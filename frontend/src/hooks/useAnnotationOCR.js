"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useWorker, OCR_MODELS } from '@/context/WorkerContext';
import { useTauriLocalOcrContext } from '@/context/TauriLocalOcrContext';
import { analyzeBubble } from '@/lib/geminiClient';
import { analyzeDeepSeekBubble } from '@/lib/deepseekClient';
import { DEEPSEEK_LABEL, getDeepSeekApiKey } from '@/lib/deepseekConfig';
import { capitalizeOcrSentenceStarts } from '@/lib/ocr-utils';
import { postOcrImage } from '@/lib/ocrProxyClient';
import { isSelectableOcrModel } from '@/lib/ocrModelAvailability';
import { cropImage, cropImageBitmap } from '@/lib/utils';
import { toast } from 'sonner';

export function useAnnotationOCR({
    imageRef,
    pageId,
    rectangle,
    pendingAnnotation,
    setPendingAnnotation,
    setIsSubmitting,
    setLoadingText,
    setIsModalOpen,
    setOcrSource,
    setDebugImageUrl,
    setShowApiKeyModal,
    isSandbox = false
}) {
    const { workers, modelStates, modelStatus, loadModel, switchModel, downloadProgress, runOcr, activeModelKey } = useWorker();
    const tauriLocalOcr = useTauriLocalOcrContext();
    const [preferLocalOCR, setPreferLocalOCR] = useState(isSandbox);
    const [geminiKey, setGeminiKey] = useState(null);
    const [hasDeepSeekKey, setHasDeepSeekKey] = useState(false);
    const [ocrResults, setOcrResults] = useState({});
    const selectableModelKeys = useMemo(() => Object.values(OCR_MODELS)
        .filter(model => isSelectableOcrModel(model, isSandbox))
        .map(model => model.key), [isSandbox]);
    const selectionStorageKey = isSandbox ? 'sandboxSelectedOcrModelKeys' : 'selectedOcrModelKeys';
    const [selectedOcrModelKeys, setSelectedOcrModelKeys] = useState(() => {
        if (typeof window === 'undefined') return ['ppocrv6Line'];
        try {
            const saved = JSON.parse(localStorage.getItem(selectionStorageKey) ?? localStorage.getItem('selectedOcrModelKeys'));
            const valid = Array.isArray(saved) ? saved.filter(key => selectableModelKeys.includes(key)) : [];
            return valid.length ? valid : ['ppocrv6Line'];
        } catch {
            return ['ppocrv6Line'];
        }
    });
    const inFlightRequests = useRef(new Map());
    const completedRequests = useRef(new Map());
    const workerWaiters = useRef(new Map());
    const deepSeekRequests = useRef(new Set());
    const deepSeekRetry = useRef(null);
    const pageGeneration = useRef(0);

    useEffect(() => {
        const requests = deepSeekRequests.current;
        const inFlight = inFlightRequests.current;
        const completed = completedRequests.current;
        return () => {
            pageGeneration.current += 1;
            if (requests.size || inFlight.size) setIsSubmitting?.(false);
            for (const controller of requests) controller.abort();
            requests.clear();
            inFlight.clear();
            completed.clear();
        };
    }, [pageId, setIsSubmitting]);

    const pendingTargetKey = pendingAnnotation
        ? JSON.stringify([pendingAnnotation.id, pendingAnnotation.x, pendingAnnotation.y, pendingAnnotation.w, pendingAnnotation.h])
        : null;
    useEffect(() => () => deepSeekRetry.current?.abort(), [pendingTargetKey]);

    useEffect(() => {
        if (typeof window === 'undefined') return;
        setPreferLocalOCR(isSandbox || localStorage.getItem('preferLocalOCR') !== 'false');
        const loadKey = () => {
            setGeminiKey(localStorage.getItem('google_api_key'));
            setHasDeepSeekKey(Boolean(getDeepSeekApiKey()));
        };
        loadKey();
        window.addEventListener('storage', loadKey);
        return () => window.removeEventListener('storage', loadKey);
    }, [isSandbox]);

    const toggleOcrPreference = useCallback(() => {
        const newValue = !preferLocalOCR;
        setPreferLocalOCR(newValue);
        localStorage.setItem('preferLocalOCR', JSON.stringify(newValue));
    }, [preferLocalOCR]);

    const toggleOcrModel = useCallback((modelKey) => {
        if (!selectableModelKeys.includes(modelKey)) return;
        setSelectedOcrModelKeys(previous => {
            const next = previous.includes(modelKey)
                ? previous.filter(key => key !== modelKey)
                : [...previous, modelKey];
            localStorage.setItem(selectionStorageKey, JSON.stringify(next));
            return next;
        });
    }, [selectableModelKeys, selectionStorageKey]);

    const getTauriTextRuntime = useCallback((modelData) => {
        const isSurya = modelData?.localModelKey === 'surya';
        return {
            canRun: isSurya ? tauriLocalOcr.canRunLocalSuryaOcr : tauriLocalOcr.canRunLocalTextOcr,
            status: isSurya ? tauriLocalOcr.localSuryaModelStatus : tauriLocalOcr.localTextModelStatus,
            isDownloading: isSurya ? tauriLocalOcr.isDownloadingLocalSuryaModel : tauriLocalOcr.isDownloadingLocalTextModel,
            runBlob: isSurya ? tauriLocalOcr.runLocalSuryaOcrBlob : tauriLocalOcr.runLocalTextOcrBlob,
            label: modelData?.label || (isSurya ? 'Surya' : 'Poneglyph'),
        };
    }, [tauriLocalOcr]);

    const waitForWorkerResult = useCallback((workerRequestId) => new Promise((resolve, reject) => {
        workerWaiters.current.set(workerRequestId, { resolve, reject });
    }), []);

    useEffect(() => {
        const loadedWorkers = Object.values(workers || {});
        const handleMessage = (event) => {
            const { status, text, error, url, requestId } = event.data;
            if (status === 'debug_image') setDebugImageUrl(url);
            if (!requestId) return;

            const waiter = workerWaiters.current.get(requestId);
            if (!waiter) return;
            if (status === 'complete') {
                workerWaiters.current.delete(requestId);
                waiter.resolve(text || '');
            }
            if (status === 'error') {
                workerWaiters.current.delete(requestId);
                waiter.reject(new Error(error || 'Erreur OCR locale'));
            }
        };
        for (const worker of loadedWorkers) worker.addEventListener('message', handleMessage);
        return () => { for (const worker of loadedWorkers) worker.removeEventListener('message', handleMessage); };
    }, [workers, setDebugImageUrl]);

    const runModel = useCallback(async (modelData, areaToCrop, requestId) => {
        if (!isSelectableOcrModel(modelData, isSandbox)) {
            throw new Error('Ce moteur OCR n’est pas disponible dans la sandbox.');
        }
        if (modelData.key === 'deepseek') {
            const controller = new AbortController();
            deepSeekRequests.current.add(controller);
            try {
                const result = await analyzeDeepSeekBubble(imageRef.current, areaToCrop, getDeepSeekApiKey(), { signal: controller.signal });
                controller.signal.throwIfAborted();
                return result.data.texte_propose;
            } finally {
                deepSeekRequests.current.delete(controller);
            }
        }
        if (modelData.key === 'lighton') {
            const blob = await cropImage(imageRef.current, areaToCrop);
            const response = await postOcrImage('/api/local_lighton', blob);
            if (!response.ok) throw new Error(`Erreur OCR ${modelData.label}`);
            const result = await response.json();
            return result.text || '';
        }

        if (modelData.runtime === 'tauri') {
            const runtime = getTauriTextRuntime(modelData);
            if (!runtime.canRun) {
                throw new Error(`${runtime.label} n'est pas chargé et prêt.`);
            }
            const blob = await cropImage(imageRef.current, areaToCrop);
            const result = await runtime.runBlob(blob);
            return result?.text || '';
        }

        if (modelData.runtime === 'onnx') {
            if (modelStates?.[modelData.key]?.status !== 'ready') {
                throw new Error(`${modelData.label} n'est pas chargé.`);
            }
            const workerRequestId = `${requestId}:${modelData.key}:${Date.now()}`;
            const bitmap = await cropImageBitmap(imageRef.current, areaToCrop);
            const result = waitForWorkerResult(workerRequestId);
            try { await runOcr(bitmap, workerRequestId, modelData.key); }
            catch (error) {
                bitmap.close();
                workerWaiters.current.get(workerRequestId)?.reject(error);
                workerWaiters.current.delete(workerRequestId);
            }
            return result;
        }

        throw new Error(`Le moteur ${modelData.label} n'est pas compatible avec la comparaison OCR.`);
    }, [modelStates, getTauriTextRuntime, imageRef, isSandbox, runOcr, waitForWorkerResult]);

    const executeSelectedOcr = useCallback((areaToCrop, requestId) => {
        // A detected bubble is first read in the background, then opened for review.
        // Reuse that result rather than charging for the same provider request twice.
        const jobKey = JSON.stringify([requestId, selectedOcrModelKeys, areaToCrop.x, areaToCrop.y, areaToCrop.w, areaToCrop.h]);
        if (completedRequests.current.has(jobKey)) return Promise.resolve(completedRequests.current.get(jobKey));
        if (inFlightRequests.current.has(jobKey)) return inFlightRequests.current.get(jobKey);

        const generation = pageGeneration.current;
        const models = selectedOcrModelKeys
            .map(key => OCR_MODELS[key])
            .filter(Boolean);
        const job = Promise.allSettled(models.map(async model => ({
            modelKey: model.key,
            label: model.label,
            text: capitalizeOcrSentenceStarts(await runModel(model, areaToCrop, requestId))
        }))).then(settled => {
            if (generation !== pageGeneration.current) throw new DOMException('Requête annulée.', 'AbortError');
            const candidates = settled
                .filter(result => result.status === 'fulfilled')
                .map(result => result.value);
            const failures = settled.flatMap((result, index) => result.status === 'rejected'
                ? [`${models[index]?.label || 'OCR'} : ${result.reason?.message || 'indisponible'}`]
                : []);

            setOcrResults(previous => ({ ...previous, [requestId]: candidates }));
            const result = { candidates, failures };
            completedRequests.current.set(jobKey, result);
            if (completedRequests.current.size > 256) completedRequests.current.delete(completedRequests.current.keys().next().value);
            return result;
        }).finally(() => {
            if (inFlightRequests.current.get(jobKey) === job) inFlightRequests.current.delete(jobKey);
        });

        inFlightRequests.current.set(jobKey, job);
        return job;
    }, [runModel, selectedOcrModelKeys]);

    const applyCandidatesToModal = useCallback((candidates) => {
        const firstText = candidates[0]?.text || '';
        setOcrSource(candidates.length > 1 ? 'multiple' : candidates[0]?.modelKey || null);
        setPendingAnnotation(previous => previous ? {
            ...previous,
            texte_propose: firstText,
            ocr_candidates: candidates
        } : previous);
        setIsModalOpen(true);
    }, [setIsModalOpen, setOcrSource, setPendingAnnotation]);

    const runBackgroundOcr = useCallback(async (areaToCrop, requestId) => {
        try {
            await executeSelectedOcr(areaToCrop, requestId);
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('Background OCR error:', error);
        }
    }, [executeSelectedOcr]);

    const runLocalOcr = useCallback(async (cropData = null, customRequestId = null) => {
        const areaToCrop = cropData || rectangle || (pendingAnnotation ? {
            x: pendingAnnotation.x,
            y: pendingAnnotation.y,
            w: pendingAnnotation.w,
            h: pendingAnnotation.h
        } : null);
        if (!areaToCrop) {
            setIsModalOpen(true);
            return;
        }

        if (selectedOcrModelKeys.includes('deepseek') && !getDeepSeekApiKey()) {
            toast.info('Configurez votre clé DeepSeek, puis relancez la transcription.');
            setShowApiKeyModal(true);
            return;
        }

        const requestId = customRequestId || Date.now();
        const generation = pageGeneration.current;
        setLoadingText(selectedOcrModelKeys.length > 1
            ? `Analyse de ${selectedOcrModelKeys.length} modèles OCR...`
            : 'Analyse OCR...');
        setIsSubmitting(true);
        setDebugImageUrl(null);

        try {
            const { candidates, failures } = await executeSelectedOcr(areaToCrop, requestId);
            if (failures.length) {
                toast.warning(`${failures.length} modèle${failures.length > 1 ? 's' : ''} OCR indisponible${failures.length > 1 ? 's' : ''}.`, {
                    description: failures.join(' · ')
                });
            }
            if (!candidates.length) {
                toast.error('Aucun modèle OCR sélectionné n’a pu traiter cette bulle.');
                setIsModalOpen(true);
                return;
            }
            applyCandidatesToModal(candidates);
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('OCR error:', error);
            toast.error(`Erreur OCR : ${error.message}`);
            setIsModalOpen(true);
        } finally {
            if (generation === pageGeneration.current) setIsSubmitting(false);
        }
    }, [applyCandidatesToModal, executeSelectedOcr, pendingAnnotation, rectangle, selectedOcrModelKeys, setDebugImageUrl, setIsModalOpen, setIsSubmitting, setLoadingText, setShowApiKeyModal]);

    const handleRetryWithDeepSeek = useCallback(async () => {
        if (!pendingAnnotation || deepSeekRetry.current) return;
        const apiKey = getDeepSeekApiKey();
        if (!apiKey) {
            setShowApiKeyModal(true);
            toast.info('Configurez votre clé DeepSeek, puis relancez la transcription.');
            return;
        }
        const target = pendingAnnotation;
        const controller = new AbortController();
        const generation = pageGeneration.current;
        deepSeekRetry.current = controller;
        deepSeekRequests.current.add(controller);
        setIsSubmitting(true);
        setLoadingText(`Transcription avec ${DEEPSEEK_LABEL}…`);
        try {
            const response = await analyzeDeepSeekBubble(imageRef.current, target, apiKey, { signal: controller.signal });
            controller.signal.throwIfAborted();
            setPendingAnnotation(previous => previous && previous.id === target.id && ['x', 'y', 'w', 'h'].every(key => previous[key] === target[key]) ? {
                ...previous, texte_propose: capitalizeOcrSentenceStarts(response.data.texte_propose),
                ocr_candidates: [{ modelKey: 'deepseek', label: DEEPSEEK_LABEL, text: capitalizeOcrSentenceStarts(response.data.texte_propose) }],
            } : previous);
            setOcrSource('deepseek');
        } catch (error) {
            if (!controller.signal.aborted) toast.error(error.message || 'Transcription DeepSeek indisponible.');
        } finally {
            deepSeekRequests.current.delete(controller);
            if (deepSeekRetry.current === controller) deepSeekRetry.current = null;
            if (generation === pageGeneration.current) setIsSubmitting(false);
        }
    }, [imageRef, pendingAnnotation, setIsSubmitting, setLoadingText, setOcrSource, setPendingAnnotation, setShowApiKeyModal]);

    const handleRetryWithCloud = useCallback((dataOverride = null) => {
        const dataToUse = dataOverride || pendingAnnotation;
        if (!dataToUse) return;
        const storedKey = localStorage.getItem('google_api_key');
        if (!storedKey) {
            if (!pendingAnnotation) setPendingAnnotation(dataToUse);
            setShowApiKeyModal(true);
            return;
        }
        setLoadingText('Analyse Cloud (Google)...');
        setIsSubmitting(true);
        setDebugImageUrl(null);
        analyzeBubble(imageRef.current, dataToUse, storedKey)
            .then(response => {
                setPendingAnnotation(previous => ({
                    ...previous,
                    texte_propose: capitalizeOcrSentenceStarts(response.data.texte_propose),
                    ocr_candidates: []
                }));
                setOcrSource('cloud');
                setIsModalOpen(true);
            })
            .catch(error => {
                console.error('Cloud OCR error:', error);
                if (error.message === 'QUOTA_EXCEEDED') toast.error('Quota API Gemini dépassé.');
                setIsModalOpen(true);
            })
            .finally(() => setIsSubmitting(false));
    }, [imageRef, pendingAnnotation, setDebugImageUrl, setIsModalOpen, setIsSubmitting, setLoadingText, setOcrSource, setPendingAnnotation, setShowApiKeyModal]);

    return {
        preferLocalOCR,
        toggleOcrPreference,
        geminiKey,
        hasDeepSeekKey,
        handleRetryWithDeepSeek,
        activeModelKey,
        modelStatus,
        loadModel,
        switchModel,
        downloadProgress,
        selectedOcrModelKeys,
        toggleOcrModel,
        runLocalOcr,
        runBackgroundOcr,
        ocrResults,
        handleRetryWithCloud
    };
}
