"use client";

import { useEffect, useRef, useCallback, useMemo } from 'react';
import { useWorker } from '@/context/WorkerContext';
import { useOcrModels } from '@/hooks/useOcrModels';
import { analyzeBubble } from '@/lib/geminiClient';
import { analyzeDeepSeekBubble } from '@/lib/deepseekClient';
import { DEEPSEEK_LABEL, getDeepSeekApiKey } from '@/lib/deepseekConfig';
import { capitalizeOcrSentenceStarts } from '@/lib/ocr-utils';
import { OCR_MODELS, getOcrModelLabel } from '@/lib/ocrModels';
import { toast } from 'sonner';

const DEEPSEEK_CANDIDATE_LABEL = getOcrModelLabel(OCR_MODELS.deepseek);

/**
 * OCR inside the annotation editor. A bubble (hand-drawn or queued by the page pipeline) is read
 * by the selected bubble models only; page models contribute through the `pageCandidates` that
 * the page pipeline rallied onto that bubble.
 */
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
    isSandbox = false,
    allowPageModels = false
}) {
    const { workers } = useWorker();
    const {
        models, selectedOcrModelKeys, toggleOcrModel, prepareSelectedModels,
        refreshModelAccess, runBubbleModel, runPageModel
    } = useOcrModels({ isSandbox, allowPageModels });
    const bubbleModels = useMemo(() => models.filter(model => model.selected && model.scope === 'bubble'), [models]);
    const pageModels = useMemo(() => models.filter(model => model.selected && model.scope === 'page'), [models]);
    const inFlightRequests = useRef(new Map());
    const completedRequests = useRef(new Map());
    const requestControllers = useRef(new Set());
    const deepSeekRetry = useRef(null);
    const pageGeneration = useRef(0);

    useEffect(() => {
        const controllers = requestControllers.current;
        const inFlight = inFlightRequests.current;
        const completed = completedRequests.current;
        return () => {
            pageGeneration.current += 1;
            if (controllers.size || inFlight.size) setIsSubmitting?.(false);
            for (const controller of controllers) controller.abort();
            controllers.clear();
            inFlight.clear();
            completed.clear();
        };
    }, [pageId, setIsSubmitting]);

    const pendingTargetKey = pendingAnnotation
        ? JSON.stringify([pendingAnnotation.id, pendingAnnotation.x, pendingAnnotation.y, pendingAnnotation.w, pendingAnnotation.h])
        : null;
    useEffect(() => () => deepSeekRetry.current?.abort(), [pendingTargetKey]);

    useEffect(() => {
        const loadedWorkers = Object.values(workers || {});
        const handleMessage = ({ data }) => {
            if (data?.status === 'debug_image') setDebugImageUrl(data.url);
        };
        for (const worker of loadedWorkers) worker.addEventListener('message', handleMessage);
        return () => { for (const worker of loadedWorkers) worker.removeEventListener('message', handleMessage); };
    }, [workers, setDebugImageUrl]);

    const executeSelectedOcr = useCallback((areaToCrop, requestId) => {
        // A detected bubble is first read in the background, then opened for review.
        // Reuse that result rather than charging for the same provider request twice.
        const jobKey = JSON.stringify([requestId, bubbleModels.map(model => model.key), areaToCrop.x, areaToCrop.y, areaToCrop.w, areaToCrop.h]);
        if (completedRequests.current.has(jobKey)) return Promise.resolve(completedRequests.current.get(jobKey));
        if (inFlightRequests.current.has(jobKey)) return inFlightRequests.current.get(jobKey);

        const generation = pageGeneration.current;
        const controller = new AbortController();
        requestControllers.current.add(controller);
        const job = Promise.allSettled(bubbleModels.map(async model => {
            const text = await runBubbleModel(model, imageRef.current, areaToCrop, { requestId, signal: controller.signal });
            return { modelKey: model.key, label: model.label,
                text: model.preserveTranscript ? text : capitalizeOcrSentenceStarts(text) };
        })).then(settled => {
            if (generation !== pageGeneration.current) throw new DOMException('Requête annulée.', 'AbortError');
            const candidates = settled
                .filter(result => result.status === 'fulfilled')
                .map(result => result.value);
            const failures = settled.flatMap((result, index) => result.status === 'rejected'
                ? [`${bubbleModels[index].label} : ${result.reason?.message || 'indisponible'}`]
                : []);
            const result = { candidates, failures };
            completedRequests.current.set(jobKey, result);
            if (completedRequests.current.size > 256) completedRequests.current.delete(completedRequests.current.keys().next().value);
            return result;
        }).finally(() => {
            requestControllers.current.delete(controller);
            if (inFlightRequests.current.get(jobKey) === job) inFlightRequests.current.delete(jobKey);
        });

        inFlightRequests.current.set(jobKey, job);
        return job;
    }, [bubbleModels, imageRef, runBubbleModel]);

    const applyCandidatesToModal = useCallback((candidates) => {
        setOcrSource(candidates.length > 1 ? 'multiple' : candidates[0]?.modelKey || null);
        setPendingAnnotation(previous => previous ? {
            ...previous,
            texte_propose: candidates[0]?.text || '',
            ocr_candidates: candidates
        } : previous);
        setIsModalOpen(true);
    }, [setIsModalOpen, setOcrSource, setPendingAnnotation]);

    const runBackgroundOcr = useCallback(async (areaToCrop, requestId) => {
        if (!bubbleModels.length) return;
        try {
            await executeSelectedOcr(areaToCrop, requestId);
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('Background OCR error:', error);
        }
    }, [bubbleModels.length, executeSelectedOcr]);

    /**
     * Reads one bubble with the selected bubble models and opens the editor.
     * `pageCandidates` (page OCR) are offered alongside; a retry without a crop reuses the open bubble's.
     */
    const runLocalOcr = useCallback(async (cropData = null, customRequestId = null, { pageCandidates } = {}) => {
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
        const extraCandidates = pageCandidates ?? (cropData ? [] : pendingAnnotation?.page_ocr_candidates ?? []);

        if (bubbleModels.some(model => model.engine === 'deepseek') && !getDeepSeekApiKey()) {
            toast.info('Configurez votre clé DeepSeek, puis relancez la transcription.');
            setShowApiKeyModal(true);
            return;
        }
        if (!bubbleModels.length) {
            applyCandidatesToModal(extraCandidates);
            return;
        }

        const requestId = customRequestId || Date.now();
        const generation = pageGeneration.current;
        setLoadingText(bubbleModels.length > 1
            ? `Analyse de ${bubbleModels.length} modèles OCR...`
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
            if (!candidates.length && !extraCandidates.length) toast.error('Aucun modèle OCR n’a pu lire cette bulle.');
            applyCandidatesToModal([...candidates, ...extraCandidates]);
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('OCR error:', error);
            toast.error(`Erreur OCR : ${error.message}`);
            setIsModalOpen(true);
        } finally {
            if (generation === pageGeneration.current) setIsSubmitting(false);
        }
    }, [applyCandidatesToModal, bubbleModels, executeSelectedOcr, pendingAnnotation, rectangle, setDebugImageUrl, setIsModalOpen, setIsSubmitting, setLoadingText, setShowApiKeyModal]);

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
        requestControllers.current.add(controller);
        setIsSubmitting(true);
        setLoadingText(`Transcription avec ${DEEPSEEK_LABEL}…`);
        try {
            const response = await analyzeDeepSeekBubble(imageRef.current, target, apiKey, { signal: controller.signal });
            controller.signal.throwIfAborted();
            setPendingAnnotation(previous => previous && previous.id === target.id && ['x', 'y', 'w', 'h'].every(key => previous[key] === target[key]) ? {
                ...previous, texte_propose: capitalizeOcrSentenceStarts(response.data.texte_propose),
                ocr_candidates: [{ modelKey: 'deepseek', label: DEEPSEEK_CANDIDATE_LABEL, text: capitalizeOcrSentenceStarts(response.data.texte_propose) }],
            } : previous);
            setOcrSource('deepseek');
        } catch (error) {
            if (!controller.signal.aborted) toast.error(error.message || 'Transcription DeepSeek indisponible.');
        } finally {
            requestControllers.current.delete(controller);
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
        models,
        selectedOcrModelKeys,
        toggleOcrModel,
        prepareSelectedModels,
        refreshModelAccess,
        bubbleModels,
        pageModels,
        runPageModel,
        runLocalOcr,
        runBackgroundOcr,
        handleRetryWithCloud,
        handleRetryWithDeepSeek
    };
}
