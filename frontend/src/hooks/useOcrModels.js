"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import { useWorker } from '@/context/WorkerContext';
import { useTauriLocalOcrContext } from '@/context/TauriLocalOcrContext';
import { useAiModelConfig } from '@/hooks/useAiModelConfig';
import { canDownloadMissingLocalModel } from '@/components/localModelRecovery';
import { getChatGptStatus, runChatGptPageOcr, subscribeToChatGptAuth } from '@/lib/chatGptDesktop';
import { generateDeepSeekOneShotBubbles } from '@/lib/deepseekClient';
import { getDeepSeekApiKey } from '@/lib/deepseekConfig';
import { generateOneShotBubbles } from '@/lib/geminiClient';
import { getMistralApiKey } from '@/lib/mistralConfig';
import { generateGlmOneShotBubbles } from '@/lib/mistralClient';
import { capitalizeOcrSentenceStarts } from '@/lib/ocr-utils';
import { DEFAULT_OCR_MODEL_KEYS, OCR_MODELS, getOcrModelLabel, getOcrModelName, isSelectableOcrModel } from '@/lib/ocrModels';
import { cropImage, cropImageBitmap } from '@/lib/utils';

const GEMINI_KEY_STORAGE_KEY = 'google_api_key';

/** TauriLocalOcrContext field names for each desktop model. */
const LOCAL_MODEL_FIELDS = Object.freeze({
    text: { status: 'localTextModelStatus', downloading: 'isDownloadingLocalTextModel', progress: 'localTextDownloadProgress', loading: 'isLoadingLocalTextModel', download: 'downloadLocalTextModel', load: 'loadLocalTextModel', run: 'runLocalTextOcrBlob' },
    surya: { status: 'localSuryaModelStatus', downloading: 'isDownloadingLocalSuryaModel', progress: 'localSuryaDownloadProgress', loading: 'isLoadingLocalSuryaModel', download: 'downloadLocalSuryaModel', load: 'loadLocalSuryaModel', run: 'runLocalSuryaOcrBlob' },
    bbox: { status: 'localModelStatus', downloading: 'isDownloadingLocalModel', progress: 'localDownloadProgress', loading: 'isLoadingLocalModel', download: 'downloadLocalModel', load: 'loadLocalModel', run: 'runLocalOcrBlob' },
    suryaBbox: { status: 'localSuryaBBoxModelStatus', downloading: 'isDownloadingLocalSuryaBBoxModel', progress: 'localSuryaBBoxDownloadProgress', loading: 'isLoadingLocalSuryaBBoxModel', download: 'downloadLocalSuryaBBoxModel', load: 'loadLocalSuryaBBoxModel', run: 'runLocalSuryaBBoxOcrBlob' },
});

const LOCAL_CONNECTION_ISSUES = Object.freeze({
    checking: 'Connexion au serveur local…',
    reconnecting: 'Reconnexion au serveur local…',
    offline: 'Serveur local hors ligne',
    unavailable: 'Serveur local indisponible',
});

const ACCESS_REQUIREMENTS = Object.freeze({
    gemini: { access: 'gemini', message: 'Clé Gemini requise' },
    deepseek: { access: 'deepseek', message: 'Clé DeepSeek requise' },
    mistral: { access: 'mistral', message: 'Clé Mistral requise' },
    chatgpt: { access: 'chatgpt', message: 'Connexion ChatGPT requise' },
});

function getLocalRuntime(local, localModel) {
    const fields = LOCAL_MODEL_FIELDS[localModel];
    return Object.fromEntries(Object.entries(fields).map(([name, field]) => [name, local?.[field]]));
}

function getReadyLocalRuntime(local, model) {
    const runtime = getLocalRuntime(local, model.localModel);
    if (!runtime.status?.ready || !runtime.run) throw new Error('modèle non chargé.');
    return runtime;
}

function readStoredKey(storageKey) {
    try {
        // Node's server-side localStorage getter emits an ExperimentalWarning; only browsers hold keys.
        if (typeof window === 'undefined') return '';
        return window.localStorage?.getItem(storageKey)?.trim() || '';
    }
    catch { return ''; }
}

function readKeyAccess() {
    return { gemini: Boolean(readStoredKey(GEMINI_KEY_STORAGE_KEY)), deepseek: Boolean(getDeepSeekApiKey()), mistral: Boolean(getMistralApiKey()) };
}

function readSelection(storageKey, selectableKeys) {
    try {
        const saved = JSON.parse(localStorage.getItem(storageKey) ?? localStorage.getItem('selectedOcrModelKeys'));
        const valid = Array.isArray(saved) ? saved.filter(key => selectableKeys.includes(key)) : [];
        return valid.length ? valid : [...DEFAULT_OCR_MODEL_KEYS];
    } catch {
        return [...DEFAULT_OCR_MODEL_KEYS];
    }
}

function abortError() {
    return new DOMException('Requête annulée.', 'AbortError');
}

/**
 * Readiness of one model: `ready`, `loading` (with `progress`), `load`, `download`,
 * `setup` (API key or account missing) or `unavailable` (local server unreachable).
 */
export function getOcrModelState(model, { modelStates, local, access }) {
    if (model.engine === 'onnx') {
        const state = modelStates?.[model.key];
        if (state?.status === 'ready') return { status: 'ready' };
        if (state?.status === 'loading') return { status: 'loading', progress: state.progress ?? 0, message: 'Chargement' };
        return { status: 'load', message: state?.status === 'error' ? state.error || 'Échec du chargement' : null };
    }
    if (model.engine === 'tauri') {
        const connectionIssue = LOCAL_CONNECTION_ISSUES[local?.localConnectionState?.status];
        if (connectionIssue) return { status: 'unavailable', message: connectionIssue };
        const runtime = getLocalRuntime(local, model.localModel);
        const status = runtime.status;
        const downloading = Boolean(runtime.downloading || status?.download?.active);
        if (downloading) {
            return { status: 'loading', progress: Number.isFinite(runtime.progress) ? runtime.progress : null, message: 'Téléchargement' };
        }
        if (runtime.loading || status?.loading) return { status: 'loading', progress: null, message: 'Chargement' };
        if (status?.ready) return { status: 'ready' };
        if (canDownloadMissingLocalModel(status, downloading)) return { status: 'download', message: status?.download?.error || null };
        return { status: 'load', message: status?.error || null };
    }
    const requirement = ACCESS_REQUIREMENTS[model.engine];
    if (requirement && !access[requirement.access]) return { status: 'setup', message: requirement.message };
    return { status: 'ready' };
}

/**
 * The editor's OCR models: availability, selection, readiness, preparation and execution.
 * Bubble and page models share one list; only the scope decides how they are run.
 */
export function useOcrModels({ isSandbox = false, allowPageModels = false } = {}) {
    const { workers, modelStates, loadModel, runOcr } = useWorker();
    const local = useTauriLocalOcrContext();
    const aiModelConfig = useAiModelConfig();
    const [access, setAccess] = useState(() => ({ ...readKeyAccess(), chatgpt: false, chatgptAvailable: false }));
    const latest = useRef({ workers, modelStates, local });
    useEffect(() => {
        latest.current = { workers, modelStates, local };
    });
    const workerWaiters = useRef(new Map());

    const selectionStorageKey = isSandbox ? 'sandboxSelectedOcrModelKeys' : 'selectedOcrModelKeys';
    const selectableKeys = useMemo(() => Object.values(OCR_MODELS)
        .filter(model => isSelectableOcrModel(model, isSandbox))
        .map(model => model.key), [isSandbox]);
    const [selectedOcrModelKeys, setSelectedOcrModelKeys] = useState(() => (
        typeof window === 'undefined' ? [...DEFAULT_OCR_MODEL_KEYS] : readSelection(selectionStorageKey, selectableKeys)
    ));

    const toggleOcrModel = useCallback((modelKey) => {
        if (!selectableKeys.includes(modelKey)) return;
        setSelectedOcrModelKeys(previous => {
            const next = previous.includes(modelKey)
                ? previous.filter(key => key !== modelKey)
                : [...previous, modelKey];
            localStorage.setItem(selectionStorageKey, JSON.stringify(next));
            return next;
        });
    }, [selectableKeys, selectionStorageKey]);

    const mounted = useRef(false);
    const applyChatGptStatus = useCallback((status) => {
        if (!mounted.current) return;
        setAccess(previous => ({ ...previous, chatgptAvailable: Boolean(status?.available), chatgpt: Boolean(status?.available && status?.connected) }));
    }, []);

    /** Re-reads API keys and the ChatGPT session, e.g. after the API settings dialog closes. */
    const refreshModelAccess = useCallback(async () => {
        setAccess(previous => ({ ...previous, ...readKeyAccess() }));
        applyChatGptStatus(await getChatGptStatus());
    }, [applyChatGptStatus]);

    useEffect(() => {
        mounted.current = true;
        getChatGptStatus().then(applyChatGptStatus);
        const handleStorage = () => setAccess(previous => ({ ...previous, ...readKeyAccess() }));
        window.addEventListener('storage', handleStorage);
        const unsubscribe = subscribeToChatGptAuth(status => setAccess(previous => ({ ...previous, chatgptAvailable: true, chatgpt: Boolean(status?.connected) })));
        return () => {
            mounted.current = false;
            window.removeEventListener('storage', handleStorage);
            unsubscribe();
        };
    }, [applyChatGptStatus]);

    const models = useMemo(() => Object.values(OCR_MODELS)
        .filter(model => isSelectableOcrModel(model, isSandbox)
            && (model.scope === 'bubble' || allowPageModels)
            && (model.engine !== 'tauri' || local?.isTauri)
            && (model.engine !== 'chatgpt' || access.chatgptAvailable))
        .map(model => ({
            ...model,
            name: getOcrModelName(model, aiModelConfig),
            label: getOcrModelLabel(model, aiModelConfig),
            selected: selectedOcrModelKeys.includes(model.key),
            ...getOcrModelState(model, { modelStates, local, access }),
        })), [access, aiModelConfig, allowPageModels, isSandbox, local, modelStates, selectedOcrModelKeys]);

    const prepareSelectedModels = useCallback(() => {
        for (const model of models) {
            if (!model.selected || !['load', 'download'].includes(model.status)) continue;
            if (model.engine === 'onnx') {
                loadModel(model.key);
                continue;
            }
            if (model.engine !== 'tauri') continue;
            const runtime = getLocalRuntime(local, model.localModel);
            const prepare = model.status === 'download' ? runtime.download : runtime.load;
            Promise.resolve(prepare?.()).then(result => {
                if (result?.ok === false || result?.error) toast.error(`${model.label} : ${result.error || 'préparation impossible.'}`);
            });
        }
    }, [loadModel, local, models]);

    useEffect(() => {
        const entries = Object.entries(workers || {});
        const listeners = entries.map(([modelKey, worker]) => {
            const onMessage = ({ data }) => {
                const waiter = data?.requestId ? workerWaiters.current.get(data.requestId) : null;
                if (!waiter || (data.status !== 'complete' && data.status !== 'error')) return;
                workerWaiters.current.delete(data.requestId);
                if (data.status === 'complete') waiter.resolve(data);
                else waiter.reject(new Error(data.error || 'Erreur OCR locale'));
            };
            const onError = (event) => {
                for (const [requestId, waiter] of workerWaiters.current) {
                    if (waiter.modelKey !== modelKey) continue;
                    workerWaiters.current.delete(requestId);
                    waiter.reject(new Error(event?.message || 'Worker OCR indisponible.'));
                }
            };
            worker.addEventListener('message', onMessage);
            worker.addEventListener('error', onError);
            return { worker, onMessage, onError };
        });
        return () => listeners.forEach(({ worker, onMessage, onError }) => {
            worker.removeEventListener('message', onMessage);
            worker.removeEventListener('error', onError);
        });
    }, [workers]);

    /** Sends an image to a loaded browser model and resolves with the worker's `complete` message. */
    const runWorker = useCallback((model, image, requestId, signal) => new Promise((resolve, reject) => {
        if (latest.current.modelStates?.[model.key]?.status !== 'ready') {
            image.close?.();
            reject(new Error('modèle non chargé.'));
            return;
        }
        if (signal?.aborted) {
            image.close?.();
            reject(abortError());
            return;
        }
        const onAbort = () => {
            if (!workerWaiters.current.delete(requestId)) return;
            latest.current.workers?.[model.key]?.postMessage({ type: 'cancel', requestId });
            reject(abortError());
        };
        const settle = (callback) => (value) => {
            signal?.removeEventListener('abort', onAbort);
            callback(value);
        };
        workerWaiters.current.set(requestId, { modelKey: model.key, resolve: settle(resolve), reject: settle(reject) });
        signal?.addEventListener('abort', onAbort, { once: true });
        runOcr(image, requestId, model.key).catch(error => {
            image.close?.();
            workerWaiters.current.get(requestId)?.reject(error);
            workerWaiters.current.delete(requestId);
        });
    }), [runOcr]);

    /** Reads one bubble crop. Resolves with the raw transcript. */
    const runBubbleModel = useCallback(async (model, image, area, { requestId, signal } = {}) => {
        switch (model.engine) {
            case 'onnx': {
                const data = await runWorker(model, await cropImageBitmap(image, area), `${requestId}:${model.key}:${Date.now()}`, signal);
                return data.text || '';
            }
            case 'tauri': {
                const runtime = getReadyLocalRuntime(latest.current.local, model);
                const result = await runtime.run(await cropImage(image, area));
                return result?.text || '';
            }
            default:
                throw new Error('lecture de bulle non prise en charge.');
        }
    }, [runWorker]);

    /**
     * Reads the whole page. Resolves with `[{ content, bbox }]`, `bbox` normalized to 0–1000.
     * `getPageJpeg` lets one pipeline run share a single page encoding across models.
     */
    const runPageModel = useCallback(async (model, image, { signal, getPageJpeg } = {}) => {
        const pageJpeg = () => (getPageJpeg ? getPageJpeg() : cropImage(image, { x: 0, y: 0, w: image.naturalWidth, h: image.naturalHeight }));
        let bubbles;
        switch (model.engine) {
            case 'onnx': {
                const data = await runWorker(model, await createImageBitmap(image), `${model.key}:${crypto.randomUUID()}`, signal);
                if (data.tokenLimit || data.invalidBlocks) {
                    toast.warning(`${model.label} : ${data.tokenLimit ? 'lecture tronquée, vérifiez les dernières bulles.' : 'zones invalides ignorées.'}`);
                }
                bubbles = data.bubbles;
                break;
            }
            case 'tauri': {
                const runtime = getReadyLocalRuntime(latest.current.local, model);
                const result = await runtime.run(await pageJpeg());
                if (result?.error) throw new Error(result.error);
                bubbles = result?.bubbles;
                break;
            }
            case 'gemini': {
                const apiKey = readStoredKey(GEMINI_KEY_STORAGE_KEY);
                if (!apiKey) throw new Error('clé Gemini manquante.');
                try {
                    bubbles = (await generateOneShotBubbles(image, apiKey)).data;
                } catch (error) {
                    throw error?.message === 'QUOTA_EXCEEDED' ? new Error('quota Gemini dépassé.') : error;
                }
                break;
            }
            case 'chatgpt':
                bubbles = (await runChatGptPageOcr(await pageJpeg()))?.bubbles;
                break;
            case 'deepseek': {
                const apiKey = getDeepSeekApiKey();
                if (!apiKey) throw new Error('clé DeepSeek manquante.');
                bubbles = (await generateDeepSeekOneShotBubbles(image, apiKey, { signal })).data;
                break;
            }
            case 'mistral': {
                const apiKey = getMistralApiKey();
                if (!apiKey) throw new Error('clé Mistral manquante.');
                bubbles = (await generateGlmOneShotBubbles(image, apiKey, { signal })).data;
                break;
            }
            default:
                throw new Error('lecture de page non prise en charge.');
        }
        signal?.throwIfAborted();
        if (!Array.isArray(bubbles)) throw new Error('réponse invalide.');
        return bubbles.map(bubble => ({
            bbox: bubble.bbox,
            content: model.preserveTranscript ? String(bubble.content ?? '') : capitalizeOcrSentenceStarts(bubble.content),
        }));
    }, [runWorker]);

    return {
        models,
        selectedOcrModelKeys,
        toggleOcrModel,
        prepareSelectedModels,
        refreshModelAccess,
        runBubbleModel,
        runPageModel,
    };
}
