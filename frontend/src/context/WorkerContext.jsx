"use client";
import React, { createContext, useContext, useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { OCR_MODELS } from '@/lib/ocrModels';

const WorkerContext = createContext();

export const useWorker = () => useContext(WorkerContext);

const DEFAULT_OCR_MODEL_KEY = 'indraBubble';

export const WorkerProvider = ({ children }) => {
    const entries = useRef(new Map());
    const [workers, setWorkers] = useState({});
    const [modelStates, setModelStates] = useState({});
    const [activeModelKey, setActiveModelKey] = useState(() => {
        if (typeof window !== 'undefined') {
            const storedKey = localStorage.getItem('ocrModelKey');
            return OCR_MODELS[storedKey] ? storedKey : DEFAULT_OCR_MODEL_KEY;
        }
        return DEFAULT_OCR_MODEL_KEY;
    });

    useEffect(() => () => {
        for (const entry of entries.current.values()) entry.worker.terminate();
        entries.current.clear();
    }, []);

    const loadModel = useCallback((modelKey) => {
        const key = modelKey || activeModelKey;
        if (OCR_MODELS[key]?.engine !== 'onnx') return;
        let entry = entries.current.get(key);
        if (entry?.status === 'loading' || entry?.status === 'ready') return;
        if (!entry) {
            const worker = key === 'lighton3BubbleWebgpu'
                ? new Worker(new URL('../workers/lighton3Bubble.worker.js', import.meta.url), { type: 'module' })
                : key === 'lighton3Webgpu'
                ? new Worker(new URL('../workers/lighton3.worker.js', import.meta.url), { type: 'module' })
                : key === 'falconWebgpu'
                ? new Worker(new URL('../workers/falcon.worker.js', import.meta.url), { type: 'module' })
                : key === 'suryaWebgpu'
                ? new Worker(new URL('../workers/surya.worker.js', import.meta.url), { type: 'module' })
                : key === 'indraBubble'
                ? new Worker(new URL('../workers/indra.worker.js', import.meta.url), { type: 'module' })
                : new Worker(new URL('../workers/ocr.worker.js', import.meta.url), { type: 'module' });
            entry = { worker, status: 'idle' };
            entries.current.set(key, entry);
            const update = (state) => {
                entry.status = state.status;
                setModelStates(previous => ({ ...previous, [key]: { ...previous[key], ...state } }));
            };
            worker.addEventListener('message', ({ data }) => {
                if (data.status === 'download_progress') update({ status: 'loading', progress: Math.round(data.progress || 0), file: data.file || '' });
                if (data.status === 'ready') update({ status: 'ready', progress: 100 });
                if (data.status === 'error' && !data.requestId) update({ status: 'error', error: data.error });
            });
            worker.addEventListener('error', (event) => update({ status: 'error', error: event.message }));
            setWorkers(previous => ({ ...previous, [key]: worker }));
        }
        entry.status = 'loading';
        setModelStates(previous => ({ ...previous, [key]: { status: 'loading', progress: 0 } }));
        entry.worker.postMessage({ type: 'init', modelKey: key });
    }, [activeModelKey]);

    const switchModel = useCallback((key) => {
        if (!OCR_MODELS[key]) return;
        localStorage.setItem('ocrModelKey', key);
        setActiveModelKey(key);
    }, []);

    const runOcr = useCallback(async (imageInput, requestId = null, modelKey = activeModelKey) => {
        const entry = entries.current.get(modelKey);
        if (entry?.status !== 'ready') throw new Error(`${OCR_MODELS[modelKey]?.name || modelKey} n'est pas chargé.`);
        const isBitmap = typeof ImageBitmap !== 'undefined' && imageInput instanceof ImageBitmap;
        const payload = isBitmap
            ? { type: 'run', imageBitmap: imageInput, requestId }
            : { type: 'run', imageBlob: imageInput, requestId };
        entry.worker.postMessage(payload, isBitmap ? [imageInput] : []);
    }, [activeModelKey]);

    const value = useMemo(() => ({
        workers,
        modelStates,
        worker: workers[activeModelKey] || null,
        modelStatus: modelStates[activeModelKey]?.status || (OCR_MODELS[activeModelKey]?.runtime === 'cloud' ? 'ready' : 'idle'),
        downloadProgress: modelStates[activeModelKey]?.progress || 0,
        loadModel,
        switchModel,
        runOcr,
        activeModelKey,
    }), [workers, modelStates, loadModel, switchModel, runOcr, activeModelKey]);

    return <WorkerContext.Provider value={value}>{children}</WorkerContext.Provider>;
};
