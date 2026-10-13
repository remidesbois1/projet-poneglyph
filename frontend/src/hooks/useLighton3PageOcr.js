'use client';
import { useEffect, useRef, useState } from 'react';
import { useWorker } from '@/context/WorkerContext';
import { lighton3BubbleToPixels } from '@/lib/lighton3Grounding';
import { toast } from 'sonner';

export function useLighton3PageOcr({ imageRef, contextKey, canRun = true, busy = false, onApply }) {
    const { workers, modelStates, loadModel, runOcr } = useWorker();
    const worker = workers.lighton3Webgpu;
    const pending = useRef(null);
    const [jobKey, setJobKey] = useState(null);
    const [stream, setStream] = useState('');
    const cancel = () => {
        const job = pending.current;
        if (!job) return;
        job.controller.abort();
        job.worker.postMessage({ type: 'cancel', requestId: job.requestId });
        job.reject(new DOMException('Lecture annulée.', 'AbortError'));
        pending.current = null;
    };
    useEffect(() => () => cancel(), [contextKey]);
    useEffect(() => {
        if (!worker) return;
        const message = ({ data }) => {
            const job = pending.current;
            if (!job || data.requestId !== job.requestId) return;
            if (data.status === 'stream') setStream(data.text);
            if (data.status === 'complete') job.resolve(data);
            if (data.status === 'error') job.reject(new Error(data.error));
        };
        const error = event => pending.current?.reject(new Error(event.message || 'Worker OCR indisponible.'));
        worker.addEventListener('message', message); worker.addEventListener('error', error);
        return () => { worker.removeEventListener('message', message); worker.removeEventListener('error', error); cancel(); };
    }, [worker]);
    const handleLighton3OneShot = async () => {
        const image = imageRef.current;
        if (!canRun || busy || pending.current || !image?.naturalWidth || !image?.naturalHeight) return;
        if (modelStates.lighton3Webgpu?.status !== 'ready') { loadModel('lighton3Webgpu'); return; }
        const controller = new AbortController(), requestId = crypto.randomUUID();
        const width = image.naturalWidth, height = image.naturalHeight, src = image.src;
        const isCurrent = () => !controller.signal.aborted && imageRef.current === image && image.src === src;
        const job = { controller, requestId, worker, reject: () => {} };
        pending.current = job;
        setJobKey(contextKey); setStream('');
        try {
            const bitmap = await createImageBitmap(image);
            if (!isCurrent()) { bitmap.close(); return; }
            const result = new Promise((resolve, reject) => {
                Object.assign(job, { resolve, reject });
            });
            try { await runOcr(bitmap, requestId, 'lighton3Webgpu'); }
            catch (error) { bitmap.close(); pending.current?.reject(error); }
            const data = await result;
            if (!isCurrent()) return;
            if (data.tokenLimit || data.invalidBlocks) {
                toast.warning(data.tokenLimit ? 'Lecture tronquée : vérifiez les dernières zones.' : 'Certaines zones générées sont invalides.');
            }
            const bubbles = data.bubbles.map(bubble => lighton3BubbleToPixels(bubble, width, height))
                .filter(bubble => bubble.w > 0 && bubble.h > 0);
            if (!bubbles.length) { toast.info('LightOnOCR 3 n’a détecté aucun texte exploitable.'); return; }
            await onApply(bubbles, { signal: controller.signal, isCurrent });
        } catch (error) {
            if (isCurrent()) toast.error(error.message || 'Lecture LightOnOCR 3 indisponible.');
        } finally {
            if (pending.current?.controller === controller) pending.current = null;
            setJobKey(previous => previous === contextKey ? null : previous);
        }
    };
    return { handleLighton3OneShot, cancelLighton3: cancel, lighton3State: modelStates.lighton3Webgpu,
        isLighton3Loading: jobKey === contextKey && jobKey !== null, lighton3Stream: stream };
}
