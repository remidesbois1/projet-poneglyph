"use client";

import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { generateDeepSeekOneShotBubbles } from '@/lib/deepseekClient';
import { getDeepSeekApiKey } from '@/lib/deepseekConfig';
import { reconcileOcrBubblesWithYolo } from '@/lib/ocrBboxFusion';
import { cropImage } from '@/lib/utils';

/** One explicit request, bound to the image/page that launched it. No automatic paid retries. */
export function useDeepSeekPageOcr({ imageRef, contextKey, canRun = true, busy = false, detectionStatus, detectBubbles, onApply, onConfigure }) {
    const activeJob = useRef(null);
    const [loadingJob, setLoadingJob] = useState(null);

    useEffect(() => () => {
        activeJob.current?.abort();
        activeJob.current = null;
    }, [contextKey]);

    const handleDeepSeekOneShot = async () => {
        const image = imageRef.current;
        if (!canRun || busy || activeJob.current || !image?.naturalWidth || !image?.naturalHeight) return;
        const key = getDeepSeekApiKey();
        if (!key) {
            onConfigure?.();
            toast.info('Ajoutez votre clé DeepSeek dans les réglages API, puis relancez la lecture.');
            return;
        }
        const controller = new AbortController();
        activeJob.current = controller;
        setLoadingJob({ contextKey, controller });
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        const src = image.src;
        const isCurrent = () => !controller.signal.aborted && imageRef.current === image && image.src === src;
        try {
            const yoloPromise = detectionStatus === 'ready' && detectBubbles
                ? cropImage(image, { x: 0, y: 0, w: width, h: height }).then(blob => {
                    controller.signal.throwIfAborted();
                    return detectBubbles(blob);
                }).catch(() => null)
                : Promise.resolve(null);
            const [result, yolo] = await Promise.all([
                generateDeepSeekOneShotBubbles(image, key, { signal: controller.signal }), yoloPromise,
            ]);
            if (!isCurrent()) return;
            const bubbles = reconcileOcrBubblesWithYolo(result.data, yolo, width, height);
            if (!bubbles.length) {
                toast.info('DeepSeek n’a détecté aucun texte exploitable.');
                return;
            }
            await onApply(bubbles, { signal: controller.signal, isCurrent });
        } catch (error) {
            if (isCurrent()) toast.error(error.message || 'Extraction DeepSeek indisponible.');
        } finally {
            if (activeJob.current === controller) {
                activeJob.current = null;
            }
            setLoadingJob(previous => previous?.controller === controller ? null : previous);
        }
    };

    return { handleDeepSeekOneShot, isDeepSeekLoading: Boolean(loadingJob && loadingJob.contextKey === contextKey && !loadingJob.controller.signal.aborted) };
}
