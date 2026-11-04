"use client";

import { useState, useRef, useCallback, useEffect } from 'react';
import { useDetection } from '@/context/DetectionContext';
import { boxIou, rallyPageBubblesToYolo } from '@/lib/ocrBboxFusion';
import { cropImage } from '@/lib/utils';
import { toast } from 'sonner';

// Same threshold as the batch OCR: a detection this close to an annotation is the same bubble.
const KNOWN_BUBBLE_IOU = 0.9;

/**
 * Page OCR: YOLO detects the bubbles and their reading order, page models read the whole page and
 * are rallied onto those boxes (YOLO geometry and order win), bubble models read each crop.
 * Every new bubble is then reviewed in the editor, in YOLO order.
 */
export function useAnnotationDetection({
    imageRef,
    pageId,
    existingBubbles = [],
    setRectangle,
    setPendingAnnotation,
    setDebugImageUrl,
    runLocalOcr,
    runBackgroundOcr,
    runPageModel,
    pageModels = [],
    detectionDebugEnabled = false,
    setDetectionDebugData
}) {
    const { detectBubbles, detectionStatus, loadDetectionModel, downloadProgress: detectionProgress } = useDetection();
    const [isAutoDetecting, setIsAutoDetecting] = useState(false);
    const [isPageOcrRunning, setIsPageOcrRunning] = useState(false);
    const [queueLength, setQueueLength] = useState(0);
    const reviewQueueRef = useRef([]);
    const runRef = useRef(null);

    useEffect(() => () => {
        runRef.current?.abort();
        runRef.current = null;
        reviewQueueRef.current = [];
        setIsPageOcrRunning(false);
        setIsAutoDetecting(false);
        setQueueLength(0);
    }, [pageId]);

    const processNextBubble = useCallback(() => {
        if (!imageRef.current) return;

        const next = reviewQueueRef.current.shift();
        if (!next) {
            setIsAutoDetecting(false);
            setQueueLength(0);
            setRectangle(null);
            toast.success("OCR de la page terminé.");
            return;
        }

        setQueueLength(reviewQueueRef.current.length);
        const { pageCandidates, ...box } = next;
        setRectangle(box);
        setPendingAnnotation({ id_page: parseInt(pageId, 10), ...box, texte_propose: '', page_ocr_candidates: pageCandidates });
        setDebugImageUrl(null);
        runLocalOcr(box, box.id, { pageCandidates });
    }, [imageRef, pageId, setRectangle, setPendingAnnotation, setDebugImageUrl, runLocalOcr]);

    const runPageOcr = useCallback(async () => {
        const image = imageRef.current;
        if (runRef.current || detectionStatus !== 'ready' || !image?.naturalWidth || !image?.naturalHeight) return;

        const controller = new AbortController();
        const { signal } = controller;
        const source = image.src;
        const width = image.naturalWidth;
        const height = image.naturalHeight;
        const isCurrent = () => !signal.aborted && imageRef.current === image && image.src === source;
        runRef.current = controller;
        setIsPageOcrRunning(true);
        // Providers that cannot be interrupted keep running; cancelling stops waiting for them.
        const aborted = new Promise((_, reject) => signal.addEventListener('abort', () => reject(new DOMException('OCR annulé.', 'AbortError')), { once: true }));
        aborted.catch(() => {});

        try {
            let pageJpeg = null;
            const getPageJpeg = () => (pageJpeg ??= cropImage(image, { x: 0, y: 0, w: width, h: height }));
            const pageReads = pageModels.map(model => runPageModel(model, image, { signal, getPageJpeg })
                .then(bubbles => ({ model, bubbles }), error => ({ model, error })));

            const imageBlob = await fetch(source, { signal }).then(response => response.blob());
            const detection = await detectBubbles(imageBlob, {
                signal,
                debug: detectionDebugEnabled,
                returnDebug: detectionDebugEnabled
            });
            if (!isCurrent()) return;
            if (detectionDebugEnabled) setDetectionDebugData?.(detection?.debug || null);
            const yoloBoxes = (detectionDebugEnabled ? detection?.boxes : detection) || [];

            const runId = Date.now();
            const bubbles = yoloBoxes.map((box, index) => ({ ...box, id: `auto-${runId}-${index}`, pageCandidates: [] }));
            const queue = bubbles.filter(bubble => !existingBubbles.some(existing => boxIou(existing, bubble) >= KNOWN_BUBBLE_IOU));
            if (!queue.length) {
                toast.info(bubbles.length ? 'Toutes les bulles détectées sont déjà annotées.' : 'Aucune bulle détectée.');
                return;
            }
            for (const bubble of queue) runBackgroundOcr(bubble, bubble.id);

            const failures = [];
            for (const { model, bubbles: pageBubbles, error } of await Promise.race([Promise.all(pageReads), aborted])) {
                if (error) {
                    if (error.name !== 'AbortError') failures.push(`${model.label} : ${error.message || 'indisponible'}`);
                    continue;
                }
                // Rally against every detection so text never drifts onto a neighbour of an annotated bubble.
                const { matches } = rallyPageBubblesToYolo(pageBubbles, yoloBoxes, width, height);
                for (const match of matches) {
                    bubbles[match.yoloIndex].pageCandidates.push({ modelKey: model.key, label: model.label, text: match.text });
                }
            }
            if (!isCurrent()) return;
            if (failures.length) {
                toast.warning(`${failures.length} lecture${failures.length > 1 ? 's' : ''} de page indisponible${failures.length > 1 ? 's' : ''}.`, {
                    description: failures.join(' · ')
                });
            }

            reviewQueueRef.current = queue;
            setQueueLength(queue.length);
            setIsAutoDetecting(true);
            processNextBubble();
        } catch (error) {
            if (isCurrent()) toast.error(error?.message || "OCR de la page indisponible.");
        } finally {
            controller.abort();
            if (runRef.current === controller) {
                runRef.current = null;
                setIsPageOcrRunning(false);
            }
        }
    }, [detectBubbles, detectionDebugEnabled, detectionStatus, existingBubbles, imageRef, pageModels, processNextBubble, runBackgroundOcr, runPageModel, setDetectionDebugData]);

    const cancelPageOcr = useCallback(() => {
        runRef.current?.abort();
    }, []);

    return {
        isAutoDetecting,
        setIsAutoDetecting,
        queueLength,
        detectionStatus,
        loadDetectionModel,
        detectionProgress,
        isPageOcrRunning,
        runPageOcr,
        cancelPageOcr,
        processNextBubble
    };
}
