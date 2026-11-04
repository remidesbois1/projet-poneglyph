import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { toast } from 'sonner';
import { useAnnotationDetection } from './useAnnotationDetection';

const detection = vi.hoisted(() => ({ detectBubbles: null }));

vi.mock('@/context/DetectionContext', () => ({
    useDetection: () => ({
        detectBubbles: detection.detectBubbles,
        detectionStatus: 'ready',
        loadDetectionModel: vi.fn(),
        downloadProgress: 100,
    }),
}));
vi.mock('@/lib/utils', () => ({ cropImage: vi.fn(async () => new Blob(['page'])) }));
vi.mock('sonner', () => ({
    toast: { info: vi.fn(), success: vi.fn(), warning: vi.fn(), error: vi.fn() },
}));

const BOX_A = { x: 100, y: 100, w: 200, h: 200 };
const BOX_B = { x: 600, y: 600, w: 200, h: 200 };
const DEEPSEEK = { key: 'deepseekPage', label: 'DeepSeek V4.1 Flash · Cloud', scope: 'page' };
const GEMINI = { key: 'geminiPage', label: 'Gemini 3 Flash · Cloud', scope: 'page' };
const PAGE_BUBBLES = [
    { content: 'Texte B', bbox: [600, 600, 800, 800] },
    { content: 'Texte A', bbox: [100, 100, 300, 300] },
    { content: 'Fantôme', bbox: [900, 0, 1000, 50] },
];

let props;

function deferred() {
    let resolve;
    const promise = new Promise(res => { resolve = res; });
    return { promise, resolve };
}

function renderDetection(overrides = {}) {
    props = {
        imageRef: { current: { src: 'blob:page', naturalWidth: 1000, naturalHeight: 1000 } },
        pageId: '12',
        setRectangle: vi.fn(),
        setPendingAnnotation: vi.fn(),
        setDebugImageUrl: vi.fn(),
        runLocalOcr: vi.fn(),
        runBackgroundOcr: vi.fn(),
        runPageModel: vi.fn(async () => PAGE_BUBBLES),
        pageModels: [DEEPSEEK],
        ...overrides,
    };
    return renderHook(currentProps => useAnnotationDetection(currentProps), { initialProps: props });
}

function reviewedTexts() {
    return JSON.stringify([props.runLocalOcr.mock.calls, props.setPendingAnnotation.mock.calls]);
}

describe('useAnnotationDetection page OCR', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        detection.detectBubbles = vi.fn(async () => [BOX_A, BOX_B]);
        vi.stubGlobal('fetch', vi.fn(async () => ({ blob: async () => new Blob(['img']) })));
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('reviews YOLO boxes in YOLO order with the rallied page text and drops unmatched page bubbles', async () => {
        const { result } = renderDetection();

        await act(async () => { await result.current.runPageOcr(); });

        expect(result.current.isAutoDetecting).toBe(true);
        expect(result.current.isPageOcrRunning).toBe(false);
        expect(result.current.queueLength).toBe(1);
        expect(props.runLocalOcr).toHaveBeenCalledTimes(1);
        const [boxA, idA, optionsA] = props.runLocalOcr.mock.calls[0];
        expect(boxA).toEqual(expect.objectContaining(BOX_A));
        expect(idA).toBe(boxA.id);
        expect(optionsA).toEqual({ pageCandidates: [{ modelKey: 'deepseekPage', label: DEEPSEEK.label, text: 'Texte A' }] });
        expect(props.setPendingAnnotation).toHaveBeenLastCalledWith(expect.objectContaining({
            ...BOX_A,
            id_page: 12,
            page_ocr_candidates: [{ modelKey: 'deepseekPage', label: DEEPSEEK.label, text: 'Texte A' }],
        }));

        act(() => result.current.processNextBubble());

        expect(result.current.isAutoDetecting).toBe(true);
        expect(result.current.queueLength).toBe(0);
        const [boxB, , optionsB] = props.runLocalOcr.mock.calls[1];
        expect(boxB).toEqual(expect.objectContaining(BOX_B));
        expect(boxB.id).not.toBe(boxA.id);
        expect(optionsB).toEqual({ pageCandidates: [{ modelKey: 'deepseekPage', label: DEEPSEEK.label, text: 'Texte B' }] });
        expect(props.setPendingAnnotation).toHaveBeenLastCalledWith(expect.objectContaining({ ...BOX_B, texte_propose: '' }));

        act(() => result.current.processNextBubble());

        expect(result.current.isAutoDetecting).toBe(false);
        expect(props.setRectangle).toHaveBeenLastCalledWith(null);
        expect(toast.success).toHaveBeenCalledWith('OCR de la page terminé.');
        expect(props.runLocalOcr).toHaveBeenCalledTimes(2);
        expect(reviewedTexts()).not.toContain('Fantôme');
    });

    it('starts the bubble models on every queued YOLO box before the review', async () => {
        const { result } = renderDetection();

        await act(async () => { await result.current.runPageOcr(); });
        act(() => result.current.processNextBubble());

        expect(props.runBackgroundOcr).toHaveBeenCalledTimes(2);
        const backgroundBoxes = props.runBackgroundOcr.mock.calls.map(([box]) => box);
        expect(backgroundBoxes).toEqual([expect.objectContaining(BOX_A), expect.objectContaining(BOX_B)]);
        expect(props.runBackgroundOcr.mock.calls.map(([, id]) => id)).toEqual(backgroundBoxes.map(box => box.id));
        // The review reads the same bubbles (same ids) whose background OCR already started.
        expect(props.runLocalOcr.mock.calls.map(([, id]) => id)).toEqual(backgroundBoxes.map(box => box.id));
        expect(Math.max(...props.runBackgroundOcr.mock.invocationCallOrder))
            .toBeLessThan(props.runLocalOcr.mock.invocationCallOrder[0]);
    });

    it('skips detections already annotated without shifting their page text onto a neighbour', async () => {
        const boxC = { x: 320, y: 100, w: 200, h: 200 };
        detection.detectBubbles = vi.fn(async () => [BOX_A, boxC]);
        // Overlaps A best (IoU ≈ 0.77) but also C (IoU ≈ 0.095), so it would land on C if A were not rallied.
        const runPageModel = vi.fn(async () => [{ content: 'Texte A', bbox: [100, 100, 360, 300] }]);
        const { result } = renderDetection({ runPageModel, existingBubbles: [{ ...BOX_A }] });

        await act(async () => { await result.current.runPageOcr(); });

        expect(props.runBackgroundOcr).toHaveBeenCalledTimes(1);
        expect(props.runBackgroundOcr.mock.calls[0][0]).toEqual(expect.objectContaining(boxC));
        expect(props.runLocalOcr).toHaveBeenCalledTimes(1);
        expect(props.runLocalOcr.mock.calls[0][0]).toEqual(expect.objectContaining(boxC));
        expect(props.runLocalOcr.mock.calls[0][2]).toEqual({ pageCandidates: [] });
        expect(result.current.queueLength).toBe(0);

        act(() => result.current.processNextBubble());

        expect(result.current.isAutoDetecting).toBe(false);
        expect(reviewedTexts()).not.toContain('Texte A');
    });

    it('reports a failing page model and still reviews the YOLO boxes with the other candidates', async () => {
        const runPageModel = vi.fn(async (model) => {
            if (model.key === GEMINI.key) throw new Error('quota dépassé');
            return PAGE_BUBBLES;
        });
        const { result } = renderDetection({ runPageModel, pageModels: [DEEPSEEK, GEMINI] });

        await act(async () => { await result.current.runPageOcr(); });

        expect(toast.warning).toHaveBeenCalledWith('1 lecture de page indisponible.', {
            description: 'Gemini 3 Flash · Cloud : quota dépassé',
        });
        expect(toast.error).not.toHaveBeenCalled();
        expect(result.current.isAutoDetecting).toBe(true);
        expect(props.runLocalOcr.mock.calls[0][0]).toEqual(expect.objectContaining(BOX_A));
        expect(props.runLocalOcr.mock.calls[0][2]).toEqual({
            pageCandidates: [{ modelKey: 'deepseekPage', label: DEEPSEEK.label, text: 'Texte A' }],
        });
    });

    it('reviews YOLO boxes without page candidates when the only page model fails', async () => {
        const runPageModel = vi.fn(async () => { throw new Error('hors ligne'); });
        const { result } = renderDetection({ runPageModel });

        await act(async () => { await result.current.runPageOcr(); });

        expect(toast.warning).toHaveBeenCalledWith('1 lecture de page indisponible.', {
            description: 'DeepSeek V4.1 Flash · Cloud : hors ligne',
        });
        expect(props.runBackgroundOcr).toHaveBeenCalledTimes(2);
        expect(props.runLocalOcr.mock.calls[0][0]).toEqual(expect.objectContaining(BOX_A));
        expect(props.runLocalOcr.mock.calls[0][2]).toEqual({ pageCandidates: [] });
    });

    it('announces an empty detection without opening the review', async () => {
        detection.detectBubbles = vi.fn(async () => []);
        const { result } = renderDetection();

        await act(async () => { await result.current.runPageOcr(); });

        expect(toast.info).toHaveBeenCalledWith('Aucune bulle détectée.');
        expect(props.runLocalOcr).not.toHaveBeenCalled();
        expect(props.setPendingAnnotation).not.toHaveBeenCalled();
        expect(props.runBackgroundOcr).not.toHaveBeenCalled();
        expect(result.current.isAutoDetecting).toBe(false);
        expect(result.current.isPageOcrRunning).toBe(false);
    });

    it('drops a late page result after navigating to another page', async () => {
        const pending = deferred();
        const runPageModel = vi.fn(() => pending.promise);
        const { result, rerender } = renderDetection({ runPageModel });

        let run;
        act(() => { run = result.current.runPageOcr(); });
        expect(result.current.isPageOcrRunning).toBe(true);
        await waitFor(() => expect(props.runBackgroundOcr).toHaveBeenCalledTimes(2));

        rerender({ ...props, pageId: '13' });
        expect(result.current.isPageOcrRunning).toBe(false);

        await act(async () => {
            pending.resolve(PAGE_BUBBLES);
            await run;
        });

        expect(result.current.isPageOcrRunning).toBe(false);
        expect(result.current.isAutoDetecting).toBe(false);
        expect(props.runLocalOcr).not.toHaveBeenCalled();
        expect(props.setPendingAnnotation).not.toHaveBeenCalled();
        expect(toast.error).not.toHaveBeenCalled();
        expect(toast.warning).not.toHaveBeenCalled();
    });

    it('stops waiting for a page model that never answers when cancelled', async () => {
        const runPageModel = vi.fn(() => new Promise(() => {}));
        const { result } = renderDetection({ runPageModel });

        let run;
        act(() => { run = result.current.runPageOcr(); });
        await waitFor(() => expect(props.runBackgroundOcr).toHaveBeenCalledTimes(2));
        expect(result.current.isPageOcrRunning).toBe(true);

        await act(async () => {
            result.current.cancelPageOcr();
            await run;
        });

        expect(result.current.isPageOcrRunning).toBe(false);
        expect(result.current.isAutoDetecting).toBe(false);
        expect(props.runLocalOcr).not.toHaveBeenCalled();
        expect(props.setPendingAnnotation).not.toHaveBeenCalled();
        expect(toast.error).not.toHaveBeenCalled();
        expect(toast.warning).not.toHaveBeenCalled();
    });

    it('ignores a second launch while a page OCR is already running', async () => {
        const { result } = renderDetection();

        await act(async () => {
            await Promise.all([result.current.runPageOcr(), result.current.runPageOcr()]);
        });

        expect(detection.detectBubbles).toHaveBeenCalledTimes(1);
        expect(props.runPageModel).toHaveBeenCalledTimes(1);
        expect(props.runLocalOcr).toHaveBeenCalledTimes(1);
    });
});
