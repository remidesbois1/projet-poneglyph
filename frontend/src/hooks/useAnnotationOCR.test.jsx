import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAnnotationOCR } from './useAnnotationOCR';
import { analyzeDeepSeekBubble } from '@/lib/deepseekClient';
import { analyzeBubble } from '@/lib/geminiClient';

const workerContext = vi.hoisted(() => ({ value: {} }));
vi.mock('@/lib/utils', async (importOriginal) => ({ ...await importOriginal(), cropImageBitmap: vi.fn(async () => ({ close: vi.fn() })) }));

vi.mock('@/context/WorkerContext', async (importOriginal) => ({
    ...await importOriginal(),
    useWorker: () => workerContext.value,
}));
vi.mock('@/context/TauriLocalOcrContext', () => ({ useTauriLocalOcrContext: () => ({}) }));
vi.mock('@/lib/deepseekClient', () => ({ analyzeDeepSeekBubble: vi.fn() }));
vi.mock('@/lib/geminiClient', () => ({ analyzeBubble: vi.fn() }));

describe('sandbox OCR selection', () => {
    beforeEach(() => { localStorage.clear(); workerContext.value = { modelStatus: 'idle', activeModelKey: 'ppocrv6Line' }; });

    it('filters saved Modal models while retaining local selections', () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['lighton', 'suryaLocal']));
        const { result } = renderHook(() => useAnnotationOCR({ isSandbox: true }));
        expect(result.current.selectedOcrModelKeys).toEqual(['suryaLocal']);
        act(() => result.current.toggleOcrModel('lighton'));
        expect(result.current.selectedOcrModelKeys).toEqual(['suryaLocal']);
        act(() => result.current.toggleOcrModel('ppocrv6Line'));
        expect(JSON.parse(localStorage.getItem('selectedOcrModelKeys'))).toEqual(['lighton', 'suryaLocal']);
        expect(JSON.parse(localStorage.getItem('sandboxSelectedOcrModelKeys'))).toEqual(['suryaLocal', 'ppocrv6Line']);
    });

    it('falls back to browser OCR when only a Modal model was saved', () => {
        localStorage.setItem('sandboxSelectedOcrModelKeys', JSON.stringify(['lighton']));
        const { result } = renderHook(() => useAnnotationOCR({ isSandbox: true }));
        expect(result.current.selectedOcrModelKeys).toEqual(['ppocrv6Line']);
    });

    it('keeps Modal selection available outside the sandbox', () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['lighton']));
        const { result } = renderHook(() => useAnnotationOCR({ isSandbox: false }));
        expect(result.current.selectedOcrModelKeys).toEqual(['lighton']);
    });
});


describe('simultaneous browser OCR', () => {
    it('collects replies from both loaded workers even when Falcon is not active', async () => {
        localStorage.setItem('sandboxSelectedOcrModelKeys', JSON.stringify(['falconWebgpu', 'ppocrv6Line']));
        const workers = { falconWebgpu: new EventTarget(), ppocrv6Line: new EventTarget() };
        const runOcr = vi.fn(async (_bitmap, requestId, key) => {
            queueMicrotask(() => workers[key].dispatchEvent(new MessageEvent('message', {
                data: { status: 'complete', requestId, text: key === 'falconWebgpu' ? 'Falcon text' : 'Paddle text' }
            })));
        });
        workerContext.value = { workers, modelStates: { falconWebgpu: { status: 'ready' }, ppocrv6Line: { status: 'ready' } }, activeModelKey: 'ppocrv6Line', runOcr };
        const { result } = renderHook(() => useAnnotationOCR({ isSandbox: true, imageRef: { current: {} }, setDebugImageUrl: vi.fn() }));
        await act(async () => { await result.current.runBackgroundOcr({ x: 0, y: 0, w: 10, h: 10 }, 'shared'); });
        expect(runOcr.mock.calls.map(call => call[2])).toEqual(['falconWebgpu', 'ppocrv6Line']);
        expect(result.current.ocrResults.shared.map(candidate => candidate.text)).toEqual(['Falcon text', 'Paddle text']);
    });
});

describe('DeepSeek bubble OCR', () => {
    function props() {
        return { pageId: '1', isSandbox: true, imageRef: { current: {} },
            pendingAnnotation: { id: 'b1', x: 1, y: 2, w: 100, h: 120, texte_propose: 'Existing text' },
            setPendingAnnotation: vi.fn(), setIsSubmitting: vi.fn(), setLoadingText: vi.fn(),
            setIsModalOpen: vi.fn(), setOcrSource: vi.fn(), setDebugImageUrl: vi.fn(), setShowApiKeyModal: vi.fn(),
        };
    }
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        workerContext.value = { modelStates: {}, workers: {} };
        analyzeDeepSeekBubble.mockResolvedValue({ data: { texte_propose: 'Salut ! comment vas-tu ?' } });
    });

    it('supports explicit DeepSeek comparison selection in sandbox without invoking Gemini', async () => {
        localStorage.setItem('sandboxSelectedOcrModelKeys', JSON.stringify(['deepseek']));
        localStorage.setItem('deepseek_api_key', 'sk-deepseek-own-key');
        const options = props();
        const { result } = renderHook(() => useAnnotationOCR(options));
        expect(result.current.selectedOcrModelKeys).toEqual(['deepseek']);
        await act(async () => result.current.runLocalOcr());
        expect(analyzeDeepSeekBubble).toHaveBeenCalledWith(options.imageRef.current, expect.objectContaining({ w: 100, h: 120 }), 'sk-deepseek-own-key', { signal: expect.any(AbortSignal) });
        const update = options.setPendingAnnotation.mock.lastCall[0](options.pendingAnnotation);
        expect(update.texte_propose).toBe('Salut ! Comment vas-tu ?');
        expect(update.ocr_candidates[0].modelKey).toBe('deepseek');
        expect(analyzeBubble).not.toHaveBeenCalled();
    });

    it('offers settings when the selected DeepSeek model lacks a key and never auto-runs on save', async () => {
        localStorage.setItem('sandboxSelectedOcrModelKeys', JSON.stringify(['deepseek']));
        const options = props();
        const { result } = renderHook(() => useAnnotationOCR(options));
        await act(async () => result.current.runLocalOcr());
        expect(options.setShowApiKeyModal).toHaveBeenCalledWith(true);
        expect(analyzeDeepSeekBubble).not.toHaveBeenCalled();
        act(() => { localStorage.setItem('deepseek_api_key', 'sk-key-saved'); window.dispatchEvent(new Event('storage')); });
        expect(result.current.hasDeepSeekKey).toBe(true);
        expect(analyzeDeepSeekBubble).not.toHaveBeenCalled();
    });

    it('can retry a single bubble with DeepSeek even when another engine is selected', async () => {
        localStorage.setItem('deepseek_api_key', 'sk-own-key');
        const options = props();
        const { result } = renderHook(() => useAnnotationOCR(options));
        await act(async () => result.current.handleRetryWithDeepSeek());
        expect(analyzeDeepSeekBubble).toHaveBeenCalledOnce();
        expect(analyzeBubble).not.toHaveBeenCalled();
        expect(options.setOcrSource).toHaveBeenCalledWith('deepseek');
        const update = options.setPendingAnnotation.mock.lastCall[0];
        expect(update({ id: 'different', texte_propose: 'Do not overwrite' }).texte_propose).toBe('Do not overwrite');
        expect(update(null)).toBeNull();
    });

    it('reuses background bubble results during review rather than billing twice', async () => {
        localStorage.setItem('sandboxSelectedOcrModelKeys', JSON.stringify(['deepseek']));
        localStorage.setItem('deepseek_api_key', 'sk-own-key');
        const options = props();
        const { result } = renderHook(() => useAnnotationOCR(options));
        const box = { x: 10, y: 20, w: 100, h: 120 };
        await act(async () => result.current.runBackgroundOcr(box, 'detected-bubble-1'));
        await act(async () => result.current.runLocalOcr(box, 'detected-bubble-1'));
        expect(analyzeDeepSeekBubble).toHaveBeenCalledOnce();
        expect(options.setPendingAnnotation).toHaveBeenCalled();
        // An explicit retry uses a new request ID and still works normally.
        await act(async () => result.current.runLocalOcr(box, 'explicit-retry'));
        expect(analyzeDeepSeekBubble).toHaveBeenCalledTimes(2);
    });

    it('cancels a pending DeepSeek retry when leaving the page', async () => {
        let resolve;
        analyzeDeepSeekBubble.mockImplementation(() => new Promise(res => { resolve = res; }));
        localStorage.setItem('deepseek_api_key', 'sk-own-key');
        const options = props();
        const { result, rerender } = renderHook(input => useAnnotationOCR(input), { initialProps: options });
        let job;
        act(() => { job = result.current.handleRetryWithDeepSeek(); });
        const signal = analyzeDeepSeekBubble.mock.calls[0][3].signal;
        rerender({ ...options, pageId: '2' });
        expect(signal.aborted).toBe(true);
        await act(async () => { resolve({ data: { texte_propose: 'Obsolete' } }); await job; });
        expect(options.setPendingAnnotation).not.toHaveBeenCalled();
        expect(options.setIsSubmitting).toHaveBeenLastCalledWith(false);
    });
});
