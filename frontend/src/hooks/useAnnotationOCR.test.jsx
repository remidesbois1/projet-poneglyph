import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAnnotationOCR } from './useAnnotationOCR';
import { analyzeDeepSeekBubble, generateDeepSeekOneShotBubbles } from '@/lib/deepseekClient';
import { analyzeBubble } from '@/lib/geminiClient';

const workerContext = vi.hoisted(() => ({ value: {} }));
vi.mock('@/lib/utils', async (importOriginal) => ({ ...await importOriginal(), cropImageBitmap: vi.fn(async () => ({ close: vi.fn() })) }));

vi.mock('@/context/WorkerContext', async (importOriginal) => ({
    ...await importOriginal(),
    useWorker: () => workerContext.value,
}));
vi.mock('@/context/TauriLocalOcrContext', () => ({ useTauriLocalOcrContext: () => ({}) }));
vi.mock('@/hooks/useAiModelConfig', () => ({ useAiModelConfig: () => aiModelConfig }));
vi.mock('@/lib/chatGptDesktop', () => ({
    getChatGptStatus: vi.fn(async () => ({ available: false })),
    subscribeToChatGptAuth: () => () => {},
    runChatGptPageOcr: vi.fn(),
}));
vi.mock('@/lib/deepseekClient', () => ({ analyzeDeepSeekBubble: vi.fn(), generateDeepSeekOneShotBubbles: vi.fn() }));
vi.mock('@/lib/geminiClient', () => ({ analyzeBubble: vi.fn(), generateOneShotBubbles: vi.fn() }));

const aiModelConfig = {};

function editorProps(overrides = {}) {
    return { pageId: '1', imageRef: { current: {} },
        pendingAnnotation: { id: 'b1', x: 1, y: 2, w: 100, h: 120, texte_propose: 'Existing text' },
        setPendingAnnotation: vi.fn(), setIsSubmitting: vi.fn(), setLoadingText: vi.fn(),
        setIsModalOpen: vi.fn(), setOcrSource: vi.fn(), setDebugImageUrl: vi.fn(), setShowApiKeyModal: vi.fn(),
        ...overrides,
    };
}

/** Browser workers that answer each request with the given text per model. */
function browserWorkers(texts) {
    const workers = Object.fromEntries(Object.keys(texts).map(key => [key, new EventTarget()]));
    const runOcr = vi.fn(async (_bitmap, requestId, key) => {
        queueMicrotask(() => workers[key].dispatchEvent(new MessageEvent('message', {
            data: { status: 'complete', requestId, text: texts[key] }
        })));
    });
    const modelStates = Object.fromEntries(Object.keys(texts).map(key => [key, { status: 'ready' }]));
    workerContext.value = { workers, modelStates, loadModel: vi.fn(), runOcr };
    return runOcr;
}

/** The annotation the editor ends up showing, from the last updater given to `setPendingAnnotation`. */
function editedAnnotation(options) {
    return options.setPendingAnnotation.mock.lastCall[0](options.pendingAnnotation);
}

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


describe('bubble OCR in the editor', () => {
    beforeEach(() => { vi.clearAllMocks(); localStorage.clear(); });

    it('reads a bubble with bubble models only and keeps the LightOnOCR 3 transcript intact', async () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['lighton3BubbleWebgpu', 'lighton3Webgpu', 'deepseekPage']));
        const runOcr = browserWorkers({ lighton3BubbleWebgpu: '"BOUM !" ... oui ?', lighton3Webgpu: 'Page model text' });
        const options = editorProps({ allowPageModels: true });
        const { result } = renderHook(() => useAnnotationOCR(options));
        await act(async () => result.current.runLocalOcr({ x: 0, y: 0, w: 10, h: 10 }, 'bubble3'));
        expect(runOcr).toHaveBeenCalledOnce();
        expect(runOcr.mock.calls[0][2]).toBe('lighton3BubbleWebgpu');
        expect(generateDeepSeekOneShotBubbles).not.toHaveBeenCalled();
        const annotation = editedAnnotation(options);
        expect(annotation.texte_propose).toBe('"BOUM !" ... oui ?');
        expect(annotation.ocr_candidates.map(candidate => candidate.modelKey)).toEqual(['lighton3BubbleWebgpu']);
    });

    it('collects replies from both loaded browser models', async () => {
        localStorage.setItem('sandboxSelectedOcrModelKeys', JSON.stringify(['falconWebgpu', 'ppocrv6Line']));
        const runOcr = browserWorkers({ falconWebgpu: 'Falcon text', ppocrv6Line: 'Paddle text' });
        const options = editorProps({ isSandbox: true });
        const { result } = renderHook(() => useAnnotationOCR(options));
        await act(async () => result.current.runLocalOcr({ x: 0, y: 0, w: 10, h: 10 }, 'shared'));
        expect(runOcr.mock.calls.map(call => call[2]).sort()).toEqual(['falconWebgpu', 'ppocrv6Line']);
        expect(editedAnnotation(options).ocr_candidates.map(candidate => candidate.text).sort()).toEqual(['Falcon text', 'Paddle text']);
    });

    it('offers page candidates after the bubble model candidates', async () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['ppocrv6Line', 'lighton3Webgpu']));
        browserWorkers({ ppocrv6Line: 'Bubble text' });
        const pageCandidates = [{ modelKey: 'lighton3Webgpu', label: 'LightOnOCR 3 BBox · Navigateur', text: 'Page text' }];
        const options = editorProps({ allowPageModels: true });
        const { result } = renderHook(() => useAnnotationOCR(options));
        await act(async () => result.current.runLocalOcr({ x: 0, y: 0, w: 10, h: 10 }, 'detected-1', { pageCandidates }));
        const annotation = editedAnnotation(options);
        expect(annotation.ocr_candidates.map(candidate => [candidate.modelKey, candidate.text])).toEqual([
            ['ppocrv6Line', 'Bubble text'],
            ['lighton3Webgpu', 'Page text'],
        ]);
        expect(annotation.texte_propose).toBe('Bubble text');
        expect(options.setIsModalOpen).toHaveBeenCalledWith(true);
    });

    it('opens the editor with the page candidates alone when no bubble model is selected', async () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['lighton3Webgpu']));
        const runOcr = browserWorkers({ lighton3Webgpu: 'unused' });
        const pageCandidates = [
            { modelKey: 'lighton3Webgpu', label: 'LightOnOCR 3 BBox · Navigateur', text: 'Première lecture' },
            { modelKey: 'deepseekPage', label: 'DeepSeek V4.1 Flash · Cloud', text: 'Seconde lecture' },
        ];
        const options = editorProps({ allowPageModels: true });
        const { result } = renderHook(() => useAnnotationOCR(options));
        await act(async () => result.current.runLocalOcr({ x: 0, y: 0, w: 10, h: 10 }, 'detected-1', { pageCandidates }));
        expect(runOcr).not.toHaveBeenCalled();
        expect(options.setIsModalOpen).toHaveBeenCalledWith(true);
        const annotation = editedAnnotation(options);
        expect(annotation.ocr_candidates).toEqual(pageCandidates);
        expect(annotation.texte_propose).toBe('Première lecture');
    });
});

describe('DeepSeek bubble OCR', () => {
    function props() {
        return editorProps({ isSandbox: true });
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
        expect(result.current.models.find(model => model.key === 'deepseek').status).toBe('setup');
        await act(async () => { localStorage.setItem('deepseek_api_key', 'sk-key-saved'); window.dispatchEvent(new Event('storage')); });
        expect(result.current.models.find(model => model.key === 'deepseek').status).toBe('ready');
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
