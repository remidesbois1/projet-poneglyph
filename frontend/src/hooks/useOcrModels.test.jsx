import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useOcrModels } from './useOcrModels';
import { getChatGptStatus } from '@/lib/chatGptDesktop';
import { generateDeepSeekOneShotBubbles } from '@/lib/deepseekClient';
import { formatProviderModelName } from '@/lib/ocrModels';

const contexts = vi.hoisted(() => ({ worker: {}, tauri: {}, aiConfig: {} }));

vi.mock('@/context/WorkerContext', () => ({ useWorker: () => contexts.worker }));
vi.mock('@/context/TauriLocalOcrContext', () => ({ useTauriLocalOcrContext: () => contexts.tauri }));
vi.mock('@/hooks/useAiModelConfig', () => ({ useAiModelConfig: () => contexts.aiConfig }));
vi.mock('@/lib/chatGptDesktop', () => ({
    getChatGptStatus: vi.fn(async () => ({ available: false })),
    subscribeToChatGptAuth: () => () => {},
    runChatGptPageOcr: vi.fn(),
}));
vi.mock('@/lib/deepseekClient', () => ({ analyzeDeepSeekBubble: vi.fn(), generateDeepSeekOneShotBubbles: vi.fn() }));
vi.mock('@/lib/geminiClient', () => ({ generateOneShotBubbles: vi.fn() }));

const modelOf = (result, key) => result.current.models.find(model => model.key === key);
const keysOf = result => result.current.models.map(model => model.key);

async function renderModels(options) {
    const hook = renderHook(props => useOcrModels(props), { initialProps: options });
    await act(async () => {});
    return hook;
}

beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    getChatGptStatus.mockImplementation(async () => ({ available: false }));
    contexts.worker = { workers: {}, modelStates: {}, loadModel: vi.fn(), runOcr: vi.fn() };
    contexts.tauri = {};
    contexts.aiConfig = {};
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe('local desktop models', () => {
    it('goes from download to load to ready and prepares with the matching action', async () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['lighton2BboxLocal']));
        const downloadLocalModel = vi.fn(async () => ({ ok: true }));
        const loadLocalModel = vi.fn(async () => ({}));
        const tauri = status => ({ isTauri: true, localConnectionState: { status: 'online' }, localModelStatus: status, downloadLocalModel, loadLocalModel });
        contexts.tauri = tauri({ installed: false });
        const { result, rerender } = await renderModels({ allowPageModels: true });

        expect(modelOf(result, 'lighton2BboxLocal').status).toBe('download');
        await act(async () => result.current.prepareSelectedModels());
        expect(downloadLocalModel).toHaveBeenCalledOnce();
        expect(loadLocalModel).not.toHaveBeenCalled();

        contexts.tauri = tauri({ installed: true });
        rerender({ allowPageModels: true });
        expect(modelOf(result, 'lighton2BboxLocal').status).toBe('load');
        await act(async () => result.current.prepareSelectedModels());
        expect(loadLocalModel).toHaveBeenCalledOnce();
        expect(downloadLocalModel).toHaveBeenCalledOnce();

        contexts.tauri = tauri({ installed: true, ready: true });
        rerender({ allowPageModels: true });
        expect(modelOf(result, 'lighton2BboxLocal').status).toBe('ready');
    });

    it('reports an offline local server and prepares nothing', async () => {
        localStorage.setItem('selectedOcrModelKeys', JSON.stringify(['lighton2BboxLocal']));
        const downloadLocalModel = vi.fn(async () => ({ ok: true }));
        const loadLocalModel = vi.fn(async () => ({}));
        contexts.tauri = { isTauri: true, localConnectionState: { status: 'offline' }, localModelStatus: { installed: false }, downloadLocalModel, loadLocalModel };
        const { result } = await renderModels({ allowPageModels: true });

        expect(modelOf(result, 'lighton2BboxLocal')).toMatchObject({ status: 'unavailable', message: 'Serveur local hors ligne' });
        await act(async () => result.current.prepareSelectedModels());
        expect(downloadLocalModel).not.toHaveBeenCalled();
        expect(loadLocalModel).not.toHaveBeenCalled();
    });
});

describe('model availability', () => {
    it('hides Modal and Gemini page reading in the sandbox but keeps DeepSeek', async () => {
        const { result } = await renderModels({ isSandbox: true, allowPageModels: true });
        const keys = keysOf(result);
        expect(keys).not.toContain('lighton');
        expect(keys).not.toContain('lighton2BboxCloud');
        expect(keys).not.toContain('geminiPage');
        expect(keys).toEqual(expect.arrayContaining(['deepseek', 'deepseekPage']));
    });

    it('hides every page model when page models are not allowed', async () => {
        contexts.tauri = { isTauri: true };
        getChatGptStatus.mockImplementation(async () => ({ available: true }));
        const { result } = await renderModels({ allowPageModels: false });
        expect(result.current.models.length).toBeGreaterThan(0);
        expect(result.current.models.filter(model => model.scope === 'page')).toEqual([]);
    });

    it('offers desktop models only inside Tauri', async () => {
        const { result: web } = await renderModels({ allowPageModels: true });
        expect(web.current.models.filter(model => model.engine === 'tauri')).toEqual([]);

        contexts.tauri = { isTauri: true };
        const { result: desktop } = await renderModels({ allowPageModels: true });
        expect(keysOf(desktop)).toEqual(expect.arrayContaining(['poneglyphLocal', 'suryaLocal', 'lighton2BboxLocal', 'suryaBboxLocal']));
    });

    it('offers ChatGPT page reading only once the desktop bridge reports it available', async () => {
        const { result: unavailable } = await renderModels({ allowPageModels: true });
        expect(keysOf(unavailable)).not.toContain('chatgptPage');

        getChatGptStatus.mockImplementation(async () => ({ available: true }));
        const { result } = await renderModels({ allowPageModels: true });
        await waitFor(() => expect(keysOf(result)).toContain('chatgptPage'));
        expect(modelOf(result, 'chatgptPage')).toMatchObject({ status: 'setup', message: 'Connexion ChatGPT requise' });
    });
});

describe('DeepSeek page model', () => {
    it('requires a key and never reaches the paid API without one', async () => {
        const { result } = await renderModels({ allowPageModels: true });
        const model = modelOf(result, 'deepseekPage');
        expect(model).toMatchObject({ status: 'setup', message: 'Clé DeepSeek requise' });
        await expect(result.current.runPageModel(model, { naturalWidth: 10, naturalHeight: 10 })).rejects.toThrow();
        expect(generateDeepSeekOneShotBubbles).not.toHaveBeenCalled();
    });
});

describe('standardized model names', () => {
    it('names Gemini page reading after the configured provider model', async () => {
        contexts.aiConfig = { model_ocr: 'gemini-3.1-flash-lite' };
        const { result } = await renderModels({ allowPageModels: true });
        expect(modelOf(result, 'geminiPage')).toMatchObject({ name: 'Gemini 3.1 Flash Lite', label: 'Gemini 3.1 Flash Lite · Cloud' });
    });

    it('formats GPT model identifiers', () => {
        expect(formatProviderModelName('gpt-5.6-luna')).toBe('GPT-5.6 Luna');
    });
});

describe('browser page model', () => {
    const bbox = [100, 200, 300, 400];

    function setUpWorker(onRun) {
        const worker = new EventTarget();
        worker.postMessage = vi.fn();
        const runOcr = vi.fn(async (_bitmap, requestId) => onRun?.(worker, requestId));
        contexts.worker = { workers: { lighton3Webgpu: worker }, modelStates: { lighton3Webgpu: { status: 'ready' } }, loadModel: vi.fn(), runOcr };
        vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ close: vi.fn() })));
        return { worker, runOcr };
    }

    it('returns the page bubbles with the transcript untouched', async () => {
        const { runOcr } = setUpWorker((worker, requestId) => queueMicrotask(() => worker.dispatchEvent(new MessageEvent('message', {
            data: { status: 'complete', requestId, bubbles: [{ bbox, content: 'étonnant !' }] },
        }))));
        const { result } = await renderModels({ allowPageModels: true });
        let bubbles;
        await act(async () => { bubbles = await result.current.runPageModel(modelOf(result, 'lighton3Webgpu'), { naturalWidth: 10, naturalHeight: 10 }); });
        expect(runOcr.mock.calls[0][2]).toBe('lighton3Webgpu');
        expect(bubbles).toEqual([{ bbox, content: 'étonnant !' }]);
    });

    it('cancels the worker request when aborted before completion', async () => {
        const { worker, runOcr } = setUpWorker();
        const { result } = await renderModels({ allowPageModels: true });
        const controller = new AbortController();
        const job = result.current.runPageModel(modelOf(result, 'lighton3Webgpu'), { naturalWidth: 10, naturalHeight: 10 }, { signal: controller.signal });
        await waitFor(() => expect(runOcr).toHaveBeenCalled());
        const requestId = runOcr.mock.calls[0][1];
        controller.abort();
        await expect(job).rejects.toMatchObject({ name: 'AbortError' });
        expect(worker.postMessage).toHaveBeenCalledWith({ type: 'cancel', requestId });
    });
});
