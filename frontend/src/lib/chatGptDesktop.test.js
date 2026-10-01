import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));

vi.mock('./aiModelConfig', () => ({
    getAiModelConfig: vi.fn().mockResolvedValue({
        model_chatgpt_ocr: 'gpt-5.6-luna',
        chatgpt_reasoning_effort: 'low',
        chatgpt_fast_mode: false,
    }),
}));

vi.mock('./promptConfig', () => ({
    getPrompt: vi.fn().mockResolvedValue('prompt-ocr-page'),
}));

vi.mock('@tauri-apps/api/core', () => ({
    invoke,
    isTauri: () => true,
}));

import {
    getChatGptStatus,
    logoutChatGpt,
    runChatGptPageOcr,
    runChatGptPageDescription,
} from './chatGptDesktop';

describe('chatGptDesktop', () => {
    beforeEach(() => {
        invoke.mockReset();
    });

    afterEach(() => vi.unstubAllGlobals());

    it('reads the in-memory desktop authentication status', async () => {
        invoke.mockResolvedValue({ connected: true, email: 'reader@example.com', model: 'gpt-5.6-luna' });
        await expect(getChatGptStatus()).resolves.toMatchObject({
            available: true,
            connected: true,
            model: 'gpt-5.6-luna',
        });
        expect(invoke).toHaveBeenCalledWith('get_chatgpt_auth_status', {});
    });

    it('sends an image only through the scoped Tauri OCR command', async () => {
        invoke.mockResolvedValue({ bubbles: [] });
        const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' });
        await runChatGptPageOcr(blob);
        expect(invoke).toHaveBeenCalledWith('run_chatgpt_page_ocr', {
            image_bytes_base64: 'AQID',
            mime_type: 'image/png',
            model: 'gpt-5.6-luna',
            fast_mode: false,
            reasoning_effort: 'low',
            prompt: 'prompt-ocr-page',
        });
    });

    it('can override the global OpenAI OCR settings and prompt for a request', async () => {
        invoke.mockResolvedValue({ bubbles: [] });
        await runChatGptPageOcr(new Blob(['image'], { type: 'image/jpeg' }), {
            model: 'gpt-5.6-terra',
            fastMode: true,
            reasoningEffort: 'high',
            prompt: 'prompt-personnalise',
        });
        expect(invoke).toHaveBeenCalledWith('run_chatgpt_page_ocr', expect.objectContaining({
            model: 'gpt-5.6-terra',
            fast_mode: true,
            reasoning_effort: 'high',
            prompt: 'prompt-personnalise',
        }));
    });

    it('clears the desktop session without touching browser storage', async () => {
        invoke.mockResolvedValue({ connected: false, model: 'gpt-5.6-luna' });
        await logoutChatGpt();
        expect(invoke).toHaveBeenCalledWith('chatgpt_logout', {});
    });

    it('sends the original blob via OAuth Desktop with fixed Astra/high and strict JSON suffix', async () => {
        const description = { content: 'Luffy attaque.', metadata: { arc: '', characters: ['Luffy'] } };
        invoke.mockResolvedValue({ text: JSON.stringify(description) });
        await expect(runChatGptPageDescription(new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' })))
            .resolves.toEqual(description);
        expect(invoke).toHaveBeenCalledWith('run_chatgpt_page_description', {
            image_bytes_base64: 'AQID', mime_type: 'image/png',
            model: 'gpt-6-astra', reasoning_effort: 'high',
            prompt: 'prompt-ocr-page\n\nprompt-ocr-page',
        });
        expect(invoke.mock.calls[0][1]).not.toHaveProperty('api_key');
    });

    it.each([
        ['description', runChatGptPageDescription, 'run_chatgpt_page_description'],
        ['OCR', runChatGptPageOcr, 'run_chatgpt_page_ocr'],
    ])('converts AVIF before the %s Tauri call', async (_, run, command) => {
        const description = { content: 'Action visible.', metadata: { arc: '', characters: [] } };
        invoke.mockResolvedValue({ text: JSON.stringify(description), bubbles: [] });
        const bitmap = { width: 1200, height: 1800, close: vi.fn() };
        vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap));
        vi.stubGlobal('OffscreenCanvas', vi.fn(function () {
            return {
                getContext: () => ({ drawImage: vi.fn() }),
                convertToBlob: async () => new Blob([new Uint8Array([4, 5, 6])], { type: 'image/png' }),
            };
        }));
        const original = new Blob(['avif-original'], { type: 'image/avif' });
        await run(original);
        expect(createImageBitmap).toHaveBeenCalledWith(original);
        expect(invoke).toHaveBeenCalledWith(command, expect.objectContaining({
            image_bytes_base64: 'BAUG', mime_type: 'image/png',
        }));
    });

    it('does not call ChatGPT when the original cannot be decoded', async () => {
        vi.stubGlobal('createImageBitmap', vi.fn().mockRejectedValue(new Error('Invalid AVIF')));
        await expect(runChatGptPageDescription(new Blob(['broken'], { type: 'image/avif' })))
            .rejects.toThrow(/convertir l’image originale en PNG/);
        expect(invoke).not.toHaveBeenCalled();
    });

    it.each([
        'invalid json', '[]', '{}',
        '{"content":"","metadata":{"arc":"","characters":[]}}',
        '{"content":"Action","metadata":{"arc":"","characters":"Luffy"}}',
        '{"content":"Action","metadata":{"arc":"","characters":[]},"extra":"invented"}',
        '```json\n{"content":"Action","metadata":{"arc":"","characters":[]}}\n```',
    ])('rejects malformed or nonconforming Astra descriptions: %s', async (text) => {
        invoke.mockResolvedValue({ text });
        await expect(runChatGptPageDescription(new Blob(['original'], { type: 'image/png' }))).rejects.toThrow(/Description invalide/);
    });
});
