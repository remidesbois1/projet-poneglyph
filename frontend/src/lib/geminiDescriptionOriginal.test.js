import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ generate: vi.fn(), crop: vi.fn() }));
vi.mock('@google/generative-ai', () => ({
    GoogleGenerativeAI: class { getGenerativeModel() { return { generateContent: mocks.generate }; } },
}));
vi.mock('./utils', () => ({ cropImage: mocks.crop }));
vi.mock('./aiModelConfig', () => ({ getAiModelConfig: vi.fn().mockResolvedValue({ model_description: 'gemini-description', gemini_thinking_level: 'default' }) }));
vi.mock('./promptConfig', () => ({ getPrompt: vi.fn().mockResolvedValue('prompt') }));

import { generateGeminiEmbedding, generatePageDescription } from './geminiClient';

describe('Gemini original page blobs', () => {
    beforeEach(() => vi.clearAllMocks());

    it('sends original bytes and MIME unchanged to description generation', async () => {
        const description = { content: 'Action visible', metadata: { arc: '', characters: [] } };
        mocks.generate.mockResolvedValue({ response: { text: () => JSON.stringify(description) } });
        await expect(generatePageDescription(new Blob(['original'], { type: 'image/png' }), 'google-key'))
            .resolves.toEqual({ data: description });
        expect(mocks.generate).toHaveBeenCalledWith([
            'prompt\n\nprompt',
            { inlineData: { data: 'b3JpZ2luYWw=', mimeType: 'image/png' } },
        ]);
        expect(mocks.crop).not.toHaveBeenCalled();
    });

    it('sends original bytes and MIME unchanged to multimodal Gemini embedding', async () => {
        const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ embedding: { values: [0.1] } }) });
        try {
            await generateGeminiEmbedding('Description', new Blob(['original'], { type: 'image/webp' }), 'google-key');
            const body = JSON.parse(fetchMock.mock.calls[0][1].body);
            expect(body.content.parts[1]).toEqual({ inlineData: { data: 'b3JpZ2luYWw=', mimeType: 'image/webp' } });
            expect(mocks.crop).not.toHaveBeenCalled();
        } finally { fetchMock.mockRestore(); }
    });
});
