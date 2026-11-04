import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { generateGlmOneShotBubbles, parseGlmBubbles } from './mistralClient';
import { cropImage } from './utils';
import { getPrompt } from './promptConfig';

vi.mock('./utils', () => ({ cropImage: vi.fn() }));
vi.mock('./promptConfig', () => ({ getPrompt: vi.fn(async key => `Prompt ${key}`) }));
const image = { naturalWidth: 1000, naturalHeight: 1500 };
const key = 'mistral-test-key-1234';
let fetchMock;
beforeEach(() => {
    vi.clearAllMocks();
    cropImage.mockResolvedValue(new Blob(['image'], { type: 'image/jpeg' }));
    fetchMock = vi.fn(async () => Response.json({ text: '{"bubbles":[]}', model: 'zai-glm-5-3-flash' }));
    vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe('Mistral GLM OCR client', () => {
    it('uses the whole original page and an independent key header rather than the Gemini key', async () => {
        localStorage.setItem('google_api_key', 'GOOGLE-SECRET');
        localStorage.setItem('deepseek_api_key', 'sk-deepseek-secret');
        fetchMock.mockResolvedValue(Response.json({ text: '{"bubbles":[{"content":"bonjour","bbox":[100,300,400,900]}]}', model: 'zai-glm-5-3-flash' }));
        const result = await generateGlmOneShotBubbles(image, key);
        expect(result.data).toEqual([{ content: 'bonjour', bbox: [100,200,400,600] }]);
        expect(cropImage).toHaveBeenCalledWith(image, { x: 0, y: 0, w: 1000, h: 1500 });
        expect(getPrompt).toHaveBeenCalledWith('ocr_page_bbox');
        expect(getPrompt).toHaveBeenCalledWith('strict_json_suffix');
        const [endpoint, options] = fetchMock.mock.calls[0];
        expect(endpoint).toBe('/api/mistral');
        expect(options.headers['X-Mistral-API-Key']).toBe(key);
        expect(options.body).not.toContain(key);
        expect(JSON.stringify(options)).not.toContain('GOOGLE-SECRET');
        expect(JSON.stringify(options)).not.toContain('sk-deepseek-secret');
        expect(JSON.parse(options.body)).toMatchObject({ image: expect.stringMatching(/^data:image\/jpeg;base64,/) });
    });

    it('does not upload an image with a missing key or invalid dimensions', async () => {
        await expect(generateGlmOneShotBubbles(image, '')).rejects.toThrow('Clé API');
        await expect(generateGlmOneShotBubbles({ naturalWidth: 0, naturalHeight: 100 }, key)).rejects.toThrow('invalide');
        await expect(generateGlmOneShotBubbles({ naturalWidth: 10000, naturalHeight: 100 }, key)).rejects.toThrow('trop grande');
        expect(cropImage).not.toHaveBeenCalled();
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it('surfaces provider errors without silently falling back to another provider', async () => {
        fetchMock.mockResolvedValue(Response.json({ error: 'Crédits Mistral insuffisants.' }, { status: 402 }));
        await expect(generateGlmOneShotBubbles(image, key)).rejects.toThrow('Crédits Mistral');
        expect(fetchMock).toHaveBeenCalledOnce();
    });

    it('honors an already aborted request without cropping or calling a provider', async () => {
        const controller = new AbortController();
        controller.abort();
        await expect(generateGlmOneShotBubbles(image, key, { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
        expect(cropImage).not.toHaveBeenCalled();
        expect(fetchMock).not.toHaveBeenCalled();
    });
});

describe('GLM bbox parsing', () => {
    const dims = { width: 1000, height: 2000 };

    it('rescales pixel coordinates to the 0-1000 page convention and accepts an explicitly empty page', () => {
        expect(parseGlmBubbles('```json\n{"bubbles":[{"content":"Salut","bbox":[0,0,1000,2000]}]}\n```', dims))
            .toEqual([{ content: 'Salut', bbox: [0,0,1000,1000] }]);
        expect(parseGlmBubbles('{"bubbles":[]}', dims)).toEqual([]);
    });

    it('clamps pixel coordinates beyond the image bounds instead of failing the whole page', () => {
        expect(parseGlmBubbles('{"bubbles":[{"content":"Débordement","bbox":[0,20,1010,60]}]}', dims))
            .toEqual([{ content: 'Débordement', bbox: [0,10,1000,30] }]);
    });

    it('drops only the bubbles whose geometry stays invalid after rescaling', () => {
        expect(parseGlmBubbles('{"bubbles":[{"content":"ok","bbox":[10000,20,10500,60]},{"content":"Valide","bbox":[100,20,500,60]}]}', dims))
            .toEqual([{ content: 'Valide', bbox: [100,10,500,30] }]);
        expect(parseGlmBubbles('{"bubbles":[{"content":"Hors page","bbox":[1100,2100,1300,2400]}]}', dims)).toEqual([]);
    });

    it('falls back to clamping raw values when no image dimensions are provided', () => {
        expect(parseGlmBubbles('{"bubbles":[{"content":"Legacy","bbox":[0,20,1010,30]}]}'))
            .toEqual([{ content: 'Legacy', bbox: [0,20,1000,30] }]);
    });

    it.each([
        '{"bubbles":[', '{}', 'null', 'Commentaire hors JSON',
        '{"bubbles":[{"content":"ok","bbox":[1,2,3]}]}',
        '{"bubbles":[{"content":"ok","bbox":["0",20,30,40]}]}',
        '{"bubbles":[{"content":"","bbox":[0,0,30,40]}]}',
    ])('rejects malformed/unsafe output without salvaging partial annotations (%#)', text => {
        expect(() => parseGlmBubbles(text, dims)).toThrow();
    });
});
