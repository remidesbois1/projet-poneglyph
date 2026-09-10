import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { analyzeDeepSeekBubble, generateDeepSeekOneShotBubbles, parseDeepSeekBubbles } from './deepseekClient';
import { cropImage } from './utils';
import { getPrompt } from './promptConfig';

vi.mock('./utils', () => ({ cropImage: vi.fn() }));
vi.mock('./promptConfig', () => ({ getPrompt: vi.fn(async key => `Prompt ${key}`) }));
const image = { naturalWidth: 1000, naturalHeight: 1500 };
const rect = { x: 10, y: 20, w: 200, h: 150 };
const key = 'sk-deepseek-test';
let fetchMock;
beforeEach(() => {
    vi.clearAllMocks();
    cropImage.mockResolvedValue(new Blob(['image'], { type: 'image/jpeg' }));
    fetchMock = vi.fn(async () => Response.json({ text: 'Bonjour', model: 'deepseek-flash' }));
    vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe('DeepSeek OCR client', () => {
    it('crops a bubble and uses an independent key header rather than the Gemini key', async () => {
        localStorage.setItem('google_api_key', 'GOOGLE-SECRET');
        const result = await analyzeDeepSeekBubble(image, rect, key);
        expect(result.data.texte_propose).toBe('Bonjour');
        expect(cropImage).toHaveBeenCalledWith(image, rect);
        expect(getPrompt).toHaveBeenCalledWith('ocr_bubble');
        const [endpoint, options] = fetchMock.mock.calls[0];
        expect(endpoint).toBe('/api/deepseek');
        expect(options.headers['X-DeepSeek-API-Key']).toBe(key);
        expect(options.body).not.toContain(key);
        expect(JSON.stringify(options)).not.toContain('GOOGLE-SECRET');
        expect(JSON.parse(options.body)).toMatchObject({ mode: 'bubble', image: expect.stringMatching(/^data:image\/jpeg;base64,/) });
    });

    it('uses the whole original page and shares the configured bbox prompt', async () => {
        fetchMock.mockResolvedValue(Response.json({ text: '{"bubbles":[{"content":"bonjour","bbox":[100,200,400,600]}]}' }));
        const result = await generateDeepSeekOneShotBubbles(image, key);
        expect(cropImage).toHaveBeenCalledWith(image, { x: 0, y: 0, w: 1000, h: 1500 });
        expect(result.data).toEqual([{ content: 'bonjour', bbox: [100,200,400,600] }]);
        expect(JSON.parse(fetchMock.mock.calls[0][1].body).prompt).toContain('ocr_page_bbox');
        expect(getPrompt).toHaveBeenCalledWith('strict_json_suffix');
    });

    it('does not upload an image with a missing key or invalid dimensions', async () => {
        await expect(analyzeDeepSeekBubble(image, rect, '')).rejects.toThrow('Clé API');
        await expect(analyzeDeepSeekBubble(image, { ...rect, w: -1 }, key)).rejects.toThrow('invalide');
        await expect(generateDeepSeekOneShotBubbles({ naturalWidth: 10000, naturalHeight: 100 }, key)).rejects.toThrow('trop grande');
        expect(cropImage).not.toHaveBeenCalled();
        expect(fetchMock).not.toHaveBeenCalled();
    });

    it('surfaces provider errors without silently falling back to another provider', async () => {
        fetchMock.mockResolvedValue(Response.json({ error: 'Solde DeepSeek insuffisant.' }, { status: 402 }));
        await expect(analyzeDeepSeekBubble(image, rect, key)).rejects.toThrow('Solde DeepSeek');
        expect(fetchMock).toHaveBeenCalledOnce();
    });

    it('honors an already aborted request without cropping or calling a provider', async () => {
        const controller = new AbortController();
        controller.abort();
        await expect(analyzeDeepSeekBubble(image, rect, key, { signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
        expect(cropImage).not.toHaveBeenCalled();
        expect(fetchMock).not.toHaveBeenCalled();
    });
});

describe('DeepSeek bbox parsing', () => {
    it('accepts a valid fenced response and an explicitly empty page', () => {
        expect(parseDeepSeekBubbles('```json\n{"bubbles":[{"content":"Salut","bbox":[0,0,1000,1000]}]}\n```')).toEqual([{ content: 'Salut', bbox: [0,0,1000,1000] }]);
        expect(parseDeepSeekBubbles('{"bubbles":[]}')).toEqual([]);
    });
    it.each([
        '{"bubbles":[', '{}', 'null', 'Commentaire hors JSON',
        '{"bubbles":[{"content":"ok","bbox":[1,2,3]}]}',
        '{"bubbles":[{"content":"ok","bbox":[10,20,5,30]}]}',
        '{"bubbles":[{"content":"ok","bbox":[0,20,1001,30]}]}',
        '{"bubbles":[{"content":"ok","bbox":["0",20,30,40]}]}',
        '{"bubbles":[{"content":"","bbox":[0,0,30,40]}]}',
    ])('rejects malformed/unsafe output without salvaging partial annotations (%#)', text => {
        expect(() => parseDeepSeekBubbles(text)).toThrow();
    });
});
