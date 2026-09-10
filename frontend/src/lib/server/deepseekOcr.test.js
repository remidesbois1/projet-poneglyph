// @vitest-environment node
import { beforeAll, describe, expect, it, vi } from 'vitest';
import sharp from 'sharp';
import { createDeepSeekOcrHandler } from './deepseekOcr';
import { DEEPSEEK_MODEL } from '../deepseekConfig';

const key = 'sk-test-deepseek-only';
let image;
beforeAll(async () => {
    const bytes = await sharp({ create: { width: 20, height: 30, channels: 3, background: '#fff' } }).png().toBuffer();
    image = `data:image/png;base64,${bytes.toString('base64')}`;
});
const request = (input = {}, headers = {}, options = {}) => new Request('https://poneglyph.fr/api/deepseek', {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-DeepSeek-API-Key': key, Origin: 'https://poneglyph.fr', ...headers },
    body: JSON.stringify({ mode: 'bubble', prompt: 'Transcribe this image.', image, ...input }), ...options,
});
const completion = (content = 'Bonjour', finish_reason = 'stop') => Response.json({
    choices: [{ finish_reason, message: { content, reasoning_content: 'Private reasoning' } }],
});

describe('DeepSeek BYOK proxy', () => {
    it('sends images only to the fixed official endpoint with the supplied DeepSeek key', async () => {
        const fetchImpl = vi.fn(async () => completion());
        const response = await createDeepSeekOcrHandler({ fetchImpl })(request({ model: 'untrusted', endpoint: 'https://attacker.invalid' }, { Authorization: 'Bearer SUPABASE_TOKEN', Cookie: 'session=private' }));
        expect(response.status).toBe(200);
        expect(await response.json()).toEqual({ text: 'Bonjour', model: DEEPSEEK_MODEL });
        expect(response.headers.get('cache-control')).toContain('no-store');
        expect(fetchImpl).toHaveBeenCalledOnce();
        const [url, options] = fetchImpl.mock.calls[0];
        expect(url).toBe('https://api.deepseek.com/chat/completions');
        expect(options.headers).toEqual({ 'Content-Type': 'application/json', Authorization: `Bearer ${key}` });
        expect(options.redirect).toBe('error');
        expect(options.credentials).toBe('omit');
        const sent = JSON.parse(options.body);
        expect(sent.model).toBe('deepseek-flash');
        expect(sent.thinking).toEqual({ type: 'disabled' });
        expect(sent.response_format).toBeUndefined();
        expect(sent.messages[0].content[1]).toEqual({ type: 'image_url', image_url: { url: image, detail: 'original' } });
    });

    it('requests strict JSON and the normalized bbox convention for full-page OCR', async () => {
        const fetchImpl = vi.fn(async () => completion('{"bubbles":[]}'));
        await createDeepSeekOcrHandler({ fetchImpl })(request({ mode: 'bbox' }));
        const sent = JSON.parse(fetchImpl.mock.calls[0][1].body);
        expect(sent.response_format).toEqual({ type: 'json_object' });
        expect(sent.max_tokens).toBe(16384);
        expect(sent.messages[0].content[0].text).toContain('[x1,y1,x2,y2]');
    });

    it.each([
        [{}, { 'X-DeepSeek-API-Key': '' }, 401],
        [{}, { Origin: 'https://attacker.invalid' }, 403],
        [{}, { 'Sec-Fetch-Site': 'cross-site' }, 403],
        [{}, { 'Content-Type': 'text/plain' }, 415],
        [{}, { 'Content-Encoding': 'gzip' }, 415],
        [{}, { 'Content-Length': '20000000' }, 413],
        [{ mode: 'chat' }, {}, 400],
        [{ prompt: '' }, {}, 400],
        [{ prompt: 'a'.repeat(50001) }, {}, 400],
        [{ image: 'https://private.invalid/image.png' }, {}, 400],
        [{ image: 'data:image/png;base64,aW52YWxpZA==' }, {}, 400],
    ])('rejects invalid requests before upstream access (%#)', async (input, headers, status) => {
        const fetchImpl = vi.fn();
        const response = await createDeepSeekOcrHandler({ fetchImpl })(request(input, headers));
        expect(response.status).toBe(status);
        expect(fetchImpl).not.toHaveBeenCalled();
        expect(await response.text()).not.toContain(key);
    });

    it('rejects invalid JSON, MIME mismatches, and oversized streamed bodies', async () => {
        const fetchImpl = vi.fn();
        const handler = createDeepSeekOcrHandler({ fetchImpl });
        expect((await handler(request({}, {}, { body: '{' }))).status).toBe(400);
        expect((await handler(request({ image: image.replace('image/png', 'image/jpeg') }))).status).toBe(400);
        expect((await handler(request({}, {}, { body: 'x'.repeat(15_000_000) }))).status).toBe(413);
        expect(fetchImpl).not.toHaveBeenCalled();
    });

    it.each([400, 401, 402, 403, 404, 429, 500])('sanitizes provider errors and never retries (%i)', async status => {
        const fetchImpl = vi.fn(async () => Response.json({ error: `${key} private-image private-prompt` }, { status }));
        const response = await createDeepSeekOcrHandler({ fetchImpl })(request());
        expect(response.status).toBe(status === 500 ? 502 : status);
        const text = await response.text();
        expect(text).not.toContain(key);
        expect(text).not.toContain('private-image');
        expect(fetchImpl).toHaveBeenCalledOnce();
    });

    it.each(['length', 'content_filter', 'tool_calls', 'aborted'])('rejects incomplete completions (%s)', async reason => {
        const fetchImpl = vi.fn(async () => completion('{"bubbles":[', reason));
        expect((await createDeepSeekOcrHandler({ fetchImpl })(request())).status).toBe(502);
    });

    it('bounds provider response size and rejects non-JSON envelopes', async () => {
        const fetchImpl = vi.fn().mockResolvedValueOnce(new Response('x'.repeat(1_100_000))).mockResolvedValueOnce(new Response('<html>error</html>'));
        const handler = createDeepSeekOcrHandler({ fetchImpl });
        expect((await handler(request())).status).toBe(502);
        expect((await handler(request())).status).toBe(502);
    });

    it('times out upstream calls and cancels when the client disconnects', async () => {
        const fetchImpl = vi.fn((_url, { signal }) => new Promise((_resolve, reject) => {
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
        }));
        const timeoutResponse = await createDeepSeekOcrHandler({ fetchImpl, timeoutMs: 30 })(request());
        expect(timeoutResponse.status).toBe(504);
        const controller = new AbortController();
        controller.abort();
        const response = await createDeepSeekOcrHandler({ fetchImpl })(request({}, {}, { signal: controller.signal }));
        expect(response.status).toBe(499);
    });
});
