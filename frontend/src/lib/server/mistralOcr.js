import sharp from 'sharp';
import { MISTRAL_IMAGE_MAX_BYTES, MISTRAL_MODEL, MISTRAL_TIMEOUT_MS } from '../mistralConfig';

const ENDPOINT = 'https://api.mistral.ai/v1/chat/completions';
const MAX_BODY_BYTES = Math.ceil(MISTRAL_IMAGE_MAX_BYTES * 4 / 3) + 128_000;
const MAX_RESPONSE_BYTES = 1024 * 1024;
const HEADERS = { 'Cache-Control': 'private, no-store, max-age=0', 'X-Content-Type-Options': 'nosniff' };

class RequestError extends Error {
    constructor(status, message) {
        super(message);
        this.status = status;
    }
}

async function readBoundedBody(body, limit, signal) {
    signal?.throwIfAborted();
    if (!body) throw new RequestError(400, 'Requête vide.');
    const reader = body.getReader();
    const chunks = [];
    let length = 0;
    const cancel = () => { void reader.cancel().catch(() => {}); };
    signal?.addEventListener('abort', cancel, { once: true });
    try {
        while (true) {
            const { done, value } = await reader.read();
            signal?.throwIfAborted();
            if (done) break;
            length += value.byteLength;
            if (length > limit) {
                cancel();
                throw new RequestError(413, 'Données trop volumineuses.');
            }
            chunks.push(Buffer.from(value));
        }
        return Buffer.concat(chunks, length).toString('utf8');
    } finally {
        signal?.removeEventListener('abort', cancel);
        reader.releaseLock();
    }
}

async function validateImage(dataUrl) {
    if (typeof dataUrl !== 'string' || dataUrl.length > MAX_BODY_BYTES) {
        throw new RequestError(400, 'Image requise.');
    }
    const match = /^data:image\/(jpeg|png|webp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(dataUrl);
    if (!match || match[2].length % 4 !== 0) {
        throw new RequestError(400, 'Image JPEG, PNG ou WebP encodée en base64 requise.');
    }
    const bytes = Buffer.from(match[2], 'base64');
    if (bytes.length > MISTRAL_IMAGE_MAX_BYTES) throw new RequestError(413, 'Image limitée à 10 Mo.');
    let metadata;
    try {
        metadata = await sharp(bytes, { limitInputPixels: 24_000_000, animated: false }).metadata();
    } catch {
        throw new RequestError(400, 'Image invalide ou trop grande.');
    }
    if (metadata.format !== match[1] || !metadata.width || !metadata.height || (metadata.pages || 1) !== 1) {
        throw new RequestError(400, 'Image invalide.');
    }
    if (metadata.width > 8192 || metadata.height > 8192) throw new RequestError(413, 'Image limitée à 8192 pixels par côté.');
    return { width: metadata.width, height: metadata.height };
}

const UPSTREAM_ERRORS = Object.freeze({
    400: 'Requête Mistral refusée.',
    401: 'Clé API Mistral invalide.',
    402: 'Crédits Mistral insuffisants.',
    403: 'Accès Mistral refusé.',
    404: 'Modèle Mistral indisponible.',
    429: 'Quota Mistral dépassé.',
});

function upstreamError(status) {
    return new RequestError(UPSTREAM_ERRORS[status] ? status : 502, UPSTREAM_ERRORS[status] || 'Mistral est momentanément indisponible.');
}

/** BYOK only: no server-owned key, persistence, cookies, third-party URLs or automatic retries. */
export function createMistralOcrHandler({ fetchImpl = globalThis.fetch, timeoutMs = MISTRAL_TIMEOUT_MS } = {}) {
    return async function POST(request) {
        const timeout = new AbortController();
        const timer = setTimeout(() => timeout.abort(), timeoutMs);
        const signal = AbortSignal.any([request.signal, timeout.signal]);
        try {
            const origin = request.headers.get('origin');
            const host = request.headers.get('host') || new URL(request.url).host;
            let validOrigin = true;
            try {
                if (origin) {
                    const parsed = new URL(origin);
                    validOrigin = ['https:', 'http:'].includes(parsed.protocol) && parsed.host === host;
                }
            } catch { validOrigin = false; }
            if (request.headers.get('sec-fetch-site') === 'cross-site' || !validOrigin) {
                throw new RequestError(403, 'Origine de la requête non autorisée.');
            }
            const apiKey = request.headers.get('x-mistral-api-key')?.trim();
            if (!apiKey || !/^[\x21-\x7e]{10,512}$/.test(apiKey)) throw new RequestError(401, 'Clé API Mistral manquante ou invalide.');
            if (!/^application\/json(?:;|$)/i.test(request.headers.get('content-type') || '') ||
                !['identity', null].includes(request.headers.get('content-encoding'))) {
                throw new RequestError(415, 'Une requête JSON non compressée est requise.');
            }
            const length = request.headers.get('content-length');
            if (length && (!/^\d+$/.test(length) || Number(length) > MAX_BODY_BYTES)) throw new RequestError(413, 'Requête trop volumineuse.');
            let input;
            try {
                input = JSON.parse(await readBoundedBody(request.body, MAX_BODY_BYTES, signal));
            } catch (error) {
                if (error instanceof SyntaxError) throw new RequestError(400, 'Requête JSON invalide.');
                throw error;
            }
            if (!input || typeof input.prompt !== 'string' || !input.prompt.trim() || input.prompt.length > 50_000) {
                throw new RequestError(400, 'Instruction OCR invalide.');
            }
            const { width, height } = await validateImage(input.image);
            signal.throwIfAborted();
            const response = await fetchImpl(ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
                body: JSON.stringify({
                    model: MISTRAL_MODEL,
                    messages: [{ role: 'user', content: [
                        { type: 'text', text: input.prompt + `\nReturn a JSON object with a bubbles array; bbox order is [x1,y1,x2,y2] in pixel coordinates of the provided image: it is exactly ${width} pixels wide and ${height} pixels tall, [0,0] is the top-left corner and [${width},${height}] is the bottom-right corner. Never normalize coordinates.` },
                        { type: 'image_url', image_url: input.image },
                    ] }],
                    temperature: 0,
                    max_tokens: 16_384,
                    stream: false,
                    response_format: { type: 'json_object' },
                    reasoning_effort: 'low',
                }),
                signal, cache: 'no-store', redirect: 'error', credentials: 'omit',
            });
            if (!response.ok) {
                void response.body?.cancel().catch(() => {});
                throw upstreamError(response.status);
            }
            let payload;
            try {
                payload = JSON.parse(await readBoundedBody(response.body, MAX_RESPONSE_BYTES, signal));
            } catch (error) {
                if (signal.aborted) throw error;
                throw new RequestError(502, 'Réponse Mistral invalide ou trop volumineuse.');
            }
            const choice = payload?.choices?.[0];
            if (choice?.finish_reason === 'length') throw new RequestError(502, 'Réponse GLM tronquée : aucune annotation n’a été importée.');
            // GLM sometimes answers with content chunks (reasoning + text) instead of a plain string.
            const content = Array.isArray(choice?.message?.content)
                ? choice.message.content.filter(chunk => chunk?.type === 'text' && typeof chunk.text === 'string').map(chunk => chunk.text).join('')
                : choice?.message?.content;
            if (choice?.finish_reason !== 'stop' || typeof content !== 'string' || !content.trim()) {
                throw new RequestError(502, 'GLM n’a pas renvoyé de résultat OCR complet.');
            }
            // Only the final answer is returned; reasoning and the upstream envelope are discarded.
            return Response.json({ text: content, model: MISTRAL_MODEL }, { headers: HEADERS });
        } catch (error) {
            const status = timeout.signal.aborted ? 504 : request.signal.aborted ? 499 : error instanceof RequestError ? error.status : 502;
            const message = timeout.signal.aborted ? 'Mistral a mis trop de temps à répondre.' : request.signal.aborted ? 'Requête annulée.' : error instanceof RequestError ? error.message : 'Impossible de joindre Mistral.';
            return Response.json({ error: message }, { status, headers: HEADERS });
        } finally {
            clearTimeout(timer);
        }
    };
}
