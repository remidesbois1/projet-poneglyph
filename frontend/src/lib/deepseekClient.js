import { z } from 'zod';
import { cropImage } from './utils';
import { getPrompt } from './promptConfig';
import { capitalizeOcrSentenceStarts } from './ocr-utils';
import { DEEPSEEK_IMAGE_MAX_BYTES, DEEPSEEK_MODEL, DEEPSEEK_TIMEOUT_MS } from './deepseekConfig';

const boxSchema = z.tuple([z.number().min(0).max(1000), z.number().min(0).max(1000), z.number().min(0).max(1000), z.number().min(0).max(1000)])
    .refine(([x1, y1, x2, y2]) => x2 > x1 && y2 > y1);
const bubblesSchema = z.array(z.object({ content: z.string().trim().min(1).max(10_000), bbox: boxSchema })).max(1000);

export function parseDeepSeekBubbles(text) {
    if (typeof text !== 'string') throw new Error('Réponse OCR DeepSeek invalide.');
    const candidate = text.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/i, '$1');
    let parsed;
    try { parsed = JSON.parse(candidate); }
    catch { throw new Error('Réponse DeepSeek non JSON : aucune annotation n’a été importée.'); }
    const result = bubblesSchema.safeParse(Array.isArray(parsed) ? parsed : parsed?.bubbles);
    if (!result.success) throw new Error('Texte ou coordonnées DeepSeek invalides : aucune annotation n’a été importée.');
    return result.data.map(bubble => ({ ...bubble, content: capitalizeOcrSentenceStarts(bubble.content) }));
}

function toDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Impossible de préparer l’image pour DeepSeek.'));
        reader.onabort = () => reject(new DOMException('Requête annulée.', 'AbortError'));
        reader.readAsDataURL(blob);
    });
}

async function requestOcr(imageSource, coordinates, apiKey, mode, { signal } = {}) {
    if (typeof apiKey !== 'string' || !apiKey.trim()) throw new Error('Clé API DeepSeek manquante. Configurez-la dans les réglages API.');
    if (!imageSource || !coordinates || !['x', 'y', 'w', 'h'].every(key => Number.isFinite(coordinates[key])) ||
        coordinates.x < 0 || coordinates.y < 0 || coordinates.w <= 0 || coordinates.h <= 0) {
        throw new Error('Image ou zone OCR invalide.');
    }
    if (coordinates.w > 8192 || coordinates.h > 8192 || coordinates.w * coordinates.h > 24_000_000) {
        throw new Error('Image trop grande pour DeepSeek : réduisez-la à 8192 pixels par côté et 24 mégapixels.');
    }
    signal?.throwIfAborted();
    const blob = await cropImage(imageSource, coordinates);
    if (!(blob instanceof Blob) || !blob.size || blob.size > DEEPSEEK_IMAGE_MAX_BYTES) throw new Error('Image DeepSeek invalide ou supérieure à 10 Mo.');
    const prompt = mode === 'bbox'
        ? `${await getPrompt('ocr_page_bbox')}\n\n${await getPrompt('strict_json_suffix')}`
        : await getPrompt('ocr_bubble');
    const image = await toDataUrl(blob);
    signal?.throwIfAborted();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), DEEPSEEK_TIMEOUT_MS + 10_000);
    const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
    try {
        const response = await fetch('/api/deepseek', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-DeepSeek-API-Key': apiKey.trim() },
            body: JSON.stringify({ mode, prompt, image }),
            cache: 'no-store', credentials: 'same-origin', signal: combined,
        });
        let payload;
        try { payload = await response.json(); }
        catch { throw new Error('Réponse du service DeepSeek invalide.'); }
        if (!response.ok) throw new Error(typeof payload?.error === 'string' ? payload.error : 'Service DeepSeek indisponible.');
        if (typeof payload?.text !== 'string') throw new Error('DeepSeek n’a pas renvoyé de texte.');
        return { text: payload.text.trim(), model: payload.model || DEEPSEEK_MODEL };
    } catch (error) {
        if (signal?.aborted) throw new DOMException('Requête annulée.', 'AbortError');
        if (controller.signal.aborted) throw new Error('DeepSeek a mis trop de temps à répondre.');
        if (error instanceof TypeError) throw new Error('Impossible de joindre DeepSeek. Vérifiez votre connexion.');
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

export async function analyzeDeepSeekBubble(imageSource, coordinates, apiKey, options) {
    const result = await requestOcr(imageSource, coordinates, apiKey, 'bubble', options);
    return { data: { texte_propose: result.text }, model: result.model };
}

export async function generateDeepSeekOneShotBubbles(imageSource, apiKey, options) {
    const result = await requestOcr(imageSource, {
        x: 0, y: 0, w: imageSource?.naturalWidth, h: imageSource?.naturalHeight,
    }, apiKey, 'bbox', options);
    return { data: parseDeepSeekBubbles(result.text), model: result.model };
}
