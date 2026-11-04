import { z } from 'zod';
import { cropImage } from './utils';
import { getPrompt } from './promptConfig';
import { capitalizeOcrSentenceStarts } from './ocr-utils';
import { MISTRAL_IMAGE_MAX_BYTES, MISTRAL_MODEL, MISTRAL_TIMEOUT_MS } from './mistralConfig';

const rawBoxSchema = z.tuple([z.number(), z.number(), z.number(), z.number()]);
const rawBubblesSchema = z.array(z.object({ content: z.string().trim().min(1).max(10_000), bbox: rawBoxSchema })).max(1000);

const clampToPage = (value) => Math.round(Math.min(1000, Math.max(0, value)));

/**
 * The server route asks GLM for pixel coordinates with the image's explicit dimensions
 * (measured 12/12 in-bounds, versus ~50% of calls out of bounds with a 0-1000 prompt).
 * `width`/`height` are the source image's pixel size: each axis is rescaled back to the
 * pipeline's 0-1000 convention, then clamped. A bubble whose geometry stays invalid is
 * dropped on its own instead of failing the whole page.
 */
export function parseGlmBubbles(text, { width, height } = {}) {
    if (typeof text !== 'string') throw new Error('Réponse OCR GLM invalide.');
    const candidate = text.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/i, '$1');
    let parsed;
    try { parsed = JSON.parse(candidate); }
    catch { throw new Error('Réponse GLM non JSON : aucune annotation n’a été importée.'); }
    const result = rawBubblesSchema.safeParse(Array.isArray(parsed) ? parsed : parsed?.bubbles);
    if (!result.success) throw new Error('Texte ou coordonnées GLM invalides : aucune annotation n’a été importée.');
    const canRescale = Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0;
    return result.data.flatMap(({ content, bbox: [x1, y1, x2, y2] }) => {
        const toPageX = (value) => clampToPage(canRescale ? value / width * 1000 : value);
        const toPageY = (value) => clampToPage(canRescale ? value / height * 1000 : value);
        const bbox = [toPageX(x1), toPageY(y1), toPageX(x2), toPageY(y2)];
        if (!(bbox[2] > bbox[0] && bbox[3] > bbox[1])) return [];
        return [{ content: capitalizeOcrSentenceStarts(content), bbox }];
    });
}

function toDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Impossible de préparer l’image pour Mistral.'));
        reader.onabort = () => reject(new DOMException('Requête annulée.', 'AbortError'));
        reader.readAsDataURL(blob);
    });
}

/** Reads the whole page with GLM 5.3 Flash (hosted by Mistral) and returns `[{ content, bbox }]`. */
export async function generateGlmOneShotBubbles(imageSource, apiKey, { signal } = {}) {
    if (typeof apiKey !== 'string' || !apiKey.trim()) throw new Error('Clé API Mistral manquante. Configurez-la dans les réglages API.');
    if (!imageSource || !Number.isFinite(imageSource.naturalWidth) || !Number.isFinite(imageSource.naturalHeight) ||
        imageSource.naturalWidth <= 0 || imageSource.naturalHeight <= 0) {
        throw new Error('Image ou zone OCR invalide.');
    }
    if (imageSource.naturalWidth > 8192 || imageSource.naturalHeight > 8192 ||
        imageSource.naturalWidth * imageSource.naturalHeight > 24_000_000) {
        throw new Error('Image trop grande pour Mistral : réduisez-la à 8192 pixels par côté et 24 mégapixels.');
    }
    signal?.throwIfAborted();
    const blob = await cropImage(imageSource, { x: 0, y: 0, w: imageSource.naturalWidth, h: imageSource.naturalHeight });
    if (!(blob instanceof Blob) || !blob.size || blob.size > MISTRAL_IMAGE_MAX_BYTES) throw new Error('Image Mistral invalide ou supérieure à 10 Mo.');
    const prompt = `${await getPrompt('ocr_page_bbox')}\n\n${await getPrompt('strict_json_suffix')}`;
    const image = await toDataUrl(blob);
    signal?.throwIfAborted();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MISTRAL_TIMEOUT_MS + 10_000);
    const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
    try {
        const response = await fetch('/api/mistral', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Mistral-API-Key': apiKey.trim() },
            body: JSON.stringify({ prompt, image }),
            cache: 'no-store', credentials: 'same-origin', signal: combined,
        });
        let payload;
        try { payload = await response.json(); }
        catch { throw new Error('Réponse du service Mistral invalide.'); }
        if (!response.ok) throw new Error(typeof payload?.error === 'string' ? payload.error : 'Service Mistral indisponible.');
        if (typeof payload?.text !== 'string') throw new Error('GLM n’a pas renvoyé de texte.');
        return { data: parseGlmBubbles(payload.text, { width: imageSource.naturalWidth, height: imageSource.naturalHeight }), model: payload.model || MISTRAL_MODEL };
    } catch (error) {
        if (signal?.aborted) throw new DOMException('Requête annulée.', 'AbortError');
        if (controller.signal.aborted) throw new Error('Mistral a mis trop de temps à répondre.');
        if (error instanceof TypeError) throw new Error('Impossible de joindre Mistral. Vérifiez votre connexion.');
        throw error;
    } finally {
        clearTimeout(timer);
    }
}
