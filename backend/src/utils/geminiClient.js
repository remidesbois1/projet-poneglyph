const axios = require('axios');
const { readPageImage } = require('./pageStorage');
const { logger } = require('./logger');

const GEMINI_EMBED_MODEL = 'gemini-embedding-2-preview';

async function generateGeminiEmbedding(text, taskType = "RETRIEVAL_QUERY", imageUrl = null, { requireImage = false, fallbackApiKey = null, onFallback } = {}) {
    const serverKey = process.env.GOOGLE_API_KEY?.trim();
    const userKey = typeof fallbackApiKey === 'string' ? fallbackApiKey.trim() : '';
    const credentials = [];
    if (serverKey) credentials.push({ key: serverKey, source: 'server' });
    if (userKey && userKey !== serverKey) credentials.push({ key: userKey, source: 'user' });
    if (!credentials.length) throw new Error('GOOGLE_API_KEY is not defined.');

    const parts = [];

    if (text) parts.push({ text: text.trim() });

    if (requireImage && !imageUrl) throw new Error('Original page image is required.');
    if (imageUrl) {
        try {
            const { buffer, contentType } = await readPageImage(imageUrl);
            const imgBase64 = buffer.toString('base64');
            parts.push({ inlineData: { mimeType: contentType, data: imgBase64 } });
        } catch (imgError) {
            if (requireImage) throw imgError;
            logger.warn('gemini_embedding_image_skipped', {
                error_code: imgError?.code || imgError?.name || 'IMAGE_READ_FAILED',
            });
        }
    }

    if (parts.length === 0) throw new Error('No content to embed.');

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_EMBED_MODEL}:embedContent`;
    const payload = {
        model: `models/${GEMINI_EMBED_MODEL}`,
        content: { parts },
        taskType
    };
    let failure;
    for (const { key, source } of credentials) {
        try {
            const response = await axios.post(url, payload, {
                headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
            });
            const embedding = response.data?.embedding?.values;
            if (!Array.isArray(embedding) || !embedding.length) throw new Error('No embedding returned from Gemini API.');
            if (source === 'user') onFallback?.();
            return embedding;
        } catch (error) {
            // Provider errors can contain request headers. Expose only status and key source.
            const status = error.response?.status;
            const errorDetails = error.response?.data?.error?.details;
            const invalidKey = [401, 403].includes(status)
                || (Array.isArray(errorDetails) && errorDetails.some((detail) => detail.reason === 'API_KEY_INVALID'));
            failure = new Error('Gemini embedding request failed.');
            failure.code = source === 'user' && invalidKey ? 'GEMINI_USER_KEY_REJECTED' : 'GEMINI_EMBEDDING_FAILED';
            failure.geminiKeySource = source;
            if (status) failure.response = { status };
        }
    }
    throw failure;
}

module.exports = { generateGeminiEmbedding };
