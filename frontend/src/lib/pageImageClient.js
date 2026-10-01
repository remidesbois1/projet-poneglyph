import { refreshSupabaseSession } from './supabaseSession';

function getBackendApiUrl() {
    return (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001/api').replace(/\/$/, '');
}

export async function fetchOriginalPageImage(pageId, accessToken, {
    signal, fetchImpl = fetch, thumbnail = false, width = 640, expectedUserId,
    refreshSession = refreshSupabaseSession,
} = {}) {
    if (!pageId) throw new Error('Page id is required');
    if (!accessToken) throw new Error('Authentication is required to load the original page');

    const imagePath = thumbnail
        ? `/image/original/thumbnail?width=${encodeURIComponent(width)}`
        : '/image/original';
    const request = (token) => fetchImpl(`${getBackendApiUrl()}/pages/${encodeURIComponent(pageId)}${imagePath}`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
        credentials: 'omit',
        signal,
    });
    let response = await request(accessToken);
    let detail;
    if (!response.ok && (response.status === 401 || response.status === 503)) {
        detail = await response.json?.().catch(() => null);
        if (response.status === 401 || detail?.code === 'SUPABASE_UNAVAILABLE') {
            signal?.throwIfAborted();
            const token = response.status === 401
                ? (await refreshSession(accessToken, { expectedUserId })).access_token
                : accessToken;
            signal?.throwIfAborted();
            response = await request(token);
            detail = null;
        }
    }

    if (!response.ok) {
        detail = detail || await response.json?.().catch(() => null);
        const error = new Error(detail?.error || (response.status === 401
            ? 'Session Supabase expirée. Reconnectez-vous pour charger la page.'
            : "Impossible de charger l'image originale."));
        error.code = detail?.code;
        error.status = response.status;
        throw error;
    }

    return response.blob();
}

export function fetchOriginalPageThumbnail(pageId, accessToken, options = {}) {
    return fetchOriginalPageImage(pageId, accessToken, { ...options, thumbnail: true });
}
