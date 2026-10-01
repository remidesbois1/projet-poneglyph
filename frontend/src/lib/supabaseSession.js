import { supabase } from './supabaseClient';

const REFRESH_MARGIN_SECONDS = 120;

function sessionError(error) {
    const transient = error && (error instanceof TypeError || error.name === 'AuthRetryableFetchError'
        || error.status === 0 || error.status === 429 || error.status >= 500
        || /fetch failed|failed to fetch|network|timeout/i.test(error.message || ''));
    const failure = new Error(transient
        ? 'Supabase est temporairement indisponible. Réessayez la reprise dans un instant.'
        : 'Session Supabase expirée ou refusée. Reconnectez-vous puis reprenez.');
    failure.code = transient ? 'SUPABASE_UNAVAILABLE' : 'SUPABASE_AUTH_REQUIRED';
    failure.status = transient ? 503 : 401;
    return failure;
}

function checkSession(session, expectedUserId) {
    if (!session?.access_token || !session.user?.id || (expectedUserId && session.user.id !== expectedUserId)) {
        throw sessionError();
    }
    return session;
}

// The image fetches and Axios requests share one refresh, including parallel pages.
export function createSupabaseSessionManager(auth, now = () => Date.now()) {
    let refreshPromise = null;
    async function readSession() {
        try {
            const { data, error } = await auth.getSession();
            if (error) throw error;
            return data?.session || null;
        } catch (error) { throw sessionError(error); }
    }

    async function refreshSession(rejectedAccessToken, { expectedUserId } = {}) {
        const current = checkSession(await readSession(), expectedUserId);
        // A timer or another request may already have replaced the rejected token.
        if (current.access_token !== rejectedAccessToken) return current;
        if (!refreshPromise) {
            refreshPromise = (async () => {
                const latest = checkSession(await readSession(), current.user.id);
                if (latest.access_token !== rejectedAccessToken) return latest;
                let result;
                try { result = await auth.refreshSession(); }
                catch (error) { throw sessionError(error); }
                if (result.error) throw sessionError(result.error);
                return checkSession(result.data?.session, current.user.id);
            })().finally(() => { refreshPromise = null; });
        }
        return checkSession(await refreshPromise, expectedUserId || current.user.id);
    }

    async function getSession({ required = false, expectedUserId } = {}) {
        const session = await readSession();
        if (!session && !required && !expectedUserId) return null;
        checkSession(session, expectedUserId);
        if (session.expires_at && session.expires_at <= now() / 1000 + REFRESH_MARGIN_SECONDS) {
            return refreshSession(session.access_token, { expectedUserId: session.user.id });
        }
        return session;
    }
    return { getSession, refreshSession };
}

const sessions = createSupabaseSessionManager(supabase.auth);
export const getSupabaseSession = sessions.getSession;
export const refreshSupabaseSession = sessions.refreshSession;
