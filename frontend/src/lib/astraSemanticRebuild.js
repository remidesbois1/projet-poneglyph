import { ASTRA_DESCRIPTION_MODEL, DESCRIPTION_PROMPT_VERSION, isCurrentAstraPage, parsePageDescription } from '@poneglyph/shared/page-description';
import { fetchOriginalPageImage } from './pageImageClient';
import { runChatGptPageDescription } from './chatGptDesktop';
import { rebuildPageSemanticData } from './api';
import { getSupabaseSession } from './supabaseSession';

export const DEFAULT_ASTRA_CONCURRENCY = 3;
export const MAX_ASTRA_CONCURRENCY = 3;

export function requireSupabaseSession(options = {}) {
    return getSupabaseSession({ ...options, required: true });
}

export function getRebuildFailure(error) {
    const message = error?.response?.data?.error || error?.message || (typeof error === 'string' ? error : 'Erreur inconnue.');
    const code = error?.response?.data?.code || error?.code || '';
    const status = error?.status || error?.response?.status;
    if (message.includes('CHATGPT_QUOTA_EXCEEDED')) return { status: 'quota', message };
    if (message.includes('CHATGPT_REQUEST_REJECTED')
        || /Appel ChatGPT.*HTTP (400|404|422)/i.test(message)) return { status: 'chatgpt_request', message };
    if (message.includes('CHATGPT_SERVICE_UNAVAILABLE')) return { status: 'chatgpt_unavailable', message };
    if (['INVALID_GEMINI_API_KEY', 'GEMINI_USER_KEY_REJECTED'].includes(code)) return { status: 'gemini_key', message };
    if (message.includes('CHATGPT_AUTH_REQUIRED') || /session ChatGPT|connectez-vous a ChatGPT/i.test(message)) {
        return { status: 'chatgpt_auth', message: 'Session ChatGPT expirée ou refusée. Reconnectez-vous à ChatGPT puis reprenez.' };
    }
    if (code === 'SUPABASE_UNAVAILABLE') return { status: 'supabase_unavailable', message };
    if (code === 'SUPABASE_PERMISSION_DENIED' || status === 403) {
        return { status: 'supabase_permissions', message: 'Accès Supabase refusé : vérifiez les permissions admin du compte avant de reprendre.' };
    }
    if (code === 'SUPABASE_AUTH_REQUIRED' || status === 401) {
        return { status: 'supabase_auth', message: 'Session Supabase expirée ou refusée malgré le renouvellement. Reconnectez-vous puis reprenez.' };
    }
    if (code === 'EMBEDDING_QUOTA_EXCEEDED' || status === 429) return { status: 'quota', message };
    if (code === 'SEMANTIC_REBUILD_UNAVAILABLE') return { status: 'unavailable', message };
    return { status: null, message };
}

export function createPendingDescriptionCache(scope, storage = typeof window === 'undefined' ? null : window.localStorage) {
    const memory = new Map();
    const key = (id) => `poneglyph:astra-description:${DESCRIPTION_PROMPT_VERSION}:${scope}:${id}`;
    return {
        get(id) {
            try {
                const value = memory.get(id) || JSON.parse(storage?.getItem(key(id)) || 'null');
                if (value?.model !== ASTRA_DESCRIPTION_MODEL || value?.version !== DESCRIPTION_PROMPT_VERSION) return null;
                return parsePageDescription(value.description);
            } catch { return null; }
        },
        put(id, description) {
            const value = { model: ASTRA_DESCRIPTION_MODEL, version: DESCRIPTION_PROMPT_VERSION, description: parsePageDescription(description) };
            memory.set(id, value);
            try {
                storage?.setItem(key(id), JSON.stringify(value));
                return Boolean(storage);
            } catch { return false; }
        },
        remove(id) {
            memory.delete(id);
            try { storage?.removeItem(key(id)); } catch { /* DB provenance still allows resuming completed pages. */ }
        },
    };
}

// Stop scheduling new pages, then wait for the started descriptions and atomic saves.
export async function runAstraSemanticRebuild({
    pages, mangaSlug, force = false, cache, shouldStop = () => false, onProgress = () => {},
    getSession = requireSupabaseSession, fetchImage = fetchOriginalPageImage,
    describeImage = runChatGptPageDescription, rebuildPage = rebuildPageSemanticData, prompt,
    concurrency = DEFAULT_ASTRA_CONCURRENCY,
}) {
    const candidates = force ? pages : pages.filter((page) => !isCurrentAstraPage(page));
    if (force) for (const page of pages) cache.remove(page.id);
    const limit = Math.max(1, Math.min(MAX_ASTRA_CONCURRENCY, Math.floor(Number(concurrency)) || DEFAULT_ASTRA_CONCURRENCY));
    const active = new Map();
    let nextIndex = 0;
    let haltStatus = null;
    const stopScheduling = () => Boolean(haltStatus || shouldStop());
    let progress = {
        total: pages.length, completed: pages.length - candidates.length,
        skipped: pages.length - candidates.length, saved: 0, geminiUserKeyPages: 0, errors: [], log: [],
        activePages: [], concurrency: limit, status: 'running',
    };
    const report = (changes = {}, line) => {
        progress = { ...progress, ...changes, activePages: [...active.values()] };
        if (line) progress.log = [{ time: Date.now(), message: line }, ...progress.log].slice(0, 30);
        onProgress(progress);
    };
    const setPhase = (page, phase) => {
        active.set(page.id, { page, phase });
        report();
    };
    report({}, `Démarrage : ${candidates.length} pages à traiter avec GPT-6 Astra, prompt v${DESCRIPTION_PROMPT_VERSION}, ${limit} pages à la fois.`);
    const worker = async () => {
        while (!stopScheduling() && nextIndex < candidates.length) {
            const page = candidates[nextIndex++];
            active.set(page.id, { page, phase: 'Image originale' });
            report({}, `Traitement page ${page.id}.`);
            try {
                const session = await getSession();
                let description = cache.get(page.id);
                if (!description) {
                    if (stopScheduling()) return;
                    const image = await fetchImage(page.id, session.access_token, { expectedUserId: session.user?.id });
                    // A loaded image has not consumed ChatGPT quota yet.
                    if (stopScheduling()) return;
                    setPhase(page, 'Description Astra');
                    description = parsePageDescription(await describeImage(image, { prompt }));
                    if (!cache.put(page.id, description)) {
                        report({}, `Cache local indisponible pour ${page.id} : conservez cet écran ouvert jusqu’à la sauvegarde.`);
                    }
                } else {
                    report({}, `Description Astra déjà obtenue pour ${page.id} : réutilisation sans nouvel appel ChatGPT.`);
                }
                setPhase(page, 'Voyage + Gemini');
                const rebuilt = await rebuildPage({ id_page: page.id, description }, mangaSlug);
                cache.remove(page.id);
                const usedUserKey = rebuilt?.data?.gemini_key_source === 'user';
                if (usedUserKey) report({}, `Gemini page ${page.id} : repli sur la clé personnelle du profil.`);
                report({
                    saved: progress.saved + 1, completed: progress.completed + 1,
                    geminiUserKeyPages: progress.geminiUserKeyPages + (usedUserKey ? 1 : 0),
                }, `Page ${page.id} enregistrée avec les deux embeddings.`);
            } catch (error) {
                const failure = getRebuildFailure(error);
                if (failure.status && !haltStatus) haltStatus = failure.status;
                report({
                    errors: [...progress.errors, { pageId: page.id, message: failure.message }],
                    ...(haltStatus ? { status: haltStatus } : {}),
                }, `Erreur page ${page.id} : ${failure.message}`);
            } finally {
                active.delete(page.id);
                report();
            }
        }
    };
    await Promise.all(Array.from({ length: Math.min(limit, candidates.length) }, worker));
    const status = haltStatus || (shouldStop() ? 'stopped' : progress.errors.length ? 'finished_with_errors' : 'completed');
    const line = status === 'stopped'
        ? 'Arrêt propre effectué. Les pages enregistrées seront ignorées à la reprise.'
        : haltStatus ? 'Traitement suspendu après la fin des pages déjà lancées. Reprise disponible.' : 'Traitement terminé.';
    report({ status }, line);
    return progress;
}
