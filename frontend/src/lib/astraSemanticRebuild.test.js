import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./api', () => ({ rebuildPageSemanticData: vi.fn() }));
vi.mock('./chatGptDesktop', () => ({ runChatGptPageDescription: vi.fn() }));
vi.mock('./supabaseClient', () => ({ supabase: { auth: { getSession: vi.fn() } } }));

import { createPendingDescriptionCache, runAstraSemanticRebuild } from './astraSemanticRebuild';

const description = { content: 'Luffy protège Zoro.', metadata: { arc: '', characters: ['Luffy', 'Zoro'] } };
const oldPage = (id) => ({ id, has_description: true, has_voyage: true, has_gemini: true, has_f2llm: true, description });
const currentPage = (id) => ({ ...oldPage(id), description_model: 'gpt-6-astra', description_prompt_version: 2 });

function setup(pages = [oldPage(1), oldPage(2)]) {
    const image = new Blob(['original'], { type: 'image/png' });
    return {
        pages, mangaSlug: 'one-piece', concurrency: 1, cache: createPendingDescriptionCache('user:one-piece'),
        getSession: vi.fn().mockResolvedValue({ access_token: 'supabase-token', user: { id: 'user' } }),
        fetchImage: vi.fn().mockResolvedValue(image),
        describeImage: vi.fn().mockResolvedValue(description),
        rebuildPage: vi.fn().mockResolvedValue({ data: { success: true } }),
        onProgress: vi.fn(),
    };
}

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
}

describe('Astra semantic rebuild', () => {
    beforeEach(() => localStorage.clear());

    it.each([undefined, 2, 99])('bounds concurrent work across description and save with concurrency %s', async (concurrency) => {
        const limit = concurrency === 2 ? 2 : 3;
        const deps = setup(Array.from({ length: 6 }, (_, index) => oldPage(index + 1)));
        const descriptions = deps.pages.map(() => deferred());
        const saves = deps.pages.map(() => deferred());
        const started = [];
        deps.fetchImage.mockImplementation(async (id) => new Blob([String(id)], { type: 'image/png' }));
        deps.describeImage.mockImplementation(async (image) => {
            const id = Number(await image.text());
            started.push(id);
            return descriptions[id - 1].promise;
        });
        deps.rebuildPage.mockImplementation(({ id_page: id }) => saves[id - 1].promise);
        const running = runAstraSemanticRebuild({ ...deps, concurrency });
        await vi.waitFor(() => expect(started).toEqual(Array.from({ length: limit }, (_, index) => index + 1)));
        descriptions[0].resolve(description);
        await vi.waitFor(() => expect(deps.rebuildPage).toHaveBeenCalledOnce());
        expect(started).toHaveLength(limit);
        expect(deps.onProgress.mock.lastCall[0].activePages).toContainEqual({ page: oldPage(1), phase: 'Voyage + Gemini' });
        expect(deps.onProgress.mock.lastCall[0].activePages).toContainEqual({ page: oldPage(2), phase: 'Description Astra' });
        saves[0].resolve();
        await vi.waitFor(() => expect(started).toHaveLength(limit + 1));
        descriptions.forEach((gate) => gate.resolve(description));
        saves.forEach((gate) => gate.resolve());
        const result = await running;
        expect(result).toMatchObject({ status: 'completed', saved: 6, completed: 6, activePages: [], concurrency: limit });
        expect(started).toHaveLength(6);
        expect(new Set(deps.rebuildPage.mock.calls.map(([body]) => body.id_page)).size).toBe(6);
        expect(deps.onProgress.mock.calls.every(([next]) => next.activePages.length <= limit)).toBe(true);
        expect(deps.pages.every((page) => deps.cache.get(page.id) === null)).toBe(true);
    });

    it('drains both started descriptions and saves on stop without starting a queued image', async () => {
        const deps = setup([oldPage(1), oldPage(2), oldPage(3), oldPage(4)]);
        const loading = deferred();
        const firstDescription = deferred();
        const secondDescription = deferred();
        const firstSave = deferred();
        const secondSave = deferred();
        let stop = false;
        let finished = false;
        deps.fetchImage.mockImplementation(async (id) => id === 3 ? loading.promise : new Blob([String(id)], { type: 'image/png' }));
        deps.describeImage.mockImplementation(async (image) => Number(await image.text()) === 1 ? firstDescription.promise : secondDescription.promise);
        deps.rebuildPage.mockImplementation(({ id_page: id }) => id === 1 ? firstSave.promise : secondSave.promise);
        const running = runAstraSemanticRebuild({ ...deps, concurrency: 3, shouldStop: () => stop });
        running.then(() => { finished = true; });
        await vi.waitFor(() => expect(deps.describeImage).toHaveBeenCalledTimes(2));
        stop = true;
        loading.resolve(new Blob(['3'], { type: 'image/png' }));
        secondDescription.resolve(description);
        await vi.waitFor(() => expect(deps.rebuildPage).toHaveBeenCalledWith({ id_page: 2, description }, 'one-piece'));
        secondSave.resolve();
        await vi.waitFor(() => expect(deps.onProgress.mock.lastCall[0].saved).toBe(1));
        expect(finished).toBe(false);
        expect(deps.describeImage).toHaveBeenCalledTimes(2);
        firstDescription.resolve(description);
        await vi.waitFor(() => expect(deps.rebuildPage).toHaveBeenCalledTimes(2));
        expect(finished).toBe(false);
        firstSave.resolve();
        expect(await running).toMatchObject({ saved: 2, completed: 2, status: 'stopped', activePages: [] });
        expect(deps.fetchImage.mock.calls.map(([id]) => id)).toEqual([1, 2, 3]);
        expect(deps.pages.every((page) => deps.cache.get(page.id) === null)).toBe(true);
    });

    it('stops new calls on quota while preserving a successful concurrent page', async () => {
        const deps = setup([oldPage(1), oldPage(2), oldPage(3), oldPage(4)]);
        const firstDescription = deferred();
        const secondDescription = deferred();
        const loading = deferred();
        let finished = false;
        deps.fetchImage.mockImplementation(async (id) => id === 3 ? loading.promise : new Blob([String(id)], { type: 'image/png' }));
        deps.describeImage.mockImplementation(async (image) => Number(await image.text()) === 1 ? firstDescription.promise : secondDescription.promise);
        const running = runAstraSemanticRebuild({ ...deps, concurrency: 3 });
        running.then(() => { finished = true; });
        await vi.waitFor(() => expect(deps.describeImage).toHaveBeenCalledTimes(2));
        firstDescription.reject('CHATGPT_QUOTA_EXCEEDED: usage limit reached');
        await vi.waitFor(() => expect(deps.onProgress.mock.lastCall[0].status).toBe('quota'));
        expect(finished).toBe(false);
        loading.resolve(new Blob(['3'], { type: 'image/png' }));
        secondDescription.resolve(description);
        const result = await running;
        expect(result).toMatchObject({ status: 'quota', saved: 1, completed: 1, activePages: [] });
        expect(result.errors).toEqual([{ pageId: 1, message: 'CHATGPT_QUOTA_EXCEEDED: usage limit reached' }]);
        expect(deps.fetchImage.mock.calls.map(([id]) => id)).toEqual([1, 2, 3]);
        expect(deps.describeImage).toHaveBeenCalledTimes(2);
        expect(deps.rebuildPage).toHaveBeenCalledWith({ id_page: 2, description }, 'one-piece');
    });

    it('keeps the first global failure and caches another description that could not be saved', async () => {
        const deps = setup([oldPage(1), oldPage(2), oldPage(3)]);
        const firstDescription = deferred();
        const secondDescription = deferred();
        deps.fetchImage.mockImplementation(async (id) => new Blob([String(id)], { type: 'image/png' }));
        deps.describeImage.mockImplementation(async (image) => Number(await image.text()) === 1 ? firstDescription.promise : secondDescription.promise);
        deps.rebuildPage.mockRejectedValue({ response: { status: 503, data: { code: 'SEMANTIC_REBUILD_UNAVAILABLE', error: 'Serveur indisponible' } } });
        const running = runAstraSemanticRebuild({ ...deps, concurrency: 2 });
        await vi.waitFor(() => expect(deps.describeImage).toHaveBeenCalledTimes(2));
        firstDescription.reject('CHATGPT_QUOTA_EXCEEDED: usage limit reached');
        await vi.waitFor(() => expect(deps.onProgress.mock.lastCall[0].status).toBe('quota'));
        secondDescription.resolve(description);
        expect(await running).toMatchObject({ status: 'quota', saved: 0, completed: 0 });
        expect(deps.fetchImage).toHaveBeenCalledTimes(2);
        expect(deps.cache.get(2)).toEqual(description);
        expect(deps.onProgress.mock.lastCall[0].errors).toHaveLength(2);
    });

    it('counts isolated parallel failures accurately and retries only incomplete pages', async () => {
        const deps = setup([oldPage(1), oldPage(2), oldPage(3)]);
        deps.rebuildPage.mockImplementation(async ({ id_page: id }) => {
            if (id === 2) throw new Error('Embedding failed');
        });
        const first = await runAstraSemanticRebuild({ ...deps, concurrency: 3 });
        expect(first).toMatchObject({ status: 'finished_with_errors', completed: 2, saved: 2 });
        expect(first.errors).toEqual([{ pageId: 2, message: 'Embedding failed' }]);
        expect(deps.cache.get(2)).toEqual(description);
        const resumed = setup([currentPage(1), oldPage(2), currentPage(3)]);
        expect(await runAstraSemanticRebuild({ ...resumed, concurrency: 3 })).toMatchObject({ completed: 3, skipped: 2, saved: 1, status: 'completed' });
        expect(resumed.describeImage).not.toHaveBeenCalled();
        expect(resumed.fetchImage).not.toHaveBeenCalled();
        expect(resumed.rebuildPage).toHaveBeenCalledOnce();
        expect(resumed.cache.get(2)).toBeNull();
    });

    it('processes all old pages even if description and all vectors already exist', async () => {
        const deps = setup();
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ completed: 2, saved: 2, skipped: 0, status: 'completed' });
        expect(deps.fetchImage).toHaveBeenCalledWith(1, 'supabase-token', { expectedUserId: 'user' });
        expect(deps.describeImage).toHaveBeenCalledWith(await deps.fetchImage.mock.results[0].value, { prompt: undefined });
        expect(deps.rebuildPage).toHaveBeenCalledWith({ id_page: 1, description }, 'one-piece');
    });

    it('resumes only complete current Astra pages, retrying partial or obsolete ones', async () => {
        const deps = setup([currentPage(1), { ...currentPage(2), has_gemini: false }, { ...currentPage(3), description_prompt_version: 1 }, { ...currentPage(4), description: '{}' }]);
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ completed: 4, skipped: 1, saved: 3 });
        expect(deps.fetchImage.mock.calls.map(([id]) => id)).toEqual([2, 3, 4]);
    });

    it('resumes without requiring any F2LLM vector on completed Astra pages', async () => {
        const deps = setup([{ ...currentPage(1), has_f2llm: false }]);
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ completed: 1, skipped: 1, saved: 0, status: 'completed' });
        expect(deps.describeImage).not.toHaveBeenCalled();
        expect(deps.rebuildPage).not.toHaveBeenCalled();
    });

    it('can explicitly force a new rebuild of current Astra pages', async () => {
        const deps = setup([currentPage(1)]);
        await runAstraSemanticRebuild({ ...deps, force: true });
        expect(deps.describeImage).toHaveBeenCalledOnce();
    });

    it('finishes the in-flight save then stops before the next page', async () => {
        const deps = setup();
        let stop = false;
        deps.describeImage.mockImplementation(async () => { stop = true; return description; });
        const result = await runAstraSemanticRebuild({ ...deps, shouldStop: () => stop });
        expect(result).toMatchObject({ saved: 1, completed: 1, status: 'stopped' });
        expect(deps.rebuildPage).toHaveBeenCalledOnce();
        expect(deps.describeImage).toHaveBeenCalledOnce();
    });

    it('does not consume quota when stopped while fetching the image', async () => {
        const deps = setup();
        let stop = false;
        deps.fetchImage.mockImplementation(async () => { stop = true; return new Blob(['original']); });
        expect(await runAstraSemanticRebuild({ ...deps, shouldStop: () => stop })).toMatchObject({ saved: 0, status: 'stopped' });
        expect(deps.describeImage).not.toHaveBeenCalled();
    });

    it.each([
        ['CHATGPT_QUOTA_EXCEEDED: usage limit reached', 'quota'],
        ['CHATGPT_AUTH_REQUIRED: Session ChatGPT refusee.', 'chatgpt_auth'],
        ['Session ChatGPT expiree. Reconnectez-vous.', 'chatgpt_auth'],
        ['CHATGPT_REQUEST_REJECTED: Requete ChatGPT refusee (HTTP 400). Instructions are required', 'chatgpt_request'],
        ['Appel ChatGPT interrompu ou refuse (HTTP 400).', 'chatgpt_request'],
        ['CHATGPT_SERVICE_UNAVAILABLE: Service ChatGPT indisponible (HTTP 503).', 'chatgpt_unavailable'],
    ])('stops on global ChatGPT failure %s without marking the page complete', async (error, status) => {
        const deps = setup();
        deps.describeImage.mockRejectedValue(error);
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ completed: 0, saved: 0, status });
        expect(result.errors).toHaveLength(1);
        expect(deps.describeImage).toHaveBeenCalledOnce();
        expect(deps.rebuildPage).not.toHaveBeenCalled();
    });

    it('continues after an isolated image failure and exposes page-specific errors', async () => {
        const deps = setup();
        deps.fetchImage.mockRejectedValueOnce(new Error('Image corrompue'));
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ completed: 1, status: 'finished_with_errors', errors: [{ pageId: 1, message: 'Image corrompue' }] });
        expect(deps.rebuildPage).toHaveBeenCalledWith({ id_page: 2, description }, 'one-piece');
    });

    it('rejects invalid description before any write and continues to the next page', async () => {
        const deps = setup();
        deps.describeImage.mockResolvedValueOnce({ content: 'missing metadata' });
        const result = await runAstraSemanticRebuild(deps);
        expect(result.saved).toBe(1);
        expect(deps.rebuildPage).toHaveBeenCalledOnce();
        expect(deps.cache.get(1)).toBeNull();
    });

    it('reuses a persisted Astra description after an embedding failure and restart', async () => {
        const deps = setup([oldPage(1)]);
        deps.rebuildPage.mockRejectedValueOnce(new Error('Embedding failed'));
        await runAstraSemanticRebuild(deps);
        expect(deps.cache.get(1)).toEqual(description);
        const resumed = setup([oldPage(1)]);
        const result = await runAstraSemanticRebuild(resumed);
        expect(result.saved).toBe(1);
        expect(resumed.describeImage).not.toHaveBeenCalled();
        expect(resumed.cache.get(1)).toBeNull();
    });

    it('stops on an expired Supabase session before describing another page', async () => {
        const deps = setup();
        deps.fetchImage.mockRejectedValueOnce(Object.assign(new Error('Session expirée'), { status: 401 }));
        const result = await runAstraSemanticRebuild(deps);
        expect(result.status).toBe('supabase_auth');
        expect(deps.describeImage).not.toHaveBeenCalled();
    });

    it.each([
        [429, 'EMBEDDING_QUOTA_EXCEEDED', 'quota'],
        [503, 'SEMANTIC_REBUILD_UNAVAILABLE', 'unavailable'],
    ])('stops on a global embedding failure and retains the Astra description', async (status, code, expected) => {
        const deps = setup();
        deps.rebuildPage.mockRejectedValue({ response: { status, data: { code, error: 'Embedding indisponible' } } });
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ status: expected, saved: 0, completed: 0 });
        expect(deps.describeImage).toHaveBeenCalledOnce();
        expect(deps.cache.get(1)).toEqual(description);
    });

    it('isolates pending descriptions by user and manga', () => {
        const cache = createPendingDescriptionCache('user:one-piece');
        cache.put(1, description);
        expect(createPendingDescriptionCache('other-user:one-piece').get(1)).toBeNull();
        expect(createPendingDescriptionCache('user:other-manga').get(1)).toBeNull();
    });

    it('reports personal-key usage while resuming a cached description without calling Astra again', async () => {
        const deps = setup([oldPage(1), oldPage(2), oldPage(3)]);
        deps.cache.put(1, description);
        deps.rebuildPage.mockImplementation(async ({ id_page }) => ({ data: { success: true, ...(id_page === 1 ? { gemini_key_source: 'user' } : {}) } }));
        const result = await runAstraSemanticRebuild({ ...deps, concurrency: 3 });
        expect(result).toMatchObject({ saved: 3, completed: 3, geminiUserKeyPages: 1, status: 'completed' });
        expect(deps.describeImage).toHaveBeenCalledTimes(2);
        expect(deps.fetchImage.mock.calls.map(([id]) => id)).toEqual([2, 3]);
        expect(result.log.some((entry) => entry.message.includes('clé personnelle du profil'))).toBe(true);
        expect(deps.cache.get(1)).toBeNull();
    });

    it.each(['INVALID_GEMINI_API_KEY', 'GEMINI_USER_KEY_REJECTED'])('stops safely on %s and keeps the Astra result for retry', async (code) => {
        const deps = setup();
        deps.rebuildPage.mockRejectedValue({ response: { status: 503, data: { code, error: 'Clé personnelle refusée' } } });
        const result = await runAstraSemanticRebuild(deps);
        expect(result).toMatchObject({ status: 'gemini_key', saved: 0, completed: 0 });
        expect(deps.describeImage).toHaveBeenCalledOnce();
        expect(deps.cache.get(1)).toEqual(description);
    });
});
