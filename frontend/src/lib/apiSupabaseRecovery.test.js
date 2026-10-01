import axios from 'axios';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ session: vi.fn(), refresh: vi.fn() }));
vi.mock('./supabaseClient', () => ({ supabase: { auth: { getSession: mocks.session, refreshSession: mocks.refresh } } }));
const client = axios.create();
vi.spyOn(axios, 'create').mockReturnValue(client);
const { getEmbeddingStats, rebuildPageSemanticData } = await import('./api');
const { runAstraSemanticRebuild, createPendingDescriptionCache, getRebuildFailure } = await import('./astraSemanticRebuild');

const description = { content: 'Luffy protège Zoro.', metadata: { arc: '', characters: ['Luffy', 'Zoro'] } };
let stored;
let requests;
const session = token => ({ access_token: token, user: { id: 'admin' }, expires_at: Date.now() / 1000 + 3600 });
function reject(config, status, code) {
    throw new axios.AxiosError('Request rejected', 'ERR_BAD_RESPONSE', config, null, { status, data: { code, error: code }, config });
}
function success(config, data = []) { return { status: 200, data, headers: {}, config }; }

describe('authenticated backend request recovery', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        stored = session('old');
        requests = [];
        mocks.session.mockImplementation(async () => ({ data: { session: stored }, error: null }));
        mocks.refresh.mockImplementation(async () => { stored = session('new'); return { data: { session: stored }, error: null }; });
        client.defaults.adapter = async config => {
            requests.push(config);
            if (config.headers.get('Authorization') === 'Bearer old') reject(config, 401, 'SUPABASE_AUTH_REQUIRED');
            return success(config);
        };
    });

    it('recovers statistics automatically instead of requiring a manual reconnect', async () => {
        await expect(getEmbeddingStats('one-piece')).resolves.toMatchObject({ status: 200, data: [] });
        expect(mocks.refresh).toHaveBeenCalledOnce();
        expect(requests).toHaveLength(2);
        expect(requests[1].params).toMatchObject({ manga: 'one-piece' });
    });

    it('recovers three concurrent saves with one refresh and exactly one successful write per page', async () => {
        localStorage.setItem('google_api_key', 'profile-secret');
        const writes = [];
        client.defaults.adapter = async config => {
            requests.push(config);
            if (config.headers.get('Authorization') === 'Bearer old') reject(config, 401, 'SUPABASE_AUTH_REQUIRED');
            writes.push(JSON.parse(config.data).id_page);
            expect(config.headers.get('X-Gemini-API-Key')).toBe('profile-secret');
            expect(config.params).toMatchObject({ manga: 'one-piece' });
            return success(config, { success: true });
        };
        const describeImage = vi.fn(async () => description);
        const pages = [
            { id: 1, description, description_model: 'gpt-6-astra', description_prompt_version: 2, has_description: true, has_voyage: true, has_gemini: true },
            ...[2, 3, 4].map(id => ({ id })),
        ];
        const result = await runAstraSemanticRebuild({
            pages, mangaSlug: 'one-piece', concurrency: 3, cache: createPendingDescriptionCache('admin:one-piece'),
            fetchImage: vi.fn(async () => new Blob(['original'])), describeImage,
        });
        expect(result).toMatchObject({ status: 'completed', saved: 3, skipped: 1, completed: 4, errors: [] });
        expect(mocks.refresh).toHaveBeenCalledOnce();
        expect(describeImage).toHaveBeenCalledTimes(3);
        expect(writes.sort()).toEqual([2, 3, 4]);
        expect(requests).toHaveLength(6);
    });

    it('retries an explicit authentication outage once without rotating the token', async () => {
        client.defaults.adapter = async config => {
            requests.push(config);
            if (requests.length === 1) reject(config, 503, 'SUPABASE_UNAVAILABLE');
            return success(config);
        };
        await getEmbeddingStats('one-piece');
        expect(requests).toHaveLength(2);
        expect(mocks.refresh).not.toHaveBeenCalled();
    });

    it.each([[403, 'SUPABASE_PERMISSION_DENIED'], [503, 'EMBEDDING_FAILED'], [429, 'EMBEDDING_QUOTA_EXCEEDED']])
        ('does not replay page work after HTTP %s (%s)', async (status, code) => {
            client.defaults.adapter = async config => { requests.push(config); reject(config, status, code); };
            await expect(rebuildPageSemanticData({ id_page: 2, description }, 'one-piece')).rejects.toMatchObject({ response: { status } });
            expect(requests).toHaveLength(1);
            expect(mocks.refresh).not.toHaveBeenCalled();
        });

    it('limits authentication recovery to one retry and retains the Astra description on failure', async () => {
        client.defaults.adapter = async config => { requests.push(config); reject(config, 401, 'SUPABASE_AUTH_REQUIRED'); };
        const cache = createPendingDescriptionCache('admin:one-piece');
        const result = await runAstraSemanticRebuild({
            pages: [{ id: 2 }], mangaSlug: 'one-piece', cache,
            fetchImage: vi.fn(async () => new Blob(['original'])), describeImage: vi.fn(async () => description),
        });
        expect(result).toMatchObject({ status: 'supabase_auth', saved: 0 });
        expect(mocks.refresh).toHaveBeenCalledOnce();
        expect(requests).toHaveLength(2);
        expect(cache.get(2)).toEqual(description);
    });

    it('does not replay a write after an ambiguous lost response', async () => {
        client.defaults.adapter = async config => { requests.push(config); throw new axios.AxiosError('Network Error', 'ERR_NETWORK', config); };
        await expect(rebuildPageSemanticData({ id_page: 2, description }, 'one-piece')).rejects.toMatchObject({ code: 'ERR_NETWORK' });
        expect(requests).toHaveLength(1);
        expect(mocks.refresh).not.toHaveBeenCalled();
    });

    it('preserves request cancellation while refreshing', async () => {
        const controller = new AbortController();
        mocks.refresh.mockImplementationOnce(async () => { controller.abort(); stored = session('new'); return { data: { session: stored } }; });
        await expect(client.get('/admin/ai-models/embedding-stats', { signal: controller.signal })).rejects.toBeDefined();
        expect(requests).toHaveLength(1);
    });

    it('suspends availability and permission failures with separate messages and cached descriptions', async () => {
        for (const [status, code, expected] of [[503, 'SUPABASE_UNAVAILABLE', 'supabase_unavailable'], [403, 'SUPABASE_PERMISSION_DENIED', 'supabase_permissions']]) {
            const error = { response: { status, data: { code, error: code } } };
            expect(getRebuildFailure(error).status).toBe(expected);
            const cache = createPendingDescriptionCache(`admin:${code}`);
            const result = await runAstraSemanticRebuild({
                pages: [{ id: 2 }, { id: 3 }], mangaSlug: 'one-piece', cache, concurrency: 1,
                fetchImage: vi.fn(async () => new Blob(['original'])), describeImage: vi.fn(async () => description),
                rebuildPage: vi.fn(async () => { throw error; }),
            });
            expect(result).toMatchObject({ status: expected, saved: 0 });
            expect(cache.get(2)).toEqual(description);
        }
    });
});
