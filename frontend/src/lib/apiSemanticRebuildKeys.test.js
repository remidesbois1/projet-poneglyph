import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), interceptor: vi.fn(), session: vi.fn() }));
vi.mock('axios', () => ({ default: { create: () => ({
    get: mocks.get, post: mocks.post, interceptors: { request: { use: mocks.interceptor }, response: { use: vi.fn() } },
}) } }));
vi.mock('./supabaseClient', () => ({ supabase: { auth: { getSession: mocks.session } } }));

import { getEmbeddingStats, getSemanticRebuildStatus, rebuildPageSemanticData, searchSemantic } from './api';
const authenticate = mocks.interceptor.mock.calls[0][0];
const body = { id_page: 12, description: { content: 'Action', metadata: { arc: '', characters: [] } } };

describe('Gemini profile credentials for the Astra rebuild', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        mocks.session.mockResolvedValue({ data: { session: { access_token: 'supabase-token', user: { id: 'user-1' } } } });
    });
    afterEach(() => vi.restoreAllMocks());

    it('sends the profile key only in headers and preserves Supabase authentication', async () => {
        localStorage.setItem('google_api_key', ' profile-google-secret ');
        getSemanticRebuildStatus();
        rebuildPageSemanticData(body, 'one-piece');
        expect(mocks.get).toHaveBeenCalledWith('/admin/ai-models/semantic-rebuild-status', {
            headers: { 'X-Gemini-API-Key': 'profile-google-secret' },
        });
        expect(mocks.post).toHaveBeenCalledWith('/admin/ai-models/rebuild-page-semantic-data', body, {
            params: { manga: 'one-piece' }, timeout: 600000, headers: { 'X-Gemini-API-Key': 'profile-google-secret' },
        });
        const [url, payload, config] = mocks.post.mock.calls[0];
        expect(url).not.toContain('profile-google-secret');
        expect(JSON.stringify(payload)).not.toContain('profile-google-secret');
        expect(JSON.stringify(config.params)).not.toContain('profile-google-secret');
        const authenticated = await authenticate({ ...config, headers: { ...config.headers } });
        expect(authenticated.headers).toMatchObject({ Authorization: 'Bearer supabase-token', 'X-Gemini-API-Key': 'profile-google-secret' });
    });

    it('does not attach personal credentials to stats or search requests', () => {
        localStorage.setItem('google_api_key', 'profile-google-secret');
        getEmbeddingStats('one-piece');
        searchSemantic('Luffy');
        expect(JSON.stringify(mocks.get.mock.calls)).not.toContain('profile-google-secret');
    });

    it('reads a replaced profile key on the next page and omits removed keys', () => {
        localStorage.setItem('google_api_key', 'first-profile-key');
        rebuildPageSemanticData(body, 'one-piece');
        localStorage.setItem('google_api_key', 'second-profile-key');
        rebuildPageSemanticData(body, 'one-piece');
        localStorage.removeItem('google_api_key');
        rebuildPageSemanticData(body, 'one-piece');
        expect(mocks.post.mock.calls.map(([, , config]) => config.headers)).toEqual([
            { 'X-Gemini-API-Key': 'first-profile-key' }, { 'X-Gemini-API-Key': 'second-profile-key' }, {},
        ]);
    });

    it('still permits server-only operation when browser key storage is inaccessible', () => {
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('Storage denied'); });
        expect(() => getSemanticRebuildStatus()).not.toThrow();
        expect(() => rebuildPageSemanticData(body, 'one-piece')).not.toThrow();
        expect(mocks.post.mock.calls[0][2].headers).toEqual({});
    });
});
