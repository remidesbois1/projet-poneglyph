import { afterEach, describe, expect, it, vi } from 'vitest';
vi.mock('./supabaseSession', () => ({ refreshSupabaseSession: vi.fn() }));
import { refreshSupabaseSession } from './supabaseSession';
import { fetchOriginalPageImage, fetchOriginalPageThumbnail } from './pageImageClient';

describe('fetchOriginalPageImage', () => {
    const originalBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

    afterEach(() => {
        process.env.NEXT_PUBLIC_BACKEND_URL = originalBackendUrl;
    });

    it('sends the Supabase access token only in the Authorization header', async () => {
        process.env.NEXT_PUBLIC_BACKEND_URL = 'https://api.example.test/api/';
        const blob = new Blob(['image'], { type: 'image/avif' });
        const fetchImpl = vi.fn().mockResolvedValue({ ok: true, blob: async () => blob });

        await expect(fetchOriginalPageImage('page 42', 'access-secret', { fetchImpl })).resolves.toBe(blob);

        expect(fetchImpl).toHaveBeenCalledOnce();
        const [url, options] = fetchImpl.mock.calls[0];
        expect(url).toBe('https://api.example.test/api/pages/page%2042/image/original');
        expect(url).not.toContain('access-secret');
        expect(options.headers).toEqual({ Authorization: 'Bearer access-secret' });
        expect(options.cache).toBe('no-store');
        expect(options.credentials).toBe('omit');
    });

    it('fails before making a request when no token is available', async () => {
        const fetchImpl = vi.fn();
        await expect(fetchOriginalPageImage('42', null, { fetchImpl })).rejects.toThrow(/Authentication/);
        expect(fetchImpl).not.toHaveBeenCalled();
    });

    it('stops after one authentication retry when the replacement token is also rejected', async () => {
        const fetchImpl = vi.fn().mockResolvedValue({ ok: false, status: 401 });
        refreshSupabaseSession.mockResolvedValue({ access_token: 'new-token' });
        await expect(fetchOriginalPageImage('42', 'expired', { fetchImpl })).rejects.toThrow(/Session Supabase expirée/);
        expect(fetchImpl).toHaveBeenCalledTimes(2);
    });

    it('recovers the private original using the shared session refresh and keeps the abort signal', async () => {
        const blob = new Blob(['original'], { type: 'image/avif' });
        const signal = new AbortController().signal;
        const refreshSession = vi.fn().mockResolvedValue({ access_token: 'new-token' });
        const fetchImpl = vi.fn().mockResolvedValueOnce({ ok: false, status: 401 })
            .mockResolvedValueOnce({ ok: true, blob: async () => blob });
        await expect(fetchOriginalPageImage(42, 'expired', { fetchImpl, refreshSession, signal, expectedUserId: 'admin' })).resolves.toBe(blob);
        expect(refreshSession).toHaveBeenCalledWith('expired', { expectedUserId: 'admin' });
        expect(fetchImpl.mock.calls.map(([, options]) => options.headers.Authorization)).toEqual(['Bearer expired', 'Bearer new-token']);
        expect(fetchImpl.mock.calls.every(([, options]) => options.signal === signal && options.cache === 'no-store')).toBe(true);
    });

    it('retries an authentication outage without refreshing and does not retry a general image outage', async () => {
        const refreshSession = vi.fn();
        const fetchImpl = vi.fn().mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({ code: 'SUPABASE_UNAVAILABLE' }) })
            .mockResolvedValueOnce({ ok: true, blob: async () => new Blob(['image']) });
        await fetchOriginalPageImage(42, 'token', { fetchImpl, refreshSession });
        expect(fetchImpl).toHaveBeenCalledTimes(2);
        expect(refreshSession).not.toHaveBeenCalled();
        fetchImpl.mockClear().mockResolvedValue({ ok: false, status: 503, json: async () => ({ code: 'IMAGE_UNAVAILABLE', error: 'Image indisponible' }) });
        await expect(fetchOriginalPageImage(42, 'token', { fetchImpl, refreshSession })).rejects.toThrow('Image indisponible');
        expect(fetchImpl).toHaveBeenCalledOnce();
    });

    it('does not refresh or retry real permission failures, or fetch again after cancellation', async () => {
        const refreshSession = vi.fn();
        const fetchImpl = vi.fn().mockResolvedValue({ ok: false, status: 403 });
        await expect(fetchOriginalPageImage(42, 'token', { fetchImpl, refreshSession })).rejects.toMatchObject({ status: 403 });
        expect(fetchImpl).toHaveBeenCalledOnce();
        expect(refreshSession).not.toHaveBeenCalled();
        const controller = new AbortController();
        refreshSession.mockImplementation(async () => { controller.abort(); return { access_token: 'new-token' }; });
        fetchImpl.mockClear().mockResolvedValue({ ok: false, status: 401 });
        await expect(fetchOriginalPageImage(42, 'token', { fetchImpl, refreshSession, signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' });
        expect(fetchImpl).toHaveBeenCalledOnce();
    });

    it('requests a server-side thumbnail at the requested display width', async () => {
        process.env.NEXT_PUBLIC_BACKEND_URL = 'https://api.example.test/api';
        const blob = new Blob(['thumbnail'], { type: 'image/avif' });
        const fetchImpl = vi.fn().mockResolvedValue({ ok: true, blob: async () => blob });

        await expect(fetchOriginalPageThumbnail('42', 'access-secret', {
            width: 640,
            fetchImpl,
        })).resolves.toBe(blob);

        expect(fetchImpl).toHaveBeenCalledWith(
            'https://api.example.test/api/pages/42/image/original/thumbnail?width=640',
            expect.objectContaining({
                headers: { Authorization: 'Bearer access-secret' },
            })
        );
    });
});
