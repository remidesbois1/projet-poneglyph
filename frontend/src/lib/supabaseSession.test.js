import { describe, expect, it, vi } from 'vitest';
vi.mock('./supabaseClient', () => ({ supabase: { auth: {} } }));
import { createSupabaseSessionManager } from './supabaseSession';

const session = (token, expires_at = 5000, id = 'admin') => ({ access_token: token, expires_at, user: { id } });
function fixture(initial = session('old')) {
    let stored = initial;
    const auth = {
        getSession: vi.fn(async () => ({ data: { session: stored }, error: null })),
        refreshSession: vi.fn(async () => { stored = session('new'); return { data: { session: stored }, error: null }; }),
    };
    return { auth, manager: createSupabaseSessionManager(auth, () => 1000 * 1000), setSession: value => { stored = value; } };
}

describe('Supabase session recovery', () => {
    it('keeps guests optional but requires a logged-in account for the batch', async () => {
        const { manager, auth } = fixture(null);
        await expect(manager.getSession()).resolves.toBeNull();
        await expect(manager.getSession({ required: true })).rejects.toMatchObject({ code: 'SUPABASE_AUTH_REQUIRED' });
        expect(auth.refreshSession).not.toHaveBeenCalled();
    });

    it('refreshes near expiry even when the background timer has not run', async () => {
        const { manager, auth } = fixture(session('old', 1050));
        await expect(manager.getSession({ required: true })).resolves.toEqual(session('new'));
        expect(auth.refreshSession).toHaveBeenCalledOnce();
    });

    it('shares one rotation between concurrent failures and reuses it for a later rejection of the old token', async () => {
        const { manager, auth } = fixture();
        await expect(Promise.all(Array.from({ length: 3 }, () => manager.refreshSession('old', { expectedUserId: 'admin' }))))
            .resolves.toEqual(Array(3).fill(session('new')));
        expect(auth.refreshSession).toHaveBeenCalledOnce();
        await expect(manager.refreshSession('old')).resolves.toEqual(session('new'));
        expect(auth.refreshSession).toHaveBeenCalledOnce();
    });

    it('reuses a token already refreshed by Supabase without rotating again', async () => {
        const { manager, auth } = fixture(session('already-refreshed'));
        await expect(manager.refreshSession('old')).resolves.toEqual(session('already-refreshed'));
        expect(auth.refreshSession).not.toHaveBeenCalled();
    });

    it.each([
        [{ status: 400, code: 'refresh_token_not_found' }, 'SUPABASE_AUTH_REQUIRED'],
        [{ status: 503, name: 'AuthRetryableFetchError' }, 'SUPABASE_UNAVAILABLE'],
    ])('reports refresh failures without clearing storage, then permits another attempt', async (error, code) => {
        const { manager, auth } = fixture();
        auth.refreshSession.mockResolvedValueOnce({ data: { session: null }, error });
        await expect(manager.refreshSession('old')).rejects.toMatchObject({ code });
        await expect(manager.refreshSession('old')).resolves.toEqual(session('new'));
        expect(auth.refreshSession).toHaveBeenCalledTimes(2);
    });

    it('distinguishes getSession network failures from an expired session', async () => {
        const { manager, auth } = fixture();
        auth.getSession.mockRejectedValueOnce(new TypeError('Failed to fetch'));
        await expect(manager.getSession()).rejects.toMatchObject({ code: 'SUPABASE_UNAVAILABLE' });
    });

    it('never refreshes a signed-out or different account to retry a request from the old account', async () => {
        const { manager, auth, setSession } = fixture(session('new', 5000, 'someone-else'));
        await expect(manager.refreshSession('old', { expectedUserId: 'admin' })).rejects.toMatchObject({ code: 'SUPABASE_AUTH_REQUIRED' });
        await expect(manager.getSession({ expectedUserId: 'admin' })).rejects.toMatchObject({ code: 'SUPABASE_AUTH_REQUIRED' });
        setSession(null);
        await expect(manager.refreshSession('old')).rejects.toMatchObject({ code: 'SUPABASE_AUTH_REQUIRED' });
        expect(auth.refreshSession).not.toHaveBeenCalled();
    });
});
