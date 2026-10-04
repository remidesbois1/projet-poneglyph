import { expect, it, vi } from 'vitest';
import { redirect } from 'next/navigation';
import Page from './page';

vi.mock('next/navigation', () => ({ redirect: vi.fn() }));

it('redirects existing submissions links to the moderation tab for the same manga', async () => {
    await Page({ params: Promise.resolve({ mangaSlug: 'one-piece' }) });
    expect(redirect).toHaveBeenCalledWith('/one-piece/moderation?view=submissions');
});
