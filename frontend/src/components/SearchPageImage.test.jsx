import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ auth: vi.fn(), original: vi.fn() }));
vi.mock('@/context/AuthContext', () => ({ useAuth: mocks.auth }));
vi.mock('@/lib/pageImageClient', () => ({ fetchOriginalPageImage: mocks.original }));
import SearchPageImage from './SearchPageImage';

const account = { session: { access_token: 'access-secret', user: { id: 'reader' } }, loading: false };
const props = { pageId: 42, url: '/api/pages/42/image', alt: 'Page trouvée' };

describe('search result images', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.auth.mockReturnValue(account);
        mocks.original.mockResolvedValue(new Blob(['sharp original'], { type: 'image/avif' }));
        let count = 0;
        URL.createObjectURL = vi.fn(() => `blob:original-${++count}`);
        URL.revokeObjectURL = vi.fn();
    });
    afterEach(() => vi.restoreAllMocks());

    it('loads unblurred private thumbnails for semantic results using the authenticated fetch', async () => {
        const { container, unmount } = render(<SearchPageImage {...props} thumbnail loading="lazy" />);
        await waitFor(() => expect(screen.getByRole('img')).toHaveAttribute('src', 'blob:original-1'));
        expect(mocks.original).toHaveBeenCalledWith(42, 'access-secret', expect.objectContaining({
            thumbnail: true, width: 640, expectedUserId: 'reader', signal: expect.any(AbortSignal),
        }));
        expect(container.innerHTML).not.toContain('/pages/42/image');
        expect(container.innerHTML).not.toContain('access-secret');
        expect(screen.getByRole('img')).toHaveAttribute('loading', 'lazy');
        unmount();
        expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:original-1');
    });

    it('uses the full original for bubble crops and preserves pixel offsets', async () => {
        render(<SearchPageImage {...props} alt="Bubble crop" style={{ position: 'absolute', left: '-123px', top: '-456px' }} className="max-w-none" />);
        const image = await screen.findByRole('img', { name: 'Bubble crop' });
        expect(mocks.original).toHaveBeenCalledWith(42, 'access-secret', expect.objectContaining({ thumbnail: false }));
        expect(image).toHaveStyle({ position: 'absolute', left: '-123px', top: '-456px' });
        expect(image).toHaveClass('max-w-none');
        expect(image).toHaveAttribute('src', 'blob:original-1');
    });

    it('loads the full private original for the best OCR result', async () => {
        render(<SearchPageImage {...props} alt="Meilleur resultat OCR" />);
        expect(await screen.findByRole('img', { name: 'Meilleur resultat OCR' })).toHaveAttribute('src', 'blob:original-1');
        expect(mocks.original).toHaveBeenCalledWith(42, 'access-secret', expect.objectContaining({ thumbnail: false }));
    });

    it('keeps the protected public preview for anonymous visitors', () => {
        mocks.auth.mockReturnValue({ session: null, loading: false });
        render(<SearchPageImage {...props} />);
        expect(screen.getByRole('img')).toHaveAttribute('src', expect.stringMatching(/\/pages\/42\/image$/));
        expect(mocks.original).not.toHaveBeenCalled();
    });

    it('waits for authentication initialization and switches an existing result to the private image after login', async () => {
        mocks.auth.mockReturnValue({ session: null, loading: true });
        const { rerender } = render(<SearchPageImage {...props} />);
        expect(screen.getByRole('status')).toHaveTextContent('Chargement');
        expect(screen.queryByRole('img')).not.toBeInTheDocument();
        expect(mocks.original).not.toHaveBeenCalled();
        mocks.auth.mockReturnValue({ session: null, loading: false });
        rerender(<SearchPageImage {...props} />);
        expect(screen.getByRole('img')).toHaveAttribute('src', expect.stringMatching(/\/pages\/42\/image$/));
        mocks.auth.mockReturnValue(account);
        rerender(<SearchPageImage {...props} />);
        expect(screen.queryByRole('img')).not.toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('img')).toHaveAttribute('src', 'blob:original-1'));
    });

    it('removes the private blob immediately on logout', async () => {
        const { rerender } = render(<SearchPageImage {...props} />);
        await screen.findByRole('img');
        const signal = mocks.original.mock.calls[0][2].signal;
        mocks.auth.mockReturnValue({ session: null, loading: false });
        rerender(<SearchPageImage {...props} />);
        expect(screen.getByRole('img')).toHaveAttribute('src', expect.stringMatching(/\/pages\/42\/image$/));
        expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:original-1');
        expect(signal.aborted).toBe(true);
    });

    it('shows a loading or error state instead of silently using a blurry preview for an authenticated reader', async () => {
        let reject;
        mocks.original.mockImplementationOnce(() => new Promise((resolve, failure) => { reject = failure; }));
        render(<SearchPageImage {...props} />);
        expect(screen.getByRole('status')).toBeInTheDocument();
        expect(screen.queryByRole('img')).not.toBeInTheDocument();
        await act(async () => reject(new Error('Network unavailable')));
        expect(screen.getByRole('alert')).toHaveTextContent('Image indisponible');
        expect(screen.queryByRole('img')).not.toBeInTheDocument();
    });

    it('does not reuse a thumbnail when the same page switches to a full-size bubble crop', async () => {
        const { rerender } = render(<SearchPageImage {...props} thumbnail />);
        await screen.findByRole('img');
        let resolveFull;
        mocks.original.mockImplementationOnce(() => new Promise(resolve => { resolveFull = resolve; }));
        rerender(<SearchPageImage {...props} />);
        expect(screen.queryByRole('img')).not.toBeInTheDocument();
        expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:original-1');
        await act(async () => resolveFull(new Blob(['full original'])));
        expect(screen.getByRole('img')).toHaveAttribute('src', 'blob:original-2');
    });
});
