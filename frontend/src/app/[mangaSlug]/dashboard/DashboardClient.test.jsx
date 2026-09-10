import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useManga } from '@/context/MangaContext';
import { useAuth } from '@/context/AuthContext';
import { useUserProfile } from '@/hooks/useUserProfile';
import { useRouter } from 'next/navigation';
import { getTomes, getChapitres, getPages, deleteBubblesForPage, deleteBubblesForChapter } from '@/lib/api';
import { toast } from 'sonner';
import DashboardClient from './DashboardClient';

vi.mock('@/context/MangaContext', () => ({ useManga: vi.fn() }));
vi.mock('@/context/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/hooks/useUserProfile', () => ({ useUserProfile: vi.fn() }));
vi.mock('next/navigation', () => ({ useRouter: vi.fn() }));
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock('@/lib/api', () => ({
    getTomes: vi.fn(),
    getChapitres: vi.fn(),
    getPages: vi.fn(),
    deleteBubblesForChapter: vi.fn(),
    deleteBubblesForPage: vi.fn(),
}));
vi.mock('@/components/CoverThumbnailImage', () => ({
    default: ({ alt }) => <div aria-label={alt} />,
}));
vi.mock('@/components/ui/sheet', async () => {
    const { createContext, useContext } = await import('react');
    const Context = createContext({});
    return {
        Sheet: ({ children, open, onOpenChange }) => <Context.Provider value={{ open, onOpenChange }}>{children}</Context.Provider>,
        SheetContent: function Content({ children, className, ref }) {
            const { open, onOpenChange } = useContext(Context);
            return open ? <div ref={ref} className={className}>
                <button type="button" onClick={() => onOpenChange(false)}>Fermer le tiroir</button>
                {children}
            </div> : null;
        },
        SheetHeader: ({ children, ...props }) => <div {...props}>{children}</div>,
        SheetTitle: ({ children, ...props }) => <h2 {...props}>{children}</h2>,
        SheetDescription: ({ children, ...props }) => <p {...props}>{children}</p>,
    };
});
vi.mock('@/components/ui/scroll-area', () => ({ ScrollArea: ({ children }) => <div>{children}</div> }));

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
    });
    return { promise, resolve, reject };
}

const volumes = [
    { id: 1, numero: 1, titre: 'Premier volume', cover_url: null },
    { id: 2, numero: 2, titre: 'Second volume', cover_url: null },
];

describe('DashboardClient async states', () => {
    beforeEach(() => {
        vi.resetAllMocks();
        useManga.mockReturnValue({ mangaSlug: 'one-piece', currentManga: { titre: 'One Piece' } });
        useAuth.mockReturnValue({ session: null });
        useUserProfile.mockReturnValue({ profile: null, loading: true });
        useRouter.mockReturnValue({ push: vi.fn() });
        getChapitres.mockResolvedValue({ data: [] });
        getPages.mockResolvedValue({ data: [] });
        for (const name of ['scrollIntoView', 'scrollTo', 'hasPointerCapture', 'setPointerCapture', 'releasePointerCapture']) {
            Object.defineProperty(HTMLElement.prototype, name, { configurable: true, value: vi.fn() });
        }
    });

    afterEach(() => vi.restoreAllMocks());

    it('renders a catalogue skeleton instead of a blank page while loading', () => {
        getTomes.mockReturnValue(new Promise(() => {}));

        const { container } = render(<DashboardClient />);

        expect(screen.getByRole('heading', { name: /Bibliothèque One Piece/i })).toBeInTheDocument();
        expect(screen.getByRole('status', { name: 'Chargement des tomes' })).toBeInTheDocument();
        expect(container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(24);
    });

    it('distinguishes a failed catalogue from an empty catalogue and retries', async () => {
        getTomes
            .mockRejectedValueOnce(new Error('Catalogue indisponible'))
            .mockResolvedValueOnce({ data: [] });

        render(<DashboardClient />);

        expect(await screen.findByRole('alert')).toHaveTextContent('Catalogue indisponible');
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));

        expect(await screen.findByText('Aucun volume disponible')).toBeInTheDocument();
        expect(getTomes).toHaveBeenCalledTimes(2);
    });

    it('waits for the manga slug before requesting a catalogue', async () => {
        useManga.mockReturnValue({ mangaSlug: null, currentManga: null });
        getTomes.mockResolvedValue({ data: volumes });
        const { rerender } = render(<DashboardClient />);
        expect(getTomes).not.toHaveBeenCalled();
        expect(screen.getByRole('status', { name: 'Chargement des tomes' })).toBeInTheDocument();

        useManga.mockReturnValue({ mangaSlug: 'one-piece', currentManga: { titre: 'One Piece' } });
        rerender(<DashboardClient />);
        expect(await screen.findByRole('button', { name: 'Ouvrir le tome 1 : Premier volume' })).toBeInTheDocument();
        expect(getTomes).toHaveBeenCalledWith('one-piece');
    });

    it('resets the search and catalogue when switching to another manga', async () => {
        const nextCatalogue = deferred();
        getTomes.mockResolvedValueOnce({ data: volumes }).mockReturnValueOnce(nextCatalogue.promise);
        const { rerender } = render(<DashboardClient />);
        await screen.findByRole('button', { name: 'Ouvrir le tome 1 : Premier volume' });
        fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'Premier' } });

        useManga.mockReturnValue({ mangaSlug: 'naruto', currentManga: { titre: 'Naruto' } });
        rerender(<DashboardClient />);
        expect(screen.getByRole('heading', { name: 'Bibliothèque Naruto' })).toBeInTheDocument();
        expect(screen.getByRole('searchbox')).toHaveValue('');
        expect(screen.getByRole('status', { name: 'Chargement des tomes' })).toBeInTheDocument();
        expect(screen.queryByText('Premier volume')).not.toBeInTheDocument();

        nextCatalogue.resolve({ data: [{ id: 5, numero: 1, titre: 'Nouveau manga' }] });
        expect(await screen.findByRole('button', { name: 'Ouvrir le tome 1 : Nouveau manga' })).toBeInTheDocument();
        expect(getTomes).toHaveBeenLastCalledWith('naruto');
    });

    it('ignores an older chapter response after another volume is selected', async () => {
        const firstRequest = deferred();
        getTomes.mockResolvedValue({ data: volumes });
        getChapitres
            .mockReturnValueOnce(firstRequest.promise)
            .mockResolvedValueOnce({ data: [{ id: 22, numero: 22, titre: 'Chapitre récent', global_status: 'empty' }] });

        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir le tome 1 : Premier volume' }));
        fireEvent.click(screen.getByRole('button', { name: 'Ouvrir le tome 2 : Second volume' }));

        expect(await screen.findByRole('button', { name: /^Ouvrir le chapitre 22/ })).toBeInTheDocument();
        firstRequest.resolve({ data: [{ id: 11, numero: 11, titre: 'Chapitre obsolète', global_status: 'empty' }] });

        await waitFor(() => expect(screen.queryByRole('button', { name: /^Ouvrir le chapitre 11/ })).not.toBeInTheDocument());
        expect(screen.getByRole('button', { name: /^Ouvrir le chapitre 22/ })).toBeInTheDocument();
    });

    it('does not expose chapters from the previous volume when the next request fails', async () => {
        getTomes.mockResolvedValue({ data: volumes });
        getChapitres
            .mockResolvedValueOnce({ data: [{ id: 11, numero: 11, titre: 'Ancien chapitre', global_status: 'empty' }] })
            .mockRejectedValueOnce(new Error('Chapitres indisponibles'));

        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir le tome 1 : Premier volume' }));
        expect(await screen.findByRole('button', { name: /^Ouvrir le chapitre 11/ })).toBeInTheDocument();

        fireEvent.click(screen.getByRole('button', { name: 'Ouvrir le tome 2 : Second volume' }));

        expect(await screen.findByRole('alert')).toHaveTextContent('Chapitres indisponibles');
        expect(screen.queryByRole('button', { name: /^Ouvrir le chapitre 11/ })).not.toBeInTheDocument();
    });

    it('shows a recoverable page error in the drawer and retries the selected chapter', async () => {
        getTomes.mockResolvedValue({ data: [volumes[0]] });
        getChapitres.mockResolvedValue({
            data: [{ id: 11, numero: 11, titre: 'Chapitre test', global_status: 'empty' }],
        });
        getPages
            .mockRejectedValueOnce(new Error('Pages indisponibles'))
            .mockResolvedValueOnce({ data: [{ id: 101, numero_page: 1, statut: 'not_started' }] });

        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir le tome 1 : Premier volume' }));
        fireEvent.click(await screen.findByRole('button', { name: /^Ouvrir le chapitre 11/ }));

        expect(await screen.findByRole('alert')).toHaveTextContent('Pages indisponibles');
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));

        expect(await screen.findByTitle('Page 1 - not_started')).toBeInTheDocument();
        expect(getPages).toHaveBeenCalledTimes(2);
    });

    it('opens the correct annotation page and keeps public pages without private status readable', async () => {
        getTomes.mockResolvedValue({ data: [volumes[0]] });
        getChapitres.mockResolvedValue({ data: [{ id: 11, numero: 11, titre: 'Départ' }] });
        getPages.mockResolvedValue({ data: [{ id: 101, numero_page: 1 }] });
        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: /^Ouvrir le tome 1/ }));
        fireEvent.click(await screen.findByRole('button', { name: /^Ouvrir le chapitre 11/ }));
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir la page 1 — Terminé' }));
        expect(useRouter().push).toHaveBeenCalledWith('/one-piece/annotate/101');
        expect(screen.queryByRole('button', { name: /^Actions/ })).not.toBeInTheDocument();
    });

    it('ignores late pages after navigating to another chapter and returns to the summary without a request', async () => {
        const latePages = deferred();
        getTomes.mockResolvedValue({ data: volumes });
        getChapitres.mockResolvedValue({ data: [{ id: 12, numero: 12 }, { id: 11, numero: 11 }] });
        getPages.mockReturnValueOnce(latePages.promise).mockResolvedValueOnce({ data: [{ id: 202, numero_page: 20, statut: 'completed' }] });
        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: /^Ouvrir le tome 1/ }));
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir le chapitre 11' }));
        expect(screen.getByRole('button', { name: 'Chapitre précédent' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Chapitre suivant' }));
        expect(await screen.findByRole('button', { name: 'Ouvrir la page 20 — Terminé' })).toBeInTheDocument();
        await act(async () => latePages.resolve({ data: [{ id: 101, numero_page: 1, statut: 'not_started' }] }));
        expect(screen.queryByTitle('Page 1 - not_started')).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Chapitre suivant' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Tous les chapitres' }));
        expect(screen.getByRole('button', { name: 'Ouvrir le chapitre 12' })).toHaveFocus();
        expect(getChapitres).toHaveBeenCalledTimes(1);
    });

    it('ignores a pending chapter response after closing and reopening the drawer', async () => {
        const lateChapters = deferred();
        getTomes.mockResolvedValue({ data: volumes });
        getChapitres.mockReturnValueOnce(lateChapters.promise).mockResolvedValueOnce({ data: [{ id: 22, numero: 22 }] });
        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: /^Ouvrir le tome 1/ }));
        fireEvent.click(screen.getByRole('button', { name: 'Fermer le tiroir' }));
        fireEvent.click(screen.getByRole('button', { name: /^Ouvrir le tome 2/ }));
        expect(await screen.findByRole('button', { name: 'Ouvrir le chapitre 22' })).toBeInTheDocument();
        await act(async () => lateChapters.resolve({ data: [{ id: 11, numero: 11 }] }));
        expect(screen.queryByRole('button', { name: 'Ouvrir le chapitre 11' })).not.toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Tome 2' })).toBeInTheDocument();
    });

    async function openAdminPages() {
        useAuth.mockReturnValue({ session: { user: { id: 'admin' } } });
        useUserProfile.mockReturnValue({ profile: { role: 'Admin' }, loading: false });
        getTomes.mockResolvedValue({ data: volumes });
        getChapitres.mockResolvedValue({ data: [{ id: 11, numero: 11, global_status: 'completed' }, { id: 12, numero: 12 }] });
        getPages.mockResolvedValue({ data: [{ id: 101, numero_page: 1, statut: 'completed' }] });
        render(<DashboardClient />);
        fireEvent.click(await screen.findByRole('button', { name: /^Ouvrir le tome 1/ }));
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir le chapitre 11' }));
        await screen.findByRole('button', { name: 'Actions de la page 1' });
    }

    async function requestDeletion(kind) {
        fireEvent.keyDown(screen.getByRole('button', { name: kind === 'page' ? 'Actions de la page 1' : 'Actions du chapitre 11' }), { key: 'ArrowDown' });
        fireEvent.click(await screen.findByRole('menuitem', { name: kind === 'page' ? 'Supprimer les bulles' : 'Vider le chapitre' }));
        return screen.findByRole('alertdialog');
    }

    it('requires confirmation, cancels safely and updates the page after an admin deletion', async () => {
        deleteBubblesForPage.mockResolvedValue({ data: { deleted: 3 } });
        await openAdminPages();
        let dialog = await requestDeletion('page');
        expect(deleteBubblesForPage).not.toHaveBeenCalled();
        expect(within(dialog).getByRole('button', { name: 'Annuler' })).toHaveFocus();
        fireEvent.click(within(dialog).getByRole('button', { name: 'Annuler' }));
        expect(deleteBubblesForPage).not.toHaveBeenCalled();
        dialog = await requestDeletion('page');
        fireEvent.click(within(dialog).getByRole('button', { name: 'Supprimer les bulles' }));
        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        expect(deleteBubblesForPage).toHaveBeenCalledWith(101);
        expect(screen.getByRole('button', { name: 'Ouvrir la page 1 — Vide' })).toBeInTheDocument();
        expect(useRouter().push).not.toHaveBeenCalled();
        expect(toast.success).toHaveBeenCalledWith('3 bulle(s) supprimée(s) sur la page 1.');
    });

    it('keeps failed deletions recoverable without changing the page status', async () => {
        deleteBubblesForPage.mockRejectedValueOnce(new Error('Hors ligne')).mockResolvedValueOnce({ data: { deleted: 1 } });
        await openAdminPages();
        const dialog = await requestDeletion('page');
        fireEvent.click(within(dialog).getByRole('button', { name: 'Supprimer les bulles' }));
        await waitFor(() => expect(toast.error).toHaveBeenCalled());
        expect(screen.getByRole('alertdialog')).toBeInTheDocument();
        expect(screen.getByTitle('Page 1 - completed')).toBeInTheDocument();
        fireEvent.click(within(dialog).getByRole('button', { name: 'Supprimer les bulles' }));
        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        expect(deleteBubblesForPage).toHaveBeenCalledTimes(2);
        expect(screen.getByTitle('Page 1 - not_started')).toBeInTheDocument();
    });

    it('clears a chapter only once and updates its pages and summary status', async () => {
        const deletion = deferred();
        deleteBubblesForChapter.mockReturnValue(deletion.promise);
        await openAdminPages();
        const dialog = await requestDeletion('chapter');
        const confirm = within(dialog).getByRole('button', { name: 'Supprimer les bulles' });
        fireEvent.click(confirm);
        fireEvent.click(confirm);
        expect(deleteBubblesForChapter).toHaveBeenCalledTimes(1);
        expect(deleteBubblesForChapter).toHaveBeenCalledWith(11);
        expect(within(dialog).getByRole('button', { name: 'Annuler' })).toBeDisabled();
        fireEvent.keyDown(dialog, { key: 'Escape' });
        expect(screen.getByRole('alertdialog')).toBeInTheDocument();
        await act(async () => deletion.resolve({ data: { deleted: 7 } }));
        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        expect(screen.getByTitle('Page 1 - not_started')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Tous les chapitres' }));
        expect(screen.getByRole('button', { name: 'Ouvrir le chapitre 11' })).toHaveTextContent('Vide');
    });

    it('does not let an old chapter deletion reset pages loaded in another tome', async () => {
        const deletion = deferred();
        deleteBubblesForChapter.mockReturnValue(deletion.promise);
        await openAdminPages();
        const secondTome = screen.getByRole('button', { name: /^Ouvrir le tome 2/ });
        const dialog = await requestDeletion('chapter');
        fireEvent.click(within(dialog).getByRole('button', { name: 'Supprimer les bulles' }));
        getChapitres.mockResolvedValueOnce({ data: [{ id: 22, numero: 22, global_status: 'completed' }] });
        getPages.mockResolvedValueOnce({ data: [{ id: 202, numero_page: 2, statut: 'completed' }] });
        // Force a context change while the request is pending, independently of the modal focus trap.
        fireEvent.click(secondTome);
        fireEvent.click(await screen.findByRole('button', { name: 'Ouvrir le chapitre 22' }));
        expect(await screen.findByTitle('Page 2 - completed')).toBeInTheDocument();
        await act(async () => deletion.resolve({ data: { deleted: 1 } }));
        expect(screen.getByTitle('Page 2 - completed')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Chapitre 22' })).toBeInTheDocument();
    });
});
