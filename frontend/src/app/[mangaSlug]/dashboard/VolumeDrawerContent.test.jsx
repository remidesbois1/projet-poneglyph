import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Sheet, SheetContent } from '@/components/ui/sheet';
import VolumeDrawerContent from './VolumeDrawerContent';

vi.mock('@/components/CoverThumbnailImage', () => ({
    // eslint-disable-next-line @next/next/no-img-element
    default: ({ src, alt, onError }) => <img src={src} alt={alt} onError={onError} />,
}));

const tome = { id: 1, numero: 1, titre: 'À l’aube d’une grande aventure' };
const chapters = [
    { id: 12, numero: 12, titre: 'Un titre très long qui reste entièrement lisible sur les petits écrans', global_status: 'in_progress' },
    { id: 2, numero: 2, titre: 'Le départ', global_status: 'completed' },
    { id: 3, numero: 3, global_status: 'empty' },
];
const pages = [
    { id: 110, numero_page: 10, statut: 'rejected' },
    { id: 101, numero_page: 1, statut: 'completed' },
    { id: 102, numero_page: 2, statut: 'in_progress' },
    { id: 103, numero_page: 3, statut: 'pending_review' },
    { id: 104, numero_page: 4, statut: 'not_started' },
];

function Drawer(props) {
    return <Sheet open><SheetContent side="bottom"><VolumeDrawerContent {...props} /></SheetContent></Sheet>;
}

function setup(overrides = {}) {
    const props = {
        tome, mangaTitle: 'One Piece', chapter: null, chapters, pages: [],
        state: { status: 'chapters-ready', error: null }, isPublicViewer: false, isAdmin: false,
        deletingTarget: null, onOpenChapter: vi.fn(), onReturnToChapters: vi.fn(),
        onRetry: vi.fn(), onOpenPage: vi.fn(), onDeletePage: vi.fn(), onDeleteChapter: vi.fn(), ...overrides,
    };
    return { ...render(<Drawer {...props} />), props };
}

const pageButtons = () => within(screen.getByRole('list', { name: 'Pages du chapitre' })).getAllByRole('button', { name: /^Ouvrir la page/ });
const pageProps = { chapter: chapters[1], pages, state: { status: 'pages-ready', error: null } };

describe('VolumeDrawerContent', () => {
    beforeEach(() => {
        for (const name of ['scrollIntoView', 'scrollTo', 'hasPointerCapture', 'setPointerCapture', 'releasePointerCapture']) {
            Object.defineProperty(HTMLElement.prototype, name, { configurable: true, value: vi.fn() });
        }
    });
    afterEach(() => vi.restoreAllMocks());

    it('renders a named sheet with the existing serif typography and a scrollable body', () => {
        setup();
        expect(screen.getByRole('dialog', { name: 'Tome 1' })).toHaveAccessibleDescription(tome.titre);
        expect(screen.getByRole('heading', { name: 'Tome 1' })).toHaveClass('font-serif', 'font-black');
        expect(screen.getByRole('region', { name: 'Contenu du tome' })).toHaveClass('min-h-0', 'overflow-y-auto', 'overscroll-y-contain');
        expect(screen.getByText('3 chapitres')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /^Actions/ })).not.toBeInTheDocument();
    });

    it('uses ordered, native chapter buttons with full titles and no lifting hover', () => {
        const { props } = setup();
        const buttons = within(screen.getByRole('list', { name: 'Chapitres du tome' })).getAllByRole('button');
        expect(buttons.map(button => button.getAttribute('aria-label'))).toEqual([
            'Ouvrir le chapitre 2 : Le départ', 'Ouvrir le chapitre 3', `Ouvrir le chapitre 12 : ${chapters[0].titre}`,
        ]);
        expect(chapters.map(chapter => chapter.numero)).toEqual([12, 2, 3]);
        expect(buttons[2]).toHaveTextContent(chapters[0].titre);
        expect(buttons[2].className).not.toMatch(/translate|scale|shadow/);
        fireEvent.click(buttons[2]);
        expect(props.onOpenChapter).toHaveBeenCalledWith(chapters[0]);
    });

    it('supports title aliases and never invents a manga title', () => {
        const { props, rerender } = setup({ tome: { id: 4, numero: 4, title: 'Autre titre' } });
        expect(screen.getByRole('dialog')).toHaveAccessibleDescription('Autre titre');
        rerender(<Drawer {...props} tome={{ id: 5, numero: 5 }} />);
        expect(screen.getByRole('dialog')).toHaveAccessibleDescription('Sommaire et pages du tome.');
        expect(screen.queryByText(tome.titre)).not.toBeInTheDocument();
    });

    it.each([
        ['loading-chapters', null, 'Chargement des chapitres', 'status'],
        ['loading-pages', chapters[1], 'Chargement des pages', 'status'],
        ['chapters-empty', null, 'Aucun chapitre dans ce volume', null],
        ['pages-empty', chapters[1], 'Aucune page dans ce chapitre', null],
    ])('distinguishes %s from errors and ready content', (status, chapter, message, role) => {
        setup({ chapter, chapters: [], pages: [], state: { status, error: null } });
        expect(role ? screen.getByRole(role, { name: message }) : screen.getByText(message)).toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(screen.queryByRole('list', { name: 'Pages du chapitre' })).not.toBeInTheDocument();
        expect(screen.queryByRole('list', { name: 'Chapitres du tome' })).not.toBeInTheDocument();
        if (role) expect(screen.queryByText('0 chapitre')).not.toBeInTheDocument();
    });

    it.each(['error-chapters', 'error-pages'])('retries %s without displaying it as an empty result', status => {
        const { props } = setup({ chapter: status === 'error-pages' ? chapters[1] : null, state: { status, error: 'Réseau indisponible' } });
        expect(screen.getByRole('alert')).toHaveTextContent('Réseau indisponible');
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));
        expect(props.onRetry).toHaveBeenCalledOnce();
    });

    it('restores summary scroll and the last visited chapter after returning from pages', () => {
        const { props, rerender } = setup();
        const scroller = screen.getByRole('region', { name: 'Contenu du tome' });
        scroller.scrollTop = 280;
        fireEvent.click(screen.getByRole('button', { name: /^Ouvrir le chapitre 12/ }));
        rerender(<Drawer {...props} {...pageProps} chapter={chapters[0]} />);
        expect(scroller.scrollTop).toBe(0);
        expect(screen.getByRole('heading', { name: 'Chapitre 12' })).toHaveFocus();
        expect(screen.getByRole('button', { name: 'Chapitre suivant' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Chapitre précédent' }));
        expect(props.onOpenChapter).toHaveBeenLastCalledWith(chapters[2]);
        fireEvent.click(screen.getByRole('button', { name: 'Tous les chapitres' }));
        expect(props.onReturnToChapters).toHaveBeenCalledOnce();
        rerender(<Drawer {...props} />);
        expect(scroller.scrollTop).toBe(280);
        expect(screen.getByRole('button', { name: /^Ouvrir le chapitre 12/ })).toHaveFocus();
    });

    it('sorts pages numerically and labels every status, including rejected, without colored cards', () => {
        const { props } = setup(pageProps);
        expect(pageButtons().map(button => button.title)).toEqual([
            'Page 1 - completed', 'Page 2 - in_progress', 'Page 3 - pending_review', 'Page 4 - not_started', 'Page 10 - rejected',
        ]);
        expect(screen.getByRole('button', { name: 'Ouvrir la page 10 — Rejeté' })).toHaveTextContent('Rejeté');
        expect(pageButtons()[0].className).not.toMatch(/bg-green|translate|scale|shadow/);
        fireEvent.click(pageButtons()[0]);
        expect(props.onOpenPage).toHaveBeenCalledWith(pages[1]);
        expect(pages[0].numero_page).toBe(10);
    });

    it('filters pages with the real shadcn select and resets when that filtered status disappears', async () => {
        const { props, rerender } = setup(pageProps);
        fireEvent.keyDown(screen.getByRole('combobox', { name: 'Filtrer les pages par statut' }), { key: 'ArrowDown' });
        fireEvent.click(await screen.findByRole('option', { name: /Rejeté/ }));
        expect(pageButtons()).toHaveLength(1);
        expect(screen.getByRole('status')).toHaveTextContent('1 page affichée');
        rerender(<Drawer {...props} pages={pages.filter(page => page.statut !== 'rejected')} />);
        expect(screen.getByText('Aucune page avec ce statut.')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Afficher toutes les pages' }));
        expect(pageButtons()).toHaveLength(4);
    });

    it('does not carry a page filter over to the next chapter', async () => {
        const { props, rerender } = setup(pageProps);
        fireEvent.keyDown(screen.getByRole('combobox'), { key: 'ArrowDown' });
        fireEvent.click(await screen.findByRole('option', { name: /En cours/ }));
        expect(pageButtons()).toHaveLength(1);
        rerender(<Drawer {...props} chapter={chapters[2]} />);
        expect(pageButtons()).toHaveLength(5);
        expect(screen.getByRole('combobox')).toHaveTextContent('Tous les statuts');
    });

    it.each([[true, 'Terminé'], [false, 'Vide']])('preserves the public/authenticated fallback for absent page metadata (%s)', (isPublicViewer, status) => {
        setup({ ...pageProps, isPublicViewer, pages: [{ id: 100, numero_page: 1 }] });
        expect(screen.getByRole('button', { name: `Ouvrir la page 1 — ${status}` })).toBeInTheDocument();
    });

    it('places visible admin action buttons outside of page navigation buttons', () => {
        setup({ ...pageProps, isAdmin: true });
        const actions = screen.getAllByRole('button', { name: /^Actions de la page/ });
        expect(actions).toHaveLength(pages.length);
        for (const button of actions) {
            expect(button).toHaveClass('size-11');
            expect(button.parentElement.closest('button')).toBeNull();
            expect(button.className).not.toMatch(/hidden|group-hover/);
        }
        expect(screen.getByRole('button', { name: 'Actions du chapitre 2' })).toBeInTheDocument();
    });

    it('handles a missing or broken cover without breaking the summary', () => {
        setup({ tome: { ...tome, cover_url: '/cover.jpg' } });
        fireEvent.error(screen.getByRole('img', { name: 'Couverture du tome 1' }));
        expect(screen.getByRole('img', { name: 'Couverture indisponible' })).toBeInTheDocument();
        expect(screen.getByRole('list', { name: 'Chapitres du tome' })).toBeInTheDocument();
    });
});
