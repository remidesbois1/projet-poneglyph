import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import VolumeLibrary from './VolumeLibrary';

vi.mock('@/components/CoverThumbnailImage', () => ({
    // A native image preserves non-bubbling error events without invoking Next's image loader.
    // eslint-disable-next-line @next/next/no-img-element
    default: ({ src, alt, onError }) => <img src={src} alt={alt} onError={onError} />,
}));

const makeTomes = count => Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    numero: index + 1,
    titre: `Aventure ${index + 1}`,
    cover_url: null,
}));

function setup(overrides = {}) {
    const props = { mangaTitle: 'One Piece', tomes: makeTomes(3), status: 'ready', onRetry: vi.fn(), onOpenTome: vi.fn(), ...overrides };
    return { ...render(<VolumeLibrary {...props} />), props };
}

const search = value => fireEvent.change(screen.getByRole('searchbox', { name: 'Rechercher un tome' }), { target: { value } });
const tomeButtons = () => within(screen.getByRole('list', { name: 'Tomes' })).getAllByRole('button');

describe('VolumeLibrary', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        // Radix uses these browser APIs when its real select is opened in jsdom.
        for (const name of ['scrollIntoView', 'scrollTo', 'hasPointerCapture', 'setPointerCapture', 'releasePointerCapture']) {
            Object.defineProperty(HTMLElement.prototype, name, { configurable: true, value: vi.fn() });
        }
    });

    afterEach(() => vi.restoreAllMocks());

    it('keeps the existing title style and confines scrolling to the catalogue', () => {
        setup();
        expect(screen.getByRole('heading', { name: 'Bibliothèque One Piece' })).toHaveClass('poneglyph-title', 'font-extrabold');
        expect(screen.getByRole('region', { name: 'Bibliothèque One Piece' })).toHaveClass('h-full', 'overflow-y-auto');
        expect(screen.getByText('3 tomes')).toBeInTheDocument();
        expect(screen.queryByRole('navigation', { name: 'Pagination des tomes' })).not.toBeInTheDocument();
    });

    it('shows a geometry-matched loading state, not an empty catalogue', () => {
        const { container } = setup({ tomes: [], status: 'loading' });
        expect(screen.getByRole('status', { name: 'Chargement des tomes' })).toBeInTheDocument();
        expect(container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(24);
        expect(container.querySelector('[aria-busy="true"]')).toBeInTheDocument();
        expect(screen.queryByText('Aucun volume disponible')).not.toBeInTheDocument();
    });

    it('sorts numerically without mutating the input and opens a tome through a native button', () => {
        const tomes = [{ id: 10, numero: 10 }, { id: 2, numero: 2 }, { id: 1, numero: 1 }];
        const { props } = setup({ tomes });
        expect(tomeButtons().map(button => button.getAttribute('aria-label'))).toEqual(['Ouvrir le tome 1', 'Ouvrir le tome 2', 'Ouvrir le tome 10']);
        expect(tomes.map(tome => tome.numero)).toEqual([10, 2, 1]);
        const button = screen.getByRole('button', { name: 'Ouvrir le tome 2' });
        expect(button).toHaveAttribute('type', 'button');
        expect(button).toHaveAttribute('aria-haspopup', 'dialog');
        fireEvent.click(button);
        expect(props.onOpenTome).toHaveBeenCalledWith(tomes[1]);
    });

    it.each(['12', 'tome 12', 'Volume 12', '0012'])('finds the exact tome number for "%s"', query => {
        setup({ tomes: [{ id: 12, numero: 12 }, { id: 120, numero: 120 }] });
        search(query);
        expect(tomeButtons()).toHaveLength(1);
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 12');
        expect(screen.getByRole('status')).toHaveTextContent('1–1 sur 1 tome trouvé');
    });

    it('supports volume zero and accent-insensitive multiword title search', () => {
        setup({ tomes: [{ id: 0, numero: 0, titre: 'Été sur une île' }, { id: 1, numero: 1, titre: 'Ailleurs' }] });
        search('0');
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 0 : Été sur une île');
        search(' ILE  ete ');
        expect(tomeButtons()).toHaveLength(1);
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 0 : Été sur une île');
    });

    it('uses the available title aliases without inventing a title for a missing one', () => {
        setup({ tomes: [{ id: 1, numero: 1, title: 'Titre alternatif' }, { id: 2, numero: 2, nom: 'Autre titre' }, { id: 3, numero: 3 }] });
        expect(screen.getByText('Titre alternatif')).toBeInTheDocument();
        expect(screen.getByText('Autre titre')).toBeInTheDocument();
        expect(screen.getByText('Sans titre')).toBeInTheDocument();
        search('alternatif');
        expect(tomeButtons()).toHaveLength(1);
    });

    it('provides an explicit no-results state with a working reset and no unusable pagination', () => {
        setup({ tomes: makeTomes(50) });
        search('introuvable');
        expect(screen.getByText('Aucun tome trouvé')).toBeInTheDocument();
        expect(screen.queryByText('Aucun volume disponible')).not.toBeInTheDocument();
        expect(screen.queryByRole('navigation', { name: 'Pagination des tomes' })).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Réinitialiser la recherche' }));
        expect(tomeButtons()).toHaveLength(24);
        expect(screen.getByRole('searchbox')).toHaveFocus();
        expect(screen.getByRole('searchbox')).toHaveValue('');
    });

    it('clears a search using the clear button or Escape without losing input focus', () => {
        setup();
        search('Aventure 1');
        fireEvent.click(screen.getByRole('button', { name: 'Effacer la recherche' }));
        expect(screen.getByRole('searchbox')).toHaveFocus();
        expect(tomeButtons()).toHaveLength(3);
        search('Aventure 2');
        fireEvent.keyDown(screen.getByRole('searchbox'), { key: 'Escape' });
        expect(screen.getByRole('searchbox')).toHaveValue('');
        expect(tomeButtons()).toHaveLength(3);
    });

    it('paginates by 24, disables the boundaries and focuses the newly displayed first tome', () => {
        setup({ tomes: makeTomes(49) });
        expect(tomeButtons()).toHaveLength(24);
        expect(screen.getByRole('button', { name: 'Page précédente' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        expect(tomeButtons()).toHaveLength(24);
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 25 : Aventure 25');
        expect(tomeButtons()[0]).toHaveFocus();
        expect(HTMLElement.prototype.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'instant' });
        expect(screen.getByRole('button', { name: 'Page 2', exact: true })).toHaveAttribute('aria-current', 'page');
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        expect(tomeButtons()).toHaveLength(1);
        expect(screen.getByRole('status')).toHaveTextContent('49–49 sur 49 tomes');
        expect(screen.getByRole('button', { name: 'Page suivante' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Page 1', exact: true }));
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 1 : Aventure 1');
    });

    it('resets to the first page when searching or changing the sort with the real shadcn select', async () => {
        setup({ tomes: makeTomes(50) });
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        search('Aventure');
        expect(screen.getByRole('button', { name: 'Page 1', exact: true })).toHaveAttribute('aria-current', 'page');
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        fireEvent.keyDown(screen.getByRole('combobox', { name: 'Ordre des tomes' }), { key: 'ArrowDown' });
        fireEvent.click(await screen.findByRole('option', { name: 'Numéro décroissant' }));
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 50 : Aventure 50');
        expect(screen.getByRole('button', { name: 'Page 1', exact: true })).toHaveAttribute('aria-current', 'page');
    });

    it('preserves the selection of page and search when switching between grid and list', () => {
        setup({ tomes: makeTomes(50) });
        search('Aventure');
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        fireEvent.click(screen.getByRole('button', { name: 'Afficher en liste' }));
        expect(screen.getByRole('button', { name: 'Afficher en liste' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('button', { name: 'Afficher en grille' })).toHaveAttribute('aria-pressed', 'false');
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 25 : Aventure 25');
        expect(screen.getByRole('searchbox')).toHaveValue('Aventure');
        fireEvent.click(screen.getByRole('button', { name: 'Afficher en grille' }));
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 25 : Aventure 25');
    });

    it('clamps pagination if a refreshed catalogue is shorter', () => {
        const { props, rerender } = setup({ tomes: makeTomes(50) });
        fireEvent.click(screen.getByRole('button', { name: 'Page 3', exact: true }));
        rerender(<VolumeLibrary {...props} tomes={makeTomes(25)} />);
        expect(tomeButtons()).toHaveLength(1);
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 25 : Aventure 25');
        expect(screen.getByRole('button', { name: 'Page suivante' })).toBeDisabled();
        expect(screen.getByRole('status')).toHaveTextContent('25–25 sur 25 tomes');
    });

    it('keeps first, last and adjacent pages accessible in a long catalogue', () => {
        setup({ tomes: makeTomes(240) });
        fireEvent.click(screen.getByRole('button', { name: 'Page 4', exact: true }));
        fireEvent.click(screen.getByRole('button', { name: 'Page 5', exact: true }));
        for (const number of [1, 4, 5, 6, 10]) expect(screen.getByRole('button', { name: `Page ${number}`, exact: true })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Page 5', exact: true })).toHaveAttribute('aria-current', 'page');
        fireEvent.click(screen.getByRole('button', { name: 'Page 10', exact: true }));
        expect(tomeButtons()[0]).toHaveAccessibleName('Ouvrir le tome 217 : Aventure 217');
    });

    it('renders a fallback for a broken cover without preventing the tome from opening', () => {
        const tome = { id: 1, numero: 1, titre: 'Premier tome', cover_url: '/s3-proxy/covers/one-piece/1.jpg' };
        const { props } = setup({ tomes: [tome] });
        fireEvent.error(screen.getByRole('img', { name: 'Couverture du tome 1' }));
        expect(screen.queryByRole('img', { name: 'Couverture du tome 1' })).not.toBeInTheDocument();
        expect(screen.getByText('Couverture indisponible')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Ouvrir le tome 1 : Premier tome' }));
        expect(props.onOpenTome).toHaveBeenCalledWith(tome);
    });

    it('distinguishes an empty catalogue from a recoverable error', () => {
        const { props, rerender } = setup({ tomes: [], status: 'empty' });
        expect(screen.getByText('Aucun volume disponible')).toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        rerender(<VolumeLibrary {...props} status="error" error="Catalogue indisponible" />);
        expect(screen.getByRole('alert')).toHaveTextContent('Catalogue indisponible');
        expect(screen.queryByText('Aucun volume disponible')).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));
        expect(props.onRetry).toHaveBeenCalledOnce();
    });
});
