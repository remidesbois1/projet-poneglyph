import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import SearchResults from './SearchResults';
import { getSearchMode } from './searchModes';

vi.mock('@/components/SearchPageImage', () => ({
    // A plain image keeps these tests on the layout, away from the authenticated image pipeline.
    // eslint-disable-next-line @next/next/no-img-element
    default: ({ alt, pageId, thumbnail }) => <img alt={alt} data-page={pageId} data-thumbnail={String(Boolean(thumbnail))} />,
}));

const bubble = (id, content, page = id) => ({
    type: 'bubble', id, page_id: 100 + id, url_image: `/api/pages/${100 + id}/image`,
    coords: { x: 10, y: 20, w: 100, h: 80 }, content, context: `Tome 1 - Chap. 2 - Page ${page}`,
});
const semanticPage = (id, similarity) => ({
    type: 'semantic', id: `page-${id}`, page_id: id, url_image: `/api/pages/${id}/image`,
    content: 'Description de la page', context: 'Tome 6 - Chap. 51 - Page 18', similarity,
});

function setup(overrides = {}) {
    const props = {
        view: 'results', mode: getSearchMode('keyword'), mangaSlug: 'one-piece', results: [], totalCount: 0, term: 'luffy',
        topOcrResult: null, isLoadingMore: false, isStale: false, hasMore: false, onLoadMore: vi.fn(), feedback: null,
        error: null, onRetry: vi.fn(), onTrySemantic: vi.fn(), examples: [], onExample: vi.fn(),
        scan: { status: '', progress: null, bubbleCount: 0 },
        ...overrides,
    };
    return { ...render(<SearchResults {...props} />), props };
}

describe('SearchResults', () => {
    it('stays out of the way before any search, apart from the example scenes it is given', () => {
        const { container, unmount } = setup({ view: 'idle' });
        expect(container).toBeEmptyDOMElement();
        unmount();

        const { props } = setup({ view: 'idle', mode: getSearchMode('semantic'), examples: ['le combat entre Zoro et Mihawk'] });
        fireEvent.click(screen.getByRole('button', { name: 'le combat entre Zoro et Mihawk' }));
        expect(props.onExample).toHaveBeenCalledWith('le combat entre Zoro et Mihawk');
    });

    it('names the term that found nothing and proposes the semantic search from keyword mode only', () => {
        const { props, unmount } = setup({ view: 'empty', term: 'zzqx' });
        expect(screen.getByText('Zone inconnue')).toBeInTheDocument();
        expect(screen.getByText(/Aucune occurrence de «\s*zzqx\s*»/)).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Rechercher par concept' }));
        expect(props.onTrySemantic).toHaveBeenCalledTimes(1);
        unmount();

        setup({ view: 'empty', mode: getSearchMode('semantic'), term: 'zzqx' });
        expect(screen.queryByRole('button', { name: 'Rechercher par concept' })).not.toBeInTheDocument();
    });

    it('reports a failed search as an alert that can be retried', () => {
        const { props } = setup({ view: 'error', error: 'Erreur moteur de recherche' });
        expect(screen.getByRole('alert')).toHaveTextContent('La recherche n’a pas abouti');
        expect(screen.getByRole('alert')).toHaveTextContent('Erreur moteur de recherche');
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));
        expect(props.onRetry).toHaveBeenCalledTimes(1);
    });

    it('shows a loading placeholder instead of results while a text search runs', () => {
        const { container } = setup({ view: 'loading' });
        expect(screen.getByRole('status', { name: 'Recherche en cours' })).toBeInTheDocument();
        expect(container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(8);
        expect(screen.queryByRole('link')).not.toBeInTheDocument();
    });

    it('describes the progress of an image scan', () => {
        setup({ view: 'loading', mode: getSearchMode('ocr'), scan: { status: 'Lecture des bulles · 3 / 12', progress: 0.25, bubbleCount: 0 } });
        expect(screen.getByRole('status')).toHaveTextContent('Lecture de votre image');
        expect(screen.getByRole('status')).toHaveTextContent('Lecture des bulles · 3 / 12');
    });

    it('lists bubbles with their place, a link to the page and the searched words highlighted', () => {
        setup({ results: [bubble(1, 'Hé, Luffy, où vas-tu ?', 3), bubble(2, '[Concept] luffy (rit)', 4)], totalCount: 130 });
        expect(screen.getByRole('status')).toHaveTextContent(/130\s*bulles pour «\s*luffy\s*»/);
        expect(screen.getByText('2 affichées')).toBeInTheDocument();

        const links = screen.getAllByRole('link');
        expect(links.map(link => link.getAttribute('href'))).toEqual(['/one-piece/annotate/101?from=search', '/one-piece/annotate/102?from=search']);
        expect(within(links[0]).getByText('Luffy').tagName).toBe('MARK');
        expect(within(links[0]).getByText('p. 3')).toBeInTheDocument();
        expect(links[1]).not.toHaveTextContent('[Concept]');
    });

    it('treats the searched term as plain text, whatever characters it contains', () => {
        setup({ results: [bubble(1, 'Quoi ?! (rires)')], totalCount: 1, term: '?! (' });
        expect(screen.getByText('?! (').tagName).toBe('MARK');
        expect(screen.getByRole('status')).toHaveTextContent(/1\s*bulle pour/);
    });

    it('shows semantic pages with their score and collects one relevance vote per page', () => {
        const onVote = vi.fn();
        const results = [semanticPage(1371, 0.921875), semanticPage(1361, 0.6)];
        setup({ mode: getSearchMode('semantic'), results, totalCount: 2, term: 'zoro contre mihawk', feedback: { given: { 'page-1361': true }, onVote } });

        const [first, second] = screen.getAllByRole('listitem');
        expect(within(first).getByRole('link')).toHaveAttribute('href', '/one-piece/annotate/1371?from=search');
        expect(within(first).getByRole('img')).toHaveAttribute('data-thumbnail', 'true');
        expect(within(first).getByText('92 %')).toBeInTheDocument();
        fireEvent.click(within(first).getByRole('button', { name: 'Résultat pertinent' }));
        expect(onVote).toHaveBeenLastCalledWith(expect.anything(), results[0], true);
        fireEvent.click(within(first).getByRole('button', { name: 'Résultat non pertinent' }));
        expect(onVote).toHaveBeenLastCalledWith(expect.anything(), results[0], false);

        expect(within(second).getByText('Merci pour votre retour')).toBeInTheDocument();
        expect(within(second).queryByRole('button')).not.toBeInTheDocument();
    });

    it('loads the next page on demand and replaces the button with placeholders meanwhile', () => {
        const results = [bubble(1, 'Luffy !'), bubble(2, 'Luffy ?')];
        const { props, rerender, container } = setup({ results, totalCount: 59, hasMore: true });
        fireEvent.click(screen.getByRole('button', { name: /Déchiffrer la suite/ }));
        expect(props.onLoadMore).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: /Déchiffrer la suite/ })).toHaveTextContent('2 / 59');

        rerender(<SearchResults {...props} isLoadingMore />);
        expect(screen.queryByRole('button', { name: /Déchiffrer la suite/ })).not.toBeInTheDocument();
        expect(container.querySelectorAll('[data-slot="skeleton"]')).toHaveLength(8);
        expect(screen.getAllByRole('link')).toHaveLength(2);
    });

    it('presents a confident image match as the page found, with the bubbles it recognised', () => {
        const top = {
            type: 'ocr', id: 'page-278', page_id: 278, url_image: '/api/pages/278/image', context: 'Tome 1 - Chap. 1 - Page 10', similarity: 0.94,
            ocr: { matches: [
                { query_index: 0, query_text: 'LUFFY JE TE PREPARE', matched_text: 'Luffy, je te prépare quelque chose à manger ?', bubble_id: 7, score: 0.97, coords: { x: 1, y: 2, w: 3, h: 4 } },
                { query_index: 1, query_text: '', matched_text: 'Hi, hi, je compte sur toi.', bubble_id: 8, score: 0.7, coords: null },
            ] },
        };
        setup({ mode: getSearchMode('ocr'), results: [top], totalCount: 1, topOcrResult: top });

        expect(screen.getByText('Page retrouvée')).toBeInTheDocument();
        expect(screen.getByRole('heading', { name: 'Tome 1' })).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /Ouvrir la page$/ })).toHaveAttribute('href', '/one-piece/annotate/278?from=search');
        const bubbles = within(screen.getByRole('list', { name: 'Bulles reconnues' })).getAllByRole('listitem');
        expect(bubbles).toHaveLength(2);
        expect(bubbles[0]).toHaveTextContent('Luffy, je te prépare quelque chose à manger ?');
        expect(bubbles[0]).toHaveTextContent('Lu dans votre image');
        expect(bubbles[0]).toHaveTextContent('97 %');
        expect(bubbles[1]).not.toHaveTextContent('Lu dans votre image');
    });
});
