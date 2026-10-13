import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
    searchBubbles: vi.fn(),
    searchF2llmLocal: vi.fn(),
    searchOcrPageMatch: vi.fn(),
    getMetadataSuggestions: vi.fn(),
    getTomes: vi.fn(),
    submitSearchFeedback: vi.fn(),
}));
vi.mock('@/lib/api', () => api);
vi.mock('@/lib/f2llmBrowserEmbedding', () => ({ generateF2llmBrowserQueryEmbedding: vi.fn() }));
vi.mock('@/context/DetectionContext', () => ({ useDetection: () => ({ detectBubbles: vi.fn(), detectionStatus: 'idle', loadDetectionModel: vi.fn() }) }));
vi.mock('@/context/WorkerContext', () => ({ useWorker: () => ({ worker: null, modelStatus: 'idle', activeModelKey: null }) }));
const manga = vi.hoisted(() => ({ current: null }));
vi.mock('@/context/MangaContext', () => ({ useManga: () => manga.current }));
vi.mock('sonner', () => ({ toast: { error: vi.fn(), info: vi.fn(), success: vi.fn() } }));
vi.mock('@/components/SearchPageImage', () => ({
    // eslint-disable-next-line @next/next/no-img-element
    default: ({ alt }) => <img alt={alt} />,
}));
import SearchClient from './SearchClient';

const found = (results, totalCount = results.length) => ({ data: { results, totalCount } });
const bubble = { type: 'bubble', id: 99, page_id: 271, url_image: '/api/pages/271/image', coords: { x: 1, y: 2, w: 30, h: 40 }, content: 'Hé, Luffy !', context: 'Tome 1 - Chap. 1 - Page 3' };
const field = () => screen.getByRole('textbox');
const type = value => fireEvent.change(field(), { target: { value } });
const pause = ms => act(() => new Promise(resolve => setTimeout(resolve, ms)));

describe('search page states', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        sessionStorage.clear();
        manga.current = { mangaSlug: 'one-piece' };
        // jsdom has no matchMedia; asking for reduced motion also keeps the decorative glyph timers out of the tests.
        window.matchMedia = vi.fn(() => ({ matches: true }));
        api.getMetadataSuggestions.mockResolvedValue({ data: { characters: [], arcs: [] } });
        api.getTomes.mockResolvedValue({ data: [] });
        api.searchBubbles.mockResolvedValue(found([]));
    });

    it('opens on the title, the three modes and an empty field, with nothing announced yet', () => {
        render(<SearchClient />);
        expect(screen.getByRole('heading', { level: 1, name: 'Voix de Toute Chose' })).toHaveClass('poneglyph-title');
        expect(screen.getAllByRole('radio').map(radio => radio.textContent)).toEqual(['Textuel', 'Sémantique', 'Visuelle']);
        expect(screen.getByRole('radio', { name: 'Textuel' })).toBeChecked();
        expect(field()).toHaveValue('');
        expect(screen.queryByText('Zone inconnue')).not.toBeInTheDocument();
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('shows a keyword search as running from the first keystroke, and the empty archive only once it has answered', async () => {
        let answer;
        api.searchBubbles.mockReturnValueOnce(new Promise(resolve => { answer = resolve; }));
        render(<SearchClient />);

        type('zz');
        expect(screen.getByRole('status', { name: 'Recherche en cours' })).toBeInTheDocument();
        expect(screen.queryByText('Zone inconnue')).not.toBeInTheDocument();

        await waitFor(() => expect(api.searchBubbles).toHaveBeenCalledTimes(1));
        expect(api.searchBubbles.mock.calls[0].slice(0, 4)).toEqual(['zz', 1, 24, 'keyword']);
        expect(screen.queryByText('Zone inconnue')).not.toBeInTheDocument();

        await act(async () => answer(found([])));
        expect(screen.getByText('Zone inconnue')).toBeInTheDocument();
        expect(screen.getByText(/Aucune occurrence de «\s*zz\s*»/)).toBeInTheDocument();
    });

    it('labels results with the term they answer and dims them while the field moves on', async () => {
        api.searchBubbles.mockResolvedValueOnce(found([bubble], 130));
        render(<SearchClient initialQuery="luffy" />);

        expect(await screen.findByRole('link')).toHaveAttribute('href', '/one-piece/annotate/271?from=search');
        expect(screen.getByRole('status')).toHaveTextContent(/130\s*bulles pour «\s*luffy\s*»/);
        expect(screen.getByRole('region', { name: 'Résultats' })).toHaveAttribute('aria-busy', 'false');

        type('luffy et zoro');
        expect(screen.getByRole('status')).toHaveTextContent(/bulles pour «\s*luffy\s*»/);
        expect(screen.getByRole('region', { name: 'Résultats' })).toHaveAttribute('aria-busy', 'true');
    });

    it('still answers a search from the URL when the manga is only known after the first render', async () => {
        // On a hard load the manga context resolves its slug in an effect, one render after the page has mounted.
        manga.current = { mangaSlug: null };
        api.searchBubbles.mockResolvedValue(found([bubble]));
        const { rerender } = render(<SearchClient initialQuery="luffy" />);
        expect(screen.getByRole('status', { name: 'Recherche en cours' })).toBeInTheDocument();

        manga.current = { mangaSlug: 'one-piece' };
        rerender(<SearchClient initialQuery="luffy" />);
        expect(await screen.findByRole('link')).toHaveAttribute('href', '/one-piece/annotate/271?from=search');
        expect(screen.queryByRole('status', { name: 'Recherche en cours' })).not.toBeInTheDocument();
    });

    it('waits for Enter in semantic mode instead of announcing an empty archive while typing', async () => {
        render(<SearchClient />);
        fireEvent.click(screen.getByRole('radio', { name: 'Sémantique' }));
        type('zoro contre mihawk');
        await pause(500);

        expect(api.searchBubbles).not.toHaveBeenCalled();
        expect(screen.queryByText('Zone inconnue')).not.toBeInTheDocument();
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'le combat entre Zoro et Mihawk' })).toBeInTheDocument();

        fireEvent.keyDown(field(), { key: 'Enter' });
        await waitFor(() => expect(api.searchBubbles).toHaveBeenCalledTimes(1));
        expect(api.searchBubbles.mock.calls[0].slice(0, 4)).toEqual(['zoro contre mihawk', 1, 24, 'semantic']);
        expect(await screen.findByText('Zone inconnue')).toBeInTheDocument();
    });

    it('reports a failed search rather than an empty archive, and retries it', async () => {
        api.searchBubbles.mockRejectedValueOnce({ response: { data: { error: 'Erreur moteur de recherche' } } });
        vi.spyOn(console, 'error').mockImplementation(() => {});
        render(<SearchClient initialQuery="luffy" />);

        expect(await screen.findByRole('alert')).toHaveTextContent('Erreur moteur de recherche');
        expect(screen.queryByText('Zone inconnue')).not.toBeInTheDocument();

        api.searchBubbles.mockResolvedValueOnce(found([bubble]));
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));
        expect(await screen.findByRole('link')).toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('runs the same query in semantic mode from the empty state', async () => {
        render(<SearchClient initialQuery="rencontre avec zoro" />);
        fireEvent.click(await screen.findByRole('button', { name: 'Rechercher par concept' }));

        await waitFor(() => expect(api.searchBubbles).toHaveBeenCalledTimes(2));
        expect(api.searchBubbles.mock.calls[1].slice(0, 4)).toEqual(['rencontre avec zoro', 1, 24, 'semantic']);
        expect(screen.getByRole('radio', { name: 'Sémantique' })).toBeChecked();
    });

    it('searches an example scene as soon as it is picked', async () => {
        render(<SearchClient />);
        fireEvent.click(screen.getByRole('radio', { name: 'Sémantique' }));
        fireEvent.click(screen.getByRole('button', { name: 'le combat entre Zoro et Mihawk' }));

        expect(field()).toHaveValue('le combat entre Zoro et Mihawk');
        await waitFor(() => expect(api.searchBubbles).toHaveBeenCalledTimes(1));
        expect(api.searchBubbles.mock.calls[0].slice(0, 4)).toEqual(['le combat entre Zoro et Mihawk', 1, 24, 'semantic']);
    });

    it('filters by tome from the filters panel and shows the active filter as a removable chip', async () => {
        sessionStorage.setItem('search_state_one-piece', JSON.stringify({ query: 'luffy', searchMode: 'keyword', selectedTome: '3', results: [], totalCount: 0 }));
        render(<SearchClient />);

        await waitFor(() => expect(api.searchBubbles).toHaveBeenCalledTimes(1));
        expect(api.searchBubbles.mock.calls[0][4]).toEqual({ characters: [], arc: '', tome: '3' });

        fireEvent.click(screen.getByRole('button', { name: 'Retirer le filtre tome 3' }));
        await waitFor(() => expect(api.searchBubbles).toHaveBeenCalledTimes(2));
        expect(api.searchBubbles.mock.calls[1][4]).toEqual({ characters: [], arc: '', tome: '' });
        expect(screen.queryByRole('button', { name: 'Retirer le filtre tome 3' })).not.toBeInTheDocument();
    });
});
