import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAuth } from '@/context/AuthContext';
import { useManga } from '@/context/MangaContext';
import { getMySubmissions } from '@/lib/api';
import SubmissionsList from './SubmissionsList';

vi.mock('@/context/AuthContext', () => ({ useAuth: vi.fn() }));
vi.mock('@/context/MangaContext', () => ({ useManga: vi.fn() }));
vi.mock('@/lib/api', () => ({ getMySubmissions: vi.fn() }));

const submission = {
    id: 1, texte_propose: 'Je deviendrai le roi des pirates !', statut: 'Proposé', created_at: '2026-10-01T12:00:00Z',
    pages: { id: 42, numero_page: 3, chapitres: { numero: 1, tomes: { numero: 1, mangas: { titre: 'One Piece', slug: 'one-piece' } } } },
};
const response = (results = [submission], totalCount = results.length) => ({ data: { results, totalCount } });
function deferred() {
    let resolve;
    const promise = new Promise(r => { resolve = r; });
    return { promise, resolve };
}

describe('SubmissionsList', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        useAuth.mockReturnValue({ session: { user: { id: 'author' } } });
        useManga.mockReturnValue({ mangaSlug: 'one-piece' });
        getMySubmissions.mockResolvedValue(response());
    });

    it('loads the current manga history with complete text, date, status and page links', async () => {
        render(<SubmissionsList />);
        expect(screen.getByRole('status', { name: 'Chargement des soumissions' })).toBeInTheDocument();
        expect(await screen.findByText(submission.texte_propose)).toBeInTheDocument();
        expect(getMySubmissions).toHaveBeenCalledWith(1, 15, 'one-piece');
        expect(screen.getByText('En attente')).toBeInTheDocument();
        expect(screen.getByText('Tome 1 · Chapitre 1 · Page 3')).toBeInTheDocument();
        expect(screen.getByRole('link', { name: 'Voir la page' })).toHaveAttribute('href', '/one-piece/annotate/42');
        expect(screen.getByText(/Soumis le/).tagName).toBe('TIME');
    });

    it('distinguishes corrections from other moderation comments and handles incomplete metadata', async () => {
        getMySubmissions.mockResolvedValue(response([
            { ...submission, statut: 'Rejeté', commentaire_moderation: 'Corriger la ponctuation.' },
            { ...submission, id: 2, statut: 'Validé', commentaire_moderation: 'Merci pour la correction.' },
            { id: 3, texte_propose: 'Métadonnées absentes', statut: 'other', created_at: 'invalid' },
        ]));
        render(<SubmissionsList />);
        expect(await screen.findByText('Rejeté')).toBeInTheDocument();
        expect(screen.getByText('Corrections demandées')).toBeInTheDocument();
        expect(screen.getByText('Retour de modération')).toBeInTheDocument();
        expect(screen.getByRole('link', { name: 'Reprendre la page' })).toHaveAttribute('href', '/one-piece/annotate/42');
        expect(screen.getByText('Statut indisponible')).toBeInTheDocument();
        expect(screen.getByText('Date indisponible')).toBeInTheDocument();
        expect(screen.getAllByRole('link')).toHaveLength(2);
    });

    it('shows an explicit error and retries successfully without presenting failure as an empty history', async () => {
        getMySubmissions.mockRejectedValueOnce(new Error('network'));
        render(<SubmissionsList />);
        expect(await screen.findByRole('alert')).toHaveTextContent('Impossible de charger vos soumissions');
        expect(screen.queryByText('Aucune soumission pour le moment')).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Réessayer' }));
        expect(await screen.findByText(submission.texte_propose)).toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(getMySubmissions).toHaveBeenCalledTimes(2);
    });

    it('provides a library link for an empty history', async () => {
        getMySubmissions.mockResolvedValue(response([]));
        render(<SubmissionsList />);
        expect(await screen.findByText('Aucune soumission pour le moment')).toBeInTheDocument();
        expect(screen.getByRole('link', { name: 'Explorer la bibliothèque' })).toHaveAttribute('href', '/one-piece/dashboard');
        expect(screen.queryByRole('navigation', { name: 'Pagination des soumissions' })).not.toBeInTheDocument();
    });

    it('locks pagination during requests and refreshes the current page', async () => {
        const next = deferred();
        getMySubmissions.mockResolvedValueOnce(response([submission], 16)).mockReturnValueOnce(next.promise)
            .mockResolvedValueOnce(response([{ ...submission, id: 2, texte_propose: 'Seconde page actualisée' }], 16));
        render(<SubmissionsList />);
        await screen.findByText(submission.texte_propose);
        expect(screen.getByRole('button', { name: 'Page précédente' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        expect(screen.getByRole('button', { name: 'Page suivante' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Page précédente' })).toBeDisabled();
        await act(async () => next.resolve(response([{ ...submission, id: 2, texte_propose: 'Seconde page' }], 16)));
        expect(screen.getByText('Seconde page')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Page suivante' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Actualiser mes soumissions' }));
        expect(await screen.findByText('Seconde page actualisée')).toBeInTheDocument();
        expect(getMySubmissions).toHaveBeenLastCalledWith(2, 15, 'one-piece');
    });

    it('returns to the last available page when the refreshed history shrinks', async () => {
        getMySubmissions.mockResolvedValueOnce(response([submission], 16)).mockResolvedValueOnce(response([], 1));
        render(<SubmissionsList />);
        await screen.findByText(submission.texte_propose);
        fireEvent.click(screen.getByRole('button', { name: 'Page suivante' }));
        await waitFor(() => expect(getMySubmissions).toHaveBeenCalledTimes(3));
        expect(getMySubmissions).toHaveBeenLastCalledWith(1, 15, 'one-piece');
        expect(await screen.findByText(submission.texte_propose)).toBeInTheDocument();
    });

    it('resets history when the manga changes and ignores late responses from the previous manga', async () => {
        const previous = deferred();
        getMySubmissions.mockReturnValueOnce(previous.promise).mockResolvedValueOnce(response([{ ...submission, texte_propose: 'Nouveau manga' }]));
        const { rerender } = render(<SubmissionsList />);
        useManga.mockReturnValue({ mangaSlug: 'new-manga' });
        rerender(<SubmissionsList />);
        expect(await screen.findByText('Nouveau manga')).toBeInTheDocument();
        await act(async () => previous.resolve(response()));
        expect(screen.queryByText(submission.texte_propose)).not.toBeInTheDocument();
        expect(getMySubmissions).toHaveBeenLastCalledWith(1, 15, 'new-manga');
    });

    it('does not request history before the session is available', () => {
        useAuth.mockReturnValue({ session: null });
        render(<SubmissionsList />);
        expect(getMySubmissions).not.toHaveBeenCalled();
    });
});
