import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    useAuth: vi.fn(), status: vi.fn(), session: vi.fn(), stats: vi.fn(), ready: vi.fn(), run: vi.fn(),
    toast: { info: vi.fn(), error: vi.fn(), success: vi.fn() },
}));
vi.mock('@/context/AuthContext', () => ({ useAuth: mocks.useAuth }));
vi.mock('@/lib/api', () => ({ getEmbeddingStats: mocks.stats, getSemanticRebuildStatus: mocks.ready }));
vi.mock('@/lib/chatGptDesktop', () => ({ getChatGptStatus: mocks.status, subscribeToChatGptAuth: () => () => {} }));
vi.mock('@/lib/astraSemanticRebuild', async (importOriginal) => ({
    ...await importOriginal(), requireSupabaseSession: mocks.session, runAstraSemanticRebuild: mocks.run,
}));
vi.mock('@/lib/promptConfig', () => ({ getPrompt: vi.fn().mockResolvedValue('prompt'), invalidatePromptCache: vi.fn() }));
vi.mock('sonner', () => ({ toast: mocks.toast }));

import AstraSemanticRebuild from './AstraSemanticRebuild';

const page = { id: 1, description: { content: 'Action', metadata: { arc: '', characters: [] } }, has_description: true, has_voyage: true, has_gemini: true, has_f2llm: false };
const currentPage = { ...page, description_model: 'gpt-6-astra', description_prompt_version: 2 };
const buttonName = 'Refaire descriptions + embeddings avec Astra';

describe('Astra rebuild controls', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.useAuth.mockReturnValue({ role: 'Admin' });
        mocks.status.mockResolvedValue({ available: true, connected: true });
        mocks.session.mockResolvedValue({ user: { id: 'admin' }, access_token: 'token' });
        mocks.stats.mockResolvedValue({ data: [page] });
        mocks.ready.mockResolvedValue({ data: { ready: true } });
        mocks.run.mockResolvedValue({ status: 'completed', saved: 1, errors: [] });
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    it('is restricted to admins', () => {
        mocks.useAuth.mockReturnValue({ role: 'User' });
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        expect(screen.queryByRole('button', { name: buttonName })).not.toBeInTheDocument();
    });

    it.each([
        [{ available: false, connected: false }, 'Poneglyph Desktop requis'],
        [{ available: true, connected: false }, 'déconnecté'],
    ])('requires Desktop and an active ChatGPT connection', async (status, text) => {
        mocks.status.mockResolvedValue(status);
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(text));
        expect(screen.getByRole('button', { name: buttonName })).toBeDisabled();
    });

    it('checks session/server and confirms that every old page is included before launching', async () => {
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        await waitFor(() => expect(mocks.run).toHaveBeenCalledOnce());
        expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('y compris celles qui ont déjà une description et des embeddings'));
        expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('embeddings Voyage et Gemini'));
        expect(mocks.session).toHaveBeenCalledOnce();
        expect(mocks.ready).toHaveBeenCalledOnce();
        expect(mocks.stats).toHaveBeenCalledWith('one-piece');
        expect(mocks.run).toHaveBeenCalledWith(expect.objectContaining({ pages: [page], mangaSlug: 'one-piece', force: false, prompt: 'prompt\n\nprompt', concurrency: 3 }));
        expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('3 pages à la fois'));
    });

    it('can reduce concurrency to two before starting', async () => {
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        expect(screen.getByRole('combobox', { name: 'Pages simultanées' })).toHaveValue('3');
        fireEvent.change(screen.getByRole('combobox', { name: 'Pages simultanées' }), { target: { value: '2' } });
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        await waitFor(() => expect(mocks.run).toHaveBeenCalledOnce());
        expect(mocks.run).toHaveBeenCalledWith(expect.objectContaining({ concurrency: 2 }));
        expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('2 pages à la fois'));
    });

    it('does not launch after a cancelled confirmation or invalid Supabase session', async () => {
        window.confirm.mockReturnValue(false);
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        await waitFor(() => expect(window.confirm).toHaveBeenCalledOnce());
        expect(mocks.run).not.toHaveBeenCalled();
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        mocks.session.mockRejectedValue(new Error('Session Supabase expirée'));
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        await waitFor(() => expect(mocks.toast.error).toHaveBeenCalledWith('Session Supabase expirée'));
        expect(mocks.run).not.toHaveBeenCalled();
    });

    it('offers resuming by default and a distinct explicitly forced restart', async () => {
        mocks.stats.mockResolvedValue({ data: [currentPage, { ...page, id: 2 }] });
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[currentPage, { ...page, id: 2 }]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        await waitFor(() => expect(mocks.run).toHaveBeenCalledOnce());
        expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Reprendre 1 pages sur 2'));
        await waitFor(() => expect(screen.getByRole('button', { name: 'Tout recommencer, y compris les pages Astra' })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: 'Tout recommencer, y compris les pages Astra' }));
        await waitFor(() => expect(mocks.run).toHaveBeenCalledTimes(2));
        expect(mocks.run.mock.calls[1][0].force).toBe(true);
    });

    it('shows progress and page-specific errors and allows a clean stop', async () => {
        let finish;
        mocks.run.mockImplementation(async (options) => {
            options.onProgress({ total: 3, completed: 1, saved: 1, skipped: 0, concurrency: 3,
                errors: [{ pageId: 2, message: 'Image corrompue' }], log: [{ time: 1, message: 'Traitement' }],
                activePages: [{ page: { id: 2 }, phase: 'Description Astra' }, { page: { id: 3 }, phase: 'Voyage + Gemini' }], status: 'running' });
            return new Promise((resolve) => { finish = resolve; });
        });
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        await screen.findByText('33%');
        expect(screen.getByText('2/3 pages en cours')).toBeInTheDocument();
        expect(screen.getByText(/Page 2 .*Description Astra/)).toBeInTheDocument();
        expect(screen.getByText(/Page 3 .*Voyage \+ Gemini/)).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Pages simultanées' })).toBeDisabled();
        expect(screen.getByText('Page 2 : Image corrompue')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Arrêter' }));
        expect(mocks.run.mock.calls[0][0].shouldStop()).toBe(true);
        expect(screen.getByRole('button', { name: 'Arrêt après les pages en cours…' })).toBeDisabled();
        await act(async () => finish({ status: 'stopped', saved: 1, errors: [] }));
    });

    it('shows a rejected request as a global stop with its diagnostic detail', async () => {
        mocks.run.mockImplementation(async (options) => {
            const result = { total: 1, completed: 0, saved: 0, skipped: 0, currentPage: null, phase: '', status: 'chatgpt_request',
                errors: [{ pageId: 1, message: 'HTTP 400. Instructions are required' }],
                log: [{ time: 1, message: 'HTTP 400. Instructions are required' }] };
            options.onProgress(result);
            return result;
        });
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        expect(await screen.findByText('Requête ChatGPT refusée — consultez le détail de l’erreur')).toBeInTheDocument();
        expect(screen.getByText('Page 1 : HTTP 400. Instructions are required')).toBeInTheDocument();
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
    });

    it('shows personal-key usage without showing the credential itself', async () => {
        mocks.run.mockImplementation(async (options) => {
            const result = { total: 1, completed: 1, saved: 1, skipped: 0, geminiUserKeyPages: 1,
                activePages: [], errors: [], log: [], status: 'completed' };
            options.onProgress(result);
            return result;
        });
        render(<AstraSemanticRebuild mangaSlug="one-piece" pages={[page]} />);
        await waitFor(() => expect(screen.getByRole('button', { name: buttonName })).toBeEnabled());
        fireEvent.click(screen.getByRole('button', { name: buttonName }));
        expect(await screen.findByText('Gemini via votre clé personnelle : 1 pages enregistrées.')).toBeInTheDocument();
    });
});
