import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
    stats: vi.fn(), models: vi.fn(), describe: vi.fn(), gemini: vi.fn(), voyage: vi.fn(), save: vi.fn(),
    original: vi.fn(), session: vi.fn(),
}));
vi.mock('@/lib/api', () => ({
    getAiModels: mocks.models, getEmbeddingStats: mocks.stats, savePageData: mocks.save, generateVoyageEmbedding: mocks.voyage,
    updateAiModels: vi.fn(), triggerGeminiBackfill: vi.fn(), triggerVoyageBackfill: vi.fn(),
}));
vi.mock('@/lib/geminiClient', () => ({ generatePageDescription: mocks.describe, generateGeminiEmbedding: mocks.gemini }));
vi.mock('@/lib/pageImageClient', () => ({ fetchOriginalPageImage: mocks.original }));
vi.mock('@/lib/astraSemanticRebuild', () => ({ requireSupabaseSession: mocks.session, getRebuildFailure: error => ({ message: error.message }) }));
vi.mock('@/lib/aiModelConfig', () => ({ cacheAiModelConfig: vi.fn() }));
vi.mock('@/components/AstraSemanticRebuild', () => ({ default: ({ onComplete }) => <button onClick={onComplete}>Astra rebuild</button> }));
vi.mock('@/components/ModelBenchmarkRegistry', () => ({ default: () => null }));
vi.mock('sonner', () => ({ toast: { info: vi.fn(), error: vi.fn(), success: vi.fn() } }));

import AiModelManager from './AiModelManager';
import { toast } from 'sonner';

describe('legacy Gemini client batch', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.setItem('google_api_key', 'test-google-key');
        mocks.models.mockResolvedValue({ data: { model_ocr: 'gemini-2.5-flash', model_description: 'gemini-2.5-flash', model_chatgpt_ocr: 'gpt-5.6-luna', gemini_thinking_level: 'default', chatgpt_reasoning_effort: 'low', chatgpt_fast_mode: false } });
        mocks.stats.mockResolvedValue({ data: [{ id: 12, url_image: '/api/pages/12/image', has_description: false, has_voyage: false, has_gemini: false }] });
        mocks.session.mockResolvedValue({ access_token: 'supabase-token' });
        mocks.describe.mockResolvedValue({ data: { content: 'Action visible', metadata: { arc: '', characters: [] } } });
        mocks.voyage.mockResolvedValue({ data: { embedding: [0.1] } });
        mocks.gemini.mockResolvedValue([0.2]);
        mocks.save.mockResolvedValue({ data: { success: true } });
    });

    it('sends the authenticated original blob to description and Gemini embedding generation', async () => {
        const blob = new Blob(['original'], { type: 'image/png' });
        mocks.original.mockResolvedValue(blob);
        render(<AiModelManager mangaSlug="one-piece" />);
        fireEvent.click(await screen.findByRole('button', { name: 'Lancer' }));
        await waitFor(() => expect(mocks.save).toHaveBeenCalledOnce());
        expect(mocks.original).toHaveBeenCalledWith(12, 'supabase-token');
        expect(mocks.describe).toHaveBeenCalledWith(blob, 'test-google-key');
        expect(mocks.gemini).toHaveBeenCalledWith('Action visible', blob, 'test-google-key');
        await waitFor(() => expect(screen.getByRole('button', { name: 'Lancer' })).toBeEnabled(), { timeout: 3000 });
    });

    it('keeps known page stats when completion refresh fails and offers a manual retry', async () => {
        render(<AiModelManager mangaSlug="one-piece" />);
        await screen.findByText('Total : 1 pages');
        mocks.stats.mockRejectedValueOnce(new Error('Supabase temporairement indisponible.'));
        fireEvent.click(screen.getByRole('button', { name: 'Astra rebuild' }));
        await screen.findByRole('alert');
        expect(screen.getByText('Total : 1 pages')).toBeInTheDocument();
        expect(toast.error).toHaveBeenCalledWith('Impossible d’actualiser les statistiques.', { description: 'Supabase temporairement indisponible.' });
        mocks.stats.mockResolvedValueOnce({ data: [{ id: 12 }, { id: 13 }] });
        fireEvent.click(screen.getByRole('button', { name: 'Actualiser les statistiques' }));
        await screen.findByText('Total : 2 pages');
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('does not present a failed initial stats request as an empty manga', async () => {
        mocks.stats.mockRejectedValueOnce(new Error('Session Supabase expirée.'));
        render(<AiModelManager mangaSlug="one-piece" />);
        expect(await screen.findByRole('alert')).toHaveTextContent('Session Supabase expirée.');
        expect(screen.queryByText("Aucune donnée d'embedding trouvée.")).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Actualiser les statistiques' }));
        await screen.findByText('Total : 1 pages');
    });
});
