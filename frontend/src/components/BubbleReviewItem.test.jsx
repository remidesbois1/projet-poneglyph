import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import BubbleReviewItem from './BubbleReviewItem';

vi.mock('@/context/AuthContext', () => ({ useAuth: () => ({ session: null }) }));
vi.mock('@/lib/api', () => ({ getBubbleCrop: vi.fn(), getBubbleHistory: vi.fn() }));

const bubble = { id: 12, texte_propose: 'Je deviendrai le roi des pirates !' };

describe('Bubble review actions', () => {
    it('keeps the contribution available after opening or cancelling a rejection', async () => {
        const onAction = vi.fn().mockResolvedValue(undefined);
        const onEdit = vi.fn();
        render(<BubbleReviewItem bubble={bubble} onAction={onAction} onEdit={onEdit} />);
        fireEvent.click(screen.getByRole('button', { name: 'Refuser' }));
        expect(onAction).toHaveBeenCalledWith('reject', 12);
        await waitFor(() => expect(screen.getByRole('button', { name: 'Corriger' })).toBeEnabled());
        expect(screen.getByText(bubble.texte_propose)).toBeVisible();
        fireEvent.click(screen.getByRole('button', { name: 'Corriger' }));
        expect(onEdit).toHaveBeenCalledWith(bubble);
    });

    it('prevents duplicate decisions while validation is pending', async () => {
        let resolve;
        const onAction = vi.fn(() => new Promise(done => { resolve = done; }));
        render(<BubbleReviewItem bubble={bubble} onAction={onAction} onEdit={vi.fn()} />);
        fireEvent.click(screen.getByRole('button', { name: 'Valider' }));
        expect(screen.getByRole('button', { name: 'Valider' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Refuser' })).toBeDisabled();
        fireEvent.click(screen.getByRole('button', { name: 'Valider' }));
        expect(onAction).toHaveBeenCalledTimes(1);
        await act(async () => resolve());
        expect(screen.getByRole('button', { name: 'Valider' })).toBeEnabled();
    });
});
