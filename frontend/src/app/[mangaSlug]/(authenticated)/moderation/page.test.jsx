import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useRouter, useSearchParams } from 'next/navigation';
import ModerationPage from './page';

vi.mock('next/navigation', () => ({ useSearchParams: vi.fn(), useRouter: vi.fn() }));
vi.mock('@/context/MangaContext', () => ({ useManga: () => ({ mangaSlug: 'one-piece', currentManga: { titre: 'One Piece' } }) }));
vi.mock('@/components/BubbleReviewList', () => ({ default: () => <div>Liste des bulles</div> }));
vi.mock('@/components/PageReviewList', () => ({ default: () => <div>Liste des pages</div> }));
vi.mock('@/components/moderation/SubmissionsList', () => ({ default: () => <div>Historique personnel</div> }));

describe('Moderation tabs', () => {
    const replace = vi.fn();
    beforeEach(() => {
        vi.clearAllMocks();
        useRouter.mockReturnValue({ replace });
        useSearchParams.mockReturnValue(new URLSearchParams());
    });

    it('opens the submissions tab directly and keeps the shared moderation heading', async () => {
        useSearchParams.mockReturnValue(new URLSearchParams('view=submissions'));
        render(<ModerationPage />);
        expect(await screen.findByText('Historique personnel')).toBeInTheDocument();
        expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Modération');
        expect(screen.getByRole('tab', { name: 'Mes Soumissions' })).toHaveAttribute('aria-selected', 'true');
        expect(screen.getAllByRole('tab')).toHaveLength(3);
        fireEvent.mouseDown(screen.getByRole('tab', { name: 'Pages complètes' }), { button: 0, ctrlKey: false });
        expect(replace).toHaveBeenCalledWith('/one-piece/moderation?view=pages', { scroll: false });
        fireEvent.mouseDown(screen.getByRole('tab', { name: 'Bulles' }), { button: 0, ctrlKey: false });
        expect(replace).toHaveBeenCalledWith('/one-piece/moderation', { scroll: false });
    });

    it('navigates to submissions from the default tab without loading the other lists', async () => {
        render(<ModerationPage />);
        expect(await screen.findByText('Liste des bulles')).toBeInTheDocument();
        expect(screen.queryByText('Historique personnel')).not.toBeInTheDocument();
        fireEvent.mouseDown(screen.getByRole('tab', { name: 'Mes Soumissions' }), { button: 0, ctrlKey: false });
        expect(replace).toHaveBeenCalledWith('/one-piece/moderation?view=submissions', { scroll: false });
    });
});
