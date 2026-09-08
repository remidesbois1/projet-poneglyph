import React from 'react';
import { render, screen, within, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import BubbleReviewList from '@/components/BubbleReviewList';
import PageReviewList from '@/components/PageReviewList';
import { getPendingBubbles, getPagesForReview, validateAllBubbles, approveAllPages } from '@/lib/api';
vi.mock('@/context/AuthContext', () => ({ useAuth: () => ({ session: { access_token: 'test' } }) }));
vi.mock('@/context/MangaContext', () => ({ useManga: () => ({ mangaSlug: 'one-piece' }) }));
vi.mock('@/hooks/useUserProfile', () => ({ useUserProfile: () => ({ profile: { role: 'Admin' } }) }));
vi.mock('@/lib/api', () => ({ getPendingBubbles: vi.fn(), getPagesForReview: vi.fn(), validateAllBubbles: vi.fn(), approveAllPages: vi.fn(), validateBubble: vi.fn(), rejectBubble: vi.fn() }));
vi.mock('@/components/BubbleReviewItem', () => ({ default: () => <div>Bulle</div> }));
vi.mock('@/components/moderation/ReviewPageThumbnail', () => ({ default: () => null }));
vi.mock('@/components/ValidationForm', () => ({ default: () => null }));
beforeEach(() => {
 vi.clearAllMocks();
 getPendingBubbles.mockResolvedValue({ data: { results: [{ id: 1 }], totalCount: 12 } });
 getPagesForReview.mockResolvedValue({ data: [{ id: 1, numero_page: 1 }, { id: 2, numero_page: 2 }] });
});
describe.each([
 ['bulles', BubbleReviewList, validateAllBubbles, 12],
 ['pages', PageReviewList, approveAllPages, 2],
])('bulk confirmation for %s', (resource, List, approve, count) => {
 it('opens the shared modal with the total and cancels without approving', async () => {
  render(<List />);
  fireEvent.click(await screen.findByRole('button', { name: 'Tout valider' }));
  const dialog = screen.getByRole('dialog');
  expect(dialog).toHaveAccessibleName('Valider toutes les ' + resource + ' ?');
  expect(dialog).toHaveTextContent(count + ' ' + resource + ' en attente seront validées.');
  expect(approve).not.toHaveBeenCalled();
  fireEvent.click(within(dialog).getByRole('button', { name: 'Annuler' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(approve).not.toHaveBeenCalled();
 });
 it('waits for success and blocks duplicate submissions', async () => {
  let finish;
  approve.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  render(<List />);
  fireEvent.click(await screen.findByRole('button', { name: 'Tout valider' }));
  const dialog = screen.getByRole('dialog');
  const submit = within(dialog).getByRole('button', { name: 'Tout valider' });
  fireEvent.click(submit); fireEvent.click(submit);
  expect(approve).toHaveBeenCalledOnce();
  expect(within(dialog).getByRole('button', { name: 'Validation en cours…' })).toBeDisabled();
  expect(within(dialog).getByRole('button', { name: 'Annuler' })).toBeDisabled();
  await act(async () => finish());
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
 });
 it('keeps API failures inside the modal and allows retrying', async () => {
  approve.mockRejectedValueOnce({ response: { data: { error: 'Validation indisponible.' } } }).mockResolvedValue({});
  render(<List />);
  fireEvent.click(await screen.findByRole('button', { name: 'Tout valider' }));
  const dialog = screen.getByRole('dialog');
  fireEvent.click(within(dialog).getByRole('button', { name: 'Tout valider' }));
  expect(await within(dialog).findByRole('alert')).toHaveTextContent('Validation indisponible.');
  fireEvent.click(within(dialog).getByRole('button', { name: 'Tout valider' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(approve).toHaveBeenCalledTimes(2);
 });
});
