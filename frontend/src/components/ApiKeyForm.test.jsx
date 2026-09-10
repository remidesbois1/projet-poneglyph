import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ApiKeyForm from './ApiKeyForm';

vi.mock('@/lib/chatGptDesktop', () => ({
    getChatGptStatus: async () => ({ available: false }),
    subscribeToChatGptAuth: () => () => {}, loginChatGpt: vi.fn(), logoutChatGpt: vi.fn(),
}));

beforeEach(() => { localStorage.clear(); vi.clearAllMocks(); });

describe('API keys per provider', () => {
    it('stores DeepSeek separately without invoking the legacy Gemini callback or a paid call', async () => {
        const googleSave = vi.fn();
        const deepseekSave = vi.fn();
        localStorage.setItem('google_api_key', 'GOOGLE-KEY-UNCHANGED');
        render(<ApiKeyForm onSave={googleSave} onSaveDeepSeek={deepseekSave} />);
        const card = screen.getByRole('region', { name: 'Clé API DeepSeek' });
        const input = within(card).getByLabelText('Clé API DeepSeek');
        expect(input).toHaveAttribute('type', 'password');
        fireEvent.change(input, { target: { value: '  sk-deepseek-secret-value  ' } });
        fireEvent.click(within(card).getByRole('button', { name: 'Enregistrer' }));
        expect(localStorage.getItem('deepseek_api_key')).toBe('sk-deepseek-secret-value');
        expect(localStorage.getItem('google_api_key')).toBe('GOOGLE-KEY-UNCHANGED');
        expect(googleSave).not.toHaveBeenCalled();
        expect(deepseekSave).toHaveBeenCalledWith('sk-deepseek-secret-value');
        expect(card.textContent).not.toContain('sk-deepseek-secret-value');
        expect(await within(card).findByText('Clé enregistrée')).toBeInTheDocument();
    });

    it('retains Gemini save semantics and never overwrites DeepSeek', async () => {
        localStorage.setItem('deepseek_api_key', 'sk-existing-deepseek');
        const onSave = vi.fn();
        const onSaveDeepSeek = vi.fn();
        render(<ApiKeyForm onSave={onSave} onSaveDeepSeek={onSaveDeepSeek} />);
        const card = screen.getByRole('region', { name: 'Clé API Google Gemini' });
        fireEvent.change(within(card).getByLabelText('Clé API Google Gemini'), { target: { value: 'AIza-valid-test' } });
        fireEvent.click(within(card).getByRole('button', { name: 'Enregistrer' }));
        expect(onSave).toHaveBeenCalledWith('AIza-valid-test');
        expect(onSaveDeepSeek).not.toHaveBeenCalled();
        expect(localStorage.getItem('deepseek_api_key')).toBe('sk-existing-deepseek');
        expect(await within(card).findByText('Clé enregistrée')).toBeInTheDocument();
    });

    it('can edit, cancel and delete only the DeepSeek key', async () => {
        localStorage.setItem('deepseek_api_key', 'sk-initial-deepseek');
        localStorage.setItem('google_api_key', 'AIza-original');
        render(<ApiKeyForm />);
        fireEvent.click(screen.getByRole('button', { name: 'Changer la clé DeepSeek' }));
        const card = screen.getByRole('region', { name: 'Clé API DeepSeek' });
        fireEvent.change(within(card).getByLabelText('Clé API DeepSeek'), { target: { value: 'sk-replacement-test' } });
        fireEvent.click(within(card).getByRole('button', { name: 'Annuler' }));
        expect(localStorage.getItem('deepseek_api_key')).toBe('sk-initial-deepseek');
        fireEvent.click(screen.getByRole('button', { name: 'Supprimer la clé DeepSeek' }));
        expect(localStorage.getItem('deepseek_api_key')).toBeNull();
        expect(localStorage.getItem('google_api_key')).toBe('AIza-original');
        expect(await within(card).findByLabelText('Clé API DeepSeek')).toHaveValue('');
    });
});
