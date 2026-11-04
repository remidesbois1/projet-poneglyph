import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import AnnotateLeftSidebar, { formatPageStatus } from './AnnotateLeftSidebar';

const model = (overrides = {}) => ({
    key: 'lighton-3-browser',
    name: 'LightOnOCR 3',
    scope: 'bubble',
    runtime: 'browser',
    engine: 'lighton3',
    size: '1,7 Go',
    selected: true,
    status: 'ready',
    progress: null,
    message: null,
    ...overrides,
});

const handlers = {
    toggleOcrModel: vi.fn(),
    onPrepareModels: vi.fn(),
    onRunPageOcr: vi.fn(),
    onCancelPageOcr: vi.fn(),
    setShowApiKeyModal: vi.fn(),
    setShowDescModal: vi.fn(),
    handleSubmitPage: vi.fn(),
};

const renderSidebar = (props = {}) =>
    render(
        <AnnotateLeftSidebar
            mangaSlug="one-piece"
            page={{ numero_page: 3, statut: 'in_progress', chapitres: { numero: 1 } }}
            chapterPages={[1, 2, 3]}
            navContext={{ prev: null, next: null }}
            role="Admin"
            detectionStatus="ready"
            canRunPageOcr
            {...handlers}
            {...props}
        />
    );

const nextStep = () => within(document.querySelector('footer')).getAllByRole('button')[0];

describe('AnnotateLeftSidebar', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders the public reader when private workflow metadata is absent', () => {
        render(
            <AnnotateLeftSidebar
                mangaSlug="one-piece"
                page={{ id: 42, numero_page: 3 }}
                chapterPages={[]}
                navContext={{ prev: null, next: null }}
                isGuest
                role="Guest"
            />
        );

        expect(screen.getByText('lecture publique')).toBeInTheDocument();
    });

    it('keeps formatting authenticated workflow statuses', () => {
        expect(formatPageStatus('pending_review')).toBe('En revue');
    });

    describe('model list', () => {
        it('groups bubble and page models under their own fieldsets', () => {
            renderSidebar({
                ocrModels: [
                    model(),
                    model({ key: 'deepseek-page', name: 'DeepSeek OCR', scope: 'page', runtime: 'cloud' }),
                ],
            });

            const bubbles = screen.getByRole('group', { name: 'Bulles' });
            const page = screen.getByRole('group', { name: 'Page entière' });
            expect(within(bubbles).getByRole('checkbox', { name: 'LightOnOCR 3 · Navigateur' })).toBeChecked();
            expect(within(page).getByRole('checkbox', { name: 'DeepSeek OCR · Cloud' })).toBeChecked();
            expect(within(bubbles).queryByRole('checkbox', { name: /DeepSeek/ })).not.toBeInTheDocument();
        });

        it('omits the page group when no page model is available', () => {
            renderSidebar({ ocrModels: [model()] });

            expect(screen.getByRole('group', { name: 'Bulles' })).toBeInTheDocument();
            expect(screen.queryByRole('group', { name: 'Page entière' })).not.toBeInTheDocument();
        });

        it('hides the model list from non-staff users', () => {
            renderSidebar({ role: 'User', ocrModels: [model()] });

            expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
            expect(screen.queryByRole('button', { name: /OCR de la page/ })).not.toBeInTheDocument();
        });

        it('toggles a model by key', () => {
            renderSidebar({ ocrModels: [model({ selected: false })] });

            const checkbox = screen.getByRole('checkbox', { name: 'LightOnOCR 3 · Navigateur' });
            expect(checkbox).not.toBeChecked();
            fireEvent.click(checkbox);

            expect(handlers.toggleOcrModel).toHaveBeenCalledWith('lighton-3-browser');
        });

        it('shows the size of a selected model waiting to be loaded', () => {
            renderSidebar({ ocrModels: [model({ status: 'load', size: '1,7 Go' })] });
            expect(screen.getByText('À charger · 1,7 Go')).toBeInTheDocument();
        });

        it('shows the size of a selected model waiting to be downloaded', () => {
            renderSidebar({ ocrModels: [model({ status: 'download', size: '2 Go' })] });
            expect(screen.getByText('À télécharger · 2 Go')).toBeInTheDocument();
        });

        it('shows loading progress of a selected model', () => {
            renderSidebar({
                ocrModels: [model({ status: 'loading', progress: 42, message: 'Téléchargement' })],
            });

            expect(screen.getByText('Téléchargement · 42 %')).toBeInTheDocument();
            expect(screen.getByRole('progressbar')).toHaveAttribute('value', '42');
        });

        it('shows the setup requirement of a selected model', () => {
            renderSidebar({
                ocrModels: [model({ runtime: 'cloud', status: 'setup', message: 'Clé DeepSeek requise' })],
            });
            expect(screen.getByText('Clé DeepSeek requise')).toBeInTheDocument();
        });

        it('shows no status for unselected models', () => {
            renderSidebar({
                ocrModels: [
                    model({ selected: false, status: 'download', size: '2 Go' }),
                    model({ key: 'b', name: 'Surya', selected: false, status: 'loading', progress: 42, message: 'Téléchargement' }),
                ],
            });

            expect(screen.queryByText(/À télécharger/)).not.toBeInTheDocument();
            expect(screen.queryByText(/Téléchargement/)).not.toBeInTheDocument();
            expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
        });
    });

    describe('next-step button', () => {
        it('asks to choose a model when none is selected', () => {
            renderSidebar({ ocrModels: [model({ selected: false })] });

            expect(nextStep()).toHaveTextContent('Choisissez un modèle');
            expect(nextStep()).toBeDisabled();
        });

        it('loads models when a selected model is not loaded', () => {
            renderSidebar({ ocrModels: [model({ status: 'load' })] });

            expect(nextStep()).toHaveTextContent('Charger les modèles');
            fireEvent.click(nextStep());
            expect(handlers.onPrepareModels).toHaveBeenCalledTimes(1);
        });

        it('loads models when the detector is idle', () => {
            renderSidebar({ ocrModels: [model()], detectionStatus: 'idle' });

            expect(nextStep()).toHaveTextContent('Charger les modèles');
            fireEvent.click(nextStep());
            expect(handlers.onPrepareModels).toHaveBeenCalledTimes(1);
        });

        it('waits for the detector with its progress', () => {
            renderSidebar({ ocrModels: [model()], detectionStatus: 'loading', detectionProgress: 42 });

            expect(nextStep()).toHaveTextContent('Chargement du détecteur · 42 %');
            expect(nextStep()).toBeDisabled();
        });

        it('waits while a selected model is loading', () => {
            renderSidebar({ ocrModels: [model(), model({ key: 'b', name: 'Surya', status: 'loading', progress: 10 })] });

            expect(nextStep()).toHaveTextContent('Chargement des modèles…');
            expect(nextStep()).toBeDisabled();
        });

        it('opens API settings when a selected model needs setup', () => {
            renderSidebar({ ocrModels: [model({ runtime: 'cloud', status: 'setup', message: 'Clé DeepSeek requise' })] });

            expect(nextStep()).toHaveTextContent('Configurer l’accès API');
            fireEvent.click(nextStep());
            expect(handlers.setShowApiKeyModal).toHaveBeenCalledWith(true);
            expect(handlers.onRunPageOcr).not.toHaveBeenCalled();
        });

        it('is blocked when a selected model is unavailable', () => {
            renderSidebar({ ocrModels: [model({ status: 'unavailable' })] });

            expect(nextStep()).toHaveTextContent('Modèle indisponible');
            expect(nextStep()).toBeDisabled();
        });

        it('runs page OCR when every selected model is ready', () => {
            renderSidebar({ ocrModels: [model(), model({ key: 'b', selected: false, status: 'download' })] });

            expect(nextStep()).toHaveTextContent('Lancer l’OCR de la page');
            fireEvent.click(nextStep());
            expect(handlers.onRunPageOcr).toHaveBeenCalledTimes(1);
        });

        it('cannot run page OCR when the page is not runnable', () => {
            renderSidebar({ ocrModels: [model()], canRunPageOcr: false });

            expect(nextStep()).toHaveTextContent('Lancer l’OCR de la page');
            expect(nextStep()).toBeDisabled();
        });

        it('cannot run page OCR while submitting', () => {
            renderSidebar({ ocrModels: [model()], isSubmitting: true });

            expect(nextStep()).toHaveTextContent('Lancer l’OCR de la page');
            expect(nextStep()).toBeDisabled();
        });

        it('locks the editor and offers cancellation while page OCR runs', () => {
            renderSidebar({ ocrModels: [model()], isPageOcrRunning: true });

            expect(nextStep()).toHaveTextContent('OCR en cours…');
            expect(nextStep()).toBeDisabled();
            expect(screen.getByRole('checkbox', { name: 'LightOnOCR 3 · Navigateur' })).toBeDisabled();
            expect(screen.getByRole('button', { name: 'Envoyer en validation' })).toBeDisabled();

            fireEvent.click(screen.getByRole('button', { name: 'Annuler' }));
            expect(handlers.onCancelPageOcr).toHaveBeenCalledTimes(1);
        });

        it('counts the bubbles left to review', () => {
            renderSidebar({ ocrModels: [model()], isReviewingPage: true, reviewRemaining: 2 });

            expect(nextStep()).toHaveTextContent('3 bulles à vérifier');
            expect(nextStep()).toBeDisabled();
            expect(screen.queryByRole('button', { name: 'Annuler' })).not.toBeInTheDocument();
        });
    });
});
