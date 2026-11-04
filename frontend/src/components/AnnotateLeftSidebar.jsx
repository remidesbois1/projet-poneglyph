import React from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import {
    ArrowLeft,
    ChevronLeft,
    ChevronRight,
    Download,
    FileText,
    KeyRound,
    Loader2,
    ScanText,
    Settings2,
} from 'lucide-react';
import { OCR_RUNTIME_LABELS } from '@/lib/ocrModels';
import { cn } from '@/lib/utils';

const PAGE_STATUSES = [
    { value: 'not_started', label: 'Non commencée' },
    { value: 'in_progress', label: 'En cours' },
    { value: 'pending_review', label: 'En revue' },
    { value: 'completed', label: 'Validée' },
];

const MODEL_GROUPS = [
    { scope: 'bubble', title: 'Bulles' },
    { scope: 'page', title: 'Page entière' },
];

const STEP_ICONS = { prepare: Download, setup: KeyRound, run: ScanText };

export function formatPageStatus(status, isGuest = false) {
    return (
        PAGE_STATUSES.find((item) => item.value === status)?.label ||
        (typeof status === 'string' && status.trim()
            ? status.replace(/_/g, ' ')
            : isGuest
              ? 'lecture publique'
              : 'Statut inconnu')
    );
}

function formatPercent(value) {
    return Number.isFinite(value) ? `${Math.round(value)} %` : null;
}

function describeModelState(model) {
    if (model.status === 'loading') return [model.message, formatPercent(model.progress)].filter(Boolean).join(' · ');
    if (model.message) return model.message;
    if (model.status === 'download') return ['À télécharger', model.size].filter(Boolean).join(' · ');
    if (model.status === 'load') return ['À charger', model.size].filter(Boolean).join(' · ');
    return null;
}

/** The single page OCR button always shows the next step: load, configure, then run. */
function getPageOcrStep({ models, detectionStatus, detectionProgress, isRunning, isReviewing, reviewRemaining }) {
    if (isRunning) return { label: 'OCR en cours…', pending: true };
    if (isReviewing) {
        const count = reviewRemaining + 1;
        return { label: `${count} bulle${count > 1 ? 's' : ''} à vérifier`, pending: true };
    }
    const selected = models.filter((model) => model.selected);
    if (!selected.length) return { label: 'Choisissez un modèle' };
    if (detectionStatus === 'idle' || selected.some((model) => model.status === 'load' || model.status === 'download')) {
        return { label: 'Charger les modèles', action: 'prepare' };
    }
    if (detectionStatus !== 'ready') {
        const progress = formatPercent(detectionProgress);
        return { label: `Chargement du détecteur${progress ? ` · ${progress}` : '…'}`, pending: true };
    }
    if (selected.some((model) => model.status === 'loading')) return { label: 'Chargement des modèles…', pending: true };
    if (selected.some((model) => model.status === 'setup')) return { label: 'Configurer l’accès API', action: 'setup' };
    if (selected.some((model) => model.status === 'unavailable')) return { label: 'Modèle indisponible' };
    return { label: 'Lancer l’OCR de la page', action: 'run' };
}

function OcrModelOption({ model, disabled, onToggle }) {
    const runtime = OCR_RUNTIME_LABELS[model.runtime];
    const detail = model.selected && model.status !== 'ready' ? describeModelState(model) : null;
    const progress = model.selected && model.status === 'loading' && Number.isFinite(model.progress)
        ? Math.max(0, Math.min(100, Math.round(model.progress)))
        : null;

    return (
        <li>
            <label className="flex min-h-9 cursor-pointer items-center gap-3 rounded-sm text-[13px] text-slate-300 hover:text-white has-[:disabled]:cursor-default has-[:disabled]:hover:text-slate-300">
                <input
                    type="checkbox"
                    aria-label={`${model.name} · ${runtime}`}
                    checked={model.selected}
                    disabled={disabled}
                    onChange={() => onToggle(model.key)}
                    className="h-4 w-4 shrink-0 cursor-pointer [color-scheme:dark] accent-sky-400 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-sky-400 disabled:cursor-default"
                />
                <span className="min-w-0 flex-1 truncate">{model.name}</span>
                <span className="shrink-0 text-[11px] text-slate-500">{runtime}</span>
            </label>
            {detail && (
                <p
                    className={cn(
                        'mb-1.5 ml-7 text-[11px] leading-4',
                        model.status === 'setup' || model.status === 'unavailable' ? 'text-amber-200/80' : 'text-slate-400'
                    )}
                >
                    {detail}
                </p>
            )}
            {progress !== null && (
                <progress
                    aria-label={`Préparation de ${model.name}`}
                    value={progress}
                    max={100}
                    className="mb-2 ml-7 block h-1 w-[calc(100%-1.75rem)] accent-sky-400"
                />
            )}
        </li>
    );
}

export default function AnnotateLeftSidebar({
    fromSearch,
    mangaSlug,
    page,
    chapterPages,
    navContext,
    goToPrev,
    goToNext,
    isGuest,
    role,
    isSandbox = false,
    ocrModels = [],
    toggleOcrModel,
    onPrepareModels,
    detectionStatus,
    detectionProgress,
    canRunPageOcr = false,
    isPageOcrRunning = false,
    isReviewingPage = false,
    reviewRemaining = 0,
    onRunPageOcr,
    onCancelPageOcr,
    isSubmitting = false,
    setShowDescModal,
    setShowApiKeyModal,
    handleSubmitPage,
    handlePageStatusChange,
    isUpdatingPageStatus,
}) {
    const isStaff = role === 'Admin' || role === 'Modo';
    const showOcr = !isGuest && isStaff;
    const canInlineEditStatus =
        role === 'Admin' &&
        page?.statut &&
        handlePageStatusChange &&
        !isSandbox;
    const busy = Boolean(isSubmitting || isPageOcrRunning || isReviewingPage);
    const step = getPageOcrStep({
        models: ocrModels,
        detectionStatus,
        detectionProgress,
        isRunning: isPageOcrRunning,
        isReviewing: isReviewingPage,
        reviewRemaining,
    });
    const StepIcon = step.pending ? Loader2 : STEP_ICONS[step.action];
    const runStep = {
        prepare: onPrepareModels,
        setup: () => setShowApiKeyModal(true),
        run: onRunPageOcr,
    }[step.action];
    const stepDisabled = !runStep || step.pending || (step.action === 'run' && (!canRunPageOcr || isSubmitting));

    return (
        <aside
            aria-label="Outils d’annotation"
            className="relative z-40 hidden h-full w-[288px] shrink-0 flex-col border-r border-white/10 bg-[#0b1420] text-sm text-slate-200 lg:flex"
        >
            <header className="shrink-0 border-b border-white/10 p-4">
                <Link
                    href={
                        !mangaSlug
                            ? '/'
                            : fromSearch
                              ? `/${mangaSlug}/search`
                              : `/${mangaSlug}/dashboard`
                    }
                    className="mb-5 inline-flex items-center gap-2 rounded-sm text-xs text-slate-400 transition-colors hover:text-white focus-visible:outline-2 focus-visible:outline-sky-400"
                >
                    <ArrowLeft size={14} />
                    {!mangaSlug
                        ? 'Accueil'
                        : fromSearch
                          ? 'Recherche'
                          : 'Tableau de bord'}
                </Link>
                <h2 className="text-base font-semibold text-white">
                    {page.chapitres
                        ? `Chapitre ${page.chapitres.numero}`
                        : 'Page de test'}
                    {page.chapitres?.tomes && (
                        <span className="ml-2 text-xs font-normal text-slate-400">
                            Tome {page.chapitres.tomes.numero}
                        </span>
                    )}
                </h2>
                <div className="mt-3 flex items-center justify-between gap-2">
                    {!isGuest && !isSandbox && (
                        <Button
                            variant="ghost"
                            size="icon"
                            aria-label="Page précédente"
                            disabled={!navContext.prev}
                            onClick={goToPrev}
                            className="text-slate-300 hover:bg-white/10 hover:text-white"
                        >
                            <ChevronLeft />
                        </Button>
                    )}
                    <span className="flex-1 text-center text-sm tabular-nums text-slate-200">
                        Page {page.numero_page || 1}
                        {chapterPages.length > 0 && (
                            <span className="text-slate-400">
                                {' '}
                                / {chapterPages.length}
                            </span>
                        )}
                    </span>
                    {!isGuest && !isSandbox && (
                        <Button
                            variant="ghost"
                            size="icon"
                            aria-label="Page suivante"
                            disabled={!navContext.next}
                            onClick={goToNext}
                            className="text-slate-300 hover:bg-white/10 hover:text-white"
                        >
                            <ChevronRight />
                        </Button>
                    )}
                </div>
                {!isSandbox && (
                    <div className="mt-3 flex items-center justify-between gap-3 border-t border-white/[0.06] pt-3">
                        <span className="text-xs text-slate-400">Statut</span>
                        {canInlineEditStatus ? (
                            <Select
                                value={page.statut}
                                onValueChange={handlePageStatusChange}
                                disabled={isUpdatingPageStatus}
                            >
                                <SelectTrigger
                                    aria-label="État de la page"
                                    className="h-8 w-auto gap-3 border-white/10 bg-transparent text-xs text-slate-200 shadow-none"
                                >
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {PAGE_STATUSES.map((status) => (
                                        <SelectItem
                                            key={status.value}
                                            value={status.value}
                                        >
                                            {status.label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        ) : (
                            <span className="text-xs text-slate-300">
                                {formatPageStatus(page?.statut, isGuest)}
                            </span>
                        )}
                    </div>
                )}
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
                {showOcr && (
                    <section aria-labelledby="page-ocr-heading" className="space-y-4 p-4">
                        <h3 id="page-ocr-heading" className="font-medium text-white">
                            OCR de la page
                        </h3>
                        {MODEL_GROUPS.map((group) => {
                            const models = ocrModels.filter((model) => model.scope === group.scope);
                            if (!models.length) return null;
                            return (
                                <fieldset key={group.scope}>
                                    <legend className="mb-1 text-xs text-slate-400">
                                        {group.title}
                                    </legend>
                                    <ul>
                                        {models.map((model) => (
                                            <OcrModelOption
                                                key={model.key}
                                                model={model}
                                                disabled={busy}
                                                onToggle={toggleOcrModel}
                                            />
                                        ))}
                                    </ul>
                                </fieldset>
                            );
                        })}
                    </section>
                )}
                {role === 'User' && !isGuest && !isSandbox && (
                    <section className="p-4 text-sm leading-relaxed text-slate-300">
                        <h3 className="mb-3 font-medium text-white">
                            Description
                        </h3>
                        <p>
                            {page.description_semantique?.content ||
                                'Aucune description.'}
                        </p>
                        {page.description_semantique?.arc && (
                            <p className="mt-4">
                                <span className="text-slate-400">Arc : </span>
                                {page.description_semantique.arc}
                            </p>
                        )}
                        {page.description_semantique?.characters?.length >
                            0 && (
                            <p className="mt-2">
                                <span className="text-slate-400">
                                    Personnages :{' '}
                                </span>
                                {page.description_semantique.characters.join(
                                    ', '
                                )}
                            </p>
                        )}
                    </section>
                )}
            </div>

            {showOcr && (
                <footer className="shrink-0 border-t border-white/10 p-3">
                    <Button
                        onClick={runStep}
                        disabled={stepDisabled}
                        className="h-10 w-full bg-[#dbe5ef] text-[13px] font-medium text-[#0b1420] shadow-none hover:bg-[#f1f5f9]"
                    >
                        {StepIcon && <StepIcon className={step.pending ? 'animate-spin' : undefined} />}
                        {step.label}
                    </Button>
                    {isPageOcrRunning && onCancelPageOcr && (
                        <Button
                            variant="ghost"
                            onClick={onCancelPageOcr}
                            className="mt-1 h-9 w-full text-[13px] text-slate-400 hover:bg-white/5 hover:text-white"
                        >
                            Annuler
                        </Button>
                    )}
                    <div className="mt-3 border-t border-white/[0.06] pt-2">
                        {!isSandbox && (
                            <Button
                                variant="ghost"
                                onClick={() => setShowDescModal(true)}
                                className="h-10 w-full justify-start text-[13px] text-slate-300 hover:bg-white/5 hover:text-white"
                            >
                                <FileText />
                                Métadonnées
                                <ChevronRight className="ml-auto text-slate-500" />
                            </Button>
                        )}
                        <Button
                            variant="ghost"
                            onClick={() => setShowApiKeyModal(true)}
                            className="h-10 w-full justify-start text-[13px] text-slate-300 hover:bg-white/5 hover:text-white"
                        >
                            <Settings2 />
                            Réglages API
                            <ChevronRight className="ml-auto text-slate-500" />
                        </Button>
                    </div>
                    {!isSandbox && (
                        <Button
                            onClick={handleSubmitPage}
                            disabled={
                                busy ||
                                page.statut === 'pending_review' ||
                                page.statut === 'completed'
                            }
                            className="mt-2 h-10 w-full border border-white/20 bg-transparent text-[13px] font-medium text-slate-200 shadow-none hover:bg-white/10"
                        >
                            {page.statut === 'completed'
                                ? 'Page validée'
                                : page.statut === 'pending_review'
                                  ? 'En attente de validation'
                                  : 'Envoyer en validation'}
                        </Button>
                    )}
                </footer>
            )}
        </aside>
    );
}
