"use client";

import { useEffect, useRef, useState } from 'react';
import { AlertCircle, ArrowLeft, BookOpen, ChevronLeft, ChevronRight, FileText, Loader2, MoreHorizontal, RefreshCcw, Trash2 } from 'lucide-react';
import CoverThumbnailImage from '@/components/CoverThumbnailImage';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Skeleton } from '@/components/ui/skeleton';
import { cn, getCoverThumbnailUrl, getPageDisplayStatus } from '@/lib/utils';

const PAGE_GRID = 'grid grid-cols-2 gap-2.5 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8';
const CHAPTER_GRID = 'grid gap-x-8 lg:grid-cols-2 xl:gap-x-12';
const FOCUS_STYLE = 'outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#8dbbff]/60';
const STATUS = {
    not_started: { label: 'Vide', dot: 'bg-slate-400' },
    in_progress: { label: 'En cours', dot: 'bg-orange-400' },
    pending_review: { label: 'À valider', dot: 'bg-yellow-400' },
    completed: { label: 'Terminé', dot: 'bg-green-400' },
    rejected: { label: 'Rejeté', dot: 'bg-red-400' },
};

function getStatus(status) {
    return STATUS[status === 'empty' ? 'not_started' : status] || STATUS.not_started;
}

function StatusLabel({ status }) {
    const { label, dot } = getStatus(status);
    return (
        <span className="inline-flex min-w-0 items-center gap-2 text-xs leading-5 text-slate-400">
            <span aria-hidden="true" className={cn('size-1.5 shrink-0 rounded-full', dot)} />
            {label}
        </span>
    );
}

function DrawerCover({ tome }) {
    const src = tome.cover_url ? getCoverThumbnailUrl(tome.cover_url, 256) : null;
    const [failedSource, setFailedSource] = useState(null);
    return (
        <div className="relative aspect-[2/3] w-14 shrink-0 overflow-hidden rounded-md border border-white/10 bg-[#071625] sm:w-20">
            {src && failedSource !== src ? (
                <CoverThumbnailImage src={src} crossOrigin="anonymous" alt={`Couverture du tome ${tome.numero}`} sizes="(max-width: 639px) 56px, 80px" className="h-full w-full object-contain" onError={() => setFailedSource(src)} />
            ) : (
                <div role="img" aria-label="Couverture indisponible" className="flex h-full items-center justify-center text-slate-400">
                    <BookOpen aria-hidden="true" className="size-6" strokeWidth={1.25} />
                </div>
            )}
        </div>
    );
}

function ContentState({ loading, error, empty, pages = false, onRetry }) {
    if (loading) {
        return (
            <div role="status" aria-label={pages ? 'Chargement des pages' : 'Chargement des chapitres'}>
                <span className="sr-only">{pages ? 'Chargement des pages…' : 'Chargement des chapitres…'}</span>
                <div aria-hidden="true" className={pages ? PAGE_GRID : CHAPTER_GRID}>
                    {Array.from({ length: pages ? 16 : 8 }, (_, index) => (
                        pages ? <Skeleton key={index} className="h-28 rounded-lg bg-white/5 motion-reduce:animate-none" /> : (
                            <div key={index} className="flex min-h-24 items-center gap-4 border-b border-white/8 py-4">
                                <Skeleton className="h-7 w-9 shrink-0 bg-white/8 motion-reduce:animate-none" />
                                <div className="flex-1 space-y-3">
                                    <Skeleton className="h-4 w-3/4 bg-white/8 motion-reduce:animate-none" />
                                    <div className="h-3 w-16 rounded bg-white/5" />
                                </div>
                            </div>
                        )
                    ))}
                </div>
            </div>
        );
    }
    const Icon = error ? AlertCircle : pages ? FileText : BookOpen;
    return (
        <div role={error ? 'alert' : undefined} className="flex min-h-48 flex-col items-center justify-center px-3 py-10 text-center">
            <Icon aria-hidden="true" className={cn('size-7', error ? 'text-red-300' : 'text-slate-400')} strokeWidth={1.5} />
            <p className="mt-4 text-sm font-semibold text-slate-100">{error ? `Impossible de charger les ${pages ? 'pages' : 'chapitres'}` : empty}</p>
            <p className="mt-2 max-w-md break-words text-sm leading-6 text-slate-400">{error || (pages ? 'Les pages ajoutées à ce chapitre apparaîtront ici.' : 'Les chapitres ajoutés à ce tome apparaîtront ici.')}</p>
            {error && <Button type="button" variant="outline" size="sm" onClick={onRetry} className="mt-5 h-11 text-slate-200"><RefreshCcw aria-hidden="true" className="size-4" /> Réessayer</Button>}
        </div>
    );
}

/** The trigger stays visible on touch screens; destructive actions never overlap the reading target. */
function DeleteMenu({ label, actionLabel, disabled, onRequest, className }) {
    const triggerRef = useRef(null);
    const opensDialogRef = useRef(false);
    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button ref={triggerRef} type="button" variant="ghost" size="icon" aria-label={label} disabled={disabled} className={cn('size-11 shrink-0 rounded-md text-slate-400 hover:bg-white/8 hover:text-white', className)}>
                    <MoreHorizontal aria-hidden="true" className="size-4" />
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="z-[60] max-w-[calc(100vw-2rem)]" onCloseAutoFocus={event => {
                if (opensDialogRef.current) event.preventDefault();
                opensDialogRef.current = false;
            }}>
                <DropdownMenuItem disabled={disabled} onSelect={() => {
                    opensDialogRef.current = true;
                    onRequest(triggerRef.current);
                }} className="min-h-11 cursor-pointer gap-2 text-red-300 focus:bg-red-400/10 focus:text-red-200">
                    <Trash2 aria-hidden="true" className="size-4 text-red-300" />
                    {actionLabel}
                </DropdownMenuItem>
            </DropdownMenuContent>
        </DropdownMenu>
    );
}

function PageCollection({ chapter, pages, isPublicViewer, isAdmin, deletingTarget, onOpenPage, onRequestDelete }) {
    const [filter, setFilter] = useState('all');
    const items = [...pages].sort((a, b) => Number(a.numero_page) - Number(b.numero_page)).map(page => ({ page, status: getPageDisplayStatus(page.statut, isPublicViewer) }));
    const counts = items.reduce((acc, { status }) => {
        const key = Object.hasOwn(STATUS, status) ? status : 'not_started';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});
    const visible = filter === 'all' ? items : items.filter(({ status }) => (Object.hasOwn(STATUS, status) ? status : 'not_started') === filter);

    return (
        <>
            <div className="mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
                <h3 className="text-sm font-semibold text-slate-100">Pages <span className="ml-2 font-normal tabular-nums text-slate-400">{pages.length}</span></h3>
                <div className="ml-auto flex min-w-0 items-center gap-1">
                    <Select value={filter} onValueChange={setFilter}>
                        <SelectTrigger aria-label="Filtrer les pages par statut" className="w-40 rounded-lg border-white/12 text-slate-300 shadow-none data-[size=default]:h-11 sm:w-44"><SelectValue /></SelectTrigger>
                        <SelectContent position="popper" align="end" className="z-[60]">
                            <SelectItem value="all">Tous les statuts</SelectItem>
                            {Object.entries(STATUS).filter(([key]) => counts[key] || filter === key).map(([key, value]) => (
                                <SelectItem key={key} value={key}><span className="flex w-full items-center gap-3"><span>{value.label}</span><span className="ml-auto tabular-nums text-slate-400">{counts[key] || 0}</span></span></SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    {isAdmin && <DeleteMenu label={`Actions du chapitre ${chapter.numero}`} actionLabel="Vider le chapitre" disabled={Boolean(deletingTarget)} onRequest={trigger => onRequestDelete({ kind: 'chapter', item: chapter }, trigger)} />}
                </div>
            </div>
            <p role="status" aria-live="polite" className="sr-only">{visible.length} {visible.length === 1 ? 'page affichée' : 'pages affichées'}</p>
            {visible.length ? (
                <ul aria-label="Pages du chapitre" className={PAGE_GRID}>
                    {visible.map(({ page, status }) => (
                        <li key={page.id} className="relative min-w-0 rounded-lg border border-white/10 bg-white/[0.025]">
                            <button type="button" title={`Page ${page.numero_page} - ${status}`} aria-label={`Ouvrir la page ${page.numero_page} — ${getStatus(status).label}`} onClick={() => onOpenPage(page)} className={cn('flex min-h-28 w-full cursor-pointer flex-col justify-between gap-3 rounded-lg p-3 text-left transition-colors hover:bg-white/[0.045] motion-reduce:transition-none sm:p-4', FOCUS_STYLE)}>
                                <span className={cn('block min-w-0', isAdmin && 'pr-8')}>
                                    <span className="block text-xs text-slate-400">Page</span>
                                    <span className="mt-1 block break-words font-serif text-2xl font-black leading-none tabular-nums text-white">{page.numero_page}</span>
                                </span>
                                <StatusLabel status={status} />
                            </button>
                            {isAdmin && <DeleteMenu label={`Actions de la page ${page.numero_page}`} actionLabel="Supprimer les bulles" disabled={Boolean(deletingTarget)} onRequest={trigger => onRequestDelete({ kind: 'page', item: page }, trigger)} className="absolute right-0 top-0" />}
                        </li>
                    ))}
                </ul>
            ) : (
                <div className="py-12 text-center">
                    <p className="text-sm text-slate-300">Aucune page avec ce statut.</p>
                    <Button type="button" variant="outline" onClick={() => setFilter('all')} className="mt-4 h-11">Afficher toutes les pages</Button>
                </div>
            )}
        </>
    );
}

function DeleteConfirmation({ target, open, busy, onClose, onConfirm, onRestoreFocus }) {
    const cancelRef = useRef(null);
    const chapter = target?.kind === 'chapter';
    const title = chapter ? `Vider le chapitre ${target.item.numero} ?` : `Supprimer les bulles de la page ${target?.item.numero_page} ?`;
    return (
        <Dialog open={open} onOpenChange={open => { if (!open && !busy) onClose(); }}>
            <DialogContent role="alertdialog" showCloseButton={false} className="gap-6 border-white/12 bg-[#071625] p-5 text-slate-100 sm:p-6" onOpenAutoFocus={event => { event.preventDefault(); cancelRef.current?.focus(); }} onCloseAutoFocus={onRestoreFocus} onEscapeKeyDown={event => { if (busy) event.preventDefault(); }} onPointerDownOutside={event => { if (busy) event.preventDefault(); }}>
                <DialogHeader className="gap-3 text-left">
                    <DialogTitle className="text-lg leading-snug text-white">{title}</DialogTitle>
                    <DialogDescription className="text-sm leading-6 text-slate-300">
                        {chapter ? 'Toutes les bulles de ce chapitre seront supprimées et ses pages repasseront à l’état « Vide ».' : 'Toutes les bulles de cette page seront supprimées. Elle repassera à l’état « Vide ».'}
                        <span className="mt-2 block text-slate-400">Cette action est irréversible.</span>
                    </DialogDescription>
                </DialogHeader>
                <DialogFooter className="gap-2">
                    <Button ref={cancelRef} type="button" variant="outline" disabled={busy} onClick={onClose} className="h-11 text-slate-200">Annuler</Button>
                    <Button type="button" variant="outline" disabled={busy} onClick={onConfirm} className="h-11 border-red-400/25 bg-red-400/10 text-red-200 hover:bg-red-400/20 hover:text-red-100">
                        {busy ? <Loader2 aria-hidden="true" className="size-4 animate-spin" /> : <Trash2 aria-hidden="true" className="size-4" />}
                        {busy ? 'Suppression…' : 'Supprimer les bulles'}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}

/** Drawer body only: the existing Sheet and its drag/open animations belong to DashboardClient. */
export default function VolumeDrawerContent({ tome, mangaTitle, chapter, chapters, pages, state, isPublicViewer, isAdmin, deletingTarget, onOpenChapter, onReturnToChapters, onRetry, onOpenPage, onDeletePage, onDeleteChapter }) {
    const scrollRef = useRef(null);
    const titleRef = useRef(null);
    const chapterButtonsRef = useRef(new Map());
    const previousChapterRef = useRef(chapter?.id);
    const chapterScrollRef = useRef(0);
    const deleteTriggerRef = useRef(null);
    const confirmationBusyRef = useRef(false);
    const [pendingDeletion, setPendingDeletion] = useState(null);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const chapterId = chapter?.id;
    const sortedChapters = [...chapters].sort((a, b) => Number(a.numero) - Number(b.numero));
    const chapterIndex = sortedChapters.findIndex(item => item.id === chapter?.id);
    const loading = state.status === 'loading-chapters' || state.status === 'loading-pages';
    const error = state.status === 'error-chapters' || state.status === 'error-pages';
    const ready = state.status === 'chapters-ready' || state.status === 'pages-ready';
    const title = chapter ? chapter.titre : tome?.titre || tome?.title || tome?.nom;

    useEffect(() => {
        const previous = previousChapterRef.current;
        if (previous === chapterId) return;
        previousChapterRef.current = chapterId;
        if (scrollRef.current) scrollRef.current.scrollTop = chapterId != null ? 0 : chapterScrollRef.current;
        // Returning to the summary restores the exact chapter, including its scroll position.
        const focusTarget = chapterId != null ? titleRef.current : chapterButtonsRef.current.get(previous) || titleRef.current;
        focusTarget?.focus({ preventScroll: true });
    }, [chapterId]);

    const openChapter = item => {
        if (!chapter) chapterScrollRef.current = scrollRef.current?.scrollTop || 0;
        onOpenChapter(item);
    };

    const requestDelete = (target, trigger) => {
        deleteTriggerRef.current = trigger;
        setPendingDeletion(target);
        setDeleteDialogOpen(true);
    };

    const confirmDelete = async () => {
        if (!pendingDeletion || !deleteDialogOpen || deletingTarget || confirmationBusyRef.current) return;
        confirmationBusyRef.current = true;
        try {
            const success = pendingDeletion.kind === 'chapter' ? await onDeleteChapter(pendingDeletion.item) : await onDeletePage(pendingDeletion.item);
            if (success) setDeleteDialogOpen(false);
        } finally {
            confirmationBusyRef.current = false;
        }
    };

    if (!tome) return null;

    return (
        <>
            <div ref={scrollRef} role="region" aria-label="Contenu du tome" className="min-h-0 flex-1 overflow-y-auto overscroll-y-contain [scrollbar-gutter:stable]">
                {chapter && (
                    <nav aria-label="Navigation des chapitres" className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-white/10 bg-[#071625] px-3 py-1 sm:px-7 lg:px-9">
                        <Button type="button" variant="ghost" onClick={onReturnToChapters} className="h-11 min-w-0 gap-2 px-2 text-slate-300 hover:bg-white/8 hover:text-white"><ArrowLeft aria-hidden="true" className="size-4" /><span className="truncate">Tous les chapitres</span></Button>
                        <div className="flex shrink-0 items-center gap-0.5">
                            <Button type="button" variant="ghost" size="icon" aria-label="Chapitre précédent" title="Chapitre précédent" disabled={chapterIndex <= 0} onClick={() => openChapter(sortedChapters[chapterIndex - 1])} className="size-11 text-slate-300 hover:bg-white/8 hover:text-white"><ChevronLeft aria-hidden="true" className="size-4" /></Button>
                            <Button type="button" variant="ghost" size="icon" aria-label="Chapitre suivant" title="Chapitre suivant" disabled={chapterIndex < 0 || chapterIndex >= sortedChapters.length - 1} onClick={() => openChapter(sortedChapters[chapterIndex + 1])} className="size-11 text-slate-300 hover:bg-white/8 hover:text-white"><ChevronRight aria-hidden="true" className="size-4" /></Button>
                        </div>
                    </nav>
                )}

                <div className="px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:px-8 lg:px-10">
                    <SheetHeader className="flex-row items-start gap-4 px-0 pb-6 pt-3 text-left sm:items-center sm:gap-5 sm:pb-7 sm:pt-4">
                        <DrawerCover tome={tome} />
                        <div className="min-w-0 flex-1">
                            <p className="mb-1.5 break-words text-xs leading-5 text-slate-400">{mangaTitle}{chapter ? ` · Tome ${tome.numero}` : ''}</p>
                            <SheetTitle ref={titleRef} tabIndex={-1} data-drawer-title className="w-fit max-w-full rounded-sm font-serif text-2xl font-black leading-tight text-white outline-none sm:text-3xl">{chapter ? `Chapitre ${chapter.numero}` : `Tome ${tome.numero}`}</SheetTitle>
                            <SheetDescription className="mt-2 max-w-3xl break-words text-sm leading-6 text-slate-300 [overflow-wrap:anywhere]">{title || (chapter ? 'Sélectionnez une page pour l’ouvrir.' : 'Sommaire et pages du tome.')}</SheetDescription>
                        </div>
                    </SheetHeader>

                    <section aria-label={chapter ? 'Pages' : 'Sommaire'} aria-busy={loading} className="border-t border-white/10 pt-4">
                        {!chapter && (
                            <div className="mb-2 flex items-center justify-between gap-4">
                                <h3 className="text-sm font-semibold text-slate-100">Sommaire</h3>
                                {!loading && !error && <p className="text-xs tabular-nums text-slate-400">{chapters.length} {chapters.length === 1 ? 'chapitre' : 'chapitres'}</p>}
                            </div>
                        )}
                        {loading || error || !ready ? (
                            <ContentState loading={loading} error={error ? state.error || 'Veuillez réessayer dans quelques instants.' : null} pages={Boolean(chapter)} empty={chapter ? 'Aucune page dans ce chapitre' : 'Aucun chapitre dans ce volume'} onRetry={onRetry} />
                        ) : chapter ? (
                            <PageCollection key={chapter.id} chapter={chapter} pages={pages} isPublicViewer={isPublicViewer} isAdmin={isAdmin} deletingTarget={deletingTarget} onOpenPage={onOpenPage} onRequestDelete={requestDelete} />
                        ) : (
                            <ol aria-label="Chapitres du tome" className={CHAPTER_GRID}>
                                {sortedChapters.map(item => (
                                    <li key={item.id} className="min-w-0 border-b border-white/8">
                                        <button ref={node => { if (node) chapterButtonsRef.current.set(item.id, node); else chapterButtonsRef.current.delete(item.id); }} type="button" aria-label={`Ouvrir le chapitre ${item.numero}${item.titre ? ` : ${item.titre}` : ''}`} onClick={() => openChapter(item)} className={cn('flex min-h-24 w-full cursor-pointer items-center gap-4 rounded-md px-2 py-4 text-left transition-colors hover:bg-white/[0.035] motion-reduce:transition-none sm:gap-5 sm:px-3', FOCUS_STYLE)}>
                                            <span aria-hidden="true" className="w-10 shrink-0 text-right font-serif text-2xl font-black leading-none tabular-nums text-[#bdd6ff] sm:w-14">{String(item.numero).padStart(2, '0')}</span>
                                            <span className="min-w-0 flex-1">
                                                <span className="block break-words text-sm font-medium leading-6 text-slate-100 [overflow-wrap:anywhere]">{item.titre || `Chapitre ${item.numero}`}</span>
                                                <span className="mt-1 block"><StatusLabel status={item.global_status} /></span>
                                            </span>
                                            <ChevronRight aria-hidden="true" className="size-4 shrink-0 text-slate-500" />
                                        </button>
                                    </li>
                                ))}
                            </ol>
                        )}
                    </section>
                </div>
            </div>
            <DeleteConfirmation target={pendingDeletion} open={deleteDialogOpen} busy={Boolean(deletingTarget)} onClose={() => setDeleteDialogOpen(false)} onConfirm={confirmDelete} onRestoreFocus={event => {
                event.preventDefault();
                const trigger = deleteTriggerRef.current;
                (trigger?.isConnected ? trigger : titleRef.current)?.focus({ preventScroll: true });
            }} />
        </>
    );
}
