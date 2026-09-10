"use client";

import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { AlertCircle, BookOpen, ChevronLeft, ChevronRight, LayoutGrid, List, RefreshCcw, Search, X } from 'lucide-react';
import CoverThumbnailImage from '@/components/CoverThumbnailImage';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { cn, getCoverThumbnailUrl } from '@/lib/utils';

const PAGE_SIZE = 24;
const GRID_CLASS = 'grid grid-cols-2 gap-x-2 gap-y-4 sm:grid-cols-3 sm:gap-x-3 md:grid-cols-4 xl:grid-cols-6';

function getTitle(tome) {
    return tome.titre || tome.title || tome.nom || '';
}

function normalizeSearch(value) {
    return String(value ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
}

function getPageNumbers(page, total) {
    const visible = new Set([1, total, page - 1, page, page + 1]);
    if (page <= 3) [2, 3, 4].forEach(value => visible.add(value));
    if (page >= total - 2) [total - 3, total - 2, total - 1].forEach(value => visible.add(value));

    const numbers = [...visible].filter(value => value >= 1 && value <= total).sort((a, b) => a - b);
    return numbers.flatMap((value, index) => {
        const previous = numbers[index - 1];
        if (previous && value - previous === 2) return [previous + 1, value];
        if (previous && value - previous > 2) return [`gap-${previous}`, value];
        return [value];
    });
}

function VolumeCover({ tome, compact = false }) {
    const src = tome.cover_url ? getCoverThumbnailUrl(tome.cover_url, compact ? 128 : 512) : null;
    const [failedSource, setFailedSource] = useState(null);

    return (
        <span className={cn('relative block aspect-[2/3] shrink-0 overflow-hidden rounded-md border border-white/10 bg-[#071625]', compact ? 'w-12 sm:w-14' : 'w-full')}>
            {src && failedSource !== src ? (
                <CoverThumbnailImage
                    src={src}
                    crossOrigin="anonymous"
                    alt={`Couverture du tome ${tome.numero}`}
                    fill
                    sizes={compact ? '56px' : '(max-width: 639px) 45vw, (max-width: 767px) 30vw, (max-width: 1279px) 22vw, 220px'}
                    className="h-full w-full object-contain"
                    loading="lazy"
                    onError={() => setFailedSource(src)}
                    unoptimized
                />
            ) : (
                <span className="flex h-full flex-col items-center justify-center gap-3 px-2 text-slate-400">
                    <BookOpen aria-hidden="true" className={compact ? 'size-5' : 'size-8'} strokeWidth={1.25} />
                    {!compact && <span className="text-center text-xs leading-relaxed">Couverture indisponible</span>}
                </span>
            )}
        </span>
    );
}

function VolumeItem({ tome, view, onOpen }) {
    const title = getTitle(tome);
    const compact = view === 'list';

    return (
        <li className="min-w-0">
            <button
                type="button"
                aria-label={`Ouvrir le tome ${tome.numero}${title ? ` : ${title}` : ''}`}
                aria-haspopup="dialog"
                onClick={() => onOpen(tome)}
                className={cn(
                    'w-full cursor-pointer rounded-lg border border-transparent text-left outline-none transition-colors hover:border-white/12 hover:bg-white/[0.035] focus-visible:border-[#8dbbff]/60 focus-visible:ring-2 focus-visible:ring-[#8dbbff]/40 motion-reduce:transition-none',
                    compact ? 'flex items-center gap-4 p-3 sm:px-4' : 'block h-full p-2 sm:p-3',
                )}
            >
                <VolumeCover tome={tome} compact={compact} />
                <span className={cn('block min-w-0', compact ? 'flex-1' : 'mt-3')}>
                    <span className="block font-serif text-lg font-black leading-snug text-white sm:text-xl">
                        Tome <span className="tabular-nums">{tome.numero}</span>
                    </span>
                    <span title={title || undefined} className={cn('mt-1 block text-sm leading-5 text-slate-300', compact ? 'line-clamp-2' : 'line-clamp-2 min-h-10')}>
                        {title || <span className="text-slate-400">Sans titre</span>}
                    </span>
                </span>
                {compact && <ChevronRight aria-hidden="true" className="size-4 shrink-0 text-slate-400" />}
            </button>
        </li>
    );
}

function LibrarySkeleton({ view }) {
    const compact = view === 'list';
    return (
        <div role="status" aria-label="Chargement des tomes">
            <span className="sr-only">Chargement des tomes…</span>
            <div aria-hidden="true" className={compact ? 'divide-y divide-white/8' : GRID_CLASS}>
                {Array.from({ length: PAGE_SIZE }, (_, index) => (
                    <div key={index} className={compact ? 'flex items-center gap-4 p-3 sm:px-4' : 'p-2 sm:p-3'}>
                        <Skeleton className={cn('aspect-[2/3] rounded-md bg-white/8 motion-reduce:animate-none', compact ? 'w-12 shrink-0 sm:w-14' : 'w-full')} />
                        <div className={compact ? 'flex-1 space-y-2' : 'mt-3 space-y-2 pb-5'}>
                            <div className="h-5 w-20 rounded bg-white/8" />
                            <div className="h-4 w-3/4 max-w-64 rounded bg-white/5" />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

/** Catalog-only UI. Chapter/page loading and the existing drawer stay in DashboardClient. */
export default function VolumeLibrary({ mangaTitle, tomes, status, error, onRetry, onOpenTome }) {
    const [query, setQuery] = useState('');
    const [sort, setSort] = useState('asc');
    const [view, setView] = useState('grid');
    const [requestedPage, setRequestedPage] = useState(1);
    const scrollRef = useRef(null);
    const searchRef = useRef(null);
    const resultsRef = useRef(null);
    const focusPageRef = useRef(false);
    const resultsId = useId();
    const loading = status === 'loading';

    const filteredTomes = useMemo(() => {
        const normalized = normalizeSearch(query);
        const numberQuery = normalized.match(/^(?:(?:tome|volume)\s*)?(\d+)$/);
        const terms = normalized.split(/\s+/).filter(Boolean);
        return tomes.filter(tome => {
            // A number, "tome 12" or "volume 12" takes the reader straight to that volume.
            if (numberQuery) return Number(tome.numero) === Number(numberQuery[1]);
            const searchable = normalizeSearch(`${tome.numero} ${getTitle(tome)}`);
            return terms.every(term => searchable.includes(term));
        }).sort((a, b) => {
            const difference = (Number(a.numero) || 0) - (Number(b.numero) || 0);
            return sort === 'asc' ? difference : -difference;
        });
    }, [tomes, query, sort]);

    const totalPages = Math.max(1, Math.ceil(filteredTomes.length / PAGE_SIZE));
    // A refreshed catalogue can be shorter without leaving the user on an empty page.
    const page = Math.min(requestedPage, totalPages);
    const firstIndex = (page - 1) * PAGE_SIZE;
    const visibleTomes = filteredTomes.slice(firstIndex, firstIndex + PAGE_SIZE);

    useEffect(() => {
        if (!focusPageRef.current) return;
        focusPageRef.current = false;
        scrollRef.current?.scrollTo?.({ top: 0, behavior: 'instant' });
        resultsRef.current?.querySelector('button')?.focus({ preventScroll: true });
    }, [page]);

    const clearSearch = () => {
        setQuery('');
        setRequestedPage(1);
        searchRef.current?.focus();
    };

    const changePage = (nextPage) => {
        if (nextPage === page || nextPage < 1 || nextPage > totalPages) return;
        focusPageRef.current = true;
        setRequestedPage(nextPage);
    };

    return (
        <section ref={scrollRef} aria-label={`Bibliothèque ${mangaTitle}`} className="h-full min-h-0 w-full overflow-y-auto overscroll-y-contain pb-2 pr-1 [scrollbar-gutter:stable] sm:pr-3">
            <div className="mb-7 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <h1 className="poneglyph-title min-w-0 text-4xl font-extrabold sm:text-5xl">
                    Bibliothèque {mangaTitle}
                </h1>
                {status === 'ready' && <p className="shrink-0 pb-1 text-sm tabular-nums text-slate-400">{tomes.length} {tomes.length === 1 ? 'tome' : 'tomes'}</p>}
            </div>

            <Card className="poneglyph-panel gap-0 py-0">
                <div className="flex flex-col gap-3 border-b border-white/10 p-4 sm:px-5 lg:flex-row lg:items-center lg:justify-between">
                    <div role="search" className="relative min-w-0 lg:w-full lg:max-w-md">
                        <Search aria-hidden="true" className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
                        <Input
                            ref={searchRef}
                            type="search"
                            value={query}
                            onChange={event => { setQuery(event.target.value); setRequestedPage(1); }}
                            onKeyDown={event => { if (event.key === 'Escape' && query) { event.preventDefault(); clearSearch(); } }}
                            aria-label="Rechercher un tome"
                            aria-controls={resultsId}
                            placeholder="Numéro ou titre…"
                            className="poneglyph-input h-11 rounded-lg pl-10 pr-11 [&::-webkit-search-cancel-button]:appearance-none"
                        />
                        {query && (
                            <Button type="button" variant="ghost" size="icon" aria-label="Effacer la recherche" onClick={clearSearch} className="absolute right-0.5 top-0.5 size-10 rounded-md text-slate-400 hover:bg-white/8 hover:text-white">
                                <X aria-hidden="true" className="size-4" />
                            </Button>
                        )}
                    </div>
                    <div className="flex min-w-0 items-center gap-3">
                        <Select value={sort} onValueChange={value => { setSort(value); setRequestedPage(1); }}>
                            <SelectTrigger aria-label="Ordre des tomes" className="min-w-0 flex-1 rounded-lg text-slate-300 shadow-none data-[size=default]:h-11 lg:w-48">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent position="popper" align="end">
                                <SelectItem value="asc">Numéro croissant</SelectItem>
                                <SelectItem value="desc">Numéro décroissant</SelectItem>
                            </SelectContent>
                        </Select>
                        <div role="group" aria-label="Affichage des tomes" className="flex shrink-0 gap-0.5 rounded-lg border border-white/12 p-0.5">
                            {[{ value: 'grid', label: 'Afficher en grille', Icon: LayoutGrid }, { value: 'list', label: 'Afficher en liste', Icon: List }].map(({ value, label, Icon }) => (
                                <Button key={value} type="button" variant="ghost" size="icon" aria-label={label} title={label} aria-pressed={view === value} onClick={() => setView(value)} className={cn('size-10 rounded-md transition-colors hover:bg-white/8 hover:text-white', view === value ? 'bg-white/10 text-[#8dbbff]' : 'text-slate-400')}>
                                    <Icon aria-hidden="true" className="size-4" />
                                </Button>
                            ))}
                        </div>
                    </div>
                </div>

                <div id={resultsId} ref={resultsRef} aria-busy={loading} className="min-h-64 p-2 sm:p-3">
                    {loading ? <LibrarySkeleton view={view} /> : status === 'error' ? (
                        <div role="alert" className="flex min-h-64 flex-col items-center justify-center px-4 py-10 text-center">
                            <AlertCircle aria-hidden="true" className="size-7 text-red-300" />
                            <p className="mt-4 font-semibold text-slate-100">Impossible de charger la bibliothèque</p>
                            <p className="mt-2 max-w-md text-sm text-slate-400">{error || 'Le catalogue est momentanément indisponible.'}</p>
                            <Button type="button" variant="outline" size="sm" onClick={onRetry} className="mt-5 gap-2 text-slate-200">
                                <RefreshCcw aria-hidden="true" className="size-4" /> Réessayer
                            </Button>
                        </div>
                    ) : tomes.length === 0 ? (
                        <div className="flex min-h-64 flex-col items-center justify-center px-4 py-10 text-center">
                            <BookOpen aria-hidden="true" className="size-8 text-slate-400" strokeWidth={1.5} />
                            <p className="mt-4 font-semibold text-slate-100">Aucun volume disponible</p>
                            <p className="mt-2 text-sm text-slate-400">Les volumes publiés apparaîtront ici.</p>
                        </div>
                    ) : filteredTomes.length === 0 ? (
                        <div className="flex min-h-64 flex-col items-center justify-center px-4 py-10 text-center">
                            <Search aria-hidden="true" className="size-7 text-slate-400" strokeWidth={1.5} />
                            <p className="mt-4 font-semibold text-slate-100">Aucun tome trouvé</p>
                            <p className="mt-2 max-w-md break-words text-sm text-slate-400">Aucun résultat pour « {query.trim()} ».</p>
                            <Button type="button" variant="outline" size="sm" onClick={clearSearch} className="mt-5 text-slate-200">Réinitialiser la recherche</Button>
                        </div>
                    ) : (
                        <ul aria-label="Tomes" className={view === 'list' ? 'divide-y divide-white/8' : GRID_CLASS}>
                            {visibleTomes.map(tome => <VolumeItem key={tome.id} tome={tome} view={view} onOpen={onOpenTome} />)}
                        </ul>
                    )}
                </div>

                {status === 'ready' && (
                    <div className="flex flex-col gap-3 border-t border-white/10 px-4 py-4 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between sm:px-5">
                        <p role="status" aria-live="polite" aria-atomic="true" className="text-center text-xs tabular-nums text-slate-400 sm:text-left">
                            {filteredTomes.length > 0 ? `${firstIndex + 1}–${Math.min(firstIndex + PAGE_SIZE, filteredTomes.length)} sur ${filteredTomes.length} ${filteredTomes.length === 1 ? 'tome' : 'tomes'}` : '0 tome'}
                            {query.trim() && filteredTomes.length > 0 ? (filteredTomes.length === 1 ? ' trouvé' : ' trouvés') : ''}
                        </p>
                        {totalPages > 1 && (
                            <nav aria-label="Pagination des tomes" className="flex items-center justify-center gap-1">
                                <Button type="button" variant="ghost" size="icon" aria-label="Page précédente" disabled={page === 1} onClick={() => changePage(page - 1)} className="size-11 text-slate-300 hover:bg-white/8 hover:text-white">
                                    <ChevronLeft aria-hidden="true" className="size-4" />
                                </Button>
                                <span className="min-w-20 text-center text-sm tabular-nums text-slate-300 sm:hidden">{page} / {totalPages}</span>
                                <div className="hidden items-center gap-1 sm:flex">
                                    {getPageNumbers(page, totalPages).map(value => typeof value === 'string' ? (
                                        <span key={value} aria-hidden="true" className="px-1 text-sm text-slate-500">…</span>
                                    ) : (
                                        <Button key={value} type="button" variant="ghost" size="icon" aria-label={`Page ${value}`} aria-current={value === page ? 'page' : undefined} onClick={() => changePage(value)} className={cn('size-10 rounded-md tabular-nums transition-colors hover:bg-white/8 hover:text-white', value === page ? 'bg-white/10 text-[#8dbbff]' : 'text-slate-400')}>
                                            {value}
                                        </Button>
                                    ))}
                                </div>
                                <Button type="button" variant="ghost" size="icon" aria-label="Page suivante" disabled={page === totalPages} onClick={() => changePage(page + 1)} className="size-11 text-slate-300 hover:bg-white/8 hover:text-white">
                                    <ChevronRight aria-hidden="true" className="size-4" />
                                </Button>
                            </nav>
                        )}
                    </div>
                )}
            </Card>
        </section>
    );
}
