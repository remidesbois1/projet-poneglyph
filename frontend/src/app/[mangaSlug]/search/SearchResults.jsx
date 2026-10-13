"use client";

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { AlertCircle, ArrowRight, BookOpen, Check, Loader2, RefreshCcw, Sparkles, ThumbsDown, ThumbsUp } from 'lucide-react';
import SearchPageImage from '@/components/SearchPageImage';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import { getSearchMode } from './searchModes';

const BUBBLE_GRID = 'grid grid-cols-1 gap-3 sm:grid-cols-2 sm:gap-4 lg:grid-cols-3 xl:grid-cols-4';
const PAGE_GRID = 'grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 lg:grid-cols-4 xl:grid-cols-5';
// The visual search returns three candidates at most: one row, kept at the size of the other page cards.
const CANDIDATE_GRID = 'grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4';
const CARD = 'group relative flex min-w-0 flex-col overflow-hidden rounded-xl border border-white/10 bg-white/[0.025] transition-colors duration-200 hover:border-(--mode)/40 hover:bg-white/[0.045] motion-reduce:transition-none';
const CARD_ENTRANCE = 'animate-in fade-in slide-in-from-bottom-1 duration-300 [animation-fill-mode:backwards] motion-reduce:animate-none';
const CARD_LINK = 'flex min-w-0 flex-1 rounded-xl outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-(--mode)/70';
const FOCUS = 'outline-none focus-visible:ring-2 focus-visible:ring-(--mode)/60';
const SKELETON = 'bg-white/8 motion-reduce:animate-none';

function parsePlace(context) {
    const match = String(context ?? '').match(/Tome\s+(\S+)\s+-\s+Chap\.\s+(\S+)\s+-\s+Page\s+(\S+)/);
    return match ? { tome: match[1], chapter: match[2], page: match[3] } : null;
}

function highlightTerm(text, term) {
    const clean = String(text ?? '').replace(/^\[Concept\]\s*/, '');
    const needle = term?.trim();
    if (!needle) return clean;
    const parts = clean.split(new RegExp(`(${needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'));
    // split() with a capture group alternates plain text and matches.
    return parts.map((part, index) => (index % 2 === 1
        ? <mark key={index} className="rounded-[3px] bg-(--mode)/25 px-0.5 font-semibold text-white">{part}</mark>
        : part));
}

/** Scales the bubble to fit the frame: small ones are enlarged a little, large ones shrink until they show whole. */
function fitBubble(coords, stage) {
    const width = Math.max(1, Number(coords.w) || 1);
    const height = Math.max(1, Number(coords.h) || 1);
    const scale = Math.min(1.75, (stage.w * 0.84) / width, (stage.h * 0.84) / height);
    return {
        scale,
        offset: `translate(${-Number(coords.x) * scale}px, ${-Number(coords.y) * scale}px) scale(${scale})`,
        box: {
            left: (stage.w - width * scale) / 2,
            top: (stage.h - height * scale) / 2,
            width: width * scale,
            height: height * scale,
        },
    };
}

/** The bubble alone, cut out of its page and centred in the frame. */
function BubbleCrop({ url, pageId, coords, className }) {
    const stageRef = useRef(null);
    const [stage, setStage] = useState(null);
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        const node = stageRef.current;
        if (!node || typeof ResizeObserver === 'undefined') return;
        const observer = new ResizeObserver(() => setStage({ w: node.clientWidth, h: node.clientHeight }));
        observer.observe(node);
        return () => observer.disconnect();
    }, []);

    const fit = stage && stage.w > 0 && stage.h > 0 ? fitBubble(coords, stage) : null;

    return (
        <div ref={stageRef} className={cn('relative overflow-hidden bg-[#030c16]/65', className)}>
            {fit && (
                // Until the page has loaded, the wrapper spans the frame so the loading or error message has room.
                <div
                    className={cn('absolute overflow-hidden transition-transform duration-300 group-hover:scale-[1.04] motion-reduce:transition-none', loaded ? 'rounded-md shadow-[0_3px_12px_#0006]' : 'inset-0')}
                    style={loaded ? fit.box : undefined}
                >
                    <SearchPageImage
                        url={url}
                        pageId={pageId}
                        alt=""
                        draggable={false}
                        onLoad={() => setLoaded(true)}
                        className={cn('absolute left-0 top-0 max-w-none origin-top-left transition-opacity duration-300', loaded ? 'opacity-100' : 'opacity-0')}
                        style={{ transform: fit.offset }}
                    />
                </div>
            )}
        </div>
    );
}

function PageThumbnail({ url, pageId, alt }) {
    const [landscape, setLandscape] = useState(false);
    return (
        <div className="relative aspect-[2/3] w-full overflow-hidden border-b border-white/6 bg-[#030c16]/65">
            <SearchPageImage
                url={url}
                pageId={pageId}
                thumbnail
                alt={alt}
                loading="lazy"
                onLoad={event => setLandscape(event.currentTarget.naturalWidth > event.currentTarget.naturalHeight)}
                className={cn(
                    'h-full w-full transition-transform duration-500 group-hover:scale-[1.03] motion-reduce:transition-none',
                    // A double-page spread would lose half its artwork if it were cropped to a portrait frame.
                    landscape ? 'object-contain' : 'object-cover object-top',
                )}
            />
        </div>
    );
}

function Place({ place, context, className }) {
    if (!place) return <p className={cn('text-xs leading-5 text-slate-400', className)}>{context}</p>;
    return (
        <p className={cn('flex items-center gap-1.5 text-xs leading-5 tabular-nums text-slate-400', className)}>
            <span>Tome {place.tome}</span>
            <span aria-hidden="true" className="text-slate-600">·</span>
            <span>Chap. {place.chapter}</span>
            <span className="ml-auto pl-2 font-medium text-slate-200">p. {place.page}</span>
        </p>
    );
}

function Score({ value }) {
    return (
        <span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium tabular-nums text-slate-200">
            <span aria-hidden="true" className={cn('size-1.5 rounded-full', value >= 80 ? 'bg-emerald-400' : 'bg-(--mode)')} />
            {value} %
        </span>
    );
}

function BubbleCard({ item, href, term, delay }) {
    return (
        <li className={cn(CARD, CARD_ENTRANCE)} style={{ animationDelay: delay }}>
            <Link href={href} prefetch={false} className={cn(CARD_LINK, 'flex-row sm:flex-col')}>
                <BubbleCrop url={item.url_image} pageId={item.page_id} coords={item.coords} className="min-h-32 w-[38%] max-w-44 shrink-0 border-r border-white/6 sm:aspect-[4/3] sm:w-full sm:max-w-none sm:border-b sm:border-r-0" />
                <span className="flex min-w-0 flex-1 flex-col gap-3 p-3.5 sm:p-4">
                    <span className="line-clamp-5 break-words text-sm leading-relaxed text-slate-100 sm:text-[15px]">
                        «&nbsp;{highlightTerm(item.content, term)}&nbsp;»
                    </span>
                    <Place place={parsePlace(item.context)} context={item.context} className="mt-auto" />
                </span>
            </Link>
        </li>
    );
}

function PageCard({ item, href, delay, feedback }) {
    const place = parsePlace(item.context);
    const score = item.similarity > 0 ? Math.round(item.similarity * 100) : null;
    const matchedCount = item.ocr?.matches?.length || 0;

    return (
        <li className={cn(CARD, CARD_ENTRANCE)} style={{ animationDelay: delay }}>
            <Link href={href} prefetch={false} className={cn(CARD_LINK, 'flex-col')}>
                <PageThumbnail url={item.url_image} pageId={item.page_id} alt={place ? `Tome ${place.tome}, chapitre ${place.chapter}, page ${place.page}` : 'Page trouvée'} />
                <span className="block min-w-0 p-3 sm:p-3.5">
                    <span className="flex items-baseline justify-between gap-2">
                        <span className="min-w-0 truncate font-serif text-lg font-black leading-snug text-white">
                            {place ? <>Tome <span className="tabular-nums">{place.tome}</span></> : item.context}
                        </span>
                        {score !== null && <Score value={score} />}
                    </span>
                    {place && <span className="mt-0.5 block text-xs leading-5 tabular-nums text-slate-400">Chapitre {place.chapter} · page {place.page}</span>}
                    {item.type === 'ocr' && matchedCount > 0 && (
                        <span className="mt-2 block text-xs leading-5 text-slate-300">
                            {matchedCount} {matchedCount === 1 ? 'bulle reconnue' : 'bulles reconnues'}
                        </span>
                    )}
                </span>
            </Link>

            {feedback && (
                <div className="flex min-h-10 items-center justify-between gap-2 border-t border-white/8 py-1 pl-3 pr-1.5 text-xs text-slate-400 sm:pl-3.5">
                    {feedback.given ? (
                        <span className="flex items-center gap-1.5 text-emerald-300"><Check aria-hidden="true" className="size-3.5" /> Merci pour votre retour</span>
                    ) : (
                        <>
                            <span>Pertinent ?</span>
                            <span className="flex gap-0.5">
                                <button type="button" aria-label="Résultat pertinent" title="Pertinent" onClick={event => feedback.onVote(event, item, true)} className={cn('flex size-8 cursor-pointer items-center justify-center rounded-md transition-colors hover:bg-emerald-400/10 hover:text-emerald-300', FOCUS)}>
                                    <ThumbsUp aria-hidden="true" className="size-3.5" />
                                </button>
                                <button type="button" aria-label="Résultat non pertinent" title="Non pertinent" onClick={event => feedback.onVote(event, item, false)} className={cn('flex size-8 cursor-pointer items-center justify-center rounded-md transition-colors hover:bg-red-400/10 hover:text-red-300', FOCUS)}>
                                    <ThumbsDown aria-hidden="true" className="size-3.5" />
                                </button>
                            </span>
                        </>
                    )}
                </div>
            )}
        </li>
    );
}

/** A confident visual match: the page with the recognised bubbles outlined, next to what was read in the uploaded image. */
function TopOcrResult({ item, href }) {
    const place = parsePlace(item.context);
    const matches = item.ocr?.matches || [];
    const [natural, setNatural] = useState(null);
    const [activeIndex, setActiveIndex] = useState(null);

    return (
        <article className="animate-in fade-in overflow-hidden rounded-2xl border border-(--mode)/25 bg-[#071321]/85 duration-300 motion-reduce:animate-none">
            <div className="grid lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
                <Link href={href} prefetch={false} aria-label="Ouvrir la page retrouvée" className={cn('group block border-white/10 bg-[#030c16]/65 p-4 sm:p-6 lg:border-r', FOCUS, 'focus-visible:ring-inset')}>
                    <span className="relative mx-auto block w-full max-w-sm overflow-hidden rounded-md shadow-[0_12px_40px_#0008]">
                        <SearchPageImage
                            url={item.url_image}
                            pageId={item.page_id}
                            alt={place ? `Page retrouvée : tome ${place.tome}, chapitre ${place.chapter}, page ${place.page}` : 'Page retrouvée'}
                            onLoad={event => setNatural({ w: event.currentTarget.naturalWidth, h: event.currentTarget.naturalHeight })}
                            className="block h-auto w-full"
                        />
                        {natural && matches.map((match, index) => match.coords && (
                            <span
                                key={`${match.bubble_id || index}-${index}`}
                                aria-hidden="true"
                                className={cn('absolute rounded-md border-2 transition-colors duration-200', activeIndex === index ? 'border-(--mode) bg-(--mode)/30' : 'border-(--mode)/80 bg-(--mode)/12')}
                                style={{
                                    left: `${(match.coords.x / natural.w) * 100}%`,
                                    top: `${(match.coords.y / natural.h) * 100}%`,
                                    width: `${(match.coords.w / natural.w) * 100}%`,
                                    height: `${(match.coords.h / natural.h) * 100}%`,
                                }}
                            >
                                <span className="absolute -left-2.5 -top-2.5 flex size-5 items-center justify-center rounded-full bg-(--mode-strong) text-[10px] font-bold tabular-nums text-(--mode-ink) shadow-[0_2px_8px_#0008]">{index + 1}</span>
                            </span>
                        ))}
                    </span>
                </Link>

                <div className="flex min-w-0 flex-col p-5 sm:p-7">
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-(--mode)">Page retrouvée</p>
                    <h2 className="mt-3 font-serif text-3xl font-black leading-tight text-white sm:text-4xl">
                        {place ? <>Tome <span className="tabular-nums">{place.tome}</span></> : item.context}
                    </h2>
                    {place && <p className="mt-2 text-sm tabular-nums text-slate-300">Chapitre {place.chapter} · page {place.page}</p>}

                    <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-4">
                        <div>
                            <dt className="text-xs text-slate-400">Meilleure bulle</dt>
                            <dd className="mt-1 text-2xl font-semibold tabular-nums text-white">{Math.round(matches[0].score * 100)} %</dd>
                        </div>
                        <div>
                            <dt className="text-xs text-slate-400">Bulles reconnues</dt>
                            <dd className="mt-1 text-2xl font-semibold tabular-nums text-white">{matches.length}</dd>
                        </div>
                    </dl>

                    <ol aria-label="Bulles reconnues" className="mt-6 divide-y divide-white/8 border-y border-white/8">
                        {matches.map((match, index) => (
                            <li
                                key={`${match.bubble_id || index}-${index}`}
                                onMouseEnter={() => setActiveIndex(index)}
                                onMouseLeave={() => setActiveIndex(null)}
                                className="flex gap-3.5 py-3.5"
                            >
                                <span aria-hidden="true" className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border border-(--mode)/45 text-[10px] font-bold tabular-nums text-(--mode)">{index + 1}</span>
                                <div className="min-w-0 flex-1">
                                    <p className="break-words text-[15px] leading-relaxed text-white">«&nbsp;{match.matched_text}&nbsp;»</p>
                                    {match.query_text && (
                                        <p className="mt-1 break-words text-xs leading-5 text-slate-400">Lu dans votre image : «&nbsp;{match.query_text}&nbsp;»</p>
                                    )}
                                </div>
                                <span className={cn('mt-1 shrink-0 text-xs font-medium tabular-nums', match.score >= 0.85 ? 'text-(--mode)' : 'text-slate-400')}>{Math.round(match.score * 100)} %</span>
                            </li>
                        ))}
                    </ol>

                    <div className="mt-auto pt-6">
                        <Link href={href} prefetch={false} className={cn('inline-flex h-11 items-center gap-2 rounded-lg bg-(--mode-strong) px-5 text-sm font-semibold text-(--mode-ink) shadow-[0_8px_24px_rgb(var(--mode-rgb)/0.24)] transition-colors hover:bg-(--mode-strong-hover)', FOCUS, 'focus-visible:ring-offset-2 focus-visible:ring-offset-[#071321]')}>
                            Ouvrir la page <ArrowRight aria-hidden="true" className="size-4" />
                        </Link>
                    </div>
                </div>
            </div>
        </article>
    );
}

function ResultSkeletons({ pages, count = 8 }) {
    return Array.from({ length: count }, (_, index) => (
        <li key={`skeleton-${index}`} aria-hidden="true" className={cn('flex min-w-0 overflow-hidden rounded-xl border border-white/8', pages ? 'flex-col' : 'flex-row sm:flex-col')}>
            <Skeleton className={cn('rounded-none', SKELETON, pages ? 'aspect-[2/3] w-full' : 'min-h-32 w-[38%] max-w-44 shrink-0 sm:aspect-[4/3] sm:w-full sm:max-w-none')} />
            <div className="flex-1 space-y-2.5 p-3.5 sm:p-4">
                <div className="h-4 w-4/5 rounded bg-white/8" />
                {!pages && <div className="h-4 w-3/5 rounded bg-white/8" />}
                <div className={cn('h-3 w-1/2 rounded bg-white/5', !pages && 'mt-5')} />
            </div>
        </li>
    ));
}

function ScanProgress({ status, progress }) {
    return (
        <div role="status" aria-live="polite" className="mx-auto flex min-h-64 max-w-md flex-col items-center justify-center px-4 py-12 text-center">
            <Loader2 aria-hidden="true" className="size-7 animate-spin text-(--mode) motion-reduce:animate-none" />
            <p className="mt-4 font-semibold text-slate-100">Lecture de votre image</p>
            <p className="mt-2 text-sm leading-6 text-slate-400">{status || 'Préparation…'}</p>
            <div className="mt-5 h-1.5 w-56 max-w-full overflow-hidden rounded-full bg-white/10">
                <div className="h-full rounded-full bg-(--mode) transition-[width] duration-300 motion-reduce:transition-none" style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
            </div>
        </div>
    );
}

function StateMessage({ icon: Icon, tone = 'text-slate-400', title, children, action, alert = false }) {
    return (
        <div role={alert ? 'alert' : undefined} className="flex min-h-64 flex-col items-center justify-center px-4 py-12 text-center animate-in fade-in duration-300 motion-reduce:animate-none">
            <Icon aria-hidden="true" className={cn('size-8', tone)} strokeWidth={1.5} />
            <p className="mt-4 text-lg font-semibold text-slate-100">{title}</p>
            <p className="mt-2 max-w-md break-words text-sm leading-6 text-slate-400">{children}</p>
            {action}
        </div>
    );
}

/** Everything under the console: examples, loading, results, empty and error states. `view` is decided by SearchClient. */
export default function SearchResults({ view, mode, mangaSlug, results, totalCount, term, topOcrResult, isLoadingMore, isStale, hasMore, onLoadMore, feedback, error, onRetry, onTrySemantic, examples, onExample, scan }) {
    const pages = mode.id !== 'keyword';
    const hrefFor = item => `/${mangaSlug}/annotate/${item.page_id}?from=search`;

    if (view === 'idle') {
        if (examples.length === 0) return null;
        return (
            <div className="mx-auto mt-6 flex max-w-3xl animate-in flex-wrap items-center justify-center gap-2 text-xs text-slate-400 fade-in duration-300 motion-reduce:animate-none">
                <span className="mr-1">Essayez :</span>
                {examples.map(label => (
                    <button key={label} type="button" onClick={() => onExample(label)} className={cn('cursor-pointer rounded-full border border-white/10 bg-white/[0.035] px-3 py-1.5 text-slate-300 transition-colors hover:border-(--mode)/45 hover:bg-(--mode)/15 hover:text-white', FOCUS)}>
                        {label}
                    </button>
                ))}
            </div>
        );
    }

    if (view === 'error') {
        return (
            <div className="mt-6">
                <StateMessage
                    alert
                    icon={AlertCircle}
                    tone="text-red-300"
                    title="La recherche n’a pas abouti"
                    action={(
                        <Button type="button" variant="outline" size="sm" onClick={onRetry} className="mt-5 gap-2 text-slate-200">
                            <RefreshCcw aria-hidden="true" className="size-4" /> Réessayer
                        </Button>
                    )}
                >
                    {error}
                </StateMessage>
            </div>
        );
    }

    if (view === 'empty') {
        return (
            <div className="mt-6">
                <StateMessage
                    icon={BookOpen}
                    title="Zone inconnue"
                    action={mode.id === 'keyword' && (
                        <button
                            type="button"
                            onClick={onTrySemantic}
                            style={getSearchMode('semantic').theme}
                            className={cn('mt-6 inline-flex h-11 cursor-pointer items-center gap-2 rounded-lg bg-(--mode-strong) px-5 text-sm font-semibold text-(--mode-ink) shadow-[0_8px_24px_rgb(var(--mode-rgb)/0.24)] transition-colors hover:bg-(--mode-strong-hover)', FOCUS)}
                        >
                            <Sparkles aria-hidden="true" className="size-4" /> Rechercher par concept
                        </button>
                    )}
                >
                    {mode.id === 'ocr' ? 'Aucune page proche dans nos archives.' : <>Aucune occurrence de «&nbsp;{term}&nbsp;» dans nos archives.</>}
                    {mode.id === 'keyword' && ' L’IA pourrait vous aider par analogie.'}
                </StateMessage>
            </div>
        );
    }

    if (view === 'loading' && mode.id === 'ocr') {
        return <div className="mt-6"><ScanProgress status={scan.status} progress={scan.progress} /></div>;
    }

    if (view === 'loading') {
        return (
            <div role="status" aria-label="Recherche en cours" className="mt-8">
                <span className="sr-only">Recherche en cours…</span>
                <div aria-hidden="true" className="mb-5 h-8 w-56 rounded bg-white/5" />
                <ul aria-hidden="true" className={pages ? PAGE_GRID : BUBBLE_GRID}><ResultSkeletons pages={pages} /></ul>
            </div>
        );
    }

    if (topOcrResult) {
        return <div className="mx-auto mt-8 max-w-6xl"><TopOcrResult item={topOcrResult} href={hrefFor(topOcrResult)} /></div>;
    }

    const noun = mode.id === 'keyword'
        ? (totalCount === 1 ? 'bulle' : 'bulles')
        : mode.id === 'ocr' ? (totalCount === 1 ? 'page candidate' : 'pages candidates') : (totalCount === 1 ? 'page' : 'pages');

    return (
        <section aria-label="Résultats" aria-busy={isStale || isLoadingMore} className={cn('mt-8', mode.id === 'ocr' && 'mx-auto max-w-4xl')}>
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 px-1">
                <p role="status" className="min-w-0 break-words text-sm text-slate-400">
                    <span className="mr-2 text-2xl font-semibold tabular-nums text-white">{totalCount}</span>
                    {noun}
                    {mode.id !== 'ocr' && term && <> pour <span className="text-slate-200">«&nbsp;{term}&nbsp;»</span></>}
                </p>
                {mode.id === 'ocr' && scan.bubbleCount > 0 && (
                    <p className="text-xs tabular-nums text-slate-400">{scan.bubbleCount} {scan.bubbleCount === 1 ? 'bulle lue' : 'bulles lues'} dans votre image</p>
                )}
                {mode.id === 'keyword' && results.length < totalCount && (
                    <p className="text-xs tabular-nums text-slate-400">{results.length} affichées</p>
                )}
            </div>

            <ul className={cn(mode.id === 'ocr' ? CANDIDATE_GRID : pages ? PAGE_GRID : BUBBLE_GRID, 'transition-opacity duration-200 motion-reduce:transition-none', isStale && 'opacity-50')}>
                {results.map((item, index) => {
                    const key = `${item.id}-${index}`;
                    // New cards arrive in a short cascade; cards loaded earlier keep their key and stay put.
                    const delay = `${(index % 24) * 18}ms`;
                    return item.coords && item.type !== 'semantic'
                        ? <BubbleCard key={key} item={item} href={hrefFor(item)} term={term} delay={delay} />
                        : <PageCard key={key} item={item} href={hrefFor(item)} delay={delay} feedback={feedback && { given: Boolean(feedback.given[item.id]), onVote: feedback.onVote }} />;
                })}
                {isLoadingMore && <ResultSkeletons pages={pages} />}
            </ul>

            {!isLoadingMore && hasMore && (
                <div className="mt-10 flex justify-center">
                    <button type="button" onClick={onLoadMore} className={cn('inline-flex h-12 cursor-pointer items-center justify-center gap-3 rounded-lg border border-white/16 bg-[#071625]/58 px-7 text-sm font-semibold text-slate-200 backdrop-blur-md transition hover:border-white/28 hover:bg-white/8 hover:text-white', FOCUS)}>
                        Déchiffrer la suite
                        <span className="text-xs font-normal tabular-nums text-slate-400">{results.length} / {totalCount}</span>
                    </button>
                </div>
            )}
        </section>
    );
}
