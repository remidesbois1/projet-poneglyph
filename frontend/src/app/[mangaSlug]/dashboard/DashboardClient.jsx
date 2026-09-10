"use client";

import React, { useCallback, useRef, useState, useEffect } from 'react';
import { useManga } from '@/context/MangaContext';
import { useAuth } from '@/context/AuthContext';
import { useUserProfile } from '@/hooks/useUserProfile';
import { useRouter } from 'next/navigation';
import { deleteBubblesForChapter, deleteBubblesForPage, getTomes, getChapitres, getPages } from '@/lib/api';
import VolumeLibrary from './VolumeLibrary';
import VolumeDrawerContent from './VolumeDrawerContent';
import { toast } from 'sonner';

import { Sheet, SheetContent } from '@/components/ui/sheet';

const LOAD_STATUS = Object.freeze({
    LOADING: 'loading',
    READY: 'ready',
    EMPTY: 'empty',
    ERROR: 'error',
});

const DRAWER_STATUS = Object.freeze({
    CLOSED: 'closed',
    LOADING_CHAPTERS: 'loading-chapters',
    CHAPTERS_READY: 'chapters-ready',
    CHAPTERS_EMPTY: 'chapters-empty',
    LOADING_PAGES: 'loading-pages',
    PAGES_READY: 'pages-ready',
    PAGES_EMPTY: 'pages-empty',
    ERROR_CHAPTERS: 'error-chapters',
    ERROR_PAGES: 'error-pages',
});

export default function DashboardPage() {
    const { mangaSlug, currentManga } = useManga();

    // A different manga starts with its own catalogue and drawer, never the previous one's data.
    return <MangaDashboard key={mangaSlug} mangaSlug={mangaSlug} currentManga={currentManga} />;
}

function MangaDashboard({ mangaSlug, currentManga }) {
    const { profile } = useUserProfile();
    const { session } = useAuth();
    const router = useRouter();

    const [tomes, setTomes] = useState([]);
    const [chapters, setChapters] = useState([]);
    const [pages, setPages] = useState([]);

    const [isSheetOpen, setIsSheetOpen] = useState(false);
    const [selectedTome, setSelectedTome] = useState(null);
    const [selectedChapter, setSelectedChapter] = useState(null);
    const [catalogState, setCatalogState] = useState({ status: LOAD_STATUS.LOADING, error: null });
    const [drawerState, setDrawerState] = useState({ status: DRAWER_STATUS.CLOSED, error: null });
    const [deletingTarget, setDeletingTarget] = useState(null);
    const drawerRef = useRef(null);
    const drawerOpenerRef = useRef(null);
    const drawerDragStartYRef = useRef(null);
    const drawerDragYRef = useRef(0);
    const drawerDragFrameRef = useRef(null);
    const catalogRequestRef = useRef(0);
    const drawerRequestRef = useRef(0);
    const sheetCleanupTimerRef = useRef(null);
    const isAdmin = profile?.role === 'Admin';

    const loadTomes = useCallback(() => {
        if (!mangaSlug) return;
        const requestId = ++catalogRequestRef.current;
        return getTomes(mangaSlug).then((response) => {
            if (requestId !== catalogRequestRef.current) return;
            const nextTomes = Array.isArray(response.data) ? response.data : [];
            setTomes(nextTomes);
            setCatalogState({
                status: nextTomes.length > 0 ? LOAD_STATUS.READY : LOAD_STATUS.EMPTY,
                error: null,
            });
        }).catch((error) => {
            if (requestId !== catalogRequestRef.current) return;
            setTomes([]);
            setCatalogState({
                status: LOAD_STATUS.ERROR,
                error: error?.response?.data?.error || error?.message || 'Impossible de charger les volumes.',
            });
        });
    }, [mangaSlug]);

    useEffect(() => {
        void loadTomes();
        return () => {
            catalogRequestRef.current += 1;
        };
    }, [loadTomes]);

    useEffect(() => {
        if (isSheetOpen) {
            const drawer = drawerRef.current;
            if (drawer) {
                drawer.style.transform = '';
                drawer.style.transition = '';
                drawer.style.willChange = '';
            }
            drawerDragYRef.current = 0;
            drawerDragStartYRef.current = null;
        }
    }, [isSheetOpen]);

    useEffect(() => {
        return () => {
            if (drawerDragFrameRef.current) {
                cancelAnimationFrame(drawerDragFrameRef.current);
            }
            if (sheetCleanupTimerRef.current) {
                clearTimeout(sheetCleanupTimerRef.current);
            }
            drawerRequestRef.current += 1;
        };
    }, []);

    const applyDrawerDrag = (dragY) => {
        drawerDragYRef.current = dragY;

        if (drawerDragFrameRef.current) return;

        drawerDragFrameRef.current = requestAnimationFrame(() => {
            drawerDragFrameRef.current = null;
            const drawer = drawerRef.current;
            if (!drawer) return;

            drawer.style.transition = 'none';
            drawer.style.transform = `translate3d(0, ${drawerDragYRef.current}px, 0)`;
        });
    };

    const handleDrawerPointerDown = (event) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;

        drawerDragStartYRef.current = event.clientY;
        drawerDragYRef.current = 0;
        const drawer = drawerRef.current;
        if (drawer) {
            drawer.style.transition = 'none';
            drawer.style.willChange = 'transform';
        }
        event.currentTarget.setPointerCapture(event.pointerId);
    };

    const handleDrawerPointerMove = (event) => {
        if (drawerDragStartYRef.current === null) return;

        const rawDragY = event.clientY - drawerDragStartYRef.current;
        const dragY = rawDragY <= 0
            ? Math.max(-14, rawDragY * 0.14)
            : rawDragY > 420
                ? 420 + (rawDragY - 420) * 0.28
                : rawDragY;

        applyDrawerDrag(dragY);
    };

    const handleDrawerPointerUp = (event) => {
        if (drawerDragStartYRef.current === null) return;

        if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }

        if (drawerDragFrameRef.current) {
            cancelAnimationFrame(drawerDragFrameRef.current);
            drawerDragFrameRef.current = null;
        }

        const drawer = drawerRef.current;
        const shouldClose = event.type !== 'pointercancel' && drawerDragYRef.current > 130;

        drawerDragStartYRef.current = null;
        drawerDragYRef.current = 0;

        if (shouldClose) {
            handleSheetChange(false);
            return;
        }

        if (drawer) {
            drawer.style.transition = 'transform 180ms cubic-bezier(0.22, 1, 0.36, 1)';
            drawer.style.transform = 'translate3d(0, 0, 0)';
            window.setTimeout(() => {
                if (drawerDragStartYRef.current !== null || !drawerRef.current) return;
                drawer.style.transform = '';
                drawer.style.transition = '';
                drawer.style.willChange = '';
            }, 190);
        }
    };

    const openTome = async (tome) => {
        if (!isSheetOpen) drawerOpenerRef.current = document.activeElement;
        if (sheetCleanupTimerRef.current) {
            clearTimeout(sheetCleanupTimerRef.current);
            sheetCleanupTimerRef.current = null;
        }
        const requestId = ++drawerRequestRef.current;
        setSelectedTome(tome);
        setSelectedChapter(null);
        setChapters([]);
        setPages([]);
        setIsSheetOpen(true);
        setDrawerState({ status: DRAWER_STATUS.LOADING_CHAPTERS, error: null });

        try {
            const res = await getChapitres(tome.id);
            if (requestId !== drawerRequestRef.current) return;
            const nextChapters = Array.isArray(res.data) ? res.data : [];
            setChapters(nextChapters);
            setDrawerState({
                status: nextChapters.length > 0 ? DRAWER_STATUS.CHAPTERS_READY : DRAWER_STATUS.CHAPTERS_EMPTY,
                error: null,
            });
        } catch (e) {
            if (requestId !== drawerRequestRef.current) return;
            setChapters([]);
            setDrawerState({
                status: DRAWER_STATUS.ERROR_CHAPTERS,
                error: e?.response?.data?.error || e?.message || 'Impossible de charger les chapitres.',
            });
        }
    };

    const openChapter = async (chapter) => {
        const requestId = ++drawerRequestRef.current;
        setSelectedChapter(chapter);
        setPages([]);
        setDrawerState({ status: DRAWER_STATUS.LOADING_PAGES, error: null });
        try {
            const res = await getPages(chapter.id);
            if (requestId !== drawerRequestRef.current) return;
            const nextPages = Array.isArray(res.data) ? res.data : [];
            setPages(nextPages);
            setDrawerState({
                status: nextPages.length > 0 ? DRAWER_STATUS.PAGES_READY : DRAWER_STATUS.PAGES_EMPTY,
                error: null,
            });
        } catch (e) {
            if (requestId !== drawerRequestRef.current) return;
            setPages([]);
            setDrawerState({
                status: DRAWER_STATUS.ERROR_PAGES,
                error: e?.response?.data?.error || e?.message || 'Impossible de charger les pages.',
            });
        }
    };

    const returnToChapters = () => {
        drawerRequestRef.current += 1;
        setSelectedChapter(null);
        setPages([]);
        setDrawerState({
            status: chapters.length > 0 ? DRAWER_STATUS.CHAPTERS_READY : DRAWER_STATUS.CHAPTERS_EMPTY,
            error: null,
        });
    };

    const retryDrawerLoad = () => {
        if (drawerState.status === DRAWER_STATUS.ERROR_CHAPTERS && selectedTome) {
            void openTome(selectedTome);
        } else if (drawerState.status === DRAWER_STATUS.ERROR_PAGES && selectedChapter) {
            void openChapter(selectedChapter);
        }
    };

    const handleSheetChange = (open) => {
        setIsSheetOpen(open);
        if (open) {
            if (sheetCleanupTimerRef.current) {
                clearTimeout(sheetCleanupTimerRef.current);
                sheetCleanupTimerRef.current = null;
            }
        } else {
            drawerRequestRef.current += 1;
            // Keep the current content intact until the existing closing animation finishes.
            if (sheetCleanupTimerRef.current) clearTimeout(sheetCleanupTimerRef.current);
            sheetCleanupTimerRef.current = setTimeout(() => {
                setSelectedTome(null);
                setSelectedChapter(null);
                setChapters([]);
                setPages([]);
                setDrawerState({ status: DRAWER_STATUS.CLOSED, error: null });
                sheetCleanupTimerRef.current = null;
            }, 300);
        }
    };

    const handleDeletePageBubbles = async (page) => {
        if (!isAdmin || deletingTarget) return false;
        const requestId = drawerRequestRef.current;
        const target = `page-${page.id}`;
        setDeletingTarget(target);
        try {
            const { data } = await deleteBubblesForPage(page.id);
            if (requestId === drawerRequestRef.current) {
                setPages(prev => prev.map(item => item.id === page.id ? { ...item, statut: 'not_started' } : item));
            }
            toast.success(`${data?.deleted || 0} bulle(s) supprimée(s) sur la page ${page.numero_page}.`);
            return true;
        } catch (error) {
            toast.error(error?.response?.data?.error || "Suppression des bulles de la page impossible.");
            return false;
        } finally {
            setDeletingTarget(null);
        }
    };

    const handleDeleteChapterBubbles = async (chapter) => {
        if (!isAdmin || !chapter || deletingTarget) return false;
        const requestId = drawerRequestRef.current;
        const target = `chapter-${chapter.id}`;
        setDeletingTarget(target);
        try {
            const { data } = await deleteBubblesForChapter(chapter.id);
            if (requestId === drawerRequestRef.current) {
                setPages(prev => prev.map(page => ({ ...page, statut: 'not_started' })));
                setChapters(prev => prev.map(item => item.id === chapter.id ? { ...item, global_status: 'empty' } : item));
                setSelectedChapter(prev => prev?.id === chapter.id ? { ...prev, global_status: 'empty' } : prev);
            }
            toast.success(`${data?.deleted || 0} bulle(s) supprimée(s) sur le chapitre ${chapter.numero}.`);
            return true;
        } catch (error) {
            toast.error(error?.response?.data?.error || "Suppression des bulles du chapitre impossible.");
            return false;
        } finally {
            setDeletingTarget(null);
        }
    };

    return (
        <div className="h-full min-h-0 w-full">
            <VolumeLibrary
                mangaTitle={currentManga?.titre || 'Poneglyph'}
                tomes={tomes}
                status={catalogState.status}
                error={catalogState.error}
                onRetry={() => {
                    setCatalogState({ status: LOAD_STATUS.LOADING, error: null });
                    void loadTomes();
                }}
                onOpenTome={openTome}
            />

            <Sheet open={isSheetOpen} onOpenChange={handleSheetChange}>
                <SheetContent
                    ref={drawerRef}
                    side="bottom"
                    onOpenAutoFocus={event => {
                        event.preventDefault();
                        drawerRef.current?.querySelector('[data-drawer-title]')?.focus({ preventScroll: true });
                    }}
                    onCloseAutoFocus={event => {
                        event.preventDefault();
                        if (drawerOpenerRef.current?.isConnected) drawerOpenerRef.current.focus({ preventScroll: true });
                    }}
                    className="mx-auto h-[min(82vh,760px)] w-[calc(100%-1.5rem)] max-w-[1460px] gap-0 overflow-hidden rounded-t-[28px] border border-white/14 bg-[#06111e]/96 p-0 text-slate-100 shadow-[0_-22px_80px_rgba(0,0,0,0.48)] backdrop-blur-xl sm:w-[calc(100%-4rem)] [&>button:last-child]:top-0.5 [&>button:last-child]:right-2 [&>button:last-child]:flex [&>button:last-child]:size-11 [&>button:last-child]:items-center [&>button:last-child]:justify-center [&>button:last-child]:rounded-md"
                >

                    <div
                        data-drawer-handle
                        className="flex h-11 shrink-0 touch-none cursor-grab items-center justify-center active:cursor-grabbing"
                        onPointerDown={handleDrawerPointerDown}
                        onPointerMove={handleDrawerPointerMove}
                        onPointerUp={handleDrawerPointerUp}
                        onPointerCancel={handleDrawerPointerUp}
                    >
                        <span className="h-1.5 w-16 rounded-full bg-white/24" />
                    </div>

                    <VolumeDrawerContent
                        key={selectedTome?.id}
                        tome={selectedTome}
                        mangaTitle={currentManga?.titre || 'Poneglyph'}
                        chapter={selectedChapter}
                        chapters={chapters}
                        pages={pages}
                        state={drawerState}
                        isPublicViewer={!session}
                        isAdmin={isAdmin}
                        deletingTarget={deletingTarget}
                        onOpenChapter={openChapter}
                        onReturnToChapters={returnToChapters}
                        onRetry={retryDrawerLoad}
                        onOpenPage={page => router.push(`/${mangaSlug}/annotate/${page.id}`)}
                        onDeletePage={handleDeletePageBubbles}
                        onDeleteChapter={handleDeleteChapterBubbles}
                    />

                </SheetContent>
            </Sheet>
        </div>
    );
}
