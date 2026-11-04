"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { getPageById, getBubblesForPage, deleteBubble, submitPageForReview, updatePageStatus, reorderBubbles, getPages } from '@/lib/api';
import AiAccessDialog from '@/components/AiAccessDialog';
import { useAuth } from '@/context/AuthContext';
import { useManga } from '@/context/MangaContext';
import { arrayMove } from '@dnd-kit/sortable';
import { useAnnotationInteractions } from '@/hooks/useAnnotationInteractions';
import { useAnnotationOCR } from '@/hooks/useAnnotationOCR';
import { useAnnotationDetection } from '@/hooks/useAnnotationDetection';
import { useAnnotationMetadata } from '@/hooks/useAnnotationMetadata';
import { getProxiedImageUrl } from '@/lib/utils';
import { fetchOriginalPageImage } from '@/lib/pageImageClient';
import { canCreateBubble, canEditBubble, canReorderBubbles } from '@/lib/bubblePermissions';
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog } from "@/components/ui/dialog";
import { AlertCircle, ArrowLeft, Send, X, Shield, FileText, Loader2, RefreshCcw } from "lucide-react";
import { toast } from "sonner";
import { Skeleton } from "@/components/ui/skeleton";
import AnnotateLeftSidebar from '@/components/AnnotateLeftSidebar';
import AnnotateCanvas from '@/components/AnnotateCanvas';
import AnnotateAnnotationSidebar from '@/components/AnnotateAnnotationSidebar';
import AnnotateEditorDialog from '@/components/AnnotateEditorDialog';
import AnnotateMetadataModal from '@/components/AnnotateMetadataModal';
import SubmitPageDialog from '@/components/SubmitPageDialog';

const PAGE_STATUSES = [
    { value: 'not_started', label: 'Non commencée' },
    { value: 'in_progress', label: 'En cours' },
    { value: 'pending_review', label: 'En revue' },
    { value: 'completed', label: 'Validée' },
];

const PREFETCH_BEHIND_COUNT = 1;
const PREFETCH_AHEAD_COUNT = 2;

const RESOURCE_STATUS = Object.freeze({
    IDLE: 'idle',
    LOADING: 'loading',
    READY: 'ready',
    ERROR: 'error',
});

function normalizePageId(id) {
    return id == null ? null : String(id);
}

function sortBubblesForAnnotation(bubbles = []) {
    return [...bubbles].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
}

function unwrapApiData(value) {
    return value?.data ?? value;
}

function getNearbyChapterPages(
    chapterPages,
    currentPageId,
    behindCount = PREFETCH_BEHIND_COUNT,
    aheadCount = PREFETCH_AHEAD_COUNT
) {
    const currentIndex = chapterPages.findIndex(p => normalizePageId(p.id) === normalizePageId(currentPageId));
    if (currentIndex === -1) return [];
    return chapterPages.slice(
        Math.max(0, currentIndex - behindCount),
        currentIndex + aheadCount + 1
    ).filter(p => normalizePageId(p.id) !== normalizePageId(currentPageId));
}

function AnnotationLoadingState({ message = 'Chargement de la page…' }) {
    return (
        <div className="-mx-4 -my-7 flex h-[calc(100%+3.5rem)] min-h-[60vh] overflow-hidden bg-[#030a13] sm:-mx-8 lg:-mx-10" role="status" aria-live="polite">
            <div className="hidden w-[280px] shrink-0 space-y-5 border-r border-white/10 bg-[#06111e] p-5 lg:block">
                <Skeleton className="h-8 w-36 bg-white/10" />
                <Skeleton className="h-20 w-full bg-white/10" />
                <Skeleton className="h-40 w-full bg-white/10" />
            </div>
            <div className="flex min-w-0 flex-1 flex-col">
                <div className="flex h-16 items-center justify-between border-b border-white/10 px-5">
                    <Skeleton className="h-8 w-44 bg-white/10" />
                    <span className="flex items-center gap-2 text-sm font-semibold text-slate-300">
                        <Loader2 className="h-4 w-4 animate-spin text-[#8dbbff]" />
                        {message}
                    </span>
                </div>
                <div className="flex min-h-0 flex-1 items-center justify-center p-6">
                    <Skeleton className="h-full min-h-[360px] w-full max-w-[760px] bg-white/10" />
                </div>
            </div>
        </div>
    );
}

function AnnotationErrorState({ title, message, onRetry }) {
    return (
        <div className="flex min-h-[60vh] items-center justify-center px-4" role="alert">
            <div className="max-w-md rounded-2xl border border-red-400/25 bg-red-950/20 p-6 text-center text-slate-100">
                <AlertCircle className="mx-auto h-8 w-8 text-red-300" />
                <h2 className="mt-3 text-lg font-bold">{title}</h2>
                <p className="mt-2 text-sm text-slate-300">{message}</p>
                <Button type="button" variant="outline" onClick={onRetry} className="mt-5 border-white/15 bg-white/10 text-white hover:bg-white/15 hover:text-white">
                    <RefreshCcw className="mr-2 h-4 w-4" />
                    Réessayer
                </Button>
            </div>
        </div>
    );
}

export default function AnnotatePage() {
    const { user, session, isGuest, role } = useAuth();
    const params = useParams();
    const router = useRouter();
    const searchParams = useSearchParams();
    const fromSearch = searchParams.get('from') === 'search';
    const detectionDebugEnabled = searchParams.get('debug') === '1';
    const paramsPageId = params?.pageId;
    const [pageId, setPageId] = useState(paramsPageId);
    const { mangaSlug, currentManga } = useManga();

    const [page, setPage] = useState(null);
    const [pageLoadStatus, setPageLoadStatus] = useState(RESOURCE_STATUS.LOADING);
    const [pageLoadError, setPageLoadError] = useState(null);
    const [pageRetryGeneration, setPageRetryGeneration] = useState(0);
    const [existingBubbles, setExistingBubbles] = useState([]);
    const [bubblesLoadStatus, setBubblesLoadStatus] = useState(RESOURCE_STATUS.LOADING);
    const [bubblesLoadError, setBubblesLoadError] = useState(null);
    const [imageLoadStatus, setImageLoadStatus] = useState(RESOURCE_STATUS.LOADING);
    const [imageLoadError, setImageLoadError] = useState(null);
    const [imageRetryGeneration, setImageRetryGeneration] = useState(0);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isUpdatingPageStatus, setIsUpdatingPageStatus] = useState(false);
    const [reviewPage, setReviewPage] = useState(null);
    const [loadingText, setLoadingText] = useState("Analyse en cours...");
    const [pendingAnnotation, setPendingAnnotation] = useState(null);
    const [rectangle, setRectangle] = useState(null);
    const [imageDimensions, setImageDimensions] = useState(null);
    const [ocrSource, setOcrSource] = useState(null);
    const [debugImageUrl, setDebugImageUrl] = useState(null);
    const [originalImageUrl, setOriginalImageUrl] = useState(null);
    const [detectionDebugData, setDetectionDebugData] = useState(null);
    const [showApiKeyModal, setShowApiKeyModal] = useState(false);
    const [showDescModal, setShowDescModal] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const canEdit = bubblesLoadStatus === RESOURCE_STATUS.READY
        && canCreateBubble({ page, user, role, isGuest });

    const containerRef = useRef(null);
    const imageRef = useRef(null);

    const [chapterPages, setChapterPages] = useState([]);
    const [navContext, setNavContext] = useState({ prev: null, next: null });
    const [isMobile, setIsMobile] = useState(false);
    const canReorder = canReorderBubbles({ page, user, role, isGuest, bubbles: existingBubbles });
    const canEditExistingBubble = useCallback(
        (bubble) => canEditBubble({ page, user, role, isGuest, bubble }),
        [page, user, role, isGuest]
    );

    const chapterPagesRef = useRef([]);
    chapterPagesRef.current = chapterPages;

    const pageIdRef = useRef(pageId);
    pageIdRef.current = pageId;

    const navGenerationRef = useRef(0);
    const pageCacheRef = useRef(new Map());
    const bubbleCacheRef = useRef(new Map());
    const imageBlobCacheRef = useRef(new Map());

    const beginPageTransition = useCallback(() => {
        setPage(null);
        setExistingBubbles([]);
        setOriginalImageUrl(null);
        setPageLoadStatus(RESOURCE_STATUS.LOADING);
        setPageLoadError(null);
        setBubblesLoadStatus(RESOURCE_STATUS.LOADING);
        setBubblesLoadError(null);
        setImageLoadStatus(RESOURCE_STATUS.LOADING);
        setImageLoadError(null);
    }, []);

    useEffect(() => {
        imageBlobCacheRef.current.clear();
    }, [session?.access_token]);

    useEffect(() => {
        if (paramsPageId && String(paramsPageId) !== String(pageIdRef.current)) {
            navGenerationRef.current += 1;
            beginPageTransition();
            setPageId(paramsPageId);
        }
    }, [beginPageTransition, paramsPageId]);

    useEffect(() => {
        if (!detectionDebugEnabled) {
            setDetectionDebugData(null);
        }
    }, [detectionDebugEnabled]);

    useEffect(() => {
        const handlePopState = () => {
            const pathParts = window.location.pathname.split('/');
            const urlPageId = pathParts[pathParts.length - 1];
            if (urlPageId && String(urlPageId) !== String(pageIdRef.current)) {
                navGenerationRef.current += 1;
                beginPageTransition();
                setPageId(urlPageId);
                setImageDimensions(null);
                setPendingAnnotation(null);
                setRectangle(null);
                setIsModalOpen(false);
                setDebugImageUrl(null);
                setDetectionDebugData(null);
            }
        };
        window.addEventListener('popstate', handlePopState);
        return () => window.removeEventListener('popstate', handlePopState);
    }, [beginPageTransition]);

    const navigateToPage = useCallback((newPageId) => {
        navGenerationRef.current += 1;
        beginPageTransition();
        setPageId(newPageId);
        setImageDimensions(null);
        window.history.pushState({}, '', `/${mangaSlug}/annotate/${newPageId}`);
        setPendingAnnotation(null);
        setRectangle(null);
        setIsModalOpen(false);
        setDebugImageUrl(null);
        setDetectionDebugData(null);
        const pages = chapterPagesRef.current;
        if (pages.length > 0) {
            const currentIndex = pages.findIndex(p => p.id === parseInt(newPageId));
            setNavContext({
                prev: currentIndex > 0 ? pages[currentIndex - 1] : null,
                next: currentIndex < pages.length - 1 ? pages[currentIndex + 1] : null
            });
        }
    }, [beginPageTransition, mangaSlug]);

    const cacheBubblesForPage = useCallback((targetPageId, bubbles) => {
        const cacheKey = normalizePageId(targetPageId);
        if (!cacheKey) return;
        bubbleCacheRef.current.set(cacheKey, sortBubblesForAnnotation(bubbles));
    }, []);

    const getOriginalPageBlob = useCallback((targetPageId) => {
        const token = session?.access_token;
        const cacheKey = `${normalizePageId(targetPageId)}:${token || 'none'}`;
        if (!token) return Promise.reject(new Error('Session manquante'));

        const cached = imageBlobCacheRef.current.get(cacheKey);
        if (cached instanceof Blob) return Promise.resolve(cached);
        if (cached) return cached;

        const request = fetchOriginalPageImage(targetPageId, token)
            .then((blob) => {
                imageBlobCacheRef.current.set(cacheKey, blob);
                return blob;
            })
            .catch((error) => {
                imageBlobCacheRef.current.delete(cacheKey);
                throw error;
            });
        imageBlobCacheRef.current.set(cacheKey, request);
        return request;
    }, [session?.access_token]);

    useEffect(() => {
        let active = true;
        let objectUrl = null;
        setOriginalImageUrl(null);
        setImageDimensions(null);
        setImageLoadError(null);
        setImageLoadStatus(RESOURCE_STATUS.LOADING);

        if (!pageId || !session?.access_token) return undefined;

        getOriginalPageBlob(pageId)
            .then((blob) => {
                if (!active) return;
                objectUrl = URL.createObjectURL(blob);
                setOriginalImageUrl(objectUrl);
            })
            .catch((imageError) => {
                if (!active) return;
                setImageLoadStatus(RESOURCE_STATUS.ERROR);
                setImageLoadError(imageError.message || "Impossible de charger l'image originale.");
            });

        return () => {
            active = false;
            if (objectUrl) URL.revokeObjectURL(objectUrl);
        };
    }, [getOriginalPageBlob, imageRetryGeneration, pageId, session?.access_token]);

    const retryImageLoad = useCallback(() => {
        const token = session?.access_token;
        const cacheKey = `${normalizePageId(pageId)}:${token || 'none'}`;
        imageBlobCacheRef.current.delete(cacheKey);
        setImageLoadError(null);
        setImageLoadStatus(RESOURCE_STATUS.LOADING);
        setImageRetryGeneration(generation => generation + 1);
    }, [pageId, session?.access_token]);

    const prefetchAnnotatePage = useCallback((targetPage) => {
        const targetPageId = normalizePageId(targetPage?.id);
        if (!targetPageId || !session?.access_token) return;

        if (mangaSlug) {
            router.prefetch(`/${mangaSlug}/annotate/${targetPageId}`);
        }

        if (!pageCacheRef.current.has(targetPageId)) {
            const pagePromise = getPageById(targetPageId)
                .then(response => {
                    pageCacheRef.current.set(targetPageId, response.data);
                    return response.data;
                })
                .catch(error => {
                    pageCacheRef.current.delete(targetPageId);
                    throw error;
                });
            pageCacheRef.current.set(targetPageId, pagePromise);
            pagePromise.catch(() => {});
        }

        if (!bubbleCacheRef.current.has(targetPageId)) {
            const bubblesPromise = getBubblesForPage(targetPageId)
                .then(response => {
                    const sortedBubbles = sortBubblesForAnnotation(response.data);
                    bubbleCacheRef.current.set(targetPageId, sortedBubbles);
                    return sortedBubbles;
                })
                .catch(error => {
                    bubbleCacheRef.current.delete(targetPageId);
                    throw error;
                });
            bubbleCacheRef.current.set(targetPageId, bubblesPromise);
            bubblesPromise.catch(() => {});
        }

        getOriginalPageBlob(targetPageId).catch(() => {});
    }, [getOriginalPageBlob, mangaSlug, router, session?.access_token]);

    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth < 1024);
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    const {
        formData, setFormData, suggestions, charInput, setCharInput,
        isSavingDesc, isGeneratingAI, tabMode, setTabMode, jsonInput,
        jsonError, handleJsonChange, handleSaveDescription, handleGenerateAI,
        addCharacter, removeCharacter
    } = useAnnotationMetadata({
        page, setPage, pageId, imageRef, showDescModal, setShowDescModal, setShowApiKeyModal
    });

    const {
        models: ocrModels, toggleOcrModel, prepareSelectedModels, refreshModelAccess,
        bubbleModels, pageModels, runPageModel, runLocalOcr, runBackgroundOcr,
        handleRetryWithCloud, handleRetryWithDeepSeek
    } = useAnnotationOCR({
        imageRef, pageId, rectangle, pendingAnnotation, setPendingAnnotation,
        setIsSubmitting, setLoadingText, setIsModalOpen, setOcrSource,
        setDebugImageUrl, setShowApiKeyModal, allowPageModels: !isGuest && role === 'Admin'
    });

    const {
        isAutoDetecting, setIsAutoDetecting, queueLength, detectionStatus,
        loadDetectionModel, detectionProgress, isPageOcrRunning, runPageOcr,
        cancelPageOcr, processNextBubble
    } = useAnnotationDetection({
        imageRef, pageId, existingBubbles, setRectangle, setPendingAnnotation, setDebugImageUrl,
        runLocalOcr, runBackgroundOcr, runPageModel, pageModels,
        detectionDebugEnabled, setDetectionDebugData
    });

    const prepareOcrModels = useCallback(() => {
        loadDetectionModel();
        prepareSelectedModels();
    }, [loadDetectionModel, prepareSelectedModels]);

    // Keys and the ChatGPT session are edited in the API dialog.
    useEffect(() => {
        if (!showApiKeyModal) void refreshModelAccess();
    }, [refreshModelAccess, showApiKeyModal]);

    const setExistingBubblesAndCache = useCallback((updater) => {
        setExistingBubbles(prev => {
            const nextBubbles = typeof updater === 'function' ? updater(prev) : updater;
            if (Array.isArray(nextBubbles)) {
                cacheBubblesForPage(pageIdRef.current, nextBubbles);
            }
            return nextBubbles;
        });
    }, [cacheBubblesForPage]);

    const {
        isDrawing, startPoint, endPoint, mousePos, isShiftPressed,
        hoveredBubble, setHoveredBubble, handleMouseDown, handleMouseMove,
        handleMouseUp, handleInteractionStart
    } = useAnnotationInteractions({
        containerRef, imageRef, imageDimensions, existingBubbles, setExistingBubbles: setExistingBubblesAndCache,
        pendingAnnotation, setPendingAnnotation, setRectangle, canEdit, canEditBubble: canEditExistingBubble, isMobile,
        pageStatus: page?.statut, isSubmitting: isSubmitting || isPageOcrRunning, showApiKeyModal, showDescModal
    });

    const fetchBubbles = useCallback(({ force = false } = {}) => {
        if (pageId && (session?.access_token || isGuest)) {
            const gen = navGenerationRef.current;
            const cacheKey = normalizePageId(pageId);
            if (force && cacheKey) bubbleCacheRef.current.delete(cacheKey);
            setBubblesLoadStatus(RESOURCE_STATUS.LOADING);
            setBubblesLoadError(null);
            const cachedBubbles = cacheKey ? bubbleCacheRef.current.get(cacheKey) : null;
            const bubblesRequest = cachedBubbles
                ? Promise.resolve(cachedBubbles).catch(() => {
                    bubbleCacheRef.current.delete(cacheKey);
                    return getBubblesForPage(pageId).then(response => response.data);
                })
                : getBubblesForPage(pageId).then(response => response.data);

            bubblesRequest
                .then(response => {
                    if (gen !== navGenerationRef.current) return;
                    const sortedBubbles = sortBubblesForAnnotation(unwrapApiData(response));
                    cacheBubblesForPage(pageId, sortedBubbles);
                    setExistingBubbles(sortedBubbles);
                    setBubblesLoadStatus(RESOURCE_STATUS.READY);
                })
                .catch(bubblesError => {
                    if (gen !== navGenerationRef.current) return;
                    setExistingBubbles([]);
                    setBubblesLoadStatus(RESOURCE_STATUS.ERROR);
                    setBubblesLoadError(bubblesError?.response?.data?.error || bubblesError?.message || 'Impossible de charger les annotations.');
                });
        }
    }, [cacheBubblesForPage, pageId, session?.access_token, isGuest]);

    const retryBubblesLoad = useCallback(() => {
        void fetchBubbles({ force: true });
    }, [fetchBubbles]);

    useEffect(() => {
        if (pageId && (session?.access_token || isGuest)) {
            const gen = navGenerationRef.current;
            const cacheKey = normalizePageId(pageId);
            setPageLoadStatus(RESOURCE_STATUS.LOADING);
            setPageLoadError(null);
            const cachedPage = cacheKey ? pageCacheRef.current.get(cacheKey) : null;
            const pageRequest = cachedPage
                ? Promise.resolve(cachedPage).catch(() => {
                    pageCacheRef.current.delete(cacheKey);
                    return getPageById(pageId).then(response => response.data);
                })
                : getPageById(pageId).then(response => response.data);

            pageRequest
                .then(response => {
                    if (gen !== navGenerationRef.current) return;
                    const pageData = unwrapApiData(response);
                    if (!pageData) throw new Error("Page non trouvée");
                    pageCacheRef.current.set(cacheKey, pageData);
                    setPage(pageData);
                    setPageLoadStatus(RESOURCE_STATUS.READY);
                    if (pageData.id_chapitre) {
                        getPages(pageData.id_chapitre)
                            .then(pagesRes => {
                                if (gen !== navGenerationRef.current) return;
                                const pages = pagesRes.data;
                                setChapterPages(pages);
                                const currentIndex = pages.findIndex(p => p.id === parseInt(pageId));
                                setNavContext({
                                    prev: currentIndex > 0 ? pages[currentIndex - 1] : null,
                                    next: currentIndex < pages.length - 1 ? pages[currentIndex + 1] : null
                                });
                            })
                            .catch(() => {
                                if (gen !== navGenerationRef.current) return;
                                setChapterPages([]);
                                setNavContext({ prev: null, next: null });
                            });
                    }
                })
                .catch((pageError) => {
                    if (gen !== navGenerationRef.current) return;
                    setPage(null);
                    setPageLoadStatus(RESOURCE_STATUS.ERROR);
                    setPageLoadError(pageError?.response?.data?.error || pageError?.message || "Impossible de charger la page.");
                });
            fetchBubbles();
        }
    }, [pageId, session?.access_token, isGuest, fetchBubbles, pageRetryGeneration]);

    const retryPageLoad = useCallback(() => {
        const cacheKey = normalizePageId(pageId);
        if (cacheKey) pageCacheRef.current.delete(cacheKey);
        setPage(null);
        setPageLoadError(null);
        setPageLoadStatus(RESOURCE_STATUS.LOADING);
        setPageRetryGeneration(generation => generation + 1);
    }, [pageId]);

    useEffect(() => {
        if (!pageId || chapterPages.length === 0 || !session?.access_token) return;
        getNearbyChapterPages(chapterPages, pageId).forEach(prefetchAnnotatePage);
    }, [chapterPages, pageId, prefetchAnnotatePage, session?.access_token]);

    useEffect(() => {
        if (detectionStatus === 'idle') {
            loadDetectionModel();
        }
    }, [detectionStatus, loadDetectionModel]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (reviewPage) return;
            if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
                if (e.key === 'Escape') {
                    if (pendingAnnotation) setPendingAnnotation(null);
                    if (showDescModal) setShowDescModal(false);
                    if (showApiKeyModal) setShowApiKeyModal(false);
                }
                return;
            }
            switch (e.key) {
                case 'ArrowLeft':
                    if (!isGuest && navContext.prev) navigateToPage(navContext.prev.id);
                    break;
                case 'ArrowRight':
                    if (!isGuest && navContext.next) navigateToPage(navContext.next.id);
                    break;
                case 'Escape':
                    if (isDrawing) { /* dealt with in hook but can be here too */ }
                    if (pendingAnnotation) setPendingAnnotation(null);
                    break;
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [navContext, navigateToPage, pendingAnnotation, showDescModal, showApiKeyModal, mangaSlug, isDrawing, isGuest, reviewPage]);

    const goToPrev = useCallback(() => navContext.prev && navigateToPage(navContext.prev.id), [navContext.prev, navigateToPage]);
    const goToNext = useCallback(() => navContext.next && navigateToPage(navContext.next.id), [navContext.next, navigateToPage]);

    useEffect(() => {
        if (isAutoDetecting) return;
        if (rectangle && imageRef.current) {
            const analysisData = {
                id: `manual-${Date.now()}`,
                id_page: parseInt(pageId, 10), 
                ...rectangle, 
                texte_propose: '' 
            };
            setPendingAnnotation(analysisData);
            setDebugImageUrl(null);
            runLocalOcr(rectangle);
        }
    }, [rectangle, pageId, isAutoDetecting]);

    const handleSaveApiKey = (key) => {
        localStorage.setItem('google_api_key', key);
        setShowApiKeyModal(false);
        if (pendingAnnotation) handleRetryWithCloud();
        if (showDescModal) handleSaveDescription();
    };

    const handleEditBubble = (bubble) => {
        if (!canEdit || !canEditExistingBubble(bubble) || isMobile) return;
        setPendingAnnotation(bubble);
        setIsModalOpen(true);
    };

    const handleDeleteBubble = async (bubbleId) => {
        if (isGuest || (isMobile && role !== 'Admin')) return;
        if (window.confirm("Supprimer cette annotation ?")) {
            const previousBubbles = [...existingBubbles];
            setExistingBubblesAndCache(prev => prev.filter(b => b.id !== bubbleId));
            try {
                await deleteBubble(bubbleId);
                toast.success("Annotation supprimée.");
            } catch (error) {
                setExistingBubblesAndCache(previousBubbles);
                toast.error("Erreur lors de la suppression.");
            }
        }
    };

    const handleSuccess = (newData, tempId = null) => {
        const isOptimistic = !!newData?.isOptimistic;
        const isBackgroundResult = !!tempId;

        if (isOptimistic) {
            setPendingAnnotation(null);
            setDebugImageUrl(null);
            setIsModalOpen(false);
            if (isAutoDetecting) {
                setTimeout(() => processNextBubble(), 100);
            } else {
                setRectangle(null);
            }
        }

        if (newData) {
            setExistingBubblesAndCache(prev => {
                const results = [...prev];
                const idx = results.findIndex(b => b.id === newData.id || (tempId && b.id === tempId));

                if (idx !== -1) {
                    results[idx] = { ...results[idx], ...newData };
                    if (isBackgroundResult) results[idx].isOptimistic = false;
                } else if (!isBackgroundResult) {
                    results.push(newData);
                }

                return results.sort((a, b) => a.order - b.order);
            });
        }
    };

    const commitPageUpdate = useCallback((targetPageId, nextPage) => {
        const targetKey = normalizePageId(targetPageId);
        if (!targetKey || !nextPage) return;

        const cachedPage = pageCacheRef.current.get(targetKey);
        const mergedPage = cachedPage ? { ...cachedPage, ...nextPage } : nextPage;
        pageCacheRef.current.set(targetKey, mergedPage);
        setChapterPages(previousPages => previousPages.map(chapterPage => (
            normalizePageId(chapterPage.id) === targetKey
                ? { ...chapterPage, ...mergedPage }
                : chapterPage
        )));

        if (normalizePageId(pageIdRef.current) === targetKey) {
            setPage(previousPage => previousPage ? { ...previousPage, ...mergedPage } : mergedPage);
        }
    }, []);

    const handleSubmitPage = () => {
        if (isGuest || isMobile || !page) return;
        setReviewPage({ id: pageId, number: page.numero_page });
    };

    const confirmSubmitPage = async () => {
        const response = await submitPageForReview(reviewPage.id);
        commitPageUpdate(reviewPage.id, response.data);
        toast.success("Page soumise pour validation !");
    };

    const handlePageStatusChange = async (statut) => {
        if (role !== 'Admin' || !pageId || statut === page?.statut) return;

        const targetPageId = pageId;
        setIsUpdatingPageStatus(true);
        try {
            const response = await updatePageStatus(targetPageId, statut);
            commitPageUpdate(targetPageId, response.data);
            toast.success("Statut de la page mis à jour.");
        } catch (error) {
            toast.error("Erreur lors du changement de statut.");
        } finally {
            setIsUpdatingPageStatus(false);
        }
    };

    const handleDragEnd = (event) => {
        if (!canReorder || !pageId) return;
        const { active, over } = event;
        if (active && over && active.id !== over.id) {
            setExistingBubblesAndCache((bubbles) => {
                const oldIndex = bubbles.findIndex(b => b.id === active.id);
                const newIndex = bubbles.findIndex(b => b.id === over.id);
                const newOrder = arrayMove(bubbles, oldIndex, newIndex);
                const orderedBubblesForApi = newOrder.map((b, index) => ({ id: b.id, order: index + 1 }));
                reorderBubbles(pageId, orderedBubblesForApi).catch(() => fetchBubbles());
                return newOrder;
            });
        }
    };

    if (pageLoadStatus === RESOURCE_STATUS.ERROR) {
        return (
            <AnnotationErrorState
                title="Page indisponible"
                message={pageLoadError || 'Impossible de charger cette page.'}
                onRetry={retryPageLoad}
            />
        );
    }
    if (pageLoadStatus !== RESOURCE_STATUS.READY || !page) {
        return <AnnotationLoadingState message="Chargement des métadonnées…" />;
    }
    if (imageLoadStatus === RESOURCE_STATUS.ERROR) {
        return (
            <AnnotationErrorState
                title="Image indisponible"
                message={imageLoadError || "Impossible de charger l'image de la page."}
                onRetry={retryImageLoad}
            />
        );
    }
    if (session?.access_token && !originalImageUrl) {
        return <AnnotationLoadingState message="Chargement sécurisé de l’image…" />;
    }

    return (
        <div className="relative -mx-4 -my-7 flex h-[calc(100%+3.5rem)] flex-col overflow-hidden bg-[#030a13] sm:-mx-8 lg:-mx-10 lg:flex-row">
            <AnnotateLeftSidebar
                fromSearch={fromSearch}
                mangaSlug={mangaSlug}
                page={page}
                chapterPages={chapterPages}
                navContext={navContext}
                goToPrev={goToPrev}
                goToNext={goToNext}
                isGuest={isGuest}
                role={role}
                ocrModels={ocrModels}
                toggleOcrModel={toggleOcrModel}
                onPrepareModels={prepareOcrModels}
                detectionStatus={detectionStatus}
                detectionProgress={detectionProgress}
                canRunPageOcr={canEdit && imageLoadStatus === RESOURCE_STATUS.READY}
                isPageOcrRunning={isPageOcrRunning}
                isReviewingPage={isAutoDetecting}
                reviewRemaining={queueLength}
                onRunPageOcr={runPageOcr}
                onCancelPageOcr={cancelPageOcr}
                isSubmitting={isSubmitting}
                setShowDescModal={setShowDescModal}
                setShowApiKeyModal={setShowApiKeyModal}
                handleSubmitPage={handleSubmitPage}
                handlePageStatusChange={handlePageStatusChange}
                isUpdatingPageStatus={isUpdatingPageStatus}
            />

            <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#030a13]">

                <header className="z-20 flex h-auto min-h-16 flex-none items-center justify-between border-b border-white/10 bg-[#06111e] px-4 py-3 shadow-sm lg:hidden">
                    <div className="flex items-center gap-3 shrink-0">
                        <Link href={`/${mangaSlug}/dashboard`}>
                            <Button variant="ghost" size="icon" className="h-9 w-9">
                                <ArrowLeft className="h-5 w-5" />
                            </Button>
                        </Link>
                        <div className="flex flex-col">
                            <h2 className="max-w-[120px] truncate text-sm font-bold text-white">
                                T.{page.chapitres?.tomes?.numero} - Ch.{page.chapitres?.numero}
                            </h2>
                            <span className="text-[10px] text-slate-500">Page {page.numero_page}</span>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {!isGuest && (role === 'Admin' || role === 'Modo') && (
                            <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => setShowDescModal(true)}>
                                <FileText size={16} />
                            </Button>
                        )}
                        <Button variant="default" size="sm" className="h-9" disabled={isPageOcrRunning || page.statut === 'pending_review' || page.statut === 'completed' || isGuest || role === 'User'} onClick={handleSubmitPage}>
                            <Send size={14} className="mr-2" /> Soumettre
                        </Button>
                    </div>
                </header>

                {!isGuest && role === 'Admin' && (
                    <div className="lg:hidden border-b border-slate-200 bg-white px-4 py-3">
                        <Select value={page.statut} onValueChange={handlePageStatusChange} disabled={isUpdatingPageStatus}>
                            <SelectTrigger className="w-full h-9 bg-white border-slate-200 text-[12px] font-bold text-slate-700">
                                <SelectValue placeholder="Choisir un état" />
                            </SelectTrigger>
                            <SelectContent>
                                {PAGE_STATUSES.map(status => (
                                    <SelectItem key={status.value} value={status.value}>
                                        {status.label}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                )}

                {bubblesLoadStatus === RESOURCE_STATUS.LOADING && (
                    <div className="flex items-center justify-center gap-2 border-b border-sky-300/20 bg-sky-400/10 px-5 py-2 text-xs font-semibold text-sky-100" role="status" aria-live="polite">
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Chargement des annotations…
                    </div>
                )}

                {bubblesLoadStatus === RESOURCE_STATUS.ERROR && (
                    <div className="flex flex-wrap items-center justify-center gap-3 border-b border-amber-300/25 bg-amber-400/10 px-5 py-2 text-xs font-semibold text-amber-100" role="alert">
                        <AlertCircle className="h-4 w-4" />
                        <span>{bubblesLoadError || 'Impossible de charger les annotations. Les modifications sont désactivées.'}</span>
                        <Button type="button" variant="outline" size="sm" onClick={retryBubblesLoad} className="h-7 border-amber-200/30 bg-white/10 px-2 text-[11px] text-amber-50 hover:bg-white/15 hover:text-white">
                            <RefreshCcw className="mr-1.5 h-3 w-3" />
                            Réessayer
                        </Button>
                    </div>
                )}

                {isGuest && (
                    <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 flex items-center justify-center gap-2 text-amber-800 text-sm font-medium">
                        <Shield className="h-4 w-4" />
                        Mode Visiteur : Modification et navigation limitées. Connectez-vous pour tout débloquer.
                    </div>
                )}

                {page.commentaire_moderation && page.statut !== 'completed' && (
                    <div className="bg-red-50 border-b border-red-200 px-6 py-3 flex items-center gap-3 text-red-800 text-sm animate-in slide-in-from-top duration-300">
                        <div className="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center shrink-0">
                            <X className="h-4 w-4 text-red-600" />
                        </div>
                        <div className="flex-1">
                            <p className="font-bold">Cette page a été refusée par la modération</p>
                            <p className="text-red-700/80 italic font-medium">&quot;{page.commentaire_moderation}&quot;</p>
                        </div>
                    </div>
                )}

                <div className="flex flex-col lg:flex-row flex-1 overflow-hidden min-h-0">
                    <AnnotateCanvas
                        imageKey={`${normalizePageId(pageId)}:${imageRetryGeneration}`}
                        canEdit={canEdit}
                        canEditBubble={canEditExistingBubble}
                        imageDimensions={imageDimensions}
                        setImageDimensions={setImageDimensions}
                        containerRef={containerRef}
                        imageRef={imageRef}
                        handleMouseDown={handleMouseDown}
                        handleMouseMove={handleMouseMove}
                        handleMouseUp={handleMouseUp}
                        imageUrl={session?.access_token
                            ? originalImageUrl
                            : getProxiedImageUrl(page.url_image, pageId)}
                        isImageLoading={imageLoadStatus === RESOURCE_STATUS.LOADING}
                        onImageLoad={() => {
                            setImageLoadError(null);
                            setImageLoadStatus(RESOURCE_STATUS.READY);
                        }}
                        onImageError={() => {
                            setImageLoadStatus(RESOURCE_STATUS.ERROR);
                            setImageLoadError("Impossible d’afficher l’image de la page.");
                        }}
                        isSubmitting={isSubmitting || isPageOcrRunning}
                        loadingText={isPageOcrRunning ? 'Lecture de la page…' : loadingText}
                        rectangle={rectangle}
                        pendingAnnotation={pendingAnnotation}
                        isAutoDetecting={isAutoDetecting}
                        isShiftPressed={isShiftPressed}
                        handleInteractionStart={handleInteractionStart}
                        setIsModalOpen={setIsModalOpen}
                        isDrawing={isDrawing}
                        startPoint={startPoint}
                        endPoint={endPoint}
                        existingBubbles={existingBubbles}
                        setHoveredBubble={setHoveredBubble}
                        hoveredBubble={hoveredBubble}
                        mousePos={mousePos}
                        handleEditBubble={handleEditBubble}
                        detectionDebugEnabled={detectionDebugEnabled}
                        detectionDebugData={detectionDebugData}
                    />

                    <AnnotateAnnotationSidebar
                        existingBubbles={existingBubbles}
                        handleDragEnd={handleDragEnd}
                        user={user}
                        handleEditBubble={handleEditBubble}
                        handleDeleteBubble={handleDeleteBubble}
                        canEdit={canEdit}
                        canEditBubble={canEditExistingBubble}
                        canReorder={canReorder}
                        role={role}
                    />
                </div>
            </div>

            {reviewPage && <SubmitPageDialog
                pageNumber={reviewPage.number}
                onClose={() => setReviewPage(null)}
                onConfirm={confirmSubmitPage}
            />}
            <AnnotateEditorDialog
                isOpen={isModalOpen}
                setIsModalOpen={setIsModalOpen}
                setIsSubmitting={setIsSubmitting}
                isAutoDetecting={isAutoDetecting}
                setIsAutoDetecting={setIsAutoDetecting}
                setPendingAnnotation={setPendingAnnotation}
                setDebugImageUrl={setDebugImageUrl}
                setRectangle={setRectangle}
                pendingAnnotation={pendingAnnotation}
                ocrSource={ocrSource}
                handleSuccess={handleSuccess}
                processNextBubble={processNextBubble}
                debugImageUrl={debugImageUrl}
                runLocalOcr={runLocalOcr}
                handleRetryWithCloud={handleRetryWithCloud}
                handleRetryWithDeepSeek={handleRetryWithDeepSeek}
                isSubmitting={isSubmitting || isPageOcrRunning}
                ocrModelCount={bubbleModels.length}
            />

            <Dialog open={showApiKeyModal} onOpenChange={setShowApiKeyModal}>
                <AiAccessDialog onSave={handleSaveApiKey} onSaveDeepSeek={() => setShowApiKeyModal(false)} />
            </Dialog>

            {!isGuest && (
                <AnnotateMetadataModal
                    isOpen={showDescModal}
                    onOpenChange={setShowDescModal}
                    tabMode={tabMode}
                    setTabMode={setTabMode}
                    formData={formData}
                    setFormData={setFormData}
                    charInput={charInput}
                    setCharInput={setCharInput}
                    suggestions={suggestions}
                    isGeneratingAI={isGeneratingAI}
                    handleGenerateAI={handleGenerateAI}
                    handleSaveDescription={handleSaveDescription}
                    isSavingDesc={isSavingDesc}
                    jsonInput={jsonInput}
                    handleJsonChange={handleJsonChange}
                    jsonError={jsonError}
                />
            )}
        </div>
    );
}
