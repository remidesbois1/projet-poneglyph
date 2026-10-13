"use client";

import React, { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import { searchBubbles, searchF2llmLocal, searchOcrPageMatch, getMetadataSuggestions, getTomes, submitSearchFeedback } from '@/lib/api';
import { generateF2llmBrowserQueryEmbedding } from '@/lib/f2llmBrowserEmbedding';
import {
    createAbortError,
    createSearchRequestLifecycle,
    isAbortError,
    throwIfAborted,
} from '@/lib/searchRequestLifecycle';
import { useDebounce } from '@/hooks/useDebounce';
import { useDetection } from '@/context/DetectionContext';
import { useWorker } from '@/context/WorkerContext';
import { useManga } from '@/context/MangaContext';
import { toast } from "sonner";
import SearchConsole from './SearchConsole';
import { SearchBackdrop, SearchHero, SearchModeSwitch } from './SearchHero';
import SearchResults from './SearchResults';
import { SEMANTIC_EXAMPLES, getSearchMode } from './searchModes';

const RESULTS_PER_PAGE = 24;
const OCR_RESULTS_LIMIT = 3;
const OCR_SEARCH_PROVIDER = 'readernet+ppocrv6';

const waitForCondition = (predicate, timeoutMs, errorMessage, signal) => {
    const started = Date.now();
    return new Promise((resolve, reject) => {
        let timer = null;
        const cleanup = () => {
            if (timer !== null) window.clearTimeout(timer);
            signal?.removeEventListener('abort', handleAbort);
        };
        const handleAbort = () => {
            cleanup();
            reject(createAbortError());
        };
        const tick = () => {
            if (signal?.aborted) {
                handleAbort();
                return;
            }
            if (predicate()) {
                cleanup();
                resolve();
                return;
            }
            if (Date.now() - started >= timeoutMs) {
                cleanup();
                reject(new Error(errorMessage));
                return;
            }
            timer = window.setTimeout(tick, 100);
        };
        signal?.addEventListener('abort', handleAbort, { once: true });
        tick();
    });
};

const clampDetectedBox = (box, imageWidth, imageHeight) => {
    const pad = 4;
    const x = Math.max(0, Math.floor(Number(box.x || 0)) - pad);
    const y = Math.max(0, Math.floor(Number(box.y || 0)) - pad);
    const right = Math.min(imageWidth, Math.ceil(Number(box.x || 0) + Number(box.w || 0)) + pad);
    const bottom = Math.min(imageHeight, Math.ceil(Number(box.y || 0) + Number(box.h || 0)) + pad);
    const w = Math.max(0, right - x);
    const h = Math.max(0, bottom - y);
    return { x, y, w, h };
};

export default function SearchPage({ initialQuery = '', initialMode = '' }) {
    const { mangaSlug } = useManga();
    const {
        detectBubbles,
        detectionStatus,
        loadDetectionModel,
    } = useDetection();
    const {
        worker: ocrWorker,
        modelStatus: ocrModelStatus,
        activeModelKey: activeOcrModelKey,
    } = useWorker();
    const getSavedState = () => {
        if (typeof window === 'undefined') return {};
        try {
            const saved = sessionStorage.getItem(`search_state_${mangaSlug}`);
            return saved ? JSON.parse(saved) : {};
        } catch { return {}; }
    };

    const savedState = getSavedState();
    const landingQuery = initialQuery.trim();
    const isLandingSemanticSearch = initialMode === 'semantic' && landingQuery.length >= 2;
    const savedOcrProvider = savedState.ocrProvider === OCR_SEARCH_PROVIDER ? savedState.ocrProvider : null;
    const savedSearchMode = isLandingSemanticSearch ? 'semantic' : savedState.searchMode || (savedState.useSemantic ? 'semantic' : 'keyword');
    const canRestoreResults = savedSearchMode !== 'ocr' || Boolean(savedOcrProvider);
    const restoredResults = canRestoreResults ? savedState.results || [] : [];
    const startingQuery = landingQuery || savedState.query || '';
    // Same conditions as the search effect below: the very first render already shows the search as running.
    const searchesOnMount = isLandingSemanticSearch
        || (savedSearchMode === 'keyword' && restoredResults.length === 0 && startingQuery.trim().length >= 2);

    const [query, setQuery] = useState(startingQuery);
    const debouncedQuery = useDebounce(query, 400);

    const [results, setResults] = useState(restoredResults);
    const [totalCount, setTotalCount] = useState(canRestoreResults ? savedState.totalCount || 0 : 0);
    const [isLoading, setIsLoading] = useState(searchesOnMount);
    // The term the displayed results (or their absence) answer, which the field may already have moved on from.
    const [settledTerm, setSettledTerm] = useState(
        restoredResults.length > 0 ? String(savedState.settledTerm ?? savedState.query ?? '').trim() : null
    );
    const [searchError, setSearchError] = useState(null);
    const [page, setPage] = useState(canRestoreResults ? savedState.page || 1 : 1);
    const [hasMore, setHasMore] = useState(canRestoreResults ? savedState.hasMore || false : false);
    const [searchMode, setSearchMode] = useState(savedSearchMode);
    const [localOnly, setLocalOnly] = useState(savedState.localOnly || false);
    const useSemantic = searchMode === 'semantic';
    const useOcrSearch = searchMode === 'ocr';
    const [feedbackGiven, setFeedbackGiven] = useState({});
    const [ocrImageFile, setOcrImageFile] = useState(null);
    const [ocrImagePreviewUrl, setOcrImagePreviewUrl] = useState(null);
    const [ocrExtractedBubbles, setOcrExtractedBubbles] = useState(savedOcrProvider ? savedState.ocrExtractedBubbles || [] : []);
    const [ocrProvider, setOcrProvider] = useState(savedOcrProvider);
    const [ocrStatus, setOcrStatus] = useState('');
    const [localModelStatus, setLocalModelStatus] = useState('');
    // Share of the current background task that is done (model download, bubbles read), or null when unknown.
    const [progress, setProgress] = useState(null);
    const [isDraggingImage, setIsDraggingImage] = useState(false);
    const [ocrHasSearched, setOcrHasSearched] = useState(savedOcrProvider ? savedState.ocrHasSearched || false : false);

    const [selectedCharacters, setSelectedCharacters] = useState(savedState.selectedCharacters || []);
    const [selectedArc, setSelectedArc] = useState(savedState.selectedArc || 'all');
    const [selectedTome, setSelectedTome] = useState(savedState.selectedTome || 'all');

    const [characterSuggestions, setCharacterSuggestions] = useState([]);
    const [arcSuggestions, setArcSuggestions] = useState([]);
    const [tomes, setTomes] = useState([]);

    const searchLifecycleRef = useRef(null);
    if (!searchLifecycleRef.current) {
        searchLifecycleRef.current = createSearchRequestLifecycle();
    }
    const inputRef = useRef(null);
    const scrollRef = useRef(null);
    const consoleAnchorRef = useRef(null);
    const isFirstRun = useRef(true);
    const autoSemanticSearchRef = useRef(isLandingSemanticSearch);
    const detectionStatusRef = useRef(detectionStatus);
    const ocrModelStatusRef = useRef(ocrModelStatus);
    const activeOcrModelKeyRef = useRef(activeOcrModelKey);
    const ocrWorkerRef = useRef(ocrWorker);
    const activeLocalSearchRequestIdRef = useRef(null);

    const invalidateActiveSearch = useCallback(() => {
        searchLifecycleRef.current.invalidate();
        activeLocalSearchRequestIdRef.current = null;
        setIsLoading(false);
        setSearchError(null);
        setProgress(null);
    }, []);

    const clearSearchResults = useCallback(() => {
        setResults([]);
        setTotalCount(0);
        setHasMore(false);
        setSettledTerm(null);
    }, []);

    // A new search started from the sticky console: bring the first results back under it without moving the field.
    const showFirstResults = useCallback(() => {
        const scroller = scrollRef.current;
        const anchor = consoleAnchorRef.current;
        if (scroller && anchor && scroller.scrollTop > anchor.offsetTop) scroller.scrollTop = anchor.offsetTop;
    }, []);

    // Same phase as the search effect below, and declared before it: when the manga changes, the previous search is
    // dropped first and the new one starts after. As a passive effect it would run later and cancel the new search.
    useLayoutEffect(() => () => {
        searchLifecycleRef.current.invalidate();
        activeLocalSearchRequestIdRef.current = null;
    }, [mangaSlug]);

    useEffect(() => {
        detectionStatusRef.current = detectionStatus;
    }, [detectionStatus]);

    useEffect(() => {
        ocrModelStatusRef.current = ocrModelStatus;
    }, [ocrModelStatus]);

    useEffect(() => {
        activeOcrModelKeyRef.current = activeOcrModelKey;
    }, [activeOcrModelKey]);

    useEffect(() => {
        ocrWorkerRef.current = ocrWorker;
    }, [ocrWorker]);

    // Persistence Effect
    useEffect(() => {
        if (!mangaSlug) return;
        const state = {
            query, settledTerm, results, totalCount, page, hasMore, useSemantic, searchMode, localOnly,
            ocrExtractedBubbles, ocrProvider, ocrHasSearched,
            selectedCharacters, selectedArc, selectedTome
        };
        sessionStorage.setItem(`search_state_${mangaSlug}`, JSON.stringify(state));
    }, [query, settledTerm, results, totalCount, page, hasMore, useSemantic, searchMode, localOnly, ocrExtractedBubbles, ocrProvider, ocrHasSearched, selectedCharacters, selectedArc, selectedTome, mangaSlug]);

    useEffect(() => {
        return () => {
            if (ocrImagePreviewUrl) URL.revokeObjectURL(ocrImagePreviewUrl);
        };
    }, [ocrImagePreviewUrl]);

    useEffect(() => {
        const handleProgress = (event) => {
            const requestId = activeLocalSearchRequestIdRef.current;
            if (!requestId || !searchLifecycleRef.current.isCurrent(requestId)) return;
            const percentage = Math.round(event.detail?.progress || 0);
            setLocalModelStatus(`Chargement de F2LLM · ${percentage} %`);
            setProgress(percentage / 100);
        };

        window.addEventListener('f2llm-progress', handleProgress);
        return () => window.removeEventListener('f2llm-progress', handleProgress);
    }, []);

    useEffect(() => {
        if (inputRef.current) inputRef.current.focus();
        const controller = new AbortController();

        const fetchMetadata = async () => {
            try {
                const [metadataRes, tomesRes] = await Promise.all([
                    getMetadataSuggestions(mangaSlug, { signal: controller.signal }),
                    getTomes(mangaSlug, { signal: controller.signal })
                ]);
                if (controller.signal.aborted) return;
                setCharacterSuggestions(metadataRes.data.characters || []);
                setArcSuggestions(metadataRes.data.arcs || []);
                setTomes(tomesRes.data || []);
            } catch (err) {
                if (!isAbortError(err, controller.signal)) {
                    console.error('Erreur chargement metadata:', err);
                }
            }
        };
        fetchMetadata();
        return () => controller.abort();
    }, [mangaSlug]);

    // A layout effect, so the search starts before the browser paints the frame in between (no flash of the idle guide).
    useLayoutEffect(() => {
        if (useSemantic) {
            if (!mangaSlug) return;
            if (autoSemanticSearchRef.current && debouncedQuery.trim().length >= 2) {
                autoSemanticSearchRef.current = false;
                setPage(1);
                fetchResults(debouncedQuery, 1, true);
            }
            return;
        }
        if (useOcrSearch) return;

        if (isFirstRun.current) {
            isFirstRun.current = false;
            // Si on a déjà des résultats restaurés, on ne déclenche pas la recherche initiale
            if (results.length > 0) return;
        }

        if (debouncedQuery.trim().length >= 2) {
            setPage(1);
            fetchResults(debouncedQuery, 1, true);
        } else {
            searchLifecycleRef.current.invalidate();
            setIsLoading(false);
            setResults([]);
            setTotalCount(0);
            setHasMore(false);
        }
    }, [mangaSlug, debouncedQuery, useSemantic, useOcrSearch, selectedCharacters, selectedArc, selectedTome]);

    const getActiveFilters = () => ({
        characters: selectedCharacters,
        arc: selectedArc !== 'all' ? selectedArc : '',
        tome: selectedTome !== 'all' ? selectedTome : ''
    });

    const normalizeOcrBubbles = (bubbles) => (Array.isArray(bubbles) ? bubbles : [])
        .map((bubble) => {
            const content = String(bubble?.content || bubble?.text || bubble?.texte_propose || '').trim();
            const bbox = bubble?.bbox || bubble?.pos || null;
            return content ? { content, bbox } : null;
        })
        .filter(Boolean);

    const ensureReaderNetReady = async (request) => {
        throwIfAborted(request.signal);
        if (detectionStatusRef.current === 'ready') return;
        searchLifecycleRef.current.commit(request.requestId, () => {
            setOcrStatus("Chargement de ReaderNet…");
        });
        loadDetectionModel();
        await waitForCondition(
            () => detectionStatusRef.current === 'ready',
            120000,
            "Le pipeline ReaderNet n’a pas pu être chargé.",
            request.signal
        );
        throwIfAborted(request.signal);
    };

    const ensurePpocrReady = async (request) => {
        throwIfAborted(request.signal);
        if (ocrWorkerRef.current && ocrModelStatusRef.current === 'ready' && activeOcrModelKeyRef.current === 'ppocrv6Line') return;
        searchLifecycleRef.current.commit(request.requestId, () => {
            setOcrStatus("Chargement de PP-OCRv6…");
        });
        await waitForCondition(
            () => Boolean(ocrWorkerRef.current),
            10000,
            "Worker PP-OCRv6 indisponible.",
            request.signal
        );
        throwIfAborted(request.signal);
        ocrWorkerRef.current.postMessage({ type: 'init', modelKey: 'ppocrv6Line' });
        await waitForCondition(
            () => ocrWorkerRef.current && ocrModelStatusRef.current === 'ready' && activeOcrModelKeyRef.current === 'ppocrv6Line',
            120000,
            "PP-OCRv6 n’a pas pu être chargé.",
            request.signal
        );
        throwIfAborted(request.signal);
    };

    const runPpocrSearchCrop = (imageBitmap, workerRequestId, signal) => new Promise((resolve, reject) => {
        const worker = ocrWorkerRef.current;
        if (!worker) {
            imageBitmap?.close?.();
            reject(new Error("Worker PP-OCRv6 indisponible."));
            return;
        }

        let settled = false;
        let timeout = null;
        const cleanup = () => {
            if (timeout !== null) window.clearTimeout(timeout);
            worker.removeEventListener('message', handleMessage);
            signal?.removeEventListener('abort', handleAbort);
        };
        const settle = (callback, value) => {
            if (settled) return;
            settled = true;
            cleanup();
            callback(value);
        };
        const handleAbort = () => {
            worker.postMessage({ type: 'cancel', requestId: workerRequestId });
            settle(reject, createAbortError());
        };

        const handleMessage = (event) => {
            const data = event.data || {};
            if (data.requestId !== workerRequestId) return;
            if (data.status === 'complete') {
                settle(resolve, data);
            } else {
                settle(reject, new Error(data.error || "Erreur PP-OCRv6."));
            }
        };

        if (signal?.aborted) {
            imageBitmap?.close?.();
            settle(reject, createAbortError());
            return;
        }

        timeout = window.setTimeout(() => {
            settle(reject, new Error("Timeout PP-OCRv6 sur une bulle."));
        }, 90000);
        worker.addEventListener('message', handleMessage);
        signal?.addEventListener('abort', handleAbort, { once: true });
        worker.postMessage({ type: 'run', imageBitmap, requestId: workerRequestId }, [imageBitmap]);
    });

    const extractOcrBubblesFromImage = async (imageFile, request) => {
        if (!imageFile) throw new Error("Image manquante.");

        await ensureReaderNetReady(request);
        searchLifecycleRef.current.commit(request.requestId, () => {
            setOcrStatus("Détection des bulles…");
        });
        const boxes = await detectBubbles(imageFile, { signal: request.signal });
        throwIfAborted(request.signal);
        if (!boxes?.length) {
            throw new Error("Aucune bulle détectée par le pipeline ReaderNet.");
        }

        await ensurePpocrReady(request);
        const pageBitmap = await createImageBitmap(imageFile);
        const bubbles = [];

        try {
            throwIfAborted(request.signal);
            for (let index = 0; index < boxes.length; index += 1) {
                throwIfAborted(request.signal);
                const bbox = clampDetectedBox(boxes[index], pageBitmap.width, pageBitmap.height);
                if (bbox.w <= 0 || bbox.h <= 0) continue;

                searchLifecycleRef.current.commit(request.requestId, () => {
                    setOcrStatus(`Lecture des bulles · ${index + 1} / ${boxes.length}`);
                    setProgress(index / boxes.length);
                });
                const crop = await createImageBitmap(pageBitmap, bbox.x, bbox.y, bbox.w, bbox.h);
                throwIfAborted(request.signal);
                const result = await runPpocrSearchCrop(
                    crop,
                    `search-ppocr-${request.requestId}-${index}`,
                    request.signal
                );
                throwIfAborted(request.signal);
                const content = String(result?.text || '').trim();
                if (!content) continue;

                bubbles.push({
                    content,
                    bbox,
                    score: boxes[index]?.score || boxes[index]?.conf || null,
                    lineCount: result?.lineCount || null,
                });
            }
        } finally {
            pageBitmap.close?.();
        }

        return {
            provider: OCR_SEARCH_PROVIDER,
            bubbles: normalizeOcrBubbles(bubbles),
            rawText: bubbles.map(bubble => bubble.content).join('\n'),
        };
    };

    const loadOcrImageFile = useCallback((file) => {
        if (!file) return;
        if (!file.type.startsWith('image/')) {
            toast.error("Image invalide.");
            return;
        }

        invalidateActiveSearch();
        if (ocrImagePreviewUrl) URL.revokeObjectURL(ocrImagePreviewUrl);
        setOcrImageFile(file);
        setOcrImagePreviewUrl(URL.createObjectURL(file));
        setOcrExtractedBubbles([]);
        setOcrProvider(null);
        setOcrStatus('');
        setOcrHasSearched(false);
        clearSearchResults();
    }, [clearSearchResults, invalidateActiveSearch, ocrImagePreviewUrl]);

    const clearOcrImage = () => {
        invalidateActiveSearch();
        if (ocrImagePreviewUrl) URL.revokeObjectURL(ocrImagePreviewUrl);
        setOcrImageFile(null);
        setOcrImagePreviewUrl(null);
        setOcrExtractedBubbles([]);
        setOcrProvider(null);
        setOcrStatus('');
        setOcrHasSearched(false);
        clearSearchResults();
    };

    // In visual mode the whole page takes an image, pasted or dropped. Without this, a drop next to the field would navigate away to the file.
    useEffect(() => {
        if (!useOcrSearch) return;

        const handlePaste = (event) => {
            const items = Array.from(event.clipboardData?.items || []);
            const imageItem = items.find(item => item.type?.startsWith('image/'));
            const file = imageItem?.getAsFile();
            if (!file) return;

            event.preventDefault();
            loadOcrImageFile(file);
        };
        const carriesFiles = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');
        const handleDragOver = (event) => {
            if (!carriesFiles(event)) return;
            event.preventDefault();
            setIsDraggingImage(true);
        };
        const handleDragLeave = (event) => {
            if (!event.relatedTarget) setIsDraggingImage(false);
        };
        const handleDrop = (event) => {
            if (!carriesFiles(event)) return;
            event.preventDefault();
            setIsDraggingImage(false);
            loadOcrImageFile(event.dataTransfer.files?.[0]);
        };

        window.addEventListener('paste', handlePaste);
        window.addEventListener('dragover', handleDragOver);
        window.addEventListener('dragleave', handleDragLeave);
        window.addEventListener('drop', handleDrop);
        return () => {
            window.removeEventListener('paste', handlePaste);
            window.removeEventListener('dragover', handleDragOver);
            window.removeEventListener('dragleave', handleDragLeave);
            window.removeEventListener('drop', handleDrop);
            setIsDraggingImage(false);
        };
    }, [useOcrSearch, loadOcrImageFile]);

    const runOcrSearch = async (pageToFetch = 1, isNewSearch = true) => {
        if (!ocrImageFile && (!ocrExtractedBubbles || ocrExtractedBubbles.length === 0)) {
            toast.error("Ajoutez une image pour lancer la recherche OCR.");
            return;
        }

        activeLocalSearchRequestIdRef.current = null;
        const request = searchLifecycleRef.current.begin();
        setIsLoading(true);
        setSearchError(null);
        setProgress(null);
        if (isNewSearch) {
            setResults([]);
            setFeedbackGiven({});
            showFirstResults();
        }

        try {
            let extracted = {
                provider: ocrProvider === OCR_SEARCH_PROVIDER ? ocrProvider : OCR_SEARCH_PROVIDER,
                bubbles: ocrExtractedBubbles,
                rawText: ocrExtractedBubbles.map(bubble => bubble.content).join('\n'),
            };

            if (ocrImageFile && (isNewSearch || !extracted.bubbles?.length)) {
                extracted = await extractOcrBubblesFromImage(ocrImageFile, request);
                throwIfAborted(request.signal);
                if (!extracted.bubbles.length) {
                    throw new Error("Aucune bulle lisible détectée dans l’image.");
                }
                searchLifecycleRef.current.commit(request.requestId, () => {
                    setOcrExtractedBubbles(extracted.bubbles);
                    setOcrProvider(extracted.provider);
                });
            }

            searchLifecycleRef.current.commit(request.requestId, () => {
                setOcrStatus("Recherche de la page…");
                setProgress(1);
            });
            const response = await searchOcrPageMatch({
                bubbles: extracted.bubbles,
                page: pageToFetch,
                limit: OCR_RESULTS_LIMIT,
                filters: getActiveFilters(),
                provider: extracted.provider,
                rawText: extracted.rawText,
                signal: request.signal,
            });
            throwIfAborted(request.signal);

            const newResults = (response.data.results || []).slice(0, OCR_RESULTS_LIMIT);
            const total = response.data.totalCount || 0;

            searchLifecycleRef.current.commit(request.requestId, () => {
                setResults(prev => isNewSearch ? newResults : [...prev, ...newResults]);
                setTotalCount(newResults.length);
                setHasMore(false);
                setOcrHasSearched(true);
                setOcrStatus(`${extracted.bubbles.length} bulles lues · ${newResults.length} meilleures pages sur ${total} classées`);
            });
        } catch (err) {
            if (searchLifecycleRef.current.isCurrent(request.requestId) && !isAbortError(err, request.signal)) {
                const message = err?.response?.data?.error || err?.message || "Recherche OCR impossible.";
                toast.error(message);
                setOcrStatus(message);
                setSearchError(message);
            }
        } finally {
            if (searchLifecycleRef.current.commit(request.requestId, () => { setIsLoading(false); setProgress(null); })) {
                searchLifecycleRef.current.finish(request.requestId);
            }
        }
    };

    const handleManualSearch = () => {
        if (useOcrSearch) {
            setPage(1);
            runOcrSearch(1, true);
            return;
        }
        if (query.trim().length < 2) return;
        setPage(1);
        fetchResults(query, 1, true);
    };

    async function fetchResults(searchTerm, pageToFetch, isNewSearch) {
        activeLocalSearchRequestIdRef.current = null;
        const request = searchLifecycleRef.current.begin();

        setIsLoading(true);
        setSearchError(null);
        setProgress(null);
        if (isNewSearch) {
            setResults([]);
            setFeedbackGiven({});
            showFirstResults();
        }

        try {
            const filters = getActiveFilters();
            let response;
            if (useSemantic && localOnly) {
                activeLocalSearchRequestIdRef.current = request.requestId;
                searchLifecycleRef.current.commit(request.requestId, () => {
                    setLocalModelStatus('Préparation de F2LLM…');
                });
                const embedding = await generateF2llmBrowserQueryEmbedding(searchTerm, {
                    signal: request.signal,
                });
                throwIfAborted(request.signal);
                searchLifecycleRef.current.commit(request.requestId, () => {
                    setLocalModelStatus('F2LLM prêt');
                    setProgress(null);
                });
                response = await searchF2llmLocal({
                    query: searchTerm,
                    embedding,
                    page: pageToFetch,
                    limit: RESULTS_PER_PAGE,
                    filters,
                    signal: request.signal,
                });
            } else {
                response = await searchBubbles(
                    searchTerm,
                    pageToFetch,
                    RESULTS_PER_PAGE,
                    useSemantic ? 'semantic' : 'keyword',
                    filters,
                    useSemantic && !localOnly,
                    false,
                    { signal: request.signal }
                );
            }
            throwIfAborted(request.signal);

            const newResults = response.data.results || [];
            const total = response.data.totalCount || 0;

            searchLifecycleRef.current.commit(request.requestId, () => {
                setResults(prev => isNewSearch ? newResults : [...prev, ...newResults]);
                setSettledTerm(searchTerm.trim());

                if (useSemantic && pageToFetch === 1) {
                    setTotalCount(newResults.length);
                    setHasMore(false);
                } else {
                    setTotalCount(total);
                    const previousCount = isNewSearch ? 0 : results.length;
                    setHasMore(previousCount + newResults.length < total);
                }
            });
        } catch (err) {
            if (searchLifecycleRef.current.isCurrent(request.requestId) && !isAbortError(err, request.signal)) {
                console.error("Erreur recherche", err);
                setSearchError(err?.response?.data?.error || "Le moteur de recherche est momentanément indisponible.");
            }
        } finally {
            if (searchLifecycleRef.current.commit(request.requestId, () => { setIsLoading(false); setProgress(null); })) {
                activeLocalSearchRequestIdRef.current = null;
                searchLifecycleRef.current.finish(request.requestId);
            }
        }
    }

    const loadMore = () => {
        const nextPage = page + 1;
        setPage(nextPage);
        if (useOcrSearch) {
            runOcrSearch(nextPage, false);
            return;
        }
        fetchResults(query, nextPage, false);
    };

    const handleFeedback = async (e, item, isRelevant) => {
        e.preventDefault();
        e.stopPropagation();
        const searchRequestId = searchLifecycleRef.current.currentRequestId();

        if (feedbackGiven[item.id]) {
            toast.info("Déjà voté.");
            return;
        }

        try {
            await submitSearchFeedback({
                query: settledTerm ?? debouncedQuery,
                doc_id: item.id,
                doc_text: item.content,
                is_relevant: isRelevant,
                model_provider: localOnly ? 'f2llm-local-ft' : 'dual'
            });

            searchLifecycleRef.current.commit(searchRequestId, () => {
                setFeedbackGiven(prev => ({ ...prev, [item.id]: true }));
                toast.success("Merci pour votre retour !");
            });
        } catch (err) {
            if (searchLifecycleRef.current.isCurrent(searchRequestId)) {
                console.error("Feedback error", err);
            }
        }
    };

    const handleSearchModeChange = (value) => {
        invalidateActiveSearch();
        setSearchMode(value);
        setPage(1);
        clearSearchResults();
        setOcrHasSearched(false);
        setOcrStatus('');
    };

    const handleExample = (label) => {
        setQuery(label);
        setPage(1);
        fetchResults(label, 1, true);
    };

    // "Rechercher par concept" keeps the query and runs it in semantic mode, through the search effect.
    const handleTrySemantic = () => {
        autoSemanticSearchRef.current = true;
        handleSearchModeChange('semantic');
    };

    const handleLocalOnlyChange = (checked) => {
        invalidateActiveSearch();
        setLocalOnly(checked);
        setPage(1);
        clearSearchResults();
    };

    const handleQueryChange = (event) => {
        invalidateActiveSearch();
        const value = event.target.value;
        setQuery(value);
        if (value.trim().length < 2) clearSearchResults();
    };

    const handleClearQuery = () => {
        invalidateActiveSearch();
        setQuery('');
        setPage(1);
        clearSearchResults();
        inputRef.current?.focus();
    };

    const handleTomeChange = (value) => {
        invalidateActiveSearch();
        setSelectedTome(value);
        setPage(1);
        clearSearchResults();
        setOcrHasSearched(false);
        setOcrStatus('');
    };

    const handleArcChange = (value) => {
        invalidateActiveSearch();
        setSelectedArc(value);
        setPage(1);
        clearSearchResults();
        setOcrHasSearched(false);
        setOcrStatus('');
    };

    const updateSelectedCharacters = (updater) => {
        invalidateActiveSearch();
        setSelectedCharacters(updater);
        setPage(1);
        clearSearchResults();
        setOcrHasSearched(false);
        setOcrStatus('');
    };

    const resetFilters = () => {
        invalidateActiveSearch();
        setSelectedCharacters([]);
        setSelectedArc('all');
        setSelectedTome('all');
        setPage(1);
        clearSearchResults();
        setOcrHasSearched(false);
        setOcrStatus('');
    };

    const mode = getSearchMode(searchMode);
    const firstOcrMatch = results[0]?.ocr?.matches?.[0] || null;
    const topOcrResult = useOcrSearch && firstOcrMatch?.query_index === 0 && firstOcrMatch?.score >= 0.85 ? results[0] : null;

    const trimmedQuery = query.trim();
    // Keyword search follows the field, so a query the debounce has not released yet is already a search in progress.
    const isDebouncing = searchMode === 'keyword' && trimmedQuery.length >= 2 && query !== debouncedQuery;
    const hasAnswer = useOcrSearch ? ocrHasSearched : trimmedQuery.length >= 2 && settledTerm === trimmedQuery;
    const view = results.length > 0 ? 'results'
        : isLoading || isDebouncing ? 'loading'
            : searchError ? 'error'
                : hasAnswer ? 'empty' : 'idle';

    return (
        <div ref={scrollRef} style={mode.theme} className="h-full min-h-0 w-full overflow-x-hidden overflow-y-auto overscroll-y-contain [scrollbar-gutter:stable_both-edges]">
            <div className="relative isolate pb-16">
                <SearchBackdrop modeId={mode.id} />
                <SearchHero mode={mode} />
                <SearchModeSwitch value={mode.id} onChange={handleSearchModeChange} />

                <div ref={consoleAnchorRef} aria-hidden="true" className="h-4" />
                <div className="z-20 md:sticky md:top-0">
                    <SearchConsole
                        mode={mode}
                        query={query}
                        inputRef={inputRef}
                        onQueryChange={handleQueryChange}
                        onClearQuery={handleClearQuery}
                        onSubmit={handleManualSearch}
                        isLoading={isLoading}
                        progress={isLoading ? progress : null}
                        localOnly={localOnly}
                        onLocalOnlyChange={handleLocalOnlyChange}
                        localModelStatus={localModelStatus}
                        image={{
                            fileName: ocrImageFile?.name,
                            previewUrl: ocrImagePreviewUrl,
                            bubbleCount: ocrExtractedBubbles.length,
                            status: ocrStatus,
                            hasError: Boolean(searchError),
                            isDragging: isDraggingImage,
                            onPick: loadOcrImageFile,
                            onClear: clearOcrImage,
                        }}
                        filters={{
                            tomes,
                            selectedTome,
                            onTomeChange: handleTomeChange,
                            arcs: arcSuggestions,
                            selectedArc,
                            onArcChange: handleArcChange,
                            characters: characterSuggestions,
                            selectedCharacters,
                            onCharactersChange: updateSelectedCharacters,
                            onReset: resetFilters,
                        }}
                    />
                </div>

                <SearchResults
                    view={view}
                    mode={mode}
                    mangaSlug={mangaSlug}
                    results={results}
                    totalCount={totalCount}
                    term={settledTerm}
                    topOcrResult={topOcrResult}
                    isLoadingMore={isLoading}
                    isStale={isDebouncing}
                    hasMore={hasMore}
                    onLoadMore={loadMore}
                    feedback={useSemantic ? { given: feedbackGiven, onVote: handleFeedback } : null}
                    error={searchError}
                    onRetry={handleManualSearch}
                    onTrySemantic={handleTrySemantic}
                    examples={useSemantic ? SEMANTIC_EXAMPLES[mangaSlug] || [] : []}
                    onExample={handleExample}
                    scan={{ status: ocrStatus, progress, bubbleCount: ocrExtractedBubbles.length }}
                />
            </div>
        </div>
    );
}
