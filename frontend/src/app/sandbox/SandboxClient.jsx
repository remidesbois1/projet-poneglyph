"use client";

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { arrayMove } from '@dnd-kit/sortable';
import { useAnnotationInteractions } from '@/hooks/useAnnotationInteractions';
import { useAnnotationOCR } from '@/hooks/useAnnotationOCR';
import { useAnnotationDetection } from '@/hooks/useAnnotationDetection';
import { useAnnotationMetadata } from '@/hooks/useAnnotationMetadata';
import { Dialog } from "@/components/ui/dialog";
import {
    ArrowLeft,
    ArrowRight,
    BookOpen,
    Check,
    ExternalLink,
    FileImage,
    Upload,
    X,
} from "lucide-react";
import { toast } from "sonner";
import AnnotateLeftSidebar from '@/components/AnnotateLeftSidebar';
import AnnotateCanvas from '@/components/AnnotateCanvas';
import AnnotateAnnotationSidebar from '@/components/AnnotateAnnotationSidebar';
import AnnotateEditorDialog from '@/components/AnnotateEditorDialog';
import AnnotateMetadataModal from '@/components/AnnotateMetadataModal';
import LocalOcrStatusIndicator from '@/components/LocalOcrStatusIndicator';
import AiAccessDialog from '@/components/AiAccessDialog';

const PONEGLYPH_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const GLENAT_BOOK_URL = 'https://www.glenat.com/glenat-manga/one-piece-edition-originale-tome-01-9782723488525/';
const SANDBOX_GUIDE_STORAGE_KEY = 'poneglyph-sandbox-guide-seen';
// Sandbox annotations belong to the current local session and remain editable.
const canEditSandboxBubble = () => true;

const GLENAT_DEMO_PAGES = [18, 22, 24, 48].map((pageNumber) => {
    return {
        id: `glenat-${pageNumber}`,
        pageNumber,
        imageUrl: `/api/sandbox/glenat-page?page=${pageNumber}`,
    };
});

function generateSlots(count, seed = 12345) {
    const slots = [];
    let state = seed;
    const nextRand = () => {
        let t = state += 0x6D2B79F5;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let i = 0; i < count; i++) {
        slots.push({
            x: 2 + nextRand() * 92,
            y: 2 + nextRand() * 92,
            size: 20 + Math.floor(nextRand() * 32),
            rotate: Math.floor(nextRand() * 80) - 40,
            char: PONEGLYPH_LETTERS[Math.floor(nextRand() * 26)],
            opacity: 0,
        });
    }
    return slots;
}

function PoneglyphBackground({ count = 20, seed = 0 }) {
    const [glyphs, setGlyphs] = React.useState(() => generateSlots(count, seed));

    React.useEffect(() => {
        const timers = [];
        glyphs.forEach((_, i) => {
            const cycle = () => {
                const delay = 500 + Math.random() * 3500;
                const fadeIn = setTimeout(() => {
                    setGlyphs(prev => prev.map((g, idx) =>
                        idx === i ? { ...g, opacity: 1, char: PONEGLYPH_LETTERS[Math.floor(Math.random() * 26)] } : g
                    ));
                    const stayDuration = 2500 + Math.random() * 4500;
                    const fadeOut = setTimeout(() => {
                        setGlyphs(prev => prev.map((g, idx) =>
                            idx === i ? { ...g, opacity: 0 } : g
                        ));
                        const nextTimer = setTimeout(cycle, 1000 + Math.random() * 2000);
                        timers.push(nextTimer);
                    }, stayDuration);
                    timers.push(fadeOut);
                }, delay);
                timers.push(fadeIn);
            };
            cycle();
        });
        return () => timers.forEach(t => clearTimeout(t));
    }, []);

    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none select-none">
            {glyphs.map((g, i) => (
                <span
                    key={i}
                    className="absolute"
                    style={{
                        fontFamily: "'Poneglyph', serif",
                        fontSize: `${g.size}px`,
                        left: `${g.x}%`,
                        top: `${g.y}%`,
                        transform: `rotate(${g.rotate}deg)`,
                        opacity: g.opacity * 0.18,
                        transition: 'opacity 2.5s ease-in-out',
                        color: '#2F7AAF',
                        lineHeight: 1,
                    }}
                >
                    {g.char}
                </span>
            ))}
        </div>
    );
}

function SandboxGuide({ open, stepIndex, onStepChange, onClose, targets, isMobile = false }) {
    const desktopEditorSteps = [
            {
                target: 'editorSidebar',
                eyebrow: '01 · Piloter l’analyse',
                title: 'Le panneau de contrôle',
                description: 'Cochez les modèles OCR, chargez-les, puis lancez l’OCR de la page : chaque bulle détectée s’ouvre ensuite pour validation.',
            },
            {
                target: 'canvas',
                eyebrow: '02 · Annoter',
                title: 'Dessinez directement sur la page',
                description: 'Tracez un rectangle autour d’une bulle pour la reconnaître. Maintenez Maj pour déplacer ou redimensionner une annotation existante.',
            },
            {
                target: 'annotations',
                eyebrow: '03 · Vérifier',
                title: 'La liste de résultats',
                description: 'Retrouvez toutes les bulles détectées, réordonnez-les par glisser-déposer, copiez leur texte ou ouvrez-les pour les corriger.',
            },
            {
                target: 'editorToolbar',
                eyebrow: '04 · À tout moment',
                title: 'Le guide reste accessible',
                description: 'Besoin d’un rappel ? Le bouton Guide en haut à droite relance cette présentation quand vous le souhaitez.',
            },
        ];
    const mobileEditorSteps = [
        {
            target: 'editorToolbar',
            eyebrow: '01 · Repérer l’écran',
            title: 'Votre page est prête',
            description: 'Cette barre rappelle la source et le format de la page. Sur ordinateur, le panneau de contrôle permet aussi de choisir les modèles et de lancer l’OCR de la page.',
        },
        {
            target: 'canvas',
            eyebrow: '02 · Annoter',
            title: 'Dessinez directement sur la page',
            description: 'Tracez un rectangle autour d’une bulle pour la reconnaître. Le canvas s’adapte automatiquement à la taille de votre écran.',
        },
        {
            target: 'annotations',
            eyebrow: '03 · Vérifier',
            title: 'La liste de résultats',
            description: 'Retrouvez toutes les bulles détectées et ouvrez-les pour corriger leur texte. Faites défiler l’écran pour passer du canvas à la liste.',
        },
        {
            target: 'editorToolbar',
            eyebrow: '04 · À tout moment',
            title: 'Le guide reste accessible',
            description: 'Besoin d’un rappel ? Le bouton Guide en haut à droite relance cette présentation quand vous le souhaitez.',
        },
    ];
    const steps = isMobile ? mobileEditorSteps : desktopEditorSteps;

    const [targetRect, setTargetRect] = React.useState(null);
    const step = steps[Math.min(stepIndex, steps.length - 1)];

    React.useEffect(() => {
        if (!open) return undefined;

        const updateTargetRect = () => {
            const element = targets?.[step.target]?.current;
            if (!element) {
                setTargetRect(null);
                return;
            }

            const bounds = element.getBoundingClientRect();
            setTargetRect({
                top: bounds.top,
                left: bounds.left,
                width: bounds.width,
                height: bounds.height,
                bottom: bounds.bottom,
                right: bounds.right,
            });
        };

        const targetElement = targets?.[step.target]?.current;
        targetElement?.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        updateTargetRect();
        const frame = window.requestAnimationFrame(updateTargetRect);
        window.addEventListener('resize', updateTargetRect);
        window.addEventListener('scroll', updateTargetRect, true);

        return () => {
            window.cancelAnimationFrame(frame);
            window.removeEventListener('resize', updateTargetRect);
            window.removeEventListener('scroll', updateTargetRect, true);
        };
    }, [open, step.target, targets]);

    React.useEffect(() => {
        if (!open) return undefined;

        const handleKeyDown = (event) => {
            if (event.key === 'Escape') onClose();
            if (event.key === 'ArrowRight' && stepIndex < steps.length - 1) onStepChange(stepIndex + 1);
            if (event.key === 'ArrowLeft' && stepIndex > 0) onStepChange(stepIndex - 1);
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [open, onClose, onStepChange, stepIndex, steps.length]);

    if (!open) return null;

    const viewportWidth = typeof window === 'undefined' ? 1024 : window.innerWidth;
    const viewportHeight = typeof window === 'undefined' ? 768 : window.innerHeight;
    const tooltipWidth = Math.min(380, viewportWidth - 32);
    const tooltipHeight = Math.min(360, viewportHeight - 32);
    const tooltipLeft = targetRect
        ? Math.max(16, Math.min(targetRect.left, viewportWidth - tooltipWidth - 16))
        : Math.max(16, (viewportWidth - tooltipWidth) / 2);
    const desiredTooltipTop = targetRect ? targetRect.bottom + 18 : (viewportHeight - tooltipHeight) / 2;
    const tooltipTop = Math.max(16, Math.min(desiredTooltipTop, viewportHeight - tooltipHeight - 16));
    const isLastStep = stepIndex === steps.length - 1;

    return (
        <>
            <div className="fixed inset-0 z-[80] bg-[#010610]/78 backdrop-blur-[1px]" aria-hidden="true" />
            {targetRect && (
                <div
                    className="pointer-events-none fixed z-[81] rounded-2xl border-2 border-[#8dbbff] shadow-[0_0_0_9999px_rgba(1,6,16,0.78),0_0_28px_rgba(61,134,255,0.45)] transition-all duration-300"
                    style={{
                        top: `${Math.max(8, targetRect.top - 8)}px`,
                        left: `${Math.max(8, targetRect.left - 8)}px`,
                        width: `${targetRect.width + 16}px`,
                        height: `${targetRect.height + 16}px`,
                    }}
                />
            )}
            <section
                role="dialog"
                aria-modal="true"
                aria-labelledby="sandbox-guide-title"
                className="fixed z-[82] max-w-[calc(100vw-2rem)] overflow-y-auto rounded-2xl border border-[#8dbbff]/35 bg-[#071625]/[.98] p-5 text-slate-100 shadow-[0_24px_80px_rgba(0,0,0,.55)] backdrop-blur-xl transition-all duration-300 sm:p-6"
                style={{ top: `${tooltipTop}px`, left: `${tooltipLeft}px`, width: `${tooltipWidth}px`, maxHeight: `calc(100vh - 32px)` }}
            >
                <div className="mb-5 flex items-start justify-between gap-4">
                    <div>
                        <p className="mb-1 text-[10px] font-black uppercase tracking-[.18em] text-[#8dbbff]">Guide du sandbox</p>
                        <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">{step.eyebrow}</p>
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        aria-label="Fermer le guide"
                        className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-white"
                    >
                        <X size={16} />
                    </button>
                </div>

                <h2 id="sandbox-guide-title" className="text-xl font-black tracking-tight text-white">{step.title}</h2>
                <p className="mt-3 text-sm leading-relaxed text-slate-300">{step.description}</p>

                <div className="mt-6 flex items-center gap-1.5" aria-label={`Étape ${stepIndex + 1} sur ${steps.length}`}>
                    {steps.map((item, index) => (
                        <span
                            key={`${item.target}-${index}`}
                            className={`h-1.5 flex-1 rounded-full transition-colors ${index <= stepIndex ? 'bg-[#8dbbff]' : 'bg-white/10'}`}
                        />
                    ))}
                </div>

                <div className="mt-5 flex items-center justify-between gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        className="text-xs font-bold text-slate-400 transition hover:text-white"
                    >
                        Passer le guide
                    </button>
                    <div className="flex items-center gap-2">
                        {stepIndex > 0 && (
                            <button
                                type="button"
                                onClick={() => onStepChange(stepIndex - 1)}
                                className="rounded-xl border border-white/12 px-3 py-2 text-xs font-bold text-slate-300 transition hover:bg-white/10 hover:text-white"
                            >
                                Retour
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={() => isLastStep ? onClose() : onStepChange(stepIndex + 1)}
                            className="inline-flex items-center gap-2 rounded-xl bg-[#3d86ff] px-3.5 py-2 text-xs font-black text-white shadow-[0_8px_24px_rgba(61,134,255,.28)] transition hover:bg-[#5795ff]"
                        >
                            {isLastStep ? 'C’est parti' : 'Suivant'}
                            {isLastStep ? <Check size={14} /> : <ArrowRight size={14} />}
                        </button>
                    </div>
                </div>
            </section>
        </>
    );
}

export default function SandboxClient() {
    const [page, setPage] = useState(null);
    const [existingBubbles, setExistingBubbles] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [loadingText, setLoadingText] = useState("Analyse en cours...");
    const [pendingAnnotation, setPendingAnnotation] = useState(null);
    const [rectangle, setRectangle] = useState(null);
    const [imageDimensions, setImageDimensions] = useState(null);
    const [ocrSource, setOcrSource] = useState(null);
    const [debugImageUrl, setDebugImageUrl] = useState(null);
    const [showApiKeyModal, setShowApiKeyModal] = useState(false);
    const [showDescModal, setShowDescModal] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [imageUrl, setImageUrl] = useState(null);
    const [isGuideOpen, setIsGuideOpen] = useState(false);
    const [guideStep, setGuideStep] = useState(0);
    const [sampleLoadState, setSampleLoadState] = useState({});

    const containerRef = useRef(null);
    const imageRef = useRef(null);
    const fileInputRef = useRef(null);
    const editorToolbarRef = useRef(null);
    const editorSidebarRef = useRef(null);
    const editorCanvasRef = useRef(null);
    const annotationSidebarRef = useRef(null);

    const guideTargets = {
        editorToolbar: editorToolbarRef,
        editorSidebar: editorSidebarRef,
        canvas: editorCanvasRef,
        annotations: annotationSidebarRef,
    };

    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        const checkMobile = () => setIsMobile(window.innerWidth < 1024);
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return () => window.removeEventListener('resize', checkMobile);
    }, []);

    useEffect(() => {
        if (!page || window.localStorage.getItem(SANDBOX_GUIDE_STORAGE_KEY) === 'true') return undefined;

        const timer = window.setTimeout(() => {
            setGuideStep(0);
            setIsGuideOpen(true);
        }, 450);

        return () => window.clearTimeout(timer);
    }, [page]);

    useEffect(() => () => {
        if (imageUrl?.startsWith('blob:')) URL.revokeObjectURL(imageUrl);
    }, [imageUrl]);

    const closeGuide = () => {
        setIsGuideOpen(false);
        window.localStorage.setItem(SANDBOX_GUIDE_STORAGE_KEY, 'true');
    };

    const openGuide = () => {
        setGuideStep(0);
        setIsGuideOpen(true);
    };

    const setSandboxImageUrl = (nextUrl) => {
        setImageUrl((previousUrl) => {
            if (previousUrl?.startsWith('blob:')) URL.revokeObjectURL(previousUrl);
            return nextUrl;
        });
    };

    const handleImageUpload = (file) => {
        if (!file || !file.type.startsWith('image/')) {
            toast.error("Veuillez sélectionner une image valide.");
            return;
        }
        const url = URL.createObjectURL(file);
        setSandboxImageUrl(url);
        setPage({
            id: 'sandbox-upload',
            url_image: url,
            statut: 'not_started',
            numero_page: 0,
            description: null,
            source_label: 'Image importée',
            source_url: null,
        });
        setExistingBubbles([]);
        setRectangle(null);
        setPendingAnnotation(null);
        setDebugImageUrl(null);
        setImageDimensions(null);
        setOcrSource(null);
        setIsGuideOpen(false);
        toast.success("Image chargée ! Prêt pour l’annotation.");
    };

    const handleGlenatPageSelect = (demoPage) => {
        setSandboxImageUrl(demoPage.imageUrl);
        setPage({
            id: `sandbox-${demoPage.id}`,
            url_image: demoPage.imageUrl,
            statut: 'not_started',
            numero_page: demoPage.pageNumber,
            description: null,
            source_label: 'Glénat · liseuse',
            source_url: GLENAT_BOOK_URL,
        });
        setExistingBubbles([]);
        setRectangle(null);
        setPendingAnnotation(null);
        setDebugImageUrl(null);
        setImageDimensions(null);
        setOcrSource(null);
        setIsGuideOpen(false);
        toast.success('Extrait sélectionné. Prêt pour l’annotation.');
    };

    const onDrop = (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        handleImageUpload(file);
    };

    const {
        formData, setFormData, charInput, setCharInput,
        isSavingDesc, isGeneratingAI, tabMode, setTabMode, jsonInput,
        jsonError, handleJsonChange, handleSaveDescription, handleGenerateAI,
    } = useAnnotationMetadata({
        page, setPage, pageId: 'sandbox', imageRef, showDescModal, setShowDescModal, setShowApiKeyModal,
        onSaveDescription: async (payload) => {
            setPage(prev => ({ ...prev, description: JSON.stringify(payload) }));
            return { data: { success: true } };
        },
        onFetchSuggestions: async () => ({ data: { arcs: [], characters: [] } })
    });

    const {
        models: ocrModels, toggleOcrModel, prepareSelectedModels, refreshModelAccess,
        pageModels, runPageModel, runLocalOcr, runBackgroundOcr,
        handleRetryWithCloud
    } = useAnnotationOCR({
        imageRef, pageId: imageUrl, rectangle, pendingAnnotation, setPendingAnnotation,
        setIsSubmitting, setLoadingText, setIsModalOpen, setOcrSource,
        setDebugImageUrl, setShowApiKeyModal, isSandbox: true, allowPageModels: true
    });

    const {
        isAutoDetecting, setIsAutoDetecting, queueLength, detectionStatus,
        loadDetectionModel, detectionProgress, isPageOcrRunning, runPageOcr,
        cancelPageOcr, processNextBubble
    } = useAnnotationDetection({
        imageRef, pageId: imageUrl, existingBubbles, setRectangle, setPendingAnnotation, setDebugImageUrl,
        runLocalOcr, runBackgroundOcr, runPageModel, pageModels
    });

    const prepareOcrModels = () => {
        loadDetectionModel();
        prepareSelectedModels();
    };

    // Keys and the ChatGPT session are edited in the API dialog.
    useEffect(() => {
        if (!showApiKeyModal) void refreshModelAccess();
    }, [refreshModelAccess, showApiKeyModal]);

    const {
        isDrawing, startPoint, endPoint, mousePos, isShiftPressed,
        hoveredBubble, setHoveredBubble, handleMouseDown, handleMouseMove,
        handleMouseUp, handleInteractionStart
    } = useAnnotationInteractions({
        containerRef, imageRef, imageDimensions, existingBubbles, setExistingBubbles,
        pendingAnnotation, setPendingAnnotation, setRectangle, canEdit: true, canEditBubble: canEditSandboxBubble, isMobile,
        pageStatus: 'not_started', isSubmitting: isSubmitting || isPageOcrRunning, showApiKeyModal, showDescModal,
        onUpdateGeometry: (targetId, geometry) => {
            setExistingBubbles(prev => prev.map(b => b.id === targetId ? { ...b, ...geometry } : b));
            toast.success("Position mise à jour");
        }
    });

    useEffect(() => {
        if (isAutoDetecting) return;
        if (rectangle && imageRef.current) {
            const analysisData = { id: Date.now(), id_page: 'sandbox', ...rectangle, texte_propose: '' };
            setPendingAnnotation(analysisData);
            setDebugImageUrl(null);
            runLocalOcr(analysisData);
        }
    }, [rectangle, isAutoDetecting]);

    const handleSaveApiKey = (key) => {
        localStorage.setItem('google_api_key', key);
        setShowApiKeyModal(false);
        if (pendingAnnotation) handleRetryWithCloud();
        if (showDescModal) handleSaveDescription();
    };

    const handleEditBubble = (bubble) => {
        if (isMobile) return;
        setPendingAnnotation(bubble);
        setIsModalOpen(true);
    };

    const handleDeleteBubble = (bubbleId) => {
        if (window.confirm("Supprimer cette annotation locale ?")) {
            setExistingBubbles(prev => prev.filter(b => b.id !== bubbleId));
            toast.success("Annotation supprimée.");
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
            setExistingBubbles(prev => {
                const results = [...prev];
                const idx = results.findIndex(b => b.id === newData.id || (tempId && b.id === tempId));

                if (idx !== -1) {
                    results[idx] = { ...results[idx], ...newData };
                    if (isBackgroundResult) results[idx].isOptimistic = false;
                } else if (!isBackgroundResult) {
                    results.push({ ...newData, id: newData.id || Date.now() });
                }

                return results.sort((a, b) => (a.order || 0) - (b.order || 0));
            });
        }
    };

    const handleDragEnd = (event) => {
        const { active, over } = event;
        if (active && over && active.id !== over.id) {
            setExistingBubbles((bubbles) => {
                const oldIndex = bubbles.findIndex(b => b.id === active.id);
                const newIndex = bubbles.findIndex(b => b.id === over.id);
                const newOrder = arrayMove(bubbles, oldIndex, newIndex);
                return newOrder.map((b, index) => ({ ...b, order: index + 1 }));
            });
        }
    };

    if (!page) {
        return (
            <div className="poneglyph-app relative min-h-screen overflow-y-auto"
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}>
                <PoneglyphBackground count={32} seed={123} />

                <header className="relative z-20 mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-5 sm:px-6 lg:px-8">
                    <Link href="/" className="inline-flex items-center gap-2 text-[11px] font-black uppercase tracking-[.18em] text-slate-400 transition hover:text-white">
                        <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/12 bg-white/[.07] text-[#8dbbff]">
                            <ArrowLeft size={14} />
                        </span>
                        Poneglyph
                    </Link>
                    <div className="flex items-center gap-2">
                        <LocalOcrStatusIndicator align="right" />
                    </div>
                </header>

                <main className="relative z-10 mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 pb-10 sm:px-6 lg:gap-8 lg:px-8 lg:pb-14">
                    <section className="poneglyph-panel rounded-[2rem] p-5 sm:p-7">
                        <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
                            <div>
                                <div className="flex items-center gap-2">
                                    <h2 className="text-xl font-black tracking-tight text-white sm:text-2xl">Pages de démonstration</h2>
                                </div>
                                <p className="mt-1.5 text-xs text-slate-400">Des extraits chargés depuis la liseuse Glénat.</p>
                            </div>
                            <a href={GLENAT_BOOK_URL} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 text-[11px] font-bold text-slate-400 underline decoration-white/20 underline-offset-4 transition hover:text-[#bcd6ff]">
                                Voir sur Glénat <ExternalLink size={12} />
                            </a>
                        </div>

                        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                            {GLENAT_DEMO_PAGES.map((demoPage, demoIndex) => {
                                const imageState = sampleLoadState[demoPage.id];

                                return (
                                    <article key={demoPage.id} className="group rounded-2xl border border-white/10 bg-[#04101d]/70 p-2 transition duration-200 hover:-translate-y-1 hover:border-[#8dbbff]/45 hover:bg-[#091d31] hover:shadow-[0_18px_42px_rgba(0,0,0,.28)]">
                                        <button
                                            type="button"
                                            onClick={() => handleGlenatPageSelect(demoPage)}
                                            aria-label={`Choisir l’extrait ${demoIndex + 1} de démonstration`}
                                            className="block w-full text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#8dbbff] focus-visible:ring-offset-2 focus-visible:ring-offset-[#071625]"
                                        >
                                            <div className="relative aspect-[.68/1] overflow-hidden rounded-xl border border-white/10 bg-[#020812]">
                                                {imageState !== 'loaded' && (
                                                    <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-white/[.05] via-[#0b2741] to-white/[.03]" aria-hidden="true" />
                                                )}
                                                <Image
                                                    src={demoPage.imageUrl}
                                                    alt="Extrait de démonstration Glénat"
                                                    fill
                                                    unoptimized
                                                    sizes="(min-width: 1280px) 25vw, (min-width: 640px) 50vw, 100vw"
                                                    loading="lazy"
                                                    className={`h-full w-full object-cover object-top transition duration-500 group-hover:scale-[1.03] ${imageState === 'error' ? 'opacity-0' : 'opacity-100'}`}
                                                    onLoad={() => setSampleLoadState((previous) => ({ ...previous, [demoPage.id]: 'loaded' }))}
                                                    onError={() => setSampleLoadState((previous) => ({ ...previous, [demoPage.id]: 'error' }))}
                                                />
                                                {imageState === 'error' && (
                                                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-4 text-center text-xs text-slate-400">
                                                        <FileImage size={20} className="text-slate-500" />
                                                        Aperçu indisponible
                                                    </div>
                                                )}
                                                <span className="absolute bottom-2 right-2 rounded-full bg-[#020812]/80 px-2.5 py-1.5 text-[10px] font-black uppercase tracking-wider text-white opacity-0 backdrop-blur-sm transition group-hover:opacity-100">Ouvrir</span>
                                            </div>
                                        </button>
                                    </article>
                                );
                            })}
                        </div>
                    </section>

                    <section className="poneglyph-panel-soft flex flex-col gap-5 rounded-[2rem] border-dashed p-5 sm:flex-row sm:items-center sm:justify-between sm:p-7">
                        <div className="flex items-start gap-4">
                            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-[#8dbbff]/20 bg-[#3d86ff]/10 text-[#8dbbff]"><Upload size={19} /></div>
                            <div>
                                <h2 className="text-base font-black text-white">Vous préférez travailler sur votre propre planche ?</h2>
                                <p className="mt-1.5 max-w-xl text-xs leading-relaxed text-slate-400">Importez un fichier JPG, PNG ou WebP. Le fichier reste local à cette session et n’est pas envoyé tant que vous ne lancez pas un service d’analyse.</p>
                            </div>
                        </div>
                        <div className="shrink-0">
                            <input
                                type="file"
                                className="hidden"
                                ref={fileInputRef}
                                accept="image/*"
                                onChange={(event) => {
                                    handleImageUpload(event.target.files?.[0]);
                                    event.target.value = '';
                                }}
                            />
                            <button type="button" onClick={() => fileInputRef.current?.click()} className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-[#8dbbff]/30 bg-[#3d86ff]/12 px-4 py-2.5 text-xs font-black text-[#dbeafe] transition hover:border-[#8dbbff]/60 hover:bg-[#3d86ff]/22 sm:w-auto">
                                Importer une image <ArrowRight size={14} />
                            </button>
                        </div>
                    </section>

                </main>
            </div>
        );
    }

    return (
        <div className="poneglyph-app relative flex h-screen flex-col overflow-hidden lg:flex-row">
            <div ref={editorSidebarRef} className="min-h-0 shrink-0">
                <AnnotateLeftSidebar
                    fromSearch={false}
                    mangaSlug=""
                    page={page}
                    chapterPages={[]}
                    navContext={{ prev: null, next: null }}
                    goToPrev={() => { }}
                    goToNext={() => { }}
                    isGuest={false}
                    role="Admin"
                    isSandbox={true}
                    ocrModels={ocrModels}
                    toggleOcrModel={toggleOcrModel}
                    onPrepareModels={prepareOcrModels}
                    detectionStatus={detectionStatus}
                    detectionProgress={detectionProgress}
                    canRunPageOcr={Boolean(imageUrl)}
                    isPageOcrRunning={isPageOcrRunning}
                    isReviewingPage={isAutoDetecting}
                    reviewRemaining={queueLength}
                    onRunPageOcr={runPageOcr}
                    onCancelPageOcr={cancelPageOcr}
                    isSubmitting={isSubmitting}
                    setShowDescModal={() => { }}
                    setShowApiKeyModal={setShowApiKeyModal}
                    handleSubmitPage={() => { }}
                />
            </div>

            <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#030a13]">
                <div className="absolute left-3 top-3 z-30">
                    <LocalOcrStatusIndicator />
                </div>

                <div ref={editorToolbarRef} className="absolute right-3 top-3 z-30 flex items-center gap-2">
                    <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-[#06111e]/88 px-3 py-2 text-[10px] font-bold text-slate-300 shadow-lg backdrop-blur-sm sm:flex">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                        {page.source_label || 'Image importée'}
                    </div>
                    <button type="button" onClick={openGuide} className="inline-flex items-center gap-1.5 rounded-full border border-[#8dbbff]/25 bg-[#06111e]/90 px-3 py-2 text-[10px] font-black uppercase tracking-wider text-[#bcd6ff] shadow-lg backdrop-blur-sm transition hover:border-[#8dbbff]/55 hover:bg-[#3d86ff]/20 hover:text-white">
                        <BookOpen size={13} /> Guide
                    </button>
                </div>

                <div className="flex flex-col lg:flex-row flex-1 overflow-hidden min-h-0">
                    <div ref={editorCanvasRef} className="flex min-h-0 min-w-0 flex-1">
                        <AnnotateCanvas
                            canEdit={true}
                            imageDimensions={imageDimensions}
                            setImageDimensions={setImageDimensions}
                            containerRef={containerRef}
                            imageRef={imageRef}
                            handleMouseDown={handleMouseDown}
                            handleMouseMove={handleMouseMove}
                            handleMouseUp={handleMouseUp}
                            imageUrl={imageUrl}
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
                        />
                    </div>

                    <div ref={annotationSidebarRef} className="min-h-0 w-full shrink-0 lg:w-[380px]">
                        <AnnotateAnnotationSidebar
                            existingBubbles={existingBubbles}
                            handleDragEnd={handleDragEnd}
                            user={{ id: 'sandbox-user', role: 'Admin' }}
                            handleEditBubble={handleEditBubble}
                            handleDeleteBubble={handleDeleteBubble}
                            canEdit={true}
                            canEditBubble={canEditSandboxBubble}
                            canReorder={true}
                            role="Admin"
                        />
                    </div>
                </div>
            </div>

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
                isSubmitting={isSubmitting || isPageOcrRunning}
                isSandbox={true}
            />

            <Dialog open={showApiKeyModal} onOpenChange={setShowApiKeyModal}>
                <AiAccessDialog onSave={handleSaveApiKey} onSaveDeepSeek={() => setShowApiKeyModal(false)} onSaveMistral={() => setShowApiKeyModal(false)} />
            </Dialog>

            <AnnotateMetadataModal
                isOpen={showDescModal}
                onOpenChange={setShowDescModal}
                tabMode={tabMode}
                setTabMode={setTabMode}
                formData={formData}
                setFormData={setFormData}
                charInput={charInput}
                setCharInput={setCharInput}
                suggestions={{ arcs: [], characters: [] }}
                isGeneratingAI={isGeneratingAI}
                handleGenerateAI={handleGenerateAI}
                handleSaveDescription={handleSaveDescription}
                isSavingDesc={isSavingDesc}
                jsonInput={jsonInput}
                handleJsonChange={handleJsonChange}
                jsonError={jsonError}
            />

            <SandboxGuide
                open={isGuideOpen}
                stepIndex={guideStep}
                onStepChange={setGuideStep}
                onClose={closeGuide}
                targets={guideTargets}
                isMobile={isMobile}
            />
        </div>
    );
}
