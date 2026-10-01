"use client";

import { useEffect, useRef, useState } from 'react';
import { ASTRA_DESCRIPTION_MODEL, DESCRIPTION_PROMPT_VERSION, isCurrentAstraPage } from '@poneglyph/shared/page-description';
import { useAuth } from '@/context/AuthContext';
import { getEmbeddingStats, getSemanticRebuildStatus } from '@/lib/api';
import { getChatGptStatus, subscribeToChatGptAuth } from '@/lib/chatGptDesktop';
import { createPendingDescriptionCache, DEFAULT_ASTRA_CONCURRENCY, getRebuildFailure, requireSupabaseSession, runAstraSemanticRebuild } from '@/lib/astraSemanticRebuild';
import { getPrompt, invalidatePromptCache } from '@/lib/promptConfig';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Loader2, RotateCcw, Square, Zap } from 'lucide-react';
import { toast } from 'sonner';

const STATUS_LABELS = {
    running: 'Reconstruction en cours', stopped: 'Arrêté — reprise disponible', completed: 'Reconstruction terminée',
    finished_with_errors: 'Terminé avec erreurs — reprise disponible', quota: 'Quota atteint — reprise après réinitialisation',
    chatgpt_auth: 'Reconnexion ChatGPT nécessaire', supabase_auth: 'Reconnexion Supabase nécessaire',
    supabase_permissions: 'Permissions admin à vérifier',
    supabase_unavailable: 'Supabase indisponible — reprise disponible',
    unavailable: 'Configuration serveur à vérifier',
    chatgpt_request: 'Requête ChatGPT refusée — consultez le détail de l’erreur',
    chatgpt_unavailable: 'Service ChatGPT indisponible — reprise disponible',
    gemini_key: 'Clé Gemini personnelle à vérifier dans votre profil',
};

export default function AstraSemanticRebuild({ mangaSlug, pages = [], disabled = false, onBusyChange, onComplete }) {
    const { role } = useAuth();
    const [chatGpt, setChatGpt] = useState({ available: false, connected: false });
    const [checkingConnection, setCheckingConnection] = useState(true);
    const [busy, setBusy] = useState(false);
    const [stopping, setStopping] = useState(false);
    const [concurrency, setConcurrency] = useState(DEFAULT_ASTRA_CONCURRENCY);
    const [progress, setProgress] = useState(null);
    const runRef = useRef(false);
    const stopRef = useRef(false);
    const mountedRef = useRef(false);

    useEffect(() => {
        mountedRef.current = true;
        getChatGptStatus().then((status) => {
            if (mountedRef.current) { setChatGpt(status); setCheckingConnection(false); }
        });
        const unsubscribe = subscribeToChatGptAuth((status) => {
            if (mountedRef.current) setChatGpt({ available: true, ...status });
        });
        return () => { mountedRef.current = false; stopRef.current = true; unsubscribe(); };
    }, []);

    const refreshConnection = async () => {
        setCheckingConnection(true);
        setChatGpt(await getChatGptStatus());
        setCheckingConnection(false);
    };

    const start = async (force = false) => {
        if (runRef.current || role !== 'Admin' || !mangaSlug) return;
        runRef.current = true;
        stopRef.current = false;
        setBusy(true);
        setStopping(false);
        onBusyChange?.(true);
        try {
            const status = await getChatGptStatus();
            setChatGpt(status);
            if (!status.available) throw new Error('Ce traitement nécessite Poneglyph Desktop.');
            if (!status.connected) throw new Error('Connectez-vous à ChatGPT dans votre profil puis actualisez la connexion.');
            const session = await requireSupabaseSession();
            await getSemanticRebuildStatus();
            // Fresh server flags handle crashes or a lost response after a successful save.
            const freshPages = (await getEmbeddingStats(mangaSlug)).data;
            const remaining = force ? freshPages : freshPages.filter((page) => !isCurrentAstraPage(page));
            if (!remaining.length) {
                toast.info('Toutes les pages sont déjà reconstruites avec Astra et le prompt actuel.');
                return;
            }
            const scope = `${session.user.id}:${mangaSlug}`;
            const text = force || remaining.length === freshPages.length
                ? `Toutes les ${freshPages.length} pages du manga « ${mangaSlug} » seront retraitées, y compris celles qui ont déjà une description et des embeddings.`
                : `Reprendre ${remaining.length} pages sur ${freshPages.length} du manga « ${mangaSlug} ». Les ${freshPages.length - remaining.length} pages déjà reconstruites seront ignorées.`;
            if (!window.confirm(`${text}\n\nGPT-6 Astra (gpt-6-astra), raisonnement high, prompt v${DESCRIPTION_PROMPT_VERSION}, ${concurrency} pages à la fois. Ce traitement consomme votre quota ChatGPT Work/Codex. Les anciennes données sont remplacées uniquement après génération complète des embeddings Voyage et Gemini.\n\nLancer le traitement ?`)) return;
            invalidatePromptCache();
            const prompt = `${await getPrompt('page_description')}\n\n${await getPrompt('strict_json_suffix')}`;
            const result = await runAstraSemanticRebuild({
                pages: freshPages, mangaSlug, force, prompt, concurrency,
                cache: createPendingDescriptionCache(scope),
                getSession: () => requireSupabaseSession({ expectedUserId: session.user.id }),
                shouldStop: () => stopRef.current,
                onProgress: (next) => { if (mountedRef.current) setProgress(next); },
            });
            if (result.status === 'chatgpt_auth' && mountedRef.current) setChatGpt({ available: true, connected: false });
            if (mountedRef.current) {
                if (result.status === 'completed') toast.success('Reconstruction Astra terminée.');
                else toast.info(STATUS_LABELS[result.status], { description: `${result.saved} pages enregistrées, ${result.errors.length} erreurs.` });
            }
        } catch (error) {
            const failure = getRebuildFailure(error);
            if (mountedRef.current) toast.error(failure.message);
        } finally {
            runRef.current = false;
            if (mountedRef.current) {
                setBusy(false);
                setStopping(false);
                onBusyChange?.(false);
                onComplete?.();
            }
        }
    };

    if (role !== 'Admin') return null;
    const completed = pages.filter(isCurrentAstraPage).length;
    const canStart = !busy && !disabled && !checkingConnection && chatGpt.available && chatGpt.connected && pages.length > 0;
    const percentage = progress?.total ? Math.round(progress.completed / progress.total * 100) : 0;
    return (
        <div className="mb-5 space-y-4 rounded-2xl border border-violet-400/30 bg-violet-500/10 p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                    <p className="font-semibold text-white">Reconstruire avec GPT-6 Astra</p>
                    <p className="mt-1 text-xs text-slate-300">
                        {ASTRA_DESCRIPTION_MODEL} · Raisonnement high · Prompt v{DESCRIPTION_PROMPT_VERSION} · {pages.length} pages au total
                    </p>
                    <p className="mt-1 text-xs text-slate-300">
                        Images originales privées · Voyage + Gemini · {completed} pages déjà reconstruites
                    </p>
                    <p className="mt-1 text-xs text-slate-300">Gemini : clé serveur en priorité, clé personnelle du profil en secours.</p>
                    <p className="mt-2 text-xs text-violet-200" role="status">
                        Connexion ChatGPT : {checkingConnection ? 'vérification…' : !chatGpt.available ? 'Poneglyph Desktop requis' : chatGpt.connected ? 'connecté' : 'déconnecté — connectez-vous dans votre profil'}
                    </p>
                </div>
                <Button variant="outline" size="sm" onClick={refreshConnection} disabled={busy || checkingConnection}>
                    <RotateCcw className="mr-2 h-4 w-4" /> Actualiser la connexion
                </Button>
            </div>
            <div className="flex flex-wrap items-center gap-2">
                <label className="flex items-center gap-2 text-xs text-slate-300">
                    Pages simultanées
                    <select aria-label="Pages simultanées" value={concurrency} onChange={(event) => setConcurrency(Number(event.target.value))}
                        disabled={busy || disabled} className="rounded-md border border-violet-400/30 bg-[#020713] px-2 py-2 text-white disabled:opacity-50">
                        <option value={2}>2</option>
                        <option value={3}>3</option>
                    </select>
                </label>
                <Button onClick={() => start(false)} disabled={!canStart} className="bg-violet-600 hover:bg-violet-500">
                    {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
                    Refaire descriptions + embeddings avec Astra
                </Button>
                {completed > 0 && (
                    <Button onClick={() => start(true)} disabled={!canStart} variant="outline">Tout recommencer, y compris les pages Astra</Button>
                )}
                {busy && (
                    <Button variant="outline" onClick={() => { stopRef.current = true; setStopping(true); }} disabled={stopping}>
                        <Square className="mr-2 h-4 w-4" /> {stopping ? 'Arrêt après les pages en cours…' : 'Arrêter'}
                    </Button>
                )}
            </div>
            {completed > 0 && <p className="text-xs text-slate-300">Le bouton principal reprend les pages restantes sans retraiter celles déjà terminées.</p>}
            {progress && (
                <div className="space-y-3" aria-live="polite">
                    <p className="text-sm font-medium text-white">{stopping ? 'Arrêt demandé' : STATUS_LABELS[progress.status]}</p>
                    <div className="flex flex-wrap justify-between gap-2 text-xs text-slate-300">
                        <span>{progress.completed}/{progress.total} pages terminées · {progress.saved} enregistrées · {progress.skipped} déjà à jour · {progress.errors.length} erreurs</span>
                        <span>{percentage}%</span>
                    </div>
                    {progress.geminiUserKeyPages > 0 && <p className="text-xs text-violet-200">Gemini via votre clé personnelle : {progress.geminiUserKeyPages} pages enregistrées.</p>}
                    <Progress value={percentage} className="h-2" />
                    {progress.activePages?.length > 0 && (
                        <div className="space-y-1 text-xs text-violet-200">
                            <p>{progress.activePages.length}/{progress.concurrency} pages en cours</p>
                            {progress.activePages.map(({ page, phase }) => (
                                <p key={page.id}>Page {page.id} · Tome {page.tome_numero} · Chapitre {page.chapitre_numero} · Page {page.numero} — {phase}</p>
                            ))}
                        </div>
                    )}
                    <div className="max-h-40 space-y-1 overflow-y-auto rounded-lg bg-[#020713] p-3 font-mono text-xs text-slate-300">
                        {progress.log.map((entry, index) => <p key={index}>[{new Date(entry.time).toLocaleTimeString()}] {entry.message}</p>)}
                    </div>
                    {progress.errors.length > 0 && (
                        <details className="text-xs text-rose-300">
                            <summary className="cursor-pointer">Erreurs par page ({progress.errors.length})</summary>
                            <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto">
                                {progress.errors.map((error, index) => <li key={index}>Page {error.pageId} : {error.message}</li>)}
                            </ul>
                        </details>
                    )}
                </div>
            )}
        </div>
    );
}
