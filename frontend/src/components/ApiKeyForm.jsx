"use client";

import React, { useEffect, useState } from 'react';
import { ArrowRight, CheckCircle2, ExternalLink, KeyRound, Loader2, LogIn, ShieldCheck, Trash2 } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { getChatGptStatus, loginChatGpt, logoutChatGpt, subscribeToChatGptAuth } from '@/lib/chatGptDesktop';
import { DEEPSEEK_API_KEY_STORAGE_KEY, DEEPSEEK_LABEL, DEEPSEEK_MODEL } from '@/lib/deepseekConfig';

const GOOGLE_API_KEY = 'google_api_key';

function readStoredKey(storageKey) {
    try { return globalThis.localStorage?.getItem(storageKey) || ''; }
    catch { return ''; }
}

function ProviderKeyCard({ provider, storageKey, description, keyUrl, placeholder, onSave }) {
    const [savedKey, setSavedKey] = useState(() => readStoredKey(storageKey));
    const [isEditing, setIsEditing] = useState(() => !readStoredKey(storageKey));
    const [key, setKey] = useState('');
    const inputId = `api-key-${storageKey}`;

    useEffect(() => {
        const syncKey = () => setSavedKey(readStoredKey(storageKey));
        window.addEventListener('storage', syncKey);
        return () => window.removeEventListener('storage', syncKey);
    }, [storageKey]);

    const saveKey = (event) => {
        event.preventDefault();
        const nextKey = key.trim();
        if (!nextKey) return;
        try { localStorage.setItem(storageKey, nextKey); }
        catch { toast.error('Impossible d’enregistrer la clé dans ce navigateur.'); return; }
        setSavedKey(nextKey);
        setKey('');
        setIsEditing(false);
        window.dispatchEvent(new Event('storage'));
        onSave?.(nextKey);
        toast.success(`Clé API ${provider} enregistrée.`);
    };

    const removeKey = () => {
        try { localStorage.removeItem(storageKey); }
        catch { toast.error('Impossible de supprimer la clé dans ce navigateur.'); return; }
        setSavedKey(null);
        setKey('');
        setIsEditing(true);
        window.dispatchEvent(new Event('storage'));
        toast.info(`Clé API ${provider} supprimée.`);
    };

    const maskedKey = savedKey && savedKey.length >= 10
        ? `${savedKey.slice(0, 6)}••••••••••••••••${savedKey.slice(-4)}`
        : '••••••••••••••••';

    return (
        <section aria-label={`Clé API ${provider}`} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-sm sm:p-5">
            <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                <div>
                    <Label htmlFor={inputId} className="text-sm font-semibold text-slate-100">Clé API {provider}</Label>
                    <p className="mt-1 text-xs leading-relaxed text-slate-400">{description}</p>
                </div>
                <a href={keyUrl} target="_blank" rel="noopener noreferrer" aria-label={`Obtenir une clé ${provider}`} className="flex shrink-0 items-center gap-1.5 rounded-full border border-indigo-400/20 bg-indigo-400/10 px-2.5 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:border-indigo-300/40 hover:bg-indigo-400/20">
                    Obtenir une clé <ExternalLink className="h-3 w-3" />
                </a>
            </div>

            {savedKey && !isEditing ? (
                <div className="flex items-center justify-between gap-3 rounded-xl border border-emerald-400/20 bg-emerald-400/[0.08] p-3">
                    <div className="flex min-w-0 items-center gap-3">
                        <span className="rounded-full bg-emerald-400/15 p-2"><CheckCircle2 className="h-4 w-4 text-emerald-300" /></span>
                        <div className="min-w-0">
                            <p className="text-[11px] font-medium text-emerald-200">Clé enregistrée</p>
                            <code className="block truncate font-mono text-xs font-semibold tracking-wide text-slate-200 sm:text-sm">{maskedKey}</code>
                        </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                        <Button type="button" variant="ghost" size="sm" aria-label={`Changer la clé ${provider}`} className="h-10 px-2 text-xs text-slate-300 hover:bg-white/10 hover:text-white" onClick={() => { setKey(''); setIsEditing(true); }}>Changer</Button>
                        <Button type="button" variant="ghost" size="icon" className="h-10 w-10 text-rose-300 hover:bg-rose-400/10 hover:text-rose-200" onClick={removeKey} title={`Supprimer la clé ${provider}`} aria-label={`Supprimer la clé ${provider}`}><Trash2 className="h-4 w-4" /></Button>
                    </div>
                </div>
            ) : (
                <form onSubmit={saveKey} className="space-y-3">
                    <div className="group relative">
                        <KeyRound className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 transition-colors group-focus-within:text-indigo-300" />
                        <Input id={inputId} type="password" value={key} onChange={(event) => setKey(event.target.value)} placeholder={placeholder} className="h-11 border-white/10 bg-slate-950/50 pl-10 font-mono text-sm text-slate-100 placeholder:text-slate-600 focus-visible:border-indigo-400/50 focus-visible:ring-indigo-400/20" autoComplete="off" spellCheck={false} maxLength={512} />
                    </div>
                    <div className="flex items-center justify-end gap-2">
                        {savedKey && <Button type="button" variant="ghost" size="sm" onClick={() => setIsEditing(false)} className="h-9 text-slate-400 hover:bg-white/10 hover:text-slate-100">Annuler</Button>}
                        <Button type="submit" disabled={!key.trim()} className="h-9 min-w-32 gap-2 bg-indigo-500 text-xs text-white shadow-lg shadow-indigo-950/30 hover:bg-indigo-400">Enregistrer <ArrowRight className="h-3.5 w-3.5" /></Button>
                    </div>
                </form>
            )}
        </section>
    );
}

export default function ApiKeyForm({ onSave = () => {}, onSaveDeepSeek }) {
    const [chatGptStatus, setChatGptStatus] = useState({ available: false, connected: false });
    const [isChatGptPending, setIsChatGptPending] = useState(false);

    useEffect(() => {
        let cancelled = false;
        getChatGptStatus().then((status) => {
            if (!cancelled) setChatGptStatus(status);
        }).catch(() => {
            if (!cancelled) setChatGptStatus({ available: false, connected: false });
        });
        const unsubscribe = subscribeToChatGptAuth((status) => setChatGptStatus((previous) => ({ ...previous, ...status })));
        return () => {
            cancelled = true;
            unsubscribe();
        };
    }, []);

    const connectChatGpt = async () => {
        setIsChatGptPending(true);
        try {
            setChatGptStatus(await loginChatGpt());
            toast.success('Compte ChatGPT connecté.');
        } catch (error) {
            toast.error(error?.message || String(error));
        } finally {
            setIsChatGptPending(false);
        }
    };

    const disconnectChatGpt = async () => {
        setIsChatGptPending(true);
        try {
            setChatGptStatus(await logoutChatGpt());
            toast.info('Compte ChatGPT déconnecté.');
        } catch (error) {
            toast.error(error?.message || String(error));
        } finally {
            setIsChatGptPending(false);
        }
    };

    return (
        <div className="space-y-4">
            <div className="flex gap-3 rounded-2xl border border-sky-400/20 bg-sky-400/[0.08] p-4">
                <span className="h-fit rounded-full bg-sky-400/15 p-2"><ShieldCheck className="h-4 w-4 text-sky-300" /></span>
                <div>
                    <p className="text-sm font-semibold text-sky-100">Vos clés personnelles, stockées sur cet appareil</p>
                    <p className="mt-1 text-xs leading-relaxed text-sky-100/65">Clés conservées dans ce navigateur. Pour DeepSeek, l’image et la clé transitent par Poneglyph vers l’API, sans conservation côté serveur. Appels facturés sur votre compte fournisseur.{chatGptStatus.available ? ' La session ChatGPT reste dans l’application desktop.' : ''}</p>
                </div>
            </div>

            <ProviderKeyCard provider="Google Gemini" storageKey={GOOGLE_API_KEY} description="Utilisée pour les outils Gemini et Google Vision." keyUrl="https://aistudio.google.com/app/apikey" placeholder="Collez votre clé ici (ex : AIzaSy...)" onSave={onSave} />
            <ProviderKeyCard provider="DeepSeek" storageKey={DEEPSEEK_API_KEY_STORAGE_KEY} description={`${DEEPSEEK_LABEL} · ${DEEPSEEK_MODEL}. Transcription des bulles et extraction des textes avec bbox.`} keyUrl="https://platform.deepseek.com/api_keys" placeholder="Collez votre clé DeepSeek (sk-...)" onSave={onSaveDeepSeek} />

            {chatGptStatus.available && (
                <section className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-sm sm:p-5">
                    <div className="mb-4 flex items-start justify-between gap-4">
                        <div>
                            <Label className="text-sm font-semibold text-slate-100">Compte ChatGPT</Label>
                            <p className="mt-1 text-xs leading-relaxed text-slate-400">Utilisé par l&apos;OCR OpenAI depuis l&apos;application desktop.</p>
                        </div>
                        {chatGptStatus.connected && <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1 text-[10px] font-semibold text-emerald-200">Connecté</span>}
                    </div>

                    {chatGptStatus.connected ? (
                        <div className="flex items-center justify-between gap-3 rounded-xl border border-emerald-400/20 bg-emerald-400/[0.08] p-3">
                            <div className="min-w-0">
                                <p className="truncate text-sm font-semibold text-slate-100">{chatGptStatus.email || 'Compte ChatGPT'}</p>
                                <p className="mt-0.5 text-[11px] text-slate-400">Session active dans Poneglyph Desktop.</p>
                            </div>
                            <Button type="button" variant="ghost" size="sm" disabled={isChatGptPending} onClick={disconnectChatGpt} className="shrink-0 text-rose-300 hover:bg-rose-400/10 hover:text-rose-200">
                                {isChatGptPending ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Déconnecter'}
                            </Button>
                        </div>
                    ) : (
                        <Button type="button" disabled={isChatGptPending} onClick={connectChatGpt} className="h-11 w-full gap-2 bg-slate-100 text-slate-950 hover:bg-white">
                            {isChatGptPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogIn className="h-4 w-4" />}
                            {isChatGptPending ? 'Connexion dans le navigateur…' : 'Se connecter avec ChatGPT'}
                        </Button>
                    )}
                </section>
            )}
        </div>
    );
}
