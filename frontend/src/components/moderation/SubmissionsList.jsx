"use client";

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { AlertCircle, ArrowUpRight, Check, ChevronLeft, ChevronRight, Clock3, FileText, Inbox, MessageSquareText, RefreshCcw, X } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useManga } from '@/context/MangaContext';
import { getMySubmissions } from '@/lib/api';
import { getPageMiniatureUrl } from '@/lib/utils';
import reviewStyles from './Review.module.css';
import styles from './Submissions.module.css';

const RESULTS_PER_PAGE = 15;
const DATE_FORMAT = new Intl.DateTimeFormat('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' });
const STATUSES = {
    'Validé': { label: 'Validé', tone: 'approved', Icon: Check },
    'Rejeté': { label: 'Rejeté', tone: 'rejected', Icon: X },
    'Proposé': { label: 'En attente', tone: 'pending', Icon: Clock3 },
};

function SubmissionCard({ submission, mangaSlug }) {
    const page = submission.pages;
    const chapter = page?.chapitres;
    const volume = chapter?.tomes;
    const manga = volume?.mangas;
    const status = STATUSES[submission.statut] || { label: 'Statut indisponible', tone: 'unknown', Icon: AlertCircle };
    const { Icon } = status;
    const preview = getPageMiniatureUrl(page?.id);
    const [failedPreview, setFailedPreview] = useState(null);
    const date = submission.created_at ? new Date(submission.created_at) : null;
    const validDate = date && !Number.isNaN(date.getTime());

    return (
        <li className={styles.card}>
            <div className={styles.preview}>
                {preview && failedPreview !== preview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={preview} alt="" loading="lazy" decoding="async" crossOrigin="anonymous" onError={() => setFailedPreview(preview)} />
                ) : <FileText aria-hidden="true" className={styles.previewFallback} />}
                <span>Page {page?.numero_page ?? '?'}</span>
            </div>
            <article className={styles.body} aria-label={`Soumission, tome ${volume?.numero ?? '?'}, chapitre ${chapter?.numero ?? '?'}, page ${page?.numero_page ?? '?'}`}>
                <div className={styles.meta}>
                    <p className={styles.location}>
                        <span>{manga?.titre || 'Manga inconnu'}</span>
                        <span className={styles.locationDetail}>Tome {volume?.numero ?? '?'} · Chapitre {chapter?.numero ?? '?'} · Page {page?.numero_page ?? '?'}</span>
                    </p>
                    <span className={styles.status} data-tone={status.tone}><Icon aria-hidden="true" />{status.label}</span>
                </div>
                <p className={styles.text}>{submission.texte_propose || 'Aucun texte proposé.'}</p>
                {submission.commentaire_moderation && (
                    <div className={styles.feedback} data-tone={status.tone}>
                        <MessageSquareText aria-hidden="true" />
                        <div>
                            <p className={styles.feedbackTitle}>{status.tone === 'rejected' ? 'Corrections demandées' : 'Retour de modération'}</p>
                            <p>{submission.commentaire_moderation}</p>
                        </div>
                    </div>
                )}
                <footer className={styles.footer}>
                    {validDate ? <time dateTime={date.toISOString()}>Soumis le {DATE_FORMAT.format(date)}</time> : <span>Date indisponible</span>}
                    {page?.id != null && (
                        <Link href={`/${manga?.slug || mangaSlug}/annotate/${page.id}`} prefetch={false} className={styles.pageLink}>
                            {status.tone === 'rejected' ? 'Reprendre la page' : 'Voir la page'}<ArrowUpRight aria-hidden="true" />
                        </Link>
                    )}
                </footer>
            </article>
        </li>
    );
}

function SubmissionHistory({ userId, mangaSlug }) {
    const [page, setPage] = useState(1);
    const [refresh, setRefresh] = useState(0);
    const [result, setResult] = useState({ key: null, submissions: [], totalCount: 0, error: null });
    const requestKey = `${page}:${refresh}`;
    const loading = result.key !== requestKey;
    const totalPages = Math.max(1, Math.ceil(result.totalCount / RESULTS_PER_PAGE));

    useEffect(() => {
        if (!userId || !mangaSlug) return;
        let active = true;
        getMySubmissions(page, RESULTS_PER_PAGE, mangaSlug)
            .then(({ data }) => {
                if (!active) return;
                const lastPage = Math.max(1, Math.ceil(data.totalCount / RESULTS_PER_PAGE));
                if (page > lastPage) {
                    setPage(lastPage);
                    return;
                }
                setResult({ key: requestKey, submissions: data.results, totalCount: data.totalCount, error: null });
            })
            .catch(() => {
                if (active) setResult(previous => ({ ...previous, key: requestKey, error: 'Impossible de charger vos soumissions. Veuillez réessayer.' }));
            });
        return () => { active = false; };
    }, [userId, mangaSlug, page, requestKey]);

    return (
        <section className={reviewStyles.queue} aria-label="Mes soumissions" aria-busy={loading}>
            <div className={styles.toolbar}>
                <div>
                    <h2>Vos contributions{result.key !== null && <span className={styles.count}>{result.totalCount}</span>}</h2>
                    <p>Suivez vos textes proposés et les retours de modération.</p>
                </div>
                <button type="button" className={styles.refresh} disabled={loading} onClick={() => setRefresh(value => value + 1)} aria-label="Actualiser mes soumissions">
                    <RefreshCcw aria-hidden="true" /><span>Actualiser</span>
                </button>
            </div>
            <div className={reviewStyles.scroll}>
                {result.error && !loading ? (
                    <div className={styles.empty} role="alert">
                        <AlertCircle aria-hidden="true" />
                        <h3>Chargement indisponible</h3>
                        <p>{result.error}</p>
                        <button type="button" className={reviewStyles.action} onClick={() => setRefresh(value => value + 1)}><RefreshCcw aria-hidden="true" />Réessayer</button>
                    </div>
                ) : loading ? (
                    <div role="status" aria-label="Chargement des soumissions">
                        <span className="sr-only">Chargement des soumissions…</span>
                        <div aria-hidden="true" className={styles.skeletonList}>
                            {[0, 1, 2, 3].map(index => <div key={index} className={styles.skeleton}><div /><div><span /><span /><span /></div></div>)}
                        </div>
                    </div>
                ) : result.submissions.length === 0 ? (
                    <div className={styles.empty}>
                        <Inbox aria-hidden="true" />
                        <h3>Aucune soumission pour le moment</h3>
                        <p>Vos propositions apparaîtront ici après leur envoi depuis une page d’annotation.</p>
                        <Link href={`/${mangaSlug}/dashboard`} prefetch={false} className={reviewStyles.action}>Explorer la bibliothèque<ArrowUpRight aria-hidden="true" /></Link>
                    </div>
                ) : (
                    <>
                        <p role="status" aria-live="polite" className="sr-only">{result.totalCount} contributions, page {page} sur {totalPages}.</p>
                        <ul className={styles.list} aria-label="Historique des soumissions">
                            {result.submissions.map(submission => <SubmissionCard key={submission.id} submission={submission} mangaSlug={mangaSlug} />)}
                        </ul>
                    </>
                )}
            </div>
            {result.totalCount > 0 && (
                <nav className={styles.pagination} aria-label="Pagination des soumissions">
                    <span className={styles.range}>{Math.min((page - 1) * RESULTS_PER_PAGE + 1, result.totalCount)}–{Math.min(page * RESULTS_PER_PAGE, result.totalCount)} sur {result.totalCount}</span>
                    <div>
                        <button type="button" className={styles.pageButton} disabled={loading || page <= 1} onClick={() => setPage(value => value - 1)} aria-label="Page précédente"><ChevronLeft aria-hidden="true" /></button>
                        <span>Page {page} <span className={styles.pageTotal}>/ {totalPages}</span></span>
                        <button type="button" className={styles.pageButton} disabled={loading || page >= totalPages} onClick={() => setPage(value => value + 1)} aria-label="Page suivante"><ChevronRight aria-hidden="true" /></button>
                    </div>
                </nav>
            )}
        </section>
    );
}

export default function SubmissionsList() {
    const { session } = useAuth();
    const { mangaSlug } = useManga();
    const userId = session?.user?.id;
    return <SubmissionHistory key={`${userId}:${mangaSlug}`} userId={userId} mangaSlug={mangaSlug} />;
}
