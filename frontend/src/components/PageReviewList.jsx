import BulkValidationDialog from "@/components/moderation/BulkValidationDialog";
import React, { useState, useEffect, useCallback } from "react";
import { getPagesForReview, approveAllPages } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { useUserProfile } from "@/hooks/useUserProfile";
import { useManga } from "@/context/MangaContext";
import styles from "@/components/moderation/Review.module.css";
import { ArrowUpRight, Layers } from "lucide-react";
import Link from "next/link";
import ReviewPageThumbnail from "@/components/moderation/ReviewPageThumbnail";
import { Button } from "@/components/ui/button";
export default function PageReviewList() {
  const { session } = useAuth();
  const { mangaSlug } = useManga();
  const { profile } = useUserProfile();
  const [pages, setPages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [confirmationCount, setConfirmationCount] = useState(null);
  const token = session?.access_token;
  const fetchPages = useCallback(async () => {
    if (!token) return;

    try {
      const { data } = await getPagesForReview();
      setPages(
        [...data].sort(
          (a, b) =>
            (a.chapitres?.tomes?.numero ?? 0) -
              (b.chapitres?.tomes?.numero ?? 0) ||
            (a.chapitres?.numero ?? 0) - (b.chapitres?.numero ?? 0) ||
            a.numero_page - b.numero_page,
        ),
      );
      setError(null);
    } catch {
      setError("Impossible de charger les pages.");
    } finally {
      setLoading(false);
    }
  }, [token]);
  // Fetch synchronizes external data; state updates happen after the request settles.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchPages();
  }, [fetchPages]);
  const approveAll = async () => {
    setSaving(true);
    try {
      await approveAllPages();
      await fetchPages();
    } finally {
      setSaving(false);
    }
  };
  return (
    <section className={styles.queue} aria-busy={loading}>
      <div className={styles.toolbar}>
        <p className={styles.count}>
          <strong>{pages.length}</strong> pages à vérifier
        </p>
        {profile?.role === "Admin" && pages.length > 0 && (
          <Button
            className={styles.action}
            onClick={() => setConfirmationCount(pages.length)}
            disabled={saving || loading}
          >
            Tout valider
          </Button>
        )}
      </div>
      {error ? (
        <div role="alert" className="flex items-center gap-4">
          {error}
          <Button variant="outline" onClick={fetchPages}>
            Réessayer
          </Button>
        </div>
      ) : loading ? (
        <p role="status" className="py-12 text-muted-foreground">
          Chargement des pages…
        </p>
      ) : pages.length === 0 ? (
        <p className={styles.empty}>
          <Layers />
          Aucune page en attente.
        </p>
      ) : (
        <div className={styles.scroll}>
          <div className={styles.pageGrid}>
            {pages.map((page) => (
              <Link
                key={page.id}
                href={"/" + mangaSlug + "/moderation/page/" + page.id}
                prefetch={false}
                className={styles.pageCard}
              >
                <div className={styles.pageImage}>
                  <ReviewPageThumbnail page={page} />
                </div>
                <div className={styles.pageDetails}>
                  <h3>Page {page.numero_page}</h3>
                  <p className="mt-1 text-muted-foreground">
                    Tome {page.chapitres?.tomes?.numero ?? "?"} · Ch.{" "}
                    {page.chapitres?.numero ?? "?"}
                  </p>
                  <div className={styles.pageOpen}>
                    <span>Vérifier la page</span>
                    <ArrowUpRight size={15} />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
      {confirmationCount !== null && (
        <BulkValidationDialog
          resource="pages"
          count={confirmationCount}
          onClose={() => setConfirmationCount(null)}
          onConfirm={approveAll}
        />
      )}
    </section>
  );
}
