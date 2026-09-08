import BulkValidationDialog from "@/components/moderation/BulkValidationDialog";
import React, { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/context/AuthContext";

import { useUserProfile } from "@/hooks/useUserProfile";
import {
  getPendingBubbles,
  validateBubble,
  rejectBubble,
  validateAllBubbles,
} from "@/lib/api";
import styles from "@/components/moderation/Review.module.css";
import BubbleReviewItem from "./BubbleReviewItem";
import ValidationForm from "./ValidationForm";
import ModerationCommentModal from "./ModerationCommentModal";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

import {
  CheckCircle2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const RESULTS_PER_PAGE = 5;

const BubbleReviewList = () => {
  const { session } = useAuth();
  const { profile } = useUserProfile();

  const [pendingBubbles, setPendingBubbles] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [confirmationCount, setConfirmationCount] = useState(null);
  const [savingAll, setSavingAll] = useState(false);

  const [editingBubble, setEditingBubble] = useState(null);

  const token = session?.access_token;
  const fetchPending = useCallback(
    function fetchPending(pageToFetch) {
      if (!token) return;
      setIsLoading(true);
      return getPendingBubbles(pageToFetch, RESULTS_PER_PAGE)
        .then((response) => {
          if (!response.data.results.length && pageToFetch > 1) {
            return fetchPending(pageToFetch - 1);
          }
          setPendingBubbles(response.data.results);
          setTotalCount(response.data.totalCount);
          setCurrentPage(pageToFetch);
          setError(null);
        })
        .catch((err) => {
          console.error(err);
          setError("Impossible de récupérer les propositions.");
        })
        .finally(() => setIsLoading(false));
    },
    [token],
  );

  useEffect(() => {
    fetchPending(1);
  }, [fetchPending]);

  const [rejectingItem, setRejectingItem] = useState(null);

  const handleAction = async (action, id) => {
    if (!session) return;
    try {
      if (action === "validate") {
        await validateBubble(id);
        await fetchPending(currentPage);
      } else if (action === "reject") {
        setRejectingItem(id);
      }
    } catch (err) {
      alert(`Une erreur est survenue lors de l'action.`);
      console.error(err);
    }
  };

  const handleConfirmReject = async (comment) => {
    if (!rejectingItem) return;
    await rejectBubble(rejectingItem, comment);
    setRejectingItem(null);
    fetchPending(currentPage);
  };

  const handleValidateAll = async () => {
    setSavingAll(true);
    try {
      await validateAllBubbles();
      await fetchPending(1);
    } finally {
      setSavingAll(false);
    }
  };

  const handleEditSuccess = () => {
    setEditingBubble(null);
    fetchPending(currentPage);
  };

  const totalPages = Math.ceil(totalCount / RESULTS_PER_PAGE);

  if (isLoading && pendingBubbles.length === 0) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-40 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 text-red-600 p-4 rounded-lg flex items-center gap-2 border border-red-100">
        <AlertCircle className="h-5 w-5" />
        {error}
        <Button variant="outline" onClick={() => fetchPending(currentPage)}>
          Réessayer
        </Button>
      </div>
    );
  }

  return (
    <div className={styles.queue}>
      <div className={styles.toolbar}>
        <p className={styles.count}>
          <strong>{totalCount}</strong> bulles à vérifier
        </p>

        {profile?.role === "Admin" && totalCount > 0 && (
          <Button
            onClick={() => setConfirmationCount(totalCount)}
            disabled={isLoading || savingAll}
            variant="default"
            className={styles.action}
          >
            <CheckCircle2 className="mr-2 h-4 w-4" />
            Tout valider
          </Button>
        )}
      </div>

      {pendingBubbles.length === 0 ? (
        <p className={styles.empty}>Aucune bulle en attente.</p>
      ) : (
        <div className={styles.scroll}>
          <div className={styles.bubbleQueue}>
            {pendingBubbles.map((bubble) => (
              <BubbleReviewItem
                key={bubble.id}
                bubble={bubble}
                onAction={handleAction}
                onEdit={setEditingBubble}
                disabled={isLoading}
              />
            ))}
          </div>
        </div>
      )}

      {totalPages > 1 && (
        <div className={styles.pagination}>
          <Button
            variant="outline"
            onClick={() => fetchPending(currentPage - 1)}
            disabled={isLoading || currentPage === 1}
            className="w-32"
          >
            <ChevronLeft className="mr-2 h-4 w-4" /> Précédent
          </Button>

          <span className="text-sm font-medium text-slate-600">
            Page {currentPage} / {totalPages}
          </span>

          <Button
            variant="outline"
            onClick={() => fetchPending(currentPage + 1)}
            disabled={isLoading || currentPage === totalPages}
            className="w-32"
          >
            Suivant <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      )}

      <Dialog
        open={!!editingBubble}
        onOpenChange={(open) => !open && setEditingBubble(null)}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Correction de la proposition</DialogTitle>
          </DialogHeader>

          {editingBubble && (
            <ValidationForm
              annotationData={editingBubble}
              onValidationSuccess={handleEditSuccess}
              onCancel={() => setEditingBubble(null)}
              onReject={(id) => {
                setEditingBubble(null);
                setRejectingItem(id);
              }}
            />
          )}
        </DialogContent>
      </Dialog>
      <ModerationCommentModal
        isOpen={!!rejectingItem}
        onClose={() => setRejectingItem(null)}
        onSubmit={handleConfirmReject}
        title="Refuser cette bulle"
        description="L'indexeur verra votre commentaire pour s'améliorer."
      />
      {confirmationCount !== null && (
        <BulkValidationDialog
          resource="bulles"
          count={confirmationCount}
          onClose={() => setConfirmationCount(null)}
          onConfirm={handleValidateAll}
        />
      )}
    </div>
  );
};

export default BubbleReviewList;
