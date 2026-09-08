import React, { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import styles from "@/components/moderation/Review.module.css";
import { getBubbleCrop } from "@/lib/api";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

import { Check, Pencil, ImageOff, History } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getBubbleHistory } from "@/lib/api";

const BubbleReviewItem = ({ bubble, onAction, onEdit, disabled }) => {
  const { session } = useAuth();
  const [imageSrc, setImageSrc] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let localUrl;

    if (session) {
      getBubbleCrop(bubble.id)
        .then((response) => {
          if (isMounted) {
            localUrl = URL.createObjectURL(response.data);
            setImageSrc(localUrl);
          }
        })
        .catch((err) => console.error("Erreur image crop", err))
        .finally(() => {
          if (isMounted) setIsLoading(false);
        });
    }

    return () => {
      isMounted = false;
      if (localUrl) URL.revokeObjectURL(localUrl);
    };
  }, [bubble.id, session]);

  useEffect(() => {
    if (isHistoryOpen) {
      getBubbleHistory(bubble.id)
        .then((res) => setHistory(res.data))
        .catch((err) => console.error("History fetch error:", err))
        .finally(() => setLoadingHistory(false));
    }
  }, [isHistoryOpen, bubble.id]);

  const handleActionSequence = async (type) => {
    if (busy) return;
    setBusy(true);
    try {
      await onAction(type, bubble.id);
    } finally {
      setBusy(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString("fr-FR", {
      day: "2-digit",
      month: "2-digit",
      year: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className={styles.bubbleCard}>
      <div className={styles.crop}>
        {isLoading ? (
          <Skeleton className="h-24 w-full rounded" />
        ) : imageSrc ? (
          <img
            src={imageSrc}
            alt="Contexte"
            className="max-w-full max-h-[120px] object-contain rounded shadow-sm bg-white"
          />
        ) : (
          <div className="flex flex-col items-center text-slate-300 text-xs">
            <ImageOff className="h-8 w-8 mb-1" />
            <span>Image indisponible</span>
          </div>
        )}
      </div>

      <div className={styles.bubbleBody}>
        <div className={styles.bubbleMeta}>
          <div className="text-xs">Bulle proposée</div>

          <Dialog
            open={isHistoryOpen}
            onOpenChange={(open) => {
              setIsHistoryOpen(open);
              if (open) setLoadingHistory(true);
            }}
          >
            <DialogTrigger asChild>
              <Button
                variant="ghost"
                aria-label="Historique de la bulle"
                size="icon"
                className="h-6 w-6 text-slate-400 hover:text-slate-600 -mt-1 -mr-1"
              >
                <History className="h-4 w-4" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md max-h-[80vh] flex flex-col">
              <DialogHeader>
                <DialogTitle>Historique de la bulle</DialogTitle>
              </DialogHeader>
              <ScrollArea className="flex-1 pr-4">
                {loadingHistory ? (
                  <div className="space-y-2">
                    <Skeleton className="h-10 w-full" />
                    <Skeleton className="h-10 w-full" />
                  </div>
                ) : history.length === 0 ? (
                  <div className="text-center text-slate-500 py-4">
                    Aucun historique disponible.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {history.map((entry) => (
                      <div
                        key={entry.id}
                        className="text-sm border-l-2 border-slate-200 pl-3 py-1"
                      >
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-semibold text-slate-700 capitalize">
                            {entry.action.replace("_", " ")}
                          </span>
                          <span className="text-xs text-slate-400">
                            {formatDate(entry.created_at)}
                          </span>
                        </div>
                        <div className="text-xs text-slate-500 mb-1">
                          Par {entry.user?.email || "Inconnu"}
                        </div>
                        {entry.comment && (
                          <div className="text-xs text-orange-600 italic bg-orange-50 p-1 rounded">
                            « {entry.comment} »
                          </div>
                        )}
                        {entry.action === "update_text" &&
                          entry.old_data?.texte_propose && (
                            <div className="mt-1 text-xs">
                              <div className="line-through text-slate-400">
                                {entry.old_data.texte_propose}
                              </div>
                              <div className="text-green-600">
                                {entry.new_data.texte_propose}
                              </div>
                            </div>
                          )}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </DialogContent>
          </Dialog>
        </div>
        <div className={styles.bubbleText}>{bubble.texte_propose}</div>
        <div className={styles.bubbleActions}>
          <button
            type="button"
            className={styles.quiet}
            onClick={() => handleActionSequence("reject")}
            disabled={busy || disabled}
          >
            Refuser
          </button>
          <button
            type="button"
            className={styles.action}
            onClick={() => onEdit(bubble)}
            disabled={busy || disabled}
          >
            <Pencil />
            Corriger
          </button>
          <button
            type="button"
            className={styles.primary}
            onClick={() => handleActionSequence("validate")}
            disabled={busy || disabled}
          >
            <Check />
            Valider
          </button>
        </div>
      </div>
    </div>
  );
};

export default BubbleReviewItem;
