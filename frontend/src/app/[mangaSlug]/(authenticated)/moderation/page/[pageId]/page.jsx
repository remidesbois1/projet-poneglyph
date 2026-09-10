"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useManga } from "@/context/MangaContext";
import {
  getPageById,
  getBubblesForPage,
  approvePage,
  rejectPage,
  rejectBubble,
  savePageDescription,
  getMetadataSuggestions,
  reorderBubbles,
} from "@/lib/api";
import { generatePageDescription } from "@/lib/geminiClient";
import { cn } from "@/lib/utils";
import { arrayMove } from "@dnd-kit/sortable";
import SortableBubbleList from "@/components/SortableBubbleList";
import ReviewPageImage from "@/components/moderation/ReviewPageImage";
import styles from "@/components/moderation/Review.module.css";
import { useAuth } from "@/context/AuthContext";
import ValidationForm from "@/components/ValidationForm";
import ModerationCommentModal from "@/components/ModerationCommentModal";
import AiAccessDialog from "@/components/AiAccessDialog";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import {
  X,
  ArrowLeft,
  FileText,
  Save,
  Plus,
  Check,
  GripVertical,
  Loader2,
  Sparkles,
  Code,
} from "lucide-react";

export default function PageReview() {
  const params = useParams();
  const pageId = params?.pageId;
  const router = useRouter();
  const { session, isGuest } = useAuth();
  const { mangaSlug } = useManga();

  const [page, setPage] = useState(null);
  const [bubbles, setBubbles] = useState([]);
  const [loading, setLoading] = useState(true);

  const [editingBubble, setEditingBubble] = useState(null);

  const imageRef = useRef(null);

  const [selectedBubble, setSelectedBubble] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [savingOrder, setSavingOrder] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const orderLock = useRef(false);
  const token = session?.access_token;

  const [showDescModal, setShowDescModal] = useState(false);
  const [isSavingDesc, setIsSavingDesc] = useState(false);
  const [isGeneratingAI, setIsGeneratingAI] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [formData, setFormData] = useState({
    content: "",
    arc: "",
    characters: [],
  });
  const [suggestions, setSuggestions] = useState({ arcs: [], characters: [] });
  const [charInput, setCharInput] = useState("");
  const [tabMode, setTabMode] = useState("form");
  const [jsonInput, setJsonInput] = useState("");
  const [jsonError, setJsonError] = useState(null);

  const fetchPageData = useCallback(async () => {
    if (!pageId || !token) return;
    try {
      const [pageRes, bubblesRes] = await Promise.all([
        getPageById(pageId),
        getBubblesForPage(pageId),
      ]);
      setLoadError(null);
      setPage(pageRes.data);
      let description = pageRes.data.description;
      if (typeof description === "string") {
        try {
          description = JSON.parse(description);
        } catch {
          description = { content: description };
        }
      }
      setFormData({
        content: description?.content || "",
        arc: description?.metadata?.arc || "",
        characters: Array.isArray(description?.metadata?.characters)
          ? description.metadata.characters
          : [],
      });
      setBubbles(
        [...bubblesRes.data].sort(
          (a, b) =>
            (a.order ?? Infinity) - (b.order ?? Infinity) || a.id - b.id,
        ),
      );
    } catch {
      setLoadError("Impossible de charger cette page.");
    } finally {
      setLoading(false);
    }
  }, [pageId, token]);

  // Fetch synchronizes external data; state updates happen after the request settles.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchPageData();
  }, [fetchPageData]);

  const handleDragEnd = async ({ active, over }) => {
    if (
      orderLock.current ||
      submitting ||
      page?.statut !== "pending_review" ||
      !over ||
      active.id === over.id
    )
      return;
    const from = bubbles.findIndex((b) => b.id === active.id);
    const to = bubbles.findIndex((b) => b.id === over.id);
    if (from < 0 || to < 0) return;
    const previous = bubbles;
    const ordered = arrayMove(bubbles, from, to).map((bubble, index) => ({
      ...bubble,
      order: index + 1,
    }));
    orderLock.current = true;
    setSavingOrder(true);
    setBubbles(ordered);
    setSelectedBubble(active.id);
    try {
      await reorderBubbles(
        pageId,
        ordered.map(({ id, order }) => ({ id, order })),
      );
    } catch (error) {
      setBubbles(previous);
      toast.error(
        error.response?.data?.error || "Impossible de sauvegarder cet ordre.",
      );
    } finally {
      orderLock.current = false;
      setSavingOrder(false);
    }
  };

  const selectBubble = (id) => {
    setSelectedBubble(id);
    document
      .getElementById("review-bubble-" + id)
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  const changeTab = (mode) => {
    if (mode === "json") {
      setJsonInput(
        JSON.stringify(
          {
            content: formData.content,
            metadata: { arc: formData.arc, characters: formData.characters },
          },
          null,
          4,
        ),
      );
      setJsonError(null);
    }
    setTabMode(mode);
  };

  const handleJsonChange = (e) => {
    const val = e.target.value;
    setJsonInput(val);
    try {
      const parsed = JSON.parse(val);
      if (typeof parsed !== "object" || parsed === null)
        throw new Error("Le JSON doit être un objet.");
      if (!parsed.metadata || typeof parsed.metadata !== "object")
        throw new Error("L'objet doit contenir une clé 'metadata'.");

      setJsonError(null);
      setFormData({
        content: parsed.content || "",
        arc: parsed.metadata.arc || "",
        characters: Array.isArray(parsed.metadata.characters)
          ? parsed.metadata.characters
          : [],
      });
    } catch (err) {
      setJsonError(err.message);
    }
  };

  const fetchSuggestions = useCallback(async () => {
    if (!session?.access_token) return;
    try {
      const res = await getMetadataSuggestions();
      setSuggestions(res.data);
    } catch (err) {
      console.error("Erreur suggestions:", err);
    }
  }, [session]);

  const handleSaveApiKey = (key) => {
    localStorage.setItem("google_api_key", key);
    setShowApiKeyModal(false);
  };

  const handleSaveDescription = async () => {
    const payload = {
      content: formData.content,
      metadata: { arc: formData.arc, characters: formData.characters },
    };
    setIsSavingDesc(true);

    try {
      await savePageDescription(pageId, payload);
      setPage((prev) => ({ ...prev, description: JSON.stringify(payload) }));
      setShowDescModal(false);
      toast.success("Description enregistrée !");
    } catch {
      toast.error("Erreur lors de la sauvegarde.");
    } finally {
      setIsSavingDesc(false);
    }
  };

  const handleGenerateAI = async () => {
    const storedKey = localStorage.getItem("google_api_key");
    if (!storedKey) {
      setShowApiKeyModal(true);
      return;
    }
    setIsGeneratingAI(true);
    try {
      const res = await generatePageDescription(imageRef.current, storedKey);
      const aiData = res.data;
      setFormData({
        content: aiData.content || "",
        arc: aiData.metadata?.arc || "",
        characters: Array.isArray(aiData.metadata?.characters)
          ? aiData.metadata.characters
          : [],
      });
      setJsonInput(JSON.stringify(aiData, null, 4));
      setJsonError(null);
    } catch (error) {
      console.error(error);
      toast.error("Erreur lors de la génération par IA.");
    } finally {
      setIsGeneratingAI(false);
    }
  };

  const addCharacter = (char) => {
    const cleanChar = char.trim();
    if (cleanChar && !formData.characters.includes(cleanChar)) {
      setFormData((prev) => ({
        ...prev,
        characters: [...prev.characters, cleanChar],
      }));
    }
    setCharInput("");
  };

  const removeCharacter = (char) => {
    setFormData((prev) => ({
      ...prev,
      characters: prev.characters.filter((c) => c !== char),
    }));
  };

  const handleApprove = async () => {
    if (window.confirm("Confirmer l'approbation de cette page ?")) {
      try {
        setSubmitting(true);
        await approvePage(pageId);
        router.push(`/${mangaSlug}/moderation?view=pages`);
      } catch (error) {
        alert("Erreur technique lors de l'approbation.");
        console.error(error);
      } finally {
        setSubmitting(false);
      }
    }
  };

  const [showRejectModal, setShowRejectModal] = useState(false);

  const handleReject = async (comment) => {
    try {
      setSubmitting(true);
      await rejectPage(pageId, comment);
      router.push(`/${mangaSlug}/moderation?view=pages`);
    } catch (error) {
      alert("Erreur technique lors du rejet.");
      console.error(error);
    } finally {
      setSubmitting(false);
    }
  };

  const [rejectingBubbleId, setRejectingBubbleId] = useState(null);

  const handleConfirmRejectBubble = async (comment) => {
    if (!rejectingBubbleId) return;
    try {
      await rejectBubble(rejectingBubbleId, comment);
      setRejectingBubbleId(null);
      fetchPageData();
    } catch (error) {
      alert("Erreur technique lors du rejet de la bulle.");
      console.error(error);
    }
  };

  const handleEditSuccess = () => {
    setEditingBubble(null);
    fetchPageData();
  };

  if (loading)
    return (
      <p role="status" className="p-8 text-muted-foreground">
        Chargement de la page…
      </p>
    );
  if (loadError)
    return (
      <div role="alert" className="flex items-center gap-4 p-8">
        {loadError}
        <Button variant="outline" onClick={fetchPageData}>
          Réessayer
        </Button>
      </div>
    );
  if (!page) return <p className="p-8">Page introuvable.</p>;
  const canReview = page.statut === "pending_review";
  const busy = savingOrder || submitting || isSavingDesc;

  return (
    <div className={styles.workspace}>
      <title>{"Relecture · Page " + page.numero_page}</title>
      <header className={styles.reviewHeader}>
        <div className={styles.reviewTitle}>
          <button
            type="button"
            className={styles.action}
            aria-label="Retour à la modération"
            onClick={() =>
              router.push("/" + mangaSlug + "/moderation?view=pages")
            }
          >
            <ArrowLeft />
          </button>
          <div>
            <h1>Page {page.numero_page}</h1>
            <p>
              Tome {page.chapitres?.tomes?.numero ?? "?"} · Chapitre{" "}
              {page.chapitres?.numero ?? "?"}
            </p>
          </div>
        </div>
        <div className={styles.headerActions}>
          <button
            type="button"
            className={styles.quiet}
            onClick={() => {
              setShowDescModal(true);
              fetchSuggestions();
            }}
            disabled={busy}
          >
            <FileText />
            Description
          </button>
          <button
            type="button"
            className={styles.action}
            onClick={() => setShowRejectModal(true)}
            disabled={busy || !canReview}
          >
            Refuser
          </button>
          <button
            type="button"
            className={styles.primary}
            onClick={handleApprove}
            disabled={busy || !canReview}
          >
            <Check />
            Valider la page
          </button>
        </div>
      </header>
      {!canReview && (
        <p className={styles.count}>
          Cette page n’est plus en attente de validation.
        </p>
      )}
      <div className={styles.workPanels}>
        <ReviewPageImage
          key={page.id}
          page={page}
          bubbles={bubbles}
          selectedBubble={selectedBubble}
          onSelect={selectBubble}
          imageRef={imageRef}
        />
        <aside
          className={styles.readingPanel}
          aria-label="Relecture des bulles"
        >
          <div className={styles.panelHeader}>
            <h2>Ordre de lecture</h2>
            <span className={styles.count}>{bubbles.length} bulles</span>
          </div>
          <div
            className={styles.scroll}
            data-testid="review-bubble-scroll"
            tabIndex={0}
            aria-label="Liste des bulles"
          >
            {bubbles.length === 0 ? (
              <p className={styles.empty}>Aucune bulle sur cette page.</p>
            ) : (
              <SortableBubbleList
                bubbles={bubbles}
                onDragEnd={handleDragEnd}
                className={styles.readingList}
                getItemProps={(bubble) => ({
                  variant: "review",
                  disabled: busy || !canReview,
                  selected: selectedBubble === bubble.id,
                  onSelect: setSelectedBubble,
                  onEdit: setEditingBubble,
                })}
              />
            )}
            {formData.content && (
              <section className={styles.description}>
                <h2>Description</h2>
                <p>{formData.content}</p>
                <p>
                  {[formData.arc, ...formData.characters]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
              </section>
            )}
          </div>
          <div className={styles.readingFooter} role="status">
            {savingOrder ? (
              <>
                <Loader2 size={13} className="animate-spin" />
                Enregistrement de l’ordre…
              </>
            ) : (
              <>
                <GripVertical size={13} />
                Glissez les bulles pour changer l’ordre.
              </>
            )}
          </div>
        </aside>
      </div>

      <Dialog
        open={!!editingBubble}
        onOpenChange={(open) => !open && setEditingBubble(null)}
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Correction Rapide</DialogTitle>
          </DialogHeader>
          {editingBubble && (
            <ValidationForm
              annotationData={editingBubble}
              onValidationSuccess={handleEditSuccess}
              onCancel={() => setEditingBubble(null)}
              onReject={(id) => {
                setEditingBubble(null);
                setRejectingBubbleId(id);
              }}
            />
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={showDescModal} onOpenChange={setShowDescModal}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <div className="flex items-center justify-between pr-4">
              <div>
                <DialogTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-indigo-600" />
                  Description de la page
                </DialogTitle>
                <DialogDescription>
                  Définition des métadonnées pour le moteur de recherche.
                </DialogDescription>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={handleGenerateAI}
                disabled={isGeneratingAI || isGuest}
                className="gap-2 border-indigo-200 text-indigo-700 bg-indigo-50 hover:bg-indigo-100"
              >
                {isGeneratingAI ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Générer avec IA
              </Button>
            </div>
          </DialogHeader>

          <Tabs value={tabMode} onValueChange={changeTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="form">
                <FileText className="h-4 w-4 mr-2" />
                Formulaire
              </TabsTrigger>
              <TabsTrigger value="json">
                <Code className="h-4 w-4 mr-2" />
                JSON
              </TabsTrigger>
            </TabsList>

            <TabsContent value="form" className="space-y-4 outline-none">
              <div className="flex flex-col gap-3">
                <Label
                  htmlFor="scene-content"
                  className="text-sm font-semibold text-slate-700"
                >
                  Contenu Sémantique
                </Label>
                <Textarea
                  id="scene-content"
                  value={formData.content}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      content: e.target.value,
                    }))
                  }
                  className="min-h-[120px] resize-none border-slate-200 focus:ring-indigo-500"
                  placeholder="Description de l'action, des lieux..."
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="flex flex-col gap-3">
                  <Label className="text-sm font-semibold text-slate-700">
                    Arc Narratif
                  </Label>
                  <div className="relative">
                    <input
                      list="arc-suggestions-mod"
                      value={formData.arc}
                      onChange={(e) =>
                        setFormData((prev) => ({
                          ...prev,
                          arc: e.target.value,
                        }))
                      }
                      className="flex h-10 w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm focus:ring-indigo-500 focus:outline-none focus:ring-2"
                      placeholder="Ex: Water 7"
                    />
                    <datalist id="arc-suggestions-mod">
                      {suggestions.arcs.map((arc) => (
                        <option key={arc} value={arc} />
                      ))}
                    </datalist>
                  </div>
                </div>

                <div className="flex flex-col gap-3">
                  <Label className="text-sm font-semibold text-slate-700">
                    Personnages
                  </Label>
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2">
                      <input
                        list="char-suggestions-mod"
                        value={charInput}
                        onChange={(e) => setCharInput(e.target.value)}
                        onKeyDown={(e) =>
                          e.key === "Enter" && addCharacter(charInput)
                        }
                        className="flex h-10 flex-1 rounded-md border border-slate-200 px-3 text-sm focus:ring-indigo-500 focus:outline-none focus:ring-2"
                        placeholder="Ajouter..."
                      />
                      <datalist id="char-suggestions-mod">
                        {suggestions.characters.map((c) => (
                          <option key={c} value={c} />
                        ))}
                      </datalist>
                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={() => addCharacter(charInput)}
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2 min-h-[40px] p-2 bg-slate-50 rounded border border-dashed border-slate-200">
                      {formData.characters.map((char) => (
                        <button
                          key={char}
                          type="button"
                          className="inline-flex items-center gap-2 text-sm"
                          onClick={() => removeCharacter(char)}
                          aria-label={"Retirer " + char}
                        >
                          {char} <X className="h-3 w-3" />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="json" className="outline-none">
              <div className="relative">
                <Textarea
                  value={jsonInput}
                  onChange={handleJsonChange}
                  className={cn(
                    "font-mono text-xs min-h-[350px] bg-slate-900 text-slate-50 resize-none",
                    jsonError
                      ? "border-red-500 focus:ring-red-500"
                      : "border-slate-800 focus:ring-slate-700",
                  )}
                  spellCheck={false}
                />
                {jsonError && (
                  <div className="absolute bottom-4 left-4 right-4 bg-red-500/90 text-white text-xs p-2 rounded shadow-lg backdrop-blur-sm">
                    Erreur JSON: {jsonError}
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>

          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowDescModal(false)}>
              Fermer
            </Button>
            <Button
              onClick={handleSaveDescription}
              disabled={isSavingDesc || isGeneratingAI || !!jsonError}
              className="bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm"
            >
              {isSavingDesc ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Save className="mr-2 h-4 w-4" />
              )}
              Enregistrer
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showApiKeyModal} onOpenChange={setShowApiKeyModal}>
        <AiAccessDialog onSave={handleSaveApiKey} onSaveDeepSeek={() => setShowApiKeyModal(false)} />
      </Dialog>

      <ModerationCommentModal
        isOpen={showRejectModal}
        onClose={() => setShowRejectModal(false)}
        onSubmit={handleReject}
        title="Refuser cette page"
        description="L'utilisateur verra ce commentaire sur sa page 'Mes soumissions' pour comprendre les corrections nécessaires."
      />
      <ModerationCommentModal
        isOpen={!!rejectingBubbleId}
        onClose={() => setRejectingBubbleId(null)}
        onSubmit={handleConfirmRejectBubble}
        title="Refuser cette bulle"
        description="L'indexeur verra votre commentaire pour s'améliorer."
      />
    </div>
  );
}
