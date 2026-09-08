"use client";
import { Send } from "lucide-react";
import ConfirmationDialog from "@/components/ConfirmationDialog";

export default function SubmitPageDialog({ pageNumber, onClose, onConfirm }) {
  return (
    <ConfirmationDialog
      eyebrow={pageNumber != null ? "PAGE " + pageNumber : "VALIDATION"}
      title="Envoyer en validation ?"
      description="La page sera transmise à la modération pour vérification."
      confirmLabel="Envoyer"
      busyLabel="Envoi en cours…"
      cancelLabel="Continuer l’annotation"
      errorMessage="L’envoi a échoué. Réessayez dans un instant."
      icon={Send}
      onClose={onClose}
      onConfirm={onConfirm}
    />
  );
}
