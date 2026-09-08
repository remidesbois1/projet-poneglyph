import { CheckCheck } from "lucide-react";
import ConfirmationDialog from "@/components/ConfirmationDialog";

export default function BulkValidationDialog({
  resource,
  count,
  onClose,
  onConfirm,
}) {
  const noun =
    resource === "pages"
      ? count === 1
        ? "page"
        : "pages"
      : count === 1
        ? "bulle"
        : "bulles";
  return (
    <ConfirmationDialog
      eyebrow="MODÉRATION"
      title={
        resource === "pages"
          ? "Valider toutes les pages ?"
          : "Valider toutes les bulles ?"
      }
      description={
        count +
        " " +
        noun +
        (count === 1
          ? " en attente sera validée."
          : " en attente seront validées.")
      }
      confirmLabel="Tout valider"
      busyLabel="Validation en cours…"
      errorMessage="La validation a échoué. Réessayez dans un instant."
      icon={CheckCheck}
      onClose={onClose}
      onConfirm={onConfirm}
    />
  );
}
