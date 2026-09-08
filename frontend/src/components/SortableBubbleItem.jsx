import React from "react";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Button } from "@/components/ui/button";
import { GripVertical, Pencil, Trash2 } from "lucide-react";
import styles from "@/components/moderation/Review.module.css";
import { cn } from "@/lib/utils";

export const SortableBubbleItem = ({
  bubble,
  index,
  user,
  onEdit,
  onDelete,
  disabled,
  canManage,
  isAdmin,
  variant,
  selected,
  onSelect,
}) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: bubble.id, disabled: disabled });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 999 : "auto",
    position: "relative",
  };

  const canManageOwnProposedBubble =
    canManage &&
    bubble.statut === "Proposé" &&
    user &&
    bubble.id_user_createur === user.id;
  const canDelete = isAdmin || canManageOwnProposedBubble;

  if (variant === "review") {
    return (
      <li
        ref={setNodeRef}
        style={style}
        className={styles.readingItem}
        data-selected={selected}
        data-dragging={isDragging}
        id={"review-bubble-" + bubble.id}
      >
        <button
          type="button"
          {...attributes}
          {...listeners}
          className={styles.dragHandle}
          disabled={disabled}
          aria-label={"Déplacer la bulle " + (index + 1)}
          aria-roledescription="bulle déplaçable"
        >
          <GripVertical size={17} />
        </button>
        <div className="min-w-0">
          <div className={styles.itemTop}>
            <span className={styles.itemNumber}>
              Bulle {String(index + 1).padStart(2, "0")}
            </span>
            <button
              type="button"
              className={styles.quiet}
              style={{ height: 26, padding: "0 6px" }}
              disabled={disabled}
              onClick={() => onEdit(bubble)}
            >
              <Pencil size={12} />
              Corriger
            </button>
          </div>
          <button
            type="button"
            className={styles.itemText}
            onClick={() => onSelect(bubble.id)}
            aria-label={
              "Repérer la bulle " +
              (index + 1) +
              " : " +
              (bubble.texte_propose || "Sans texte")
            }
          >
            {bubble.texte_propose || "Sans texte"}
          </button>
        </div>
      </li>
    );
  }

  return (
    <li
      ref={setNodeRef}
      style={style}
      className={cn(
        "group grid grid-cols-[auto_auto_1fr_auto] items-center gap-3 p-3 bg-white border border-slate-200 rounded-lg shadow-sm transition-all w-full max-w-full box-border",
        isDragging && "opacity-50 border-dashed border-slate-400 bg-slate-50",
        !isDragging && "hover:border-slate-300 hover:shadow-md",
      )}
    >
      <div
        {...attributes}
        {...listeners}
        className={cn(
          "touch-none cursor-grab text-slate-300 hover:text-slate-500 transition-colors flex-shrink-0",
          disabled && "cursor-default opacity-50",
        )}
      >
        <GripVertical className="h-5 w-5" />
      </div>

      <span className="flex items-center justify-center h-6 w-6 rounded-full bg-slate-100 text-slate-700 text-xs font-bold flex-shrink-0">
        {index + 1}
      </span>

      <div className="min-w-0">
        <span
          className="text-sm text-slate-700 truncate font-medium block w-full"
          title={bubble.texte_propose}
        >
          {bubble.texte_propose || (
            <em className="text-slate-400">Sans texte</em>
          )}
        </span>
      </div>

      <div className="flex justify-end min-w-[60px]">
        {canDelete && (
          <div
            className={cn(
              "flex gap-1 transition-opacity",
              isAdmin ? "opacity-100" : "opacity-0 group-hover:opacity-100",
            )}
          >
            {canManageOwnProposedBubble && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                onClick={(e) => {
                  e.stopPropagation();
                  onEdit(bubble);
                }}
                title="Modifier"
              >
                <Pencil className="h-4 w-4" />
              </Button>
            )}
            {canDelete && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-red-500 hover:text-red-700 hover:bg-red-50"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(bubble.id);
                }}
                title="Supprimer"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        )}
      </div>
    </li>
  );
};
