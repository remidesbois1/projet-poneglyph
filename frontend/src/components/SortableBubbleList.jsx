import React from "react";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  SortableContext,
  verticalListSortingStrategy,
  sortableKeyboardCoordinates,
} from "@dnd-kit/sortable";
import { SortableBubbleItem } from "@/components/SortableBubbleItem";

// Annotation and moderation share drag handles, keyboard controls and sorting behavior.
export default function SortableBubbleList({
  bubbles,
  onDragEnd,
  className,
  getItemProps,
}) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={onDragEnd}
      accessibility={{
        announcements: {
          onDragStart: ({ active }) =>
            "Bulle " +
            (bubbles.findIndex((b) => b.id === active.id) + 1) +
            " saisie.",
          onDragOver: ({ over }) =>
            over
              ? "Position " +
                (bubbles.findIndex((b) => b.id === over.id) + 1) +
                " sur " +
                bubbles.length +
                "."
              : "Hors de la liste.",
          onDragEnd: ({ over }) =>
            over
              ? "Bulle déposée en position " +
                (bubbles.findIndex((b) => b.id === over.id) + 1) +
                "."
              : "Déplacement annulé.",
          onDragCancel: () => "Déplacement annulé.",
        },
        screenReaderInstructions: {
          draggable:
            "Appuyez sur Espace pour saisir une bulle, utilisez les flèches pour la déplacer, puis Espace pour la déposer. Échap pour annuler.",
        },
      }}
    >
      <SortableContext
        items={bubbles.map((b) => b.id)}
        strategy={verticalListSortingStrategy}
      >
        <ol className={className}>
          {bubbles.map((bubble, index) => (
            <SortableBubbleItem
              key={bubble.id}
              bubble={bubble}
              index={index}
              {...getItemProps(bubble, index)}
            />
          ))}
        </ol>
      </SortableContext>
    </DndContext>
  );
}
