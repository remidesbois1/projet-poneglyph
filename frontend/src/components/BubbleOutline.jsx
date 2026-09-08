import React from "react";
import { cn } from "@/lib/utils";

// Shared by annotation and review so status colors, borders and numbering stay identical.
export default function BubbleOutline({
  as: Element = "div",
  bubble,
  index,
  highlighted = false,
  className,
  children,
  ...props
}) {
  const validated = bubble.statut === "Validé";
  return (
    <Element
      {...props}
      className={cn(
        "absolute border-2 z-10 transition-colors cursor-pointer group",
        validated
          ? "border-emerald-500 bg-emerald-500/10 hover:bg-emerald-500/20"
          : "border-amber-500 bg-amber-500/10 hover:bg-amber-500/20",
        highlighted && (validated ? "bg-emerald-500/20" : "bg-amber-500/20"),
        className,
      )}
    >
      <span
        className={cn(
          "absolute -top-6 -left-[2px] text-white text-[10px] font-bold px-1.5 py-0.5 rounded shadow-sm",
          validated ? "bg-emerald-500" : "bg-amber-500",
        )}
      >
        #{index + 1}
      </span>
      {children}
    </Element>
  );
}
