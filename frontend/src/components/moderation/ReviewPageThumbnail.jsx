import React from "react";
import { useOriginalPageImage } from "@/hooks/useOriginalPageImage";
export default function ReviewPageThumbnail({ page }) {
  const { url, error } = useOriginalPageImage(page.id, {
    thumbnail: true,
    width: 480,
  });
  if (error)
    return <span className="text-xs text-slate-400">Aperçu indisponible</span>;
  if (!url)
    return (
      <span className="text-xs text-slate-400" role="status">
        Chargement…
      </span>
    );
  return <img src={url} alt={"Page " + page.numero_page} loading="lazy" />;
}
