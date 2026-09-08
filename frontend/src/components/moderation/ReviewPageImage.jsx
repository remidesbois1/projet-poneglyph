import React, { useEffect, useRef, useState } from "react";
import BubbleOutline from "@/components/BubbleOutline";
import { Minus, Plus, Scan } from "lucide-react";
import { useOriginalPageImage } from "@/hooks/useOriginalPageImage";
import styles from "./Review.module.css";

export default function ReviewPageImage({
  page,
  bubbles,
  selectedBubble,
  onSelect,
  imageRef,
}) {
  const viewport = useRef(null);
  const { url, error, loading, retry } = useOriginalPageImage(page.id);
  const [bounds, setBounds] = useState({ width: 0, height: 0 });
  const [natural, setNatural] = useState(null);
  const [zoom, setZoom] = useState(null);
  const [decodeError, setDecodeError] = useState(false);
  useEffect(() => {
    const element = viewport.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() =>
      setBounds({
        width: Math.max(1, element.clientWidth - 48),
        height: Math.max(1, element.clientHeight - 48),
      }),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const fitScale =
    natural && bounds.width
      ? Math.min(
          bounds.width / natural.width,
          bounds.height / natural.height,
          1,
        )
      : 1;
  const scale = zoom ?? fitScale;
  return (
    <section className={styles.imagePanel} aria-label="Aperçu de la page">
      <div className={styles.panelHeader}>
        <h2>Page originale</h2>
        <div className={styles.imageControls}>
          <button
            type="button"
            className={styles.iconButton}
            aria-label="Dézoomer"
            disabled={scale <= 0.25}
            onClick={() => setZoom(Math.max(0.25, scale - 0.25))}
          >
            <Minus />
          </button>
          <span style={{ minWidth: 40, textAlign: "center" }}>
            {Math.round(scale * 100)} %
          </span>
          <button
            type="button"
            className={styles.iconButton}
            aria-label="Zoomer"
            disabled={scale >= 2}
            onClick={() => setZoom(Math.min(2, scale + 0.25))}
          >
            <Plus />
          </button>
          <button
            type="button"
            className={styles.iconButton}
            aria-label="Ajuster la page"
            onClick={() => setZoom(null)}
          >
            <Scan />
          </button>
        </div>
      </div>
      <div
        className={styles.imageViewport}
        ref={viewport}
        data-fit={zoom === null}
      >
        {error || decodeError ? (
          <div role="alert" className={styles.empty}>
            <p>{error || "Image indisponible."}</p>
            <button
              type="button"
              className={styles.action}
              onClick={() => {
                setDecodeError(false);
                retry();
              }}
            >
              Réessayer
            </button>
          </div>
        ) : loading ? (
          <p role="status" className={styles.empty}>
            Chargement de l’original…
          </p>
        ) : (
          <div className={styles.imageStage}>
            <img
              ref={imageRef}
              src={url}
              crossOrigin="anonymous"
              alt={"Page " + page.numero_page}
              style={
                natural
                  ? {
                      width: natural.width * scale,
                      height: natural.height * scale,
                    }
                  : { maxWidth: "100%" }
              }
              onLoad={(e) =>
                setNatural({
                  width: e.target.naturalWidth,
                  height: e.target.naturalHeight,
                })
              }
              onError={() => setDecodeError(true)}
            />
            {natural &&
              bubbles.map((bubble, index) => (
                <BubbleOutline
                  as="button"
                  bubble={bubble}
                  index={index}
                  key={bubble.id}
                  type="button"
                  className="focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                  highlighted={selectedBubble === bubble.id}
                  aria-label={"Sélectionner la bulle " + (index + 1)}
                  aria-pressed={selectedBubble === bubble.id}
                  onClick={() => onSelect(bubble.id)}
                  style={{
                    left: (bubble.x / natural.width) * 100 + "%",
                    top: (bubble.y / natural.height) * 100 + "%",
                    width: (bubble.w / natural.width) * 100 + "%",
                    height: (bubble.h / natural.height) * 100 + "%",
                  }}
                />
              ))}
          </div>
        )}
      </div>
    </section>
  );
}
