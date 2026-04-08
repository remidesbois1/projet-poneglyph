"use client";

import { useRef, useEffect } from "react";
import { getProxiedImageUrl } from "@/lib/utils";

export default function EvidenceCard({ item, isHighlighted }) {
    const cardRef = useRef(null);

    useEffect(() => {
        if (isHighlighted && cardRef.current) {
            cardRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    }, [isHighlighted]);

    const imageUrl = item.url_image?.startsWith("/api/")
        ? `${process.env.NEXT_PUBLIC_BACKEND_URL}${item.url_image.replace("/api", "")}`
        : getProxiedImageUrl(item.url_image);

    return (
        <div
            ref={cardRef}
            className={`evidence-card ${isHighlighted ? "highlighted" : ""}`}
            id={`evidence-doc-${item.doc_id}`}
        >
            {item.url_image && (
                <img
                    className={`evidence-card-image ${item.type === "bubble" ? "is-crop" : ""}`}
                    src={imageUrl}
                    alt={item.context}
                    loading="lazy"
                    crossOrigin="anonymous"
                />
            )}
        </div>
    );
}
