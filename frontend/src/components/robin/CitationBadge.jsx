"use client";

import { useState } from "react";
import { getProxiedImageUrl } from "@/lib/utils";

export default function CitationBadge({ docId, doc, onClick }) {
    const [showHover, setShowHover] = useState(false);

    return (
        <span
            className="citation-badge"
            onClick={onClick}
            onMouseEnter={() => setShowHover(true)}
            onMouseLeave={() => setShowHover(false)}
        >
            Doc {docId}
            {showHover && doc && (
                <div className="citation-hovercard">
                    {doc.url_image && (
                        <img
                            src={getProxiedImageUrl(doc.url_image)}
                            alt={doc.context}
                            crossOrigin="anonymous"
                        />
                    )}
                    <div className="citation-hovercard-footer">{doc.context}</div>
                </div>
            )}
        </span>
    );
}
