"use client";

import EvidenceCard from "./EvidenceCard";

export default function EvidenceWall({ evidence, highlightedDoc }) {
    if (!evidence || evidence.length === 0) return <div className="evidence-wall" />;

    return (
        <div className="evidence-wall">
            <div className="evidence-wall-header">
                <div className="evidence-wall-title">
                    Pièces à Conviction
                </div>
                <div style={{ fontSize: '11px', color: 'var(--robin-text-dim)', marginTop: '4px', fontWeight: 600, textTransform: 'uppercase' }}>
                    {evidence.length} fragments identifiés
                </div>
            </div>
            <div className="evidence-wall-inner">
                {evidence.map((item) => (
                    <EvidenceCard
                        key={item.page_id}
                        item={item}
                        isHighlighted={highlightedDoc === item.doc_id}
                    />
                ))}
            </div>
        </div>
    );
}
