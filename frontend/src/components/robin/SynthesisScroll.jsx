"use client";

import { useRef, useEffect } from "react";
import { Search } from "lucide-react";
import CitationBadge from "./CitationBadge";

function parseContent(text, evidence, onCitationClick) {
    if (!text) return null;
    const parts = [];
    const outerRegex = /\[Doc\s*([^\]]+)\]/gi;
    let lastIndex = 0;
    let match;

    while ((match = outerRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
        }
        
        const idsStr = match[1]; // e.g. "1, 2, 5"
        const ids = idsStr.split(',').map(s => parseInt(s.match(/\d+/)?.[0])).filter(Boolean);
        
        ids.forEach((docId, i) => {
            const doc = evidence.find(e => e.doc_id === docId);
            parts.push(
                <CitationBadge
                    key={`cite-${match.index}-${docId}`}
                    docId={docId}
                    doc={doc}
                    onClick={() => onCitationClick(docId)}
                />
            );
            if (i < ids.length - 1) parts.push(", ");
        });
        
        lastIndex = outerRegex.lastIndex;
    }

    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }

    return parts;
}

function MessageContent({ text, evidence, onCitationClick, isLast, isStreaming }) {
    const visibleText = text.includes("@@@answer") 
        ? text.split("@@@answer")[1] 
        : text;

    const paragraphs = visibleText.split("\n").filter(Boolean);

    return (
        <>
            {paragraphs.map((p, i) => (
                <div key={i} className="mb-2">
                    {parseContent(p, evidence, onCitationClick)}
                    {isLast && isStreaming && i === paragraphs.length - 1 && (
                        <span className="streaming-cursor" />
                    )}
                </div>
            ))}
            {paragraphs.length === 0 && isLast && isStreaming && (
                <div className="mb-2"><span className="streaming-cursor" /></div>
            )}
        </>
    );
}

import ArchaeologicalTerminal from "./ArchaeologicalTerminal";

export default function SynthesisScroll({ 
    messages, 
    isStreaming, 
    evidence, 
    onCitationClick,
    onSubmit,
    isThinking,
    thinkingStep
}) {
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isThinking]);

    return (
        <div className="synthesis-scroll" ref={scrollRef}>
            {messages.map((msg, i) => (
                <div
                    key={i}
                    className={`message-bubble ${msg.role === "user" ? "message-user" : "message-assistant"}`}
                >
                    {msg.role === "user" ? (
                        <div className="message-user-inner">{msg.content}</div>
                    ) : (
                        <div className="message-assistant">
                            <MessageContent
                                text={msg.content}
                                evidence={evidence}
                                onCitationClick={onCitationClick}
                                isLast={i === messages.length - 1}
                                isStreaming={isStreaming}
                            />
                        </div>
                    )}
                </div>
            ))}
            {!isStreaming && (
                <div className="message-bubble message-user">
                    <ArchaeologicalTerminal 
                        onSubmit={onSubmit} 
                        isThinking={isThinking} 
                        thinkingStep={thinkingStep} 
                    />
                </div>
            )}
            {isStreaming && (
                <div className="message-bubble message-assistant">
                    <div className="thinking-ink">En cours de rédaction...</div>
                </div>
            )}
        </div>
    );
}
