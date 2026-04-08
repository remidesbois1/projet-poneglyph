"use client";

import { useState, useRef, useCallback } from "react";
import { Search, ArrowRight, Loader2, Sparkles } from "lucide-react";

export default function ArchaeologicalTerminal({ onSubmit, isThinking, thinkingStep, useCloud, onToggleModel }) {
    const [input, setInput] = useState("");
    const textareaRef = useRef(null);

    const handleKeyDown = useCallback((e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (input.trim() && !isThinking) {
                onSubmit(input.trim());
                setInput("");
                if (textareaRef.current) textareaRef.current.style.height = "auto";
            }
        }
    }, [input, isThinking, onSubmit]);

    const handleInput = useCallback((e) => {
        setInput(e.target.value);
        const el = e.target;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }, []);

    const handleSubmit = useCallback(() => {
        if (input.trim() && !isThinking) {
            onSubmit(input.trim());
            setInput("");
            if (textareaRef.current) textareaRef.current.style.height = "auto";
        }
    }, [input, isThinking, onSubmit]);

    return (
        <div className="terminal-zone">
            <div className="terminal-input-row">
                <textarea
                    ref={textareaRef}
                    className="terminal-input"
                    value={input}
                    onChange={handleInput}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    disabled={isThinking}
                    autoFocus
                />
            </div>
        </div>
    );
}
