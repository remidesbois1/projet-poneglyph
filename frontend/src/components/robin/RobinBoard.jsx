"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import ArchaeologicalTerminal from "@/components/robin/ArchaeologicalTerminal";
import EvidenceWall from "@/components/robin/EvidenceWall";
import SynthesisScroll from "@/components/robin/SynthesisScroll";
import ModelStatusBar from "@/components/robin/ModelStatusBar";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function RobinBoard() {
    const [messages, setMessages] = useState([]);
    const [evidence, setEvidence] = useState([]);
    const [isThinking, setIsThinking] = useState(false);
    const [thinkingStep, setThinkingStep] = useState("");
    const [highlightedDoc, setHighlightedDoc] = useState(null);
    const [useCloud, setUseCloud] = useState(true);
    const [modelStatus, setModelStatus] = useState("idle");
    const [isStreaming, setIsStreaming] = useState(false);
    const workerRef = useRef(null);

    const fetchRAG = useCallback(async (query) => {
        setThinkingStep(`Recherche : "${query}"...`);
        try {
            const res = await fetch(`${BACKEND_URL}/search/rag?q=${encodeURIComponent(query)}&limit=6`);
            const data = await res.json();
            if (data.results && data.results.length > 0) {
                setEvidence(data.results);
            }
            return data.results || [];
        } catch (err) {
            console.error("RAG fetch error:", err);
            return [];
        }
    }, []);

    // Initialisation du worker WebGPU
    useEffect(() => {
        if (!useCloud && !workerRef.current) {
            workerRef.current = new Worker(new URL('../../workers/gemma.worker.js', import.meta.url), { type: 'module' });
            
            workerRef.current.onmessage = async (e) => {
                const { type, text, status, message, fullText, name, args } = e.data;
                
                switch (type) {
                    case 'status':
                        setModelStatus(status);
                        if (message) setThinkingStep(message);
                        break;
                    case 'token':
                        setMessages(prev => {
                            const last = prev.length - 1;
                            const copy = [...prev];
                            copy[last] = { role: "assistant", content: (copy[last].content || "") + text };
                            return copy;
                        });
                        break;
                    case 'tool_call':
                        if (name === 'search_poneglyph') {
                            const results = await fetchRAG(args.query);
                            const toolContent = results.length > 0
                                ? results.map(r => `[Doc ${r.doc_id}] (${r.context}) : ${r.content}`).join('\n\n')
                                : "Aucun résultat trouvé.";
                            
                            workerRef.current.postMessage({
                                type: 'generate',
                                messages: [], // Chat messages already in worker state or re-sent
                                toolResults: { results: toolContent }
                            });
                        }
                        break;
                    case 'done':
                        setIsThinking(false);
                        setIsStreaming(false);
                        setThinkingStep("");
                        break;
                    case 'error':
                        console.error("Worker error:", message);
                        setMessages(prev => {
                            const last = prev.length - 1;
                            const copy = [...prev];
                            copy[last] = { role: "assistant", content: `❌ Erreur modèle local : ${message}` };
                            return copy;
                        });
                        setIsThinking(false);
                        setIsStreaming(false);
                        break;
                }
            };

            workerRef.current.postMessage({ type: 'load' });
        }

        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
        };
    }, [useCloud, fetchRAG]);

    const handleSubmitCloud = useCallback(async (userQuery, currentMessages) => {
        const assistantIndex = currentMessages.length;
        setMessages([...currentMessages, { role: "assistant", content: "" }]);

        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: currentMessages,
                model: 'gemma-4-31b-it',
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: "Erreur serveur" }));
            throw new Error(err.error || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';
        let currentEvent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        switch (currentEvent) {
                            case 'thinking':
                                setThinkingStep(data.step);
                                break;
                            case 'evidence':
                                setEvidence(data.results || []);
                                break;
                            case 'token':
                                fullText += data.text;
                                setMessages(prev => {
                                    const copy = [...prev];
                                    copy[assistantIndex] = { role: "assistant", content: fullText };
                                    return copy;
                                });
                                break;
                            case 'done':
                                setMessages(prev => {
                                    const copy = [...prev];
                                    copy[assistantIndex] = { role: "assistant", content: data.fullText || fullText };
                                    return copy;
                                });
                                break;
                            case 'error':
                                throw new Error(data.message);
                        }
                    } catch (e) {
                        if (e.message && !e.message.includes('JSON')) throw e;
                    }
                    currentEvent = '';
                }
            }
        }
    }, []);

    const handleSubmit = useCallback(async (userQuery) => {
        if (!userQuery.trim() || isThinking) return;

        const newMessages = [...messages, { role: "user", content: userQuery }];
        setMessages(newMessages);
        setIsThinking(true);
        setIsStreaming(true);
        setThinkingStep("Analyse de la requête...");
        setEvidence([]);
        setHighlightedDoc(null);

        try {
            if (useCloud) {
                await handleSubmitCloud(userQuery, newMessages);
            } else {
                if (modelStatus !== 'ready') {
                    throw new Error("Le modèle local n'est pas encore prêt.");
                }
                const assistantIndex = newMessages.length;
                setMessages([...newMessages, { role: "assistant", content: "" }]);
                workerRef.current.postMessage({
                    type: 'generate',
                    messages: newMessages
                });
            }
        } catch (err) {
            console.error("Agent error:", err);
            setMessages(prev => {
                const last = prev.length - 1;
                const copy = [...prev];
                copy[last] = { role: "assistant", content: `❌ Erreur : ${err.message || "Une erreur est survenue."}` };
                return copy;
            });
            setIsThinking(false);
            setIsStreaming(false);
        } finally {
            if (useCloud) {
                setIsThinking(false);
                setIsStreaming(false);
                setThinkingStep("");
            }
        }
    }, [messages, isThinking, useCloud, handleSubmitCloud, modelStatus]);

    const handleCitationClick = useCallback((docId) => {
        setHighlightedDoc(docId);
        setTimeout(() => setHighlightedDoc(null), 3000);
    }, []);

    const getVisibleEvidence = useCallback(() => {
        if (messages.length === 0) return [];
        const lastAssistant = [...messages].reverse().find(m => m.role === "assistant");
        if (!lastAssistant) return evidence;

        const content = lastAssistant.content;
        const answerPart = content.includes("@@@answer") 
            ? content.split("@@@answer")[1] 
            : content;
        
        const citedIds = new Set();
        const outerRegex = /\[Doc\s*([^\]]+)\]/gi;
        let match;
        while ((match = outerRegex.exec(answerPart)) !== null) {
            const idsStr = match[1];
            idsStr.split(',').forEach(s => {
                const idMatch = s.match(/\d+/);
                if (idMatch) citedIds.add(parseInt(idMatch[0]));
            });
        }
        
        if (citedIds.size === 0) return isStreaming ? evidence : [];
        return evidence.filter(e => citedIds.has(e.doc_id));
    }, [messages, evidence, isStreaming]);

    const visibleEvidence = getVisibleEvidence();

    return (
        <div className="robin-page">
            <div className="robin-layout">
                <div className={`robin-main ${visibleEvidence.length > 0 ? "evidence-open" : ""}`}>
                    <SynthesisScroll
                        messages={messages}
                        isStreaming={isStreaming}
                        evidence={evidence}
                        onCitationClick={handleCitationClick}
                        onSubmit={handleSubmit}
                        isThinking={isThinking}
                        thinkingStep={thinkingStep}
                    />
                    <EvidenceWall evidence={visibleEvidence} highlightedDoc={highlightedDoc} />
                </div>
            </div>
        </div>
    );
}

