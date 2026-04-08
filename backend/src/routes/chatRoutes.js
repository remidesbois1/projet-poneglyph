const express = require('express');
const router = express.Router();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { supabase } = require('../config/supabaseClient');
const { generateVoyageEmbedding } = require('../utils/voyageClient');
const { generateGeminiEmbedding } = require('../utils/geminiClient');

const DUAL_OVERLAP_BONUS = 1.15;

const SYSTEM_PROMPT = `Tu es Robin, l'archéologue des Ponéglyphes. Tu réponds aux questions sur One Piece en utilisant l'outil search_poneglyph.

Règles de sortie CRITIQUES :
1. Produis TOUJOURS ta synthèse finale après le marqueur @@@ANSWER@@@.
2. Tout ce qui précède ce marqueur sera ignoré (tu peux y mettre tes réflexions).
3. Ta synthèse après @@@ANSWER@@@ doit être directe, sans blabla ("D'après les docs", "En regardant...").
4. Cite tes sources avec le format [Doc X].
5. Réponds toujours en français de manière érudite et précise.`;

const SEARCH_TOOL = {
    functionDeclarations: [{
        name: "search_poneglyph",
        description: "Recherche sémantique et textuelle dans l'archive One Piece. Retourne des pages entières et des bulles de texte spécifiques (crops).",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "La requête de recherche (ex: 'Qui est Joy Boy ?', 'Combat Luffy vs Kaido')"
                }
            },
            required: ["query"]
        }
    }]
};

async function performRAGSearch(query, limit = 8) {
    const [voyageResults, geminiResults, bubbleResults] = await Promise.all([
        (async () => {
            try {
                const embedding = await generateVoyageEmbedding(query, "query");
                const { data, error } = await supabase.rpc('match_pages', {
                    query_embedding: embedding, match_threshold: 0.30, match_count: 20,
                });
                if (error) throw error;
                return data || [];
            } catch { return []; }
        })(),
        (async () => {
            try {
                const embedding = await generateGeminiEmbedding(query, "RETRIEVAL_QUERY");
                const { data, error } = await supabase.rpc('match_pages_gemini', {
                    query_embedding: embedding, match_threshold: 0.30, match_count: 20,
                });
                if (error) throw error;
                return data || [];
            } catch { return []; }
        })(),
        (async () => {
            try {
                // Keyword search in bubbles
                const { data, error } = await supabase
                    .from('bulles')
                    .select(`
                        id, texte_propose, x, y, w, h, id_page,
                        pages ( numero_page, chapters:id_chapitre ( numero, tomes:id_tome ( numero ) ) )
                    `)
                    .textSearch('fts', query.split(' ').join(' & '), { config: 'french' })
                    .limit(5);
                if (error) throw error;
                return data || [];
            } catch { return []; }
        })()
    ]);

    const pageMap = new Map();
    for (const p of voyageResults) {
        pageMap.set(p.id, { ...p, sources: ['voyage'], bestSimilarity: p.similarity });
    }
    for (const p of geminiResults) {
        if (pageMap.has(p.id)) {
            const existing = pageMap.get(p.id);
            existing.sources.push('gemini');
            existing.bestSimilarity = Math.max(existing.bestSimilarity, p.similarity);
        } else {
            pageMap.set(p.id, { ...p, sources: ['gemini'], bestSimilarity: p.similarity });
        }
    }
    for (const [, entry] of pageMap) {
        entry.similarity = entry.sources.length > 1
            ? Math.min(entry.bestSimilarity * DUAL_OVERLAP_BONUS, 1.0)
            : entry.bestSimilarity;
    }

    const sortedPages = Array.from(pageMap.values())
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5);

    const mergedEvidence = [];
    let docIdCounter = 1;

    // Add Bubble evidence first (specific)
    for (const b of bubbleResults) {
        const tome = b.pages?.chapters?.tomes?.numero || "?";
        const chap = b.pages?.chapters?.numero || "?";
        mergedEvidence.push({
            doc_id: docIdCounter++,
            type: 'bubble',
            bubble_id: b.id,
            url_image: `/api/bulles/${b.id}/crop`,
            context: `Bulle : Tome ${tome} - Chap. ${chap} - Page ${b.pages?.numero_page}`,
            content: b.texte_propose || "",
            similarity: 95,
        });
    }

    // Add Page evidence
    for (const p of sortedPages) {
        let snippet = p.description;
        try {
            if (typeof snippet === 'string') snippet = JSON.parse(snippet).content;
            else if (typeof snippet === 'object') snippet = snippet.content;
        } catch { }

        mergedEvidence.push({
            doc_id: docIdCounter++,
            type: 'page',
            page_id: p.id,
            url_image: p.url_image,
            context: `Page : Tome ${p.tome_numero} - Chap. ${p.chapitre_numero} - Page ${p.numero_page}`,
            content: snippet || "",
            similarity: Math.round(p.similarity * 100),
        });
    }

    return mergedEvidence.slice(0, limit);
}


router.post('/', async (req, res) => {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) return res.status(500).json({ error: "GOOGLE_API_KEY non configurée" });

    const { messages = [], model = "gemma-4-31b-it" } = req.body;
    if (!messages.length) return res.status(400).json({ error: "Messages requis" });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    if (res.flushHeaders) res.flushHeaders();

    const send = (event, data) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    try {
        const genAI = new GoogleGenerativeAI(apiKey);
        const modelInstance = genAI.getGenerativeModel({
            model,
            systemInstruction: SYSTEM_PROMPT,
            tools: [SEARCH_TOOL]
        });

        const chat = modelInstance.startChat({
            history: messages.slice(0, -1).map(m => ({
                role: m.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: m.content }]
            })),
        });

        const lastMessage = messages[messages.length - 1];

        send('thinking', { step: "Analyse de la requête..." });

        const result = await chat.sendMessage(lastMessage.content);
        const response = await result.response;
        const call = response.functionCalls()?.[0];

        if (call && call.name === "search_poneglyph") {
            const query = call.args.query;
            send('thinking', { step: `Recherche : "${query}"...` });
            send('tool_call', { name: call.name, args: call.args });

            const ragResults = await performRAGSearch(query);
            send('evidence', { results: ragResults });

            send('thinking', { step: "Rédaction de la synthèse..." });

            const toolResponseContent = ragResults.length > 0
                ? ragResults.map(r => `[Doc ${r.doc_id}] (${r.context}) : ${r.content}`).join('\n\n')
                : "Aucun résultat trouvé dans l'archive.";

            const secondResult = await chat.sendMessageStream([{
                functionResponse: {
                    name: call.name,
                    response: { results: toolResponseContent }
                }
            }]);

            let fullText = '';
            let isAnswerStarted = false;
            const ANSWER_MARKER = "@@@ANSWER@@@";

            for await (const chunk of secondResult.stream) {
                const text = chunk.text();
                if (!text) continue;

                fullText += text;

                if (!isAnswerStarted) {
                    const markerIndex = fullText.indexOf(ANSWER_MARKER);
                    if (markerIndex !== -1) {
                        isAnswerStarted = true;
                        const actualText = fullText.substring(markerIndex + ANSWER_MARKER.length);
                        if (actualText) send('token', { text: actualText.trimStart() });
                    }
                } else {
                    send('token', { text });
                }
            }
            send('done', { fullText: isAnswerStarted ? fullText.substring(fullText.indexOf(ANSWER_MARKER) + ANSWER_MARKER.length).trimStart() : fullText });
        } else {
            const text = response.text();
            if (text) {
                send('token', { text });
                send('done', { fullText: text });
            } else {
                send('error', { message: "Réponse vide du modèle" });
            }
        }

    } catch (err) {
        console.error('[Chat] Error:', err);
        send('error', { message: err.message || "Erreur serveur" });
    }

    res.end();
});

module.exports = router;
