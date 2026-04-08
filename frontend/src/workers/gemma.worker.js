import { pipeline, TextStreamer } from '@huggingface/transformers';

const MODEL_ID = 'onnx-community/gemma-4-E2B-it-ONNX';

let generator = null;
let isLoading = false;

const SYSTEM_PROMPT = `Tu es Robin, l'archéologue des Ponéglyphes. Tu réponds aux questions sur One Piece.
REDIRECTIONS : Produis TOUJOURS ta synthèse finale après le marqueur @@@ANSWER@@@. Tout ce qui précède sera ignoré (références, pensées). Réponds DIRECTEMENT avec les faits et cite sources avec [Doc X].`;

const TOOLS_SCHEMA = [{
    type: "function",
    function: {
        name: "search_poneglyph",
        description: "Recherche sémantique dans l'archive de mangas One Piece",
        parameters: {
            type: "object",
            properties: {
                query: { type: "string", description: "Requête de recherche" }
            },
            required: ["query"]
        }
    }
}];

async function loadModel() {
    if (generator || isLoading) return;
    isLoading = true;

    self.postMessage({ type: 'status', status: 'loading', message: 'Chargement du modèle WebGPU...' });

    try {
        generator = await pipeline('text-generation', MODEL_ID, {
            device: 'webgpu',
            dtype: 'q4',
        });
        self.postMessage({ type: 'status', status: 'ready', message: 'Modèle prêt' });
    } catch (err) {
        self.postMessage({ type: 'status', status: 'error', message: err.message });
        generator = null;
    } finally {
        isLoading = false;
    }
}

async function generate(messages, toolResults = null) {
    if (!generator) {
        self.postMessage({ type: 'error', message: 'Modèle non chargé' });
        return;
    }

    try {
        const chatMessages = [
            { role: 'system', content: SYSTEM_PROMPT },
            ...messages
        ];

        if (toolResults) {
            chatMessages.push({
                role: 'tool',
                content: JSON.stringify(toolResults),
            });
        }

        let fullText = '';

        const output = await generator(chatMessages, {
            max_new_tokens: 1024,
            temperature: 0.7,
            do_sample: true,
            tools: toolResults ? undefined : TOOLS_SCHEMA,
            return_full_text: false,
            streamer: new TextStreamer(generator.tokenizer, {
                skip_prompt: true,
                callback_function: (text) => {
                    fullText += text;
                    const ANSWER_MARKER = "@@@ANSWER@@@";
                    const markerIndex = fullText.indexOf(ANSWER_MARKER);
                    
                    if (markerIndex !== -1) {
                        // On a trouvé le marker
                        const afterMarker = fullText.substring(markerIndex + ANSWER_MARKER.length);
                        // On n'envoie que si on vient juste de le trouver ou si on est déjà après
                        // Note: le streamer peut envoyer des morceaux. C'est délicat.
                        // On va tricher : si on vient de trouver le marker, on envoie tout ce qui suit.
                        // Sinon on envoie simplement le texte courant si on est déjà "en mode réponse"
                        const isJustFound = fullText.length - text.length <= markerIndex + ANSWER_MARKER.length;
                        if (isJustFound) {
                             if (afterMarker) self.postMessage({ type: 'token', text: afterMarker.trimStart() });
                        } else {
                             self.postMessage({ type: 'token', text });
                        }
                    }
                }
            }),
        });

        const generated = output[0]?.generated_text || fullText;

        const toolCallMatch = generated.match(/"name"\s*:\s*"search_poneglyph".*?"query"\s*:\s*"([^"]+)"/s);
        if (toolCallMatch) {
            self.postMessage({
                type: 'tool_call',
                name: 'search_poneglyph',
                args: { query: toolCallMatch[1] }
            });
            return;
        }

        const ANSWER_MARKER = "@@@ANSWER@@@";
        const finalAns = fullText.includes(ANSWER_MARKER) 
            ? fullText.substring(fullText.indexOf(ANSWER_MARKER) + ANSWER_MARKER.length).trimStart()
            : fullText;

        self.postMessage({ type: 'done', fullText: finalAns });
    } catch (err) {
        self.postMessage({ type: 'error', message: err.message });
    }
}

self.onmessage = async (event) => {
    const { type, messages, toolResults } = event.data;

    switch (type) {
        case 'load':
            await loadModel();
            break;
        case 'generate':
            await generate(messages, toolResults);
            break;
        case 'abort':
            break;
    }
};
