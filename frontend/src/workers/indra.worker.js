import * as ort from 'onnxruntime-web';
import modelRegistry from '@poneglyph/shared/model-registry.json';
import { applyOcrTextRules } from '../lib/ocr-utils.js';
import { rgbaToGray, runIndraBubble } from '../lib/indraOcr.js';
import { downloadModel } from '../lib/modelDownload.js';

// Two small convolutional networks (0.4 + 5 MB) with a different input size per bubble:
// WASM runs them in a few milliseconds, without WebGPU shader compilation per shape.
ort.env.wasm.wasmPaths = new URL('/onnx/', self.location.origin).href;
ort.env.wasm.proxy = false;
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, Math.max(1, self.navigator?.hardwareConcurrency || 1))
    : 1;

const ARTIFACT = modelRegistry.models['indra-bubble'].artifact;
const MODEL_BASE = `https://huggingface.co/${ARTIFACT.repository}/resolve/${ARTIFACT.revision}`;

let model = null;
let queue = Promise.resolve();

async function load(onProgress) {
    if (model) return;
    const response = await fetch(`${MODEL_BASE}/onnx/browser_manifest.json`);
    if (!response.ok) throw new Error('Manifeste Indra OCR introuvable.');
    const manifest = await response.json();
    const sessions = {};
    for (const [key, label] of [['det', 'Indra détection'], ['rec', 'Indra reconnaissance']]) {
        const file = manifest.files[key];
        const bytes = await downloadModel(`${MODEL_BASE}/${file.path}`, {
            ...file,
            onProgress: progress => onProgress({ file: label, progress: (key === 'det' ? 0 : 10) + progress * (key === 'det' ? 0.1 : 0.9) }),
        });
        sessions[key] = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    }
    model = { detSession: sessions.det, recSession: sessions.rec, manifest };
}

async function grayFrom(input) {
    const bitmap = typeof ImageBitmap !== 'undefined' && input instanceof ImageBitmap ? input : await createImageBitmap(input);
    try {
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(bitmap, 0, 0);
        const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
        return { gray: rgbaToGray(data, bitmap.width, bitmap.height), width: bitmap.width, height: bitmap.height };
    } finally {
        bitmap.close?.();
    }
}

async function recognize(input) {
    const start = performance.now();
    const { gray, width, height } = await grayFrom(input);
    const decodeMs = performance.now() - start;
    const result = await runIndraBubble(ort, model, gray, width, height);
    const text = applyOcrTextRules(result.raw, model.manifest.postprocess);
    return {
        text,
        rawText: result.raw,
        lineCount: result.boxes.length,
        provider: 'wasm',
        timings: { decodeMs, ...result.timings, totalMs: performance.now() - start },
    };
}

self.addEventListener('message', ({ data }) => {
    queue = queue.catch(() => undefined).then(async () => {
        try {
            if (data.type === 'init') {
                await load(({ file, progress }) => self.postMessage({ status: 'download_progress', file, progress: Math.min(Math.max(progress, 0), 100) }));
                self.postMessage({ status: 'download_progress', file: '', progress: 100 });
                self.postMessage({ status: 'ready', modelKey: data.modelKey || 'indraBubble' });
            } else if (data.type === 'run' && (data.imageBitmap || data.imageBlob)) {
                if (!model) throw new Error('Modèle Indra OCR non chargé.');
                self.postMessage({ status: 'complete', requestId: data.requestId, ...await recognize(data.imageBitmap || data.imageBlob) });
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            self.postMessage(data.type === 'init'
                ? { status: 'error', error: `Initialisation Indra OCR impossible : ${message}` }
                : { status: 'error', error: `Erreur OCR Indra : ${message}`, requestId: data.requestId });
        }
    });
});
