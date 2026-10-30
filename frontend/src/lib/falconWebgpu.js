import * as ort from 'onnxruntime-web/webgpu';
import { PreTrainedTokenizer } from '@huggingface/transformers';
import { prepareImage, rotaryPositions } from './falconPreprocessing';
import { downloadModel } from './modelDownload';
import { FalconFusedDecoder } from './falconFusedDecoder';
import falconDecodeWeights from './falconDecodeWeights.json';

export const FALCON_MODEL_BASE = process.env.NEXT_PUBLIC_FALCON_MODEL_BASE ||
    'https://huggingface.co/Remidesbois/Falcon-OCR-Poneglyph/resolve/abd33103ab9fef627d89ef430e1ecb1c15fb7194/onnx';

const FALCON_LARGEST_WEIGHT = 65536 * 768 * 2;
export function canUseFalconFused(manifest, device) {
    return falconDecodeWeights.schema === 1 && falconDecodeWeights.source_sha256 === manifest.source_sha256 &&
        falconDecodeWeights.graph_sha256 === manifest.files?.['decoder.onnx']?.sha256 &&
        falconDecodeWeights.data_sha256 === manifest.files?.['decoder.onnx.data']?.sha256 &&
        device.features.has('subgroups') && device.limits.maxStorageBufferBindingSize >= FALCON_LARGEST_WEIGHT &&
        device.limits.maxBufferSize >= FALCON_LARGEST_WEIGHT;
}

class FalconFusedFailure extends Error {
    constructor(cause) { super('Le décodeur Falcon accéléré a échoué.', { cause }); }
}

export class FalconWebGPU {
    async load(onProgress = () => {}, options = {}) {
        if (this.session) return;
        if (!navigator.gpu) throw new Error('WebGPU est indisponible dans ce navigateur.');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter?.features.has('shader-f16')) throw new Error('Ce GPU ne prend pas en charge WebGPU FP16.');
        this.gpuName = adapter.info?.description || adapter.info?.device || adapter.info?.architecture || 'WebGPU';
        ort.env.wasm.wasmPaths = new URL('/onnx/', self.location.origin).href;
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.webgpu.powerPreference = 'high-performance';
        onProgress({ file: 'Configuration Falcon', progress: 0 });
        let modelBase = FALCON_MODEL_BASE;
        if (!process.env.NEXT_PUBLIC_FALCON_MODEL_BASE && process.env.NODE_ENV === 'development') {
            const localBase = new URL('/models/falcon-ocr', self.location.origin).href;
            const local = await fetch(`${localBase}/browser_manifest.json`, { method: 'HEAD' }).catch(() => null);
            if (local?.ok) modelBase = localBase;
        }
        const base = new URL(modelBase, self.location.origin).href.replace(/\/$/, '');
        const getJson = async name => { const r = await fetch(`${base}/${name}`); if (!r.ok) throw new Error(`Fichier Falcon manquant : ${name}`); return r.json(); };
        const [manifest, tokenizer, tokenizerConfig] = await Promise.all([
            getJson('browser_manifest.json'), getJson('tokenizer.json'), getJson('tokenizer_config.json')]);
        this.manifest = manifest;
        this.fusedFallback = null;
        this.options = { tokenBatch: 8, mode: 'auto', ...options };
        this.tokenizer = new PreTrainedTokenizer(tokenizer, tokenizerConfig);
        const graph = await downloadModel(`${base}/decoder.onnx`, manifest.files['decoder.onnx']);
        const data = await downloadModel(`${base}/decoder.onnx.data`, {
            ...manifest.files['decoder.onnx.data'],
            onProgress: progress => onProgress({ file: 'Poids Falcon ONNX', progress }),
        });
        onProgress({ file: 'Initialisation WebGPU', progress: 99 });
        this.session = await ort.InferenceSession.create(graph, {
            executionProviders: ['webgpu'],
            externalData: [{ path: 'decoder.onnx.data', data }],
            preferredOutputLocation: { present_key_values: 'gpu-buffer', next_token: 'cpu', logits: 'cpu' },
            graphOptimizationLevel: 'all',
            extra: { session: { disable_cpu_ep_fallback: '1' } },
        });
        if (this.options.mode !== 'onnx') {
            const device = await ort.env.webgpu.device;
            if (canUseFalconFused(manifest, device)) {
                onProgress({ file: 'Préparation du décodeur Falcon accéléré', progress: 99 });
                this.fused = new FalconFusedDecoder(device, this.options);
                try { await this.fused.load(falconDecodeWeights, data); }
                catch (error) {
                    const lost = this.fused.deviceLost;
                    this.fused.dispose(); this.fused = null;
                    if (this.options.mode === 'fused' || lost) { await this.dispose(); throw error; }
                    this.fusedFallback = error?.message || String(error);
                }
            } else if (this.options.mode === 'fused') {
                await this.dispose(); throw new Error('Modèle ou GPU incompatible avec le décodeur Falcon accéléré.');
            }
        }
        onProgress({ file: 'Falcon prêt', progress: 100 });
    }

    async recognize(imageInput, onToken = () => {}) {
        if (!this.session) throw new Error('Falcon n’est pas chargé.');
        const start = performance.now();
        const bitmap = imageInput instanceof ImageBitmap ? imageInput : await createImageBitmap(imageInput);
        let prepared;
        try { prepared = await prepareImage(this.manifest, bitmap); }
        finally { bitmap.close(); }
        const { config, max_new_tokens: cap } = this.manifest;
        const prefix = prepared.ids.length;
        if (prefix + cap > config.max_seq_len) throw new Error('Cette image dépasse le contexte de Falcon.');
        try { return await this.recognizePrepared(prepared, start, onToken, this.fused); }
        catch (error) {
            if (!(error instanceof FalconFusedFailure) || this.options.mode === 'fused' || this.fused?.deviceLost) throw error;
            this.fusedFallback = error.cause?.message || String(error.cause);
            this.fused?.dispose(); this.fused = null;
            onToken('');
            return this.recognizePrepared(prepared, start, onToken, null);
        }
    }

    async recognizePrepared(prepared, start, onToken, fused) {
        const { config, max_new_tokens: cap, stop_ids: stops } = this.manifest;
        const prefix = prepared.ids.length;
        let cache = new ort.Tensor('float16', new Uint16Array(0), [config.n_layers, 2, 1, 16, 0, 64]);
        let length = 0, next = null, finished = false, prefillMs = 0, setupMs = 0, lastStream = -Infinity;
        const generated = [];
        const pendingTokens = [];
        try {
            for (let step = 0; step < cap; step++) {
                if (step > 0 && fused) {
                    try {
                        if (step === 1) {
                            const setupStart = performance.now();
                            await fused.begin(cache, length, prepared.lastPosition + 1, next, cap);
                            setupMs = performance.now() - setupStart;
                        }
                        if (!pendingTokens.length) {
                            const count = step <= 4 ? 1 : Math.min(Math.max(1, Number(this.options.tokenBatch) || 8), 8, cap - step);
                            pendingTokens.push(...await fused.step(count));
                        }
                        next = pendingTokens.shift();
                    } catch (error) { throw new FalconFusedFailure(error); }
                } else {
                    const seq = step === 0 ? prefix : 1;
                    const rotation = step === 0 ? prepared : rotaryPositions(this.manifest, [prepared.lastPosition + step]);
                    const feeds = {
                        input_ids: new ort.Tensor('int64', BigInt64Array.from(step === 0 ? prepared.ids : [next], BigInt), [1, seq]),
                        pixel_values: new ort.Tensor('float16', step === 0 ? prepared.pixels : new Uint16Array(768), [1, seq, 768]),
                        pixel_mask: new ort.Tensor('bool', step === 0 ? prepared.mask : new Uint8Array(1), [1, seq, 1]),
                        rope_cos: new ort.Tensor('float32', rotation.cos, [1, seq, 16, 32]),
                        rope_sin: new ort.Tensor('float32', rotation.sin, [1, seq, 16, 32]),
                        attention_mask: new ort.Tensor('float16', step === 0 ? prepared.attention : new Uint16Array(length + 1), [1, 1, seq, length + seq]),
                        sink_template: new ort.Tensor('float16', new Uint16Array(16 * seq), [1, 16, seq, 1]),
                        past_key_values: cache,
                    };
                    let outputs;
                    try { outputs = await this.session.run(feeds, ['next_token', 'present_key_values']); }
                    finally { for (const [name, tensor] of Object.entries(feeds)) if (name !== 'past_key_values') tensor.dispose(); }
                    cache.dispose(); cache = outputs.present_key_values;
                    next = Number(outputs.next_token.data[0]); outputs.next_token.dispose();
                    if (step === 0) prefillMs = performance.now() - start;
                    length += seq;
                }
                if (stops.includes(next)) { finished = true; break; }
                generated.push(next);
                if (performance.now() - lastStream >= 32) {
                    onToken(this.tokenizer.decode(generated, { skip_special_tokens: false, clean_up_tokenization_spaces: false }));
                    lastStream = performance.now();
                }
            }
            const rawText = this.tokenizer.decode(generated, { skip_special_tokens: false, clean_up_tokenization_spaces: false });
            return { text: rawText.trim(), rawText,
                provider: 'webgpu', gpu: this.gpuName, tokenLimit: !finished,
                implementation: fused ? 'wgsl-fused' : 'onnx', diagnostics: fused?.stats(),
                accelerationFallback: this.fusedFallback || undefined,
                timings: { totalMs: Math.round(performance.now() - start), prefillMs: Math.round(prefillMs), setupMs: Math.round(setupMs) }, tokens: generated.length };
        } finally { cache.dispose(); fused?.end(); }
    }

    async dispose() { this.fused?.dispose(); this.fused = null; await this.session?.release(); this.session = null; this.tokenizer = null; }
}
