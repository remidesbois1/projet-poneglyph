import * as ort from 'onnxruntime-web/webgpu';
import { PreTrainedTokenizer } from '@huggingface/transformers';
import { half } from './falconPreprocessing';
import { prepareSuryaImage, suryaRotary, cleanSuryaText } from './suryaPreprocessing';
import { downloadModel } from './modelDownload';
import { SuryaFusedDecoder } from './suryaFusedDecoder';

export const SURYA_MODEL_BASE = process.env.NEXT_PUBLIC_SURYA_MODEL_BASE ||
    'https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/resolve/1d3698b88de708cc7b7cb8d349b4255715b70bc5/onnx';

export class SuryaWebGPU {
    constructor(descriptor = {}) {
        this.descriptor = { modelBase: SURYA_MODEL_BASE, localPath: '/models/surya-ocr',
            overrideBase: Boolean(process.env.NEXT_PUBLIC_SURYA_MODEL_BASE), label: 'Surya',
            prepareImage: prepareSuryaImage, cleanText: cleanSuryaText, ...descriptor };
    }
    async load(onProgress = () => {}, options = {}) {
        if (this.decoder) return;
        if (!navigator.gpu) throw new Error('WebGPU est indisponible dans ce navigateur.');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter?.features.has('shader-f16')) throw new Error('Ce GPU ne prend pas en charge WebGPU FP16.');
        this.gpuName = adapter.info?.description || adapter.info?.architecture || 'WebGPU';
        ort.env.wasm.wasmPaths = new URL('/onnx/', self.location.origin).href;
        ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
        ort.env.webgpu.powerPreference = 'high-performance';
        this.options = { tokenBatch: 8, ...options };
        let base = this.descriptor.modelBase;
        if (!this.descriptor.overrideBase && process.env.NODE_ENV === 'development') {
            const local = new URL(this.descriptor.localPath, self.location.origin).href;
            if ((await fetch(`${local}/browser_manifest.json`, { method: 'HEAD' }).catch(() => null))?.ok) base = local;
        }
        base = new URL(base, self.location.origin).href.replace(/\/$/, '');
        const getJson = async name => {
            const response = await fetch(`${base}/${name}`);
            if (!response.ok) throw new Error(`Fichier ${this.descriptor.label} manquant : ${name} (${response.status}).`);
            return response.json();
        };
        const [manifest, tokenizer, tokenizerConfig] = await Promise.all([
            getJson('browser_manifest.json'), getJson('tokenizer.json'), getJson('tokenizer_config.json')]);
        this.descriptor.validateManifest?.(manifest);
        this.manifest = manifest;
        const useFused = options.mode !== 'onnx' && !!manifest.files['decode_weights.json'];
        if (options.mode === 'fused' && !useFused) throw new Error('Index des poids fusionnés absent du modèle.');
        this.options.mode = useFused ? 'fused' : 'onnx';
        this.stateOnly = useFused && options.prefill !== 'classic';
        this.tokenizer = new PreTrainedTokenizer(tokenizer, tokenizerConfig);
        this.cacheSpec = [];
        manifest.config.text_config.layer_types.forEach((type, i) => {
            if (type === 'linear_attention') {
                this.cacheSpec.push([`conv_${i}`, 'float16', [1, 3, 6144]], [`state_${i}`, 'float32', [16, 128, 128]]);
            } else this.cacheSpec.push([`key_${i}`, 'float16', [1, 2, 0, 256]], [`value_${i}`, 'float16', [1, 2, 0, 256]]);
        });
        try {
            for (const name of ['vision', 'decoder']) {
                const graphName = name === 'decoder' && this.stateOnly ? 'prefill-fast.onnx' : `${name}.onnx`;
                const graph = await downloadModel(`${base}/${graphName}`, manifest.files[graphName]);
                const data = await downloadModel(`${base}/${name}.onnx.data`, {
                    ...manifest.files[`${name}.onnx.data`],
                    onProgress: progress => onProgress({ file: name === 'vision' ? `Encodeur d’image ${this.descriptor.label}` : `Décodeur ${this.descriptor.label}`, progress }),
                });
                onProgress({ file: `Initialisation ${name} sur WebGPU`, progress: 99 });
                const locations = name === 'vision' ? { image_features: 'gpu-buffer' } : {
                    ...Object.fromEntries(this.cacheSpec.map(([key]) => [`present_${key}`, 'gpu-buffer'])),
                    next_token: 'cpu', logits: 'cpu',
                };
                this[name] = await ort.InferenceSession.create(graph, {
                    executionProviders: [{ name: 'webgpu', validationMode: 'wgpuOnly' }], externalData: [{ path: `${name}.onnx.data`, data }],
                    preferredOutputLocation: locations, graphOptimizationLevel: 'all',
                    extra: { session: { disable_cpu_ep_fallback: '1' } },
                });
                if (name === 'decoder') {
                    if (useFused) {
                        if (!manifest.files['decode_weights.json']) throw new Error('Index des poids fusionnés absent du modèle.');
                        const indexBytes = await downloadModel(`${base}/decode_weights.json`, manifest.files['decode_weights.json']);
                        const index = JSON.parse(new TextDecoder().decode(indexBytes));
                        if (index.schema !== 1 || index.source_sha256 !== manifest.source_sha256 || index.data_sha256 !== manifest.files['decoder.onnx.data'].sha256) {
                            throw new Error('L’index GPU ne correspond pas aux poids ONNX.');
                        }
                        this.fused = new SuryaFusedDecoder(await ort.env.webgpu.device, this.options);
                        await this.fused.load(index, data);
                    } else {
                        const decodeGraph = await downloadModel(`${base}/decode.onnx`, manifest.files['decode.onnx']);
                        this.decode = await ort.InferenceSession.create(decodeGraph, {
                            executionProviders: ['webgpu'], externalData: [{ path: 'decoder.onnx.data', data }],
                            preferredOutputLocation: locations, graphOptimizationLevel: 'all',
                            extra: { session: { disable_cpu_ep_fallback: '1' } },
                        });
                    }
                }
            }
        } catch (error) { await this.dispose(); throw error; }
        onProgress({ file: `${this.descriptor.label} prêt`, progress: 100 });
    }

    async recognize(input, onToken = () => {}, options = {}) {
        if (!this.decoder) throw new Error(`${this.descriptor.label} n’est pas chargé.`);
        options.signal?.throwIfAborted();
        const started = performance.now();
        const bitmap = input instanceof ImageBitmap ? input : await createImageBitmap(input);
        let prepared;
        try { prepared = await this.descriptor.prepareImage(this.manifest, bitmap); }
        finally { bitmap.close(); }
        const p = prepared, n = p.patches;
        const feeds = {
            pixel_values: new ort.Tensor('float16', p.pixels, [n, 1536]),
            position_indices: new ort.Tensor('int64', p.indices, [4, n]),
            position_weights: new ort.Tensor('float32', p.weights, [4, n, 1]),
            vision_cos: new ort.Tensor('float32', p.cos, [1, n, 64]),
            vision_sin: new ort.Tensor('float32', p.sin, [1, n, 64]),
        };
        if (this.manifest.vision_attention_chunks > 1) {
            const chunks = this.manifest.vision_attention_chunks;
            const sizes = Array.from({ length: chunks }, (_, i) => Math.floor(n / chunks) + (i < n % chunks ? 1 : 0));
            const groups = [];
            for (let i = 0; i < 4; i++) {
                const part = sizes.slice(i * 4, i * 4 + 4);
                groups.push(part.reduce((a, b) => a + b, 0));
                feeds[`query_splits_${i}`] = new ort.Tensor('int64', BigInt64Array.from(part, BigInt), [4]);
            }
            feeds.query_group_splits = new ort.Tensor('int64', BigInt64Array.from(groups, BigInt), [4]);
        }
        let imageFeatures, cache = {}, next, prefillMs;
        const generated = [];
        try {
            try { imageFeatures = (await this.vision.run(feeds)).image_features; }
            finally { Object.values(feeds).forEach(t => t.dispose()); }
            options.signal?.throwIfAborted();
            const visionMs = performance.now() - started;
            for (const [key, dtype, shape] of this.cacheSpec) {
                const count = shape.reduce((a, b) => a * b, 1);
                cache[key] = new ort.Tensor(dtype, dtype === 'float32' ? new Float32Array(count) : new Uint16Array(count), shape);
            }
            const outputNames = [...this.cacheSpec.map(([key]) => `present_${key}`), ...(this.stateOnly ? [] : ['next_token'])];
            const run = async (ids, positions, start) => {
                const size = ids.length, total = start + size;
                const rotation = suryaRotary(positions);
                const attention = new Uint16Array(size * total);
                for (let q = 0; q < size; q++) for (let k = start + q + 1; k < total; k++) attention[q * total + k] = half(-10000);
                const identity = new Float32Array(size * size), lower = new Float32Array(size * size), strict = new Float32Array(size * size);
                for (let q = 0; q < size; q++) for (let k = 0; k <= q; k++) {
                    lower[q * size + k] = 1;
                    if (q === k) identity[q * size + k] = 1; else strict[q * size + k] = 1;
                }
                const inputs = {
                    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, size]),
                    image_indices: new ort.Tensor('int64', BigInt64Array.from(ids, (id, i) => BigInt(start < p.ids.length && id === this.manifest.image_token_id ? start + i - p.imageOffset : 0)), [size]),
                    image_mask: new ort.Tensor('bool', Uint8Array.from(ids, id => start < p.ids.length && id === this.manifest.image_token_id ? 1 : 0), [1, size, 1]),
                    rope_cos: new ort.Tensor('float32', rotation.cos, [1, 1, size, 64]),
                    rope_sin: new ort.Tensor('float32', rotation.sin, [1, 1, size, 64]),
                    attention_mask: new ort.Tensor('float16', attention, [1, 1, size, total]),
                    identity: new ort.Tensor('float32', identity, [1, size, size]),
                    lower: new ort.Tensor('float32', lower, [1, size, size]),
                    strict_lower: new ort.Tensor('float32', strict, [1, size, size]),
                };
                const session = size === 1 && this.decode ? this.decode : this.decoder;
                const sessionInputs = Object.fromEntries(Object.entries({ ...inputs, ...cache, image_features: imageFeatures })
                    .filter(([name]) => session.inputNames.includes(name)));
                let outputs;
                try { outputs = await session.run(sessionInputs, outputNames); }
                finally { Object.values(inputs).forEach(t => t.dispose()); }
                Object.values(cache).forEach(t => t.dispose());
                cache = Object.fromEntries(this.cacheSpec.map(([key]) => [key, outputs[`present_${key}`]]));
                if (this.stateOnly) return;
                const token = Number(outputs.next_token.data[0]); outputs.next_token.dispose();
                return token;
            };
            const chunk = this.manifest.chunk_size;
            const promptEnd = p.ids.length - (this.stateOnly ? 1 : 0);
            for (let start = 0; start < promptEnd; start += chunk) {
                options.signal?.throwIfAborted();
                const end = Math.min(start + chunk, promptEnd);
                next = await run(p.ids.slice(start, end), p.positions.slice(start, end), start);
            }
            let setupMs = 0;
            if (this.stateOnly) {
                const setupStart = performance.now();
                await this.fused.begin(cache, promptEnd, p.positions[promptEnd][0], p.ids[promptEnd], this.manifest.max_new_tokens + 1);
                setupMs = performance.now() - setupStart;
                [next] = await this.fused.step();
            } else if (this.fused && !this.manifest.stop_ids.includes(next)) {
                const setupStart = performance.now();
                await this.fused.begin(cache, p.ids.length, p.nextPosition, next, this.manifest.max_new_tokens);
                setupMs = performance.now() - setupStart;
            }
            prefillMs = performance.now() - started;
            const pendingTokens = [];
            let lastStream = 0;
            for (let step = 0; step < this.manifest.max_new_tokens; step++) {
                options.signal?.throwIfAborted();
                if (this.manifest.stop_ids.includes(next)) break;
                generated.push(next);
                if (performance.now() - lastStream >= 32 || step === 0) {
                    onToken(this.tokenizer.decode(generated, { skip_special_tokens: true, clean_up_tokenization_spaces: false }));
                    lastStream = performance.now();
                }
                if (step + 1 < this.manifest.max_new_tokens) {
                    const pos = p.nextPosition + step;
                    if (this.fused) {
                        if (!pendingTokens.length) {
                            const batch = this.options.profile || step < 4 ? 1 : Math.min(8, Math.max(1, Number(this.options.tokenBatch) || 8));
                            pendingTokens.push(...await this.fused.step(Math.min(batch, this.manifest.max_new_tokens - step - 1)));
                        }
                        next = pendingTokens.shift();
                    } else next = await run([next], [[pos, pos, pos]], p.ids.length + step);
                }
            }
            return { text: this.descriptor.cleanText(this.tokenizer.decode(generated, { skip_special_tokens: true, clean_up_tokenization_spaces: false })),
                provider: 'webgpu', gpu: this.gpuName, tokens: generated.length,
                tokenLimit: !this.manifest.stop_ids.includes(next),
                implementation: this.fused ? 'wgsl-fused' : 'onnx',
                diagnostics: this.fused?.stats(),
                profile: this.options.profile ? this.fused?.profile : undefined,
                timings: { totalMs: Math.round(performance.now() - started), visionMs: Math.round(visionMs), prefillMs: Math.round(prefillMs), setupMs: Math.round(setupMs) } };
        } finally { imageFeatures?.dispose(); Object.values(cache).forEach(t => t.dispose()); this.fused?.end(); }
    }

    async dispose() {
        await this.vision?.release(); await this.decoder?.release(); await this.decode?.release();
        this.fused?.dispose(); this.fused = null;
        this.vision = null; this.decoder = null; this.decode = null; this.tokenizer = null;
    }
}
