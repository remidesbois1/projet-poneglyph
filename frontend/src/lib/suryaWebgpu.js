import * as ort from 'onnxruntime-web/webgpu';
import { PreTrainedTokenizer } from '@huggingface/transformers';
import { half } from './falconPreprocessing';
import { prepareSuryaImage, suryaRotary, cleanSuryaText } from './suryaPreprocessing';
import { downloadModel } from './modelDownload';

export const SURYA_MODEL_BASE = process.env.NEXT_PUBLIC_SURYA_MODEL_BASE ||
    'https://huggingface.co/Remidesbois/surya-bubble-ocr-poneglyph/resolve/16efe756422b73054b97ea1fe082a0724627eadb/onnx';

export class SuryaWebGPU {
    async load(onProgress = () => {}) {
        if (this.decoder) return;
        if (!navigator.gpu) throw new Error('WebGPU est indisponible dans ce navigateur.');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter?.features.has('shader-f16')) throw new Error('Ce GPU ne prend pas en charge WebGPU FP16.');
        this.gpuName = adapter.info?.description || adapter.info?.architecture || 'WebGPU';
        ort.env.wasm.wasmPaths = new URL('/onnx/', self.location.origin).href;
        ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
        ort.env.webgpu.powerPreference = 'high-performance';
        let base = SURYA_MODEL_BASE;
        if (!process.env.NEXT_PUBLIC_SURYA_MODEL_BASE && process.env.NODE_ENV === 'development') {
            const local = new URL('/models/surya-ocr', self.location.origin).href;
            if ((await fetch(`${local}/browser_manifest.json`, { method: 'HEAD' }).catch(() => null))?.ok) base = local;
        }
        base = new URL(base, self.location.origin).href.replace(/\/$/, '');
        const getJson = async name => {
            const response = await fetch(`${base}/${name}`);
            if (!response.ok) throw new Error(`Fichier Surya manquant : ${name} (${response.status}).`);
            return response.json();
        };
        const [manifest, tokenizer, tokenizerConfig] = await Promise.all([
            getJson('browser_manifest.json'), getJson('tokenizer.json'), getJson('tokenizer_config.json')]);
        this.manifest = manifest;
        this.tokenizer = new PreTrainedTokenizer(tokenizer, tokenizerConfig);
        this.cacheSpec = [];
        manifest.config.text_config.layer_types.forEach((type, i) => {
            if (type === 'linear_attention') {
                this.cacheSpec.push([`conv_${i}`, 'float16', [1, 3, 6144]], [`state_${i}`, 'float32', [16, 128, 128]]);
            } else this.cacheSpec.push([`key_${i}`, 'float16', [1, 2, 0, 256]], [`value_${i}`, 'float16', [1, 2, 0, 256]]);
        });
        try {
            for (const name of ['vision', 'decoder']) {
                const graph = await downloadModel(`${base}/${name}.onnx`, manifest.files[`${name}.onnx`]);
                const data = await downloadModel(`${base}/${name}.onnx.data`, {
                    ...manifest.files[`${name}.onnx.data`],
                    onProgress: progress => onProgress({ file: name === 'vision' ? 'Encodeur d’image Surya' : 'Décodeur Surya', progress }),
                });
                onProgress({ file: `Initialisation ${name} sur WebGPU`, progress: 99 });
                const locations = name === 'vision' ? { image_features: 'gpu-buffer' } : {
                    ...Object.fromEntries(this.cacheSpec.map(([key]) => [`present_${key}`, 'gpu-buffer'])),
                    next_token: 'cpu', logits: 'cpu',
                };
                this[name] = await ort.InferenceSession.create(graph, {
                    executionProviders: ['webgpu'], externalData: [{ path: `${name}.onnx.data`, data }],
                    preferredOutputLocation: locations, graphOptimizationLevel: 'all',
                    extra: { session: { disable_cpu_ep_fallback: '1' } },
                });
                if (name === 'decoder') {
                    const decodeGraph = await downloadModel(`${base}/decode.onnx`, manifest.files['decode.onnx']);
                    this.decode = await ort.InferenceSession.create(decodeGraph, {
                        executionProviders: ['webgpu'], externalData: [{ path: 'decoder.onnx.data', data }],
                        preferredOutputLocation: locations, graphOptimizationLevel: 'all',
                        extra: { session: { disable_cpu_ep_fallback: '1' } },
                    });
                }
            }
        } catch (error) { await this.dispose(); throw error; }
        onProgress({ file: 'Surya prêt', progress: 100 });
    }

    async recognize(input, onToken = () => {}) {
        if (!this.decoder) throw new Error('Surya n’est pas chargé.');
        const started = performance.now();
        const bitmap = input instanceof ImageBitmap ? input : await createImageBitmap(input);
        let prepared;
        try { prepared = await prepareSuryaImage(this.manifest, bitmap); }
        finally { bitmap.close(); }
        const p = prepared, n = p.patches;
        const feeds = {
            pixel_values: new ort.Tensor('float16', p.pixels, [n, 1536]),
            position_indices: new ort.Tensor('int64', p.indices, [4, n]),
            position_weights: new ort.Tensor('float32', p.weights, [4, n, 1]),
            vision_cos: new ort.Tensor('float32', p.cos, [1, n, 64]),
            vision_sin: new ort.Tensor('float32', p.sin, [1, n, 64]),
        };
        let imageFeatures, cache = {}, next, prefillMs;
        const generated = [];
        try {
            try { imageFeatures = (await this.vision.run(feeds)).image_features; }
            finally { Object.values(feeds).forEach(t => t.dispose()); }
            const visionMs = performance.now() - started;
            for (const [key, dtype, shape] of this.cacheSpec) {
                const count = shape.reduce((a, b) => a * b, 1);
                cache[key] = new ort.Tensor(dtype, dtype === 'float32' ? new Float32Array(count) : new Uint16Array(count), shape);
            }
            const outputNames = [...this.cacheSpec.map(([key]) => `present_${key}`), 'next_token'];
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
                    image_indices: new ort.Tensor('int64', BigInt64Array.from(ids, (id, i) => BigInt(start < p.ids.length && id === 11 ? start + i - p.imageOffset : 0)), [size]),
                    image_mask: new ort.Tensor('bool', Uint8Array.from(ids, id => start < p.ids.length && id === 11 ? 1 : 0), [1, size, 1]),
                    rope_cos: new ort.Tensor('float32', rotation.cos, [1, 1, size, 64]),
                    rope_sin: new ort.Tensor('float32', rotation.sin, [1, 1, size, 64]),
                    attention_mask: new ort.Tensor('float16', attention, [1, 1, size, total]),
                    identity: new ort.Tensor('float32', identity, [1, size, size]),
                    lower: new ort.Tensor('float32', lower, [1, size, size]),
                    strict_lower: new ort.Tensor('float32', strict, [1, size, size]),
                };
                const session = size === 1 ? this.decode : this.decoder;
                const sessionInputs = Object.fromEntries(Object.entries({ ...inputs, ...cache, image_features: imageFeatures })
                    .filter(([name]) => session.inputNames.includes(name)));
                let outputs;
                try { outputs = await session.run(sessionInputs, outputNames); }
                finally { Object.values(inputs).forEach(t => t.dispose()); }
                Object.values(cache).forEach(t => t.dispose());
                cache = Object.fromEntries(this.cacheSpec.map(([key]) => [key, outputs[`present_${key}`]]));
                const token = Number(outputs.next_token.data[0]); outputs.next_token.dispose();
                return token;
            };
            const chunk = this.manifest.chunk_size;
            for (let start = 0; start < p.ids.length; start += chunk) {
                next = await run(p.ids.slice(start, start + chunk), p.positions.slice(start, start + chunk), start);
            }
            prefillMs = performance.now() - started;
            for (let step = 0; step < this.manifest.max_new_tokens; step++) {
                if (this.manifest.stop_ids.includes(next)) break;
                generated.push(next);
                onToken(this.tokenizer.decode(generated, { skip_special_tokens: true, clean_up_tokenization_spaces: false }));
                if (step + 1 < this.manifest.max_new_tokens) {
                    const pos = p.nextPosition + step;
                    next = await run([next], [[pos, pos, pos]], p.ids.length + step);
                }
            }
            return { text: cleanSuryaText(this.tokenizer.decode(generated, { skip_special_tokens: true, clean_up_tokenization_spaces: false })),
                provider: 'webgpu', gpu: this.gpuName, tokens: generated.length,
                tokenLimit: !this.manifest.stop_ids.includes(next),
                timings: { totalMs: Math.round(performance.now() - started), visionMs: Math.round(visionMs), prefillMs: Math.round(prefillMs) } };
        } finally { imageFeatures?.dispose(); Object.values(cache).forEach(t => t.dispose()); }
    }

    async dispose() {
        await this.vision?.release(); await this.decoder?.release(); await this.decode?.release();
        this.vision = null; this.decoder = null; this.decode = null; this.tokenizer = null;
    }
}
