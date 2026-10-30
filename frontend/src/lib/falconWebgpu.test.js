import { afterEach, describe, expect, it, vi } from 'vitest';
import index from './falconDecodeWeights.json';

vi.mock('onnxruntime-web/webgpu', () => {
    class Tensor {
        constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
        dispose() {}
    }
    return { Tensor };
});
vi.mock('./falconPreprocessing', () => ({
    prepareImage: vi.fn(async () => ({ ids: [1], pixels: new Uint16Array(768), mask: new Uint8Array(1),
        cos: new Float32Array(512), sin: new Float32Array(512), attention: new Uint16Array(1), lastPosition: 0 })),
    rotaryPositions: vi.fn(() => ({ cos: new Float32Array(512), sin: new Float32Array(512) })),
}));
import { FalconWebGPU, canUseFalconFused } from './falconWebgpu';

const manifest = () => ({ source_sha256: index.source_sha256, files: {
    'decoder.onnx': { sha256: index.graph_sha256 }, 'decoder.onnx.data': { sha256: index.data_sha256 },
} });
const device = () => ({ features: new Set(['subgroups']), limits: { maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 128 * 1024 * 1024 } });

describe('Falcon accelerated decoder safety', () => {
    afterEach(() => vi.unstubAllGlobals());

    it('enables the decoder only for the verified model and supported GPU limits', () => {
        expect(canUseFalconFused(manifest(), device())).toBe(true);
        expect(canUseFalconFused({ ...manifest(), source_sha256: 'different' }, device())).toBe(false);
        const changed = manifest(); changed.files['decoder.onnx.data'].sha256 = 'different';
        expect(canUseFalconFused(changed, device())).toBe(false);
        const noSubgroups = device(); noSubgroups.features.clear();
        expect(canUseFalconFused(manifest(), noSubgroups)).toBe(false);
        const smallBuffer = device(); smallBuffer.limits.maxStorageBufferBindingSize = 1024;
        expect(canUseFalconFused(manifest(), smallBuffer)).toBe(false);
    });

    it('indexes complete weight ranges within the published ONNX data', () => {
        expect(Object.keys(index.weights)).toHaveLength(113);
        for (const weight of Object.values(index.weights)) {
            const elements = weight.shape.reduce((a, b) => a * b, 1);
            if (weight.bits) expect(weight.bits).toHaveLength(elements);
            else {
                expect(weight.bytes).toBe(elements * (weight.dtype === 'float32' ? 4 : 2));
                expect(weight.offset).toBeGreaterThanOrEqual(0);
                expect(weight.offset + weight.bytes).toBeLessThanOrEqual(617742336);
            }
        }
    });

    it('restarts the same image with ONNX if the accelerated path fails', async () => {
        const Bitmap = class { close() {} };
        vi.stubGlobal('ImageBitmap', Bitmap);
        const engine = new FalconWebGPU();
        const dispose = vi.fn(), end = vi.fn();
        engine.fused = { begin: vi.fn().mockRejectedValue(new Error('kernel failed')), end, dispose };
        engine.options = { mode: 'auto', tokenBatch: 8 };
        engine.manifest = { config: { n_layers: 22, max_seq_len: 1024 }, max_new_tokens: 4, stop_ids: [99] };
        engine.tokenizer = { decode: values => values.join(',') };
        const tokens = [42, 42, 99];
        engine.session = { run: vi.fn(async () => ({ next_token: { data: [tokens.shift()], dispose() {} },
            present_key_values: { dispose() {} } })) };
        const stream = vi.fn();
        const result = await engine.recognize(new Bitmap(), stream);
        expect(result.text).toBe('42');
        expect(result.implementation).toBe('onnx');
        expect(result.accelerationFallback).toBe('kernel failed');
        expect(engine.session.run).toHaveBeenCalledTimes(3);
        expect(end).toHaveBeenCalled(); expect(dispose).toHaveBeenCalled();
        expect(stream).toHaveBeenCalledWith('');
        expect(engine.fused).toBeNull();
    });
});
