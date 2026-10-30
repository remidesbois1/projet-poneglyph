import { describe, expect, it } from 'vitest';
import { half, packImage, resizedDimensions, rotaryPositions, roundEven, smartDimensions } from './falconPreprocessing';

describe('Falcon browser preprocessing', () => {
    it('encodes normalized pixels as IEEE float16, including subnormals', () => {
        expect([0, -0, 1, -1, 65504, 2 ** -14, 2 ** -24].map(half))
            .toEqual([0, 0x8000, 0x3c00, 0xbc00, 0x7bff, 0x0400, 1]);
    });
    it('matches Python rounding and the two resizing budgets', () => {
        expect([2.5, 3.5, 4.5].map(roundEven)).toEqual([2, 4, 4]);
        expect(resizedDimensions(160, 96)).toEqual([160, 96]);
        expect(resizedDimensions(32, 64)).toEqual([64, 128]);
        expect(smartDimensions(72, 88)).toEqual([64, 96]);
    });
    it('preserves patch order, image bidirectionality and text causality', () => {
        const manifest = {
            prompt_chunks: [[], [100, 101]],
            config: { image_cls_token_id: 244, image_reg_1_token_id: 245, image_reg_2_token_id: 246,
                image_reg_3_token_id: 247, image_reg_4_token_id: 248, img_id: 227, img_end_id: 230 },
            golden_frequencies: Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => [1, 0])),
        };
        const rgba = new Uint8ClampedArray(32 * 32 * 4).fill(255);
        rgba[0] = 0;
        const batch = packImage(manifest, rgba, 32, 32);
        expect(batch.ids).toEqual([244,245,246,247,248,227,227,227,227,230,100,101]);
        expect(batch.pixels[5 * 768]).toBe(half(-1));
        expect(batch.pixels[5 * 768 + 1]).toBe(half(1));
        expect(batch.attention[8]).toBe(0);
        expect(batch.attention[9]).toBe(half(-10000));
        expect(batch.attention[10 * 12 + 11]).toBe(half(-10000));
        expect(batch.lastPosition).toBe(2);
        expect(batch.cos[(5 * 16) * 32 + 16]).toBeCloseTo(Math.cos(-1), 6);
    });
    it('preserves every possible normalized pixel value', () => {
        const manifest = { prompt_chunks: [[], []], config: { image_cls_token_id: 244, image_reg_1_token_id: 245,
            image_reg_2_token_id: 246, image_reg_3_token_id: 247, image_reg_4_token_id: 248, img_id: 227, img_end_id: 230 },
        golden_frequencies: Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => [1, 0])) };
        const rgba = new Uint8ClampedArray(16 * 16 * 4);
        for (let i = 0; i < 256; i++) for (let c = 0; c < 3; c++) rgba[i * 4 + c] = (i + c) % 256;
        const pixels = packImage(manifest, rgba, 16, 16).pixels;
        for (let i = 0; i < 256; i++) for (let c = 0; c < 3; c++) {
            expect(pixels[5 * 768 + i * 3 + c]).toBe(half(((i + c) % 256) / 127.5 - 1));
        }
    });
    it('matches the original temporal and spatial rotary calculations exactly', () => {
        const manifest = { golden_frequencies: Array.from({ length: 16 }, (_, h) =>
            Array.from({ length: 16 }, (_, f) => [Math.sin(h + f), Math.cos(h - f)])) };
        const times = [0, 7, 8191], spatial = [null, [-0.75, 0.625], [1.5, -0.5]];
        const result = rotaryPositions(manifest, times, spatial);
        const cos = new Float32Array(result.cos.length), sin = new Float32Array(result.sin.length);
        for (let s = 0; s < times.length; s++) for (let h = 0; h < 16; h++) for (let f = 0; f < 32; f++) {
            const frequency = manifest.golden_frequencies[h][f - 16];
            const angle = Math.fround(f < 16 ? times[s] / (10000 ** (2 * f / 32))
                : spatial[s] ? spatial[s][0] * frequency[0] + spatial[s][1] * frequency[1] : 0);
            const i = (s * 16 + h) * 32 + f;
            cos[i] = Math.cos(angle); sin[i] = Math.sin(angle);
        }
        expect(result.cos).toEqual(cos); expect(result.sin).toEqual(sin);
    });
});
