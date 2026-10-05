import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { cleanSuryaText, packSuryaPixels, suryaDimensions, suryaLayout, suryaRotary } from './suryaPreprocessing';
import { half } from './falconPreprocessing';

describe('Surya image and rotary preprocessing', () => {
    it('keeps the trained pixel budget and 32-pixel grid', () => {
        expect(suryaDimensions(51, 73)).toEqual([224, 320]);
        expect(suryaDimensions(133, 93)).toEqual([320, 224]);
        expect(suryaDimensions(1000, 2000)).toEqual([704, 1440]);
        expect(() => suryaDimensions(1, 1000)).toThrow();
    });
    it('packs spatial merge blocks, channels, and duplicate temporal frames in HF order', () => {
        const layout = suryaLayout({ prompt_chunks: [[1, 9], [10, 8]], image_token_id: 11 }, 64, 32);
        expect(layout.patchPositions).toEqual([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]);
        const rgba = new Uint8Array(64 * 32 * 4);
        for (let i = 0; i < 64 * 32; i++) { rgba[i * 4] = 255; rgba[i * 4 + 1] = 128; rgba[i * 4 + 3] = 255; }
        const pixels = packSuryaPixels(rgba, 64, layout);
        expect(Array.from(pixels.slice(0, 512)).every(v => v === half(1))).toBe(true);
        expect(pixels[512]).toBe(half(128 / 127.5 - 1));
        expect(pixels[1024]).toBe(half(-1));
        expect(layout.positions).toEqual([[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [4, 4, 4], [5, 5, 5]]);
        expect(layout.nextPosition).toBe(6);
    });
    it('matches training output normalization', () => {
        expect(cleanSuryaText('  "Vas-y\nBuchi !"  ')).toBe('Vas-y Buchi !');
        expect(cleanSuryaText("L’équipage... ")).toBe('L’équipage...');
    });
});

const root = resolve('public/models/surya-ocr');
describe.skipIf(!existsSync(`${root}/fixtures/examples.json`))('Local independent Transformers fixtures', () => {
    it('matches all token IDs, image interpolation indices, and multimodal rotary positions', () => {
        const manifest = JSON.parse(readFileSync(`${root}/browser_manifest.json`, 'utf8'));
        const examples = JSON.parse(readFileSync(`${root}/fixtures/examples.json`, 'utf8'));
        for (let i = 0; i < examples.length; i++) {
            const fixturePath = examples[i].image.replace(/bubble-(\d+)\.png$/, 'inputs-$1.json');
            const fixture = JSON.parse(readFileSync(`${root}/${fixturePath}`, 'utf8'));
            const [width, height] = suryaDimensions(examples[i].width, examples[i].height);
            expect([height / 16, width / 16]).toEqual(examples[i].grid.slice(1));
            const layout = suryaLayout(manifest, width, height);
            expect(layout.ids).toEqual(fixture.input_ids[0]);
            expect(layout.positions).toEqual(layout.ids.map((_, j) => fixture.positions.map(axis => axis[0][j])));
            expect(Array.from(layout.indices, Number)).toEqual(fixture.position_indices.flat());
            const expectedWeights = fixture.position_weights.flat(2);
            expect(layout.weights.reduce((max, v, j) => Math.max(max, Math.abs(v - expectedWeights[j])), 0)).toBeLessThan(0.00001);
            for (const name of ['cos', 'sin']) {
                const expected = fixture[`vision_${name}`].flat(2);
                expect(layout[name].reduce((max, v, j) => Math.max(max, Math.abs(v - expected[j])), 0)).toBeLessThan(0.00001);
            }
            expect(Array.from(suryaRotary(layout.positions).cos).every(Number.isFinite)).toBe(true);
        }
    });
});
