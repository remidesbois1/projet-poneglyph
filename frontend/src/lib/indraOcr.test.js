import { describe, expect, it } from 'vitest';
import {
    cropLine,
    decodeCtcIds,
    decodeLines,
    isDarkLine,
    pageToInput,
    resizeArea,
    resizeLinear,
    roundHalfEven,
    sortLineBoxes,
    stitchLines,
} from './indraOcr';
import golden from './indraOcr.fixtures.json';

const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

describe('Indra OCR preprocessing matches the Python reference (OpenCV + indra_ocr)', () => {
    it('resizes like cv2 INTER_AREA and INTER_LINEAR within one gray level', () => {
        for (const c of golden.resize) {
            const resize = c.mode === 'area' ? resizeArea : resizeLinear;
            const out = resize(Uint8Array.from(c.src), c.sw, c.sh, c.dw, c.dh);
            expect(out.length).toBe(c.out.length);
            expect(maxDiff(Array.from(out), c.out)).toBeLessThanOrEqual(1);
        }
    });

    it('builds the detector input at the bubble scale, padded to multiples of 32', () => {
        const { page } = golden;
        const input = pageToInput(Uint8Array.from(page.gray), page.width, page.height, page.long_side);
        expect([input.width, input.height]).toEqual([page.input.width, page.input.height]);
        expect(input.scale).toBeCloseTo(page.input.scale, 12);
        expect(maxDiff(Array.from(input.data, v => v * 255), page.input.levels)).toBeLessThanOrEqual(1.001);
    });

    it('crops tall line windows, inverting light text on a dark ground', () => {
        const { page } = golden;
        const gray = Uint8Array.from(page.gray);
        for (const crop of page.crops) {
            expect(isDarkLine(gray, page.width, page.height, crop.box)).toBe(crop.dark);
            const ink = cropLine(gray, page.width, page.height, crop.box);
            expect([ink.width, ink.height]).toEqual([crop.width, crop.height]);
            expect(maxDiff(Array.from(ink.data, v => v * 255), crop.levels)).toBeLessThanOrEqual(1.001);
        }
    });

    it('decodes detector maps into the same boxes, dropping one-pixel cores', () => {
        const { decode } = golden;
        const boxes = decodeLines(Float32Array.from(decode.out), 3, decode.h, decode.w);
        expect(boxes).toHaveLength(decode.boxes.length);
        boxes.forEach((box, i) => box.forEach((v, k) => expect(v).toBeCloseTo(decode.boxes[i][k], 3)));
    });

    it('rounds halves to even like Python round()', () => {
        expect([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(roundHalfEven)).toEqual([0, 2, 2, 4, 2, 3]);
    });
});

describe('Indra OCR assembly', () => {
    it('orders lines by half-pitch bands, then left to right', () => {
        const boxes = [[50, 40, 90, 60], [10, 0, 40, 20], [5, 42, 45, 61], [60, 2, 80, 21]];
        expect(sortLineBoxes(boxes)).toEqual([[10, 0, 40, 20], [60, 2, 80, 21], [5, 42, 45, 61], [50, 40, 90, 60]]);
    });

    it('puts line windows side by side and pads like pad_batch', () => {
        const a = { width: 3, height: 2, data: Float32Array.from([1, 2, 3, 4, 5, 6]) };
        const b = { width: 2, height: 2, data: Float32Array.from([7, 8, 9, 10]) };
        const out = stitchLines([a, b]);
        expect([out.width, out.validWidth]).toEqual([32, 5]);
        expect(Array.from(out.data.subarray(0, 6))).toEqual([1, 2, 3, 7, 8, 0]);
        expect(Array.from(out.data.subarray(32, 38))).toEqual([4, 5, 6, 9, 10, 0]);
    });

    it('collapses CTC repeats and blanks within the valid length only', () => {
        const ids = BigInt64Array.from([1n, 1n, 0n, 1n, 2n, 2n, 0n, 3n, 3n]);
        expect(decodeCtcIds(ids, 7, ['a', 'b', 'c'])).toBe('aab');
    });
});
