import { describe, expect, it } from 'vitest';
import { resizeUint8Bicubic } from './bicubicResize';
import golden from './bicubicResize.fixtures.json';

describe('native Torchvision uint8 resize', () => {
    it('matches independent native pixels for upscaling, downscaling and one-axis resizes', () => {
        for (const item of golden.cases) {
            expect(Array.from(resizeUint8Bicubic(Uint8ClampedArray.from(item.input), item.width, item.height,
                item.outputWidth, item.outputHeight))).toEqual(item.expected);
        }
    });
    it('preserves RGB bytes without resampling at identical dimensions', () => {
        const source = Uint8ClampedArray.from(golden.cases[0].input);
        expect(resizeUint8Bicubic(source, 3, 2, 3, 2)).toBe(source);
    });
    it('rejects invalid dimensions before allocating output buffers', () => {
        expect(() => resizeUint8Bicubic(new Uint8ClampedArray(4), 1, 1, 0, 32)).toThrow(/invalides/);
        expect(() => resizeUint8Bicubic(new Uint8ClampedArray(4), 2, 1, 32, 32)).toThrow(/invalides/);
    });
});
