import { describe, expect, it } from 'vitest';
import { lighton3Dimensions } from './lighton3Preprocessing';
import { suryaLayout } from './suryaPreprocessing';

const manifest = { min_pixels: 65536, max_pixels: 4194304, max_long_edge: 2048,
    prompt_chunks: [[248045, 846, 198, 248053], [248054, 1884, 286, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]],
    image_token_id: 248056 };

describe('Full-page native Qwen3.5 inputs', () => {
    it('preserves the training image budget and native 32-pixel grid', () => {
        expect(lighton3Dimensions(manifest, 958, 1500).size).toEqual([960, 1504]);
        expect(lighton3Dimensions(manifest, 4000, 6000)).toEqual({ thumbnailWidth: 1365, thumbnailHeight: 2048, size: [1376, 2048] });
        expect(lighton3Dimensions(manifest, 2048, 2048).size).toEqual([2048, 2048]);
    });
    it('uses the grounding prompt with disabled thinking and all visual tokens', () => {
        const layout = suryaLayout(manifest, 64, 64);
        expect(layout.ids).toEqual([248045, 846, 198, 248053, ...Array(4).fill(248056), ...manifest.prompt_chunks[1]]);
        expect(layout.positions.slice(4, 8)).toEqual([[4,4,4], [4,4,5], [4,5,4], [4,5,5]]);
        expect(layout.nextPosition).toBe(18);
    });
});
