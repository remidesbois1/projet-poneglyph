import { describe, expect, it, vi } from 'vitest';
vi.mock('./suryaWebgpu', () => ({ SuryaWebGPU: class { constructor(descriptor) { this.descriptor = descriptor; } } }));
import { Lighton3BubbleWebGPU, cleanLighton3BubbleText, validateLighton3BubbleManifest } from './lighton3BubbleWebgpu';
import { suryaDimensions } from './suryaPreprocessing';
import { prepareLighton3BubbleImage } from './lighton3BubblePreprocessing';

const manifest = { task: 'bubble-ocr', model_id: 'Remidesbois/LightOnOCR-3-Poneglyph', prompt: '', thinking: false,
    max_new_tokens: 256, min_pixels: 65536, max_pixels: 1048576, postprocess: { strip_outer_quote_pair: false } };

describe('LightOnOCR 3 bubble contract', () => {
    it('preserves visible quotes, case, accents and punctuation', () => {
        expect(cleanLighton3BubbleText('  "OÙ\n  EST-IL ?"  ')).toBe('"OÙ EST-IL ?"');
        expect(cleanLighton3BubbleText("'Boum !'" )).toBe("'Boum !'");
        expect(cleanLighton3BubbleText('... oui !')).toBe('... oui !');
    });
    it('rejects a full-page model or an incompatible generation contract before GPU loading', () => {
        expect(() => validateLighton3BubbleManifest(manifest)).not.toThrow();
        for (const change of [{ task: 'page-ocr-bbox' }, { prompt: 'grounding' }, { thinking: true },
            { max_new_tokens: 1280 }, { max_pixels: 4194304 }, { postprocess: { strip_outer_quote_pair: true } }]) {
            expect(() => validateLighton3BubbleManifest({ ...manifest, ...change })).toThrow(/bulles/);
        }
    });
    it('uses one resize with the native crop pixel budget, without the page thumbnail stage', () => {
        const engine = new Lighton3BubbleWebGPU();
        expect(engine.descriptor.prepareImage).toBe(prepareLighton3BubbleImage);
        expect(engine.descriptor.localPath).toBe('/models/lighton3-bubble');
        expect(engine.descriptor.modelBase).toMatch(/LightOnOCR-3-Poneglyph\/resolve\/[0-9a-f]{40}\/onnx$/);
        const [w, h] = suryaDimensions(80, 60, manifest.min_pixels, manifest.max_pixels);
        expect([w, h]).toEqual([320, 224]);
        expect(w * h).toBeGreaterThanOrEqual(65536);
        const large = suryaDimensions(1800, 3000, manifest.min_pixels, manifest.max_pixels);
        expect(large[0] * large[1]).toBeLessThanOrEqual(1048576);
    });
});
