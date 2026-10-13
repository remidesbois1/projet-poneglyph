import { SuryaWebGPU } from './suryaWebgpu';
import { prepareLighton3BubbleImage } from './lighton3BubblePreprocessing';
import { LIGHTON3_BUBBLE_MODEL_BASE } from './lighton3BubbleModel';

export { LIGHTON3_BUBBLE_MODEL_BASE } from './lighton3BubbleModel';

export function validateLighton3BubbleManifest(manifest) {
    if (manifest.task !== 'bubble-ocr' || manifest.model_id !== 'Remidesbois/LightOnOCR-3-Poneglyph' ||
        manifest.prompt !== '' || manifest.thinking !== false || manifest.max_new_tokens !== 256 ||
        manifest.min_pixels !== 65536 || manifest.max_pixels !== 1048576 || manifest.bbox ||
        manifest.postprocess?.strip_outer_quote_pair !== false) {
        throw new Error('Les fichiers ne correspondent pas au modèle LightOnOCR 3 pour les bulles.');
    }
}

export function cleanLighton3BubbleText(text) {
    return text.trim().replace(/\s+/g, ' ');
}

export class Lighton3BubbleWebGPU extends SuryaWebGPU {
    constructor() {
        super({ modelBase: LIGHTON3_BUBBLE_MODEL_BASE, localPath: '/models/lighton3-bubble', label: 'LightOnOCR 3 · bulles',
            overrideBase: Boolean(process.env.NEXT_PUBLIC_LIGHTON3_BUBBLE_MODEL_BASE),
            prepareImage: prepareLighton3BubbleImage, cleanText: cleanLighton3BubbleText,
            validateManifest: validateLighton3BubbleManifest });
    }
}
