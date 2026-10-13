import { SuryaWebGPU } from './suryaWebgpu';
import { prepareLighton3Image } from './lighton3Preprocessing';
import { parseLighton3Grounding } from './lighton3Grounding';

export const LIGHTON3_MODEL_BASE = process.env.NEXT_PUBLIC_LIGHTON3_MODEL_BASE ||
    'https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph-bbox/resolve/b3ae1dac8ae48a0f43a7b78dcc7aa8d48b97a63d/onnx';

export class Lighton3WebGPU extends SuryaWebGPU {
    constructor() {
        super({ modelBase: LIGHTON3_MODEL_BASE, localPath: '/models/lighton3-ocr', label: 'LightOnOCR 3',
            overrideBase: Boolean(process.env.NEXT_PUBLIC_LIGHTON3_MODEL_BASE),
            prepareImage: prepareLighton3Image, cleanText: text => text.trim() });
    }
    async recognize(input, onToken, options) {
        const result = await super.recognize(input, onToken, options);
        return { ...result, ...parseLighton3Grounding(result.text) };
    }
}
