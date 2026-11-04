import { DEEPSEEK_LABEL } from './deepseekConfig';
import { GLM_LABEL } from './mistralConfig';

export const OCR_RUNTIME_LABELS = Object.freeze({
    browser: 'Navigateur',
    local: 'Local',
    cloud: 'Cloud',
});

const defineModel = (key, fields) => Object.freeze({ key, sandbox: true, ...fields });

/**
 * Every OCR model offered by the annotation editor, in display order.
 * - `scope: 'bubble'` reads one bubble crop (page OCR and hand-drawn boxes).
 * - `scope: 'page'` reads the whole page and returns text with its own boxes; the page
 *   pipeline rallies them onto the YOLO detector, whose geometry and reading order win.
 * Names follow "<family> <version>[ BBox]"; where a model runs is `runtime`, never the name.
 */
export const OCR_MODELS = Object.freeze({
    ppocrv6Line: defineModel('ppocrv6Line', { name: 'PP-OCRv6', scope: 'bubble', runtime: 'browser', engine: 'onnx', size: '87 Mo' }),
    lighton3BubbleWebgpu: defineModel('lighton3BubbleWebgpu', { name: 'LightOnOCR 3', scope: 'bubble', runtime: 'browser', engine: 'onnx', size: '1,7 Go', preserveTranscript: true }),
    suryaWebgpu: defineModel('suryaWebgpu', { name: 'Surya OCR 2', scope: 'bubble', runtime: 'browser', engine: 'onnx', size: '1,3 Go' }),
    falconWebgpu: defineModel('falconWebgpu', { name: 'Falcon OCR', scope: 'bubble', runtime: 'browser', engine: 'onnx', size: '620 Mo' }),
    poneglyphLocal: defineModel('poneglyphLocal', { name: 'LightOnOCR 2', scope: 'bubble', runtime: 'local', engine: 'tauri', localModel: 'text', size: '2 Go' }),
    suryaLocal: defineModel('suryaLocal', { name: 'Surya OCR 2', scope: 'bubble', runtime: 'local', engine: 'tauri', localModel: 'surya', size: '1,3 Go' }),
    lighton3Webgpu: defineModel('lighton3Webgpu', { name: 'LightOnOCR 3 BBox', scope: 'page', runtime: 'browser', engine: 'onnx', size: '1,7 Go', preserveTranscript: true }),
    lighton2BboxLocal: defineModel('lighton2BboxLocal', { name: 'LightOnOCR 2 BBox', scope: 'page', runtime: 'local', engine: 'tauri', localModel: 'bbox', size: '2 Go' }),
    suryaBboxLocal: defineModel('suryaBboxLocal', { name: 'Surya OCR 2 BBox', scope: 'page', runtime: 'local', engine: 'tauri', localModel: 'suryaBbox', size: '1,3 Go' }),
    geminiPage: defineModel('geminiPage', { name: 'Gemini', scope: 'page', runtime: 'cloud', engine: 'gemini', configKey: 'model_ocr', sandbox: false }),
    chatgptPage: defineModel('chatgptPage', { name: 'ChatGPT', scope: 'page', runtime: 'cloud', engine: 'chatgpt', configKey: 'model_chatgpt_ocr' }),
    deepseekPage: defineModel('deepseekPage', { name: DEEPSEEK_LABEL, scope: 'page', runtime: 'cloud', engine: 'deepseek' }),
    glmPage: defineModel('glmPage', { name: GLM_LABEL, scope: 'page', runtime: 'cloud', engine: 'mistral' }),
});

export const DEFAULT_OCR_MODEL_KEYS = Object.freeze(['ppocrv6Line']);

/** The sandbox never offered Gemini page reading. */
export function isSelectableOcrModel(model, isSandbox = false) {
    return Boolean(model && (!isSandbox || model.sandbox));
}

/** `gemini-3.1-flash-lite` → `Gemini 3.1 Flash Lite`, `gpt-5.6-luna` → `GPT-5.6 Luna`. */
export function formatProviderModelName(modelId) {
    const [family, ...rest] = String(modelId || '').trim().split('-').filter(Boolean);
    if (!family) return '';
    const capitalize = token => token.charAt(0).toUpperCase() + token.slice(1);
    if (family.toLowerCase() === 'gpt') {
        const [version, ...variant] = rest;
        return [version ? `GPT-${version}` : 'GPT', ...variant.map(capitalize)].join(' ');
    }
    return [family, ...rest].map(capitalize).join(' ');
}

export function getOcrModelName(model, aiModelConfig = null) {
    const configuredModel = model.configKey ? aiModelConfig?.[model.configKey] : null;
    return configuredModel ? formatProviderModelName(configuredModel) : model.name;
}

export function getOcrModelLabel(model, aiModelConfig = null) {
    return `${getOcrModelName(model, aiModelConfig)} · ${OCR_RUNTIME_LABELS[model.runtime]}`;
}
