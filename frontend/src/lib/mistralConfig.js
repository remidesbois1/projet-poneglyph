// Z.ai GLM 5.3 Flash, hosted by Mistral, checked 2026-11-04:
// https://docs.mistral.ai/models/zai-glm-5-3-flash (multimodal: text + image input)
// The model is served by Mistral without modification; only the Flash variant accepts images.
export const MISTRAL_MODEL = 'zai-glm-5-3-flash';
export const GLM_LABEL = 'GLM 5.3 Flash';
export const MISTRAL_API_KEY_STORAGE_KEY = 'mistral_api_key';
export const MISTRAL_IMAGE_MAX_BYTES = 10 * 1024 * 1024;
export const MISTRAL_TIMEOUT_MS = 120_000;

export function getMistralApiKey() {
    try {
        // Node's server-side localStorage getter emits an ExperimentalWarning; only browsers hold the key.
        if (typeof window === 'undefined') return '';
        return window.localStorage?.getItem(MISTRAL_API_KEY_STORAGE_KEY)?.trim() || '';
    } catch {
        return '';
    }
}
