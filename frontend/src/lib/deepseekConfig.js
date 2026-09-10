// Official API model table, checked 2026-09-10:
// https://api-docs.deepseek.com/quick_start/pricing/
// The retired deepseek-v4-flash-vision-exp alias now routes to V4.1 Flash.
export const DEEPSEEK_MODEL = 'deepseek-flash';
export const DEEPSEEK_LABEL = 'DeepSeek V4.1 Flash';
export const DEEPSEEK_API_KEY_STORAGE_KEY = 'deepseek_api_key';
export const DEEPSEEK_IMAGE_MAX_BYTES = 10 * 1024 * 1024;
export const DEEPSEEK_TIMEOUT_MS = 120_000;

export function getDeepSeekApiKey() {
    try {
        return globalThis.localStorage?.getItem(DEEPSEEK_API_KEY_STORAGE_KEY)?.trim() || '';
    } catch {
        return '';
    }
}
