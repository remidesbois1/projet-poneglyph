export function isSelectableOcrModel(model, isSandbox = false) {
    return Boolean(model && model.key !== 'gemini' && model.task !== 'page-ocr-bbox' && (!isSandbox || model.type !== 'api' || model.byok === true));
}
