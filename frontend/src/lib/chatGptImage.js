const CHATGPT_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp']);

export async function prepareChatGptImage(originalBlob) {
    if (!(originalBlob instanceof Blob) || !originalBlob.size) throw new Error('Image originale manquante.');
    if (CHATGPT_IMAGE_TYPES.has(originalBlob.type)) return originalBlob;

    // Decode the private original at its full size. PNG adds no further lossy compression.
    let bitmap;
    let canvas;
    try {
        bitmap = await createImageBitmap(originalBlob);
        canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas indisponible.');
        context.drawImage(bitmap, 0, 0);
        const png = await canvas.convertToBlob({ type: 'image/png' });
        if (!png.size || png.type !== 'image/png') throw new Error('Conversion PNG invalide.');
        return png;
    } catch (error) {
        throw new Error("Impossible de convertir l’image originale en PNG pour ChatGPT.", { cause: error });
    } finally {
        bitmap?.close();
        if (canvas) {
            canvas.width = 0;
            canvas.height = 0;
        }
    }
}
