import { suryaDimensions, suryaLayout, packSuryaPixels } from './suryaPreprocessing';
import { roundEven } from './falconPreprocessing';

export function lighton3Dimensions(manifest, width, height) {
    if (!(width > 0 && height > 0)) throw new Error('Dimensions de page invalides.');
    const scale = Math.min(1, manifest.max_long_edge / Math.max(width, height));
    // The native pipeline first applies Pillow.thumbnail, then Qwen smart_resize.
    const thumbnailWidth = Math.max(1, roundEven(width * scale));
    const thumbnailHeight = Math.max(1, roundEven(height * scale));
    return { thumbnailWidth, thumbnailHeight,
        size: suryaDimensions(thumbnailWidth, thumbnailHeight, manifest.min_pixels, manifest.max_pixels) };
}

export async function prepareLighton3Image(manifest, bitmap) {
    const { thumbnailWidth, thumbnailHeight, size: [width, height] } = lighton3Dimensions(manifest, bitmap.width, bitmap.height);
    let source = bitmap;
    if (thumbnailWidth !== bitmap.width || thumbnailHeight !== bitmap.height) {
        source = new OffscreenCanvas(thumbnailWidth, thumbnailHeight);
        const ctx = source.getContext('2d');
        ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, thumbnailWidth, thumbnailHeight);
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(bitmap, 0, 0, thumbnailWidth, thumbnailHeight);
    }
    const canvas = new OffscreenCanvas(width, height), ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, width, height);
    const layout = suryaLayout(manifest, width, height);
    return { ...layout, pixels: packSuryaPixels(ctx.getImageData(0, 0, width, height).data, width, layout),
        originalWidth: bitmap.width, originalHeight: bitmap.height };
}
