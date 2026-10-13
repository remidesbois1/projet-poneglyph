import { suryaDimensions, suryaLayout, packSuryaPixels } from './suryaPreprocessing';
import { resizeUint8Bicubic } from './bicubicResize';

export async function prepareLighton3BubbleImage(manifest, bitmap) {
    const [width, height] = suryaDimensions(bitmap.width, bitmap.height, manifest.min_pixels, manifest.max_pixels);
    // Decode RGB at the source dimensions. Canvas never resizes the bubble:
    // the uint8 bicubic implementation matches the native Torchvision processor.
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, bitmap.width, bitmap.height);
    ctx.drawImage(bitmap, 0, 0);
    const source = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const rgba = resizeUint8Bicubic(source, bitmap.width, bitmap.height, width, height);
    const layout = suryaLayout(manifest, width, height);
    return { ...layout, pixels: packSuryaPixels(rgba, width, layout) };
}
