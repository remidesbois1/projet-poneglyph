import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prepareChatGptImage } from './chatGptImage';

describe('ChatGPT original image preparation', () => {
    let bitmap;
    let canvas;
    let drawImage;
    let png;

    beforeEach(() => {
        bitmap = { width: 1824, height: 2736, close: vi.fn() };
        drawImage = vi.fn();
        png = new Blob(['png-image'], { type: 'image/png' });
        canvas = { getContext: vi.fn(() => ({ drawImage })), convertToBlob: vi.fn().mockResolvedValue(png) };
        vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap));
        vi.stubGlobal('OffscreenCanvas', vi.fn(function (width, height) {
            canvas.width = width;
            canvas.height = height;
            return canvas;
        }));
    });

    afterEach(() => vi.unstubAllGlobals());

    it.each(['image/jpeg', 'image/png', 'image/gif', 'image/webp'])('preserves supported %s bytes', async (type) => {
        const original = new Blob(['original'], { type });
        await expect(prepareChatGptImage(original)).resolves.toBe(original);
        expect(createImageBitmap).not.toHaveBeenCalled();
    });

    it.each(['image/avif', 'application/octet-stream', ''])('converts %s from the full private original without resizing', async (type) => {
        const original = new Blob(['private-original'], { type });
        await expect(prepareChatGptImage(original)).resolves.toBe(png);
        expect(createImageBitmap).toHaveBeenCalledWith(original);
        expect(OffscreenCanvas).toHaveBeenCalledWith(1824, 2736);
        expect(drawImage).toHaveBeenCalledWith(bitmap, 0, 0);
        expect(canvas.convertToBlob).toHaveBeenCalledWith({ type: 'image/png' });
        expect(bitmap.close).toHaveBeenCalledOnce();
        expect(canvas.width).toBe(0);
        expect(canvas.height).toBe(0);
    });

    it('rejects a corrupt original before making a ChatGPT request', async () => {
        createImageBitmap.mockRejectedValue(new Error('Invalid image'));
        await expect(prepareChatGptImage(new Blob(['corrupt'], { type: 'image/avif' })))
            .rejects.toThrow(/convertir l’image originale en PNG/);
        expect(OffscreenCanvas).not.toHaveBeenCalled();
    });

    it('releases decoded pixels if PNG encoding fails', async () => {
        canvas.convertToBlob.mockRejectedValue(new Error('Encoding failed'));
        await expect(prepareChatGptImage(new Blob(['image'], { type: 'image/avif' })))
            .rejects.toThrow(/convertir l’image originale en PNG/);
        expect(bitmap.close).toHaveBeenCalledOnce();
        expect(canvas.width).toBe(0);
        expect(canvas.height).toBe(0);
    });

    it('rejects an empty or incorrectly typed encoder result', async () => {
        for (const invalid of [new Blob([], { type: 'image/png' }), new Blob(['avif'], { type: 'image/avif' })]) {
            canvas.convertToBlob.mockResolvedValue(invalid);
            await expect(prepareChatGptImage(new Blob(['image'], { type: 'image/avif' })))
                .rejects.toThrow(/convertir l’image originale en PNG/);
        }
    });
});
