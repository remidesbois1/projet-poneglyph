import { half, roundEven } from './falconPreprocessing';

export function suryaDimensions(width, height, minPixels = 65536, maxPixels = 1048576) {
    if (!(width > 0 && height > 0) || Math.max(width, height) / Math.min(width, height) > 200) {
        throw new Error('Dimensions d’image incompatibles avec Surya.');
    }
    let w = roundEven(width / 32) * 32, h = roundEven(height / 32) * 32;
    if (w * h > maxPixels) {
        const scale = Math.sqrt(width * height / maxPixels);
        w = Math.max(32, Math.floor(width / scale / 32) * 32);
        h = Math.max(32, Math.floor(height / scale / 32) * 32);
    } else if (w * h < minPixels) {
        const scale = Math.sqrt(minPixels / (width * height));
        w = Math.ceil(width * scale / 32) * 32;
        h = Math.ceil(height * scale / 32) * 32;
    }
    return [w, h];
}

export function suryaRotary(positions) {
    const cos = new Float32Array(positions.length * 64), sin = new Float32Array(cos.length);
    for (let s = 0; s < positions.length; s++) for (let f = 0; f < 32; f++) {
        const axis = f % 3 === 1 && f < 33 ? 1 : f % 3 === 2 && f < 30 ? 2 : 0;
        const angle = Math.fround(positions[s][axis] * Math.fround(1 / (10000000 ** (f / 32))));
        cos[s * 64 + f] = cos[s * 64 + f + 32] = Math.cos(angle);
        sin[s * 64 + f] = sin[s * 64 + f + 32] = Math.sin(angle);
    }
    return { cos, sin };
}

export function suryaLayout(manifest, width, height) {
    const rows = height / 16, cols = width / 16, patches = rows * cols;
    const indices = new BigInt64Array(4 * patches), weights = new Float32Array(4 * patches);
    const cos = new Float32Array(patches * 64), sin = new Float32Array(cos.length);
    const patchPositions = [];
    for (let by = 0; by < rows; by += 2) for (let bx = 0; bx < cols; bx += 2) {
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const y = by + dy, x = bx + dx, n = patchPositions.length;
            patchPositions.push([y, x]);
            const py = Math.fround(y * 47 / (rows - 1)), px = Math.fround(x * 47 / (cols - 1));
            const y0 = Math.floor(py), x0 = Math.floor(px), y1 = Math.min(47, y0 + 1), x1 = Math.min(47, x0 + 1);
            const fy = py - y0, fx = px - x0;
            [y0 * 48 + x0, y0 * 48 + x1, y1 * 48 + x0, y1 * 48 + x1].forEach((v, i) => { indices[i * patches + n] = BigInt(v); });
            [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx].forEach((v, i) => { weights[i * patches + n] = v; });
            for (let f = 0; f < 32; f++) {
                const angle = Math.fround((f < 16 ? y : x) * Math.fround(1 / (10000 ** ((f % 16) / 16))));
                cos[n * 64 + f] = cos[n * 64 + f + 32] = Math.cos(angle);
                sin[n * 64 + f] = sin[n * 64 + f + 32] = Math.sin(angle);
            }
        }
    }
    const imageTokens = patches / 4, offset = manifest.prompt_chunks[0].length;
    const ids = [...manifest.prompt_chunks[0], ...Array(imageTokens).fill(manifest.image_token_id), ...manifest.prompt_chunks[1]];
    const positions = [];
    for (let i = 0; i < offset; i++) positions.push([i, i, i]);
    for (let y = 0; y < rows / 2; y++) for (let x = 0; x < cols / 2; x++) positions.push([offset, offset + y, offset + x]);
    const afterImage = offset + Math.max(rows, cols) / 2;
    for (let i = 0; i < manifest.prompt_chunks[1].length; i++) positions.push([afterImage + i, afterImage + i, afterImage + i]);
    return { rows, cols, patches, patchPositions, indices, weights, cos, sin, ids, positions,
        imageOffset: offset, nextPosition: afterImage + manifest.prompt_chunks[1].length };
}

export function packSuryaPixels(rgba, width, layout) {
    const pixels = new Uint16Array(layout.patches * 1536);
    let index = 0;
    for (const [py, px] of layout.patchPositions) {
        for (let c = 0; c < 3; c++) for (let t = 0; t < 2; t++) {
            for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
                const value = rgba[((py * 16 + y) * width + px * 16 + x) * 4 + c];
                pixels[index++] = half(value / 127.5 - 1);
            }
        }
    }
    return pixels;
}

export async function prepareSuryaImage(manifest, bitmap) {
    const [width, height] = suryaDimensions(bitmap.width, bitmap.height, manifest.min_pixels, manifest.max_pixels);
    const canvas = new OffscreenCanvas(width, height), ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, width, height);
    const layout = suryaLayout(manifest, width, height);
    return { ...layout, pixels: packSuryaPixels(ctx.getImageData(0, 0, width, height).data, width, layout) };
}

export function cleanSuryaText(text) {
    text = text.trim();
    if (text.length >= 2 && text[0] === text.at(-1) && ['"', "'", '`'].includes(text[0])) text = text.slice(1, -1).trim();
    return text.replace(/\s+/g, ' ').trim();
}
