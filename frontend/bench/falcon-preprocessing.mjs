// Reproduce the CPU-only preprocessing comparison with synthetic pixels.
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { half, packImage } from '../src/lib/falconPreprocessing.js';

function oldRotary(manifest, times, spatial) {
    const cos = new Float32Array(times.length * 16 * 32), sin = new Float32Array(cos.length);
    for (let s = 0; s < times.length; s++) for (let h = 0; h < 16; h++) for (let f = 0; f < 32; f++) {
        const frequency = manifest.golden_frequencies[h][f - 16], pos = spatial[s];
        const angle = Math.fround(f < 16 ? times[s] / (10000 ** (2 * f / 32))
            : pos ? pos[0] * frequency[0] + pos[1] * frequency[1] : 0);
        const i = (s * 16 + h) * 32 + f; cos[i] = Math.cos(angle); sin[i] = Math.sin(angle);
    }
    return { cos, sin };
}
function oldPack(manifest, rgba, width, height) {
    const c = manifest.config, columns = width / 16, rows = height / 16, count = columns * rows;
    const ids = [...manifest.prompt_chunks[0], c.image_cls_token_id, c.image_reg_1_token_id, c.image_reg_2_token_id,
        c.image_reg_3_token_id, c.image_reg_4_token_id, ...Array(count).fill(c.img_id), c.img_end_id, ...manifest.prompt_chunks[1]];
    const start = manifest.prompt_chunks[0].length, imageEnd = start + 5 + count;
    const pixels = new Uint16Array(ids.length * 768), mask = new Uint8Array(ids.length), spatial = Array(ids.length).fill(null);
    for (let py = 0; py < rows; py++) for (let px = 0; px < columns; px++) {
        const token = start + 5 + py * columns + px; mask[token] = 1;
        spatial[token] = [Math.fround(Math.sqrt(rows / columns) * (rows === 1 ? -1 : -1 + 2 * py / (rows - 1))),
            Math.fround(Math.sqrt(columns / rows) * (columns === 1 ? -1 : -1 + 2 * px / (columns - 1)))];
        let i = token * 768;
        for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
            const offset = ((py * 16 + y) * width + px * 16 + x) * 4;
            for (let channel = 0; channel < 3; channel++) pixels[i++] = half(rgba[offset + channel] / 127.5 - 1);
        }
    }
    const noIncrease = new Set([c.img_id, c.img_end_id, c.image_reg_1_token_id, c.image_reg_2_token_id, c.image_reg_3_token_id, c.image_reg_4_token_id]);
    let position = -1;
    const positions = ids.map(id => { if (!noIncrease.has(id)) position++; return Math.max(position, 0); });
    const attention = new Uint16Array(ids.length ** 2);
    for (let q = 0; q < ids.length; q++) for (let k = 0; k < ids.length; k++)
        if (!(k <= q || (q >= start && q < imageEnd && k >= start && k < imageEnd))) attention[q * ids.length + k] = half(-10000);
    return { ids, pixels, mask, attention, ...oldRotary(manifest, positions, spatial), lastPosition: position };
}
const manifest = { prompt_chunks: [[], [46021, 790, 2757, 3463, 1211, 1112, 6883, 537, 709, 257]],
    config: { image_cls_token_id: 244, image_reg_1_token_id: 245, image_reg_2_token_id: 246, image_reg_3_token_id: 247,
        image_reg_4_token_id: 248, img_id: 227, img_end_id: 230 },
    golden_frequencies: Array.from({ length: 16 }, (_, h) => Array.from({ length: 16 }, (_, f) => [Math.sin(h + f), Math.cos(h - f)])) };
const width = 224, height = 320, rgba = new Uint8ClampedArray(width * height * 4);
for (let i = 0; i < rgba.length; i++) rgba[i] = i % 256;
const a = oldPack(manifest, rgba, width, height), b = packImage(manifest, rgba, width, height);
for (const key of Object.keys(a)) assert.deepEqual(b[key], a[key], key);
function bench(fn) {
    for (let i = 0; i < 3; i++) fn();
    const values = [];
    for (let i = 0; i < 10; i++) { const start = performance.now(); fn(); values.push(performance.now() - start); }
    return values.sort((x, y) => x - y)[5];
}
console.log(JSON.stringify({ width, height, equal: true, baseline_ms: bench(() => oldPack(manifest, rgba, width, height)),
    optimized_ms: bench(() => packImage(manifest, rgba, width, height)) }));
