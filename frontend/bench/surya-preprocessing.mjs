// Reproducible CPU microbenchmark of exact pixel packing, independent of a GPU.
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({ configFile: false, root: fileURLToPath(new URL('../', import.meta.url)),
    server: { middlewareMode: true } });
try {
    const { suryaLayout, packSuryaPixels } = await server.ssrLoadModule('/src/lib/suryaPreprocessing.js');
    const { half } = await server.ssrLoadModule('/src/lib/falconPreprocessing.js');
    const width = 224, height = 320;
    const layout = suryaLayout({ prompt_chunks: [[1], [2]], image_token_id: 11 }, width, height);
    const rgba = Uint8Array.from({ length: width * height * 4 }, (_, i) => (i * 37 + Math.floor(i / 7)) % 256);
    const reference = () => {
        const pixels = new Uint16Array(layout.patches * 1536); let index = 0;
        for (const [py, px] of layout.patchPositions) for (let c = 0; c < 3; c++) for (let t = 0; t < 2; t++) {
            for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
                pixels[index++] = half(rgba[((py * 16 + y) * width + px * 16 + x) * 4 + c] / 127.5 - 1);
            }
        }
        return pixels;
    };
    const optimized = () => packSuryaPixels(rgba, width, layout);
    assert.deepEqual(optimized(), reference());
    for (let i = 0; i < 10; i++) { reference(); optimized(); }
    const measure = fn => {
        const samples = [];
        for (let run = 0; run < 5; run++) {
            const start = performance.now();
            for (let i = 0; i < 50; i++) fn();
            samples.push((performance.now() - start) / 50);
        }
        return samples.sort((a, b) => a - b)[2];
    };
    const before = measure(reference), after = measure(optimized);
    console.log(JSON.stringify({ width, height, reference_ms: before, optimized_ms: after,
        speedup: before / after, identical_fp16_values: true }, null, 2));
} finally { await server.close(); }
