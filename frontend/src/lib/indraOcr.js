/**
 * Indra OCR bubble pipeline (Poneglyph fine-tune of datalab-to/indra-ocr v1.1.0).
 *
 * Indra det.onnx finds the text lines of a bubble crop, the lines are put side by side
 * (each as Indra's tall 96 px window) and Indra rec.onnx reads them in one CTC pass.
 * Every step reproduces the reference Python (indra_ocr 1.1.2 + OpenCV) used for
 * training and evaluation: docker_scripts/eval_indra_ocr. The functions are pure;
 * the ONNX Runtime module and sessions are passed in, so the worker and the Node
 * parity harness run the same code.
 */

const AREA_EPS = 1e-3;

/** Python round(): half to even. */
export function roundHalfEven(value) {
    const floor = Math.floor(value);
    const diff = value - floor;
    if (diff > 0.5) return floor + 1;
    if (diff < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
}

/** cv2.COLOR_RGB2GRAY on RGBA bytes (OpenCV fixed point, 14 bits). */
export function rgbaToGray(rgba, width, height) {
    const gray = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
        gray[i] = (rgba[j] * 4899 + rgba[j + 1] * 9617 + rgba[j + 2] * 1868 + 8192) >> 14;
    }
    return gray;
}

function areaTable(srcSize, dstSize) {
    const scale = srcSize / dstSize;
    const table = [];
    for (let d = 0; d < dstSize; d++) {
        const fs1 = d * scale, fs2 = fs1 + scale;
        const cell = Math.min(scale, srcSize - fs1);
        let s1 = Math.ceil(fs1), s2 = Math.floor(fs2);
        s2 = Math.min(s2, srcSize - 1);
        s1 = Math.min(s1, s2);
        const taps = [];
        if (s1 - fs1 > AREA_EPS) taps.push([s1 - 1, (s1 - fs1) / cell]);
        for (let s = s1; s < s2; s++) taps.push([s, 1 / cell]);
        if (fs2 - s2 > AREA_EPS) taps.push([s2, Math.min(Math.min(fs2 - s2, 1), cell) / cell]);
        table.push(taps);
    }
    return table;
}

/** cv2.resize(INTER_AREA) for a single-channel uint8 image shrunk on both axes. */
export function resizeArea(src, sw, sh, dw, dh) {
    const tx = areaTable(sw, dw), ty = areaTable(sh, dh);
    const rows = new Float64Array(sh * dw);
    for (let y = 0; y < sh; y++) {
        const off = y * sw;
        for (let x = 0; x < dw; x++) {
            let acc = 0;
            for (const [s, w] of tx[x]) acc += src[off + s] * w;
            rows[y * dw + x] = acc;
        }
    }
    const out = new Uint8Array(dw * dh);
    for (let y = 0; y < dh; y++) {
        for (let x = 0; x < dw; x++) {
            let acc = 0;
            for (const [s, w] of ty[y]) acc += rows[s * dw + x] * w;
            out[y * dw + x] = Math.min(255, Math.max(0, Math.round(acc)));
        }
    }
    return out;
}

function linearTable(srcSize, dstSize) {
    const scale = srcSize / dstSize;
    const index = new Int32Array(dstSize), coef = new Int32Array(dstSize);
    for (let d = 0; d < dstSize; d++) {
        let f = (d + 0.5) * scale - 0.5;
        let s = Math.floor(f);
        f -= s;
        if (s < 0) { f = 0; s = 0; }
        if (s >= srcSize - 1) { f = 0; s = srcSize - 1; }
        index[d] = s;
        coef[d] = Math.round((1 - f) * 2048);
    }
    return { index, coef };
}

/** cv2.resize(INTER_LINEAR) for single-channel uint8 (OpenCV 11-bit fixed point). */
export function resizeLinear(src, sw, sh, dw, dh) {
    const tx = linearTable(sw, dw), ty = linearTable(sh, dh);
    const rows = new Int32Array(sh * dw);
    for (let y = 0; y < sh; y++) {
        const off = y * sw;
        for (let x = 0; x < dw; x++) {
            const s = tx.index[x], a = tx.coef[x];
            const next = s + 1 < sw ? src[off + s + 1] : src[off + s];
            rows[y * dw + x] = src[off + s] * a + next * (2048 - a);
        }
    }
    const out = new Uint8Array(dw * dh);
    for (let y = 0; y < dh; y++) {
        const s = ty.index[y], b0 = ty.coef[y], b1 = 2048 - b0;
        const r0 = s * dw, r1 = (s + 1 < sh ? s + 1 : s) * dw;
        for (let x = 0; x < dw; x++) {
            const v = ((((rows[r0 + x] >> 4) * b0) >> 16) + (((rows[r1 + x] >> 4) * b1) >> 16) + 2) >> 2;
            out[y * dw + x] = v > 255 ? 255 : v < 0 ? 0 : v;
        }
    }
    return out;
}

/** OpenCV interpolation as Indra chooses it: area when shrinking, linear otherwise. */
export function resizeGray(src, sw, sh, dw, dh, shrink) {
    if (sw === dw && sh === dh) return src;
    return shrink && dw <= sw && dh <= sh ? resizeArea(src, sw, sh, dw, dh) : resizeLinear(src, sw, sh, dw, dh);
}

/** indra_ocr.det_post.page_to_input: ink map padded to multiples of 32. */
export function pageToInput(gray, width, height, longSide) {
    const scale = longSide / Math.max(width, height);
    const nh = Math.max(1, roundHalfEven(height * scale)), nw = Math.max(1, roundHalfEven(width * scale));
    const resized = resizeGray(gray, width, height, nw, nh, scale < 1);
    const ph = Math.ceil(nh / 32) * 32, pw = Math.ceil(nw / 32) * 32;
    const data = new Float32Array(ph * pw);
    for (let y = 0; y < nh; y++) {
        for (let x = 0; x < nw; x++) data[y * pw + x] = 1 - resized[y * nw + x] / 255;
    }
    return { data, height: ph, width: pw, scale };
}

/** indra_ocr.det_post.decode: core components (4-connected) with mean top/bottom distances. */
export function decodeLines(out, channels, h, w, { stride = 2, coreXShrink = 0.15 } = {}) {
    const plane = h * w;
    const labels = new Int32Array(plane);
    const stack = new Int32Array(plane);
    const boxes = [];
    let next = 0;
    for (let start = 0; start < plane; start++) {
        if (labels[start] || out[start] <= 0) continue; // sigmoid(logit) > 0.5
        next++;
        let top = 0, bottom = 0, score = 0, count = 0, minX = w, maxX = -1, sp = 0;
        stack[sp++] = start;
        labels[start] = next;
        while (sp) {
            const p = stack[--sp];
            const y = (p / w) | 0, x = p - y * w;
            const cy = (y + 0.5) * stride;
            top += cy - out[plane + p];
            bottom += cy + out[2 * plane + p];
            score += 1 / (1 + Math.exp(-out[p]));
            count++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
                if (q >= 0 && !labels[q] && out[q] > 0) { labels[q] = next; stack[sp++] = q; }
            }
        }
        if (count < 2) continue;
        const y0 = top / count, y1 = bottom / count;
        const lh = Math.max(1, y1 - y0);
        const ex = Math.min(coreXShrink * lh, 0.5 * lh);
        boxes.push([minX * stride - ex, y0, (maxX + 1) * stride + ex, y1, score / count]);
    }
    return boxes;
}

/** Production line order (ocr.worker.js sortLineBoxes): half-pitch bands, then x. */
export function sortLineBoxes(boxes) {
    if (!boxes.length) return boxes;
    const heights = boxes.map(b => b[3] - b[1]).sort((a, b) => a - b);
    const band = Math.max(10, heights[Math.floor(heights.length / 2)] * 0.65);
    const key = b => Math.round((b[1] + b[3]) / 2 / band);
    return [...boxes].sort((a, b) => key(a) - key(b) || a[0] - b[0]);
}

/** Line boxes of a bubble in crop pixels, in reading order. */
export async function detectLines(ort, session, gray, width, height, scale) {
    const longSide = Math.max(32, roundHalfEven(Math.max(width, height) * scale));
    const input = pageToInput(gray, width, height, longSide);
    const result = await session.run({ x: new ort.Tensor('float32', input.data, [1, 1, input.height, input.width]) });
    const out = result[session.outputNames[0]];
    const [, channels, oh, ow] = out.dims;
    const boxes = [];
    for (const [x0, y0, x1, y1] of decodeLines(out.data, channels, oh, ow)) {
        const b = [
            Math.min(Math.max(x0 / input.scale, 0), width), Math.min(Math.max(y0 / input.scale, 0), height),
            Math.min(Math.max(x1 / input.scale, 0), width), Math.min(Math.max(y1 / input.scale, 0), height),
        ];
        if (b[2] - b[0] >= 2 && b[3] - b[1] >= 2) boxes.push(b);
    }
    return sortLineBoxes(boxes);
}

/** numpy.percentile (linear) of a uint8 region from its histogram (rec_preproc._percentiles). */
function percentiles(values, qs) {
    const hist = new Float64Array(256);
    for (const v of values) hist[v]++;
    const cum = new Float64Array(256);
    let total = 0;
    for (let i = 0; i < 256; i++) { total += hist[i]; cum[i] = total; }
    const searchRight = target => { let i = 0; while (i < 256 && cum[i] <= target) i++; return i; };
    const n = values.length;
    return qs.map(q => {
        const x = q * (n - 1), i = Math.floor(x), g = x - i;
        const lo = searchRight(i), hi = searchRight(Math.min(i + 1, n - 1));
        const d = hi - lo;
        return g >= 0.5 ? hi - d * (1 - g) : lo + d * g;
    });
}

/** rec_preproc.is_dark_line: light text on a dark ground. */
export function isDarkLine(gray, width, height, box) {
    const x0 = Math.max(0, Math.trunc(box[0])), y0 = Math.max(0, Math.trunc(box[1]));
    const x1 = Math.min(width, Math.trunc(box[2]) + 1), y1 = Math.min(height, Math.trunc(box[3]) + 1);
    if (x1 <= x0 || y1 <= y0) return false;
    const region = new Uint8Array((x1 - x0) * (y1 - y0));
    let k = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) region[k++] = gray[y * width + x];
    const [lo, med, hi] = percentiles(region, [0.02, 0.5, 0.98]);
    return hi - med > 2 * (med - lo) + 8;
}

/** rec_preproc.crop_line with the tall window: ink map [lineHeight * window, W]. */
export function cropLine(gray, width, height, box, { lineHeight = 32, window = 3, padX = 0.3, padY = 0.2, maxWidth = 4096 } = {}) {
    const pady = (window * (1 + 2 * padY) - 1) / 2;
    const outH = lineHeight * window;
    const [x0, y0, x1, y1] = box;
    const h = Math.max(1, y1 - y0);
    const ix0 = Math.floor(x0 - padX * h), iy0 = Math.floor(y0 - pady * h);
    const ix1 = Math.ceil(x1 + padX * h), iy1 = Math.ceil(y1 + pady * h);
    const cw = Math.max(1, ix1 - ix0), ch = Math.max(1, iy1 - iy0);
    const dark = isDarkLine(gray, width, height, box);
    const crop = new Uint8Array(cw * ch).fill(dark ? 0 : 255);
    const sx0 = Math.max(0, ix0), sy0 = Math.max(0, iy0), sx1 = Math.min(width, ix1), sy1 = Math.min(height, iy1);
    for (let y = sy0; y < sy1; y++) {
        crop.set(gray.subarray(y * width + sx0, y * width + sx1), (y - iy0) * cw + (sx0 - ix0));
    }
    const outW = Math.max(8, Math.min(maxWidth, roundHalfEven(cw * outH / ch)));
    const resized = resizeGray(crop, cw, ch, outW, outH, ch > outH);
    const ink = new Float32Array(outW * outH);
    for (let i = 0; i < ink.length; i++) ink[i] = dark ? resized[i] / 255 : 1 - resized[i] / 255;
    return { data: ink, width: outW, height: outH };
}

/** The bubble's line windows side by side, right-padded like rec_preproc.pad_batch. */
export function stitchLines(crops, { multiple = 8, minWidth = 32 } = {}) {
    const height = crops[0].height;
    const total = crops.reduce((sum, c) => sum + c.width, 0);
    const padded = Math.ceil(Math.max(total, minWidth) / multiple) * multiple;
    const data = new Float32Array(height * padded);
    let x = 0;
    for (const c of crops) {
        for (let y = 0; y < height; y++) data.set(c.data.subarray(y * c.width, (y + 1) * c.width), y * padded + x);
        x += c.width;
    }
    return { data, height, width: padded, validWidth: total };
}

/** Greedy CTC on rec.onnx's argmax ids: drop repeats and blanks (class 0). */
export function decodeCtcIds(ids, length, chars) {
    let text = '', prev = -1;
    for (let t = 0; t < length; t++) {
        const id = Number(ids[t]);
        if (id !== 0 && id !== prev && id <= chars.length) text += chars[id - 1];
        prev = id;
    }
    return text;
}

/** Full bubble OCR: returns the raw recognizer text (before the shared text rules). */
export async function runIndraBubble(ort, { detSession, recSession, manifest }, gray, width, height) {
    const t0 = performance.now();
    const found = await detectLines(ort, detSession, gray, width, height, manifest.det_scale);
    const t1 = performance.now();
    const boxes = found.length ? found : [[0, 0, width, height]];
    const crops = boxes.map(box => cropLine(gray, width, height, box, { lineHeight: manifest.rec_height, window: manifest.rec_window }));
    const input = stitchLines(crops);
    const result = await recSession.run({
        x: new ort.Tensor('float32', input.data, [1, 1, input.height, input.width]),
        widths: new ort.Tensor('int64', BigInt64Array.from([BigInt(input.validWidth)]), [1]),
    });
    const raw = decodeCtcIds(result.ids.data, Number(result.lengths.data[0]), manifest.vocab);
    const t2 = performance.now();
    return { raw, boxes: found, timings: { detectMs: t1 - t0, recognizeMs: t2 - t1 } };
}
