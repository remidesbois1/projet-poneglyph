// Match the native uint8 Torchvision/PyTorch 2.8 antialiased bicubic path.
// Reference: aten/src/ATen/native/cpu/UpSampleKernel.cpp (Keys a=-0.5,
// normalized double coefficients, adaptive int16 precision, two uint8 passes).
function coefficients(inputSize, outputSize) {
    const scale = inputSize / outputSize, support = 2 * Math.max(1, scale), inverse = 1 / Math.max(1, scale);
    const maxSize = 2 * Math.ceil(support) + 1;
    let maximum = 0;
    const rows = Array.from({ length: outputSize }, (_, i) => {
        const center = scale * (i + .5);
        const start = Math.max(0, Math.trunc(center - support + .5));
        const count = Math.max(0, Math.min(maxSize, Math.min(inputSize, Math.trunc(center + support + .5)) - start));
        const weights = new Float64Array(count);
        let sum = 0;
        for (let j = 0; j < count; j++) {
            const x = Math.abs((j + start - center + .5) * inverse);
            weights[j] = x < 1 ? (1.5 * x - 2.5) * x * x + 1 : x < 2 ? ((-.5 * x + 2.5) * x - 4) * x + 2 : 0;
            sum += weights[j];
        }
        for (let j = 0; j < count; j++) { weights[j] /= sum; maximum = Math.max(maximum, weights[j]); }
        return { start, weights };
    });
    let precision = 0;
    while (precision < 22 && Math.trunc(.5 + maximum * 2 ** (precision + 1)) < 32768) precision++;
    const unit = 2 ** precision;
    for (const row of rows) row.weights = Int32Array.from(row.weights, w => Math.trunc(w * unit + (w < 0 ? -.5 : .5)));
    return { rows, unit, half: unit / 2 };
}

export function resizeUint8Bicubic(rgba, width, height, outputWidth, outputHeight) {
    if (![width, height, outputWidth, outputHeight].every(n => Number.isInteger(n) && n > 0) || rgba.length !== width * height * 4) {
        throw new Error('Dimensions ou pixels invalides pour le redimensionnement.');
    }
    let source = rgba, sourceWidth = width;
    if (width !== outputWidth) {
        const { rows, unit, half } = coefficients(width, outputWidth);
        const horizontal = new Uint8ClampedArray(outputWidth * height * 4);
        for (let y = 0; y < height; y++) for (let x = 0; x < outputWidth; x++) {
            const { start, weights } = rows[x], input = (y * width + start) * 4, output = (y * outputWidth + x) * 4;
            for (let channel = 0; channel < 3; channel++) {
                let value = half;
                for (let i = 0; i < weights.length; i++) value += source[input + i * 4 + channel] * weights[i];
                horizontal[output + channel] = Math.max(0, Math.min(255, Math.floor(value / unit)));
            }
            horizontal[output + 3] = 255;
        }
        source = horizontal; sourceWidth = outputWidth;
    }
    if (height === outputHeight) return source;
    const { rows, unit, half } = coefficients(height, outputHeight);
    const output = new Uint8ClampedArray(outputWidth * outputHeight * 4);
    for (let y = 0; y < outputHeight; y++) for (let x = 0; x < outputWidth; x++) {
        const { start, weights } = rows[y], input = (start * sourceWidth + x) * 4, dst = (y * outputWidth + x) * 4;
        for (let channel = 0; channel < 3; channel++) {
            let value = half;
            for (let i = 0; i < weights.length; i++) value += source[input + i * sourceWidth * 4 + channel] * weights[i];
            output[dst + channel] = Math.max(0, Math.min(255, Math.floor(value / unit)));
        }
        output[dst + 3] = 255;
    }
    return output;
}
