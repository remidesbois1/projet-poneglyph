const OCR_LETTER_RE = /\p{L}/u;

export function capitalizeOcrSentenceStarts(text) {
    let capitalizeNextLetter = false;

    return [...String(text || '')].map(char => {
        if (capitalizeNextLetter && OCR_LETTER_RE.test(char)) {
            capitalizeNextLetter = false;
            return char.toLocaleUpperCase('fr-FR');
        }

        if (char === '?' || char === '!' || char === '.' || char === '\u2026') {
            capitalizeNextLetter = true;
        }
        return char;
    }).join('');
}

const OCR_RULE_TOKEN_RE = /[\p{L}]+(?:['\u2019][\p{L}]+)?/gu;

function normalizeOcrRuleSpacing(text) {
    return String(text || '')
        .trim()
        .replace(/\u2026/g, '...')
        .replace(/\s+/g, ' ')
        .replace(/\s*\.\s*\.\s*\./g, '<ELLIPSIS>')
        .replace(/\s+([,.])/g, '$1')
        .replace(/([,.])(\S)/g, '$1 $2')
        .replace(/\s*([!?;:]+)/g, ' $1')
        .replace(/<ELLIPSIS>/g, '...')
        .replace(/\s+/g, ' ')
        .trim();
}

function isOcrWeirdCase(token) {
    const letters = [...token].filter(char => OCR_LETTER_RE.test(char));
    if (letters.length < 2) return false;
    const hasUpper = letters.some(char => char === char.toLocaleUpperCase('fr-FR') && char !== char.toLocaleLowerCase('fr-FR'));
    const hasLower = letters.some(char => char === char.toLocaleLowerCase('fr-FR') && char !== char.toLocaleUpperCase('fr-FR'));
    if (!hasUpper || !hasLower) return false;
    const titleLike = token[0] === token[0].toLocaleUpperCase('fr-FR')
        && token.slice(1) === token.slice(1).toLocaleLowerCase('fr-FR');
    return !titleLike;
}

/**
 * Text rules learned from the training labels only (browser manifest `postprocess`):
 * French punctuation spacing, lexicon casing, then sentence starts. Shared by the
 * PP-OCRv6 and Indra bubble recognizers.
 */
export function applyOcrTextRules(text, rules) {
    if (!rules?.enabled || !rules.lexicon) return capitalizeOcrSentenceStarts(text);

    const lexicon = rules.lexicon || {};
    const minCount = Number(rules.min_count || 1);
    const minRatio = Number(rules.min_ratio || 1);
    const maxLowerRatio = Number(rules.max_lower_ratio || 0);
    const sentenceStart = Boolean(rules.sentence_start);
    const spaced = normalizeOcrRuleSpacing(text);

    const lexiconCorrected = spaced.replace(OCR_RULE_TOKEN_RE, (token, offset) => {
        const key = token.toLocaleLowerCase('fr-FR');
        const entry = lexicon[key];
        if (!entry?.best) return token;
        if (isOcrWeirdCase(token)) return entry.best;

        const total = Number(entry.total || 0);
        const bestRatio = Number(entry.best_ratio || 0);
        const lowerRatio = Number(entry.lower_ratio || 0);
        const tokenIsLower = token === token.toLocaleLowerCase('fr-FR');
        if (
            total >= minCount
            && bestRatio >= minRatio
            && lowerRatio <= maxLowerRatio
            && entry.best !== key
            && (tokenIsLower || (sentenceStart && offset === 0))
        ) {
            return entry.best;
        }
        return token;
    });
    return capitalizeOcrSentenceStarts(lexiconCorrected);
}

export function fixFrenchPunctuation(text) {
    if (!text) return "";

    text = text.replace(/"/g, '');
    text = text.replace(/'/g, "'");
    text = text.replace(/([^\s!?;:])([!?;:])/g, '$1 $2');
    text = text.replace(/([.,!?:;])(?=[a-zA-Z])/g, '$1 ');
    text = text.replace(/\bI'\b/g, "l'");
    text = text.replace(/\bI\b/g, "Il");
    text = text.replace(/([?!])\s+(\1)/g, '$1$1');
    text = text.replace(/([?!])\s+(\1)/g, '$1$1');
    text = text.replace(/\? !/g, '?!').replace(/! \?/g, '!?');
    text = text.replace(/\s+/g, ' ').trim();

    return text;
}

export function joinLines(lineTexts) {
    if (lineTexts.length === 0) return '';
    let result = lineTexts[0];

    for (let i = 1; i < lineTexts.length; i++) {
        const next = lineTexts[i];
        const lastChar = result.slice(-1);
        const endsWithSentence = /[.!?…]$/.test(result) || result.endsWith('...');

        if (lastChar === "'" || lastChar === "\u2019" || lastChar === "-") {
            result += endsWithSentence ? next : (next[0]?.toLowerCase() + next.slice(1));
        } else if (endsWithSentence) {
            result += ' ' + next;
        } else {
            result += ' ' + (next[0]?.toLowerCase() + next.slice(1));
        }
    }

    return result;
}

export function splitIntoLines(imageData, width, height) {
    const projection = new Float32Array(height);

    for (let y = 0; y < height; y++) {
        let darkPixels = 0;
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const gray = imageData[idx] * 0.299 + imageData[idx + 1] * 0.587 + imageData[idx + 2] * 0.114;
            if (gray < 128) darkPixels++;
        }
        projection[y] = darkPixels / width;
    }

    const threshold = 0.02;
    const minLineHeight = Math.max(8, Math.round(height * 0.05));
    const minGapHeight = Math.max(2, Math.round(height * 0.02));

    const lines = [];
    let inLine = false;
    let lineStart = 0;

    for (let y = 0; y < height; y++) {
        if (!inLine && projection[y] > threshold) {
            inLine = true;
            lineStart = y;
        } else if (inLine && projection[y] <= threshold) {
            let gapEnd = y;
            while (gapEnd < height && projection[gapEnd] <= threshold) gapEnd++;
            const gapSize = gapEnd - y;

            if (gapSize >= minGapHeight && (y - lineStart) >= minLineHeight) {
                lines.push({ top: lineStart, bottom: y });
                inLine = false;
            }
        }
    }

    if (inLine && (height - lineStart) >= minLineHeight) {
        lines.push({ top: lineStart, bottom: height });
    }

    if (lines.length <= 1) return null;

    const padding = Math.round(height * 0.03);
    return lines.map(l => ({
        top: Math.max(0, l.top - padding),
        bottom: Math.min(height, l.bottom + padding),
    }));
}

export async function cropLineFromBlob(blob, lineRegion, imgWidth) {
    const bitmap = await createImageBitmap(blob);
    const lineHeight = lineRegion.bottom - lineRegion.top;
    const canvas = new OffscreenCanvas(imgWidth, lineHeight);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, lineRegion.top, imgWidth, lineHeight, 0, 0, imgWidth, lineHeight);
    bitmap.close();
    return canvas.convertToBlob({ type: 'image/png' });
}

export async function getImageInfo(blob) {
    const bitmap = await createImageBitmap(blob);
    const width = bitmap.width;
    const height = bitmap.height;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const imageData = ctx.getImageData(0, 0, width, height);
    return { width, height, data: imageData.data };
}
