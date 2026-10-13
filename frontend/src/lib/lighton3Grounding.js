const TEXT_LABELS = new Set(['text', 'title', 'list', 'header', 'footer', 'page_number',
    'footnote', 'caption', 'formula', 'code', 'table', 'aside_text']);

export function parseLighton3Grounding(raw) {
    const headers = [...raw.matchAll(/^!\[([a-z_]+\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/gm)];
    const bubbles = [];
    let invalidBlocks = 0;
    headers.forEach((header, index) => {
        const label = header[1].replace(/\+$/, '');
        const bbox = header.slice(2).map(Number);
        const content = raw.slice(header.index + header[0].length, headers[index + 1]?.index ?? raw.length)
            .normalize('NFC').replace(/\s+/g, ' ').trim();
        if (bbox.some(value => value > 1000) || bbox[0] >= bbox[2] || bbox[1] >= bbox[3] || !content) {
            invalidBlocks++; return;
        }
        if (TEXT_LABELS.has(label)) bubbles.push({ bbox, content, label, order: bubbles.length + 1 });
    });
    const markers = [...raw.matchAll(/!\[[^\]\n]+\]\(/g)].length;
    invalidBlocks += Math.max(0, markers - headers.length);
    if (raw.trim() && (!headers.length || raw.slice(0, headers[0].index).trim())) invalidBlocks++;
    return { bubbles, invalidBlocks };
}

export function lighton3BubbleToPixels(bubble, width, height) {
    if (!(width > 0 && height > 0)) throw new Error('Dimensions de page invalides.');
    const [x1, y1, x2, y2] = bubble.bbox;
    const x = Math.round(x1 * width / 1000), y = Math.round(y1 * height / 1000);
    return { x, y, w: Math.round(x2 * width / 1000) - x, h: Math.round(y2 * height / 1000) - y,
        texte_propose: bubble.content, order: bubble.order };
}
