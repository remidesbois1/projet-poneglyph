const MIN_RALLY_IOU = 0.05;

function isFiniteBox(box) {
    return box && ['x', 'y', 'w', 'h'].every(key => Number.isFinite(Number(box[key])))
        && Number(box.w) > 0
        && Number(box.h) > 0;
}

function normalizedToPixels(bubble, imageWidth, imageHeight) {
    if (!Array.isArray(bubble?.bbox) || bubble.bbox.length !== 4) return null;
    const [x1, y1, x2, y2] = bubble.bbox.map(Number);
    if (![x1, y1, x2, y2, imageWidth, imageHeight].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return null;
    return {
        x: Math.round((x1 / 1000) * imageWidth),
        y: Math.round((y1 / 1000) * imageHeight),
        w: Math.round(((x2 - x1) / 1000) * imageWidth),
        h: Math.round(((y2 - y1) / 1000) * imageHeight),
    };
}

export function boxIou(a, b) {
    const intersectionWidth = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const intersectionHeight = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const intersection = intersectionWidth * intersectionHeight;
    const union = (a.w * a.h) + (b.w * b.h) - intersection;
    return union > 0 ? intersection / union : 0;
}

/**
 * Rallies full-page OCR bubbles (`bbox` normalized to 0–1000) onto YOLO boxes, as the batch OCR does:
 * one-to-one assignment by descending IoU (> 0.05). YOLO is the truth, so page bubbles without a YOLO
 * match are dropped. Matches come back in YOLO order with the YOLO indices that kept no page text.
 *
 * Gemini vision models emit 2D boxes either in the prompt's [x1, y1, x2, y2] order or in their
 * native spatial order [ymin, xmin, ymax, xmax], non-deterministically per response. Both
 * orientations are rallied and the one agreeing best with the detector boxes wins.
 */
export function rallyPageBubblesToYolo(pageBubbles, yoloBoxes, imageWidth, imageHeight) {
    const yoloList = Array.isArray(yoloBoxes) ? yoloBoxes : [];
    const rawPage = Array.isArray(pageBubbles) ? pageBubbles : [];
    const yolo = yoloList
        .map((box, index) => ({ index, box: isFiniteBox(box) ? {
            x: Math.round(Number(box.x)),
            y: Math.round(Number(box.y)),
            w: Math.round(Number(box.w)),
            h: Math.round(Number(box.h)),
        } : null }))
        .filter(item => item.box);

    const direct = rawPage
        .map((bubble, index) => ({ index, text: String(bubble?.content ?? ''), box: normalizedToPixels(bubble, imageWidth, imageHeight) }))
        .filter(item => isFiniteBox(item.box));
    const transposed = rawPage
        .map((bubble, index) => {
            if (!Array.isArray(bubble?.bbox) || bubble.bbox.length !== 4) return null;
            const [a, b, c, d] = bubble.bbox.map(Number);
            return { index, text: String(bubble?.content ?? ''), box: normalizedToPixels({ bbox: [b, a, d, c] }, imageWidth, imageHeight) };
        })
        .filter(item => item && isFiniteBox(item.box));

    const directMatches = rallyBestAssignment(direct, yolo);
    const transposedMatches = transposed.length ? rallyBestAssignment(transposed, yolo) : [];
    const totalIou = matches => matches.reduce((sum, match) => sum + match.iou, 0);
    const matches = totalIou(transposedMatches) > totalIou(directMatches) ? transposedMatches : directMatches;

    const usedYolo = new Set(matches.map(match => match.yoloIndex));
    return {
        matches,
        unmatchedYoloIndices: yoloList.map((_, index) => index).filter(index => !usedYolo.has(index)),
    };
}

function rallyBestAssignment(page, yolo) {
    const candidates = [];
    for (const pageItem of page) {
        for (const yoloItem of yolo) {
            const iou = boxIou(pageItem.box, yoloItem.box);
            if (iou > MIN_RALLY_IOU) candidates.push({ pageItem, yoloIndex: yoloItem.index, iou });
        }
    }
    candidates.sort((a, b) => b.iou - a.iou);

    const usedPage = new Set();
    const usedYolo = new Set();
    const matches = [];
    for (const { pageItem, yoloIndex, iou } of candidates) {
        if (usedPage.has(pageItem.index) || usedYolo.has(yoloIndex)) continue;
        usedPage.add(pageItem.index);
        usedYolo.add(yoloIndex);
        matches.push({ yoloIndex, pageIndex: pageItem.index, iou, text: pageItem.text });
    }
    matches.sort((a, b) => a.yoloIndex - b.yoloIndex);
    return matches;
}
