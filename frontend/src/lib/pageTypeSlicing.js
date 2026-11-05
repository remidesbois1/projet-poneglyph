export const ANNEXE_AUTO_DELETE_THRESHOLD = 0.9;
export const SUMMARY_CANDIDATE_THRESHOLD = 0.8;
const DOUBLE_PAGE_ASPECT_FACTOR = 1.65;
const MIN_DOUBLE_PAGE_ASPECT_RATIO = 1.1;

function aspectRatio(page) {
    const width = page.width;
    const height = page.height;
    return Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0
        ? width / height
        : null;
}

/** A wide scan occupies one CBZ image but two numbered pages in the book. */
export function markDoublePageSpreads(pages) {
    const ratios = pages.map(aspectRatio).filter((ratio) => ratio !== null).sort((a, b) => a - b);
    const usualRatio = ratios[Math.floor((ratios.length - 1) / 2)];

    return pages.map((page) => {
        const ratio = aspectRatio(page);
        return {
            ...page,
            isDoublePage: ratio !== null && usualRatio !== undefined
                && ratio >= Math.max(MIN_DOUBLE_PAGE_ASPECT_RATIO, usualRatio * DOUBLE_PAGE_ASPECT_FACTOR),
        };
    });
}

function probabilityFor(page, label) {
    const value = page.pageType?.probabilities?.[label];
    if (Number.isFinite(value)) return value;
    return page.pageType?.label === label ? page.pageType.confidence : 0;
}

function normalizeChapter(chapter, pageCount) {
    const number = Number.parseInt(chapter?.number, 10);
    const startPage = Number.parseInt(chapter?.startPage, 10);
    const printedPage = Number.parseInt(chapter?.printedPage, 10);
    const title = typeof chapter?.title === 'string' ? chapter.title.trim() : '';
    if (!Number.isInteger(number) || number < 1) return null;
    const validStart = Number.isInteger(startPage) && startPage >= 1 && startPage <= pageCount;
    const validPrinted = Number.isInteger(printedPage) && printedPage > 0;
    if (!validStart && !validPrinted) return null;
    return {
        number,
        startPage: validStart ? startPage : null,
        printedPage: validPrinted ? printedPage : null,
        title: title || `Chapitre ${number}`,
    };
}

function resolveChapterStarts(orderedPages, chapters) {
    // Gemini sees only the table of contents. Its first CBZ position serves as
    // an anchor; all later positions come from the actual image widths.
    const anchor = chapters.find((chapter) => chapter.startPage !== null && chapter.printedPage !== null);
    if (!anchor || !orderedPages.every((page) => aspectRatio(page) !== null)) return chapters;

    const anchorIndex = orderedPages.findIndex((page) => page.index === anchor.startPage - 1);
    if (anchorIndex < 0) return chapters;

    const printedToCbz = new Map();
    let printedPage = anchor.printedPage;
    for (let index = anchorIndex; index < orderedPages.length; index += 1) {
        const page = orderedPages[index];
        const span = page.isDoublePage ? 2 : 1;
        for (let offset = 0; offset < span; offset += 1) {
            printedToCbz.set(printedPage + offset, page.index + 1);
        }
        printedPage += span;
    }

    printedPage = anchor.printedPage;
    for (let index = anchorIndex - 1; index >= 0; index -= 1) {
        const page = orderedPages[index];
        const span = page.isDoublePage ? 2 : 1;
        printedPage -= span;
        for (let offset = 0; offset < span; offset += 1) {
            printedToCbz.set(printedPage + offset, page.index + 1);
        }
    }

    return chapters.map((chapter) => ({
        ...chapter,
        startPage: printedToCbz.get(chapter.printedPage) ?? chapter.startPage,
    }));
}

/**
 * Build the import proposal from the page-type model and the chapter list
 * extracted from the volume summary by Gemini. Covers are deliberately not a
 * chapter-boundary signal: only Gemini's validated summary positions are.
 */
export function buildPageTypeImportPlan(pages, summaryChapters = []) {
    const orderedPages = markDoublePageSpreads(pages).sort((left, right) => left.index - right.index);
    const autoDeletedPages = orderedPages.filter(
        (page) => probabilityFor(page, 'annexe') > ANNEXE_AUTO_DELETE_THRESHOLD,
    );
    const summaryCandidates = orderedPages
        .map((page) => ({ page, confidence: probabilityFor(page, 'summary') }))
        .filter((candidate) => candidate.confidence >= SUMMARY_CANDIDATE_THRESHOLD)
        .sort((left, right) => right.confidence - left.confidence || left.page.index - right.page.index);
    const summaryPage = summaryCandidates[0]?.page || null;

    const excludedIds = new Set([
        ...autoDeletedPages.map((page) => page.id),
        ...summaryCandidates.map(({ page }) => page.id),
    ]);
    const retainedPages = orderedPages.filter((page) => !excludedIds.has(page.id));

    const proposedChapters = summaryChapters
        .map((chapter) => normalizeChapter(chapter, orderedPages.length))
        .filter(Boolean);
    const normalizedChapters = resolveChapterStarts(orderedPages, proposedChapters)
        .filter((chapter) => chapter.startPage !== null)
        .sort((left, right) => left.startPage - right.startPage || left.number - right.number)
        .filter((chapter, index, all) => index === 0 || chapter.startPage !== all[index - 1].startPage);

    const chapters = normalizedChapters.map((chapter, index) => {
        const startIndex = chapter.startPage - 1;
        const nextChapter = normalizedChapters[index + 1];
        const nextStartIndex = nextChapter ? nextChapter.startPage - 1 : Number.POSITIVE_INFINITY;
        // Prefix pages (front cover, credits) remain with chapter one instead
        // of blocking the import as unassigned pages.
        const pageIds = retainedPages
            .filter((page) => (
                index === 0
                    ? page.index < nextStartIndex
                    : page.index >= startIndex && page.index < nextStartIndex
            ))
            .map((page) => page.id);
        return {
            chapterNumber: chapter.number,
            title: chapter.title,
            startPage: chapter.startPage,
            pageIds,
        };
    }).filter((chapter) => chapter.pageIds.length > 0);

    return {
        autoDeletedPages,
        summaryCandidates,
        summaryPage,
        retainedPages,
        chapters,
    };
}
