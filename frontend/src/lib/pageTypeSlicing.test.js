import { describe, expect, it } from 'vitest';

import { buildPageTypeImportPlan, markDoublePageSpreads } from './pageTypeSlicing';

function makePage(position, { double = false, width, height = 1200, pageType } = {}) {
    return {
        id: `page-${position}`,
        index: position - 1,
        width: width ?? (double ? 1600 : 800),
        height,
        pageType: pageType ? { label: pageType, confidence: 0.99 } : null,
    };
}

describe('CBZ double-page detection', () => {
    it('uses aspect ratio so a high-resolution portrait is not mistaken for a spread', () => {
        const pages = [
            makePage(1),
            makePage(2, { width: 1600, height: 2400 }),
            makePage(3, { double: true }),
            makePage(4),
            makePage(5, { width: 2400, height: 1800 }),
        ];

        expect(markDoublePageSpreads(pages).map((page) => page.isDoublePage)).toEqual([
            false, false, true, false, true,
        ]);
    });
});

describe('chapter boundaries from a printed table of contents', () => {
    it('counts both halves of every spread even when Gemini overestimates later CBZ indices', () => {
        const pages = Array.from({ length: 22 }, (_, index) => makePage(index + 1, {
            double: index + 1 === 8 || index + 1 === 14,
            pageType: index + 1 === 2 ? 'summary' : index + 1 === 22 ? 'annexe' : 'story_page',
        }));
        const chapters = [
            { number: 45, title: 'Premier', printedPage: 3, startPage: 5 },
            { number: 46, title: 'Deuxième', printedPage: 10, startPage: 12 },
            { number: 47, title: 'Troisième', printedPage: 15, startPage: 17 },
        ];

        const plan = buildPageTypeImportPlan(pages, chapters);

        expect(plan.chapters.map((chapter) => chapter.startPage)).toEqual([5, 11, 15]);
        expect(plan.chapters[0].pageIds).toContain('page-1');
        expect(plan.chapters[0].pageIds).toContain('page-10');
        expect(plan.chapters[1].pageIds).toEqual(['page-11', 'page-12', 'page-13', 'page-14']);
        expect(plan.chapters[2].pageIds[0]).toBe('page-15');
        expect(plan.retainedPages.map((page) => page.id)).not.toContain('page-2');
        expect(plan.retainedPages.map((page) => page.id)).not.toContain('page-22');
        expect(plan.chapters.flatMap((chapter) => chapter.pageIds)).toEqual(
            plan.retainedPages.map((page) => page.id),
        );
    });

    it('maps a printed chapter boundary on the second half of a spread to that one image', () => {
        const pages = Array.from({ length: 16 }, (_, index) => makePage(index + 1, {
            double: index + 1 === 8,
        }));

        const plan = buildPageTypeImportPlan(pages, [
            { number: 1, title: 'Un', printedPage: 1, startPage: 5 },
            { number: 2, title: 'Deux', printedPage: 5, startPage: null },
            { number: 3, title: 'Trois', printedPage: 6, startPage: null },
        ]);

        expect(plan.chapters.map((chapter) => chapter.startPage)).toEqual([5, 8, 9]);
        expect(plan.chapters[1].pageIds).toEqual(['page-8']);
    });

    it('preserves old CBZ positions when printed numbers or image dimensions are unavailable', () => {
        const pages = Array.from({ length: 12 }, (_, index) => makePage(index + 1));
        const legacy = [
            { number: 1, startPage: 3, title: 'Un' },
            { number: 2, startPage: 9, title: 'Deux' },
        ];

        expect(buildPageTypeImportPlan(pages, legacy).chapters.map((chapter) => chapter.startPage))
            .toEqual([3, 9]);

        const withoutSizes = pages.map(({ id, index, pageType }) => ({ id, index, pageType }));
        const withPrinted = legacy.map((chapter, index) => ({
            ...chapter,
            printedPage: index === 0 ? 2 : 15,
        }));
        expect(buildPageTypeImportPlan(withoutSizes, withPrinted).chapters.map((chapter) => chapter.startPage))
            .toEqual([3, 9]);
    });
});
