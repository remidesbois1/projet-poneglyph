import { describe, expect, it } from 'vitest';

import { getGeminiGenerationConfig, normalizeGeneratedBubbles, parseVolumeSummaryResponse } from './geminiClient';
import { rallyPageBubblesToYolo } from './ocrBboxFusion';

describe('getGeminiGenerationConfig', () => {
    it('leaves Gemini thinking automatic when no level is selected', () => {
        expect(getGeminiGenerationConfig(
            { gemini_thinking_level: 'default' },
            { responseMimeType: 'application/json' }
        )).toEqual({ responseMimeType: 'application/json' });
    });

    it('maps none to a zero token budget', () => {
        expect(getGeminiGenerationConfig({ gemini_thinking_level: 'none' })).toEqual({
            thinkingConfig: { thinkingBudget: 0 },
        });
    });

    it('maps named levels to the Gemini enum value', () => {
        expect(getGeminiGenerationConfig({ gemini_thinking_level: 'medium' })).toEqual({
            thinkingConfig: { thinkingLevel: 'MEDIUM' },
        });
    });
});

describe('parseVolumeSummaryResponse', () => {
    it('accepts printed page numbers with a CBZ anchor only on the first chapter', () => {
        const response = JSON.stringify({
            chapters: [
                { number: 45, title: 'Premier', printed_page: 3, start_page: 5 },
                { number: 46, title: 'Deuxième', printed_page: 25, start_page: null },
                { number: 47, title: 'Troisième', printed_page: 45, start_page: 200 },
            ],
        });

        expect(parseVolumeSummaryResponse(response, 60).chapters).toEqual([
            { number: 45, title: 'Premier', printedPage: 3, startPage: 5 },
            { number: 46, title: 'Deuxième', printedPage: 25, startPage: null },
            { number: 47, title: 'Troisième', printedPage: 45, startPage: null },
        ]);
    });

    it('keeps accepting older summaries with CBZ positions but no printed numbers', () => {
        const response = JSON.stringify({
            chapters: [
                { number: 1, title: 'Un', start_page: 5 },
                { number: 2, title: 'Deux', start_page: 25 },
            ],
        });

        expect(parseVolumeSummaryResponse(response, 60).chapters.map((chapter) => chapter.startPage))
            .toEqual([5, 25]);
    });

    it('rejects printed pages given in reverse order', () => {
        const response = JSON.stringify({
            chapters: [
                { number: 1, printed_page: 15, start_page: 3 },
                { number: 2, printed_page: 12, start_page: null },
            ],
        });

        expect(() => parseVolumeSummaryResponse(response, 40)).toThrow('Sommaire Gemini invalide');
    });
});

describe('normalizeGeneratedBubbles', () => {
    it('passes bbox arrays through untouched; the rally resolves the orientation', () => {
        const raw = { bubbles: [{ content: 'Regardez qui vient !', bbox: [121, 781, 149, 888] }] };
        expect(normalizeGeneratedBubbles(raw)).toEqual([
            { content: 'Regardez qui vient !', bbox: [121, 781, 149, 888] },
        ]);
    });

    it('supports object coordinate formats and filters incomplete boxes', () => {
        const raw = [
            { text: 'Bulle objet', box: { x: 100, y: 200, w: 50, h: 60 } },
            { text: 'Invalide', box: [100, 200] },
            { text: '', box: [10, 20, 30, 40] },
        ];

        expect(normalizeGeneratedBubbles(raw)).toEqual([
            { content: 'Bulle objet', bbox: [100, 200, 150, 260] },
        ]);
    });
});

describe('rallyPageBubblesToYolo with Gemini bbox orientation', () => {
    // Real YOLO detections for page 1817 (958x1500): first bubble top-right, last bubble bottom-left.
    const yoloBoxes = [
        { x: 740, y: 170, w: 121, h: 65 },
        { x: 86, y: 1124, w: 194, h: 146 },
    ];
    const firstText = 'Regardez qui vient !';
    const lastText = 'Oui !! Des rescapés ont sûrement réussi à prendre contact avec le gouvernement !!';
    const byYoloOrder = matches => matches.map(m => ({ yoloIndex: m.yoloIndex, text: m.text }));
    const expected = [
        { yoloIndex: 0, text: firstText },
        { yoloIndex: 1, text: lastText },
    ];

    it('rallies a response in Gemini native [ymin, xmin, ymax, xmax] order (captured response)', () => {
        const bubbles = normalizeGeneratedBubbles({
            bubbles: [
                { content: firstText, bbox: [121, 781, 149, 888] },
                { content: lastText, bbox: [757, 105, 836, 279] },
            ],
        });
        expect(byYoloOrder(rallyPageBubblesToYolo(bubbles, yoloBoxes, 958, 1500).matches)).toEqual(expected);
    });

    it('rallies a response in prompt [x1, y1, x2, y2] order (live response)', () => {
        const bubbles = normalizeGeneratedBubbles({
            bubbles: [
                { content: firstText, bbox: [782, 121, 885, 149] },
                { content: lastText, bbox: [105, 757, 279, 836] },
            ],
        });
        expect(byYoloOrder(rallyPageBubblesToYolo(bubbles, yoloBoxes, 958, 1500).matches)).toEqual(expected);
    });
});
