import { describe, expect, it } from 'vitest';
import { rallyPageBubblesToYolo } from './ocrBboxFusion';

const BOX_A = { x: 100, y: 100, w: 200, h: 200 };
const BOX_B = { x: 600, y: 600, w: 200, h: 200 };

describe('rallyPageBubblesToYolo', () => {
    it('returns matches in YOLO order, each carrying the text of its overlapping page bubble', () => {
        const { matches, unmatchedYoloIndices } = rallyPageBubblesToYolo([
            { content: 'Deuxième', bbox: [600, 600, 800, 800] },
            { content: 'Premier', bbox: [100, 100, 300, 300] },
        ], [BOX_A, BOX_B], 1000, 1000);

        expect(matches.map(({ yoloIndex, pageIndex, text }) => ({ yoloIndex, pageIndex, text }))).toEqual([
            { yoloIndex: 0, pageIndex: 1, text: 'Premier' },
            { yoloIndex: 1, pageIndex: 0, text: 'Deuxième' },
        ]);
        expect(unmatchedYoloIndices).toEqual([]);
    });

    it('scales normalized page bboxes to the image size before matching', () => {
        const { matches } = rallyPageBubblesToYolo([
            { content: 'Haut', bbox: [100, 100, 300, 300] },
        ], [{ x: 200, y: 50, w: 400, h: 100 }, { x: 0, y: 400, w: 400, h: 100 }], 2000, 500);

        expect(matches).toEqual([expect.objectContaining({ yoloIndex: 0, text: 'Haut' })]);
    });

    it('assigns each YOLO box to its best-overlapping page bubble only and drops the other one', () => {
        const { matches, unmatchedYoloIndices } = rallyPageBubblesToYolo([
            { content: 'Recouvrement partiel', bbox: [150, 150, 350, 350] },
            { content: 'Recouvrement exact', bbox: [100, 100, 300, 300] },
        ], [BOX_A, BOX_B], 1000, 1000);

        expect(matches).toHaveLength(1);
        expect(matches[0]).toEqual(expect.objectContaining({ yoloIndex: 0, pageIndex: 1, text: 'Recouvrement exact' }));
        expect(matches[0].iou).toBeCloseTo(1);
        expect(unmatchedYoloIndices).toEqual([1]);
    });

    it('ignores page bubbles with invalid bboxes', () => {
        const { matches, unmatchedYoloIndices } = rallyPageBubblesToYolo([
            { content: 'Inversée', bbox: [300, 100, 100, 300] },
            { content: 'Plate', bbox: [100, 300, 300, 300] },
            { content: 'Texte', bbox: ['a', 100, 300, 300] },
            { content: 'Courte', bbox: [100, 100, 300] },
            { content: 'Sans bbox' },
        ], [BOX_A], 1000, 1000);

        expect(matches).toEqual([]);
        expect(unmatchedYoloIndices).toEqual([0]);
    });

    it('requires an IoU above 0.05 to rally a page bubble', () => {
        // A 10 px horizontal overlap gives IoU 10/390 ≈ 0.026; 30 px gives 30/370 ≈ 0.081.
        const barely = rallyPageBubblesToYolo([
            { content: 'Effleure', bbox: [290, 100, 490, 300] },
        ], [BOX_A], 1000, 1000);
        const enough = rallyPageBubblesToYolo([
            { content: 'Chevauche', bbox: [270, 100, 470, 300] },
        ], [BOX_A], 1000, 1000);

        expect(barely.matches).toEqual([]);
        expect(barely.unmatchedYoloIndices).toEqual([0]);
        expect(enough.matches).toEqual([expect.objectContaining({ yoloIndex: 0, text: 'Chevauche' })]);
    });
});
