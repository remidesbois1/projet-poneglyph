import { describe, expect, it } from 'vitest';
import { parseLighton3Grounding, lighton3BubbleToPixels } from './lighton3Grounding';

describe('LightOnOCR 3 native grounding contract', () => {
    it('preserves text, accents, quotes and reading order across inline and multiline blocks', () => {
        const value = parseLighton3Grounding('![text](100,200,300,400)\n"Cafe\u0301 !"\n\n![title+](10,20,30,40) Ça va ?');
        expect(value.invalidBlocks).toBe(0);
        expect(value.bubbles.map(b => b.content)).toEqual(['"Café !"', 'Ça va ?']);
        expect(value.bubbles.map(b => b.order)).toEqual([1, 2]);
    });
    it('penalizes malformed, empty and out-of-range boxes instead of clipping them', () => {
        const value = parseLighton3Grounding('![text](-1,0,10,10) incorrect\n![text](0,0,1001,10) incorrect\n![text](20,0,10,10) incorrect\n![text](0,0,10,10)');
        // The leading malformed marker is also a non-grounding preamble,
        // matching the native Python scorer's invalid-block accounting.
        expect(value.invalidBlocks).toBe(5);
        expect(value.bubbles).toEqual([]);
    });
    it('does not import panel illustrations as textual annotations', () => {
        const value = parseLighton3Grounding('![image](0,0,500,500) Illustration\n![text](100,100,200,200) Bonjour');
        expect(value.bubbles.map(b => b.content)).toEqual(['Bonjour']);
        expect(value.invalidBlocks).toBe(0);
    });
    it('maps coordinates to the original page rather than the resized input', () => {
        const box = lighton3BubbleToPixels({ bbox: [250, 200, 750, 800], content: 'Texte', order: 1 }, 4000, 6000);
        expect(box).toEqual({ x: 1000, y: 1200, w: 2000, h: 3600, texte_propose: 'Texte', order: 1 });
    });
    it('flags unstructured output and leading explanatory text', () => {
        expect(parseLighton3Grounding('Bonjour').invalidBlocks).toBe(1);
        expect(parseLighton3Grounding('Réponse :\n![text](0,0,10,10) Bonjour').invalidBlocks).toBe(1);
    });
});
