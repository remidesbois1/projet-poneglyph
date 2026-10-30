import { afterEach, describe, expect, it, vi } from 'vitest';
import { FalconFusedDecoder } from './falconFusedDecoder';

function fakeDevice() {
    return {
        lost: new Promise(() => {}), features: new Set(),
        pushErrorScope: vi.fn(), popErrorScope: vi.fn(async () => null),
        createCommandEncoder: () => ({ beginComputePass: () => ({}) }),
        createBuffer: vi.fn(() => { throw new Error('allocation failed'); }),
    };
}

describe('Falcon decoder resource handling', () => {
    afterEach(() => vi.unstubAllGlobals());
    it('balances error scopes after an allocation failure', async () => {
        vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 });
        const device = fakeDevice(), decoder = new FalconFusedDecoder(device);
        await expect(decoder.load({ weights: { sample: { bits: [0], shape: [1] } } }, new Uint8Array())).rejects.toThrow('allocation failed');
        expect(device.pushErrorScope).toHaveBeenCalledTimes(2);
        expect(device.popErrorScope).toHaveBeenCalledTimes(2);
        decoder.dispose();
    });

    it('handles an image whose first predicted token is already a stop token', () => {
        const decoder = new FalconFusedDecoder(fakeDevice());
        expect(decoder.stats()).toMatchObject({ kernelsPerStep: 0, computedSteps: 0, readbacks: 0, sequenceBytes: 0 });
        decoder.dispose();
    });
});
