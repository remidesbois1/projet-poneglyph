import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useDeepSeekPageOcr } from './useDeepSeekPageOcr';
import { generateDeepSeekOneShotBubbles } from '@/lib/deepseekClient';
import { cropImage } from '@/lib/utils';
import { toast } from 'sonner';

vi.mock('@/lib/deepseekClient', () => ({ generateDeepSeekOneShotBubbles: vi.fn() }));
vi.mock('@/lib/utils', () => ({ cropImage: vi.fn(async () => new Blob(['image'])) }));
vi.mock('sonner', () => ({ toast: { info: vi.fn(), error: vi.fn() } }));
const data = [{ content: 'Bonjour', bbox: [100,200,400,600] }];
function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
}
function setup(overrides = {}) {
    const props = {
        imageRef: { current: { src: 'blob:first-page', naturalWidth: 1000, naturalHeight: 1500 } },
        contextKey: 'page-1', onApply: vi.fn(), onConfigure: vi.fn(), ...overrides,
    };
    return { ...renderHook(input => useDeepSeekPageOcr(input), { initialProps: props }), props };
}
beforeEach(() => {
    localStorage.clear();
    localStorage.setItem('deepseek_api_key', 'sk-own-key');
    vi.clearAllMocks();
    generateDeepSeekOneShotBubbles.mockResolvedValue({ data });
});

describe('DeepSeek page lifecycle', () => {
    it('requires a key without launching a paid call or applying annotations', async () => {
        localStorage.removeItem('deepseek_api_key');
        const { result, props } = setup();
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(props.onConfigure).toHaveBeenCalledOnce();
        expect(generateDeepSeekOneShotBubbles).not.toHaveBeenCalled();
        expect(props.onApply).not.toHaveBeenCalled();
        expect(result.current.isDeepSeekLoading).toBe(false);
    });

    it.each([{ busy: true }, { canRun: false }])('respects busy and permission guards (%j)', async overrides => {
        const { result } = setup(overrides);
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(generateDeepSeekOneShotBubbles).not.toHaveBeenCalled();
    });

    it('converts normalized boxes to pixels and preserves OCR when YOLO is unavailable', async () => {
        const { result, props } = setup({ detectionStatus: 'ready', detectBubbles: vi.fn(async () => { throw new Error('offline'); }) });
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(generateDeepSeekOneShotBubbles).toHaveBeenCalledWith(props.imageRef.current, 'sk-own-key', { signal: expect.any(AbortSignal) });
        expect(props.onApply.mock.calls[0][0][0]).toMatchObject({ content: 'Bonjour', x: 100, y: 300, w: 300, h: 600, ralliedWithYolo: false });
        expect(cropImage).toHaveBeenCalledOnce();
        expect(result.current.isDeepSeekLoading).toBe(false);
    });

    it('uses loaded detector coordinates only when a matching box exists', async () => {
        const { result, props } = setup({ detectionStatus: 'ready', detectBubbles: vi.fn(async () => [{ x: 90, y: 290, w: 320, h: 620 }]) });
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(props.onApply.mock.calls[0][0][0]).toMatchObject({ x: 90, y: 290, w: 320, h: 620, ralliedWithYolo: true });
    });

    it('prevents double-clicking from issuing duplicate billed requests', async () => {
        const pending = deferred();
        generateDeepSeekOneShotBubbles.mockReturnValue(pending.promise);
        const { result, props } = setup();
        let job;
        act(() => { job = result.current.handleDeepSeekOneShot(); });
        expect(result.current.isDeepSeekLoading).toBe(true);
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(generateDeepSeekOneShotBubbles).toHaveBeenCalledOnce();
        await act(async () => { pending.resolve({ data }); await job; });
        expect(props.onApply).toHaveBeenCalledOnce();
        expect(result.current.isDeepSeekLoading).toBe(false);
    });

    it('does not cancel an active request when a transient UI guard changes', async () => {
        const pending = deferred();
        generateDeepSeekOneShotBubbles.mockReturnValueOnce(pending.promise);
        const { result, rerender, props } = setup({ canRun: true });
        let job;
        act(() => { job = result.current.handleDeepSeekOneShot(); });
        const signal = generateDeepSeekOneShotBubbles.mock.calls[0][2].signal;

        rerender({ ...props, canRun: false });

        expect(signal.aborted).toBe(false);
        expect(result.current.isDeepSeekLoading).toBe(true);
        await act(async () => { pending.resolve({ data }); await job; });
        expect(props.onApply).toHaveBeenCalledOnce();
        expect(result.current.isDeepSeekLoading).toBe(false);
    });

    it('aborts on navigation and never applies the previous image to the new page', async () => {
        const pending = deferred();
        generateDeepSeekOneShotBubbles.mockReturnValueOnce(pending.promise);
        const { result, rerender, props } = setup();
        let job;
        act(() => { job = result.current.handleDeepSeekOneShot(); });
        const signal = generateDeepSeekOneShotBubbles.mock.calls[0][2].signal;
        rerender({ ...props, contextKey: 'page-2' });
        expect(signal.aborted).toBe(true);
        expect(result.current.isDeepSeekLoading).toBe(false);
        await act(async () => { pending.resolve({ data }); await job; });
        expect(props.onApply).not.toHaveBeenCalled();
        expect(toast.error).not.toHaveBeenCalled();
    });

    it('does not overwrite annotations when the image source changes during extraction', async () => {
        const pending = deferred();
        generateDeepSeekOneShotBubbles.mockReturnValueOnce(pending.promise);
        const { result, props } = setup();
        let job;
        act(() => { job = result.current.handleDeepSeekOneShot(); });
        props.imageRef.current.src = 'blob:another-image';
        await act(async () => { pending.resolve({ data }); await job; });
        expect(props.onApply).not.toHaveBeenCalled();
    });

    it('does not import an empty extraction or retry provider failures', async () => {
        generateDeepSeekOneShotBubbles.mockResolvedValueOnce({ data: [] }).mockRejectedValueOnce(new Error('Solde insuffisant'));
        const { result, props } = setup();
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(props.onApply).not.toHaveBeenCalled();
        await act(async () => result.current.handleDeepSeekOneShot());
        expect(toast.error).toHaveBeenCalledWith('Solde insuffisant');
        expect(generateDeepSeekOneShotBubbles).toHaveBeenCalledTimes(2);
    });
});
