import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useLighton3PageOcr } from './useLighton3PageOcr';

const context = vi.hoisted(() => ({ value: {} }));
vi.mock('@/context/WorkerContext', () => ({ useWorker: () => context.value }));
vi.mock('sonner', () => ({ toast: { info: vi.fn(), warning: vi.fn(), error: vi.fn() } }));

describe('LightOn full-page request ownership', () => {
    let worker, runOcr, loadModel, imageRef, onApply;
    beforeEach(() => {
        worker = new EventTarget(); worker.postMessage = vi.fn();
        runOcr = vi.fn(async () => undefined); loadModel = vi.fn(); onApply = vi.fn();
        imageRef = { current: { naturalWidth: 2000, naturalHeight: 3000, src: 'page-a.png' } };
        vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ close: vi.fn() })));
        context.value = { workers: { lighton3Webgpu: worker }, modelStates: { lighton3Webgpu: { status: 'ready' } }, runOcr, loadModel };
    });
    const send = data => worker.dispatchEvent(new MessageEvent('message', { data }));
    it('imports generated boxes in original pixels without another detector or text rewriting', async () => {
        const { result } = renderHook(() => useLighton3PageOcr({ imageRef, contextKey: 'a', onApply }));
        let job;
        act(() => { job = result.current.handleLighton3OneShot(); });
        await waitFor(() => expect(runOcr).toHaveBeenCalledOnce());
        const requestId = runOcr.mock.calls[0][1];
        await act(async () => {
            send({ status: 'complete', requestId, bubbles: [{ bbox: [100,200,300,400], content: 'Étonnant !', order: 1 }] });
            await job;
        });
        expect(onApply.mock.calls[0][0]).toEqual([{ x: 200, y: 600, w: 400, h: 600, texte_propose: 'Étonnant !', order: 1 }]);
        expect(result.current.isLighton3Loading).toBe(false);
    });
    it('cancels an old page and ignores its late result', async () => {
        const { result, rerender } = renderHook(({ key }) => useLighton3PageOcr({ imageRef, contextKey: key, onApply }), { initialProps: { key: 'a' } });
        let job;
        act(() => { job = result.current.handleLighton3OneShot(); });
        await waitFor(() => expect(runOcr).toHaveBeenCalledOnce());
        const requestId = runOcr.mock.calls[0][1];
        act(() => rerender({ key: 'b' }));
        await act(async () => { await job; });
        expect(worker.postMessage).toHaveBeenCalledWith({ type: 'cancel', requestId });
        act(() => send({ status: 'complete', requestId, bubbles: [{ bbox: [0,0,100,100], content: 'Stale', order: 1 }] }));
        expect(onApply).not.toHaveBeenCalled();
    });
    it('prepares the full-page engine before starting inference', async () => {
        context.value.modelStates = {};
        const { result } = renderHook(() => useLighton3PageOcr({ imageRef, contextKey: 'a', onApply }));
        await act(() => result.current.handleLighton3OneShot());
        expect(loadModel).toHaveBeenCalledWith('lighton3Webgpu');
        expect(runOcr).not.toHaveBeenCalled();
    });
});
