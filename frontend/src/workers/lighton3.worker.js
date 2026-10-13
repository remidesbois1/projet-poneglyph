import { Lighton3WebGPU } from '../lib/lighton3Webgpu';

const engine = new Lighton3WebGPU();
let queue = Promise.resolve();
const controllers = new Map();
self.addEventListener('message', ({ data }) => {
    if (data.type === 'cancel') { controllers.get(data.requestId)?.abort(); return; }
    const controller = data.type === 'run' ? new AbortController() : null;
    if (controller) controllers.set(data.requestId, controller);
    queue = queue.catch(() => undefined).then(async () => {
        try {
            if (data.type === 'init') {
                await engine.load(info => self.postMessage({ status: 'download_progress', ...info }));
                self.postMessage({ status: 'ready', modelKey: 'lighton3Webgpu', gpu: engine.gpuName });
            } else if (data.type === 'run') {
                const result = await engine.recognize(data.imageBitmap || data.imageBlob,
                    text => self.postMessage({ status: 'stream', text, requestId: data.requestId }),
                    { signal: controller.signal });
                self.postMessage({ status: 'complete', requestId: data.requestId, ...result });
            } else if (data.type === 'dispose') { await engine.dispose(); self.postMessage({ status: 'disposed' }); }
        } catch (error) {
            self.postMessage({ status: 'error', requestId: data.requestId,
                error: error?.message || String(error), detail: error?.stack || String(error),
                cancelled: controller?.signal.aborted || false });
        } finally { data.imageBitmap?.close(); if (controller) controllers.delete(data.requestId); }
    });
});
