import { pipeline, env, RawImage } from '@huggingface/transformers';

env.allowLocalModels = false;
env.useBrowserCache = true;

let extractor = null;

const MODEL_ID = 'onnx-community/siglip2-base-patch16-naflex-ONNX';

self.addEventListener('message', async (event) => {
    const { type, imageBlob } = event.data;

    if (type === 'init') {
        try {
            if (extractor) {
                self.postMessage({ status: 'ready' });
                return;
            }

            const progressCallback = (data) => {
                if (data.status === 'progress' || data.status === 'download' || data.status === 'initiate') {
                    self.postMessage({ status: 'download_progress', file: data.file, progress: data.progress, name: data.name });
                }
            };

            console.log("[SigLIP Worker] Chargement de SigLIP (fp16, WebGPU)...");

            extractor = await pipeline('image-feature-extraction', MODEL_ID, {
                device: 'webgpu',
                dtype: 'fp16',
                progress_callback: progressCallback
            });

            console.log("[SigLIP Worker] Modèle chargé et prêt.");
            self.postMessage({ status: 'ready' });
        } catch (err) {
            console.error("[SigLIP Worker Init Error]", err);
            const errorMsg = err instanceof Error ? err.message : String(err);
            self.postMessage({ status: 'error', error: `Initialisation impossible : ${errorMsg}` });
        }
    }

    if (type === 'run' && imageBlob) {
        const { requestId } = event.data;
        if (!extractor) {
            self.postMessage({ status: 'error', error: 'Modèle non chargé.', requestId });
            return;
        }

        try {
            const image = await RawImage.fromBlob(imageBlob);


            const output = await extractor(image);

            let embedding = Array.from(output.data);

            console.log(`[SigLIP Worker] Embedding généré, dimensions: ${output.dims.join('x')} (${embedding.length} elements)`);

            if (output.dims.length === 3) {
                console.log("[SigLIP Worker] Note: Le tenseur a 3 dimensions. Pooling pourrait être nécessaire.");
            }

            self.postMessage({ status: 'complete', embedding, requestId });

        } catch (err) {
            console.error("[SigLIP Worker Run Error]", err);
            const errorMsg = err instanceof Error ? err.message : String(err);
            self.postMessage({ status: 'error', error: `Erreur SigLIP : ${errorMsg}`, requestId });
        }
    }
});
