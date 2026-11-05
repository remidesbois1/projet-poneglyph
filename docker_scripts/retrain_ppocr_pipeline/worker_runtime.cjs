/** Load the actual production OCR functions; never reimplement crop/stitch/resize. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ort = require('onnxruntime-node');
const canvasModule = require.resolve('@napi-rs/canvas', {
  paths: [__dirname, path.resolve(__dirname, '../audit_ppocr_pipeline/cache/runtime')],
});
const { createCanvas, loadImage } = require(canvasModule);
const ROOT = path.resolve(__dirname, '../..');

function createRuntime(models, { recognizer, threads = 4 } = {}) {
  const options = { executionProviders: ['cpu'], intraOpNumThreads: threads, interOpNumThreads: 1, logSeverityLevel: 3 };
  const context = vm.createContext({
    ort: { Tensor: ort.Tensor, InferenceSession: ort.InferenceSession, env: { wasm: {}, webgpu: {} } },
    OffscreenCanvas: function (w, h) { return createCanvas(w, h); },
    self: { location: { origin: 'http://localhost' }, navigator: { hardwareConcurrency: threads }, addEventListener() {}, postMessage() {} },
    modelRegistry: JSON.parse(fs.readFileSync(path.join(ROOT, 'packages/shared/src/model-registry.json'))),
    console, URL, performance, Float32Array, Uint8Array, Map, Set, Math, setTimeout, clearTimeout, fetch,
    _manifest: JSON.parse(fs.readFileSync(path.join(models, 'browser_manifest.json'))),
    _detector: path.join(models, 'bubble_line_detector_yolo26n.onnx'), _recognizer: recognizer, _options: options,
  });
  const worker = fs.readFileSync(path.join(ROOT, 'frontend/src/workers/ocr.worker.js'), 'utf8').replace(/^import .*;\r?\n/gm, '');
  const utils = fs.readFileSync(path.join(ROOT, 'frontend/src/lib/ocr-utils.js'), 'utf8').replace(/^export /gm, '');
  vm.runInContext(utils + '\n' + worker + `
    api = {
      async init() {
        ppocrManifest = _manifest;
        ppocrLineDetectorSession = await ort.InferenceSession.create(_detector, _options);
        if (_recognizer) ppocrRecSession = await ort.InferenceSession.create(_recognizer, _options);
      },
      detect: detectLineBoxes, stitch: stitchLineBoxes, tensor: preprocessPpocrCanvas,
      async recognize(tensor) {
        const result = await ppocrRecSession.run({[ppocrRecSession.inputNames[0]]: tensor});
        const raw = decodePpocrCtc(result[ppocrRecSession.outputNames[0]].data);
        return { raw, text: postprocessPpocrText(raw) };
      },
      post: postprocessPpocrText,
      constants: { lineConfidence: PPOCR_LINE_CONF, pad: PPOCR_LINE_PAD, gap: PPOCR_LINE_GAP }
    };`, context, { filename: 'production-ocr-worker.js' });
  return context.api;
}

function tensorToPng(tensor) {
  const [, channels, height, width] = tensor.dims;
  if (channels !== 3) throw new Error('Expected BGR tensor');
  const pixels = width * height;
  const canvas = createCanvas(width, height), ctx = canvas.getContext('2d');
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < pixels; i++) {
    image.data[4 * i] = Math.round((tensor.data[2 * pixels + i] + 1) * 127.5);
    image.data[4 * i + 1] = Math.round((tensor.data[pixels + i] + 1) * 127.5);
    image.data[4 * i + 2] = Math.round((tensor.data[i] + 1) * 127.5);
    image.data[4 * i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toBuffer('image/png');
}

function createDetectionRuntime(models, { threads = 4 } = {}) {
  const context = vm.createContext({
    ort: { Tensor: ort.Tensor, InferenceSession: ort.InferenceSession, env: { wasm: {}, webgpu: {} } },
    OffscreenCanvas: function (w, h) { return createCanvas(w, h); },
    self: { location: { origin: 'http://localhost' }, navigator: { hardwareConcurrency: threads }, addEventListener() {}, postMessage() {} },
    modelRegistry: JSON.parse(fs.readFileSync(path.join(ROOT, 'packages/shared/src/model-registry.json'))),
    console, URL, performance, Float32Array, Uint8Array, Map, Set, Math, setTimeout, clearTimeout, fetch,
    _models: models.replaceAll('\\', '/'),
    _options: { executionProviders: ['cpu'], intraOpNumThreads: threads, interOpNumThreads: 1, logSeverityLevel: 3 },
  });
  const source = fs.readFileSync(path.join(ROOT, 'frontend/src/workers/detection.worker.js'), 'utf8')
    .replace(/^import .*;\r?\n/gm, '').replace(/const BUBBLE_SCORE_THRESHOLD =/g, 'let BUBBLE_SCORE_THRESHOLD =');
  vm.runInContext(source + `
    api = {
      async init() {
        bubbleSession = await ort.InferenceSession.create(_models + '/bubble_detector.onnx', _options);
        panelSession = await ort.InferenceSession.create(_models + '/panel_detector.onnx', _options);
        orderingSession = await ort.InferenceSession.create(_models + '/ordering.onnx', _options);
      },
      async detect(image, threshold = .45) {
        BUBBLE_SCORE_THRESHOLD = threshold;
        const {height, width} = getImageInputSize(bubbleSession, 800, 800);
        const p = preprocessBubble(image, height, width);
        const result = await bubbleSession.run({[bubbleSession.inputNames[0]]: p.inputTensor});
        const boxes = simplifyPostProcess(result[bubbleSession.outputNames[0]].data, p.scale, p.padX, p.padY);
        const panels = await rankPanels(await detectPanels(image), image);
        let ordered = boxes;
        if (boxes.length > 1 && panels.length) ordered = (await sortBubblesWithReadingOrderDetails(boxes, panels, assignBubblesToPanelsDetailed(boxes, panels, image), image)).boxes;
        else if (boxes.length > 1) ordered = mangaOrderSort(boxes);
        return {boxes, ordered, panels};
      }
    };`, context, { filename: 'production-detection-worker.js' });
  return context.api;
}
module.exports = { createRuntime, createDetectionRuntime, tensorToPng, loadImage, createCanvas, ROOT };
