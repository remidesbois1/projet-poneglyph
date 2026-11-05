/** Config A (YOLO lines + PP-OCRv6) through the production worker functions, on annotated bubble crops.
 * Also writes every bubble crop and its production YOLO line boxes for the Indra configs and fine-tuning.
 * Usage: node run_ppocr.cjs --split val|test|train [--limit N]
 * train: crops + YOLO boxes only (no recognition, not timed). */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const R = path.resolve(__dirname, '../retrain_ppocr_pipeline');
const {createRuntime, createCanvas, loadImage} = require(path.join(R, 'worker_runtime.cjs'));
const DATASET = path.join(R, 'outputs/dataset-v5');
const MODELS = path.join(R, 'outputs/models-published');
const OUT = path.join(__dirname, 'outputs');
const sha = x => crypto.createHash('sha256').update(x).digest('hex');

async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  const split = value('--split');
  if (!['train', 'val', 'test'].includes(split)) throw new Error('--split train|val|test required');
  const limit = args.includes('--limit') ? Number(value('--limit')) : Infinity;
  const recognize = split !== 'train';
  const plan = JSON.parse(fs.readFileSync(path.join(DATASET, 'plan.json')));
  const ocr = createRuntime(MODELS, {recognizer: recognize ? path.join(MODELS, 'ppocrv6_bubble_line_rec_webgpu.onnx') : null, threads: 4});
  const excluded = new Set(JSON.parse(fs.readFileSync(path.join(R, 'outputs/training-pair-exclusions.json'))).map(e => e.bubble_id));
  await ocr.init();
  const cropDir = path.join(OUT, 'crops', split);
  fs.mkdirSync(cropDir, {recursive: true});
  const file = path.join(OUT, `ppocr_${split}.jsonl`);
  const protocol = {split, models: Object.fromEntries(['browser_manifest.json', 'bubble_line_detector_yolo26n.onnx', 'ppocrv6_bubble_line_rec_webgpu.onnx']
      .map(n => [n, sha(fs.readFileSync(path.join(MODELS, n)))])),
    plan_sha256: sha(fs.readFileSync(path.join(DATASET, 'plan.json'))),
    worker_sha256: sha(fs.readFileSync(path.join(__dirname, '../../frontend/src/workers/ocr.worker.js'))),
    runtime: 'onnxruntime-node CPU, intra-op 4 threads, sequential, one bubble at a time, warm sessions'};
  fs.writeFileSync(path.join(OUT, `ppocr_${split}.protocol.json`), JSON.stringify(protocol, null, 2));
  const rows = [];
  let warm = !recognize;
  const pages = plan.pages.filter(p => p.split === split);
  for (const page of pages) {
    if (rows.length >= limit) break;
    const image = await loadImage(path.join(DATASET, 'pages', `${page.id}.png`));
    for (const b of page.bubbles) {
      if (rows.length >= limit) break;
      // Same crop as evaluate_pipeline.cjs --gt-crops (integer canvas copy of the annotated box).
      const w = Math.max(1, Math.round(b.w)), h = Math.max(1, Math.round(b.h));
      const crop = createCanvas(w, h);
      crop.getContext('2d').drawImage(image, Math.round(b.x), Math.round(b.y), w, h, 0, 0, w, h);
      const cropRel = `crops/${split}/${b.id}.png`;
      fs.writeFileSync(path.join(OUT, cropRel), crop.toBuffer('image/png'));
      if (!warm) { const t = ocr.tensor(ocr.stitch(crop, await ocr.detect(crop))); await ocr.recognize(t); warm = true; }
      const t0 = performance.now();
      const boxes = await ocr.detect(crop);
      const t1 = performance.now();
      const row = {bubble_id: b.id, page_id: page.id, split, reference: b.text ?? '', crop: cropRel, width: w, height: h,
        train_excluded: excluded.has(b.id), yolo_boxes: boxes.map(({x1, y1, x2, y2, conf}) => ({x1, y1, x2, y2, conf}))};
      if (recognize) {
        const tensor = ocr.tensor(ocr.stitch(crop, boxes));
        const t2 = performance.now();
        const {raw, text} = await ocr.recognize(tensor);
        const t3 = performance.now();
        Object.assign(row, {raw, post: text, ms: {detect: t1 - t0, stitch_tensor: t2 - t1, recognize: t3 - t2, total: t3 - t0}});
      }
      rows.push(row);
    }
    if (rows.length % 200 < page.bubbles.length) console.log(JSON.stringify({split, bubbles: rows.length}));
  }
  fs.writeFileSync(file, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  console.log(JSON.stringify({split, bubbles: rows.length, file}));
}
main().catch(e => { console.error(e); process.exitCode = 1; });
