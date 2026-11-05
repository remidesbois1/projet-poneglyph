/** Reproducible, production-function end-to-end evaluation on unchanged pages. */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const {createRuntime, createDetectionRuntime, createCanvas, loadImage, ROOT} = require('./worker_runtime.cjs');
const {matchBoxes, score} = require('../audit_ppocr_pipeline/runtime.cjs');
const sha = x => crypto.createHash('sha256').update(x).digest('hex');
async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  for (const k of ['--dataset', '--models', '--output']) if (!args.includes(k)) throw new Error(`${k} required`);
  const dataset = path.resolve(value('--dataset')), models = path.resolve(value('--models')), output = path.resolve(value('--output'));
  const split = args.includes('--split') ? value('--split') : 'val';
  if (!['val', 'test'].includes(split)) throw new Error('Only held-out pages may be evaluated');
  const threshold = args.includes('--confidence') ? Number(value('--confidence')) : .45;
  const limit = args.includes('--limit') ? Number(value('--limit')) : Infinity;
  const cropMode = args.includes('--gt-crops');
  const plan = JSON.parse(fs.readFileSync(path.join(dataset, 'plan.json')));
  const protocol = {split, threshold, crop_mode: cropMode, page_limit: Number.isFinite(limit) ? limit : null,
    dataset_plan_sha256: sha(fs.readFileSync(path.join(dataset, 'plan.json'))),
    worker_sha256: sha(fs.readFileSync(path.join(ROOT, 'frontend/src/workers/ocr.worker.js'))),
    detection_worker_sha256: sha(fs.readFileSync(path.join(ROOT, 'frontend/src/workers/detection.worker.js'))),
    runtime_sha256: sha(fs.readFileSync(path.join(__dirname, 'worker_runtime.cjs'))), script_sha256: sha(fs.readFileSync(__filename)),
    artifacts: Object.fromEntries(['browser_manifest.json', 'ppocrv6_bubble_line_rec_webgpu.onnx', 'bubble_line_detector_yolo26n.onnx', 'bubble_detector.onnx', 'panel_detector.onnx', 'ordering.onnx'].map(n => [n, sha(fs.readFileSync(path.join(models, n)))])),
    note: 'Previously inspected historical holdout. Native Skia/ORT CPU execution; browser tested separately. Unmatched predictions reported but not scored as false positives because annotation completeness is unproven.'};
  fs.mkdirSync(output, {recursive: true});
  const protocolPath = path.join(output, 'protocol.json');
  if (fs.existsSync(protocolPath) && JSON.stringify(JSON.parse(fs.readFileSync(protocolPath))) !== JSON.stringify(protocol)) throw new Error('Output belongs to a different protocol or model; choose a new directory');
  fs.writeFileSync(protocolPath, JSON.stringify(protocol, null, 2));
  const ocr = createRuntime(models, {recognizer: path.join(models, 'ppocrv6_bubble_line_rec_webgpu.onnx')});
  await ocr.init();
  const detector = cropMode ? null : createDetectionRuntime(models);
  if (detector) await detector.init();
  const predictionsFile = path.join(output, 'pages.jsonl');
  const rows = fs.existsSync(predictionsFile) ? fs.readFileSync(predictionsFile, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [];
  const done = new Set(rows.map(r => r.id));
  const pages = plan.pages.filter(p => p.split === split).slice(0, limit);
  for (const page of pages) {
    if (done.has(page.id)) continue;
    const image = await loadImage(path.join(dataset, 'pages', `${page.id}.png`));
    const detections = cropMode ? {ordered: page.bubbles} : await detector.detect(image, threshold);
    const matches = matchBoxes(page.bubbles, detections.ordered), matched = new Map(matches.map(m => [m.i, m.j]));
    const predictions = [];
    for (const box of detections.ordered) {
      const w = Math.max(1, Math.round(box.w)), h = Math.max(1, Math.round(box.h));
      const crop = createCanvas(w, h);
      crop.getContext('2d').drawImage(image, Math.round(box.x), Math.round(box.y), w, h, 0, 0, w, h);
      const boxes = await ocr.detect(crop), tensor = ocr.tensor(ocr.stitch(crop, boxes));
      predictions.push({...await ocr.recognize(tensor), box, line_count: boxes.length, width: tensor.dims[3]});
    }
    const bubbles = page.bubbles.map((b, i) => ({id: b.id, reference: b.text, detected: matched.has(i),
      ...(matched.has(i) ? predictions[matched.get(i)] : {raw: '', text: ''})}));
    const row = {id: page.id, predictions, matches, bubbles, unmatched_predictions: predictions.length - matches.length};
    fs.appendFileSync(predictionsFile, JSON.stringify(row) + '\n'); rows.push(row);
    if (rows.length % 10 === 0 || rows.length === pages.length) console.log(JSON.stringify({pages: rows.length, total: pages.length}));
  }
  const bubbles = rows.flatMap(r => r.bubbles);
  const summary = {pages: rows.length, bubbles: bubbles.length, matched: bubbles.filter(b => b.detected).length,
    unmatched_predictions: rows.reduce((n, r) => n + r.unmatched_predictions, 0), raw: score(bubbles, 'raw'), final: score(bubbles),
    perfect_annotated_pages: rows.filter(r => score(r.bubbles).exact_count === r.bubbles.length).length,
    limitation: protocol.note};
  fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary));
}
main().catch(e => {console.error(e); process.exitCode = 1;});
