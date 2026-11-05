/** Threshold selection uses validation only; test is evaluated after freezing it. */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const {createDetectionRuntime, loadImage} = require('./worker_runtime.cjs');
const {matchBoxes} = require('../audit_ppocr_pipeline/runtime.cjs');
async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  for (const k of ['--models', '--dataset', '--output']) if (!args.includes(k)) throw new Error(`${k} required`);
  const models = path.resolve(value('--models')), dataset = path.resolve(value('--dataset')), output = path.resolve(value('--output'));
  const plan = JSON.parse(fs.readFileSync(path.join(dataset, 'plan.json')));
  const thresholds = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6];
  const runtime = createDetectionRuntime(models); await runtime.init();
  const rows = [];
  for (const page of plan.pages.filter(p => p.split === 'val')) {
    const image = await loadImage(path.join(dataset, 'pages', `${page.id}.png`));
    const {boxes} = await runtime.detect(image, .05);
    const scores = thresholds.map(threshold => {
      const selected = boxes.filter(b => b.score >= threshold), matches = matchBoxes(page.bubbles, selected);
      return {threshold, gt: page.bubbles.length, detected: selected.length, matched: matches.length, missed: page.bubbles.length - matches.length, unmatched: selected.length - matches.length};
    });
    rows.push({id: page.id, boxes, scores});
  }
  const summary = thresholds.map(threshold => {
    const subset = rows.map(r => r.scores.find(s => s.threshold === threshold));
    return {threshold, ...Object.fromEntries(['gt', 'detected', 'matched', 'missed', 'unmatched'].map(k => [k, subset.reduce((n, r) => n + r[k], 0)]))};
  });
  const result = {split: 'val', detector_sha256: crypto.createHash('sha256').update(fs.readFileSync(path.join(models, 'bubble_detector.onnx'))).digest('hex'),
    note: 'GT-conditioned one-to-one IoU>=0.5. Unmatched detections are not verified false positives.', summary, pages: rows};
  fs.mkdirSync(path.dirname(output), {recursive: true});
  fs.writeFileSync(output, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(summary));
}
main().catch(e => {console.error(e); process.exitCode = 1;});
