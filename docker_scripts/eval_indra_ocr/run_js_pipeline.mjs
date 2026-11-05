/** Run the browser implementation (frontend/src/lib/indraOcr.js) on bubble crops with onnxruntime-node,
 * to check it reproduces the Python reference. Writes outputs/predictions/<split>/<name>.jsonl for score.cjs.
 * Usage: node run_js_pipeline.mjs --split val|test [--release outputs/release] [--name JS_indra_release] */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { rgbaToGray, runIndraBubble } from '../../frontend/src/lib/indraOcr.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');
const { createCanvas, loadImage } = require(require.resolve('@napi-rs/canvas', { paths: [path.join(here, '../audit_ppocr_pipeline/cache/runtime')] }));

const args = process.argv.slice(2), value = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const split = value('--split');
const release = path.resolve(here, value('--release', 'outputs/release'));
const name = value('--name', 'JS_indra_release');
const manifest = JSON.parse(fs.readFileSync(path.join(release, 'onnx/browser_manifest.json'), 'utf8'));
const options = { executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1, logSeverityLevel: 3 };
const detSession = await ort.InferenceSession.create(path.join(release, manifest.files.det.path), options);
const recSession = await ort.InferenceSession.create(path.join(release, manifest.files.rec.path), options);
const rows = fs.readFileSync(path.join(here, `outputs/ppocr_${split}.jsonl`), 'utf8').trim().split('\n').map(JSON.parse);
const out = [JSON.stringify({ config: { implementation: 'frontend/src/lib/indraOcr.js', runtime: 'onnxruntime-node CPU' }, split, n: rows.length })];
for (const r of rows) {
  const image = await loadImage(path.join(here, 'outputs', r.crop));
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const gray = rgbaToGray(ctx.getImageData(0, 0, image.width, image.height).data, image.width, image.height);
  const res = await runIndraBubble(ort, { detSession, recSession, manifest }, gray, image.width, image.height);
  out.push(JSON.stringify({ bubble_id: r.bubble_id, reference: r.reference, raw: res.raw, lines: res.boxes.length, ms: res.timings }));
}
const file = path.join(here, 'outputs/predictions', split, `${name}.jsonl`);
fs.writeFileSync(file, out.join('\n') + '\n');
console.log(file);
