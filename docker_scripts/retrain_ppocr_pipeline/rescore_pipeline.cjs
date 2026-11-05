/** Exact postprocess ablation: reuse immutable raw CNN outputs, never images/labels. */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const R = require('../audit_ppocr_pipeline/runtime.cjs');
async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  for (const k of ['--run', '--output']) if (!args.includes(k)) throw new Error(`${k} required`);
  const root = path.resolve(value('--run')), file = path.join(root, 'pages.jsonl');
  const rows = fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
  const api = R.ocr(); await api.init();
  for (const row of rows) for (const b of row.bubbles) b.text = api.post(b.raw || '').post;
  const bubbles = rows.flatMap(r => r.bubbles);
  const summary = {pages: rows.length, bubbles: bubbles.length, matched: bubbles.filter(b => b.detected).length,
    unmatched_predictions: rows.reduce((n, r) => n+r.unmatched_predictions, 0), raw: R.score(bubbles, 'raw'), final: R.score(bubbles),
    perfect_annotated_pages: rows.filter(r => R.score(r.bubbles).exact_count === r.bubbles.length).length,
    postprocess: 'PP-OCR lexicon/spacing/case only, selected on validation',
    raw_predictions_sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),
    original_protocol: JSON.parse(fs.readFileSync(path.join(root, 'protocol.json'))),
    limitation: 'Previously inspected historical holdout; GT-conditioned scores, unverified annotation completeness.'};
  fs.writeFileSync(path.resolve(value('--output')), JSON.stringify(summary, null, 2)); console.log(JSON.stringify(summary));
}
main().catch(e => {console.error(e); process.exitCode = 1;});
