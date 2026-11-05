/** Apply the frozen production text rules to paired native/browser predictions. */
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const R = require('../audit_ppocr_pipeline/runtime.cjs');
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const read = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const lines = p => fs.readFileSync(p, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  for (const k of ['--results', '--output']) if (!args.includes(k)) throw new Error(`${k} required`);
  const root = path.resolve(value('--results')), output = path.resolve(value('--output'));
  const manifest = read(path.join(root, 'recognizer-onnx/browser_manifest.json'));
  if (JSON.stringify(manifest.postprocess) !== JSON.stringify(R.manifest.postprocess)) throw new Error('Postprocess rules changed; comparisons require explicit isolation');
  const api = R.ocr(); await api.init();
  const report = {normalization: 'Production audit NFKC, straight apostrophes, ellipsis expansion and collapsed whitespace; case and punctuation retained.',
    evaluation_note: 'Historical page-level holdout, already evaluated and inspected. GT crops isolate recognition. Native CUDA FP32 predictions; browser subset tested independently.',
    weights_sha256: sha(path.join(root, 'recognizer-onnx/ppocrv6_bubble_line_rec_webgpu.onnx')), splits: {}};
  const diagnostic = read(path.join(R.ROOT, 'docker_scripts/audit_ppocr_pipeline/visual-labels.json'));
  for (const split of ['val', 'test']) {
    const oldRows = lines(path.join(root, 'baseline-recognizer-eval', split + '_predictions.jsonl'));
    const newRows = lines(path.join(root, 'recognizer-eval', split + '_predictions.jsonl'));
    const expected = new Map(oldRows.map(r => [r.image, r.reference]));
    if (oldRows.length !== newRows.length || newRows.some(r => r.reference !== expected.get(r.image))) throw new Error('Recognition rows/targets changed between models');
    const scoreRows = rows => rows.map(r => ({...r, ...api.post(r.raw), id: Number(path.basename(r.image, '.png'))}));
    const baseline = scoreRows(oldRows), candidate = scoreRows(newRows).map(r => ({...r, legacy_text: r.text, text: r.post}));
    report.splits[split] = {baseline: {raw: R.score(baseline, 'raw'), lexicon_only: R.score(baseline, 'post'), final: R.score(baseline)},
      candidate: {raw: R.score(candidate, 'raw'), legacy_cleanup: R.score(candidate, 'legacy_text'), final: R.score(candidate)}, diagnostic_subset: {}};
    report.postprocess = {selection: 'Validation only: 440 vs 485 character errors on candidate, 515 vs 560 on baseline.', baseline: 'Lexicon then legacy generic cleanup', candidate: 'PP-OCR lexicon/spacing/case rules; preserve quotes and literal I'};
    for (const label of ['ordinary', 'stylized_sfx']) {
      const ids = new Set(diagnostic.labels.filter(r => r.split === split && r.label === label).map(r => r.id));
      report.splits[split].diagnostic_subset[label] = {note: diagnostic.protocol + ' Error-focused exploratory subset, not representative.', baseline: R.score(baseline.filter(r => ids.has(r.id))), candidate: R.score(candidate.filter(r => ids.has(r.id)))};
    }
    if (split === 'val' && args.includes('--browser-baseline') && args.includes('--browser-candidate')) {
      const a = read(path.resolve(value('--browser-baseline'))), b = read(path.resolve(value('--browser-candidate')));
      if (a.status !== 'complete' || b.status !== 'complete' || JSON.stringify(a.protocol.ids) !== JSON.stringify(b.protocol.ids)) throw new Error('Incompatible browser comparison');
      if (b.protocol.model_sha256 !== report.weights_sha256) throw new Error('Browser tested different candidate weights');
      const refs = new Map(newRows.map(r => [Number(path.basename(r.image, '.png')), r.reference]));
      const native = new Map(newRows.map(r => [Number(path.basename(r.image, '.png')), r.raw]));
      report.browser = {split: 'val', provider: b.provider, adapter: b.adapter, samples: b.rows.length,
        input_mode: b.protocol.input_mode, baseline: R.score(a.rows.map(r => ({...r, reference: refs.get(r.id)}))),
        candidate: R.score(b.rows.map(r => ({...r, reference: refs.get(r.id)}))),
        native_candidate_text_matches: b.rows.filter(r => r.raw === native.get(r.id)).length};
    }
  }
  fs.writeFileSync(output, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
}
main().catch(e => {console.error(e); process.exitCode = 1;});
