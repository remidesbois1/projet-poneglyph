/** Score raw predictions with the production post-processing and audit metrics.
 * Usage: node score.cjs --split val [--tag name] [--files a.jsonl b.jsonl ...]   (default: every file of outputs/predictions/<split>)
 * The production PP-OCRv6 baseline (outputs/ppocr_<split>.jsonl) is always included as "A_yolo_ppocrv6".
 * "final" = postprocessPpocrText(raw) from the published browser manifest (rules built from train labels only),
 * applied identically to every configuration. */
const fs = require('node:fs'), path = require('node:path');
const R = path.resolve(__dirname, '../retrain_ppocr_pipeline');
const {createRuntime} = require(path.join(R, 'worker_runtime.cjs'));
const {score, normalize, distance} = require('../audit_ppocr_pipeline/runtime.cjs');
const OUT = path.join(__dirname, 'outputs');
const jsonl = f => fs.readFileSync(f, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const pct = x => (100 * x).toFixed(3);

async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  const split = value('--split');
  const api = createRuntime(path.join(R, 'outputs/models-published'));
  await api.init();
  const base = jsonl(path.join(OUT, `ppocr_${split}.jsonl`));
  const refs = new Map(base.map(r => [r.bubble_id, r.reference]));
  const files = args.includes('--files') ? args.slice(args.indexOf('--files') + 1).filter(a => !a.startsWith('--') && a !== value('--tag'))
    : fs.readdirSync(path.join(OUT, 'predictions', split)).filter(f => f.endsWith('.jsonl')).sort().map(f => path.join(OUT, 'predictions', split, f));
  const runs = [['A_yolo_ppocrv6', null, base.map(r => ({bubble_id: r.bubble_id, raw: r.raw, lines: r.yolo_boxes.length}))]];
  for (const f of files) {
    const [header, ...rows] = jsonl(f);
    runs.push([path.basename(f, '.jsonl'), header.config, rows]);
  }
  const report = {split, n: base.length, postprocess: 'production postprocessPpocrText (published manifest)', runs: {}};
  console.log(['run', 'n', 'CER raw %', 'CER final %', 'exact final %', 'CER truecase %', 'exact truecase %', 'content CER %', 'empty', 'short CER %'].join('\t'));
  for (const [name, config, rows] of runs) {
    if (rows.length !== base.length || rows.some(r => !refs.has(r.bubble_id))) throw new Error(`${name}: rows differ from the split`);
    // truecase: for recognizers that read the all-caps lettering literally, lowercase first and let the
    // production rules (train-label lexicon + sentence starts) restore case.
    const scored = rows.map(r => ({...r, reference: refs.get(r.bubble_id), text: api.post(r.raw),
      truecase: api.post(String(r.raw ?? '').toLocaleLowerCase('fr-FR'))}));
    const short = scored.filter(r => Array.from(normalize(r.reference)).length <= 12);
    const result = {config, raw: score(scored, 'raw'), final: score(scored), truecase: score(scored, 'truecase'), short_final: score(short),
      zero_line_bubbles: rows.filter(r => r.lines === 0).length,
      // per bubble (final text): [bubble_id, edit distance, reference chars], for paired bootstrap
      per_bubble: scored.map(r => [r.bubble_id, distance(normalize(r.reference), normalize(r.text)), Array.from(normalize(r.reference)).length])};
    report.runs[name] = result;
    console.log([name, rows.length, pct(result.raw.cer), pct(result.final.cer), pct(result.final.exact), pct(result.truecase.cer),
      pct(result.truecase.exact), pct(result.final.content_cer), result.final.empty, pct(result.short_final.cer)].join('\t'));
  }
  fs.mkdirSync(path.join(OUT, 'scores'), {recursive: true});
  const outFile = path.join(OUT, 'scores', `${split}${args.includes('--tag') ? '_' + value('--tag') : ''}.json`);
  fs.writeFileSync(outFile, JSON.stringify(report, null, 2));
}
main().catch(e => { console.error(e); process.exitCode = 1; });
