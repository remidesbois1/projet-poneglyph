/** Private local training export. Supabase/R2 access is read-only; no uploads. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const { createRuntime, tensorToPng, loadImage, ROOT } = require('./worker_runtime.cjs');
const CACHE = path.join(ROOT, 'docker_scripts/audit_ppocr_pipeline/cache');
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const read = name => JSON.parse(fs.readFileSync(name, 'utf8'));
const write = (name, value) => fs.writeFileSync(name, JSON.stringify(value, null, 2));

function buildPlan(validated, audit, limitPages = 0) {
  const heldout = new Map();
  for (const e of audit.examples) {
    if (heldout.has(e.page_id) && heldout.get(e.page_id) !== e.split) throw new Error('Historical page leakage');
    heldout.set(e.page_id, e.split);
  }
  const ids = new Set(), pages = new Map();
  for (const b of validated) {
    if (ids.has(b.id)) throw new Error(`Duplicate bubble ${b.id}`);
    ids.add(b.id);
    if (!pages.has(b.id_page)) pages.set(b.id_page, { id: b.id_page, split: heldout.get(b.id_page) || 'train', source: b.pages?.url_image, bubbles: [] });
    pages.get(b.id_page).bubbles.push({ id: b.id, x: b.x, y: b.y, w: b.w, h: b.h, text: b.texte_propose });
  }
  let selected = [...pages.values()].sort((a, b) => a.id - b.id);
  if (limitPages) selected = ['train', 'val'].flatMap(split => selected.filter(p => p.split === split).slice(0, limitPages));
  return { schema_version: 2, smoke: Boolean(limitPages), snapshot_created_at: audit.created_at,
    evaluation_note: 'Historical val/test pages are preserved, but are already inspected, not blind holdouts. Annotation completeness remains unproven.',
    counts: Object.fromEntries(['train', 'val', 'test'].map(split => [split, { pages: selected.filter(p => p.split === split).length, bubbles: selected.filter(p => p.split === split).reduce((n, p) => n + p.bubbles.length, 0) }])),
    pages: selected };
}

async function prepare({ output, limitPages = 0, planOnly = false, recognitionLimit = 0, detectorOnly = false, reuseFrom }) {
  fs.mkdirSync(output, { recursive: true });
  const plan = buildPlan(read(path.join(CACHE, 'validated.json')), read(path.join(CACHE, 'dataset.json')), limitPages);
  plan.inputs = Object.fromEntries(['validated.json', 'dataset.json'].map(n => [n, sha(fs.readFileSync(path.join(CACHE, n)))]));
  plan.worker_sha256 = sha(fs.readFileSync(path.join(ROOT, 'frontend/src/workers/ocr.worker.js')));
  plan.utils_sha256 = sha(fs.readFileSync(path.join(ROOT, 'frontend/src/lib/ocr-utils.js')));
  plan.manifest_sha256 = sha(fs.readFileSync(path.join(CACHE, 'models/browser_manifest.json')));
  plan.line_model_sha256 = sha(fs.readFileSync(path.join(CACHE, 'models/bubble_line_detector_yolo26n.onnx')));
  plan.prepare_sha256 = sha(fs.readFileSync(__filename));
  plan.recognition_limit = recognitionLimit;
  plan.detector_only = detectorOnly;
  if (reuseFrom) {
    const sourcePlan = read(path.join(reuseFrom, 'plan.json'));
    for (const key of ['inputs', 'worker_sha256', 'utils_sha256', 'manifest_sha256', 'line_model_sha256', 'recognition_limit', 'detector_only', 'pages']) {
      if (JSON.stringify(sourcePlan[key]) !== JSON.stringify(plan[key])) throw new Error(`Cannot reuse different ${key}`);
    }
    plan.reused_export = { source: reuseFrom, plan_sha256: sha(fs.readFileSync(path.join(reuseFrom, 'plan.json'))) };
  }
  const planPath = path.join(output, 'plan.json');
  if (fs.existsSync(planPath) && JSON.stringify(read(planPath)) !== JSON.stringify(plan)) throw new Error('Output already contains a different plan; use a new run directory');
  write(planPath, plan);
  console.log(JSON.stringify({ plan: plan.counts, smoke: plan.smoke }));
  if (planOnly) return plan;

  const env = require('dotenv').parse(fs.readFileSync(path.join(ROOT, 'backend/.env')));
  const db = require('@supabase/supabase-js').createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });
  const { S3Client, GetObjectCommand } = require('@aws-sdk/client-s3');
  if (!env.R2_ENDPOINT) throw new Error('R2_ENDPOINT is required in backend/.env');
  const s3 = new S3Client({ region: 'auto', endpoint: env.R2_ENDPOINT, credentials: { accessKeyId: env.R2_ACCESS_KEY_ID, secretAccessKey: env.R2_SECRET_ACCESS_KEY }, maxAttempts: 3 });
  const completenessPath = path.join(output, 'annotation_status.json');
  let status;
  if (fs.existsSync(completenessPath)) status = read(completenessPath);
  else {
    status = [];
    for (let offset = 0; ; offset += 1000) {
      const r = await db.from('bulles').select('id,id_page,statut').order('id').range(offset, offset + 999);
      if (r.error) throw new Error(`Annotation status read: ${r.error.message}`);
      status.push(...r.data);
      if (r.data.length < 1000) break;
    }
    write(completenessPath, status);
  }
  const knownIds = new Set(plan.pages.flatMap(p => p.bubbles.map(b => b.id)));
  const incompletePages = new Set(status.filter(b => b.statut !== 'Valid\u00e9' || !knownIds.has(b.id)).map(b => b.id_page));
  const runtime = createRuntime(path.join(CACHE, 'models'), { threads: 2 });
  if (!detectorOnly) await runtime.init();
  const rows = [], exclusions = [], detectorPages = [], inputHashes = new Map();
  for (const split of ['train', 'val', 'test']) {
    fs.mkdirSync(path.join(output, 'detector', split, 'images'), { recursive: true });
    fs.mkdirSync(path.join(output, 'detector', split, 'labels'), { recursive: true });
    fs.mkdirSync(path.join(output, 'recognizer', 'images', split), { recursive: true });
  }
  fs.mkdirSync(path.join(output, 'pages'), { recursive: true });
  fs.mkdirSync(path.join(output, 'prepared_pages'), { recursive: true });
  let done = 0;
  async function processPage(page) {
    const pageFile = path.join(output, 'pages', `${page.id}.png`);
    if (!fs.existsSync(pageFile)) {
      const cached = path.join(CACHE, 'pages', `${page.id}.png`);
      if (fs.existsSync(cached)) fs.copyFileSync(cached, pageFile);
      else {
        if (!page.source) throw new Error(`Page ${page.id} has no source`);
        let buffer;
        if (page.source.startsWith('r2://')) {
          const u = new URL(page.source);
          if (u.hostname !== env.R2_PAGES_BUCKET_NAME) throw new Error('Unexpected R2 bucket');
          const r = await s3.send(new GetObjectCommand({ Bucket: u.hostname, Key: decodeURIComponent(u.pathname.slice(1)) }));
          buffer = Buffer.from(await r.Body.transformToByteArray());
        } else {
          const r = await fetch(page.source, { signal: AbortSignal.timeout(60000) });
          if (!r.ok) throw new Error(`Page ${page.id}: HTTP ${r.status}`);
          buffer = Buffer.from(await r.arrayBuffer());
        }
        await sharp(buffer).rotate().png().toFile(pageFile);
      }
    }
    const pageBuffer = fs.readFileSync(pageFile), digest = sha(pageBuffer);
    if (inputHashes.has(digest) && inputHashes.get(digest) !== page.split) throw new Error(`Duplicate image across splits: page ${page.id}`);
    inputHashes.set(digest, page.split);
    const { width, height } = await sharp(pageBuffer).metadata();
    const invalid = page.bubbles.filter(b => ![b.x, b.y, b.w, b.h].every(Number.isFinite) || b.w <= 0 || b.h <= 0 || b.x >= width || b.y >= height || b.x + b.w <= 0 || b.y + b.h <= 0);
    if (invalid.length) throw new Error(`Invalid boxes on page ${page.id}: ${invalid.map(b => b.id)}`);
    const bubbles = page.bubbles.map(b => {
      const x = Math.max(0, b.x), y = Math.max(0, b.y);
      const clipped = { ...b, x, y, w: Math.min(width, b.x + b.w) - x, h: Math.min(height, b.y + b.h) - y };
      if (['x', 'y', 'w', 'h'].some(k => clipped[k] !== b[k])) exclusions.push({ bubble_id: b.id, page_id: page.id, split: page.split, stage: 'geometry', reason: 'clipped_to_page', original: b, clipped });
      return clipped;
    });
    if (incompletePages.has(page.id)) exclusions.push({ page_id: page.id, stage: 'detector', reason: 'unvalidated_or_new_annotations' });
    else {
      const stem = path.join(output, 'detector', page.split);
      fs.copyFileSync(pageFile, path.join(stem, 'images', `${page.id}.png`));
      fs.writeFileSync(path.join(stem, 'labels', `${page.id}.txt`), bubbles.map(b => `0 ${(b.x + b.w / 2) / width} ${(b.y + b.h / 2) / height} ${b.w / width} ${b.h / height}`).join('\n') + '\n');
      detectorPages.push({ id: page.id, split: page.split, sha256: digest, bubbles: page.bubbles.length });
    }
    const checkpoint = path.join(output, 'prepared_pages', `${page.id}.json`);
    if (reuseFrom && !fs.existsSync(checkpoint)) {
      const sourceCheckpoint = path.join(reuseFrom, 'prepared_pages', `${page.id}.json`);
      if (fs.existsSync(sourceCheckpoint)) {
        const saved = read(sourceCheckpoint);
        if (saved.page_sha256 !== digest) throw new Error(`Reused page ${page.id} changed`);
        for (const row of saved.rows) {
          const source = path.join(reuseFrom, 'recognizer', row.single_line_image);
          if (sha(fs.readFileSync(source)) !== row.sha256) throw new Error(`Corrupt reused input ${row.bubble_id}`);
          fs.copyFileSync(source, path.join(output, 'recognizer', row.single_line_image));
          row.trainable = Boolean(row.text?.trim());
        }
        for (const e of saved.exclusions) e.excluded_from_train = Boolean(e.reasons?.includes('empty_target'));
        write(checkpoint, saved);
      }
    }
    if (!detectorOnly && fs.existsSync(checkpoint)) {
      const saved = read(checkpoint);
      if (saved.page_sha256 !== digest || saved.rows.some(r => !fs.existsSync(path.join(output, 'recognizer', r.single_line_image)))) throw new Error(`Invalid checkpoint for page ${page.id}`);
      rows.push(...saved.rows);
      exclusions.push(...saved.exclusions);
    } else if (!detectorOnly) {
      const pageRows = [], pageExclusions = [];
      for (const b of bubbles) {
      if (recognitionLimit && rows.filter(r => r.split === page.split).length >= recognitionLimit) break;
      const left = Math.round(b.x), top = Math.round(b.y), cropWidth = Math.min(width - left, Math.round(b.w)), cropHeight = Math.min(height - top, Math.round(b.h));
      const crop = await sharp(pageBuffer).extract({ left, top, width: cropWidth, height: cropHeight }).png().toBuffer();
      const image = await loadImage(crop), boxes = await runtime.detect(image);
      const tensor = runtime.tensor(runtime.stitch(image, boxes));
      const rel = `images/${page.split}/${b.id}.png`, png = tensorToPng(tensor);
      fs.writeFileSync(path.join(output, 'recognizer', rel), png);
      const reasons = [];
      if (!boxes.length) reasons.push('zero_detected_lines');
      // Annotation line breaks describe the bubble layout, which is stitched
      // into one line. Keep the raw annotation and serialize whitespace safely.
      const text = String(b.text || '').replace(/\s+/g, ' ').trim();
      if (!text) reasons.push('empty_target');
      // Zero-line fallback is part of production and includes SFX. Retain it;
      // diagnose it without silently removing this class from training.
      const trainable = Boolean(text);
      if (reasons.length) pageExclusions.push({ bubble_id: b.id, stage: 'recognizer', split: page.split, reasons, excluded_from_train: !trainable });
      const row = { bubble_id: b.id, page_id: page.id, split: page.split, text, reference: b.text, single_line_image: rel,
        detected_lines: boxes, width: tensor.dims[3], height: 48, sha256: sha(png), trainable };
      rows.push(row);
      pageRows.push(row);
      }
      exclusions.push(...pageExclusions);
      write(checkpoint, { page_sha256: digest, rows: pageRows, exclusions: pageExclusions });
    }
    if (++done % 25 === 0 || done === plan.pages.length) console.log(JSON.stringify({ pages: done, total: plan.pages.length, recognition_rows: rows.length }));
  }
  let nextPage = 0;
  const workerCount = recognitionLimit ? 1 : 4;
  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (nextPage < plan.pages.length) await processPage(plan.pages[nextPage++]);
  }));
  rows.sort((a, b) => a.page_id - b.page_id || a.bubble_id - b.bubble_id);
  detectorPages.sort((a, b) => a.id - b.id);
  const detectorRoot = path.join(output, 'detector').replaceAll('\\', '/');
  fs.writeFileSync(path.join(output, 'detector/data.yaml'), `path: ${JSON.stringify(detectorRoot)}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: bubble\n`);
  for (const split of ['train', 'val', 'test']) fs.writeFileSync(path.join(output, 'recognizer', `rec_gt_${split}.txt`), rows.filter(r => r.split === split && r.trainable).map(r => `${r.single_line_image}\t${r.text}`).join('\n') + '\n');
  fs.writeFileSync(path.join(output, 'recognizer/line_manifest.jsonl'), rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  write(path.join(output, 'exclusions.json'), exclusions);
  write(path.join(output, 'detector/pages.json'), detectorPages);
  const summary = { smoke: plan.smoke, limited_recognition: Boolean(recognitionLimit), detector_only: detectorOnly,
    worker_sha256: plan.worker_sha256, line_model_sha256: plan.line_model_sha256, constants: runtime.constants,
    detector_pages: detectorPages.length, recognizer_rows: rows.length, exclusions: exclusions.length,
    recognizer_splits: Object.fromEntries(['train', 'val', 'test'].map(s => [s, rows.filter(r => r.split === s && r.trainable).length])) };
  write(path.join(output, 'dataset_summary.json'), summary);
  console.log(JSON.stringify(summary));
  return summary;
}

if (require.main === module) {
  const args = process.argv.slice(2), value = name => args[args.indexOf(name) + 1];
  if (!args.includes('--output')) throw new Error('--output is required');
  prepare({ output: path.resolve(value('--output')), limitPages: args.includes('--smoke-pages') ? Number(value('--smoke-pages')) : 0,
    recognitionLimit: args.includes('--recognition-limit') ? Number(value('--recognition-limit')) : 0,
    planOnly: args.includes('--plan-only'), detectorOnly: args.includes('--detector-only'),
    reuseFrom: args.includes('--reuse-prepared-pages') ? path.resolve(value('--reuse-prepared-pages')) : undefined,
  }).catch(e => { console.error(e.message); process.exitCode = 1; });
}
module.exports = { buildPlan, prepare };
