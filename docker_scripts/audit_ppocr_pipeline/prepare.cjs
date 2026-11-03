/** Read-only benchmark snapshot. Never writes to Supabase or R2. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { createClient } = require('@supabase/supabase-js');
const { S3Client, GetObjectCommand } = require('@aws-sdk/client-s3');
const sharp = require('sharp');
const ROOT = path.resolve(__dirname, '../..');
const CACHE = path.join(__dirname, 'cache');
const env = require('dotenv').parse(fs.readFileSync(path.join(ROOT, 'backend/.env')));
const registry = JSON.parse(fs.readFileSync(path.join(ROOT, 'packages/shared/src/model-registry.json')));
for (const dir of ['models', 'pages', 'crops']) fs.mkdirSync(path.join(CACHE, dir), { recursive: true });

function parseCsv(text) {
  const rows = []; let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; } else quoted = !quoted;
    } else if (c === ',' && !quoted) { row.push(field); field = ''; }
    else if (c === '\n' && !quoted) { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (row.length || field) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  const header = rows.shift();
  return rows.filter(r => r.length === header.length).map(r => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

async function pool(items, n, fn) {
  let next = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (next < items.length) await fn(items[next++]); }));
}

async function main() {
  const downloads = [
    ['ppocrv6-line', ['onnx/bubble_line_detector_yolo26n.onnx', 'onnx/ppocrv6_bubble_line_rec.onnx', 'onnx/ppocrv6_bubble_line_rec_webgpu.onnx', 'onnx/browser_manifest.json', 'onnx/pipeline_manifest.json', 'test_predictions.csv', 'validation_predictions.csv', 'dataset_stats.json']],
    ['readernet-reading-order', ['bubble_detector.onnx', 'panel_detector.onnx', 'ordering.onnx', 'model_manifest.json']],
  ];
  await pool(downloads.flatMap(([key, files]) => files.map(file => ({ key, file }))), 3, async ({ key, file }) => {
    const output = path.join(CACHE, 'models', path.basename(file));
    if (fs.existsSync(output)) return;
    const a = registry.models[key].artifact;
    const response = await fetch(`https://huggingface.co/${a.repository}/resolve/${a.revision}/${file}`, { signal: AbortSignal.timeout(180000) });
    if (!response.ok) throw new Error(`Model ${file}: HTTP ${response.status}`);
    fs.writeFileSync(output, Buffer.from(await response.arrayBuffer()));
    console.log('downloaded', file, fs.statSync(output).size);
  });
  const db = createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });
  const snapshotPath = path.join(CACHE, 'validated.json');
  let bubbles;
  if (fs.existsSync(snapshotPath)) bubbles = JSON.parse(fs.readFileSync(snapshotPath));
  else {
    bubbles = [];
    for (let offset = 0; ; offset += 1000) {
      const response = await db.from('bulles').select('id,x,y,w,h,texte_propose,id_page,order,pages(url_image)').eq('statut', 'Valid\u00e9').order('id').range(offset, offset + 999);
      if (response.error) throw new Error(`Supabase read: ${response.error.message}`);
      bubbles.push(...response.data);
      if (response.data.length < 1000) break;
    }
    fs.writeFileSync(snapshotPath, JSON.stringify(bubbles));
  }
  const byId = new Map(bubbles.map(b => [String(b.id), b]));
  const examples = [];
  const missing = [];
  for (const [split, filename] of [['test', 'test_predictions.csv'], ['val', 'validation_predictions.csv']]) {
    for (const old of parseCsv(fs.readFileSync(path.join(CACHE, 'models', filename), 'utf8'))) {
      const b = byId.get(old.bubble_id);
      if (!b) { missing.push({ id: old.bubble_id, split }); continue; }
      examples.push({ id: b.id, page_id: b.id_page, split, reference: b.texte_propose, old_reference: old.reference, old_prediction: old.prediction, old_line_count: Number(old.line_count), bbox: [b.x, b.y, b.w, b.h], order: b.order, crop: `crops/${b.id}.png`, text_changed: old.reference !== b.texte_propose });
    }
  }
  const pageIds = new Set(examples.map(e => e.page_id));
  const pages = [...pageIds].sort((a, b) => a - b).map(id => ({ id, file: `pages/${id}.png`, bubbles: bubbles.filter(b => b.id_page === id).map(b => ({ id: b.id, x: b.x, y: b.y, w: b.w, h: b.h, order: b.order, text: b.texte_propose })) }));
  const s3 = new S3Client({ endpoint: env.R2_ENDPOINT, region: 'auto', credentials: { accessKeyId: env.R2_ACCESS_KEY_ID, secretAccessKey: env.R2_SECRET_ACCESS_KEY }, maxAttempts: 3 });
  let done = 0;
  await pool(pages, 6, async page => {
    const target = path.join(CACHE, page.file);
    if (!fs.existsSync(target)) {
      const source = bubbles.find(b => b.id_page === page.id).pages.url_image;
      let buffer;
      if (source.startsWith('r2://')) {
        const u = new URL(source);
        if (u.hostname !== env.R2_PAGES_BUCKET_NAME) throw new Error('Unexpected R2 bucket');
        const response = await s3.send(new GetObjectCommand({ Bucket: u.hostname, Key: decodeURIComponent(u.pathname.slice(1)) }));
        buffer = Buffer.from(await response.Body.transformToByteArray());
      } else {
        const response = await fetch(source, { signal: AbortSignal.timeout(45000) });
        if (!response.ok) throw new Error(`Page ${page.id}: HTTP ${response.status}`);
        buffer = Buffer.from(await response.arrayBuffer());
      }
      await sharp(buffer).rotate().png().toFile(target);
    }
    const metadata = await sharp(target).metadata();
    page.width = metadata.width; page.height = metadata.height;
    for (const e of examples.filter(e => e.page_id === page.id)) {
      const cropPath = path.join(CACHE, e.crop);
      const [x, y, w, h] = e.bbox.map(Math.round);
      if (x < 0 || y < 0 || x + w > page.width || y + h > page.height) { e.invalid_bbox = true; continue; }
      if (!fs.existsSync(cropPath)) await sharp(target).extract({ left: x, top: y, width: w, height: h }).png().toFile(cropPath);
    }
    if (++done % 25 === 0) console.log('pages', done, '/', pages.length);
  });
  const artifacts = {};
  for (const name of fs.readdirSync(path.join(CACHE, 'models'))) artifacts[name] = crypto.createHash('sha256').update(fs.readFileSync(path.join(CACHE, 'models', name))).digest('hex');
  const manifest = { created_at: new Date().toISOString(), validated_total: bubbles.length, original_test_count: 1219, original_val_count: 1191, missing, artifacts, revisions: Object.fromEntries(downloads.map(([k]) => [k, registry.models[k].artifact])), pages, examples };
  fs.writeFileSync(path.join(CACHE, 'dataset.json'), JSON.stringify(manifest, null, 2));
  console.log(JSON.stringify({ pages: pages.length, examples: examples.length, missing: missing.length, text_changed: examples.filter(e => e.text_changed).length, invalid_bbox: examples.filter(e => e.invalid_bbox).length }));
}
if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });
module.exports = { parseCsv, pool };
