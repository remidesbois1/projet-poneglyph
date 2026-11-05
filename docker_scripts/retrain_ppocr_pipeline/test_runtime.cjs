const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const { buildPlan } = require('./prepare.cjs');
const { createRuntime, tensorToPng, createCanvas, ROOT } = require('./worker_runtime.cjs');

async function main() {
  const a = [{ id: 1, id_page: 10, texte_propose: 'A' }, { id: 2, id_page: 10, texte_propose: 'B' }, { id: 3, id_page: 20, texte_propose: 'C' }];
  const plan = buildPlan(a, { examples: [{ page_id: 10, split: 'test' }] });
  assert.equal(plan.pages.find(p => p.id === 10).split, 'test');
  assert.equal(plan.pages.find(p => p.id === 20).split, 'train');
  assert.throws(() => buildPlan(a, { examples: [{ page_id: 10, split: 'val' }, { page_id: 10, split: 'test' }] }), /leakage/);
  assert.throws(() => buildPlan([a[0], a[0]], { examples: [] }), /Duplicate bubble/);

  const runtime = createRuntime(path.join(ROOT, 'docker_scripts/audit_ppocr_pipeline/cache/models'));
  await runtime.init();
  assert.equal(runtime.post('"Bonjour !"'), '"Bonjour !"', 'Actual quotation marks must survive OCR postprocessing');
  assert.equal(runtime.post('Chapitre I'), 'Chapitre I', 'Roman numerals must not be rewritten as French pronouns');
  const canvas = createCanvas(64, 32), ctx = canvas.getContext('2d');
  ctx.fillStyle = '#2050c0'; ctx.fillRect(0, 0, 64, 32);
  const tensor = runtime.tensor(canvas);
  assert.deepEqual(Array.from(tensor.dims), [1, 3, 48, 960]);
  assert.ok(Math.abs(tensor.data[0] - (192 / 255 - .5) / .5) < 1e-7, 'First channel must be blue');
  assert.ok(Math.abs(tensor.data[500] - (128 / 255 - .5) / .5) < 1e-7, 'Padding must be #808080');
  const png = tensorToPng(tensor), { data, info } = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  let maxDifference = 0;
  const n = info.width * info.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
    const expected = Math.fround((data[3 * i + 2 - c] / 255 - .5) / .5);
    maxDifference = Math.max(maxDifference, Math.abs(expected - tensor.data[c * n + i]));
  }
  assert.equal(maxDifference, 0, 'Exported PNG must reconstruct every production tensor value');
  const out = path.join(__dirname, 'outputs/parity'); fs.mkdirSync(out, { recursive: true });
  fs.writeFileSync(path.join(out, 'input.png'), png);
  fs.writeFileSync(path.join(out, 'tensor.f32'), Buffer.from(tensor.data.buffer));
  console.log(JSON.stringify({ split_integrity: 'passed', tensor_png_roundtrip_max_error: maxDifference, python_fixture: out }));
}
main().catch(e => { console.error(e); process.exitCode = 1; });
