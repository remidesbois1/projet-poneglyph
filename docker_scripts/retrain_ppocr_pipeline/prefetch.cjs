/** Cache private source pages concurrently before allocating any paid GPU. */
const fs = require('node:fs'), path = require('node:path'), sharp = require('sharp');
const { S3Client, GetObjectCommand } = require('@aws-sdk/client-s3');
const ROOT = path.resolve(__dirname, '../..');
async function main() {
  const plan = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const env = require('dotenv').parse(fs.readFileSync(path.join(ROOT, 'backend/.env')));
  const s3 = new S3Client({ region: 'auto', endpoint: env.R2_ENDPOINT,
    credentials: { accessKeyId: env.R2_ACCESS_KEY_ID, secretAccessKey: env.R2_SECRET_ACCESS_KEY } });
  const cache = path.join(ROOT, 'docker_scripts/audit_ppocr_pipeline/cache/pages');
  let next = 0, done = 0;
  await Promise.all(Array.from({ length: 6 }, async () => {
    while (next < plan.pages.length) {
      const page = plan.pages[next++], target = path.join(cache, `${page.id}.png`);
      if (!fs.existsSync(target)) {
        let buffer;
        if (page.source.startsWith('r2://')) {
          const u = new URL(page.source);
          if (u.hostname !== env.R2_PAGES_BUCKET_NAME) throw new Error('Unexpected R2 bucket');
          const response = await s3.send(new GetObjectCommand({ Bucket: u.hostname, Key: decodeURIComponent(u.pathname.slice(1)) }));
          buffer = Buffer.from(await response.Body.transformToByteArray());
        } else {
          const response = await fetch(page.source, { signal: AbortSignal.timeout(60000) });
          if (!response.ok) throw new Error(`Page ${page.id}: HTTP ${response.status}`);
          buffer = Buffer.from(await response.arrayBuffer());
        }
        const png = await sharp(buffer).rotate().png().toBuffer();
        const tmp = target + '.prefetch'; fs.writeFileSync(tmp, png); fs.renameSync(tmp, target);
      }
      if (++done % 100 === 0 || done === plan.pages.length) console.log(JSON.stringify({ cached: done, total: plan.pages.length }));
    }
  }));
}
main().catch(e => { console.error(e.message); process.exitCode = 1; });
