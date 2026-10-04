const assert = require('node:assert/strict');
const http = require('node:http');
const test = require('node:test');
const express = require('express');
const sharp = require('sharp');
const { createPageRouter } = require('../src/routes/pageRoutes');
const { createPageMiniature } = require('../src/utils/imageThumbnail');
const { cacheKey, imageCache } = require('../src/utils/imageCache');

async function withServer(router, callback) {
  const app = express();
  app.use('/api/pages', router);
  const server = http.createServer(app);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    await callback(`http://127.0.0.1:${server.address().port}/api/pages/710/image/miniature`);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}

function fakePages(getPage) {
  return { from(table) {
    assert.equal(table, 'pages', 'miniatures do not need annotation queries');
    const query = {
      select: () => query, eq: () => query,
      single: async () => ({ data: getPage(), error: null }),
    };
    return query;
  } };
}

test('the public route returns only a fixed, unblurred miniature, caches it and coalesces concurrent reads', async () => {
  let sourceKey = 'miniature-test-original.png';
  let reads = 0;
  const original = await sharp(Buffer.from('<svg width="1200" height="1800"><rect width="1200" height="1800" fill="white"/><path d="M0 0L1200 1800M1200 0L0 1800" stroke="black" stroke-width="30"/></svg>')).png().toBuffer();
  const expected = await createPageMiniature(original);
  const keys = [sourceKey, 'miniature-test-replacement.png'].map(source => cacheKey.pageMiniature({ pageId: '710', source }));
  keys.forEach(key => imageCache.del(key));
  const router = createPageRouter({
    supabaseClient: fakePages(() => ({ url_image: sourceKey })),
    readImage: async () => {
      reads += 1;
      await new Promise(resolve => setImmediate(resolve));
      return { buffer: original };
    },
    previewImage: async () => { throw new Error('miniatures must not be blurred'); },
    requireAuth: () => { throw new Error('miniatures are public'); },
  });

  try {
    await withServer(router, async url => {
      const responses = await Promise.all([fetch(url), fetch(`${url}?width=9999&height=9999&original=true`)]);
      for (const response of responses) {
        assert.equal(response.status, 200);
        assert.equal(response.headers.get('content-type'), 'image/avif');
        assert.equal(response.headers.get('cache-control'), 'public, max-age=86400');
        assert.equal(response.headers.get('cross-origin-resource-policy'), 'cross-origin');
        assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
        const body = Buffer.from(await response.arrayBuffer());
        assert.deepEqual(body, expected);
        const metadata = await sharp(body).metadata();
        assert.equal(metadata.height, 128);
        assert.ok(metadata.width <= 192);
      }
      assert.equal(reads, 1);
      await (await fetch(url)).arrayBuffer();
      assert.equal(reads, 1, 'cached requests do not read the source again');
      sourceKey = 'miniature-test-replacement.png';
      await (await fetch(url)).arrayBuffer();
      assert.equal(reads, 2, 'replacing the source invalidates the server cache');
    });
  } finally {
    keys.forEach(key => imageCache.del(key));
  }
});

test('missing pages and invalid source images never fall back to full-size media', async () => {
  let page = null;
  const router = createPageRouter({
    supabaseClient: fakePages(() => page),
    readImage: async () => ({ buffer: Buffer.from('<html>storage error</html>') }),
  });
  await withServer(router, async url => {
    assert.equal((await fetch(url)).status, 404);
    page = { url_image: 'miniature-test-invalid.html' };
    const invalid = await fetch(url);
    assert.equal(invalid.status, 415);
    assert.match(invalid.headers.get('content-type'), /application\/json/);
  });
});

test('a failed source read can be retried without leaving a pending cached request', async () => {
  const original = await sharp({ create: { width: 200, height: 300, channels: 3, background: '#ffffff' } }).png().toBuffer();
  let reads = 0;
  const router = createPageRouter({
    supabaseClient: fakePages(() => ({ url_image: 'miniature-test-retry.png' })),
    readImage: async () => {
      if (++reads === 1) throw Object.assign(new Error('Storage timed out'), { code: 'PAGE_IMAGE_TIMEOUT', statusCode: 504 });
      return { buffer: original };
    },
  });
  const key = cacheKey.pageMiniature({ pageId: '710', source: 'miniature-test-retry.png' });
  imageCache.del(key);
  try {
    await withServer(router, async url => {
      assert.equal((await fetch(url)).status, 504);
      const retried = await fetch(url);
      assert.equal(retried.status, 200);
      assert.equal((await sharp(Buffer.from(await retried.arrayBuffer())).metadata()).height, 128);
    });
  } finally {
    imageCache.del(key);
  }
});
