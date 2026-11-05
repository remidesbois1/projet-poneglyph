const assert = require('node:assert/strict');
const http = require('node:http');
const test = require('node:test');
const express = require('express');
const multer = require('multer');
const { createPageUploadMiddleware } = require('../src/routes/pageUploadMiddleware');

test('page upload accepts the two required parts and rejects extra fields or files', async () => {
  const app = express();
  app.post('/upload', createPageUploadMiddleware(multer.memoryStorage()), (req, res) => {
    res.json({ key: req.body.key, content: req.file?.buffer.toString() });
  });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/upload`;
  const image = new Blob(['image'], { type: 'image/webp' });
  try {
    for (const reverse of [false, true]) {
      const form = new FormData();
      if (reverse) form.append('key', 'tome-1/chapitre-1/1.webp');
      form.append('file', image, 'page.webp');
      if (!reverse) form.append('key', 'tome-1/chapitre-1/1.webp');
      const response = await fetch(url, { method: 'POST', body: form });
      assert.equal(response.status, 200);
      assert.deepEqual(await response.json(), { key: 'tome-1/chapitre-1/1.webp', content: 'image' });
    }

    for (const extra of ['field', 'file']) {
      const form = new FormData();
      form.append('file', image, 'page.webp');
      form.append('key', 'tome-1/chapitre-1/1.webp');
      if (extra === 'field') form.append('extra', 'unexpected');
      else form.append('file', image, 'second.webp');
      const response = await fetch(url, { method: 'POST', body: form });
      assert.equal(response.status, 400);
    }
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});
