const assert = require('node:assert/strict');
const test = require('node:test');
const axios = require('axios');
const storage = require('../src/utils/pageStorage');

function loadClient(t, serverKey = 'server-gemini-secret') {
  const previousKey = process.env.GOOGLE_API_KEY;
  if (serverKey === null) delete process.env.GOOGLE_API_KEY;
  else process.env.GOOGLE_API_KEY = serverKey;
  delete require.cache[require.resolve('../src/utils/geminiClient')];
  t.after(() => {
    if (previousKey === undefined) delete process.env.GOOGLE_API_KEY;
    else process.env.GOOGLE_API_KEY = previousKey;
    delete require.cache[require.resolve('../src/utils/geminiClient')];
  });
  return require('../src/utils/geminiClient').generateGeminiEmbedding;
}

test('strict Gemini embeddings read original bytes and never silently fall back to text', async (t) => {
  const reference = 'r2://private-pages/page.png';
  const read = t.mock.method(storage, 'readPageImage', async (image) => {
    assert.equal(image, reference);
    return { buffer: Buffer.from('original'), contentType: 'image/png' };
  });
  const post = t.mock.method(axios, 'post', async (_url, body) => {
    assert.deepEqual(body.content.parts, [
      { text: 'Nouvelle description' },
      { inlineData: { mimeType: 'image/png', data: Buffer.from('original').toString('base64') } },
    ]);
    return { data: { embedding: { values: Array(3072).fill(0.1) } } };
  });
  const previousKey = process.env.GOOGLE_API_KEY;
  process.env.GOOGLE_API_KEY = 'test-google-key';
  delete require.cache[require.resolve('../src/utils/geminiClient')];
  const { generateGeminiEmbedding } = require('../src/utils/geminiClient');
  try {
    await generateGeminiEmbedding('Nouvelle description', 'RETRIEVAL_DOCUMENT', reference, { requireImage: true });
    read.mock.mockImplementation(async () => { throw new Error('Original unavailable'); });
    await assert.rejects(generateGeminiEmbedding('Nouvelle description', 'RETRIEVAL_DOCUMENT', reference, { requireImage: true }), /Original unavailable/);
    await assert.rejects(generateGeminiEmbedding('Nouvelle description', 'RETRIEVAL_DOCUMENT', null, { requireImage: true }), /Original page image is required/);
    assert.equal(post.mock.callCount(), 1);
  } finally {
    if (previousKey === undefined) delete process.env.GOOGLE_API_KEY;
    else process.env.GOOGLE_API_KEY = previousKey;
    delete require.cache[require.resolve('../src/utils/geminiClient')];
  }
});

for (const status of [400, 401, 403, 429, 503, 'network']) {
  test(`retries a Gemini server ${status} error with the personal key and the same original payload`, async (t) => {
    const read = t.mock.method(storage, 'readPageImage', async () => ({ buffer: Buffer.from('private-original'), contentType: 'image/png' }));
    const generate = loadClient(t);
    const vector = Array(3072).fill(0.2);
    const post = t.mock.method(axios, 'post', async (url, _body, options) => {
      assert.ok(!url.includes('key='));
      if (options.headers['x-goog-api-key'] === 'server-gemini-secret') {
        throw { message: 'private server error', ...(status === 'network' ? {} : { response: { status } }) };
      }
      assert.equal(options.headers['x-goog-api-key'], 'personal-gemini-secret');
      return { data: { embedding: { values: vector } } };
    });
    let fallbacks = 0;
    const result = await generate('Nouvelle description', 'RETRIEVAL_DOCUMENT', 'r2://private/page.png', {
      requireImage: true, fallbackApiKey: 'personal-gemini-secret', onFallback: () => { fallbacks++; },
    });
    assert.deepEqual(result, vector);
    assert.equal(fallbacks, 1);
    assert.equal(read.mock.callCount(), 1);
    assert.equal(post.mock.callCount(), 2);
    assert.equal(post.mock.calls[0].arguments[1], post.mock.calls[1].arguments[1]);
    assert.equal(process.env.GOOGLE_API_KEY, 'server-gemini-secret');
  });
}

test('never spends the personal key when the server request succeeds', async (t) => {
  const generate = loadClient(t);
  const post = t.mock.method(axios, 'post', async (_url, _body, options) => {
    assert.equal(options.headers['x-goog-api-key'], 'server-gemini-secret');
    return { data: { embedding: { values: [0.1] } } };
  });
  let fallbacks = 0;
  await generate('text', 'RETRIEVAL_DOCUMENT', null, { fallbackApiKey: 'personal-gemini-secret', onFallback: () => { fallbacks++; } });
  assert.equal(post.mock.callCount(), 1);
  assert.equal(fallbacks, 0);
});

test('uses the personal key directly when no server key exists', async (t) => {
  const generate = loadClient(t, null);
  const post = t.mock.method(axios, 'post', async (_url, _body, options) => {
    assert.equal(options.headers['x-goog-api-key'], 'personal-gemini-secret');
    return { data: { embedding: { values: [0.1] } } };
  });
  let fallbacks = 0;
  await generate('text', 'RETRIEVAL_DOCUMENT', null, { fallbackApiKey: 'personal-gemini-secret', onFallback: () => { fallbacks++; } });
  assert.equal(post.mock.callCount(), 1);
  assert.equal(fallbacks, 1);
});

test('does not retry with an identical key and exposes only safe provider status', async (t) => {
  const generate = loadClient(t);
  const post = t.mock.method(axios, 'post', async () => {
    throw { response: { status: 429 }, config: { headers: { 'x-goog-api-key': 'server-gemini-secret' } }, message: 'server-gemini-secret' };
  });
  await assert.rejects(generate('text', 'RETRIEVAL_DOCUMENT', null, { fallbackApiKey: 'server-gemini-secret' }), (error) => {
    assert.equal(error.response.status, 429);
    assert.equal(error.geminiKeySource, 'server');
    assert.ok(!JSON.stringify(error).includes('server-gemini-secret'));
    assert.ok(!error.message.includes('server-gemini-secret'));
    return true;
  });
  assert.equal(post.mock.callCount(), 1);
});

for (const failure of [
  { response: { status: 429 } },
  { response: { status: 403 } },
  { response: { status: 400, data: { error: { details: [{ reason: 'API_KEY_INVALID' }] } } } },
]) {
  test(`reports personal-key failure safely (${failure.response.status}) without further retries`, async (t) => {
    const generate = loadClient(t);
    const post = t.mock.method(axios, 'post', async (_url, _body, options) => {
      if (options.headers['x-goog-api-key'] === 'server-gemini-secret') throw { response: { status: 429 } };
      throw { ...failure, config: { headers: { 'x-goog-api-key': 'personal-gemini-secret' } }, message: 'personal-gemini-secret' };
    });
    await assert.rejects(generate('text', 'RETRIEVAL_DOCUMENT', null, { fallbackApiKey: 'personal-gemini-secret' }), (error) => {
      assert.equal(error.geminiKeySource, 'user');
      assert.equal(error.response.status, failure.response.status);
      assert.equal(error.code, failure.response.status === 429 ? 'GEMINI_EMBEDDING_FAILED' : 'GEMINI_USER_KEY_REJECTED');
      assert.ok(!JSON.stringify(error).includes('personal-gemini-secret'));
      assert.ok(!error.message.includes('personal-gemini-secret'));
      return true;
    });
    assert.equal(post.mock.callCount(), 2);
  });
}

test('keeps personal keys separate across concurrent page requests', async (t) => {
  const generate = loadClient(t);
  t.mock.method(axios, 'post', async (_url, payload, options) => {
    const key = options.headers['x-goog-api-key'];
    if (key === 'server-gemini-secret') throw { response: { status: 429 } };
    const text = payload.content.parts[0].text;
    assert.equal(key, text === 'page A' ? 'personal-key-A' : 'personal-key-B');
    return { data: { embedding: { values: [text === 'page A' ? 0.1 : 0.2] } } };
  });
  const results = await Promise.all([
    generate('page A', 'RETRIEVAL_DOCUMENT', null, { fallbackApiKey: 'personal-key-A' }),
    generate('page B', 'RETRIEVAL_DOCUMENT', null, { fallbackApiKey: 'personal-key-B' }),
  ]);
  assert.deepEqual(results, [[0.1], [0.2]]);
  assert.equal(process.env.GOOGLE_API_KEY, 'server-gemini-secret');
});

test('an original-image failure never triggers a credential fallback', async (t) => {
  t.mock.method(storage, 'readPageImage', async () => { throw new Error('Original unavailable'); });
  const generate = loadClient(t);
  const post = t.mock.method(axios, 'post', async () => { throw new Error('should not call Google'); });
  await assert.rejects(generate('text', 'RETRIEVAL_DOCUMENT', 'r2://private/page.png', { requireImage: true, fallbackApiKey: 'personal-key-A' }), /Original unavailable/);
  assert.equal(post.mock.callCount(), 0);
});
