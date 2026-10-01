const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { rebuildPageSemanticData } = require('../src/services/pageSemanticRebuild');
const { createPageSemanticRebuildHandler, readUserGeminiApiKey, ensureSemanticRebuildAvailable } = require('../src/routes/pageSemanticRebuildRoutes');

const description = { content: 'Luffy protège Zoro.', metadata: { arc: 'Wano', characters: ['Luffy', 'Zoro'] } };

function setup() {
  const events = [];
  let written;
  const page = {
    id: 123, url_image: 'r2://private-pages/one-piece/123.png',
    bulles: [{ texte_propose: 'Texte non validé', statut: 'pending', order: 1 }, { texte_propose: 'Second dialogue', statut: 'validated', order: 3 }, { texte_propose: 'Premier dialogue', statut: 'validated', order: 2 }],
  };
  const db = {
    from(table) {
      assert.equal(table, 'pages');
      let updating = false;
      const query = {
        select() { return query; },
        update(value) { events.push('update'); written = value; updating = true; return query; },
        eq(key, value) {
          if (key === 'chapitres.tomes.mangas.slug') assert.equal(value, 'one-piece');
          if (key === 'url_image') assert.equal(value, page.url_image);
          return query;
        },
        async maybeSingle() { return { data: updating ? { id: 123 } : page, error: null }; },
      };
      return query;
    },
  };
  const dependencies = {
    db,
    async generateVoyageEmbedding(text, task) { events.push(['voyage', text, task]); return Array(1024).fill(0.1); },
    async generateGeminiEmbedding(text, task, image, options) { events.push(['gemini', text, task, image, options]); return Array(3072).fill(0.2); },
    async generateF2llmEmbedding() { throw new Error('F2LLM must not run in the Astra rebuild'); },
    now: () => new Date('2026-10-01T12:00:00Z'),
  };
  return { dependencies, events, page, getWritten: () => written };
}

test('builds Voyage and Gemini without F2LLM and publishes them atomically', async () => {
  const fixture = setup();
  const result = await rebuildPageSemanticData({ ...fixture.dependencies, idPage: 123, mangaSlug: 'one-piece', description });
  const text = 'Luffy protège Zoro. Arc: Wano Personnages: Luffy, Zoro Dialogues: Premier dialogue Second dialogue';
  assert.deepEqual(fixture.events, [
    ['voyage', text, 'document'],
    ['gemini', text, 'RETRIEVAL_DOCUMENT', fixture.page.url_image, { requireImage: true }],
    'update',
  ]);
  assert.deepEqual(result, { success: true, id_page: 123, description_model: 'gpt-6-astra', description_prompt_version: 2, description_generated_at: '2026-10-01T12:00:00.000Z' });
  const written = fixture.getWritten();
  assert.deepEqual(written.description, description);
  assert.equal(written.embedding_voyage.length, 1024);
  assert.equal(written.embedding_gemini.length, 3072);
  assert.equal(written.embedding_f2llm, null);
});

for (const provider of ['generateVoyageEmbedding', 'generateGeminiEmbedding']) {
  test(`preserves all previous data if ${provider} fails or returns an invalid vector`, async () => {
    for (const failure of [async () => { throw new Error('provider unavailable'); }, async () => [NaN]]) {
      const fixture = setup();
      fixture.dependencies[provider] = failure;
      await assert.rejects(rebuildPageSemanticData({ ...fixture.dependencies, idPage: 123, mangaSlug: 'one-piece', description }));
      assert.equal(fixture.getWritten(), undefined);
      assert.ok(!fixture.events.includes('update'));
    }
  });
}

test('invalid descriptions are rejected before reading a page or calling embedding services', async () => {
  const fixture = setup();
  fixture.dependencies.db.from = () => { throw new Error('should not read'); };
  await assert.rejects(rebuildPageSemanticData({ ...fixture.dependencies, idPage: 123, mangaSlug: 'one-piece', description: { content: 'missing metadata' } }), { code: 'INVALID_DESCRIPTION', statusCode: 400 });
  assert.deepEqual(fixture.events, []);
});

test('does not save a page missing its original image', async () => {
  const fixture = setup();
  fixture.page.url_image = null;
  await assert.rejects(rebuildPageSemanticData({ ...fixture.dependencies, idPage: 123, mangaSlug: 'one-piece', description }), { code: 'ORIGINAL_IMAGE_MISSING' });
  assert.deepEqual(fixture.events, []);
});

function response() {
  return { statusCode: 200, status(code) { this.statusCode = code; return this; }, json(body) { this.body = body; return this; } };
}

test('admin handler validates manga/id/schema and returns safe embedding quota errors', async () => {
  const fixture = setup();
  const handler = createPageSemanticRebuildHandler(fixture.dependencies);
  for (const req of [
    { body: { id_page: -1, description }, query: { manga: 'one-piece' } },
    { body: { id_page: 123, description }, query: {} },
    { body: { id_page: 123, description: JSON.stringify(description) }, query: { manga: 'one-piece' } },
  ]) {
    const res = response();
    await handler(req, res);
    assert.equal(res.statusCode, 400);
  }
  fixture.dependencies.generateVoyageEmbedding = async () => { throw { response: { status: 429 }, message: 'secret provider url' }; };
  const res = response();
  await handler({ body: { id_page: 123, description }, query: { manga: 'one-piece' } }, res);
  assert.equal(res.statusCode, 429);
  assert.equal(res.body.code, 'EMBEDDING_QUOTA_EXCEEDED');
  assert.ok(!res.body.error.includes('secret'));
  assert.equal(fixture.getWritten(), undefined);
});

test('rebuild route and preflight are protected by Supabase auth and the admin role', () => {
  const source = fs.readFileSync(path.join(__dirname, '../src/routes/adminRoutes.js'), 'utf8');
  assert.match(source, /router\.post\('\/ai-models\/rebuild-page-semantic-data', authMiddleware, roleCheck\(\['Admin'\]\)/);
  assert.match(source, /router\.get\('\/ai-models\/semantic-rebuild-status', authMiddleware, roleCheck\(\['Admin'\]\)/);
});

test('preflight permits a personal Gemini key when the server key is missing, but still requires Voyage', () => {
  assert.doesNotThrow(() => ensureSemanticRebuildAvailable(undefined, { VOYAGE_API_KEY: 'voyage', GOOGLE_API_KEY: 'gemini' }));
  assert.doesNotThrow(() => ensureSemanticRebuildAvailable('user-key', { VOYAGE_API_KEY: 'voyage' }));
  assert.throws(() => ensureSemanticRebuildAvailable(undefined, { VOYAGE_API_KEY: 'voyage' }), { code: 'SEMANTIC_REBUILD_UNAVAILABLE' });
  assert.throws(() => ensureSemanticRebuildAvailable('user-key', { GOOGLE_API_KEY: 'gemini' }), { code: 'SEMANTIC_REBUILD_UNAVAILABLE' });
});

test('personal Gemini credentials are scoped to the request and never saved in the page', async () => {
  const fixture = setup();
  let keyPassed;
  fixture.dependencies.generateGeminiEmbedding = async (_text, _task, _image, options) => {
    keyPassed = options.fallbackApiKey;
    options.onFallback();
    return Array(3072).fill(0.2);
  };
  const handler = createPageSemanticRebuildHandler(fixture.dependencies);
  const res = response();
  await handler({ body: { id_page: 123, description }, query: { manga: 'one-piece' }, headers: { 'x-gemini-api-key': 'personal-gemini-secret' } }, res);
  assert.equal(res.statusCode, 200);
  assert.equal(keyPassed, 'personal-gemini-secret');
  assert.equal(res.body.gemini_key_source, 'user');
  assert.ok(!JSON.stringify(fixture.getWritten()).includes('personal-gemini-secret'));
  assert.ok(!JSON.stringify(res.body).includes('personal-gemini-secret'));
  assert.equal(fixture.events.filter((event) => event[0] === 'voyage').length, 1);
});

test('a server Gemini quota failure falls back within one atomic rebuild without repeating Voyage', async (t) => {
  const axios = require('axios');
  const storage = require('../src/utils/pageStorage');
  const fixture = setup();
  const read = t.mock.method(storage, 'readPageImage', async (image) => {
    assert.equal(image, fixture.page.url_image);
    return { buffer: Buffer.from('private-original'), contentType: 'image/png' };
  });
  const previousKey = process.env.GOOGLE_API_KEY;
  process.env.GOOGLE_API_KEY = 'server-gemini-secret';
  delete require.cache[require.resolve('../src/utils/geminiClient')];
  t.after(() => {
    if (previousKey === undefined) delete process.env.GOOGLE_API_KEY;
    else process.env.GOOGLE_API_KEY = previousKey;
    delete require.cache[require.resolve('../src/utils/geminiClient')];
  });
  const post = t.mock.method(axios, 'post', async (_url, _body, options) => {
    if (options.headers['x-goog-api-key'] === 'server-gemini-secret') throw { response: { status: 429 } };
    assert.equal(options.headers['x-goog-api-key'], 'personal-gemini-secret');
    return { data: { embedding: { values: Array(3072).fill(0.2) } } };
  });
  fixture.dependencies.generateGeminiEmbedding = require('../src/utils/geminiClient').generateGeminiEmbedding;
  const res = response();
  await createPageSemanticRebuildHandler(fixture.dependencies)({ body: { id_page: 123, description }, query: { manga: 'one-piece' }, headers: { 'x-gemini-api-key': 'personal-gemini-secret' } }, res);
  assert.equal(res.statusCode, 200);
  assert.equal(res.body.gemini_key_source, 'user');
  assert.equal(read.mock.callCount(), 1);
  assert.equal(post.mock.callCount(), 2);
  assert.equal(fixture.events.filter((event) => event[0] === 'voyage').length, 1);
  assert.equal(fixture.events.filter((event) => event === 'update').length, 1);
  assert.ok(!JSON.stringify(fixture.getWritten()).includes('gemini-secret'));
});

test('rejects malformed Gemini headers before processing a page without disclosing the key', async () => {
  for (const value of ['short', 'secret\nheader', ['first-key', 'second-key'], 'secret'.repeat(60)]) {
    const fixture = setup();
    const res = response();
    await createPageSemanticRebuildHandler(fixture.dependencies)({ body: { id_page: 123, description }, query: { manga: 'one-piece' }, headers: { 'x-gemini-api-key': value } }, res);
    assert.equal(res.statusCode, 400);
    assert.equal(res.body.code, 'INVALID_GEMINI_API_KEY');
    assert.deepEqual(fixture.events, []);
    assert.ok(!res.body.error.includes('secret'));
  }
  assert.equal(readUserGeminiApiKey({ headers: {} }), undefined);
});

test('does not attempt a Gemini fallback on a Voyage failure', async () => {
  const fixture = setup();
  fixture.dependencies.generateVoyageEmbedding = async () => { throw { response: { status: 429 } }; };
  fixture.dependencies.generateGeminiEmbedding = async () => { throw new Error('Gemini must not be called'); };
  const res = response();
  await createPageSemanticRebuildHandler(fixture.dependencies)({ body: { id_page: 123, description }, query: { manga: 'one-piece' }, headers: { 'x-gemini-api-key': 'personal-gemini-key' } }, res);
  assert.equal(res.statusCode, 429);
  assert.equal(res.body.code, 'EMBEDDING_QUOTA_EXCEEDED');
  assert.equal(fixture.getWritten(), undefined);
});

for (const [error, status, code, message] of [
  [{ code: 'GEMINI_USER_KEY_REJECTED' }, 503, 'GEMINI_USER_KEY_REJECTED', 'personnelle'],
  [{ response: { status: 429 }, geminiKeySource: 'user' }, 429, 'EMBEDDING_QUOTA_EXCEEDED', 'personnelle'],
  [{ response: { status: 429 }, geminiKeySource: 'server' }, 429, 'EMBEDDING_QUOTA_EXCEEDED', 'profil'],
]) {
  test(`reports ${code} safely for key source ${error.geminiKeySource || 'user'}`, async () => {
    const fixture = setup();
    fixture.dependencies.generateGeminiEmbedding = async () => { throw { ...error, message: 'private-google-key' }; };
    const res = response();
    await createPageSemanticRebuildHandler(fixture.dependencies)({ body: { id_page: 123, description }, query: { manga: 'one-piece' } }, res);
    assert.equal(res.statusCode, status);
    assert.equal(res.body.code, code);
    assert.ok(res.body.error.includes(message));
    assert.ok(!JSON.stringify(res.body).includes('private-google-key'));
    assert.equal(fixture.getWritten(), undefined);
  });
}

test('migration adds resumable provenance without clearing any existing semantic data', () => {
  const source = fs.readFileSync(path.join(__dirname, '../sql/2026-10-01_astra_semantic_rebuild.sql'), 'utf8');
  assert.match(source, /add column if not exists description_model/);
  assert.match(source, /p\.description_prompt_version/);
  assert.match(source, /revoke all on function public\.get_ai_embedding_stats\(text\) from public, anon, authenticated/);
  assert.doesNotMatch(source, /update public\.pages\s+set/i);
  assert.doesNotMatch(source, /new\.embedding_f2llm/);
  const upgrade = fs.readFileSync(path.join(__dirname, '../sql/2026-10-01_astra_rebuild_voyage_gemini.sql'), 'utf8');
  assert.match(upgrade, /create or replace function public\.invalidate_page_description_provenance/i);
  assert.doesNotMatch(upgrade, /new\.embedding_f2llm/);
});
