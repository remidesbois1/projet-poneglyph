const assert = require('node:assert/strict');
const test = require('node:test');

const configPath = require.resolve('../src/config/supabaseClient');
const previousConfig = require.cache[configPath];
require.cache[configPath] = { id: configPath, filename: configPath, loaded: true, exports: { supabase: {}, supabaseAdmin: {} } };
const { createAuthMiddleware, roleCheck } = require('../src/middleware/auth');
if (previousConfig) require.cache[configPath] = previousConfig;
else delete require.cache[configPath];

function fixture({ userError, profileError, profile = { role: 'Admin' }, throws, optional = false } = {}) {
  const calls = [];
  const authClient = { auth: { async getUser(token) {
    calls.push(['getUser', token]);
    if (throws) throw throws;
    return { data: { user: userError ? null : { id: 'verified-admin' } }, error: userError };
  } } };
  const profileClient = { from(table) {
    calls.push(['from', table]);
    const query = {
      select(column) { calls.push(['select', column]); return query; },
      eq(column, value) { calls.push(['eq', column, value]); return query; },
      async maybeSingle() { return { data: profile, error: profileError }; },
    };
    return query;
  } };
  const req = { headers: { authorization: 'Bearer access-token' } };
  const res = { statusCode: null, body: null, status(value) { this.statusCode = value; return this; }, json(value) { this.body = value; return this; } };
  let nextCalls = 0;
  const next = () => { nextCalls++; };
  return { req, res, next, calls, nextCalls: () => nextCalls, run: createAuthMiddleware({ authClient, profileClient, optional }) };
}

test('verifies the bearer token before reading only the verified user role', async () => {
  const f = fixture();
  await f.run(f.req, f.res, f.next);
  assert.equal(f.nextCalls(), 1);
  assert.deepEqual(f.req.user, { id: 'verified-admin', role: 'Admin' });
  assert.deepEqual(f.calls, [['getUser', 'access-token'], ['from', 'profiles'], ['select', 'role'], ['eq', 'id', 'verified-admin']]);
  roleCheck(['Admin'])(f.req, f.res, f.next);
  assert.equal(f.nextCalls(), 2);
});

test('rejects missing or invalid tokens with a retryable authentication code and no role lookup', async () => {
  for (const missing of [false, true]) {
    const f = fixture({ userError: { status: 401, code: 'bad_jwt' } });
    if (missing) f.req.headers.authorization = undefined;
    await f.run(f.req, f.res, f.next);
    assert.equal(f.res.statusCode, 401);
    assert.equal(f.res.body.code, 'SUPABASE_AUTH_REQUIRED');
    assert.equal(f.nextCalls(), 0);
    assert.ok(!f.calls.some(([name]) => name === 'from'));
  }
});

test('Supabase auth outages return 503 rather than an expired-session message', async () => {
  for (const options of [
    { userError: { status: 503, name: 'AuthRetryableFetchError' } },
    { userError: { status: 429 } },
    { throws: new TypeError('Failed to fetch bearer-secret') },
  ]) {
    const f = fixture(options);
    await f.run(f.req, f.res, f.next);
    assert.equal(f.res.statusCode, 503);
    assert.equal(f.res.body.code, 'SUPABASE_UNAVAILABLE');
    assert.ok(!JSON.stringify(f.res.body).includes('bearer-secret'));
    assert.equal(f.nextCalls(), 0);
  }
});

test('a failed profile read is an availability failure, with no access to the handler', async () => {
  const f = fixture({ profileError: { code: 'PGRST000', message: 'Database connection failed' } });
  await f.run(f.req, f.res, f.next);
  assert.equal(f.res.statusCode, 503);
  assert.equal(f.res.body.code, 'SUPABASE_UNAVAILABLE');
  assert.equal(f.nextCalls(), 0);
});

test('missing profiles and insufficient admin permissions remain forbidden', async () => {
  const missing = fixture({ profile: null });
  await missing.run(missing.req, missing.res, missing.next);
  assert.equal(missing.res.statusCode, 403);
  assert.equal(missing.res.body.code, 'SUPABASE_PERMISSION_DENIED');
  assert.equal(missing.nextCalls(), 0);
  const nonAdmin = fixture({ profile: { role: 'User' } });
  await nonAdmin.run(nonAdmin.req, nonAdmin.res, nonAdmin.next);
  roleCheck(['Admin'])(nonAdmin.req, nonAdmin.res, nonAdmin.next);
  assert.equal(nonAdmin.res.statusCode, 403);
  assert.equal(nonAdmin.nextCalls(), 1);
});

test('optional authentication keeps anonymous readers and validates authenticated readers without a role query', async () => {
  const f = fixture({ optional: true });
  f.req.headers.authorization = undefined;
  await f.run(f.req, f.res, f.next);
  assert.equal(f.nextCalls(), 1);
  assert.equal(f.req.user, undefined);
  f.req.headers.authorization = 'Bearer access-token';
  await f.run(f.req, f.res, f.next);
  assert.deepEqual(f.req.user, { id: 'verified-admin' });
  assert.deepEqual(f.calls, [['getUser', 'access-token']]);
});
