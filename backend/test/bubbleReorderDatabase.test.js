const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

// Optional embedded PostgreSQL runner: point PGLITE_MODULE_PATH at @electric-sql/pglite.
test('review reordering enforces staff access and persists a complete order atomically', {
  skip: !process.env.PGLITE_MODULE_PATH,
}, async () => {
  const { PGlite } = require(process.env.PGLITE_MODULE_PATH);
  const db = new PGlite();
  const admin = '00000000-0000-0000-0000-000000000001';
  const modo = '00000000-0000-0000-0000-000000000002';
  const user = '00000000-0000-0000-0000-000000000003';
  try {
    await db.exec(`
      create role anon; create role authenticated; create role service_role;
      create type public.role_type as enum ('Admin', 'Modo', 'User');
      create type public.page_status as enum ('not_started', 'in_progress', 'pending_review', 'completed');
      create type public.statut_bulle as enum ('Proposé', 'Validé');
      create table public.profiles (id uuid primary key, role public.role_type not null);
      create table public.pages (id bigint primary key, statut public.page_status not null);
      create table public.bulles (id bigint primary key, id_page bigint, id_user_createur uuid, statut public.statut_bulle, "order" integer);
      create function public.reorder_bubbles(jsonb) returns void language sql as 'select';
      insert into public.profiles values ('${admin}', 'Admin'), ('${modo}', 'Modo'), ('${user}', 'User');
      insert into public.pages values (1479, 'pending_review');
      insert into public.bulles values (1, 1479, '${user}', 'Proposé', 1), (2, 1479, '${user}', 'Proposé', 2);
    `);
    const reorder = actor => db.query('select public.reorder_page_bubbles($1, $2, $3)', [1479, actor, JSON.stringify([{ id: 2, order: 1 }, { id: 1, order: 2 }])]);
    await db.exec(fs.readFileSync(path.join(__dirname, '../sql/2026-08-01_secure_bubble_reordering.sql'), 'utf8'));
    await assert.rejects(reorder(admin), error => error.code === '42501');
    await db.exec(fs.readFileSync(path.join(__dirname, '../sql/2026-09-06_review_bubble_order.sql'), 'utf8'));
    await reorder(admin);
    await reorder(modo);
    assert.deepEqual((await db.query('select id, "order" from public.bulles order by "order"')).rows, [{ id: 2, order: 1 }, { id: 1, order: 2 }]);
    assert.equal((await db.query('select statut from public.pages where id=1479')).rows[0].statut, 'pending_review');
    await assert.rejects(reorder(user), error => error.code === '42501');
    await assert.rejects(db.query('select public.reorder_page_bubbles($1, $2, $3)', [1479, admin, JSON.stringify([{ id: 1, order: 1 }])]), error => error.code === '22023');
    await db.exec("update public.pages set statut='completed'");
    await assert.rejects(reorder(admin), error => error.code === '42501');
    await assert.rejects(reorder(modo), error => error.code === '42501');
    await db.exec("update public.pages set statut='in_progress'");
    await reorder(user);
    await db.exec("update public.bulles set statut='Validé' where id=1");
    await assert.rejects(reorder(user), error => error.code === '42501');
    await reorder(modo);
    assert.deepEqual((await db.query('select id, "order" from public.bulles order by "order"')).rows, [{ id: 2, order: 1 }, { id: 1, order: 2 }]);
  } finally { await db.close(); }
});
