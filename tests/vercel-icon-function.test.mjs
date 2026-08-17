import assert from 'node:assert/strict';
import test from 'node:test';

import { createApprovedIconProxy } from '../src/icon-proxy.mjs';
import { createVercelIconHandler } from '../src/vercel-icon-function.mjs';

const commit = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const path = 'Animals/line/cat.svg';
const pinnedUrl = `https://cdn.jsdelivr.net/gh/example/icons@${commit}/${path}`;
const manifest = {
  schemaVersion: 1,
  source: {
    repository: 'https://github.com/example/icons.git',
    commit,
  },
  classes: [{
    name: 'cat',
    candidates: [{ path, url: pinnedUrl }],
  }],
};

test('Vercel icon Function serves approved SVGs with browser and CDN security headers', async () => {
  const calls = [];
  const handler = createVercelIconHandler({
    manifest,
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return new Response('<svg xmlns="http://www.w3.org/2000/svg"/>', {
        headers: { 'Content-Type': 'image/svg+xml' },
      });
    },
  });

  const response = await handler(new Request(
    `https://autodraw.example/api/icon?path=${encodeURIComponent(path)}`,
  ));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-type'), 'image/svg+xml');
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
  assert.equal(response.headers.get('cross-origin-resource-policy'), 'same-origin');
  assert.match(response.headers.get('content-security-policy'), /default-src 'none'/);
  assert.match(response.headers.get('cache-control'), /max-age=86400/);
  assert.match(response.headers.get('vercel-cdn-cache-control'), /max-age=31536000/);
  assert.equal(await response.text(), '<svg xmlns="http://www.w3.org/2000/svg"/>');
  assert.deepEqual(calls, [{ url: pinnedUrl, options: { redirect: 'error' } }]);

  const cached = await handler(new Request(
    `https://autodraw.example/api/icon?path=${encodeURIComponent(path)}`,
  ));
  assert.equal(cached.status, 200);
  assert.equal(calls.length, 1);
});

test('Vercel icon Function handles HEAD, methods, traversal, and unknown paths', async () => {
  const handler = createVercelIconHandler({
    manifest,
    fetchImpl: async () => new Response('<svg/>', {
      headers: { 'Content-Type': 'image/svg+xml' },
    }),
  });

  const head = await handler(new Request(
    `https://autodraw.example/api/icon?path=${encodeURIComponent(path)}`,
    { method: 'HEAD' },
  ));
  assert.equal(head.status, 200);
  assert.equal(await head.text(), '');

  const responses = await Promise.all([
    handler(new Request('https://autodraw.example/api/icon', { method: 'POST' })),
    handler(new Request('https://autodraw.example/api/icon?path=../secret.svg')),
    handler(new Request('https://autodraw.example/api/icon?path=Animals/line/notes.txt')),
    // Any confined SVG under the pinned commit is serveable, not only manifest candidates.
    handler(new Request('https://autodraw.example/api/icon?path=Animals/line/missing.svg')),
  ]);
  assert.deepEqual(responses.map(response => response.status), [405, 404, 404, 200]);
  assert.equal(responses[0].headers.get('allow'), 'GET, HEAD');
});

test('Vercel icon Function rejects invalid upstream SVG responses without caching failures', async () => {
  let calls = 0;
  const handler = createVercelIconHandler({
    manifest,
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) return new Response('not svg', {
        headers: { 'Content-Type': 'image/svg+xml' },
      });
      return new Response('<svg/>', {
        headers: { 'Content-Type': 'image/svg+xml' },
      });
    },
  });

  const failed = await handler(new Request(
    `https://autodraw.example/api/icon?path=${encodeURIComponent(path)}`,
  ));
  const retried = await handler(new Request(
    `https://autodraw.example/api/icon?path=${encodeURIComponent(path)}`,
  ));
  assert.equal(failed.status, 502);
  assert.equal(retried.status, 200);
  assert.equal(calls, 2);
});

test('deployed API module exposes Vercel GET and HEAD handlers', async () => {
  const api = await import('../api/icon.mjs');
  assert.equal(typeof api.GET, 'function');
  assert.equal(typeof api.HEAD, 'function');
});

test('approved icon proxy evicts least-recently-used bodies by byte budget', async () => {
  const candidates = ['alpha', 'beta', 'gamma'].map(name => ({
    path: `Animals/line/${name}.svg`,
    url: `https://cdn.jsdelivr.net/gh/example/icons@${commit}/Animals/line/${name}.svg`,
  }));
  const calls = new Map();
  const proxy = createApprovedIconProxy({
    manifest: {
      ...manifest,
      classes: [{ name: 'cat', candidates }],
    },
    maximumCacheBytes: 12,
    fetchImpl: async url => {
      const name = new URL(url).pathname.split('/').at(-1).replace('.svg', '');
      calls.set(name, (calls.get(name) ?? 0) + 1);
      return new Response('<svg/>', {
        headers: { 'Content-Type': 'image/svg+xml' },
      });
    },
  });

  await proxy.icon(candidates[0].path);
  await proxy.icon(candidates[1].path);
  await proxy.icon(candidates[0].path);
  await proxy.icon(candidates[2].path);
  await proxy.icon(candidates[1].path);

  assert.deepEqual(Object.fromEntries(calls), {
    alpha: 1,
    beta: 2,
    gamma: 1,
  });
});

test('approved icon proxy bounds concurrent fetches for distinct icons', async () => {
  const candidates = ['alpha', 'beta', 'gamma'].map(name => ({
    path: `Animals/line/${name}.svg`,
    url: `https://cdn.jsdelivr.net/gh/example/icons@${commit}/Animals/line/${name}.svg`,
  }));
  const releases = [];
  let active = 0;
  let maximumActive = 0;
  const proxy = createApprovedIconProxy({
    manifest: { ...manifest, classes: [{ name: 'cat', candidates }] },
    maximumConcurrentFetches: 2,
    fetchImpl: async () => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      await new Promise(resolve => releases.push(resolve));
      active -= 1;
      return new Response('<svg/>', {
        headers: { 'Content-Type': 'image/svg+xml' },
      });
    },
  });

  const requests = candidates.map(candidate => proxy.icon(candidate.path));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(active, 2);
  assert.equal(releases.length, 2);
  releases.shift()();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(maximumActive, 2);
  assert.equal(releases.length, 2);
  releases.shift()();
  releases.shift()();
  await Promise.all(requests);
  assert.equal(maximumActive, 2);
});