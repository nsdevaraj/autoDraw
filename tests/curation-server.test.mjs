import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { realpath, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createCurationHandler, resolveServedFile } from '../src/curation-server.mjs';

test('the curation server only resolves explicitly served paths', async () => {
  const root = mkdtempSync(join(tmpdir(), 'curation-server-'));
  mkdirSync(join(root, 'data'));
  mkdirSync(join(root, 'models', 'quickdraw-mvp'), { recursive: true });
  mkdirSync(join(root, 'node_modules', 'onnxruntime-web', 'dist'), { recursive: true });
  mkdirSync(join(root, 'src'));
  mkdirSync(join(root, '.cache', 'svgdepot', 'Animals', 'line'), { recursive: true });
  writeFileSync(join(root, 'index.html'), '<!doctype html>');
  writeFileSync(join(root, 'curate.html'), '<!doctype html>');
  writeFileSync(join(root, 'data', 'quickdraw-candidates.json'), '{}');
  writeFileSync(join(root, 'models', 'quickdraw-mvp', 'quickdraw-mvp.onnx'), 'model');
  writeFileSync(join(root, 'node_modules', 'onnxruntime-web', 'dist', 'ort.wasm.min.mjs'), 'export default {}');
  writeFileSync(join(root, 'src', 'drawing-app.mjs'), '');
  writeFileSync(join(root, '.cache', 'svgdepot', 'Animals', 'line', 'cat.svg'), '<svg/>');

  try {
    assert.equal(await resolveServedFile(root, '/'), await realpath(join(root, 'index.html')));
    assert.equal(
      await resolveServedFile(root, '/curate.html'),
      await realpath(join(root, 'curate.html')),
    );
    assert.equal(
      await resolveServedFile(root, '/data/quickdraw-candidates.json'),
      await realpath(join(root, 'data', 'quickdraw-candidates.json')),
    );
    assert.equal(
      await resolveServedFile(root, '/models/quickdraw-mvp/quickdraw-mvp.onnx'),
      await realpath(join(root, 'models', 'quickdraw-mvp', 'quickdraw-mvp.onnx')),
    );
    assert.equal(
      await resolveServedFile(root, '/vendor/ort.wasm.min.mjs'),
      await realpath(join(root, 'node_modules', 'onnxruntime-web', 'dist', 'ort.wasm.min.mjs')),
    );
    assert.equal(
      await resolveServedFile(root, '/svgdepot/Animals/line/cat.svg'),
      await realpath(join(root, '.cache', 'svgdepot', 'Animals', 'line', 'cat.svg')),
    );
    await assert.rejects(
      () => resolveServedFile(root, '/svgdepot/../.git/config.svg'),
      /Not found/,
    );
    await assert.rejects(() => resolveServedFile(root, '/.git/config'), /Not found/);
    await assert.rejects(() => resolveServedFile(root, '/icon.html'), /Not found/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('the curation server rejects an allowed path that resolves outside the project root', async () => {
  const root = mkdtempSync(join(tmpdir(), 'curation-server-root-'));
  const outside = mkdtempSync(join(tmpdir(), 'curation-server-outside-'));
  mkdirSync(join(root, 'data'));
  mkdirSync(join(root, '.cache', 'svgdepot', 'Animals'), { recursive: true });
  writeFileSync(join(outside, 'secret.json'), '{"secret":true}');
  symlinkSync(join(outside, 'secret.json'), join(root, 'data', 'quickdraw-candidates.json'));
  symlinkSync(join(outside, 'secret.json'), join(root, '.cache', 'svgdepot', 'Animals', 'cat.svg'));

  try {
    await assert.rejects(
      () => resolveServedFile(root, '/data/quickdraw-candidates.json'),
      /Forbidden/,
    );
    await assert.rejects(
      () => resolveServedFile(root, '/svgdepot/Animals/cat.svg'),
      /Forbidden/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
    await rm(outside, { recursive: true, force: true });
  }
});

test('the curation server secures successful and error responses', async () => {
  const root = mkdtempSync(join(tmpdir(), 'curation-server-http-'));
  const outside = mkdtempSync(join(tmpdir(), 'curation-server-http-outside-'));
  mkdirSync(join(root, 'data'));
  writeFileSync(join(root, 'index.html'), '<!doctype html>');
  writeFileSync(join(outside, 'secret.json'), '{"secret":true}');
  symlinkSync(join(outside, 'secret.json'), join(root, 'data', 'quickdraw-candidates.json'));
  const server = createServer(createCurationHandler(root));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const { port } = server.address();

  try {
    const responses = await Promise.all([
      fetch(`http://127.0.0.1:${port}/`),
      fetch(`http://127.0.0.1:${port}/missing`),
      fetch(`http://127.0.0.1:${port}/data/quickdraw-candidates.json`),
      fetch(`http://127.0.0.1:${port}/`, { method: 'POST' }),
    ]);
    assert.deepEqual(responses.map(response => response.status), [200, 404, 403, 405]);

    for (const response of responses) {
      assert.equal(response.headers.get('cache-control'), 'no-store');
      assert.match(response.headers.get('content-security-policy'), /default-src 'self'/);
      assert.match(response.headers.get('content-security-policy'), /script-src 'self' 'wasm-unsafe-eval'/);
      assert.doesNotMatch(response.headers.get('content-security-policy'), /script-src[^;]*'unsafe-eval'/);
      assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
      assert.match(response.headers.get('content-type'), /^text\//);
    }
    assert.equal(responses[3].headers.get('allow'), 'GET, HEAD');
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
    await rm(outside, { recursive: true, force: true });
  }
});

test('the application server sends browser runtime binaries with exact content types', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-server-binaries-'));
  mkdirSync(join(root, 'models', 'quickdraw-mvp'), { recursive: true });
  mkdirSync(join(root, 'node_modules', 'onnxruntime-web', 'dist'), { recursive: true });
  writeFileSync(join(root, 'models', 'quickdraw-mvp', 'quickdraw-mvp.onnx'), 'model');
  writeFileSync(
    join(root, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.wasm'),
    'wasm',
  );
  const server = createServer(createCurationHandler(root));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const { port } = server.address();

  try {
    const responses = await Promise.all([
      fetch(`http://127.0.0.1:${port}/models/quickdraw-mvp/quickdraw-mvp.onnx`),
      fetch(`http://127.0.0.1:${port}/vendor/ort-wasm-simd-threaded.wasm`),
    ]);
    assert.deepEqual(responses.map(response => response.status), [200, 200]);
    assert.deepEqual(
      responses.map(response => response.headers.get('content-type')),
      ['application/octet-stream', 'application/wasm'],
    );
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
});

test('missing approved local icons proxy their pinned manifest SVG', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-server-icon-fallback-'));
  mkdirSync(join(root, 'data'));
  const approvedUrl = 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/cat.svg';
  writeFileSync(join(root, 'data', 'quickdraw-candidates.json'), JSON.stringify({
    schemaVersion: 1,
    fingerprint: 'manifest',
    source: {
      repository: 'https://github.com/example/icons.git',
      commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    },
    classes: [{
      name: 'cat',
      candidates: [{ path: 'Animals/line/cat.svg', url: approvedUrl }],
    }],
  }));
  const requests = [];
  const server = createServer(createCurationHandler(root, {
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return new Response('<svg xmlns="http://www.w3.org/2000/svg"/>', {
        status: 200,
        headers: { 'Content-Type': 'image/svg+xml' },
      });
    },
  }));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const { port } = server.address();

  try {
    const approved = await fetch(
      `http://127.0.0.1:${port}/svgdepot/Animals/line/cat.svg`,
    );
    // Retrieval reaches the whole pinned commit, so a path the manifest never listed is
    // still proxied, rebuilt to the same commit-pinned URL shape.
    const unlisted = await fetch(
      `http://127.0.0.1:${port}/svgdepot/Animals/line/unknown.svg`,
      { redirect: 'manual' },
    );
    const traversal = await fetch(
      `http://127.0.0.1:${port}/svgdepot/../secret.svg`,
      { redirect: 'manual' },
    );
    assert.equal(approved.status, 200);
    assert.equal(approved.headers.get('content-type'), 'image/svg+xml');
    assert.equal(await approved.text(), '<svg xmlns="http://www.w3.org/2000/svg"/>');
    assert.deepEqual(requests[0], {
      url: approvedUrl,
      options: { redirect: 'error' },
    });
    const cached = await fetch(`http://127.0.0.1:${port}/svgdepot/Animals/line/cat.svg`);
    assert.equal(cached.status, 200);
    assert.equal(
      requests.filter(request => request.url === approvedUrl).length,
      1,
      'the approved icon is fetched once and then cached',
    );

    assert.equal(unlisted.status, 200);
    assert.equal(
      requests.at(-1).url,
      'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/unknown.svg',
    );
    assert.equal(traversal.status, 404);
    assert.equal(
      requests.some(request => request.url.includes('secret')),
      false,
      'a traversing path must never reach the CDN',
    );
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
});

test('approved icon proxy rejects unsafe responses and retries after failure', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-server-icon-rejection-'));
  mkdirSync(join(root, 'data'));
  const commit = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  const candidates = ['mime', 'markup', 'large', 'redirect', 'retry'].map(name => ({
    path: `Animals/line/${name}.svg`,
    url: `https://cdn.jsdelivr.net/gh/example/icons@${commit}/Animals/line/${name}.svg`,
  }));
  writeFileSync(join(root, 'data', 'quickdraw-candidates.json'), JSON.stringify({
    schemaVersion: 1,
    fingerprint: 'manifest',
    source: { repository: 'https://github.com/example/icons.git', commit },
    classes: [{ name: 'cat', candidates }],
  }));
  const calls = new Map();
  const server = createServer(createCurationHandler(root, {
    fetchImpl: async (url, options) => {
      const name = new URL(url).pathname.split('/').at(-1).replace('.svg', '');
      calls.set(name, (calls.get(name) ?? 0) + 1);
      assert.deepEqual(options, { redirect: 'error' });
      if (name === 'mime') return new Response('<svg/>', { headers: { 'Content-Type': 'text/html' } });
      if (name === 'markup') return new Response('not svg', { headers: { 'Content-Type': 'image/svg+xml' } });
      if (name === 'large') return new Response('x'.repeat(2 * 1024 * 1024 + 1), { headers: { 'Content-Type': 'image/svg+xml' } });
      if (name === 'redirect') throw new TypeError('redirect blocked');
      if (name === 'retry' && calls.get(name) === 1) return new Response('bad', { headers: { 'Content-Type': 'image/svg+xml' } });
      return new Response('<svg/>', { headers: { 'Content-Type': 'image/svg+xml' } });
    },
  }));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const { port } = server.address();

  try {
    for (const name of ['mime', 'markup', 'large', 'redirect']) {
      const response = await fetch(`http://127.0.0.1:${port}/svgdepot/Animals/line/${name}.svg`);
      assert.equal(response.status, 404, name);
    }
    const failed = await fetch(`http://127.0.0.1:${port}/svgdepot/Animals/line/retry.svg`);
    const retried = await fetch(`http://127.0.0.1:${port}/svgdepot/Animals/line/retry.svg`);
    assert.equal(failed.status, 404);
    assert.equal(retried.status, 200);
    assert.equal(calls.get('retry'), 2);
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
});