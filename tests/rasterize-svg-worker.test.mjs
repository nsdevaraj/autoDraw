import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { rasterizePolylines } from '../src/sketch-rasterizer.mjs';
import { svgToPolylines } from '../src/svg-to-polylines.mjs';

const WORKER = fileURLToPath(new URL('../scripts/rasterize-svg-worker.mjs', import.meta.url));
const BITMAP_BYTES = 64 * 64;
const STATUS_OK = 0;
const STATUS_UNREADABLE = 1;
const STATUS_EMPTY = 2;
const STATUS_INK_OUT_OF_BAND = 3;

const HOUSE = '<svg viewBox="0 0 64 64"><path d="M8 32 L32 12 L56 32 L56 56 L8 56 Z"/></svg>';
const RING = '<svg viewBox="0 0 64 64"><circle cx="32" cy="32" r="20"/></svg>';

function encode(source) {
  const payload = Buffer.from(source, 'utf8');
  const header = Buffer.alloc(4);
  header.writeUInt32LE(payload.length, 0);
  return Buffer.concat([header, payload]);
}

function runWorker(sources, args = []) {
  const result = spawnSync(process.execPath, [WORKER, ...args], {
    input: Buffer.concat(sources.map(encode)),
    maxBuffer: 16 * 1024 * 1024,
  });
  return result;
}

function readResponses(stdout) {
  const responses = [];
  let offset = 0;
  while (offset < stdout.length) {
    const status = stdout[offset];
    offset += 1;
    if (status !== STATUS_OK) {
      responses.push({ status, bitmap: null });
      continue;
    }
    responses.push({ status, bitmap: stdout.subarray(offset, offset + BITMAP_BYTES) });
    offset += BITMAP_BYTES;
  }
  return responses;
}

test('worker rasterizes SVG sources with in-process parity', () => {
  const result = runWorker([HOUSE, RING]);

  assert.equal(result.status, 0, result.stderr.toString());
  const responses = readResponses(result.stdout);
  assert.equal(responses.length, 2);
  for (const [index, source] of [HOUSE, RING].entries()) {
    assert.equal(responses[index].status, STATUS_OK);
    assert.deepEqual(
      responses[index].bitmap,
      Buffer.from(rasterizePolylines(svgToPolylines(source))),
    );
  }
  assert.equal(result.stderr.length, 0);
});

test('unreadable sources yield a status byte instead of a bitmap', () => {
  const result = runWorker(['<svg><path d="M0 0 X9 9"/></svg>']);

  assert.equal(result.status, 0, result.stderr.toString());
  assert.deepEqual(readResponses(result.stdout), [{ status: STATUS_UNREADABLE, bitmap: null }]);
});

test('documents without geometry report the empty status', () => {
  const result = runWorker(['<svg><rect width="100%" height="100%"/></svg>']);

  assert.equal(result.status, 0, result.stderr.toString());
  assert.deepEqual(readResponses(result.stdout), [{ status: STATUS_EMPTY, bitmap: null }]);
});

test('the ink coverage band rejects rasters that are too sparse or too dense', () => {
  const sparse = runWorker([HOUSE], ['--min-ink', '0.99', '--max-ink', '1']);
  assert.equal(sparse.status, 0, sparse.stderr.toString());
  assert.deepEqual(readResponses(sparse.stdout), [{ status: STATUS_INK_OUT_OF_BAND, bitmap: null }]);

  const dense = runWorker([HOUSE], ['--min-ink', '0', '--max-ink', '0.001']);
  assert.equal(dense.status, 0, dense.stderr.toString());
  assert.deepEqual(readResponses(dense.stdout), [{ status: STATUS_INK_OUT_OF_BAND, bitmap: null }]);
});

test('a rejected icon does not desynchronize the responses that follow it', () => {
  const result = runWorker(['<svg><path d="M0 0 X9 9"/></svg>', HOUSE]);

  assert.equal(result.status, 0, result.stderr.toString());
  const responses = readResponses(result.stdout);
  assert.equal(responses.length, 2);
  assert.equal(responses[0].status, STATUS_UNREADABLE);
  assert.deepEqual(responses[1].bitmap, Buffer.from(rasterizePolylines(svgToPolylines(HOUSE))));
});

test('worker reports a truncated record without emitting a partial bitmap', () => {
  const result = spawnSync(process.execPath, [WORKER], { input: encode(HOUSE).subarray(0, 8) });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr.toString(), /Unexpected end of input/);
  assert.equal(result.stdout.length, 0);
});

test('worker validates raster options and the ink band', () => {
  const size = spawnSync(process.execPath, [WORKER, '--size', '1'], { input: encode(HOUSE) });
  assert.notEqual(size.status, 0);
  assert.match(size.stderr.toString(), /Raster size/);

  const band = spawnSync(process.execPath, [WORKER, '--min-ink', '0.9', '--max-ink', '0.1'], {
    input: encode(HOUSE),
  });
  assert.notEqual(band.status, 0);
  assert.match(band.stderr.toString(), /Ink coverage band/);
});
