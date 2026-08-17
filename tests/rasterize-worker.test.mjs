import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { rasterizeStroke3 } from '../src/sketch-rasterizer.mjs';

const WORKER = fileURLToPath(new URL('../scripts/rasterize-stroke3-worker.mjs', import.meta.url));

function encodeDrawing(stroke3) {
  const payload = Buffer.alloc(4 + stroke3.length * 6);
  payload.writeUInt32LE(stroke3.length, 0);
  stroke3.forEach((point, pointIndex) => {
    point.forEach((value, valueIndex) => {
      payload.writeInt16LE(value, 4 + pointIndex * 6 + valueIndex * 2);
    });
  });
  return payload;
}

test('binary worker rasterizes multiple stroke-3 records with browser parity', () => {
  const officialSample = JSON.parse(readFileSync(
    new URL('./fixtures/sketchrnn-cat-test-0.json', import.meta.url),
    'utf8',
  ));
  const simpleSample = [[5, 5, 0], [10, 0, 1]];
  const input = Buffer.concat([
    encodeDrawing(officialSample),
    encodeDrawing(simpleSample),
  ]);

  const result = spawnSync(process.execPath, [WORKER], { input });

  assert.equal(result.status, 0, result.stderr.toString());
  assert.equal(result.stdout.length, 2 * 64 * 64);
  assert.deepEqual(
    result.stdout.subarray(0, 64 * 64),
    Buffer.from(rasterizeStroke3(officialSample)),
  );
  assert.deepEqual(
    result.stdout.subarray(64 * 64),
    Buffer.from(rasterizeStroke3(simpleSample)),
  );
  assert.equal(result.stderr.length, 0);
});

test('binary worker reports a truncated record without emitting a partial bitmap', () => {
  const input = encodeDrawing([[1, 2, 1]]).subarray(0, 8);
  const result = spawnSync(process.execPath, [WORKER], { input });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr.toString(), /Unexpected end of input/);
  assert.equal(result.stdout.length, 0);
});

test('binary worker validates raster options', () => {
  const result = spawnSync(process.execPath, [WORKER, '--size', '1'], {
    input: encodeDrawing([[1, 2, 1]]),
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr.toString(), /Raster size/);
});
