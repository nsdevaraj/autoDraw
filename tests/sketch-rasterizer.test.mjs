import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  quickDrawToPolylines,
  rasterizePolylines,
  rasterizeQuickDraw,
  rasterizeStroke3,
  stroke3ToPolylines,
} from '../src/sketch-rasterizer.mjs';

const OPTIONS = { size: 24, padding: 3, strokeWidth: 2, supersample: 4 };

function inkBounds(bitmap, size) {
  const points = [];
  bitmap.forEach((value, index) => {
    if (value > 0) points.push([index % size, Math.floor(index / size)]);
  });
  if (points.length === 0) return null;
  return {
    left: Math.min(...points.map(point => point[0])),
    top: Math.min(...points.map(point => point[1])),
    right: Math.max(...points.map(point => point[0])),
    bottom: Math.max(...points.map(point => point[1])),
  };
}

test('stroke-3 pen lifts end the current polyline and move before the next one', () => {
  assert.deepEqual(
    stroke3ToPolylines([
      [10, 10, 0],
      [10, 0, 1],
      [0, 10, 0],
      [10, 0, 1],
    ]),
    [
      [[10, 10], [20, 10]],
      [[20, 20], [30, 20]],
    ],
  );
});

test('raw Quick Draw arrays convert to absolute polylines without timing data', () => {
  assert.deepEqual(
    quickDrawToPolylines([
      [[1, 4, 9], [2, 6, 8], [0, 5, 10]],
      [[12, 14], [20, 22]],
    ]),
    [
      [[1, 2], [4, 6], [9, 8]],
      [[12, 20], [14, 22]],
    ],
  );
});

test('equivalent absolute, stroke-3, and Quick Draw inputs rasterize identically', () => {
  const polylines = [
    [[10, 10], [20, 10]],
    [[20, 20], [30, 20]],
  ];
  const stroke3 = [
    [10, 10, 0],
    [10, 0, 1],
    [0, 10, 0],
    [10, 0, 1],
  ];
  const quickDraw = [
    [[10, 20], [10, 10]],
    [[20, 30], [20, 20]],
  ];

  const expected = rasterizePolylines(polylines, OPTIONS);
  assert.deepEqual(rasterizeStroke3(stroke3, OPTIONS), expected);
  assert.deepEqual(rasterizeQuickDraw(quickDraw, OPTIONS), expected);
});

test('normalization makes raster output invariant to translation and uniform scale', () => {
  const original = [[[0, 0], [10, 0], [10, 5]]];
  const transformed = [[[100, -30], [140, -30], [140, -10]]];

  assert.deepEqual(
    rasterizePolylines(transformed, OPTIONS),
    rasterizePolylines(original, OPTIONS),
  );
});

test('normalization preserves a drawing aspect ratio', () => {
  const bitmap = rasterizePolylines([
    [[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]],
  ], { ...OPTIONS, size: 40, padding: 4, strokeWidth: 2 });
  const bounds = inkBounds(bitmap, 40);

  const width = bounds.right - bounds.left + 1;
  const height = bounds.bottom - bounds.top + 1;
  assert.equal(width / height > 1.5, true);
  assert.equal(width / height < 2, true);
});

test('raster output is centered, padded, antialiased, and bounded to bytes', () => {
  const bitmap = rasterizePolylines([[[0, 0], [20, 0]]], OPTIONS);
  const bounds = inkBounds(bitmap, OPTIONS.size);

  assert.equal(bitmap instanceof Uint8Array, true);
  assert.equal(bitmap.length, OPTIONS.size * OPTIONS.size);
  assert.deepEqual(bounds, { left: 3, top: 11, right: 20, bottom: 12 });
  assert.equal(bitmap.some(value => value > 0 && value < 255), true);
  assert.equal(bitmap.every(value => value >= 0 && value <= 255), true);
});

test('pen lifts do not draw a connector between separated strokes', () => {
  const bitmap = rasterizePolylines([
    [[0, 0], [10, 0]],
    [[0, 10], [10, 10]],
  ], { ...OPTIONS, size: 25, padding: 3, strokeWidth: 1 });

  const centerColumn = 12;
  const centerRow = 12;
  assert.equal(bitmap[centerRow * 25 + centerColumn], 0);
});

test('blank drawings return a blank bitmap and invalid coordinates fail clearly', () => {
  assert.deepEqual(rasterizePolylines([], OPTIONS), new Uint8Array(OPTIONS.size ** 2));
  assert.throws(
    () => stroke3ToPolylines([]),
    /must contain at least one point/,
  );
  assert.throws(
    () => stroke3ToPolylines([[1, 2, 0]]),
    /must end with pen state 1/,
  );
  assert.throws(
    () => rasterizePolylines([[[0, 0], [Number.NaN, 2]]], OPTIONS),
    /finite numbers/,
  );
  assert.throws(
    () => stroke3ToPolylines([[1, 2, 2]]),
    /pen state must be 0 or 1/,
  );
  assert.throws(
    () => rasterizePolylines([[[0, 0], ['2', 3]]], OPTIONS),
    /coordinates must be finite numbers/,
  );
  assert.throws(
    () => rasterizePolylines([[[0, 0], [null, 3]]], OPTIONS),
    /coordinates must be finite numbers/,
  );
  assert.throws(
    () => rasterizePolylines([[[-1e308, 0], [1e308, 0]]], OPTIONS),
    /bounds exceed the supported numeric range/,
  );
  assert.throws(
    () => stroke3ToPolylines([[1, 2, '0']]),
    /pen state must be 0 or 1/,
  );
  assert.throws(
    () => rasterizePolylines([[[0, 0, 1]]], OPTIONS),
    /must contain exactly x and y coordinates/,
  );
  assert.throws(
    () => stroke3ToPolylines([[1, 2, 0, 9]]),
    /must contain exactly dx, dy, and pen state/,
  );
  assert.throws(
    () => quickDrawToPolylines([[[1], [2], [3], [4]]]),
    /must contain x and y arrays with optional timing/,
  );
});

test('official Sketch-RNN cat sample has a stable raster', () => {
  const sample = JSON.parse(readFileSync(
    new URL('./fixtures/sketchrnn-cat-test-0.json', import.meta.url),
    'utf8',
  ));
  const bitmap = rasterizeStroke3(sample);

  assert.equal(sample.length, 69);
  assert.equal(stroke3ToPolylines(sample).length, 12);
  assert.equal(
    createHash('sha256').update(bitmap).digest('hex'),
    'dde7369a95c67297a7db18207a82221aeee4895499b9cb9f5751f78fd7072529',
  );
});
