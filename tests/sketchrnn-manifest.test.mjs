import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { parseLineList } from '../src/line-list.mjs';

const PROJECT_ROOT = new URL('../', import.meta.url);
const classNames = parseLineList(readFileSync(new URL('config/mvp-classes.txt', PROJECT_ROOT), 'utf8'));
const manifest = JSON.parse(readFileSync(new URL('data/sketchrnn-mvp-manifest.json', PROJECT_ROOT), 'utf8'));

test('the tracked Sketch-RNN manifest covers the exact MVP dataset', () => {
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.source.format, 'Sketch-RNN stroke-3 NPZ');
  assert.equal(manifest.source.license, 'CC BY 4.0');
  assert.equal(manifest.source.licenseUrl, 'https://creativecommons.org/licenses/by/4.0/');
  assert.equal(manifest.source.creator, 'Google, Inc.');
  assert.equal(
    manifest.source.attribution,
    'Quick, Draw! Dataset by Google, Inc., licensed under CC BY 4.0.',
  );
  assert.equal(
    manifest.source.documentation,
    'https://github.com/googlecreativelab/quickdraw-dataset',
  );
  assert.deepEqual(manifest.classes.map(item => item.name), classNames);
  assert.equal(manifest.classes.length, 40);
  assert.equal(new Set(manifest.classes.map(item => item.sha256)).size, 40);
  assert.equal(manifest.classes.reduce((sum, item) => sum + item.bytes, 0), 438159085);

  let drawingCount = 0;
  for (const item of manifest.classes) {
    assert.match(item.sha256, /^[a-f0-9]{64}$/);
    assert.deepEqual(item.dtypes, ['int16']);
    assert.deepEqual(item.penStates, [0, 1]);
    for (const [split, count] of Object.entries({ train: 70000, valid: 2500, test: 2500 })) {
      assert.equal(item.splits[split].count, count, `${item.name}/${split}`);
      assert.equal(item.splits[split].minSequenceLength > 0, true, `${item.name}/${split}`);
      assert.equal(item.splits[split].maxSequenceLength <= 250, true, `${item.name}/${split}`);
      drawingCount += count;
    }
  }
  assert.equal(drawingCount, 3000000);
});

test('the Sketch-RNN manifest fingerprint covers all manifest metadata', () => {
  assert.match(manifest.fingerprint, /^[a-f0-9]{64}$/);
  const { fingerprint, ...content } = manifest;
  assert.equal(
    fingerprint,
    createHash('sha256').update(JSON.stringify(content)).digest('hex'),
  );
});
