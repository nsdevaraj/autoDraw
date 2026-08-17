import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

import ort from 'onnxruntime-web';

import { parseLineList } from '../src/line-list.mjs';
import { rasterizeStroke3 } from '../src/sketch-rasterizer.mjs';

const MODEL_PATH = 'models/sketch-embedder/sketch-embedder.onnx';
const METADATA_PATH = 'models/sketch-embedder/model.json';
const PYTHON = '.venv-model/bin/python3';
const EMBEDDING_DIM = 64;
const RELEASE_FLOOR = 0.7;

const metadata = JSON.parse(readFileSync(METADATA_PATH, 'utf8'));

test('the tracked embedder metadata is self-consistent and commit-pinned', () => {
  const model = readFileSync(MODEL_PATH);

  assert.equal(metadata.schemaVersion, 1);
  assert.equal(metadata.kind, 'sketch-embedder');
  assert.deepEqual(metadata.classes, parseLineList(readFileSync('config/retrieval-classes.txt', 'utf8')));
  assert.equal(new Set(metadata.classes).size, metadata.classes.length);

  assert.equal(metadata.model.filename, 'sketch-embedder.onnx');
  assert.equal(metadata.model.format, 'ONNX');
  assert.equal(metadata.model.opset, 17);
  assert.equal(metadata.model.bytes, model.length);
  assert.equal(metadata.model.bytes < 3 * 1024 * 1024, true);
  assert.equal(metadata.model.sha256, createHash('sha256').update(model).digest('hex'));
  assert.equal(metadata.model.embeddingDim, EMBEDDING_DIM);

  assert.deepEqual(metadata.model.input.shape, ['batch', 1, 64, 64]);
  assert.equal(metadata.model.input.normalization, 'uint8 / 255');
  assert.deepEqual(metadata.model.outputs.embedding.shape, ['batch', EMBEDDING_DIM]);
  assert.equal(metadata.model.outputs.embedding.normalization, 'L2');
  assert.deepEqual(metadata.model.outputs.logits.shape, ['batch', metadata.classes.length]);

  assert.match(metadata.trainingData.sketchCacheFingerprint, /^[a-f0-9]{64}$/);
  assert.match(metadata.trainingData.iconCacheFingerprint, /^[a-f0-9]{64}$/);
  assert.match(metadata.trainingData.source.commit, /^[a-f0-9]{40}$/);
  assert.match(metadata.fingerprint, /^[a-f0-9]{64}$/);
});

// Python writes the fingerprint over compact JSON, and its float formatting (4.0 vs 4)
// cannot be reproduced by JSON.stringify, so the check runs in the same runtime.
test('the metadata fingerprint covers its own content', { skip: !existsSync(PYTHON) }, () => {
  const result = spawnSync(PYTHON, ['-c', `
import json
import sys
sys.path.insert(0, '.')
from model.integrity import content_fingerprint

metadata = json.load(open('${METADATA_PATH}', encoding='utf-8'))
fingerprint = metadata.pop('fingerprint')
print('match' if fingerprint == content_fingerprint(metadata) else 'mismatch')
`]);

  assert.equal(result.status, 0, result.stderr.toString());
  assert.equal(result.stdout.toString().trim(), 'match');
});

test('the release gates the model shipped under still hold', () => {
  const { classification, retrieval } = metadata.metrics;

  assert.equal(retrieval.all.recallAt10 >= RELEASE_FLOOR, true);
  assert.equal(Math.min(...retrieval.all.perClassRecallAt10) >= RELEASE_FLOOR, true);
  assert.equal(Math.min(...classification.perClassAccuracy) >= RELEASE_FLOOR, true);
  assert.equal(classification.perClassAccuracy.length, metadata.classes.length);
  assert.equal(retrieval.all.perClassRecallAt10.length, metadata.classes.length);

  // Held-out icons stand in for the SVGDepot artwork the model never trains on.
  assert.equal(retrieval.holdoutIcons.recallAt10 >= RELEASE_FLOOR, true);
  assert.equal(metadata.metrics.maxLogitDifference < 1e-4, true);

  assert.equal(metadata.training.minRecallAt10, RELEASE_FLOOR);
  assert.equal(metadata.training.minClassRecallAt10, RELEASE_FLOOR);
  assert.equal(metadata.training.minClassAccuracy, RELEASE_FLOOR);
  assert.equal(metadata.training.deterministicAlgorithms, true);
  assert.deepEqual(metadata.training.weakClasses, { byRecall: [], byClassification: [] });
});

test('the recorded source hashes match the working tree', () => {
  for (const [path, hash] of Object.entries(metadata.source.files)) {
    assert.equal(
      hash,
      createHash('sha256').update(readFileSync(path)).digest('hex'),
      `stale source hash for ${path}`,
    );
  }
});

test('the rasterizer settings match what the browser will feed the model', () => {
  const { rasterizer } = metadata;
  assert.equal(rasterizer.size, 64);
  assert.equal(rasterizer.padding, 4);
  assert.equal(rasterizer.strokeWidth, 2.5);
  assert.equal(rasterizer.supersample, 4);
  assert.equal(rasterizeStroke3([[5, 5, 0], [10, 0, 1]], rasterizer).length, 64 * 64);
});

test('ONNX Runtime Web loads the embedder and returns unit-length embeddings', async () => {
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });

  assert.deepEqual(session.inputNames, ['bitmap']);
  assert.deepEqual(session.outputNames, ['embedding', 'logits']);

  const batch = 3;
  const pixels = new Float32Array(batch * 64 * 64);
  for (let index = 0; index < pixels.length; index += 1) {
    pixels[index] = ((index * 37) % 255) / 255;
  }
  const output = await session.run({
    bitmap: new ort.Tensor('float32', pixels, [batch, 1, 64, 64]),
  });

  assert.deepEqual(output.embedding.dims, [batch, EMBEDDING_DIM]);
  assert.deepEqual(output.logits.dims, [batch, metadata.classes.length]);
  assert.equal(output.embedding.data.every(Number.isFinite), true);

  for (let row = 0; row < batch; row += 1) {
    const vector = output.embedding.data.slice(row * EMBEDDING_DIM, (row + 1) * EMBEDDING_DIM);
    const norm = Math.hypot(...vector);
    assert.equal(Math.abs(norm - 1) < 1e-4, true, `row ${row} norm was ${norm}`);
  }
});

test('a drawn cat embeds nearer to cat sketches than to unrelated shapes', async () => {
  const sample = JSON.parse(readFileSync('tests/fixtures/sketchrnn-cat-test-0.json', 'utf8'));
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });

  const embed = async bitmap => {
    const pixels = Float32Array.from(bitmap, value => value / 255);
    const output = await session.run({
      bitmap: new ort.Tensor('float32', pixels, [1, 1, 64, 64]),
    });
    return { embedding: output.embedding.data, logits: output.logits.data };
  };

  const cat = await embed(rasterizeStroke3(sample, metadata.rasterizer));
  const blank = await embed(new Uint8Array(64 * 64));
  const dot = (left, right) => left.reduce((total, value, index) => total + value * right[index], 0);

  assert.equal(dot(cat.embedding, cat.embedding) > dot(cat.embedding, blank.embedding), true);
  const predicted = metadata.classes[cat.logits.indexOf(Math.max(...cat.logits))];
  assert.equal(predicted, 'cat');
});
