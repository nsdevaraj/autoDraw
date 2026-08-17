import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import ort from 'onnxruntime-web';

import { createQuickDrawClassifier } from '../src/quickdraw-classifier.mjs';
import { stroke3ToPolylines } from '../src/sketch-rasterizer.mjs';

const MODEL_PATH = 'models/quickdraw-mvp/quickdraw-mvp.onnx';
const METADATA_PATH = 'models/quickdraw-mvp/model.json';

test('ONNX Runtime Web WASM loads the tracked model with dynamic batching', async () => {
  const packageMetadata = JSON.parse(readFileSync('node_modules/onnxruntime-web/package.json', 'utf8'));
  const lock = JSON.parse(readFileSync('package-lock.json', 'utf8'));
  const lockedPackages = Object.values(lock.packages).filter(entry => entry.resolved);
  assert.equal(packageMetadata.version, '1.22.0');
  assert.equal(lockedPackages.every(entry => new URL(entry.resolved).hostname === 'registry.npmjs.org'), true);
  assert.equal(lockedPackages.every(entry => entry.integrity.startsWith('sha512-')), true);

  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
  });
  const bitmap = new ort.Tensor('float32', new Float32Array(3 * 64 * 64), [3, 1, 64, 64]);
  const output = await session.run({ bitmap });

  assert.deepEqual(session.inputNames, ['bitmap']);
  assert.deepEqual(session.outputNames, ['logits']);
  assert.deepEqual(output.logits.dims, [3, 40]);
  assert.equal(output.logits.data.every(Number.isFinite), true);
});

test('ONNX Runtime Web predicts cat from the shared browser rasterizer', async () => {
  const sample = JSON.parse(readFileSync('tests/fixtures/sketchrnn-cat-test-0.json', 'utf8'));
  const metadata = JSON.parse(readFileSync(METADATA_PATH, 'utf8'));

  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
  });
  const classifier = createQuickDrawClassifier({
    metadata,
    runtime: ort,
    session,
  });
  const [prediction] = await classifier.classify(stroke3ToPolylines(sample));

  assert.equal(prediction.label, 'cat');
  assert.equal(prediction.probability > 0.9, true);
});
