import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

import { parseLineList } from '../src/line-list.mjs';
import { rasterizeStroke3 } from '../src/sketch-rasterizer.mjs';

const PYTHON = '.venv-model/bin/python3';
const MODEL_PATH = 'models/quickdraw-mvp/quickdraw-mvp.onnx';
const METADATA_PATH = 'models/quickdraw-mvp/model.json';
const CACHE_METADATA_PATH = '.cache/quickdraw-training/metadata.json';
const REQUIRE_FULL_VERIFICATION = process.env.AUTODRAW_REQUIRE_MODEL_CACHE === '1';
const HAS_TRAINING_CACHE = existsSync(CACHE_METADATA_PATH);

test('tracked ONNX classifier is compact, accurate, dynamic, and self-consistent', () => {
  const metadata = JSON.parse(readFileSync(METADATA_PATH, 'utf8'));
  const classNames = parseLineList(readFileSync('config/mvp-classes.txt', 'utf8'));
  const model = readFileSync(MODEL_PATH);

  assert.equal(metadata.schemaVersion, 1);
  assert.equal(metadata.kind, 'quickdraw-classifier');
  assert.deepEqual(metadata.classes, classNames);
  assert.match(metadata.sourceCacheFingerprint, /^[a-f0-9]{64}$/);
  const datasetManifest = JSON.parse(readFileSync('data/sketchrnn-mvp-manifest.json', 'utf8'));
  assert.equal(metadata.trainingData.cacheFingerprint, metadata.sourceCacheFingerprint);
  assert.equal(metadata.trainingData.sourceManifestFingerprint, datasetManifest.fingerprint);
  assert.equal(
    metadata.trainingData.sourceManifestSha256,
    createHash('sha256').update(readFileSync('data/sketchrnn-mvp-manifest.json')).digest('hex'),
  );
  assert.deepEqual(metadata.trainingData.sampling.perClass, {
    train: 3000,
    valid: 500,
    test: 500,
  });
  assert.equal(metadata.trainingData.sampling.seed, 20260815);
  assert.equal(
    metadata.trainingData.splits.train.images.sha256,
    'f0b13c42b667dc8b368ebbc24f7d154fdc7b2736128887c2b73cd3d848e4af2c',
  );
  assert.equal(
    metadata.trainingData.splits.test.images.sha256,
    'b08466e72d7494e664ac8f8b2f8b42857ac2e23ded846aa62e19430babad3a0d',
  );
  for (const dependency of Object.values(metadata.trainingData.rasterizerFiles)) {
    assert.equal(
      dependency.sha256,
      createHash('sha256').update(readFileSync(dependency.path)).digest('hex'),
    );
  }
  assert.equal(
    metadata.trainingData.builder.sha256,
    createHash('sha256').update(readFileSync(metadata.trainingData.builder.path)).digest('hex'),
  );
  assert.deepEqual(metadata.trainingData.builder.dependencies, {
    'model/integrity.py': createHash('sha256')
      .update(readFileSync('model/integrity.py'))
      .digest('hex'),
  });
  assert.equal(metadata.trainingData.builder.runtime.numpy, '1.26.4');
  assert.match(metadata.trainingData.node.version, /^v\d+/);
  assert.match(metadata.trainingData.node.sha256, /^[a-f0-9]{64}$/);
  assert.equal(metadata.model.filename, 'quickdraw-mvp.onnx');
  assert.equal(metadata.model.format, 'ONNX');
  assert.equal(metadata.model.opset, 17);
  assert.equal(metadata.model.bytes, model.length);
  assert.equal(metadata.model.bytes < 3 * 1024 * 1024, true);
  assert.equal(metadata.model.sha256, createHash('sha256').update(model).digest('hex'));
  assert.equal(metadata.model.parameterCount, 494760);
  assert.deepEqual(metadata.model.input.shape, ['batch', 1, 64, 64]);
  assert.deepEqual(metadata.model.output.shape, ['batch', 40]);
  assert.equal(metadata.metrics.test.accuracy >= 0.88, true);
  assert.equal(metadata.metrics.test.top5Accuracy >= 0.97, true);
  assert.equal(Math.min(...metadata.metrics.test.perClassAccuracy) >= 0.70, true);
  assert.equal(metadata.metrics.onnxTest.accuracy, metadata.metrics.test.accuracy);
  assert.equal(metadata.metrics.maxLogitDifference < 1e-4, true);
  assert.deepEqual(metadata.training.optimizer, {
    name: 'AdamW',
    learningRate: 0.001,
    weightDecay: 0.0001,
  });
  assert.equal(metadata.training.numWorkers, 0);
  assert.equal(metadata.training.minimumTestAccuracy, 0.7);
  assert.equal(metadata.training.minimumPerClassAccuracy, 0.7);
  assert.equal(metadata.training.deterministicAlgorithms, true);
  assert.equal(metadata.runtime.numpy, '1.26.4');
  assert.match(metadata.runtime.python, /^3\.9\./);
  assert.match(metadata.source.repositoryRevision, /^[a-f0-9]{40}$/);
  assert.equal(metadata.source.repositoryRevision, 'd2786f128e3b1ec07cfee1b4ef14ab797f53c026');
  assert.equal(metadata.source.filesMatchRevision, true);
  const revisionAvailable = spawnSync('git', [
    'cat-file',
    '-e',
    `${metadata.source.repositoryRevision}^{commit}`,
  ]).status === 0;
  for (const [path, hash] of Object.entries(metadata.source.files)) {
    assert.equal(hash, createHash('sha256').update(readFileSync(path)).digest('hex'));
    if (revisionAvailable) {
      const committed = spawnSync('git', ['show', `${metadata.source.repositoryRevision}:${path}`]);
      assert.equal(committed.status, 0, committed.stderr.toString());
      assert.equal(hash, createHash('sha256').update(committed.stdout).digest('hex'));
    }
  }
  assert.deepEqual(metadata.browserRuntime, {
    package: 'onnxruntime-web',
    version: '1.22.0',
    lockfileSha256: createHash('sha256').update(readFileSync('package-lock.json')).digest('hex'),
  });
  assert.match(metadata.fingerprint, /^[a-f0-9]{64}$/);

  const validation = spawnSync(PYTHON, ['-c', `
import hashlib
import json
import numpy as np
import onnx
import onnxruntime as ort
import sys

model_path, metadata_path = sys.argv[1:]
model = onnx.load(model_path)
onnx.checker.check_model(model, full_check=True)
metadata = json.load(open(metadata_path))
fingerprint = metadata.pop('fingerprint')
encoded = json.dumps(metadata, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
logits = session.run(['logits'], {'bitmap': np.zeros((3, 1, 64, 64), dtype=np.float32)})[0]
print(json.dumps({
    'fingerprintMatches': fingerprint == hashlib.sha256(encoded).hexdigest(),
    'inputShape': session.get_inputs()[0].shape,
    'outputShape': session.get_outputs()[0].shape,
    'batchShape': list(logits.shape),
    'finite': bool(np.isfinite(logits).all()),
}))
`, MODEL_PATH, METADATA_PATH], { encoding: 'utf8' });
  assert.equal(validation.status, 0, validation.stderr);
  assert.deepEqual(JSON.parse(validation.stdout), {
    fingerprintMatches: true,
    inputShape: ['batch', 1, 64, 64],
    outputShape: ['batch', 40],
    batchShape: [3, 40],
    finite: true,
  });

});

test('full cache-backed metrics independently match model metadata', {
  skip: !HAS_TRAINING_CACHE && !REQUIRE_FULL_VERIFICATION,
}, () => {
  assert.equal(
    HAS_TRAINING_CACHE,
    true,
    'Build .cache/quickdraw-training before running the full model verification',
  );
  const metadata = JSON.parse(readFileSync(METADATA_PATH, 'utf8'));
  const fullVerification = spawnSync(PYTHON, ['scripts/verify-quickdraw-model.py'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  assert.equal(fullVerification.status, 0, fullVerification.stderr);
  const recomputed = JSON.parse(fullVerification.stdout);
  assert.equal(recomputed.cacheFingerprint, metadata.sourceCacheFingerprint);
  assert.equal(recomputed.modelSha256, metadata.model.sha256);
  assert.equal(Math.abs(recomputed.torch.loss - metadata.metrics.test.loss) < 1e-6, true);
  assert.equal(recomputed.torch.accuracy, metadata.metrics.test.accuracy);
  assert.equal(recomputed.torch.top5Accuracy, metadata.metrics.test.top5Accuracy);
  assert.deepEqual(recomputed.torch.perClassAccuracy, metadata.metrics.test.perClassAccuracy);
  assert.equal(Math.abs(recomputed.onnx.loss - metadata.metrics.onnxTest.loss) < 1e-6, true);
  assert.equal(recomputed.onnx.accuracy, metadata.metrics.onnxTest.accuracy);
  assert.equal(recomputed.onnx.top5Accuracy, metadata.metrics.onnxTest.top5Accuracy);
  assert.deepEqual(recomputed.onnx.perClassAccuracy, metadata.metrics.onnxTest.perClassAccuracy);
  assert.equal(recomputed.maxLogitDifference < 1e-4, true);
  assert.equal(metadata.metrics.maxLogitDifference < 1e-4, true);
});

test('official cat stroke-3 fixture predicts cat through the browser rasterizer', () => {
  const sample = JSON.parse(readFileSync('tests/fixtures/sketchrnn-cat-test-0.json', 'utf8'));
  const bitmap = Buffer.from(rasterizeStroke3(sample));
  const result = spawnSync(PYTHON, ['-c', `
import json
import numpy as np
import onnxruntime as ort
import sys

model_path = sys.argv[1]
classes = json.loads(sys.argv[2])
bitmap = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8).reshape(1, 1, 64, 64)
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
logits = session.run(['logits'], {'bitmap': bitmap.astype(np.float32) / 255})[0][0]
probabilities = np.exp(logits - logits.max())
probabilities /= probabilities.sum()
indices = np.argsort(probabilities)[::-1][:5]
print(json.dumps([{'label': classes[index], 'probability': float(probabilities[index])} for index in indices]))
`, MODEL_PATH, JSON.stringify(parseLineList(readFileSync('config/mvp-classes.txt', 'utf8')))], {
    input: bitmap,
    encoding: 'utf8',
  });

  assert.equal(result.status, 0, result.stderr);
  const predictions = JSON.parse(result.stdout);
  assert.equal(predictions[0].label, 'cat');
  assert.equal(predictions[0].probability > 0.9, true);
  assert.equal(predictions.length, 5);
});
