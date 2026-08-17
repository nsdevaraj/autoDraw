import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const PYTHON = '.venv-model/bin/python3';
const TRAINER = fileURLToPath(new URL('../scripts/train-quickdraw-model.py', import.meta.url));

function createCache(directory) {
  const source = `
import hashlib
import json
import numpy as np
import os
import sys

root = sys.argv[1]
os.makedirs(root)
rng = np.random.default_rng(23)

def create_split(per_class):
    images = np.zeros((per_class * 2, 64, 64), dtype=np.uint8)
    labels = np.repeat(np.arange(2, dtype=np.int16), per_class)
    for class_index in range(2):
        for sample_index in range(per_class):
            image = images[class_index * per_class + sample_index]
            offset = int(rng.integers(-3, 4))
            if class_index == 0:
                image[30 + offset:34 + offset, 10:54] = 255
            else:
                image[10:54, 30 + offset:34 + offset] = 255
    return images, labels

artifacts = {}
for split, count in [('train', 32), ('valid', 12), ('test', 12)]:
    images, labels = create_split(count)
    image_name = split + '-images.npy'
    label_name = split + '-labels.npy'
    np.save(os.path.join(root, image_name), images)
    np.save(os.path.join(root, label_name), labels)
    def artifact(name, array):
        path = os.path.join(root, name)
        return {
            'filename': name,
            'bytes': os.path.getsize(path),
            'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
            'shape': list(array.shape),
            'dtype': str(array.dtype),
        }
    artifacts[split] = {
        'count': len(labels),
        'images': artifact(image_name, images),
        'labels': artifact(label_name, labels),
    }

metadata = {
    'schemaVersion': 1,
  'request': {
    'sourceManifestFingerprint': 'fixture-manifest',
    'sourceManifestSha256': 'f' * 64,
    'rasterizerFiles': {
      'worker': {'path': 'scripts/rasterize-stroke3-worker.mjs', 'sha256': 'a' * 64},
      'implementation': {'path': 'src/sketch-rasterizer.mjs', 'sha256': 'b' * 64},
    },
    'builder': {
      'path': 'scripts/build-training-cache.py',
      'sha256': 'c' * 64,
      'runtime': {'python': '3.9.6', 'numpy': '1.26.4'},
    },
    'node': {'path': '/node', 'version': 'v24.14.1', 'sha256': 'd' * 64},
  },
    'classes': ['horizontal', 'vertical'],
    'sampling': {'seed': 23, 'perClass': {'train': 32, 'valid': 12, 'test': 12}},
    'rasterizer': {
        'worker': 'scripts/rasterize-stroke3-worker.mjs',
        'size': 64,
        'padding': 4,
        'strokeWidth': 2.5,
        'supersample': 4,
    },
    'splits': artifacts,
}
encoded = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
metadata['fingerprint'] = hashlib.sha256(encoded).hexdigest()
open(os.path.join(root, 'metadata.json'), 'w').write(json.dumps(metadata, indent=2) + '\\n')
`;
  const result = spawnSync(PYTHON, ['-c', source, directory], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
}

test('trainer exports an accurate browser-ready ONNX model with runtime parity', () => {
  const root = mkdtempSync(join(tmpdir(), 'quickdraw-training-'));
  const cacheDirectory = join(root, 'cache');
  const outputDirectory = join(root, 'model');
  createCache(cacheDirectory);

  try {
    const result = spawnSync(PYTHON, [
      TRAINER,
      '--cache', cacheDirectory,
      '--output', outputDirectory,
      '--epochs', '6',
      '--batch-size', '16',
      '--device', 'cpu',
      '--seed', '41',
      '--min-test-accuracy', '0.95',
      '--min-class-accuracy', '0.95',
    ], { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
    assert.equal(result.status, 0, result.stderr);

    const metadataPath = join(outputDirectory, 'model.json');
    const modelPath = join(outputDirectory, 'quickdraw-mvp.onnx');
    const metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
    assert.equal(metadata.schemaVersion, 1);
    assert.equal(metadata.kind, 'quickdraw-classifier');
    assert.deepEqual(metadata.classes, ['horizontal', 'vertical']);
    assert.equal(metadata.model.format, 'ONNX');
    assert.equal(metadata.model.input.name, 'bitmap');
    assert.deepEqual(metadata.model.input.shape, ['batch', 1, 64, 64]);
    assert.equal(metadata.model.input.normalization, 'uint8 / 255');
    assert.equal(metadata.model.output.name, 'logits');
    assert.deepEqual(metadata.model.output.shape, ['batch', 2]);
    assert.equal(metadata.metrics.test.accuracy >= 0.95, true);
    assert.equal(metadata.metrics.onnxTest.accuracy >= 0.95, true);
    assert.equal(metadata.metrics.maxLogitDifference < 1e-4, true);
    assert.equal(metadata.model.sha256, createHash('sha256').update(readFileSync(modelPath)).digest('hex'));
    assert.equal(metadata.model.bytes, readFileSync(modelPath).length);
    assert.equal(metadata.trainingData.sourceManifestFingerprint, 'fixture-manifest');
    assert.equal(metadata.trainingData.sourceManifestSha256, 'f'.repeat(64));
    assert.equal(metadata.trainingData.builder.sha256, 'c'.repeat(64));
    assert.equal(metadata.trainingData.node.sha256, 'd'.repeat(64));
    assert.deepEqual(metadata.trainingData.sampling.perClass, { train: 32, valid: 12, test: 12 });
    assert.equal(metadata.training.optimizer.name, 'AdamW');
    assert.equal(metadata.training.numWorkers, 0);
    assert.equal(metadata.training.minimumTestAccuracy, 0.95);
    assert.equal(metadata.training.minimumPerClassAccuracy, 0.95);
    assert.equal(metadata.training.deterministicAlgorithms, true);
    assert.match(metadata.runtime.python, /^3\./);
    assert.match(metadata.runtime.numpy, /^1\.|^2\./);
    assert.match(metadata.source.files['model/quickdraw_cnn.py'], /^[a-f0-9]{64}$/);
    assert.equal(typeof metadata.source.filesMatchRevision, 'boolean');
    assert.match(metadata.fingerprint, /^[a-f0-9]{64}$/);
    const fingerprintCheck = spawnSync(PYTHON, ['-c', `
import hashlib
import json
import sys
metadata = json.load(open(sys.argv[1]))
fingerprint = metadata.pop('fingerprint')
encoded = json.dumps(metadata, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
print(json.dumps({'fingerprint': fingerprint, 'computed': hashlib.sha256(encoded).hexdigest()}))
`, metadataPath], { encoding: 'utf8' });
    assert.equal(fingerprintCheck.status, 0, fingerprintCheck.stderr);
    const fingerprints = JSON.parse(fingerprintCheck.stdout);
    assert.equal(fingerprints.fingerprint, fingerprints.computed);

    const inference = spawnSync(PYTHON, ['-c', `
import json
import numpy as np
import onnxruntime as ort
session = ort.InferenceSession(${JSON.stringify(modelPath)}, providers=['CPUExecutionProvider'])
output = session.run(['logits'], {'bitmap': np.zeros((3, 1, 64, 64), dtype=np.float32)})[0]
print(json.dumps({'shape': list(output.shape), 'finite': bool(np.isfinite(output).all())}))
`], { encoding: 'utf8' });
    assert.equal(inference.status, 0, inference.stderr);
    assert.deepEqual(JSON.parse(inference.stdout), { shape: [3, 2], finite: true });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
