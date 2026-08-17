import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { rasterizeStroke3 } from '../src/sketch-rasterizer.mjs';

const SCRIPT = fileURLToPath(new URL('../scripts/build-training-cache.py', import.meta.url));
const PYTHON = process.env.PYTHON ?? 'python3';

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function createFixture(root) {
  const dataDirectory = join(root, 'archives');
  const manifestPath = join(root, 'manifest.json');
  mkdirSync(dataDirectory);
  const source = `
import numpy as np
import os
import sys

root = sys.argv[1]
samples = {
    'alpha': [[5, 5, 0], [10, 0, 1]],
    'beta': [[5, 5, 0], [0, 10, 1]],
}
for name, sample in samples.items():
    splits = {}
    for split_name, count in [('train', 3), ('valid', 2), ('test', 2)]:
        values = np.empty(count, dtype=object)
        for index in range(count):
            values[index] = np.asarray(sample, dtype=np.int16)
        splits[split_name] = values
    np.savez_compressed(os.path.join(root, name + '.npz'), **splits)
`;
  const created = spawnSync(PYTHON, ['-c', source, dataDirectory], { encoding: 'utf8' });
  assert.equal(created.status, 0, created.stderr);

  const classes = ['alpha', 'beta'].map(name => {
    const archivePath = join(dataDirectory, `${name}.npz`);
    return {
      name,
      filename: `${name}.npz`,
      bytes: readFileSync(archivePath).length,
      sha256: sha256(archivePath),
      splits: {
        train: { count: 3 },
        valid: { count: 2 },
        test: { count: 2 },
      },
      dtypes: ['int16'],
      penStates: [0, 1],
    };
  });
  const content = {
    schemaVersion: 1,
    source: { format: 'Sketch-RNN stroke-3 NPZ' },
    classes,
  };
  const manifest = {
    ...content,
    fingerprint: createHash('sha256').update(JSON.stringify(content)).digest('hex'),
  };
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return { dataDirectory, manifestPath };
}

function buildCache({ dataDirectory, manifestPath, outputDirectory, node = process.execPath }) {
  return spawnSync(PYTHON, [
    SCRIPT,
    '--manifest', manifestPath,
    '--manifest-sha256', sha256(manifestPath),
    '--data-dir', dataDirectory,
    '--output', outputDirectory,
    '--train-per-class', '2',
    '--valid-per-class', '1',
    '--test-per-class', '1',
    '--seed', '17',
    '--node', node,
    '--allow-untrusted-archives',
  ], { encoding: 'utf8' });
}

function inspectCache(outputDirectory) {
  const source = `
import hashlib
import json
import numpy as np
import os
import sys

root = sys.argv[1]
result = {}
for split in ('train', 'valid', 'test'):
    images = np.load(os.path.join(root, split + '-images.npy'), mmap_mode='r')
    labels = np.load(os.path.join(root, split + '-labels.npy'), mmap_mode='r')
    result[split] = {
        'imageShape': list(images.shape),
        'imageDtype': str(images.dtype),
        'labels': labels.tolist(),
        'labelDtype': str(labels.dtype),
        'firstImageSha256': hashlib.sha256(images[0].tobytes()).hexdigest(),
    }
print(json.dumps(result))
`;
  const result = spawnSync(PYTHON, ['-c', source, outputDirectory], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

test('training cache is balanced, browser-rasterized, fingerprinted, and reusable', () => {
  const root = mkdtempSync(join(tmpdir(), 'quickdraw-cache-'));
  const outputDirectory = join(root, 'cache');
  const fixture = createFixture(root);

  try {
    const first = buildCache({ ...fixture, outputDirectory });
    assert.equal(first.status, 0, first.stderr);
    const metadataPath = join(outputDirectory, 'metadata.json');
    const firstMetadataText = readFileSync(metadataPath, 'utf8');
    const metadata = JSON.parse(firstMetadataText);
    const inspection = inspectCache(outputDirectory);

    assert.equal(metadata.schemaVersion, 1);
    assert.deepEqual(metadata.classes, ['alpha', 'beta']);
    assert.deepEqual(metadata.rasterizer, {
      worker: 'scripts/rasterize-stroke3-worker.mjs',
      size: 64,
      padding: 4,
      strokeWidth: 2.5,
      supersample: 4,
    });
    assert.deepEqual(
      Object.keys(metadata.request.rasterizerFiles),
      ['worker', 'implementation'],
    );
    assert.equal(
      metadata.request.rasterizerFiles.worker.sha256,
      sha256('scripts/rasterize-stroke3-worker.mjs'),
    );
    assert.equal(
      metadata.request.rasterizerFiles.implementation.sha256,
      sha256('src/sketch-rasterizer.mjs'),
    );
    assert.equal(metadata.request.builder.sha256, sha256('scripts/build-training-cache.py'));
    assert.deepEqual(metadata.request.builder.dependencies, {
      'model/integrity.py': sha256('model/integrity.py'),
    });
    assert.match(metadata.request.builder.runtime.python, /^3\./);
    assert.match(metadata.request.builder.runtime.numpy, /^1\./);
    assert.match(metadata.request.node.version, /^v\d+/);
    assert.match(metadata.request.node.sha256, /^[a-f0-9]{64}$/);
    assert.deepEqual(Object.keys(metadata.request.node), ['version', 'sha256']);
    assert.equal('path' in metadata.request.node, false);
    assert.match(metadata.fingerprint, /^[a-f0-9]{64}$/);
    assert.deepEqual(inspection.train.imageShape, [4, 64, 64]);
    assert.deepEqual(inspection.valid.imageShape, [2, 64, 64]);
    assert.deepEqual(inspection.test.imageShape, [2, 64, 64]);
    assert.deepEqual(inspection.train.labels, [0, 0, 1, 1]);
    assert.deepEqual(inspection.valid.labels, [0, 1]);
    assert.deepEqual(inspection.test.labels, [0, 1]);
    assert.equal(inspection.train.imageDtype, 'uint8');
    assert.equal(inspection.train.labelDtype, 'int16');
    assert.equal(
      inspection.train.firstImageSha256,
      createHash('sha256').update(rasterizeStroke3([[5, 5, 0], [10, 0, 1]])).digest('hex'),
    );

    const artifactHashes = Object.fromEntries([
      'train-images.npy',
      'train-labels.npy',
      'valid-images.npy',
      'valid-labels.npy',
      'test-images.npy',
      'test-labels.npy',
    ].map(filename => [filename, sha256(join(outputDirectory, filename))]));

    const second = buildCache({ ...fixture, outputDirectory });
    assert.equal(second.status, 0, second.stderr);
    assert.match(second.stdout, /Training cache is current/);
    assert.equal(readFileSync(metadataPath, 'utf8'), firstMetadataText);
    for (const [filename, hash] of Object.entries(artifactHashes)) {
      assert.equal(sha256(join(outputDirectory, filename)), hash);
    }

    const nodeAlias = join(root, 'node-alias');
    symlinkSync(process.execPath, nodeAlias);
    const aliasBuild = buildCache({ ...fixture, outputDirectory, node: nodeAlias });
    assert.equal(aliasBuild.status, 0, aliasBuild.stderr);
    assert.match(aliasBuild.stdout, /Training cache is current/);

    const nodeWrapper = join(root, 'node-wrapper');
    writeFileSync(
      nodeWrapper,
      `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} "$@"\n`,
    );
    chmodSync(nodeWrapper, 0o755);
    const wrapperBuild = buildCache({ ...fixture, outputDirectory, node: nodeWrapper });
    assert.equal(wrapperBuild.status, 0, wrapperBuild.stderr);
    assert.match(wrapperBuild.stdout, /Built training cache/);
    const wrapperMetadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
    assert.equal(wrapperMetadata.request.node.sha256, sha256(nodeWrapper));
    assert.notEqual(wrapperMetadata.request.node.sha256, metadata.request.node.sha256);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('escaped artifact paths in cache metadata force a rebuild', () => {
  const root = mkdtempSync(join(tmpdir(), 'quickdraw-cache-path-'));
  const outputDirectory = join(root, 'cache');
  const fixture = createFixture(root);

  try {
    const first = buildCache({ ...fixture, outputDirectory });
    assert.equal(first.status, 0, first.stderr);
    const metadataPath = join(outputDirectory, 'metadata.json');
    const metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
    copyFileSync(join(outputDirectory, 'train-images.npy'), join(root, 'outside.npy'));
    delete metadata.fingerprint;
    metadata.splits.train.images.filename = '../outside.npy';
    metadata.fingerprint = createHash('sha256')
      .update(JSON.stringify(metadata))
      .digest('hex');
    writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

    const rebuilt = buildCache({ ...fixture, outputDirectory });
    assert.equal(rebuilt.status, 0, rebuilt.stderr);
    assert.match(rebuilt.stdout, /Built training cache/);
    const rebuiltMetadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
    assert.equal(rebuiltMetadata.splits.train.images.filename, 'train-images.npy');

    symlinkSync('loop.npy', join(outputDirectory, 'loop.npy'));
    delete rebuiltMetadata.fingerprint;
    rebuiltMetadata.splits.train.images.filename = 'loop.npy';
    rebuiltMetadata.fingerprint = createHash('sha256')
      .update(JSON.stringify(rebuiltMetadata))
      .digest('hex');
    writeFileSync(metadataPath, `${JSON.stringify(rebuiltMetadata, null, 2)}\n`);

    const loopRebuild = buildCache({ ...fixture, outputDirectory });
    assert.equal(loopRebuild.status, 0, loopRebuild.stderr);
    assert.match(loopRebuild.stdout, /Built training cache/);
    assert.equal(
      JSON.parse(readFileSync(metadataPath, 'utf8')).splits.train.images.filename,
      'train-images.npy',
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('raster worker streams maximum-size records without pipe deadlock', () => {
  const source = `
import importlib.util
import numpy as np
import sys
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('training_cache', 'scripts/build-training-cache.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = SimpleNamespace(
    node=sys.argv[1],
    worker=module.DEFAULT_WORKER,
    size=64,
    padding=4,
    stroke_width=2.5,
    supersample=4,
)
sample = np.zeros((10000, 3), dtype=np.int16)
sample[-1, 2] = 1
worker = module.RasterWorker(args)
try:
    bitmaps = worker.rasterize([sample] * 8)
    worker.close()
    assert len(bitmaps) == 8
    assert all(len(bitmap) == 4096 for bitmap in bitmaps)
finally:
    worker.terminate()
`;
  const result = spawnSync(PYTHON, ['-c', source, process.execPath], {
    encoding: 'utf8',
    timeout: 15000,
  });
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stderr);
});

test('custom pickle archives require an explicit unsafe opt-in', () => {
  const root = mkdtempSync(join(tmpdir(), 'quickdraw-cache-trust-'));
  const outputDirectory = join(root, 'cache');
  const fixture = createFixture(root);
  const result = spawnSync(PYTHON, [
    SCRIPT,
    '--manifest', fixture.manifestPath,
    '--manifest-sha256', sha256(fixture.manifestPath),
    '--data-dir', fixture.dataDirectory,
    '--output', outputDirectory,
    '--train-per-class', '1',
    '--valid-per-class', '1',
    '--test-per-class', '1',
  ], { encoding: 'utf8' });

  try {
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /require --allow-untrusted-archives/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cache setup removes temporary outputs when the raster worker fails', () => {
  const root = mkdtempSync(join(tmpdir(), 'quickdraw-cache-failure-'));
  const outputDirectory = join(root, 'cache');
  const fakeNode = join(root, 'fake-node');
  const fixture = createFixture(root);
  writeFileSync(fakeNode, '#!/bin/sh\nif [ "$1" = "--version" ]; then echo v0.0.0; exit 0; fi\nexit 1\n');
  chmodSync(fakeNode, 0o755);

  try {
    const result = spawnSync(PYTHON, [
      SCRIPT,
      '--manifest', fixture.manifestPath,
      '--manifest-sha256', sha256(fixture.manifestPath),
      '--data-dir', fixture.dataDirectory,
      '--output', outputDirectory,
      '--train-per-class', '1',
      '--valid-per-class', '1',
      '--test-per-class', '1',
      '--node', fakeNode,
      '--allow-untrusted-archives',
    ], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.equal(readdirSync(root).some(name => name.startsWith('.cache.tmp-')), false);
    assert.equal(readdirSync(root).includes('cache'), false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
