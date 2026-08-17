import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT = fileURLToPath(new URL('../scripts/fetch-sketchrnn-data.py', import.meta.url));
const PYTHON = process.env.PYTHON ?? 'python3';

function createArchive(path, penState = 1, dtype = 'np.int16') {
  const source = `
import numpy as np
import sys

def split(samples):
    output = np.empty(len(samples), dtype=object)
    for index, sample in enumerate(samples):
        output[index] = np.asarray(sample, dtype=${dtype})
    return output

train = split([
    [[4, 5, 0], [2, -1, ${penState}]],
    [[1, 1, 0], [1, 1, 0], [1, 1, 1]],
])
valid = split([[[2, 3, 0], [2, 3, 1]]])
test = split([[[3, 2, 0], [3, 2, 1]]])
np.savez_compressed(sys.argv[1], train=train, valid=valid, test=test)
`;
  const result = spawnSync(PYTHON, ['-c', source, path], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
}

function createArchiveWithSample(path, sampleExpression) {
  const source = `
import numpy as np
import sys

sample = ${sampleExpression}
split = np.empty(1, dtype=object)
split[0] = sample
np.savez_compressed(sys.argv[1], train=split, valid=split, test=split)
`;
  const result = spawnSync(PYTHON, ['-c', source, path], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
}

function runVerifier(directory, manifestPath, className = 'cat', extraArguments = []) {
  return spawnSync(PYTHON, [
    SCRIPT,
    '--class-name', className,
    '--output', directory,
    '--manifest', manifestPath,
    '--verify-only',
    '--allow-nonstandard-splits',
    ...extraArguments,
  ], { encoding: 'utf8' });
}

function runAsync(command, arguments_) {
  return new Promise(resolve => {
    const child = spawn(command, arguments_);
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr += chunk; });
    child.on('close', status => resolve({ status, stdout, stderr }));
  });
}

test('verified Sketch-RNN archives produce a deterministic split manifest', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-data-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath);

  try {
    const first = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.equal(first.status, 0, first.stderr);
    const firstContents = readFileSync(manifestPath, 'utf8');
    const manifest = JSON.parse(firstContents);

    assert.equal(manifest.schemaVersion, 1);
    assert.equal(manifest.source.format, 'Sketch-RNN stroke-3 NPZ');
    assert.equal(manifest.classes[0].name, 'cat');
    assert.match(manifest.classes[0].sha256, /^[a-f0-9]{64}$/);
    assert.equal(manifest.classes[0].bytes > 0, true);
    assert.deepEqual(manifest.classes[0].splits, {
      train: { count: 2, minSequenceLength: 2, maxSequenceLength: 3 },
      valid: { count: 1, minSequenceLength: 2, maxSequenceLength: 2 },
      test: { count: 1, minSequenceLength: 2, maxSequenceLength: 2 },
    });
    assert.deepEqual(manifest.classes[0].dtypes, ['int16']);
    assert.deepEqual(manifest.classes[0].penStates, [0, 1]);

    const second = runVerifier(directory, manifestPath);
    assert.equal(second.status, 0, second.stderr);
    assert.equal(readFileSync(manifestPath, 'utf8'), firstContents);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('archive verification rejects invalid stroke-3 pen states', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-invalid-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath, 2);

  try {
    const result = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /pen state must be 0 or 1/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('archive verification rejects data that is not int16', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-dtype-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath, 1, 'np.float64');

  try {
    const result = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /dtype must be int16/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('archive verification rejects empty stroke-3 samples', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-empty-sample-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchiveWithSample(archivePath, "np.empty((0, 3), dtype=np.int16)");

  try {
    const result = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /must contain at least one point/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('archive verification rejects stroke-3 samples without a final pen lift', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-unterminated-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchiveWithSample(archivePath, "np.asarray([[1, 2, 0]], dtype=np.int16)");

  try {
    const result = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /must end with pen state 1/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('archive verification requires a trusted checksum by default', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-trust-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath);

  try {
    const result = runVerifier(directory, manifestPath);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /No trusted checksum for cat/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('subset runs cannot overwrite the default full manifest', () => {
  const result = spawnSync(PYTHON, [SCRIPT, '--class-name', 'cat', '--verify-only'], {
    encoding: 'utf8',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /--class-name cannot write the full production manifest/);
});

test('archive verification rejects a tampered trust manifest', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-trust-manifest-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath);

  try {
    const created = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.equal(created.status, 0, created.stderr);
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    manifest.classes[0].sha256 = '0'.repeat(64);
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

    const result = runVerifier(directory, manifestPath);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Trust manifest fingerprint mismatch/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('archive verification rejects empty checksum values before loading data', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-empty-checksum-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath);

  try {
    const created = runVerifier(directory, manifestPath, 'cat', ['--allow-unverified-archives']);
    assert.equal(created.status, 0, created.stderr);
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    manifest.classes[0].sha256 = '';
    const { fingerprint: _fingerprint, ...content } = manifest;
    manifest.fingerprint = createHash('sha256').update(JSON.stringify(content)).digest('hex');
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

    const result = runVerifier(directory, manifestPath);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /invalid SHA-256 for cat/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('custom class files cannot overwrite the default full manifest', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-custom-classes-'));
  const classesPath = join(directory, 'classes.txt');
  writeFileSync(classesPath, 'cat\n');

  try {
    const result = spawnSync(PYTHON, [SCRIPT, `--classes=${classesPath}`, '--verify-only'], {
      encoding: 'utf8',
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /--classes cannot write the full production manifest/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('unsafe archive loading cannot write the production trust manifest', () => {
  const outputDirectory = mkdtempSync(join(tmpdir(), 'sketchrnn-unsafe-manifest-'));
  try {
    const result = spawnSync(PYTHON, [
      SCRIPT,
      '--output', outputDirectory,
      '--verify-only',
      '--allow-unverified-archives',
    ], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /unsafe archive loading cannot write the production manifest/);
  } finally {
    rmSync(outputDirectory, { recursive: true, force: true });
  }
});

test('unsafe archive loading cannot write the production archive cache', () => {
  const root = mkdtempSync(join(tmpdir(), 'sketchrnn-unsafe-cache-'));
  const manifestPath = join(root, 'manifest.json');

  try {
    const result = spawnSync(PYTHON, [
      SCRIPT,
      '--class-name', 'cat',
      '--manifest', manifestPath,
      '--allow-unverified-archives',
      '--verify-only',
    ], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /unsafe archive loading cannot use the production archive cache/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('equals-form explicit manifest is accepted for a class subset', () => {
  const directory = mkdtempSync(join(tmpdir(), 'sketchrnn-equals-manifest-'));
  const archivePath = join(directory, 'cat.npz');
  const manifestPath = join(directory, 'manifest.json');
  createArchive(archivePath);

  try {
    const result = spawnSync(PYTHON, [
      SCRIPT,
      '--class-name=cat',
      `--output=${directory}`,
      `--manifest=${manifestPath}`,
      '--verify-only',
      '--allow-nonstandard-splits',
      '--allow-unverified-archives',
    ], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('class selectors are mutually exclusive', () => {
  const result = spawnSync(PYTHON, [
    SCRIPT,
    '--classes=config/mvp-classes.txt',
    '--class-name=cat',
  ], { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /not allowed with argument/);
});

test('production manifest requires official source, trust, and split policy', () => {
  const cases = [
    [['--verify-only', '--base-url=https://example.test'], /official base URL/],
    [['--verify-only', '--trust-manifest=/tmp/other.json'], /production trust manifest/],
    [['--verify-only', '--allow-nonstandard-splits'], /standard split sizes/],
  ];

  for (const [arguments_, expectedError] of cases) {
    const result = spawnSync(PYTHON, [SCRIPT, ...arguments_], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, expectedError);
  }
});

test('production manifest requires the pinned MVP class list', () => {
  const root = mkdtempSync(join(tmpdir(), 'sketchrnn-production-list-'));
  mkdirSync(join(root, 'scripts'));
  mkdirSync(join(root, 'config'));
  copyFileSync(SCRIPT, join(root, 'scripts', 'fetch-sketchrnn-data.py'));
  writeFileSync(join(root, 'config', 'mvp-classes.txt'), 'cat\n');

  try {
    const result = spawnSync(PYTHON, [
      join(root, 'scripts', 'fetch-sketchrnn-data.py'),
      '--verify-only',
    ], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /MVP class-list fingerprint mismatch/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});


test('archives with encoded class names download atomically and reuse the cache', () => {
  const root = mkdtempSync(join(tmpdir(), 'sketchrnn-download-'));
  const sourceDirectory = join(root, 'source');
  const outputDirectory = join(root, 'output');
  const manifestPath = join(root, 'manifest.json');
  const className = 'coffee cup';
  const sourceArchive = join(sourceDirectory, `${className}.npz`);
  mkdirSync(sourceDirectory);
  createArchive(sourceArchive);
  const baseUrl = pathToFileURL(sourceDirectory).href;
  const arguments_ = [
    SCRIPT,
    '--class-name', className,
    '--output', outputDirectory,
    '--manifest', manifestPath,
    '--base-url', baseUrl,
    '--allow-nonstandard-splits',
    '--allow-unverified-archives',
  ];

  try {
    const first = spawnSync(PYTHON, arguments_, { encoding: 'utf8' });
    assert.equal(first.status, 0, first.stderr);
    assert.match(first.stdout, /coffee cup: downloaded/);
    assert.equal(readFileSync(join(outputDirectory, `${className}.npz`)).equals(readFileSync(sourceArchive)), true);
    assert.equal(readFileSync(manifestPath, 'utf8').includes('coffee%20cup.npz'), true);
    const partFiles = readdirSync(outputDirectory)
      .filter(filename => filename.includes('.part-'));
    assert.deepEqual(partFiles, []);

    const second = spawnSync(PYTHON, arguments_, { encoding: 'utf8' });
    assert.equal(second.status, 0, second.stderr);
    assert.match(second.stdout, /coffee cup: cached/);

    writeFileSync(join(outputDirectory, `${className}.npz`), 'tampered archive');
    const tampered = spawnSync(PYTHON, arguments_, { encoding: 'utf8' });
    assert.notEqual(tampered.status, 0);
    assert.match(tampered.stderr, /checksum mismatch/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('a corrupt forced download cannot replace a valid trusted cache', () => {
  const root = mkdtempSync(join(tmpdir(), 'sketchrnn-corrupt-download-'));
  const sourceDirectory = join(root, 'source');
  const outputDirectory = join(root, 'output');
  const manifestPath = join(root, 'manifest.json');
  const className = 'cat';
  const sourceArchive = join(sourceDirectory, `${className}.npz`);
  const cachedArchive = join(outputDirectory, `${className}.npz`);
  mkdirSync(sourceDirectory);
  mkdirSync(outputDirectory);
  createArchive(sourceArchive);
  copyFileSync(sourceArchive, cachedArchive);
  const original = readFileSync(cachedArchive);
  const baseUrl = pathToFileURL(sourceDirectory).href;

  try {
    const trusted = spawnSync(PYTHON, [
      SCRIPT,
      '--class-name', className,
      '--output', outputDirectory,
      '--manifest', manifestPath,
      '--base-url', baseUrl,
      '--verify-only',
      '--allow-nonstandard-splits',
      '--allow-unverified-archives',
    ], { encoding: 'utf8' });
    assert.equal(trusted.status, 0, trusted.stderr);

    writeFileSync(sourceArchive, Buffer.concat([original, Buffer.from([0])]));
    const result = spawnSync(PYTHON, [
      SCRIPT,
      '--class-name', className,
      '--output', outputDirectory,
      '--manifest', manifestPath,
      '--base-url', baseUrl,
      '--force',
      '--allow-nonstandard-splits',
    ], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /download size mismatch/);
    assert.equal(readFileSync(cachedArchive).equals(original), true);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('concurrent processes download the same archive without sharing temporary files', async () => {
  const root = mkdtempSync(join(tmpdir(), 'sketchrnn-concurrent-'));
  const sourceDirectory = join(root, 'source');
  const outputDirectory = join(root, 'output');
  const className = 'cat';
  const sourceArchive = join(sourceDirectory, `${className}.npz`);
  mkdirSync(sourceDirectory);
  createArchive(sourceArchive);
  const baseUrl = pathToFileURL(sourceDirectory).href;

  function argumentsFor(manifestName) {
    return [
      SCRIPT,
      '--class-name', className,
      '--output', outputDirectory,
      '--manifest', join(root, manifestName),
      '--base-url', baseUrl,
      '--allow-nonstandard-splits',
      '--allow-unverified-archives',
      '--force',
    ];
  }

  try {
    const [first, second] = await Promise.all([
      runAsync(PYTHON, argumentsFor('first.json')),
      runAsync(PYTHON, argumentsFor('second.json')),
    ]);
    assert.equal(first.status, 0, first.stderr);
    assert.equal(second.status, 0, second.stderr);
    assert.equal(readFileSync(join(outputDirectory, `${className}.npz`)).equals(readFileSync(sourceArchive)), true);
    assert.deepEqual(readdirSync(outputDirectory).filter(name => name.includes('.part-')), []);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
