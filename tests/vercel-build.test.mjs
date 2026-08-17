import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const BUILD_SCRIPT = fileURLToPath(new URL('../scripts/build-vercel.mjs', import.meta.url));

const COPIED_FILES = new Map([
  ['index.html', 'index.html'],
  ['curate.html', 'curate.html'],
  ['config/mvp-classes.txt', 'config/mvp-classes.txt'],
  ['data/quickdraw-candidates.json', 'data/quickdraw-candidates.json'],
  ['models/quickdraw-mvp/model.json', 'models/quickdraw-mvp/model.json'],
  ['models/quickdraw-mvp/quickdraw-mvp.onnx', 'models/quickdraw-mvp/quickdraw-mvp.onnx'],
  ['models/sketch-embedder/model.json', 'models/sketch-embedder/model.json'],
  ['models/sketch-embedder/sketch-embedder.onnx', 'models/sketch-embedder/sketch-embedder.onnx'],
  ['src/autodraw-suggestions.mjs', 'src/autodraw-suggestions.mjs'],
  ['src/curation-app.mjs', 'src/curation-app.mjs'],
  ['src/curation-state.mjs', 'src/curation-state.mjs'],
  ['src/drawing-app.mjs', 'src/drawing-app.mjs'],
  ['src/drawing-export.mjs', 'src/drawing-export.mjs'],
  ['src/drawing-state.mjs', 'src/drawing-state.mjs'],
  ['src/icon-candidate.mjs', 'src/icon-candidate.mjs'],
  ['src/icon-retrieval.mjs', 'src/icon-retrieval.mjs'],
  ['src/line-list.mjs', 'src/line-list.mjs'],
  ['src/quickdraw-classifier.mjs', 'src/quickdraw-classifier.mjs'],
  ['src/sketch-embedder.mjs', 'src/sketch-embedder.mjs'],
  ['src/sketch-rasterizer.mjs', 'src/sketch-rasterizer.mjs'],
  ['src/tokenize.mjs', 'src/tokenize.mjs'],
  ['node_modules/onnxruntime-web/dist/ort.wasm.min.mjs', 'vendor/ort.wasm.min.mjs'],
  ['node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs', 'vendor/ort-wasm-simd-threaded.mjs'],
  ['node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', 'vendor/ort-wasm-simd-threaded.wasm'],
]);

const ICON_INDEX_DIRECTORY = 'data/icon-embeddings';

// The icon index is generated, so the expectation is derived from the tree rather than
// pinned name by name; the allowlist assertion below still has to match it exactly.
function copiedFiles() {
  const files = new Map(COPIED_FILES);
  if (!existsSync(ICON_INDEX_DIRECTORY)) return files;
  for (const name of readdirSync(ICON_INDEX_DIRECTORY)) {
    files.set(`${ICON_INDEX_DIRECTORY}/${name}`, `${ICON_INDEX_DIRECTORY}/${name}`);
  }
  return files;
}

function filesUnder(root, directory = root) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(root, path) : [relative(root, path).replaceAll('\\', '/')];
  }).sort();
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

test('Vercel build creates an exact clean static allowlist with byte-identical assets', () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-vercel-build-'));
  const output = join(root, 'dist');
  mkdirSync(output);
  writeFileSync(join(output, 'stale.txt'), 'stale');

  try {
    const result = spawnSync(process.execPath, [BUILD_SCRIPT, '--output', output], {
      encoding: 'utf8',
    });
    assert.equal(result.status, 0, result.stderr);
    const expected = copiedFiles();
    assert.deepEqual(filesUnder(output), [...expected.values()].sort());
    for (const [source, destination] of expected) {
      assert.equal(sha256(join(output, destination)), sha256(source), destination);
    }
    assert.match(result.stdout, /Built Vercel static output/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('Vercel configuration builds dist, rewrites icons, and preserves WASM security headers', () => {
  const config = JSON.parse(readFileSync('vercel.json', 'utf8'));
  assert.equal(config.buildCommand, 'node scripts/build-vercel.mjs');
  assert.equal(config.outputDirectory, 'dist');
  assert.equal(config.devCommand, 'node scripts/serve-curation.mjs --port $PORT');
  assert.deepEqual(config.rewrites, [{
    source: '/svgdepot/:path*',
    destination: '/api/icon?path=:path*',
  }]);
  assert.equal(
    config.functions['api/icon.mjs'].includeFiles,
    'data/quickdraw-candidates.json',
  );

  const allHeaders = config.headers.flatMap(route => route.headers);
  const csp = allHeaders.find(header => header.key === 'Content-Security-Policy')?.value;
  assert.match(csp, /script-src 'self' 'wasm-unsafe-eval'/);
  assert.equal(allHeaders.some(header => header.key === 'X-Content-Type-Options' && header.value === 'nosniff'), true);

  const ignored = readFileSync('.vercelignore', 'utf8').split(/\r?\n/).filter(Boolean);
  for (const path of ['.cache/', '.venv-model/', 'datasets/', 'tests/']) {
    assert.equal(ignored.includes(path), true, path);
  }
});