import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const SCRIPT = fileURLToPath(new URL('../scripts/build-quickdraw-candidates.mjs', import.meta.url));
const SOURCE = {
  repository: 'https://github.com/nsdevaraj/SVGDepot.git',
  ref: 'main',
  commit: 'abc123',
};

function createIndex(overrides = {}) {
  return {
    schemaVersion: 1,
    tokenizer: 'filename-nfkd-lower-ascii-alpha-v1',
    source: SOURCE,
    categories: [],
    packs: [],
    icons: [],
    tokens: {},
    ...overrides,
  };
}

function writeJson(path, value) {
  writeFileSync(path, `${JSON.stringify(value)}\n`);
}

function runGenerator({ index, categories, rules, limit = 12 }) {
  const directory = mkdtempSync(join(tmpdir(), 'quickdraw-candidates-'));
  const indexPath = join(directory, 'index.json.gz');
  const categoriesPath = join(directory, 'categories.txt');
  const rulesPath = join(directory, 'rules.json');
  const outputPath = join(directory, 'candidates.json');

  try {
    writeFileSync(indexPath, gzipSync(JSON.stringify(index)));
    writeFileSync(categoriesPath, `${categories.join('\n')}\n`);
    writeJson(rulesPath, rules);

    const result = spawnSync(process.execPath, [
      SCRIPT,
      '--index', indexPath,
      '--categories', categoriesPath,
      '--rules', rulesPath,
      '--output', outputPath,
      '--limit', String(limit),
    ], { encoding: 'utf8' });

    const output = result.status === 0
      ? JSON.parse(readFileSync(outputPath, 'utf8'))
      : null;
    return { result, output };
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

test('an alias-only Quick Draw class receives matching SVG candidates', () => {
  const index = createIndex({
    categories: ['Food, Drink & Hospitality'],
    packs: [[0, 'beer-icons']],
    icons: [[0, 'bottle-cap.svg'], [0, 'bottle.svg']],
    tokens: { bottle: [0, 1], cap: [0] },
  });

  const { result, output } = runGenerator({
    index,
    categories: ['bottlecap'],
    rules: { aliases: { bottlecap: ['bottle cap'] } },
  });

  assert.equal(result.status, 0, result.stderr);
  assert.equal(output.classes[0].name, 'bottlecap');
  assert.deepEqual(output.classes[0].aliases, ['bottle cap']);
  assert.deepEqual(
    {
      name: output.classes[0].candidates[0].name,
      category: output.classes[0].candidates[0].category,
      pack: output.classes[0].candidates[0].pack,
      path: output.classes[0].candidates[0].path,
      url: output.classes[0].candidates[0].url,
    },
    {
      name: 'bottle-cap',
      category: 'Food, Drink & Hospitality',
      pack: 'beer-icons',
      path: 'Food, Drink & Hospitality/beer-icons/bottle-cap.svg',
      url: 'https://cdn.jsdelivr.net/gh/nsdevaraj/SVGDepot@abc123/Food%2C%20Drink%20%26%20Hospitality/beer-icons/bottle-cap.svg',
    },
  );
  assert.equal(Number.isFinite(output.classes[0].candidates[0].score), true);
});

test('preferred categories outrank noisy library matches', () => {
  const index = createIndex({
    categories: ['UI, Interface & Design Systems', 'Nature, Animals & Environment'],
    packs: [[0, 'interface-icons'], [1, 'weather-icons']],
    icons: [[0, 'cloud-upload.svg'], [1, 'cloud.svg']],
    tokens: { cloud: [0, 1], upload: [0] },
  });

  const { result, output } = runGenerator({
    index,
    categories: ['cloud'],
    rules: {
      preferredCategories: { cloud: ['Nature, Animals & Environment'] },
      categoryPenalties: { 'UI, Interface & Design Systems': -30 },
    },
  });

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(
    output.classes[0].candidates.map(candidate => candidate.path),
    [
      'Nature, Animals & Environment/weather-icons/cloud.svg',
      'UI, Interface & Design Systems/interface-icons/cloud-upload.svg',
    ],
  );
});

test('specific alias phrases outrank broad fallback aliases', () => {
  const index = createIndex({
    categories: ['Sports, Games & Entertainment'],
    packs: [[0, 'sports-icons']],
    icons: [[0, 'badminton-racket.svg'], [0, 'tennis-racket.svg']],
    tokens: { badminton: [0], racket: [0, 1], tennis: [1] },
  });

  const { result, output } = runGenerator({
    index,
    categories: ['tennis racquet'],
    rules: { aliases: { 'tennis racquet': ['racket', 'tennis racket'] } },
  });

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(
    output.classes[0].candidates.map(candidate => candidate.path),
    [
      'Sports, Games & Entertainment/sports-icons/tennis-racket.svg',
      'Sports, Games & Entertainment/sports-icons/badminton-racket.svg',
    ],
  );
});

test('an exact filename outranks a longer name containing the class token', () => {
  const index = createIndex({
    categories: ['Tools, Objects & Utilities'],
    packs: [[0, 'object-icons']],
    icons: [[0, 'camera-off.svg'], [0, 'camera.svg']],
    tokens: { camera: [0, 1], off: [0] },
  });

  const { result, output } = runGenerator({
    index,
    categories: ['camera'],
    rules: {},
  });

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(
    output.classes[0].candidates.map(candidate => candidate.name),
    ['camera', 'camera-off'],
  );
});

test('earlier aliases outrank equally specific fallback aliases', () => {
  const index = createIndex({
    categories: ['Travel & Transportation'],
    packs: [[0, 'ship-icons']],
    icons: [[0, 'ship.svg'], [0, 'battleship.svg']],
    tokens: { battleship: [1], ship: [0] },
  });

  const { result, output } = runGenerator({
    index,
    categories: ['aircraft carrier'],
    rules: { aliases: { 'aircraft carrier': ['battleship', 'ship'] } },
  });

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(
    output.classes[0].candidates.map(candidate => candidate.name),
    ['battleship', 'ship'],
  );
});

test('class-specific excluded tokens remove false-positive candidates', () => {
  const index = createIndex({
    categories: ['Sports, Games & Entertainment'],
    packs: [[0, 'playground-icons']],
    icons: [
      [0, 'playground.svg'],
      [0, 'swingset-playground.svg'],
      [0, 'code-playground.svg'],
    ],
    tokens: { code: [2], playground: [0, 1, 2], swingset: [1] },
  });

  const { result, output } = runGenerator({
    index,
    categories: ['see saw'],
    rules: {
      aliases: { 'see saw': ['playground'] },
      excludedTokens: { 'see saw': ['code', 'swingset'] },
    },
  });

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(
    output.classes[0].candidates.map(candidate => candidate.name),
    ['playground'],
  );
});

test('an unsupported SVGDepot index schema fails clearly', () => {
  const index = createIndex({
    schemaVersion: 2,
  });

  const { result } = runGenerator({
    index,
    categories: ['cat'],
    rules: {},
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unsupported SVGDepot index schema version: 2/);
});

test('a rule for an unknown Quick Draw class fails clearly', () => {
  const index = createIndex();

  const { result } = runGenerator({
    index,
    categories: ['cat'],
    rules: { aliases: { cot: ['cat'] } },
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unknown aliases class: cot/);
});

test('a lookalike GitHub host is rejected', () => {
  const index = createIndex({
    source: { repository: 'https://notgithub.com/nsdevaraj/SVGDepot.git', ref: 'main', commit: 'abc123' },
  });

  const { result } = runGenerator({
    index,
    categories: ['cat'],
    rules: {},
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unsupported repository URL/);
});

test('malformed class rule values fail clearly', () => {
  const index = createIndex();

  const { result } = runGenerator({
    index,
    categories: ['cat'],
    rules: { aliases: { cat: 'feline' } },
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /aliases for cat must be an array of strings/);
});