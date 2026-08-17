#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { TOKENIZER_ID, tokenize } from './lib/tokenize.mjs';

const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const DEFAULT_REPOSITORY = 'https://github.com/nsdevaraj/SVGDepot.git';
const DEFAULT_REF = 'main';
const DEFAULT_REPO_DIR = resolve(PROJECT_ROOT, '.cache/svgdepot');
const DEFAULT_OUTPUT = resolve(PROJECT_ROOT, 'data/svgdepot-index.json.gz');

function printHelp() {
  console.log(`Build a compact filename-token index for SVGDepot.

Usage:
  node scripts/build-svgdepot-index.mjs [options]

Options:
  --repo-dir <path>   Existing or destination blobless clone
  --output <path>     Output .json or .json.gz file
  --repository <url>  Git repository URL
  --ref <name>        Branch or tag to index (default: main)
  --refresh           Fetch the latest requested ref before indexing
  --help              Show this help
`);
}

function parseArgs(argv) {
  const options = {
    repoDir: DEFAULT_REPO_DIR,
    output: DEFAULT_OUTPUT,
    repository: DEFAULT_REPOSITORY,
    ref: DEFAULT_REF,
    refresh: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];

    if (argument === '--help') {
      printHelp();
      process.exit(0);
    }

    if (argument === '--refresh') {
      options.refresh = true;
      continue;
    }

    const keyByArgument = {
      '--repo-dir': 'repoDir',
      '--output': 'output',
      '--repository': 'repository',
      '--ref': 'ref',
    };
    const key = keyByArgument[argument];
    if (!key) throw new Error(`Unknown option: ${argument}`);

    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`Missing value for ${argument}`);
    }

    options[key] = key === 'repoDir' || key === 'output'
      ? resolve(process.cwd(), value)
      : value;
    index += 1;
  }

  return options;
}

function runGit(arguments_, options = {}) {
  return execFileSync('git', arguments_, {
    cwd: options.cwd,
    encoding: options.encoding ?? 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    stdio: options.stdio,
  });
}

function ensureRepository({ repoDir, repository, ref, refresh }) {
  if (!existsSync(repoDir)) {
    mkdirSync(dirname(repoDir), { recursive: true });
    const temporaryDirectory = `${repoDir}.tmp-${process.pid}`;

    try {
      runGit([
        'clone',
        '--filter=blob:none',
        '--no-checkout',
        '--depth=1',
        '--branch', ref,
        repository,
        temporaryDirectory,
      ], { stdio: 'inherit' });
      renameSync(temporaryDirectory, repoDir);
    } catch (error) {
      rmSync(temporaryDirectory, { recursive: true, force: true });
      throw error;
    }
  } else if (!existsSync(resolve(repoDir, '.git'))) {
    throw new Error(`${repoDir} exists but is not a Git repository`);
  }

  if (refresh) {
    runGit(['fetch', '--filter=blob:none', '--depth=1', 'origin', ref], {
      cwd: repoDir,
      stdio: 'inherit',
    });
    runGit(['update-ref', 'HEAD', 'FETCH_HEAD'], { cwd: repoDir });
  }
}

function tokenizeFilename(filename) {
  return tokenize(filename.replace(/\.svg$/i, ''));
}

function buildIndex(svgPaths, source) {
  const categories = [];
  const categoryIds = new Map();
  const packs = [];
  const packIds = new Map();
  const icons = [];
  const postings = new Map();

  for (const svgPath of svgPaths) {
    const segments = svgPath.split('/');
    const filename = segments.pop();
    const category = segments.shift() ?? '';
    const pack = segments.join('/');

    let categoryId = categoryIds.get(category);
    if (categoryId === undefined) {
      categoryId = categories.length;
      categoryIds.set(category, categoryId);
      categories.push(category);
    }

    const packKey = `${category}\0${pack}`;
    let packId = packIds.get(packKey);
    if (packId === undefined) {
      packId = packs.length;
      packIds.set(packKey, packId);
      packs.push([categoryId, pack]);
    }

    const iconId = icons.length;
    icons.push([packId, filename]);

    for (const token of tokenizeFilename(filename)) {
      const iconIds = postings.get(token);
      if (iconIds) iconIds.push(iconId);
      else postings.set(token, [iconId]);
    }
  }

  const tokens = Object.fromEntries(
    [...postings.entries()].sort(([left], [right]) => left.localeCompare(right)),
  );

  return {
    schemaVersion: 1,
    tokenizer: TOKENIZER_ID,
    source: {
      ...source,
      iconCount: icons.length,
      categoryCount: categories.length,
      packCount: packs.length,
      tokenCount: postings.size,
    },
    categories,
    packs,
    icons,
    tokens,
  };
}

function writeIndex(index, output) {
  mkdirSync(dirname(output), { recursive: true });

  const json = `${JSON.stringify(index)}\n`;
  const payload = output.endsWith('.gz')
    ? gzipSync(json, { level: 9 })
    : Buffer.from(json);
  const temporaryOutput = `${output}.tmp-${process.pid}`;

  writeFileSync(temporaryOutput, payload);
  renameSync(temporaryOutput, output);

  return {
    bytes: payload.length,
    sha256: createHash('sha256').update(payload).digest('hex'),
  };
}

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  ensureRepository(options);

  const commit = runGit(['rev-parse', 'HEAD'], { cwd: options.repoDir }).trim();
  const tree = runGit(['ls-tree', '-r', '-z', '--name-only', 'HEAD'], {
    cwd: options.repoDir,
    encoding: 'buffer',
  });
  const svgPaths = tree
    .toString('utf8')
    .split('\0')
    .filter(path => /\.svg$/i.test(path));

  if (svgPaths.length === 0) {
    throw new Error(`No SVG files found at ${commit}`);
  }

  const index = buildIndex(svgPaths, {
    repository: options.repository,
    ref: options.ref,
    commit,
  });
  const written = writeIndex(index, options.output);

  console.log(`Indexed ${index.source.iconCount.toLocaleString()} SVGs`);
  console.log(`Categories: ${index.source.categoryCount.toLocaleString()}`);
  console.log(`Packs: ${index.source.packCount.toLocaleString()}`);
  console.log(`Tokens: ${index.source.tokenCount.toLocaleString()}`);
  console.log(`Output: ${options.output} (${formatBytes(written.bytes)})`);
  console.log(`SHA-256: ${written.sha256}`);
}

try {
  main();
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
}