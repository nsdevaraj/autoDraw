#!/usr/bin/env node

import { copyFile, mkdir, readdir, rm, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { basename, dirname, resolve } from 'node:path';

import {
  assertSafeBuildOutput,
  cleanupFailedBuildOutput,
  publishBuildOutput,
} from '../src/vercel-build-output.mjs';

const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const DEFAULT_OUTPUT = resolve(PROJECT_ROOT, 'dist');

const STATIC_ASSETS = Object.freeze([
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

// The icon index is hundreds of generated shards, so it is copied as a tree rather than
// being pinned file by file.
const STATIC_DIRECTORIES = Object.freeze([
  ['data/icon-embeddings', 'data/icon-embeddings'],
]);

function printHelp() {
  console.log(`Build the static Vercel output.

Usage:
  node scripts/build-vercel.mjs [options]

Options:
  --output <path>  Output directory (default: dist)
  --help           Show this help
`);
}

function parseArgs(argv) {
  let output = DEFAULT_OUTPUT;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') {
      printHelp();
      process.exit(0);
    }
    if (argument !== '--output') throw new Error(`Unknown option: ${argument}`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error('Missing value for --output');
    output = resolve(process.cwd(), value);
    index += 1;
  }
  return {
    output: assertSafeBuildOutput({
      projectRoot: PROJECT_ROOT,
      defaultOutput: DEFAULT_OUTPUT,
      output,
    }),
  };
}

async function copyAsset(temporaryOutput, sourceName, destinationName) {
  const source = resolve(PROJECT_ROOT, sourceName);
  const sourceStats = await stat(source);
  if (!sourceStats.isFile()) throw new Error(`Vercel asset is not a file: ${sourceName}`);
  const destination = resolve(temporaryOutput, destinationName);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
  return sourceStats.size;
}

async function copyDirectory(temporaryOutput, sourceName, destinationName) {
  const entries = await readdir(resolve(PROJECT_ROOT, sourceName), { withFileTypes: true });
  let byteCount = 0;
  let fileCount = 0;
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    if (!entry.isFile()) throw new Error(`Vercel asset is not a file: ${sourceName}/${entry.name}`);
    byteCount += await copyAsset(
      temporaryOutput,
      `${sourceName}/${entry.name}`,
      `${destinationName}/${entry.name}`,
    );
    fileCount += 1;
  }
  return { byteCount, fileCount };
}

export async function buildVercelOutput(output) {
  const resolvedOutput = assertSafeBuildOutput({
    projectRoot: PROJECT_ROOT,
    defaultOutput: DEFAULT_OUTPUT,
    output,
  });
  const temporaryOutput = resolve(
    dirname(resolvedOutput),
    `.${basename(resolvedOutput)}.tmp-${process.pid}`,
  );
  await rm(temporaryOutput, { recursive: true, force: true });
  await mkdir(temporaryOutput, { recursive: true });

  let byteCount = 0;
  let fileCount = STATIC_ASSETS.length;
  try {
    for (const [source, destination] of STATIC_ASSETS) {
      byteCount += await copyAsset(temporaryOutput, source, destination);
    }
    for (const [source, destination] of STATIC_DIRECTORIES) {
      const copied = await copyDirectory(temporaryOutput, source, destination);
      byteCount += copied.byteCount;
      fileCount += copied.fileCount;
    }
    await publishBuildOutput({ output: resolvedOutput, temporaryOutput });
  } catch (error) {
    await cleanupFailedBuildOutput({
      temporaryOutput,
      primaryError: error,
    });
  }
  return { byteCount, fileCount, output: resolvedOutput };
}

try {
  const { output } = parseArgs(process.argv.slice(2));
  const result = await buildVercelOutput(output);
  console.log(
    `Built Vercel static output: ${result.output} `
    + `(${result.fileCount} files, ${result.byteCount.toLocaleString()} bytes)`,
  );
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
}