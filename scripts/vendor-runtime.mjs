#!/usr/bin/env node

// The bundled dev server maps /vendor/ onto node_modules, but plain static servers
// (http-server, python -m http.server, ...) can only serve real files, so the browser
// runtime is copied to ./vendor for them.

import { copyFile, mkdir, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const RUNTIME_SOURCE = 'node_modules/onnxruntime-web/dist';
const RUNTIME_FILES = Object.freeze([
  'ort.wasm.min.mjs',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
]);

async function copyRuntime() {
  const target = resolve(PROJECT_ROOT, 'vendor');
  await mkdir(target, { recursive: true });

  for (const name of RUNTIME_FILES) {
    const source = resolve(PROJECT_ROOT, RUNTIME_SOURCE, name);
    try {
      await stat(source);
    } catch {
      throw new Error(`Missing ${RUNTIME_SOURCE}/${name}. Run "npm install" first.`);
    }
    await copyFile(source, resolve(target, name));
    console.log(`vendor/${name}`);
  }
}

try {
  await copyRuntime();
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
}
