#!/usr/bin/env node

import { readSync, writeSync } from 'node:fs';

import { rasterizePolylines } from '../src/sketch-rasterizer.mjs';
import { svgToPolylines } from '../src/svg-to-polylines.mjs';

const HEADER_BYTES = 4;
const MAX_SOURCE_BYTES = 1024 * 1024;

// Status byte prefixed to every response so the caller can skip an icon without
// guessing why it produced no usable raster.
const STATUS = Object.freeze({
  ok: 0,
  unreadable: 1,
  empty: 2,
  inkOutOfBand: 3,
});

function parseArgs(argv) {
  const options = {
    size: 64,
    padding: 4,
    strokeWidth: 2.5,
    supersample: 4,
    minInk: 0.01,
    maxInk: 0.6,
  };
  const optionKeys = {
    '--size': 'size',
    '--padding': 'padding',
    '--stroke-width': 'strokeWidth',
    '--supersample': 'supersample',
    '--min-ink': 'minInk',
    '--max-ink': 'maxInk',
  };

  for (let index = 0; index < argv.length; index += 2) {
    const key = optionKeys[argv[index]];
    const value = argv[index + 1];
    if (!key || value === undefined) throw new Error(`Invalid argument: ${argv[index] ?? ''}`);
    options[key] = Number(value);
  }

  if (!(options.minInk >= 0) || !(options.maxInk > options.minInk) || options.maxInk > 1) {
    throw new Error('Ink coverage band must satisfy 0 <= min-ink < max-ink <= 1');
  }
  return options;
}

function readExact(length, allowCleanEof = false) {
  const buffer = Buffer.allocUnsafe(length);
  let offset = 0;
  while (offset < length) {
    const bytesRead = readSync(0, buffer, offset, length - offset, null);
    if (bytesRead === 0) {
      if (allowCleanEof && offset === 0) return null;
      throw new Error('Unexpected end of input');
    }
    offset += bytesRead;
  }
  return buffer;
}

function inkCoverage(bitmap) {
  let total = 0;
  for (const value of bitmap) total += value;
  return total / (bitmap.length * 255);
}

function rasterizeSource(source, rasterOptions, band) {
  let polylines;
  try {
    polylines = svgToPolylines(source);
  } catch {
    return { status: STATUS.unreadable, bitmap: null };
  }
  if (polylines.length === 0) return { status: STATUS.empty, bitmap: null };

  let bitmap;
  try {
    bitmap = rasterizePolylines(polylines, rasterOptions);
  } catch {
    return { status: STATUS.unreadable, bitmap: null };
  }

  const coverage = inkCoverage(bitmap);
  if (coverage < band.minInk || coverage > band.maxInk) {
    return { status: STATUS.inkOutOfBand, bitmap: null };
  }
  return { status: STATUS.ok, bitmap };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const rasterOptions = {
    size: options.size,
    padding: options.padding,
    strokeWidth: options.strokeWidth,
    supersample: options.supersample,
  };
  // Fail fast on bad options; per-icon raster errors are caught and reported as a status.
  rasterizePolylines([], rasterOptions);

  while (true) {
    const header = readExact(HEADER_BYTES, true);
    if (header === null) return;
    const sourceBytes = header.readUInt32LE(0);
    if (sourceBytes < 1 || sourceBytes > MAX_SOURCE_BYTES) {
      throw new Error(`Source length must be from 1 to ${MAX_SOURCE_BYTES} bytes`);
    }

    const payload = readExact(sourceBytes);
    const { status, bitmap } = rasterizeSource(
      payload.toString('utf8'),
      rasterOptions,
      { minInk: options.minInk, maxInk: options.maxInk },
    );
    writeSync(1, Buffer.from([status]));
    if (bitmap !== null) writeSync(1, bitmap);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.message || error}\n`);
  process.exitCode = 1;
}
