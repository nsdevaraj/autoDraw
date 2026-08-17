#!/usr/bin/env node

import { readSync, writeSync } from 'node:fs';

import { rasterizeStroke3 } from '../src/sketch-rasterizer.mjs';

const HEADER_BYTES = 4;
const VALUES_PER_POINT = 3;
const BYTES_PER_VALUE = 2;
const MAX_POINTS = 10000;

function parseArgs(argv) {
  const options = {
    size: 64,
    padding: 4,
    strokeWidth: 2.5,
    supersample: 4,
  };
  const optionKeys = {
    '--size': 'size',
    '--padding': 'padding',
    '--stroke-width': 'strokeWidth',
    '--supersample': 'supersample',
  };

  for (let index = 0; index < argv.length; index += 2) {
    const key = optionKeys[argv[index]];
    const value = argv[index + 1];
    if (!key || value === undefined) throw new Error(`Invalid argument: ${argv[index] ?? ''}`);
    options[key] = Number(value);
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

function decodeStroke3(payload, pointCount) {
  const drawing = new Array(pointCount);
  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const offset = pointIndex * VALUES_PER_POINT * BYTES_PER_VALUE;
    drawing[pointIndex] = [
      payload.readInt16LE(offset),
      payload.readInt16LE(offset + BYTES_PER_VALUE),
      payload.readInt16LE(offset + BYTES_PER_VALUE * 2),
    ];
  }
  return drawing;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  while (true) {
    const header = readExact(HEADER_BYTES, true);
    if (header === null) return;
    const pointCount = header.readUInt32LE(0);
    if (pointCount < 1 || pointCount > MAX_POINTS) {
      throw new Error(`Point count must be from 1 to ${MAX_POINTS}`);
    }
    const payload = readExact(pointCount * VALUES_PER_POINT * BYTES_PER_VALUE);
    const bitmap = rasterizeStroke3(decodeStroke3(payload, pointCount), options);
    writeSync(1, bitmap);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.message || error}\n`);
  process.exitCode = 1;
}
