#!/usr/bin/env node

import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { createCurationHandler } from '../src/curation-server.mjs';

const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 4173;

function printHelp() {
  console.log(`Serve AutoDraw Local and the Quick Draw curation tool.

Usage:
  node scripts/serve-curation.mjs [options]

Options:
  --host <host>  Bind host (default: ${DEFAULT_HOST})
  --port <port>  Bind port (default: ${DEFAULT_PORT})
  --help         Show this help
`);
}

function parseArgs(argv) {
  const options = { host: DEFAULT_HOST, port: DEFAULT_PORT };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') {
      printHelp();
      process.exit(0);
    }

    const value = argv[index + 1];
    if (!value) throw new Error(`Missing value for ${argument}`);
    if (argument === '--host') options.host = value;
    else if (argument === '--port') options.port = Number(value);
    else throw new Error(`Unknown option: ${argument}`);
    index += 1;
  }

  if (!Number.isInteger(options.port) || options.port < 1 || options.port > 65535) {
    throw new Error('--port must be an integer from 1 to 65535');
  }
  return options;
}

try {
  const options = parseArgs(process.argv.slice(2));
  const server = createServer(createCurationHandler(PROJECT_ROOT));
  server.listen(options.port, options.host, () => {
    console.log(`AutoDraw Local: http://${options.host}:${options.port}/`);
    console.log(`Quick Draw Curator: http://${options.host}:${options.port}/curate.html`);
  });
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
}