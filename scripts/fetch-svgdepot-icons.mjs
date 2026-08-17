#!/usr/bin/env node

// Downloads every SVGDepot icon from the commit-pinned jsDelivr CDN into a
// content-addressed store, so training never depends on a full repository checkout.

import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  statSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { dirname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';

import { expectedIconUrl, isConfinedSvgPath } from '../src/icon-candidate.mjs';
import {
  IconRejectedError,
  UNAVAILABLE_DIGEST,
  assertTrustedIconResponse,
  buildIconManifest,
  candidateIconIds,
  iconPathsFromIndex,
  selectIconIds,
} from '../src/svgdepot-corpus.mjs';

const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const DEFAULT_INDEX = resolve(PROJECT_ROOT, 'data/svgdepot-index.json.gz');
const DEFAULT_OUTPUT = resolve(PROJECT_ROOT, 'data/svgdepot-icon-manifest.json.gz');
const DEFAULT_STORE = resolve(PROJECT_ROOT, '.cache/svgdepot-icons');
const USER_AGENT = 'autoDraw-svgdepot-fetcher/1';

function printHelp() {
  console.log(`Download SVGDepot icons from the commit-pinned jsDelivr CDN.

Usage:
  node scripts/fetch-svgdepot-icons.mjs [options]

Options:
  --index <path>          SVGDepot token index (.json or .json.gz)
  --output <path>         Icon manifest output (.json or .json.gz)
  --store <path>          Content-addressed SVG store
  --candidates <path>     Fetch only the icons referenced by a candidate manifest
  --source-root <path>    Read icons from a local checkout instead of the CDN
  --concurrency <count>   Parallel requests (default: 8, maximum: 32)
  --retries <count>       Retries per icon before giving up (default: 4)
  --max-bytes <count>     Largest accepted SVG in bytes (default: 524288)
  --limit-per-pack <n>    Fetch only the first n icons of each pack
  --force                 Re-download icons already recorded in the journal
  --help                  Show this help

The checkout passed to --source-root must sit on the commit the index pins, because
the icon URLs the browser is allowed to load are rebuilt from that commit.
`);
}

function parseArgs(argv) {
  const options = {
    index: DEFAULT_INDEX,
    output: DEFAULT_OUTPUT,
    store: DEFAULT_STORE,
    candidates: null,
    sourceRoot: null,
    concurrency: 8,
    retries: 4,
    maxBytes: 512 * 1024,
    limitPerPack: Number.POSITIVE_INFINITY,
    force: false,
  };
  const pathKeys = {
    '--index': 'index',
    '--output': 'output',
    '--store': 'store',
    '--candidates': 'candidates',
    '--source-root': 'sourceRoot',
  };
  const numberKeys = {
    '--concurrency': 'concurrency',
    '--retries': 'retries',
    '--max-bytes': 'maxBytes',
    '--limit-per-pack': 'limitPerPack',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help') {
      printHelp();
      process.exit(0);
    }
    if (argument === '--force') {
      options.force = true;
      continue;
    }

    const value = argv[index + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for ${argument}`);
    }
    if (pathKeys[argument]) options[pathKeys[argument]] = resolve(process.cwd(), value);
    else if (numberKeys[argument]) options[numberKeys[argument]] = Number(value);
    else throw new Error(`Unknown option: ${argument}`);
    index += 1;
  }

  if (!Number.isInteger(options.concurrency) || options.concurrency < 1 || options.concurrency > 32) {
    throw new Error('--concurrency must be an integer from 1 to 32');
  }
  if (!Number.isInteger(options.retries) || options.retries < 0 || options.retries > 10) {
    throw new Error('--retries must be an integer from 0 to 10');
  }
  if (!Number.isInteger(options.maxBytes) || options.maxBytes < 1024) {
    throw new Error('--max-bytes must be an integer of at least 1024');
  }
  if (options.limitPerPack !== Number.POSITIVE_INFINITY
    && (!Number.isInteger(options.limitPerPack) || options.limitPerPack < 1)) {
    throw new Error('--limit-per-pack must be a positive integer');
  }
  if (options.sourceRoot !== null) {
    if (!statSync(options.sourceRoot, { throwIfNoEntry: false })?.isDirectory()) {
      throw new Error(`--source-root is not a directory: ${options.sourceRoot}`);
    }
    options.sourceRoot = realpathSync(options.sourceRoot);
  }
  return options;
}

function readJson(path) {
  const contents = readFileSync(path);
  return JSON.parse((path.endsWith('.gz') ? gunzipSync(contents) : contents).toString('utf8'));
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const json = `${JSON.stringify(value)}\n`;
  const payload = path.endsWith('.gz') ? gzipSync(json, { level: 9 }) : Buffer.from(json);
  const temporaryPath = `${path}.tmp-${process.pid}`;
  writeFileSync(temporaryPath, payload);
  renameSync(temporaryPath, path);
  return payload.length;
}

function blobPath(store, digest) {
  return resolve(store, digest.slice(0, 2), `${digest}.svg`);
}

// An append-only journal, because the manifest is only written once the whole run finishes
// and a 210k-icon fetch is far too long to restart from nothing.
function journalPath(store, commit) {
  return resolve(store, `journal-${commit}.tsv`);
}

function readJournal(store, commit, force) {
  const path = journalPath(store, commit);
  const resumed = new Map();
  if (force || !existsSync(path)) return resumed;

  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const [rawIconId, digest] = line.split('\t');
    const iconId = Number(rawIconId);
    if (!Number.isInteger(iconId) || digest === undefined) continue;
    if (digest === '-') resumed.set(iconId, null);
    else if (/^[0-9a-f]{64}$/.test(digest)) resumed.set(iconId, digest);
  }
  return resumed;
}

function openJournal(store, commit, force) {
  mkdirSync(store, { recursive: true });
  const path = journalPath(store, commit);
  if (force && existsSync(path)) writeFileSync(path, '');
  return openSync(path, 'a');
}

function retryDelay(response, attempt) {
  const header = Number(response?.headers.get('retry-after'));
  if (Number.isFinite(header) && header > 0) return Math.min(header, 60) * 1000;
  return Math.min(1000 * 2 ** attempt, 30000) + Math.floor(Math.random() * 250);
}

const wait = milliseconds => new Promise(resolve_ => {
  setTimeout(resolve_, milliseconds);
});

// jsDelivr answers 403 when it is shedding load, so it has to be retried like a 429.
const RETRYABLE_STATUSES = new Set([403, 408, 425, 429]);

async function downloadIcon(url, options) {
  for (let attempt = 0; ; attempt += 1) {
    let response = null;
    try {
      response = await fetch(url, { headers: { 'user-agent': USER_AGENT, accept: 'image/svg+xml' } });
      if (response.status === 404 || response.status === 410) return null;
      if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);

      assertTrustedIconResponse(response, options.maxBytes);
      const body = Buffer.from(await response.arrayBuffer());
      if (body.length === 0 || body.length > options.maxBytes) {
        throw new IconRejectedError(`Icon size ${body.length} is out of range: ${url}`);
      }
      return body;
    } catch (error) {
      if (error instanceof IconRejectedError) {
        process.stderr.write(`skipped: ${error.message}\n`);
        return null;
      }
      const retryable = response === null
        || RETRYABLE_STATUSES.has(response.status)
        || response.status >= 500;
      if (!retryable || attempt >= options.retries) throw error;
      await wait(retryDelay(response, attempt));
    }
  }
}

function storeIcon(store, body) {
  const digest = createHash('sha256').update(body).digest('hex');
  const target = blobPath(store, digest);
  if (!existsSync(target)) {
    mkdirSync(dirname(target), { recursive: true });
    const temporaryPath = `${target}.tmp-${process.pid}`;
    writeFileSync(temporaryPath, body);
    renameSync(temporaryPath, target);
  }
  return digest;
}

// Reads a checkout the same way the CDN path does, so a symlinked or traversing index
// entry cannot pull bytes from outside the repository it claims to describe.
function readLocalIcon(sourceRoot, iconPath, maxBytes) {
  if (!isConfinedSvgPath(iconPath)) {
    throw new IconRejectedError(`Icon path is not confined: ${iconPath}`);
  }
  const target = resolve(sourceRoot, iconPath);
  if (target !== sourceRoot && !target.startsWith(sourceRoot + sep)) {
    throw new IconRejectedError(`Icon path escapes the source root: ${iconPath}`);
  }

  let resolved;
  try {
    resolved = realpathSync(target);
  } catch {
    return null;
  }
  if (!resolved.startsWith(sourceRoot + sep)) {
    throw new IconRejectedError(`Icon path resolves outside the source root: ${iconPath}`);
  }
  const stats = statSync(resolved, { throwIfNoEntry: false });
  if (!stats?.isFile()) return null;
  if (stats.size === 0 || stats.size > maxBytes) {
    throw new IconRejectedError(`Icon size ${stats.size} is out of range: ${iconPath}`);
  }
  return readFileSync(resolved);
}

async function loadIcon(iconPath, source, options) {
  if (options.sourceRoot === null) {
    const url = expectedIconUrl(source, iconPath);
    if (url === null) throw new Error(`Cannot build a pinned URL for ${iconPath}`);
    return downloadIcon(url, options);
  }
  try {
    return readLocalIcon(options.sourceRoot, iconPath, options.maxBytes);
  } catch (error) {
    if (error instanceof IconRejectedError) {
      process.stderr.write(`skipped: ${error.message}\n`);
      return null;
    }
    throw error;
  }
}

// Indexing a checkout that has drifted would pair local artwork with URLs the browser
// resolves against a different commit, so the mismatch has to stop the run.
function assertCheckoutCommit(sourceRoot, commit) {
  const result = spawnSync('git', ['-C', sourceRoot, 'rev-parse', 'HEAD'], { encoding: 'utf8' });
  if (result.status !== 0) {
    throw new Error(`Cannot read the checkout commit at ${sourceRoot}: ${result.stderr?.trim() || 'git failed'}`);
  }
  const head = result.stdout.trim();
  if (head !== commit) {
    throw new Error(`Checkout is on ${head} but the index pins ${commit}`);
  }
}

async function fetchAll(iconIds, icons, source, options) {
  const digestIds = new Map();
  const digests = [];
  const iconDigests = new Array(icons.length).fill(UNAVAILABLE_DIGEST);
  const counts = { cached: 0, fetched: 0, unavailable: 0, failed: 0 };
  const resumed = readJournal(options.store, source.commit, options.force);
  const journal = openJournal(options.store, source.commit, options.force);

  const digestId = digest => {
    let id = digestIds.get(digest);
    if (id === undefined) {
      id = digests.length;
      digestIds.set(digest, id);
      digests.push(digest);
    }
    return id;
  };

  const record = (iconId, digest) => {
    if (digest !== null) iconDigests[iconId] = digestId(digest);
    writeSync(journal, `${iconId}\t${digest ?? '-'}\n`);
  };

  const reusableDigest = iconId => {
    if (!resumed.has(iconId)) return undefined;
    const digest = resumed.get(iconId);
    if (digest === null) return null;
    return existsSync(blobPath(options.store, digest)) ? digest : undefined;
  };

  let cursor = 0;
  let completed = 0;
  const total = iconIds.length;

  const worker = async () => {
    while (cursor < total) {
      const iconId = iconIds[cursor];
      cursor += 1;

      const cachedDigest = reusableDigest(iconId);
      if (cachedDigest !== undefined) {
        if (cachedDigest === null) {
          counts.unavailable += 1;
        } else {
          iconDigests[iconId] = digestId(cachedDigest);
          counts.cached += 1;
        }
      } else {
        try {
          const body = await loadIcon(icons[iconId].path, source, options);
          if (body === null) {
            counts.unavailable += 1;
            record(iconId, null);
          } else {
            record(iconId, storeIcon(options.store, body));
            counts.fetched += 1;
          }
        } catch (error) {
          counts.failed += 1;
          process.stderr.write(`${icons[iconId].path}: ${error.message || error}\n`);
        }
      }

      completed += 1;
      if (completed % 2000 === 0 || completed === total) {
        console.log(`Fetched ${completed.toLocaleString()}/${total.toLocaleString()} icons`);
      }
    }
  };

  try {
    await Promise.all(Array.from({ length: Math.min(options.concurrency, total) }, worker));
  } finally {
    closeSync(journal);
  }
  return { counts, digests, iconDigests };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const index = readJson(options.index);
  const icons = iconPathsFromIndex(index);
  const source = index.source;
  if (options.sourceRoot !== null) assertCheckoutCommit(options.sourceRoot, source.commit);
  const iconIds = options.candidates === null
    ? selectIconIds(icons, options.limitPerPack)
    : candidateIconIds(readJson(options.candidates), icons.length, source.commit);

  console.log(`Fetching ${iconIds.length.toLocaleString()} of ${icons.length.toLocaleString()} icons`);
  console.log(`Commit: ${source.commit}`);
  console.log(`Source: ${options.sourceRoot ?? 'jsDelivr CDN'}`);

  const { counts, digests, iconDigests } = await fetchAll(iconIds, icons, source, options);
  const bytes = writeJson(options.output, buildIconManifest({
    source,
    iconCount: icons.length,
    fetcher: {
      maxBytes: options.maxBytes,
      limitPerPack: Number.isFinite(options.limitPerPack) ? options.limitPerPack : null,
      candidatesOnly: options.candidates !== null,
      localCheckout: options.sourceRoot !== null,
    },
    counts: { ...counts, selected: iconIds.length },
    digests,
    iconDigests,
  }));

  console.log(`Downloaded: ${counts.fetched.toLocaleString()}`);
  console.log(`Reused from cache: ${counts.cached.toLocaleString()}`);
  console.log(`Unique contents: ${digests.length.toLocaleString()}`);
  console.log(`Missing at the source: ${counts.unavailable.toLocaleString()}`);
  console.log(`Failed: ${counts.failed.toLocaleString()}`);
  console.log(`Store: ${options.store}`);
  console.log(`Manifest: ${options.output} (${bytes.toLocaleString()} bytes)`);

  if (counts.failed > 0) process.exitCode = 1;
}

try {
  await main();
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
}
