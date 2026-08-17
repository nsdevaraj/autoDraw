#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { parseLineList } from '../src/line-list.mjs';
import { TOKENIZER_ID, tokenize } from './lib/tokenize.mjs';

const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const DEFAULT_INDEX = resolve(PROJECT_ROOT, 'data/svgdepot-index.json.gz');
const DEFAULT_CATEGORIES = resolve(PROJECT_ROOT, 'config/quickdraw-categories.txt');
const DEFAULT_RULES = resolve(PROJECT_ROOT, 'config/quickdraw-icon-rules.json');
const DEFAULT_OUTPUT = resolve(PROJECT_ROOT, 'data/quickdraw-candidates.json');
const DEFAULT_LIMIT = 12;
const INDEX_SCHEMA_VERSION = 1;

function printHelp() {
  console.log(`Build ranked SVGDepot candidates for every Quick Draw class.

Usage:
  node scripts/build-quickdraw-candidates.mjs [options]

Options:
  --index <path>       SVGDepot token index (.json or .json.gz)
  --categories <path>  Quick Draw category list
  --rules <path>       Alias and category scoring rules
  --output <path>      Candidate manifest output
  --limit <count>      Maximum candidates per class (default: 12)
  --help               Show this help
`);
}

function parseArgs(argv) {
  const options = {
    index: DEFAULT_INDEX,
    categories: DEFAULT_CATEGORIES,
    rules: DEFAULT_RULES,
    output: DEFAULT_OUTPUT,
    limit: DEFAULT_LIMIT,
  };
  const keys = {
    '--index': 'index',
    '--categories': 'categories',
    '--rules': 'rules',
    '--output': 'output',
    '--limit': 'limit',
  };

  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--help') {
      printHelp();
      process.exit(0);
    }

    const key = keys[argv[index]];
    const value = argv[index + 1];
    if (!key || !value) throw new Error(`Invalid argument: ${argv[index] ?? ''}`);
    options[key] = key === 'limit' ? Number(value) : resolve(process.cwd(), value);
    index += 1;
  }

  if (!Number.isInteger(options.limit) || options.limit < 1) {
    throw new Error('--limit must be a positive integer');
  }

  return options;
}

function readJson(path) {
  const contents = readFileSync(path);
  const json = path.endsWith('.gz') ? gunzipSync(contents) : contents;
  return JSON.parse(json.toString('utf8'));
}

function validateIndex(index) {
  if (index.schemaVersion !== INDEX_SCHEMA_VERSION) {
    throw new Error(`Unsupported SVGDepot index schema version: ${index.schemaVersion}`);
  }
  if (index.tokenizer !== TOKENIZER_ID) {
    throw new Error(`Unsupported SVGDepot index tokenizer: ${index.tokenizer}`);
  }
}

function validateRules(index, categories, rules) {
  const knownClasses = new Set(categories);
  const knownCategories = new Set(index.categories);
  for (const ruleGroup of ['aliases', 'preferredCategories', 'excludedTokens']) {
    for (const [name, values] of Object.entries(rules[ruleGroup] ?? {})) {
      if (!knownClasses.has(name)) throw new Error(`Unknown ${ruleGroup} class: ${name}`);
      if (!Array.isArray(values) || values.some(value => typeof value !== 'string' || value.length === 0)) {
        throw new Error(`${ruleGroup} for ${name} must be an array of strings`);
      }
      if (ruleGroup === 'preferredCategories') {
        const unknownCategory = values.find(value => !knownCategories.has(value));
        if (unknownCategory) throw new Error(`Unknown preferred category: ${unknownCategory}`);
      }
    }
  }

  for (const [category, penalty] of Object.entries(rules.categoryPenalties ?? {})) {
    if (!knownCategories.has(category)) throw new Error(`Unknown penalty category: ${category}`);
    if (!Number.isFinite(penalty)) {
      throw new Error(`Category penalty for ${category} must be a finite number`);
    }
  }
}

function matchingIconIds(index, phrase) {
  const phraseTokens = tokenize(phrase);
  if (phraseTokens.length === 0) return [];

  const postings = phraseTokens.map(token => index.tokens[token] ?? []);
  if (postings.some(iconIds => iconIds.length === 0)) return [];

  const remaining = postings.slice(1).map(iconIds => new Set(iconIds));
  return postings[0].filter(iconId => remaining.every(iconIds => iconIds.has(iconId)));
}

function iconDetails(index, iconId) {
  const [packId, filename] = index.icons[iconId];
  const [categoryId, pack] = index.packs[packId];
  const category = index.categories[categoryId];
  return {
    category,
    name: filename.replace(/\.svg$/i, ''),
    pack,
    path: [category, pack, filename].filter(Boolean).join('/'),
  };
}

function githubRepositorySlug(repository) {
  let url;
  try {
    url = new URL(repository);
  } catch {
    throw new Error(`Unsupported repository URL: ${repository}`);
  }

  const segments = url.pathname.replace(/^\/+|\/+$/g, '').split('/');
  if (url.hostname !== 'github.com' || segments.length !== 2 || !segments.every(Boolean)) {
    throw new Error(`Unsupported repository URL: ${repository}`);
  }

  return `${segments[0]}/${segments[1].replace(/\.git$/, '')}`;
}

function buildManifest(index, categories, rules, limit) {
  const repository = githubRepositorySlug(index.source.repository);
  const cdnBase = `https://cdn.jsdelivr.net/gh/${repository}@${index.source.commit}/`;

  const manifest = {
    schemaVersion: 1,
    source: { ...index.source, candidateLimit: limit },
    classes: categories.map(name => {
      const aliases = rules.aliases?.[name] ?? [];
      const matchScores = new Map();
      [name, ...aliases].forEach((phrase, phraseIndex) => {
        const phraseTokens = tokenize(phrase);
        const aliasPriority = phraseIndex === 0 ? 0 : Math.max(0, 40 - phraseIndex * 10);
        const baseRelevance = (phraseIndex === 0 ? 100 : 50)
          + phraseTokens.length * 10
          + aliasPriority;
        matchingIconIds(index, phrase).forEach(iconId => {
          const filenameTokens = tokenize(iconDetails(index, iconId).name);
          const exactBonus = filenameTokens.join(' ') === phraseTokens.join(' ') ? 20 : 0;
          const relevance = baseRelevance + exactBonus;
          matchScores.set(iconId, Math.max(matchScores.get(iconId) ?? 0, relevance));
        });
      });
      const preferredCategories = new Set(rules.preferredCategories?.[name] ?? []);
      const excludedTokens = new Set(rules.excludedTokens?.[name] ?? []);
      const candidates = [...matchScores]
        .map(([iconId, relevance]) => {
          const details = iconDetails(index, iconId);
          const score = relevance
            + (preferredCategories.has(details.category) ? 30 : 0)
            + (rules.categoryPenalties?.[details.category] ?? 0);
          const encodedPath = details.path.split('/').map(encodeURIComponent).join('/');
          return {
            id: iconId,
            ...details,
            score,
            url: `${cdnBase}${encodedPath}`,
          };
        })
        .filter(candidate => tokenize(candidate.name).every(token => !excludedTokens.has(token)))
        .sort((left, right) => right.score - left.score || left.id - right.id)
        .slice(0, limit);

      return {
        name,
        aliases,
        candidates,
      };
    }),
  };
  return {
    ...manifest,
    fingerprint: createHash('sha256').update(JSON.stringify(manifest)).digest('hex'),
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const index = readJson(options.index);
  validateIndex(index);
  const rules = readJson(options.rules);
  const categories = parseLineList(readFileSync(options.categories, 'utf8'));
  validateRules(index, categories, rules);
  const manifest = buildManifest(index, categories, rules, options.limit);

  mkdirSync(dirname(options.output), { recursive: true });
  const temporaryOutput = `${options.output}.tmp-${process.pid}`;
  writeFileSync(temporaryOutput, `${JSON.stringify(manifest, null, 2)}\n`);
  renameSync(temporaryOutput, options.output);

  const coveredClasses = manifest.classes.filter(item => item.candidates.length > 0).length;
  console.log(`Built candidates for ${manifest.classes.length} Quick Draw classes`);
  console.log(`Coverage: ${coveredClasses}/${manifest.classes.length}`);
  console.log(`Output: ${options.output}`);
}

try {
  main();
} catch (error) {
  console.error(error.message || error);
  process.exitCode = 1;
}