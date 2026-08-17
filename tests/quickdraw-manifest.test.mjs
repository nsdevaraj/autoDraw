import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parseLineList } from '../src/line-list.mjs';

const PROJECT_ROOT = new URL('../', import.meta.url);
const categoryFile = readFileSync(new URL('config/quickdraw-categories.txt', PROJECT_ROOT));
const categories = parseLineList(categoryFile.toString('utf8'));
const rules = JSON.parse(readFileSync(new URL('config/quickdraw-icon-rules.json', PROJECT_ROOT), 'utf8'));
const manifest = JSON.parse(readFileSync(new URL('data/quickdraw-candidates.json', PROJECT_ROOT), 'utf8'));
const mvpClasses = parseLineList(readFileSync(new URL('config/mvp-classes.txt', PROJECT_ROOT), 'utf8'));

function manifestClass(name) {
  return manifest.classes.find(item => item.name === name);
}

test('the shipped manifest covers every official Quick Draw class', () => {
  assert.match(manifest.fingerprint, /^[a-f0-9]{64}$/);
  const { fingerprint, ...fingerprintedContent } = manifest;
  assert.equal(
    fingerprint,
    createHash('sha256').update(JSON.stringify(fingerprintedContent)).digest('hex'),
  );
  assert.equal(
    createHash('sha256').update(categoryFile).digest('hex'),
    '6326fde883ce71b53153bdf50b3671dc6652c8c14e67c7d07d23c1af6d103f4b',
  );
  assert.equal(categories.length, 345);
  assert.deepEqual(manifest.classes.map(item => item.name), categories);

  const knownClasses = new Set(categories);
  for (const ruleGroup of ['aliases', 'preferredCategories', 'excludedTokens']) {
    for (const name of Object.keys(rules[ruleGroup] ?? {})) {
      assert.equal(knownClasses.has(name), true, `Unknown ${ruleGroup} class: ${name}`);
    }
  }

  for (const item of manifest.classes) {
    assert.equal(item.candidates.length > 0 && item.candidates.length <= 12, true, item.name);
    assert.deepEqual(item.aliases, rules.aliases[item.name] ?? []);

    const ids = new Set();
    let previousScore = Infinity;
    for (const candidate of item.candidates) {
      assert.equal(ids.has(candidate.id), false, `${item.name}: duplicate ${candidate.id}`);
      assert.equal(candidate.score <= previousScore, true, `${item.name}: scores are not sorted`);
      assert.equal(candidate.url.includes(`@${manifest.source.commit}/`), true, candidate.url);
      for (const field of ['name', 'category', 'pack', 'path']) {
        assert.equal(typeof candidate[field], 'string', `${item.name}: missing ${field}`);
      }
      ids.add(candidate.id);
      previousScore = candidate.score;
    }
  }
});

test('swing set and see saw do not share known false-positive candidates', () => {
  const swingSetCandidates = manifestClass('swing set').candidates;
  assert.equal(
    swingSetCandidates.slice(0, 3).some(candidate => /(?:^|-)swingset(?:-|$)/.test(candidate.name)),
    true,
  );

  const seeSawNames = manifestClass('see saw').candidates.map(candidate => candidate.name);
  assert.equal(seeSawNames.some(name => /(?:^|-)swingset(?:-|$)/.test(name)), false);
  assert.equal(seeSawNames.some(name => /(?:^|-)code(?:-|$)/.test(name)), false);

  const swingSetIds = new Set(swingSetCandidates.map(candidate => candidate.id));
  assert.equal(manifestClass('see saw').candidates.some(candidate => swingSetIds.has(candidate.id)), false);
});

test('the MVP curation queue contains 40 unique supported classes', () => {
  assert.equal(mvpClasses.length, 40);
  assert.equal(new Set(mvpClasses).size, 40);

  for (const className of mvpClasses) {
    const item = manifestClass(className);
    assert.notEqual(item, undefined, `Unknown MVP class: ${className}`);
    assert.equal(item.candidates.length > 0, true, `MVP class has no candidates: ${className}`);
  }
});