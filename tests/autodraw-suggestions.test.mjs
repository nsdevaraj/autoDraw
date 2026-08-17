import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createApprovedCandidateIndex,
  suggestApprovedCandidates,
} from '../src/autodraw-suggestions.mjs';

const manifest = {
  schemaVersion: 1,
  fingerprint: 'manifest-fingerprint',
  source: {
    repository: 'https://github.com/example/icons.git',
    commit: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  },
  classes: [
    {
      name: 'cat',
      aliases: [],
      candidates: [
        { id: 1, name: 'cat-one', category: 'Animals', pack: 'line', path: 'Animals/line/cat-one.svg', score: 130, url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/cat-one.svg' },
        { id: 2, name: 'cat-two', category: 'Animals', pack: 'fill', path: 'Animals/fill/cat-two.svg', score: 120, url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/fill/cat-two.svg' },
        { id: 5, name: 'cat-three', category: 'Animals', pack: 'color', path: 'Animals/color/cat-three.svg', score: 110, url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/color/cat-three.svg' },
      ],
    },
    {
      name: 'dog',
      aliases: [],
      candidates: [
        { id: 3, name: 'dog-one', category: 'Animals', pack: 'line', path: 'Animals/line/dog-one.svg', score: 130, url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/dog-one.svg' },
        { id: 4, name: 'dog-two', category: 'Animals', pack: 'fill', path: 'Animals/fill/dog-two.svg', score: 120, url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/fill/dog-two.svg' },
      ],
    },
  ],
};

test('all ranked manifest candidates are approved without curation decisions', () => {
  const index = createApprovedCandidateIndex(manifest, ['cat', 'dog']);
  const suggestions = suggestApprovedCandidates(index, [
    { classIndex: 0, label: 'cat', probability: 0.8 },
    { classIndex: 1, label: 'dog', probability: 0.15 },
  ], { limit: 4, perClass: 2 });

  assert.deepEqual(suggestions.map(item => item.id), [1, 3, 2, 4]);
  assert.deepEqual(suggestions.map(item => item.label), ['cat', 'dog', 'cat', 'dog']);
  assert.deepEqual(suggestions.map(item => item.probability), [0.8, 0.15, 0.8, 0.15]);
  assert.deepEqual(suggestions.map(item => item.path), [
    'Animals/line/cat-one.svg',
    'Animals/line/dog-one.svg',
    'Animals/fill/cat-two.svg',
    'Animals/fill/dog-two.svg',
  ]);
  assert.equal(suggestions.every(item => item.approved === true), true);
});

test('candidate index fails fast when classifier classes cannot be suggested', () => {
  assert.throws(
    () => createApprovedCandidateIndex(manifest, ['cat', 'bird']),
    /bird.*candidate/i,
  );
  assert.throws(
    () => createApprovedCandidateIndex({ ...manifest, fingerprint: '' }, ['cat']),
    /manifest/,
  );
  assert.throws(
    () => createApprovedCandidateIndex({
      ...manifest,
      classes: [{
        name: 'cat',
        aliases: [],
        candidates: [{ ...manifest.classes[0].candidates[0], url: 'javascript:alert(1)' }],
      }],
    }, ['cat']),
    /commit-pinned jsDelivr URL/,
  );
  assert.throws(
    () => createApprovedCandidateIndex({
      ...manifest,
      classes: [{
        name: 'cat',
        aliases: [],
        candidates: [{ ...manifest.classes[0].candidates[0], url: 'https://example.com/cat.svg' }],
      }],
    }, ['cat']),
    /jsDelivr/,
  );
  assert.throws(
    () => createApprovedCandidateIndex({
      ...manifest,
      classes: [{
        name: 'cat',
        aliases: [],
        candidates: [{
          ...manifest.classes[0].candidates[0],
          url: 'https://cdn.jsdelivr.net/gh/example/icons@main/Animals/line/cat-one.svg',
        }],
      }],
    }, ['cat']),
    /commit-pinned/,
  );
  assert.throws(
    () => createApprovedCandidateIndex({
      ...manifest,
      classes: [{
        name: 'cat',
        aliases: [],
        candidates: [{
          ...manifest.classes[0].candidates[0],
          url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/dog.svg',
        }],
      }],
    }, ['cat']),
    /candidate path/,
  );
  assert.throws(
    () => createApprovedCandidateIndex({
      ...manifest,
      classes: [{
        name: 'cat',
        aliases: [],
        candidates: [{ ...manifest.classes[0].candidates[0], path: '../cat.svg' }],
      }],
    }, ['cat']),
    /SVG path/,
  );
});

test('every ranked candidate remains reachable when limits are omitted', () => {
  const index = createApprovedCandidateIndex(manifest, ['cat', 'dog']);
  const suggestions = suggestApprovedCandidates(index, [
    { classIndex: 0, label: 'cat', probability: 0.8 },
    { classIndex: 1, label: 'dog', probability: 0.15 },
  ]);

  assert.deepEqual(suggestions.map(item => item.id), [1, 3, 2, 4, 5]);
});

test('suggestion limits and prediction values are validated', () => {
  const index = createApprovedCandidateIndex(manifest, ['cat', 'dog']);
  assert.throws(
    () => suggestApprovedCandidates(index, [{ label: 'cat', probability: Number.NaN }]),
    /probability/,
  );
  assert.throws(
    () => suggestApprovedCandidates(index, [], { limit: 0 }),
    /limit/,
  );
});