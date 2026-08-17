import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildCuratedManifest,
  createCurationState,
  getCurationStats,
  nextClassAfterRemoval,
  restoreCurationState,
  setCandidateDecision,
  setClassReviewed,
} from '../src/curation-state.mjs';

const manifest = {
  schemaVersion: 1,
  fingerprint: 'manifest-one',
  source: {
    repository: 'https://github.com/nsdevaraj/SVGDepot.git',
    ref: 'main',
    commit: 'abc123',
    candidateLimit: 12,
  },
  classes: [
    {
      name: 'cat',
      aliases: [],
      candidates: [
        { id: 10, name: 'cat', category: 'Animals', pack: 'outline', path: 'Animals/outline/cat.svg', score: 130, url: 'https://cdn.test/cat.svg' },
        { id: 11, name: 'cat-face', category: 'Animals', pack: 'filled', path: 'Animals/filled/cat-face.svg', score: 110, url: 'https://cdn.test/cat-face.svg' },
      ],
    },
    {
      name: 'dog',
      aliases: [],
      candidates: [
        { id: 20, name: 'dog', category: 'Animals', pack: 'outline', path: 'Animals/outline/dog.svg', score: 130, url: 'https://cdn.test/dog.svg' },
      ],
    },
  ],
};

test('candidate decisions and reviewed classes update immutable curation state', () => {
  const empty = createCurationState(manifest);
  const accepted = setCandidateDecision(empty, 'cat', 10, 'accepted');
  const rejected = setCandidateDecision(accepted, 'cat', 11, 'rejected');
  const reviewed = setClassReviewed(rejected, 'cat', true);

  assert.deepEqual(empty.decisions, {});
  assert.equal(empty.kind, 'quickdraw-curation-progress');
  assert.equal(empty.manifestFingerprint, manifest.fingerprint);
  assert.deepEqual(reviewed.decisions, { cat: { 10: 'accepted', 11: 'rejected' } });
  assert.deepEqual(reviewed.reviewedClasses, ['cat']);
  assert.deepEqual(getCurationStats(manifest, reviewed), {
    classCount: 2,
    reviewedClassCount: 1,
    acceptedCandidateCount: 1,
    rejectedCandidateCount: 1,
  });
});

test('restored curation state rejects a different source commit or unknown candidates', () => {
  assert.throws(
    () => restoreCurationState(manifest, { kind: 'quickdraw-curation-progress', schemaVersion: 1, sourceCommit: 'other', decisions: {}, reviewedClasses: [] }),
    /different SVGDepot commit/,
  );
  assert.throws(
    () => restoreCurationState(manifest, {
      schemaVersion: 1,
      kind: 'quickdraw-curation-progress',
      sourceCommit: 'abc123',
      manifestFingerprint: 'manifest-one',
      decisions: { cat: { 999: 'accepted' } },
      reviewedClasses: [],
    }),
    /Unknown candidate 999 for cat/,
  );
});

test('restored curation state rejects a different candidate manifest', () => {
  assert.throws(
    () => restoreCurationState(manifest, {
      schemaVersion: 1,
      kind: 'quickdraw-curation-progress',
      sourceCommit: 'abc123',
      manifestFingerprint: 'manifest-two',
      decisions: {},
      reviewedClasses: [],
    }),
    /different candidate manifest/,
  );
});

test('restored curation state rejects noncanonical candidate IDs', () => {
  assert.throws(
    () => restoreCurationState(manifest, {
      schemaVersion: 1,
      kind: 'quickdraw-curation-progress',
      sourceCommit: 'abc123',
      manifestFingerprint: 'manifest-one',
      decisions: { cat: { '010': 'accepted' } },
      reviewedClasses: ['cat'],
    }),
    /Invalid candidate ID for cat: 010/,
  );
});

test('restored curation state rejects malformed decision containers', () => {
  assert.throws(
    () => restoreCurationState(manifest, {
      schemaVersion: 1,
      kind: 'quickdraw-curation-progress',
      sourceCommit: 'abc123',
      manifestFingerprint: 'manifest-one',
      decisions: 42,
      reviewedClasses: [],
    }),
    /Curation decisions must be an object/,
  );
  assert.throws(
    () => restoreCurationState(manifest, {
      schemaVersion: 1,
      kind: 'quickdraw-curation-progress',
      sourceCommit: 'abc123',
      manifestFingerprint: 'manifest-one',
      decisions: { cat: 42 },
      reviewedClasses: [],
    }),
    /Decisions for cat must be an object/,
  );
});

test('removing a class from a queue preserves its former successor', () => {
  assert.equal(nextClassAfterRemoval(['apple', 'banana', 'cat'], 'banana'), 'cat');
  assert.equal(nextClassAfterRemoval(['apple', 'banana', 'cat'], 'cat'), 'banana');
  assert.equal(nextClassAfterRemoval(['apple'], 'apple'), null);
});

test('curated icon output cannot be imported as progress', () => {
  const state = setClassReviewed(createCurationState(manifest), 'cat', true);
  assert.throws(
    () => restoreCurationState(manifest, buildCuratedManifest(manifest, state)),
    /not Quick Draw curation progress/,
  );
});

test('accepted candidates export before their class is marked reviewed', () => {
  const state = setCandidateDecision(createCurationState(manifest), 'cat', 10, 'accepted');

  assert.deepEqual(buildCuratedManifest(manifest, state).classes, [
    { name: 'cat', candidates: [manifest.classes[0].candidates[0]] },
  ]);
});

test('curated export contains accepted candidates and preserves reviewed empty classes', () => {
  let state = createCurationState(manifest);
  state = setCandidateDecision(state, 'cat', 10, 'accepted');
  state = setCandidateDecision(state, 'cat', 11, 'rejected');
  state = setClassReviewed(state, 'cat', true);
  state = setClassReviewed(state, 'dog', true);

  const exported = buildCuratedManifest(manifest, state);

  assert.deepEqual(exported.source, manifest.source);
  assert.equal(exported.kind, 'quickdraw-curated-icons');
  assert.equal(exported.candidateManifestFingerprint, manifest.fingerprint);
  assert.deepEqual(exported.classes, [
    { name: 'cat', candidates: [manifest.classes[0].candidates[0]] },
    { name: 'dog', candidates: [] },
  ]);
});