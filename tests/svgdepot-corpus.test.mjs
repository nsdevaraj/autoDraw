import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

import { expectedIconUrl } from '../src/icon-candidate.mjs';
import {
  ICON_MANIFEST_SCHEMA_VERSION,
  UNAVAILABLE_DIGEST,
  assertTrustedIconResponse,
  buildIconManifest,
  candidateIconIds,
  iconPathsFromIndex,
  selectIconIds,
} from '../src/svgdepot-corpus.mjs';

const SOURCE = {
  repository: 'https://github.com/nsdevaraj/SVGDepot.git',
  ref: 'main',
  commit: '5d05c6fa39b9d193d408d203e72f98e3e78c5b3b',
};

const INDEX = {
  schemaVersion: 1,
  source: SOURCE,
  categories: ['Travel & Transportation', 'Nature, Animals & Environment'],
  packs: [[0, 'transport-vectors'], [1, 'animal-vectors'], [0, '']],
  icons: [[0, 'ship.svg'], [0, 'car.svg'], [1, 'cat.svg'], [2, 'loose.svg']],
};

function response(url, headers) {
  return { url, headers: new Headers(headers) };
}

test('icon paths join category, pack, and filename from the index', () => {
  assert.deepEqual(iconPathsFromIndex(INDEX), [
    { packId: 0, path: 'Travel & Transportation/transport-vectors/ship.svg' },
    { packId: 0, path: 'Travel & Transportation/transport-vectors/car.svg' },
    { packId: 1, path: 'Nature, Animals & Environment/animal-vectors/cat.svg' },
    { packId: 2, path: 'Travel & Transportation/loose.svg' },
  ]);
});

test('icon paths reconstruct URLs that satisfy the approved-source rule', () => {
  for (const icon of iconPathsFromIndex(INDEX)) {
    const url = expectedIconUrl(SOURCE, icon.path);
    assert.equal(url.startsWith('https://cdn.jsdelivr.net/gh/nsdevaraj/SVGDepot@'), true);
    assert.equal(url.includes(`@${SOURCE.commit}/`), true);
  }
});

test('unsupported or malformed indexes are rejected', () => {
  assert.throws(() => iconPathsFromIndex({ schemaVersion: 2 }), /schema version/);
  assert.throws(() => iconPathsFromIndex({ schemaVersion: 1 }), /icons, packs, and categories/);
  assert.throws(
    () => iconPathsFromIndex({ ...INDEX, packs: [] }),
    /entry 0 is malformed/,
  );
  assert.throws(
    () => iconPathsFromIndex({ ...INDEX, categories: [] }),
    /unknown category/,
  );
});

test('selection defaults to every icon and otherwise spans all packs', () => {
  const icons = iconPathsFromIndex(INDEX);
  assert.deepEqual(selectIconIds(icons), [0, 1, 2, 3]);
  assert.deepEqual(selectIconIds(icons, 1), [0, 2, 3]);
  assert.deepEqual(selectIconIds(icons, 2), [0, 1, 2, 3]);
});

test('selection rejects a non-positive limit', () => {
  assert.throws(() => selectIconIds(iconPathsFromIndex(INDEX), 0), /positive integer/);
  assert.throws(() => selectIconIds(iconPathsFromIndex(INDEX), 1.5), /positive integer/);
});

test('the candidate subset is the sorted unique set of referenced icon ids', () => {
  const candidates = {
    schemaVersion: 1,
    source: { commit: SOURCE.commit },
    classes: [
      { name: 'ship', candidates: [{ id: 3 }, { id: 0 }] },
      { name: 'cat', candidates: [{ id: 2 }, { id: 0 }] },
      { name: 'empty', candidates: [] },
    ],
  };
  assert.deepEqual(candidateIconIds(candidates, 4, SOURCE.commit), [0, 2, 3]);
});

test('the candidate subset refuses a mismatched commit or out-of-range id', () => {
  const candidates = {
    schemaVersion: 1,
    source: { commit: SOURCE.commit },
    classes: [{ name: 'ship', candidates: [{ id: 9 }] }],
  };
  assert.throws(
    () => candidateIconIds(candidates, 4, 'f'.repeat(40)),
    /different SVGDepot commit/,
  );
  assert.throws(() => candidateIconIds(candidates, 4, SOURCE.commit), /out of range: 9/);
  assert.throws(() => candidateIconIds({ schemaVersion: 2 }, 4, SOURCE.commit), /Invalid Quick Draw candidate manifest/);
});

test('trusted responses must stay on the CDN with an SVG content type', () => {
  assert.doesNotThrow(() => assertTrustedIconResponse(
    response('https://cdn.jsdelivr.net/gh/a/b@c/d.svg', {
      'content-type': 'image/svg+xml; charset=utf-8',
      'content-length': '1024',
    }),
    512 * 1024,
  ));
});

test('responses redirected off the CDN are refused', () => {
  assert.throws(
    () => assertTrustedIconResponse(
      response('https://example.com/d.svg', { 'content-type': 'image/svg+xml' }),
      1024,
    ),
    /redirected away from cdn\.jsdelivr\.net/,
  );
  assert.throws(
    () => assertTrustedIconResponse(response('not a url', {}), 1024),
    /unusable URL/,
  );
});

test('unexpected content types are refused', () => {
  for (const contentType of ['text/html', 'application/octet-stream', '']) {
    assert.throws(
      () => assertTrustedIconResponse(
        response('https://cdn.jsdelivr.net/gh/a/b@c/d.svg', { 'content-type': contentType }),
        1024,
      ),
      /unexpected content type/,
    );
  }
});

test('responses declaring more than the byte limit are refused', () => {
  assert.throws(
    () => assertTrustedIconResponse(
      response('https://cdn.jsdelivr.net/gh/a/b@c/d.svg', {
        'content-type': 'image/svg+xml',
        'content-length': '2048',
      }),
      1024,
    ),
    /declares 2048 bytes/,
  );
});

test('the manifest derives its counts and fingerprint from its own content', () => {
  const manifest = buildIconManifest({
    source: SOURCE,
    iconCount: 4,
    fetcher: { maxBytes: 524288, limitPerPack: null },
    counts: { cached: 1, fetched: 2, unavailable: 1, failed: 0, selected: 4 },
    digests: ['a'.repeat(64), 'b'.repeat(64)],
    iconDigests: [0, 0, 1, UNAVAILABLE_DIGEST],
  });

  assert.equal(manifest.schemaVersion, ICON_MANIFEST_SCHEMA_VERSION);
  assert.equal(manifest.source.commit, SOURCE.commit);
  assert.equal(manifest.source.iconCount, 4);
  assert.equal(manifest.counts.available, 3);
  assert.equal(manifest.counts.unique, 2);

  const { fingerprint, ...content } = manifest;
  assert.equal(
    fingerprint,
    createHash('sha256').update(JSON.stringify(content)).digest('hex'),
  );
});

test('duplicate icon contents share one digest entry', () => {
  const manifest = buildIconManifest({
    source: SOURCE,
    iconCount: 3,
    fetcher: { maxBytes: 524288, limitPerPack: null },
    counts: { cached: 0, fetched: 3, unavailable: 0, failed: 0, selected: 3 },
    digests: ['c'.repeat(64)],
    iconDigests: [0, 0, 0],
  });

  assert.equal(manifest.counts.unique, 1);
  assert.equal(manifest.counts.available, 3);
});
