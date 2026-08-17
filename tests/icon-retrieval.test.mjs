import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import test from 'node:test';
import { gzipSync } from 'node:zlib';

import { createIconRetrievalIndex, loadIconRetrievalIndex, retrieveIcons } from '../src/icon-retrieval.mjs';

const INDEX_DIRECTORY = 'data/icon-embeddings';
const INDEX_PATH = `${INDEX_DIRECTORY}/index.json`;
const DIM = 8;
const SOURCE = {
  repository: 'https://github.com/nsdevaraj/SVGDepot.git',
  commit: '5d05c6fa39b9d193d408d203e72f98e3e78c5b3b',
};

function quantize(vector) {
  const peak = Math.max(...vector.map(Math.abs));
  const scale = peak > 0 ? peak / 127 : 1;
  return { codes: vector.map(value => Math.round(value / scale)), scale };
}

function shardBinary(vectors) {
  const quantized = vectors.map(quantize);
  const codes = Int8Array.from(quantized.flatMap(item => item.codes));
  const scales = Float32Array.from(quantized.map(item => item.scale));
  return Buffer.concat([Buffer.from(codes.buffer), Buffer.from(scales.buffer)]);
}

// Two shards: one holding star-like vectors, one holding an orthogonal cluster.
function fixture() {
  const shards = [
    {
      centroid: [1, 0, 0, 0, 0, 0, 0, 0],
      vectors: [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0.94, 0.34, 0, 0, 0, 0, 0, 0],
      ],
      icons: [
        [10, 'Art, Design & Patterns/pack-a/332169-star.svg', 3, 0],
        [11, 'Art, Design & Patterns/pack-a/332170-sparkle.svg', 1, 1],
      ],
    },
    {
      centroid: [0, 1, 0, 0, 0, 0, 0, 0],
      vectors: [[0, 1, 0, 0, 0, 0, 0, 0]],
      icons: [[20, 'Travel & Transportation/pack-b/900-truck.svg', 1, 2]],
    },
  ];

  const files = new Map();
  const centroidCodes = [];
  const centroidScales = [];
  const descriptors = shards.map((shard, id) => {
    const binary = shardBinary(shard.vectors);
    const meta = gzipSync(JSON.stringify({
      shard: id,
      count: shard.icons.length,
      icons: shard.icons,
    }));
    files.set(`./shard-${id}.bin`, binary);
    files.set(`./shard-${id}.meta.json.gz`, meta);

    const quantizedCentroid = quantize(shard.centroid);
    centroidCodes.push(...quantizedCentroid.codes);
    centroidScales.push(quantizedCentroid.scale);

    return {
      id,
      count: shard.icons.length,
      bytes: binary.length,
      sha256: createHash('sha256').update(binary).digest('hex'),
      metaBytes: meta.length,
      metaSha256: createHash('sha256').update(meta).digest('hex'),
    };
  });

  const document = {
    schemaVersion: 1,
    kind: 'icon-embedding-index',
    source: { ...SOURCE },
    classes: ['star', 'sparkle', 'truck'],
    embedding: { dim: DIM, dtype: 'int8', scale: 'per-vector-float32', metric: 'dot' },
    counts: { icons: 3, vectors: 3 },
    clustering: {
      count: shards.length,
      seed: 1,
      iterations: 2,
      centroidCodes: Buffer.from(Int8Array.from(centroidCodes).buffer).toString('base64'),
      centroidScales,
    },
    shards: descriptors,
    fingerprint: 'a'.repeat(64),
  };

  const fetchImpl = async url => {
    const body = files.get(url);
    if (!body) return { ok: false, status: 404 };
    return {
      ok: true,
      status: 200,
      arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.length),
    };
  };

  return { document, files, fetchImpl };
}

function query(values) {
  const vector = new Float32Array(DIM);
  values.forEach((value, index) => {
    vector[index] = value;
  });
  return vector;
}

test('a well-formed index exposes its pinned source and geometry', () => {
  const { document } = fixture();
  const index = createIconRetrievalIndex(document);
  assert.equal(index.dim, DIM);
  assert.equal(index.source.commit, SOURCE.commit);
  assert.equal(index.shards.length, 2);
  assert.deepEqual(index.classes, ['star', 'sparkle', 'truck']);
});

test('an index built for another encoder is rejected', () => {
  const { document } = fixture();
  assert.throws(
    () => createIconRetrievalIndex(document, {
      embedder: { embeddingDim: 64, classes: document.classes },
    }),
    /dimension/,
  );
  assert.throws(
    () => createIconRetrievalIndex(document, {
      embedder: { embeddingDim: DIM, classes: ['star', 'sparkle', 'car'] },
    }),
    /classes/,
  );
});

test('malformed indexes are rejected before any shard is fetched', () => {
  const cases = [
    ['kind', document => { document.kind = 'something-else'; }],
    ['float vectors', document => { document.embedding.dtype = 'float32'; }],
    ['missing commit', document => { document.source.commit = 'abc'; }],
    ['shard byte mismatch', document => { document.shards[0].bytes += 1; }],
    ['centroid count', document => { document.clustering.centroidScales.pop(); }],
  ];
  for (const [reason, mutate] of cases) {
    const { document } = fixture();
    mutate(document);
    assert.throws(() => createIconRetrievalIndex(document), Error, reason);
  }
});

test('retrieval probes the nearest cluster and ranks by similarity', async () => {
  const { document, fetchImpl } = fixture();
  const index = createIconRetrievalIndex(document);
  const results = await retrieveIcons(index, {
    embedding: query([1]),
    probes: 1,
    lexicalWeight: 0,
    fetchImpl,
  });

  assert.deepEqual(results.map(item => item.id), [10, 11]);
  assert.equal(results[0].cosine > results[1].cosine, true);
  assert.equal(results[0].name, '332169-star');
  assert.equal(results[0].category, 'Art, Design & Patterns');
  assert.equal(results[0].pack, 'pack-a');
});

test('captions read as the icon rather than the encoder class guess', async () => {
  const { document, fetchImpl } = fixture();
  const index = createIconRetrievalIndex(document);
  const [first, second] = await retrieveIcons(index, {
    embedding: query([1]),
    probes: 1,
    lexicalWeight: 0,
    fetchImpl,
  });
  assert.equal(first.label, 'star');
  assert.equal(second.label, 'sparkle');
});

test('every retrieved icon carries a commit-pinned URL rebuilt from its path', async () => {
  const { document, fetchImpl } = fixture();
  const index = createIconRetrievalIndex(document);
  const results = await retrieveIcons(index, {
    embedding: query([1]),
    probes: 2,
    fetchImpl,
  });

  assert.equal(results.length, 3);
  for (const result of results) {
    const url = new URL(result.url);
    assert.equal(url.protocol, 'https:');
    assert.equal(url.hostname, 'cdn.jsdelivr.net');
    assert.equal(url.pathname.startsWith(`/gh/nsdevaraj/SVGDepot@${SOURCE.commit}/`), true);
    assert.equal(decodeURIComponent(url.pathname.split(`@${SOURCE.commit}/`)[1]), result.path);
    assert.equal(result.approved, true);
  }
});

test('a shard entry that escapes its path is refused', async () => {
  const { document, files, fetchImpl } = fixture();
  const poisoned = gzipSync(JSON.stringify({
    shard: 0,
    count: 2,
    icons: [
      [10, '../../etc/passwd.svg', 1, 0],
      [11, 'Art, Design & Patterns/pack-a/332170-sparkle.svg', 1, 1],
    ],
  }));
  files.set('./shard-0.meta.json.gz', poisoned);
  document.shards[0].metaSha256 = createHash('sha256').update(poisoned).digest('hex');

  const index = createIconRetrievalIndex(document);
  await assert.rejects(
    () => retrieveIcons(index, { embedding: query([1]), probes: 1, fetchImpl }),
    /malformed|confined/i,
  );
});

test('a shard whose bytes do not match the index is refused', async () => {
  const { document, files, fetchImpl } = fixture();
  files.set('./shard-0.bin', Buffer.alloc(document.shards[0].bytes));
  const index = createIconRetrievalIndex(document);
  await assert.rejects(
    () => retrieveIcons(index, { embedding: query([1]), probes: 1, fetchImpl }),
    /integrity/,
  );
});

test('the lexical prior lifts an icon whose filename matches the predicted class', async () => {
  const { document, fetchImpl } = fixture();
  const index = createIconRetrievalIndex(document);
  const [top] = await retrieveIcons(index, {
    // Geometry alone puts sparkle second; the class prediction pulls it to the front.
    embedding: query([0.94, 0.34]),
    predictions: [{ classIndex: 1, label: 'sparkle', probability: 1 }],
    probes: 1,
    lexicalWeight: 0.9,
    fetchImpl,
  });
  assert.equal(top.name, '332170-sparkle');
});

test('retrieval arguments are validated', async () => {
  const { document, fetchImpl } = fixture();
  const index = createIconRetrievalIndex(document);
  await assert.rejects(
    () => retrieveIcons(index, { embedding: new Float32Array(4), fetchImpl }),
    /Float32Array of length 8/,
  );
  await assert.rejects(
    () => retrieveIcons(index, { embedding: query([1]), limit: 0, fetchImpl }),
    /limit/,
  );
  await assert.rejects(
    () => retrieveIcons(index, { embedding: query([1]), lexicalWeight: 2, fetchImpl }),
    /Lexical weight/,
  );
});

test('loading derives the shard base URL from the index URL', async () => {
  const { document, files } = fixture();
  const fetchImpl = async url => {
    if (url === 'https://example.test/data/icon-embeddings/index.json') {
      return { ok: true, status: 200, json: async () => document };
    }
    const key = url.replace('https://example.test/data/icon-embeddings/', './');
    const body = files.get(key);
    if (!body) return { ok: false, status: 404 };
    return {
      ok: true,
      status: 200,
      arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.length),
    };
  };

  const index = await loadIconRetrievalIndex({
    indexUrl: 'https://example.test/data/icon-embeddings/index.json',
    fetchImpl,
  });
  const results = await retrieveIcons(index, { embedding: query([1]), probes: 1, fetchImpl });
  assert.equal(results[0].id, 10);
});

const artifactPresent = existsSync(INDEX_PATH);

test('the tracked icon index covers the corpus and stays within its budgets', { skip: !artifactPresent }, () => {
  const document = JSON.parse(readFileSync(INDEX_PATH, 'utf8'));
  const index = createIconRetrievalIndex(document);

  assert.equal(document.kind, 'icon-embedding-index');
  assert.match(document.source.commit, /^[a-f0-9]{40}$/);
  assert.match(document.fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(document.embedding.dim, 64);

  // The whole point of the index: every reachable icon, not just the classifier's classes.
  assert.equal(document.counts.icons > 200000, true);
  assert.equal(document.counts.vectors > 150000, true);
  assert.equal(index.shards.length, document.clustering.count);

  // First paint fetches index.json plus a few shards, so the entry point stays small.
  assert.equal(readFileSync(INDEX_PATH).length < 200 * 1024, true);
  const worstProbe = [...document.shards]
    .sort((left, right) => right.bytes - left.bytes)
    .slice(0, 4)
    .reduce((total, shard) => total + shard.bytes + shard.metaBytes, 0);
  assert.equal(worstProbe < 1024 * 1024, true, 'worst-case four-shard probe must stay under 1 MB');
});

test('every tracked shard matches the hash the index records', { skip: !artifactPresent }, () => {
  const document = JSON.parse(readFileSync(INDEX_PATH, 'utf8'));
  const present = new Set(readdirSync(INDEX_DIRECTORY));

  for (const shard of document.shards) {
    const binaryName = `shard-${shard.id}.bin`;
    const metaName = `shard-${shard.id}.meta.json.gz`;
    assert.equal(present.has(binaryName), true, `missing ${binaryName}`);
    assert.equal(present.has(metaName), true, `missing ${metaName}`);

    const binary = readFileSync(`${INDEX_DIRECTORY}/${binaryName}`);
    assert.equal(binary.length, shard.bytes);
    assert.equal(createHash('sha256').update(binary).digest('hex'), shard.sha256);

    const meta = readFileSync(`${INDEX_DIRECTORY}/${metaName}`);
    assert.equal(createHash('sha256').update(meta).digest('hex'), shard.metaSha256);
  }
});
