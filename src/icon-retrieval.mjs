import { assertApprovedIconSource, expectedIconUrl, isConfinedSvgPath } from './icon-candidate.mjs';
import { tokenize } from './tokenize.mjs';

const RETRIEVAL_INDEX = Symbol('icon-retrieval-index');
const SCALE_BYTES = 4;
const GZIP_MAGIC = [0x1f, 0x8b];

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function decodeBase64(value) {
  const binary = globalThis.atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function validatedSource(source) {
  if (
    !isRecord(source)
    || typeof source.repository !== 'string'
    || !/^[0-9a-f]{40}$/.test(source.commit ?? '')
  ) {
    throw new Error('Icon index must pin a repository and commit');
  }
  return Object.freeze({ repository: source.repository, commit: source.commit });
}

function validatedShards(shards, dim) {
  if (!Array.isArray(shards) || shards.length === 0) {
    throw new Error('Icon index must contain shards');
  }
  return Object.freeze(shards.map((shard, id) => {
    if (
      !isRecord(shard)
      || shard.id !== id
      || !Number.isInteger(shard.count)
      || shard.count < 0
      || shard.bytes !== shard.count * (dim + SCALE_BYTES)
      || !/^[0-9a-f]{64}$/.test(shard.sha256 ?? '')
      || !/^[0-9a-f]{64}$/.test(shard.metaSha256 ?? '')
    ) {
      throw new Error(`Icon index shard ${id} is malformed`);
    }
    return Object.freeze({
      id,
      count: shard.count,
      bytes: shard.bytes,
      sha256: shard.sha256,
      metaSha256: shard.metaSha256,
    });
  }));
}

export function createIconRetrievalIndex(document, { baseUrl, embedder } = {}) {
  if (
    !isRecord(document)
    || document.schemaVersion !== 1
    || document.kind !== 'icon-embedding-index'
    || typeof document.fingerprint !== 'string'
  ) {
    throw new Error('Invalid icon embedding index');
  }

  const embedding = document.embedding;
  if (
    !isRecord(embedding)
    || !Number.isInteger(embedding.dim)
    || embedding.dim < 8
    || embedding.dtype !== 'int8'
    || embedding.metric !== 'dot'
  ) {
    throw new Error('Icon index must hold int8 vectors scored by dot product');
  }

  const classes = document.classes;
  if (
    !Array.isArray(classes)
    || classes.some(label => typeof label !== 'string' || label.length === 0)
    || new Set(classes).size !== classes.length
  ) {
    throw new Error('Icon index must contain unique class labels');
  }

  // A retrieval index built against a different encoder would score in an unrelated space.
  if (embedder) {
    if (embedder.embeddingDim !== embedding.dim) {
      throw new Error('Icon index dimension does not match the embedder');
    }
    if (
      embedder.classes.length !== classes.length
      || embedder.classes.some((label, index) => label !== classes[index])
    ) {
      throw new Error('Icon index classes do not match the embedder');
    }
  }

  const clustering = document.clustering;
  const shards = validatedShards(document.shards, embedding.dim);
  if (
    !isRecord(clustering)
    || clustering.count !== shards.length
    || !Array.isArray(clustering.centroidScales)
    || clustering.centroidScales.length !== shards.length
    || clustering.centroidScales.some(value => !Number.isFinite(value))
    || typeof clustering.centroidCodes !== 'string'
  ) {
    throw new Error('Icon index clustering does not match its shards');
  }

  const centroidCodes = new Int8Array(decodeBase64(clustering.centroidCodes).buffer);
  if (centroidCodes.length !== shards.length * embedding.dim) {
    throw new Error('Icon index centroid codes do not match the shard count');
  }

  return Object.freeze({
    [RETRIEVAL_INDEX]: true,
    fingerprint: document.fingerprint,
    source: validatedSource(document.source),
    dim: embedding.dim,
    classes: Object.freeze([...classes]),
    counts: Object.freeze({ ...document.counts }),
    shards,
    centroidCodes,
    centroidScales: Float32Array.from(clustering.centroidScales),
    baseUrl: typeof baseUrl === 'string' && baseUrl.length > 0 ? baseUrl : './',
    cache: new Map(),
  });
}

function probeOrder(index, query, probes) {
  const scores = [];
  for (let shard = 0; shard < index.shards.length; shard += 1) {
    let sum = 0;
    const base = shard * index.dim;
    for (let axis = 0; axis < index.dim; axis += 1) {
      sum += index.centroidCodes[base + axis] * query[axis];
    }
    scores.push({ shard, score: sum * index.centroidScales[shard] });
  }
  return scores
    .sort((left, right) => right.score - left.score || left.shard - right.shard)
    .slice(0, probes)
    .map(entry => entry.shard)
    .filter(shard => index.shards[shard].count > 0);
}

async function assertDigest(bytes, expected, label) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return;
  const digest = await subtle.digest('SHA-256', bytes);
  const actual = [...new Uint8Array(digest)]
    .map(value => value.toString(16).padStart(2, '0'))
    .join('');
  if (actual !== expected) throw new Error(`${label} failed its integrity check`);
}

async function readShardMeta(bytes, shard) {
  // The recorded hash covers the gzip container, which a transparently decoding
  // server would have already unwrapped, so only the compressed form is verified.
  const compressed = bytes[0] === GZIP_MAGIC[0] && bytes[1] === GZIP_MAGIC[1];
  if (compressed) await assertDigest(bytes, shard.metaSha256, `Shard ${shard.id} metadata`);

  const text = compressed
    ? await new Response(
      new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),
    ).text()
    : new TextDecoder().decode(bytes);

  const meta = JSON.parse(text);
  if (
    !isRecord(meta)
    || meta.shard !== shard.id
    || meta.count !== shard.count
    || !Array.isArray(meta.icons)
    || meta.icons.length !== shard.count
  ) {
    throw new Error(`Shard ${shard.id} metadata does not match the index`);
  }
  return meta.icons;
}

async function fetchBytes(url, fetchImpl, label) {
  const response = await fetchImpl(url);
  if (!response?.ok) {
    throw new Error(`${label} request failed: ${response?.status ?? 'unknown'}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function loadShard(index, shardId, fetchImpl) {
  const cached = index.cache.get(shardId);
  if (cached) return cached;

  const shard = index.shards[shardId];
  const pending = (async () => {
    const [binary, meta] = await Promise.all([
      fetchBytes(`${index.baseUrl}shard-${shardId}.bin`, fetchImpl, `Shard ${shardId}`),
      fetchBytes(
        `${index.baseUrl}shard-${shardId}.meta.json.gz`,
        fetchImpl,
        `Shard ${shardId} metadata`,
      ),
    ]);
    if (binary.length !== shard.bytes) {
      throw new Error(`Shard ${shardId} has an unexpected size`);
    }
    await assertDigest(binary, shard.sha256, `Shard ${shardId}`);

    const codeBytes = shard.count * index.dim;
    return Object.freeze({
      codes: new Int8Array(binary.buffer, binary.byteOffset, codeBytes),
      scales: new Float32Array(
        binary.buffer.slice(binary.byteOffset + codeBytes, binary.byteOffset + shard.bytes),
      ),
      icons: await readShardMeta(meta, shard),
    });
  })();

  index.cache.set(shardId, pending);
  try {
    return await pending;
  } catch (error) {
    index.cache.delete(shardId);
    throw error;
  }
}

function describePath(path) {
  const segments = path.split('/');
  return {
    category: segments[0],
    pack: segments.length > 2 ? segments[1] : '',
    name: segments[segments.length - 1].replace(/\.svg$/i, ''),
  };
}

// SVGDepot filenames are mostly `<pack number>-<words>`, and the caption should read as
// the icon rather than as the encoder's class guess for it.
function displayLabel(name) {
  const words = name.replace(/^\d+[-_]/, '').replace(/[-_]+/g, ' ').trim();
  return words.length > 0 ? words : name;
}

// Shape alone is ambiguous across 185k icons, so a filename or aux-class agreement with the
// encoder's own class guess damps the zero-shot noise without downloading a lexical index.
function lexicalPrior(entry, predictionByClass, predictionTokens) {
  let prior = predictionByClass.get(entry[3]) ?? 0;
  if (predictionTokens.length === 0) return prior;

  const tokens = new Set(tokenize(entry[1].split('/').pop().replace(/\.svg$/i, '')));
  for (const { probability, labelTokens } of predictionTokens) {
    if (probability > prior && labelTokens.every(token => tokens.has(token))) prior = probability;
  }
  return prior;
}

export async function retrieveIcons(index, {
  embedding,
  predictions = [],
  limit = 12,
  probes = 4,
  lexicalWeight = 0.25,
  fetchImpl = globalThis.fetch,
} = {}) {
  if (!index?.[RETRIEVAL_INDEX]) throw new Error('Invalid icon retrieval index');
  if (!(embedding instanceof Float32Array) || embedding.length !== index.dim) {
    throw new Error(`Embedding must be a Float32Array of length ${index.dim}`);
  }
  if (!Number.isInteger(limit) || limit < 1) throw new Error('Retrieval limit must be positive');
  if (!Number.isInteger(probes) || probes < 1) throw new Error('Probe count must be positive');
  if (!Number.isFinite(lexicalWeight) || lexicalWeight < 0 || lexicalWeight > 1) {
    throw new Error('Lexical weight must be from 0 to 1');
  }
  if (typeof fetchImpl !== 'function') throw new Error('A fetch implementation is required');

  const predictionByClass = new Map();
  const predictionTokens = [];
  for (const prediction of predictions) {
    if (
      !isRecord(prediction)
      || !Number.isInteger(prediction.classIndex)
      || !Number.isFinite(prediction.probability)
    ) {
      throw new Error('Retrieval predictions must carry a class index and probability');
    }
    predictionByClass.set(prediction.classIndex, prediction.probability);
    predictionTokens.push({
      probability: prediction.probability,
      labelTokens: tokenize(prediction.label ?? ''),
    });
  }

  const shardIds = probeOrder(index, embedding, probes);
  const shards = await Promise.all(shardIds.map(id => loadShard(index, id, fetchImpl)));

  const scored = [];
  for (const shard of shards) {
    for (let position = 0; position < shard.icons.length; position += 1) {
      let sum = 0;
      const base = position * index.dim;
      for (let axis = 0; axis < index.dim; axis += 1) {
        sum += shard.codes[base + axis] * embedding[axis];
      }
      const entry = shard.icons[position];
      const cosine = sum * shard.scales[position];
      const prior = lexicalWeight > 0 ? lexicalPrior(entry, predictionByClass, predictionTokens) : 0;
      scored.push({
        entry,
        cosine,
        prior,
        score: (1 - lexicalWeight) * cosine + lexicalWeight * prior,
      });
    }
  }

  scored.sort((left, right) => right.score - left.score || left.entry[0] - right.entry[0]);

  const suggestions = [];
  for (const { entry, cosine, prior, score } of scored) {
    if (suggestions.length === limit) break;
    const [id, path, duplicates, auxClass] = entry;
    if (!Number.isInteger(id) || !isConfinedSvgPath(path)) {
      throw new Error(`Shard entry ${id} is malformed`);
    }
    const described = describePath(path);
    const suggestion = {
      id,
      ...described,
      path,
      score,
      cosine,
      prior,
      duplicates,
      label: displayLabel(described.name),
      auxLabel: index.classes[auxClass] ?? '',
      probability: predictionByClass.get(auxClass) ?? 0,
      url: expectedIconUrl(index.source, path),
      approved: true,
    };
    assertApprovedIconSource(suggestion, `Retrieved icon ${id}`, index.source);
    suggestions.push(Object.freeze(suggestion));
  }
  return suggestions;
}

export async function loadIconRetrievalIndex({
  indexUrl,
  embedder,
  fetchImpl = globalThis.fetch,
} = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('A fetch implementation is required');
  if (typeof indexUrl !== 'string' || indexUrl.length === 0) {
    throw new Error('An icon index URL is required');
  }
  const response = await fetchImpl(indexUrl);
  if (!response?.ok) {
    throw new Error(`Icon index request failed: ${response?.status ?? 'unknown'}`);
  }
  return createIconRetrievalIndex(await response.json(), {
    baseUrl: indexUrl.replace(/[^/]*$/, ''),
    embedder,
  });
}
