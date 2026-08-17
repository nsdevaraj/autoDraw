import { rasterizePolylines } from './sketch-rasterizer.mjs';

const DEFAULT_METADATA_URL = new URL(
  '../models/sketch-embedder/model.json',
  import.meta.url,
).href;
const DEFAULT_MODEL_URL = new URL(
  '../models/sketch-embedder/sketch-embedder.onnx',
  import.meta.url,
).href;

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function arraysEqual(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function modelContract(metadata) {
  if (!isRecord(metadata) || metadata.schemaVersion !== 1 || metadata.kind !== 'sketch-embedder') {
    throw new Error('Invalid sketch embedder metadata');
  }

  const classes = metadata.classes;
  if (
    !Array.isArray(classes)
    || classes.length < 2
    || classes.some(label => typeof label !== 'string' || label.length === 0)
    || new Set(classes).size !== classes.length
  ) {
    throw new Error('Embedder metadata must contain unique class labels');
  }

  const rasterizer = metadata.rasterizer;
  if (!isRecord(rasterizer) || !Number.isInteger(rasterizer.size)) {
    throw new Error('Embedder metadata is missing rasterizer settings');
  }
  const rasterizerOptions = {
    size: rasterizer.size,
    padding: rasterizer.padding,
    strokeWidth: rasterizer.strokeWidth,
    supersample: rasterizer.supersample,
  };
  rasterizePolylines([], rasterizerOptions);

  const model = metadata.model;
  const input = model?.input;
  const embedding = model?.outputs?.embedding;
  const logits = model?.outputs?.logits;
  if (model?.format !== 'ONNX') throw new Error('Embedder model must use ONNX format');
  if (!Number.isInteger(model.embeddingDim) || model.embeddingDim < 8) {
    throw new Error('Embedder metadata must declare an embedding dimension of at least 8');
  }
  if (
    !isRecord(input)
    || typeof input.name !== 'string'
    || input.dtype !== 'float32'
    || input.normalization !== 'uint8 / 255'
    || !arraysEqual(input.shape, ['batch', 1, rasterizer.size, rasterizer.size])
  ) {
    throw new Error('Embedder metadata has an unsupported input shape or normalization');
  }
  if (
    !isRecord(embedding)
    || typeof embedding.name !== 'string'
    || embedding.dtype !== 'float32'
    || embedding.normalization !== 'L2'
    || !arraysEqual(embedding.shape, ['batch', model.embeddingDim])
  ) {
    throw new Error('Embedder metadata has an unsupported embedding output');
  }
  if (
    !isRecord(logits)
    || typeof logits.name !== 'string'
    || logits.dtype !== 'float32'
    || !arraysEqual(logits.shape, ['batch', classes.length])
  ) {
    throw new Error('Embedder metadata has an unsupported logits output');
  }

  return {
    classes: Object.freeze([...classes]),
    embeddingDim: model.embeddingDim,
    inputName: input.name,
    inputShape: Object.freeze([...input.shape]),
    embeddingName: embedding.name,
    embeddingShape: Object.freeze([...embedding.shape]),
    logitsName: logits.name,
    logitsShape: Object.freeze([...logits.shape]),
    rasterizer: Object.freeze(rasterizerOptions),
  };
}

function matchesTensorMetadata(metadata, name, shape) {
  const entry = Array.isArray(metadata)
    ? metadata.find(candidate => candidate?.name === name)
    : undefined;
  return entry?.isTensor === true
    && entry.type === 'float32'
    && arraysEqual(entry.shape, shape);
}

function assertSessionContract(session, contract) {
  if (!session || typeof session.run !== 'function' || typeof session.release !== 'function') {
    throw new Error('Embedder session must provide run() and release()');
  }
  if (!arraysEqual(session.inputNames, [contract.inputName])) {
    throw new Error('Embedder session input does not match metadata');
  }
  if (!arraysEqual(session.outputNames, [contract.embeddingName, contract.logitsName])) {
    throw new Error('Embedder session outputs do not match metadata');
  }
  if (
    !Array.isArray(session.inputMetadata)
    || session.inputMetadata.length !== 1
    || !matchesTensorMetadata(session.inputMetadata, contract.inputName, contract.inputShape)
  ) {
    throw new Error('Embedder session input metadata does not match model metadata');
  }
  if (
    !Array.isArray(session.outputMetadata)
    || session.outputMetadata.length !== 2
    || !matchesTensorMetadata(session.outputMetadata, contract.embeddingName, contract.embeddingShape)
    || !matchesTensorMetadata(session.outputMetadata, contract.logitsName, contract.logitsShape)
  ) {
    throw new Error('Embedder session output metadata does not match model metadata');
  }
}

function readEmbedding(output, contract) {
  if (
    !isRecord(output)
    || output.type !== 'float32'
    || !arraysEqual(output.dims, [1, contract.embeddingDim])
    || output.data?.length !== contract.embeddingDim
  ) {
    throw new Error('Embedder embedding output does not match metadata');
  }
  const embedding = Float32Array.from(output.data);
  if (embedding.some(value => !Number.isFinite(value))) {
    throw new Error('Embedder must produce a finite embedding');
  }

  let norm = 0;
  for (const value of embedding) norm += value * value;
  // The index stores unit vectors, so a drifting norm would silently rescale every score.
  if (Math.abs(Math.sqrt(norm) - 1) > 1e-3) {
    throw new Error('Embedder embedding is not unit length');
  }
  return embedding;
}

function rankedPredictions(output, contract, limit) {
  if (
    !isRecord(output)
    || output.type !== 'float32'
    || !arraysEqual(output.dims, [1, contract.classes.length])
    || output.data?.length !== contract.classes.length
  ) {
    throw new Error('Embedder logits output does not match metadata');
  }

  const logits = Array.from(output.data);
  if (logits.some(value => !Number.isFinite(value))) {
    throw new Error('Embedder must produce finite logits');
  }
  const largestLogit = Math.max(...logits);
  const exponentials = logits.map(value => Math.exp(value - largestLogit));
  const total = exponentials.reduce((sum, value) => sum + value, 0);

  return exponentials
    .map((value, classIndex) => ({
      classIndex,
      label: contract.classes[classIndex],
      logit: logits[classIndex],
      probability: value / total,
    }))
    .sort((left, right) => (
      right.logit - left.logit || left.classIndex - right.classIndex
    ))
    .slice(0, Math.min(limit, contract.classes.length))
    .map(({ classIndex, label, probability }) => Object.freeze({ classIndex, label, probability }));
}

export function createSketchEmbedder({ metadata, runtime, session }) {
  if (typeof runtime?.Tensor !== 'function') {
    throw new Error('ONNX Runtime Tensor constructor is required');
  }
  const contract = modelContract(metadata);
  assertSessionContract(session, contract);
  let disposed = false;
  let disposal;
  const activeRuns = new Set();

  async function embed(polylines, { limit = 5 } = {}) {
    if (disposed) throw new Error('Sketch embedder has been disposed');
    if (!Number.isInteger(limit) || limit < 1) {
      throw new Error('Prediction limit must be a positive integer');
    }
    const bitmap = rasterizePolylines(polylines, contract.rasterizer);
    if (!bitmap.some(value => value > 0)) return null;

    const normalized = Float32Array.from(bitmap, value => value / 255);
    const tensor = new runtime.Tensor(
      'float32',
      normalized,
      [1, 1, contract.rasterizer.size, contract.rasterizer.size],
    );
    const run = Promise.resolve(session.run({ [contract.inputName]: tensor }));
    activeRuns.add(run);
    let outputs;
    try {
      outputs = await run;
    } finally {
      activeRuns.delete(run);
    }
    return Object.freeze({
      embedding: readEmbedding(outputs?.[contract.embeddingName], contract),
      predictions: Object.freeze(
        rankedPredictions(outputs?.[contract.logitsName], contract, limit),
      ),
    });
  }

  function dispose() {
    if (!disposal) {
      disposed = true;
      disposal = Promise.allSettled([...activeRuns]).then(() => session.release());
    }
    return disposal;
  }

  return Object.freeze({
    classes: contract.classes,
    embeddingDim: contract.embeddingDim,
    embed,
    dispose,
  });
}

function fileProtocol(value) {
  try {
    return new URL(value, import.meta.url).protocol === 'file:';
  } catch {
    return false;
  }
}

export async function loadSketchEmbedder({
  runtime,
  metadataUrl = DEFAULT_METADATA_URL,
  modelUrl = DEFAULT_MODEL_URL,
  fetchImpl = globalThis.fetch,
  sessionOptions = {},
} = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('A fetch implementation is required');
  if (typeof runtime?.InferenceSession?.create !== 'function') {
    throw new Error('ONNX Runtime InferenceSession is required');
  }
  if (
    fetchImpl === globalThis.fetch
    && (fileProtocol(metadataUrl) || fileProtocol(modelUrl))
  ) {
    throw new Error('Default embedder loading requires an HTTP browser context');
  }

  const response = await fetchImpl(metadataUrl);
  if (!response?.ok) {
    throw new Error(`Embedder metadata request failed: ${response?.status ?? 'unknown'}`);
  }
  const metadata = await response.json();
  modelContract(metadata);
  const session = await runtime.InferenceSession.create(modelUrl, {
    ...sessionOptions,
    executionProviders: ['wasm'],
  });
  try {
    return { embedder: createSketchEmbedder({ metadata, runtime, session }), metadata };
  } catch (error) {
    try {
      await session.release?.();
    } catch {}
    throw error;
  }
}
