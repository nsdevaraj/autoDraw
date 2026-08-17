import { rasterizePolylines } from './sketch-rasterizer.mjs';

const DEFAULT_METADATA_URL = new URL(
  '../models/quickdraw-mvp/model.json',
  import.meta.url,
).href;
const DEFAULT_MODEL_URL = new URL(
  '../models/quickdraw-mvp/quickdraw-mvp.onnx',
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
  if (!isRecord(metadata) || metadata.schemaVersion !== 1 || metadata.kind !== 'quickdraw-classifier') {
    throw new Error('Invalid Quick Draw classifier metadata');
  }

  const classes = metadata.classes;
  if (
    !Array.isArray(classes)
    || classes.length < 2
    || classes.some(label => typeof label !== 'string' || label.length === 0)
    || new Set(classes).size !== classes.length
  ) {
    throw new Error('Classifier metadata must contain unique class labels');
  }

  const rasterizer = metadata.rasterizer;
  if (!isRecord(rasterizer) || !Number.isInteger(rasterizer.size)) {
    throw new Error('Classifier metadata is missing rasterizer settings');
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
  const output = model?.output;
  if (model?.format !== 'ONNX') throw new Error('Classifier model must use ONNX format');
  if (
    !isRecord(input)
    || typeof input.name !== 'string'
    || input.dtype !== 'float32'
    || input.normalization !== 'uint8 / 255'
    || !arraysEqual(input.shape, ['batch', 1, rasterizer.size, rasterizer.size])
  ) {
    throw new Error('Classifier metadata has an unsupported input shape or normalization');
  }
  if (
    !isRecord(output)
    || typeof output.name !== 'string'
    || output.dtype !== 'float32'
    || !arraysEqual(output.shape, ['batch', classes.length])
  ) {
    throw new Error('Classifier metadata has an unsupported output shape');
  }

  return {
    classes: Object.freeze([...classes]),
    inputName: input.name,
    inputShape: Object.freeze([...input.shape]),
    outputName: output.name,
    outputShape: Object.freeze([...output.shape]),
    rasterizer: Object.freeze(rasterizerOptions),
  };
}

function matchesTensorMetadata(metadata, name, shape) {
  return Array.isArray(metadata)
    && metadata.length === 1
    && metadata[0]?.name === name
    && metadata[0]?.isTensor === true
    && metadata[0]?.type === 'float32'
    && arraysEqual(metadata[0]?.shape, shape);
}

function assertSessionContract(session, contract) {
  if (!session || typeof session.run !== 'function' || typeof session.release !== 'function') {
    throw new Error('Classifier session must provide run() and release()');
  }
  if (!arraysEqual(session.inputNames, [contract.inputName])) {
    throw new Error('Classifier session input does not match metadata');
  }
  if (!arraysEqual(session.outputNames, [contract.outputName])) {
    throw new Error('Classifier session output does not match metadata');
  }
  if (!matchesTensorMetadata(session.inputMetadata, contract.inputName, contract.inputShape)) {
    throw new Error('Classifier session input metadata does not match model metadata');
  }
  if (!matchesTensorMetadata(session.outputMetadata, contract.outputName, contract.outputShape)) {
    throw new Error('Classifier session output metadata does not match model metadata');
  }
}

function rankedPredictions(output, contract, limit) {
  if (output?.type !== 'float32') {
    throw new Error('Classifier output type must be float32');
  }
  if (
    !isRecord(output)
    || !arraysEqual(output.dims, [1, contract.classes.length])
    || !output.data
    || output.data.length !== contract.classes.length
  ) {
    throw new Error('Classifier output shape does not match metadata');
  }

  const logits = Array.from(output.data);
  if (logits.some(value => typeof value !== 'number' || !Number.isFinite(value))) {
    throw new Error('Classifier output must contain finite logits');
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
    .map(({ classIndex, label, probability }) => ({ classIndex, label, probability }));
}

export function createQuickDrawClassifier({ metadata, runtime, session }) {
  if (typeof runtime?.Tensor !== 'function') {
    throw new Error('ONNX Runtime Tensor constructor is required');
  }
  const contract = modelContract(metadata);
  assertSessionContract(session, contract);
  let disposed = false;
  let disposal;
  const activeRuns = new Set();

  async function classify(polylines, { limit = 5 } = {}) {
    if (disposed) throw new Error('Quick Draw classifier has been disposed');
    if (!Number.isInteger(limit) || limit < 1) {
      throw new Error('Prediction limit must be a positive integer');
    }
    const bitmap = rasterizePolylines(polylines, contract.rasterizer);
    if (!bitmap.some(value => value > 0)) return [];

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
    return rankedPredictions(outputs?.[contract.outputName], contract, limit);
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
    classify,
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

export async function loadQuickDrawClassifier({
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
    throw new Error('Default classifier loading requires an HTTP browser context');
  }

  const response = await fetchImpl(metadataUrl);
  if (!response?.ok) {
    throw new Error(`Classifier metadata request failed: ${response?.status ?? 'unknown'}`);
  }
  const metadata = await response.json();
  modelContract(metadata);
  const session = await runtime.InferenceSession.create(modelUrl, {
    ...sessionOptions,
    executionProviders: ['wasm'],
  });
  try {
    return createQuickDrawClassifier({ metadata, runtime, session });
  } catch (error) {
    try {
      await session.release?.();
    } catch {}
    throw error;
  }
}