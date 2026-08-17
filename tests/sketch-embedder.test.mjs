import assert from 'node:assert/strict';
import test from 'node:test';

import { createSketchEmbedder, loadSketchEmbedder } from '../src/sketch-embedder.mjs';

const EMBEDDING_DIM = 8;

function modelMetadata(overrides = {}) {
  return {
    schemaVersion: 1,
    kind: 'sketch-embedder',
    classes: ['cat', 'dog', 'bird'],
    rasterizer: {
      size: 64,
      padding: 4,
      strokeWidth: 2.5,
      supersample: 1,
    },
    model: {
      filename: 'sketch-embedder.onnx',
      format: 'ONNX',
      embeddingDim: EMBEDDING_DIM,
      input: {
        name: 'bitmap',
        shape: ['batch', 1, 64, 64],
        dtype: 'float32',
        normalization: 'uint8 / 255',
      },
      outputs: {
        embedding: {
          name: 'embedding',
          shape: ['batch', EMBEDDING_DIM],
          dtype: 'float32',
          normalization: 'L2',
        },
        logits: {
          name: 'logits',
          shape: ['batch', 3],
          dtype: 'float32',
        },
      },
    },
    ...overrides,
  };
}

class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

function unitEmbedding() {
  const values = new Float32Array(EMBEDDING_DIM);
  values[0] = 1;
  return values;
}

function fakeSession({ embedding = unitEmbedding(), logits = [1, 3, 0] } = {}) {
  const calls = [];
  return {
    calls,
    releaseCalls: 0,
    inputNames: ['bitmap'],
    outputNames: ['embedding', 'logits'],
    inputMetadata: [{
      name: 'bitmap',
      isTensor: true,
      type: 'float32',
      shape: ['batch', 1, 64, 64],
    }],
    outputMetadata: [
      {
        name: 'embedding',
        isTensor: true,
        type: 'float32',
        shape: ['batch', EMBEDDING_DIM],
      },
      {
        name: 'logits',
        isTensor: true,
        type: 'float32',
        shape: ['batch', 3],
      },
    ],
    run(feeds) {
      calls.push(feeds);
      return {
        embedding: new FakeTensor('float32', embedding, [1, EMBEDDING_DIM]),
        logits: new FakeTensor('float32', Float32Array.from(logits), [1, logits.length]),
      };
    },
    release() {
      this.releaseCalls += 1;
    },
  };
}

const runtime = { Tensor: FakeTensor };
const square = [[[8, 8], [40, 8], [40, 40], [8, 40], [8, 8]]];

test('the embedder exposes its classes and embedding dimension', () => {
  const embedder = createSketchEmbedder({
    metadata: modelMetadata(),
    runtime,
    session: fakeSession(),
  });
  assert.deepEqual(embedder.classes, ['cat', 'dog', 'bird']);
  assert.equal(embedder.embeddingDim, EMBEDDING_DIM);
});

test('embedding a sketch returns the vector and ranked auxiliary predictions', async () => {
  const embedder = createSketchEmbedder({
    metadata: modelMetadata(),
    runtime,
    session: fakeSession({ logits: [1, 3, 0] }),
  });

  const result = await embedder.embed(square, { limit: 2 });
  assert.equal(result.embedding instanceof Float32Array, true);
  assert.equal(result.embedding.length, EMBEDDING_DIM);
  assert.deepEqual(result.predictions.map(item => item.label), ['dog', 'cat']);
  assert.deepEqual(result.predictions.map(item => item.classIndex), [1, 0]);
  assert.equal(result.predictions[0].probability > result.predictions[1].probability, true);
});

test('an empty sketch is reported as nothing to embed', async () => {
  const embedder = createSketchEmbedder({
    metadata: modelMetadata(),
    runtime,
    session: fakeSession(),
  });
  assert.equal(await embedder.embed([]), null);
});

test('the model input is normalized to the rasterizer shape', async () => {
  const session = fakeSession();
  const embedder = createSketchEmbedder({ metadata: modelMetadata(), runtime, session });
  await embedder.embed(square);

  const tensor = session.calls[0].bitmap;
  assert.deepEqual(tensor.dims, [1, 1, 64, 64]);
  assert.equal(tensor.type, 'float32');
  assert.equal(tensor.data.every(value => value >= 0 && value <= 1), true);
});

test('metadata that does not describe this embedder is rejected', () => {
  const cases = [
    ['kind', { kind: 'quickdraw-classifier' }],
    ['schema version', { schemaVersion: 2 }],
    ['duplicate classes', { classes: ['cat', 'cat'] }],
  ];
  for (const [reason, overrides] of cases) {
    assert.throws(
      () => createSketchEmbedder({
        metadata: modelMetadata(overrides),
        runtime,
        session: fakeSession(),
      }),
      /embedder/i,
      reason,
    );
  }
});

test('output declarations that disagree with the contract are rejected', () => {
  const metadata = modelMetadata();
  metadata.model.outputs.embedding.normalization = 'none';
  assert.throws(
    () => createSketchEmbedder({ metadata, runtime, session: fakeSession() }),
    /embedding output/,
  );

  const mismatched = modelMetadata();
  mismatched.model.outputs.logits.shape = ['batch', 4];
  assert.throws(
    () => createSketchEmbedder({ metadata: mismatched, runtime, session: fakeSession() }),
    /logits output/,
  );
});

test('a session whose tensors differ from the metadata is rejected', () => {
  const session = fakeSession();
  session.outputNames = ['logits', 'embedding'];
  assert.throws(
    () => createSketchEmbedder({ metadata: modelMetadata(), runtime, session }),
    /session outputs/,
  );

  const reshaped = fakeSession();
  reshaped.outputMetadata[0].shape = ['batch', 16];
  assert.throws(
    () => createSketchEmbedder({ metadata: modelMetadata(), runtime, session: reshaped }),
    /output metadata/,
  );
});

// The index stores unit vectors, so a non-normalized embedding would rescale every score.
test('an embedding that is not unit length is rejected', async () => {
  const stretched = new Float32Array(EMBEDDING_DIM);
  stretched[0] = 4;
  const embedder = createSketchEmbedder({
    metadata: modelMetadata(),
    runtime,
    session: fakeSession({ embedding: stretched }),
  });
  await assert.rejects(() => embedder.embed(square), /unit length/);
});

test('disposing releases the session and blocks further embedding', async () => {
  const session = fakeSession();
  const embedder = createSketchEmbedder({ metadata: modelMetadata(), runtime, session });
  await embedder.dispose();
  assert.equal(session.releaseCalls, 1);
  await assert.rejects(() => embedder.embed(square), /disposed/);
});

test('loading releases the session when the contract check fails', async () => {
  const session = fakeSession();
  session.inputNames = ['pixels'];
  await assert.rejects(
    () => loadSketchEmbedder({
      runtime: {
        Tensor: FakeTensor,
        InferenceSession: { create: async () => session },
      },
      metadataUrl: 'https://example.test/model.json',
      modelUrl: 'https://example.test/model.onnx',
      fetchImpl: async () => ({ ok: true, json: async () => modelMetadata() }),
    }),
    /session input/,
  );
  assert.equal(session.releaseCalls, 1);
});

test('loading reports a failed metadata request', async () => {
  await assert.rejects(
    () => loadSketchEmbedder({
      runtime: { Tensor: FakeTensor, InferenceSession: { create: async () => fakeSession() } },
      metadataUrl: 'https://example.test/model.json',
      modelUrl: 'https://example.test/model.onnx',
      fetchImpl: async () => ({ ok: false, status: 503 }),
    }),
    /503/,
  );
});
