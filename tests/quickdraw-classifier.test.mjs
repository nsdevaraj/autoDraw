import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createQuickDrawClassifier,
  loadQuickDrawClassifier,
} from '../src/quickdraw-classifier.mjs';

function modelMetadata(overrides = {}) {
  return {
    schemaVersion: 1,
    kind: 'quickdraw-classifier',
    classes: ['cat', 'dog', 'bird'],
    rasterizer: {
      size: 64,
      padding: 4,
      strokeWidth: 2.5,
      supersample: 1,
    },
    model: {
      filename: 'quickdraw.onnx',
      format: 'ONNX',
      input: {
        name: 'bitmap',
        shape: ['batch', 1, 64, 64],
        dtype: 'float32',
        normalization: 'uint8 / 255',
      },
      output: {
        name: 'logits',
        shape: ['batch', 3],
        dtype: 'float32',
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

function fakeSession(logits = [1, 3, 0]) {
  const calls = [];
  const session = {
    calls,
    releaseCalls: 0,
    inputNames: ['bitmap'],
    outputNames: ['logits'],
    inputMetadata: [{
      name: 'bitmap',
      isTensor: true,
      type: 'float32',
      shape: ['batch', 1, 64, 64],
    }],
    outputMetadata: [{
      name: 'logits',
      isTensor: true,
      type: 'float32',
      shape: ['batch', logits.length],
    }],
    async run(feeds) {
      calls.push(feeds);
      return {
        logits: {
          type: 'float32',
          data: Float32Array.from(logits),
          dims: [1, logits.length],
        },
      };
    },
    async release() {
      session.releaseCalls += 1;
    },
  };
  return session;
}

test('classifier rasterizes polylines and returns stable top-k probabilities', async () => {
  const session = fakeSession();
  const classifier = createQuickDrawClassifier({
    metadata: modelMetadata(),
    runtime: { Tensor: FakeTensor },
    session,
  });

  const predictions = await classifier.classify([
    [[0, 0], [20, 0], [20, 10]],
  ], { limit: 2 });

  assert.deepEqual(classifier.classes, ['cat', 'dog', 'bird']);
  assert.deepEqual(predictions.map(item => item.label), ['dog', 'cat']);
  assert.deepEqual(predictions.map(item => item.classIndex), [1, 0]);
  assert.equal(Math.abs(predictions[0].probability - 0.8437947345) < 1e-6, true);
  assert.equal(session.calls.length, 1);

  const tensor = session.calls[0].bitmap;
  assert.equal(tensor instanceof FakeTensor, true);
  assert.equal(tensor.type, 'float32');
  assert.deepEqual(tensor.dims, [1, 1, 64, 64]);
  assert.equal(tensor.data instanceof Float32Array, true);
  assert.equal(tensor.data.some(value => value > 0), true);
  assert.equal(tensor.data.every(value => value >= 0 && value <= 1), true);
});

test('blank drawings return no predictions without running inference', async () => {
  const session = fakeSession();
  const classifier = createQuickDrawClassifier({
    metadata: modelMetadata(),
    runtime: { Tensor: FakeTensor },
    session,
  });

  assert.deepEqual(await classifier.classify([]), []);
  assert.deepEqual(await classifier.classify([[]]), []);
  assert.equal(session.calls.length, 0);
});

test('top-k ordering follows logits when softmax probabilities underflow', async () => {
  const classifier = createQuickDrawClassifier({
    metadata: modelMetadata(),
    runtime: { Tensor: FakeTensor },
    session: fakeSession([1000, -1001, -1000]),
  });

  const predictions = await classifier.classify([[[0, 0]]], { limit: 3 });

  assert.deepEqual(predictions.map(item => item.label), ['cat', 'bird', 'dog']);
  assert.deepEqual(predictions.map(item => item.probability), [1, 0, 0]);
});

test('classifier validates metadata, session, limits, and inference output', async () => {
  const create = (metadata, session = fakeSession()) => createQuickDrawClassifier({
    metadata,
    runtime: { Tensor: FakeTensor },
    session,
  });

  assert.throws(
    () => create(modelMetadata({ classes: ['cat', 'cat', 'bird'] })),
    /unique class labels/,
  );
  assert.throws(
    () => create(modelMetadata({
      rasterizer: { ...modelMetadata().rasterizer, strokeWidth: 0 },
    })),
    /stroke width must be positive/,
  );
  assert.throws(
    () => create(modelMetadata({
      model: {
        ...modelMetadata().model,
        input: { ...modelMetadata().model.input, shape: ['batch', 1, 32, 32] },
      },
    })),
    /input shape/,
  );
  assert.throws(
    () => create(modelMetadata(), { ...fakeSession(), outputNames: ['scores'] }),
    /session output/,
  );
  assert.throws(
    () => create(modelMetadata(), {
      ...fakeSession(),
      inputMetadata: [{
        name: 'bitmap',
        isTensor: true,
        type: 'float32',
        shape: ['batch', 1, 32, 32],
      }],
    }),
    /session input metadata/,
  );
  assert.throws(
    () => create(modelMetadata(), {
      ...fakeSession(),
      outputMetadata: [{
        name: 'logits',
        isTensor: true,
        type: 'int32',
        shape: ['batch', 3],
      }],
    }),
    /session output metadata/,
  );

  const classifier = create(modelMetadata());
  await assert.rejects(
    () => classifier.classify([[[0, 0]]], { limit: 0 }),
    /limit must be a positive integer/,
  );

  const malformed = create(modelMetadata(), fakeSession([1, Number.NaN, 0]));
  await assert.rejects(
    () => malformed.classify([[[0, 0]]]),
    /finite logits/,
  );

  const wrongOutputTypeSession = fakeSession();
  wrongOutputTypeSession.run = async () => ({
    logits: {
      type: 'int32',
      data: Int32Array.from([1, 3, 0]),
      dims: [1, 3],
    },
  });
  const wrongOutputType = create(modelMetadata(), wrongOutputTypeSession);
  await assert.rejects(
    () => wrongOutputType.classify([[[0, 0]]]),
    /output type must be float32/,
  );
});

test('classifier releases its session once and rejects inference after disposal', async () => {
  const session = fakeSession();
  const classifier = createQuickDrawClassifier({
    metadata: modelMetadata(),
    runtime: { Tensor: FakeTensor },
    session,
  });

  await classifier.dispose();
  await classifier.dispose();

  assert.equal(session.releaseCalls, 1);
  await assert.rejects(
    () => classifier.classify([[[0, 0]]]),
    /classifier has been disposed/,
  );
});

test('classifier waits for active inference before releasing its session', async () => {
  const session = fakeSession();
  let finishRun;
  let runSettled = false;
  session.run = () => new Promise(resolve => {
    finishRun = () => {
      runSettled = true;
      resolve({
        logits: {
          type: 'float32',
          data: Float32Array.from([1, 3, 0]),
          dims: [1, 3],
        },
      });
    };
  });
  session.release = async () => {
    assert.equal(runSettled, true);
    session.releaseCalls += 1;
  };
  const classifier = createQuickDrawClassifier({
    metadata: modelMetadata(),
    runtime: { Tensor: FakeTensor },
    session,
  });

  const classification = classifier.classify([[[0, 0]]]);
  const disposal = classifier.dispose();
  await Promise.resolve();
  assert.equal(session.releaseCalls, 0);
  await assert.rejects(
    () => classifier.classify([[[0, 0]]]),
    /classifier has been disposed/,
  );

  finishRun();
  assert.equal((await classification)[0].label, 'dog');
  await disposal;
  assert.equal(session.releaseCalls, 1);
});

test('loader fetches metadata and creates a WASM session', async () => {
  const metadata = modelMetadata();
  const session = fakeSession();
  const requests = [];
  const sessionCreations = [];
  const runtime = {
    Tensor: FakeTensor,
    InferenceSession: {
      async create(url, options) {
        sessionCreations.push({ url, options });
        return session;
      },
    },
  };

  const classifier = await loadQuickDrawClassifier({
    runtime,
    metadataUrl: '/models/quickdraw/model.json',
    modelUrl: '/models/quickdraw/model.onnx',
    fetchImpl: async url => {
      requests.push(url);
      return { ok: true, json: async () => metadata };
    },
    sessionOptions: {
      executionProviders: ['webgl'],
      graphOptimizationLevel: 'disabled',
    },
  });

  assert.deepEqual(requests, ['/models/quickdraw/model.json']);
  assert.deepEqual(sessionCreations, [{
    url: '/models/quickdraw/model.onnx',
    options: {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
    },
  }]);
  assert.deepEqual(classifier.classes, metadata.classes);

  await assert.rejects(
    () => loadQuickDrawClassifier({
      runtime,
      metadataUrl: '/invalid.json',
      modelUrl: '/model.onnx',
      fetchImpl: async () => ({
        ok: true,
        json: async () => modelMetadata({ classes: ['cat', 'cat', 'bird'] }),
      }),
    }),
    /unique class labels/,
  );
  assert.equal(sessionCreations.length, 1);

  await assert.rejects(
    () => loadQuickDrawClassifier({
      runtime,
      metadataUrl: '/missing.json',
      modelUrl: '/model.onnx',
      fetchImpl: async () => ({ ok: false, status: 404 }),
    }),
    /metadata request failed: 404/,
  );

  const invalidSession = fakeSession();
  invalidSession.outputNames = ['scores'];
  await assert.rejects(
    () => loadQuickDrawClassifier({
      runtime: {
        Tensor: FakeTensor,
        InferenceSession: { create: async () => invalidSession },
      },
      metadataUrl: '/valid.json',
      modelUrl: '/invalid.onnx',
      fetchImpl: async () => ({ ok: true, json: async () => metadata }),
    }),
    /session output/,
  );
  assert.equal(invalidSession.releaseCalls, 1);
});

test('default loader URLs require an HTTP browser context', async () => {
  await assert.rejects(
    () => loadQuickDrawClassifier({
      runtime: {
        Tensor: FakeTensor,
        InferenceSession: { create: async () => fakeSession() },
      },
    }),
    /HTTP browser context/,
  );
});