import ort from '/vendor/ort.wasm.min.mjs';

import {
  createApprovedCandidateIndex,
  createCandidateSuggestionSource,
  createRetrievalSuggestionSource,
} from './autodraw-suggestions.mjs';
import {
  clearDrawing,
  commitStroke,
  createDrawingState,
  draftPolylines,
  placeSuggestion,
  redoDrawing,
  resizeDrawing,
  undoDrawing,
} from './drawing-state.mjs';
import { createDrawingSvg } from './drawing-export.mjs';
import { loadIconRetrievalIndex } from './icon-retrieval.mjs';
import { loadQuickDrawClassifier } from './quickdraw-classifier.mjs';
import { loadSketchEmbedder } from './sketch-embedder.mjs';

const MODEL_METADATA_URL = '/models/quickdraw-mvp/model.json';
const MODEL_URL = '/models/quickdraw-mvp/quickdraw-mvp.onnx';
const EMBEDDER_METADATA_URL = '/models/sketch-embedder/model.json';
const EMBEDDER_MODEL_URL = '/models/sketch-embedder/sketch-embedder.onnx';
const ICON_INDEX_URL = '/data/icon-embeddings/index.json';
const CANDIDATES_URL = '/data/quickdraw-candidates.json';
const SUGGESTION_LIMIT = 12;
const RECOGNITION_DELAY = 120;

const stage = document.getElementById('stage');
const canvas = document.getElementById('drawingCanvas');
const context = canvas.getContext('2d');
const placedLayer = document.getElementById('placedLayer');
const suggestionList = document.getElementById('suggestionList');
const suggestionState = document.getElementById('suggestionState');
const runtimeStatus = document.getElementById('runtimeStatus');
const runtimeStatusText = document.getElementById('runtimeStatusText');
const undoButton = document.getElementById('undoButton');
const redoButton = document.getElementById('redoButton');
const clearButton = document.getElementById('clearButton');
const downloadButton = document.getElementById('downloadButton');
const strokeWidthInput = document.getElementById('strokeWidth');
const toastElement = document.getElementById('toast');

let drawing = createDrawingState();
let suggestionSource;
let activeModel;
let activeStroke;
let activePointerId;
let strokeColor = '#17212b';
let strokeWidth = Number(strokeWidthInput.value);
let recognitionTimer;
let recognitionVersion = 0;
let suggestions = [];
let toastTimer;
let renderedIcons;
let stageSize;

function setRuntimeStatus(message, state = '') {
  runtimeStatusText.textContent = message;
  runtimeStatus.className = `runtime-status${state ? ` ${state}` : ''}`;
}

function toast(message) {
  toastElement.textContent = message;
  toastElement.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastElement.classList.remove('visible'), 1800);
}

function stagePoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return [
    Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)),
    Math.max(0, Math.min(bounds.height, event.clientY - bounds.top)),
  ];
}

function appendPoint(points, point) {
  const previous = points[points.length - 1];
  if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 0.45) {
    points.push(point);
  }
}

function drawStroke(stroke) {
  const points = stroke.points;
  if (points.length === 0) return;
  context.strokeStyle = stroke.color;
  context.fillStyle = stroke.color;
  context.lineWidth = stroke.width;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  if (points.length === 1) {
    context.beginPath();
    context.arc(points[0][0], points[0][1], stroke.width / 2, 0, Math.PI * 2);
    context.fill();
    return;
  }
  context.beginPath();
  context.moveTo(points[0][0], points[0][1]);
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    const midpointX = (previous[0] + current[0]) / 2;
    const midpointY = (previous[1] + current[1]) / 2;
    context.quadraticCurveTo(previous[0], previous[1], midpointX, midpointY);
  }
  const last = points[points.length - 1];
  context.lineTo(last[0], last[1]);
  context.stroke();
}

function renderCanvas() {
  const bounds = stage.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  const targetWidth = Math.round(width * pixelRatio);
  const targetHeight = Math.round(height * pixelRatio);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  drawing.draftStrokes.forEach(drawStroke);
  if (activeStroke) drawStroke(activeStroke);
}

// Icons come from the commit-pinned CDN URL by default: it is the approved URL the index
// derives, and it works under any static server. The local /svgdepot/ route only exists when
// the bundled server runs, so it is the fallback (and can be forced with ?icons=local for an
// offline .cache/svgdepot). Whichever source answers first wins for the rest of the session.
let preferLocalIcons = new URLSearchParams(location.search).get('icons') === 'local';

function localIconUrl(path) {
  return new URL(`/svgdepot/${path.split('/').map(encodeURIComponent).join('/')}`, location.href).href;
}

function iconSources(icon) {
  const local = localIconUrl(icon.path);
  const primaryIsLocal = preferLocalIcons;
  return primaryIsLocal
    ? { primary: local, fallback: icon.url, primaryIsLocal }
    : { primary: icon.url, fallback: local, primaryIsLocal };
}

function loadIconImage(image, icon) {
  const { primary, fallback, primaryIsLocal } = iconSources(icon);
  image.addEventListener('error', () => {
    if (image.src === fallback) return;
    // Assigning the opposite of the source that just failed keeps a whole batch of
    // simultaneous failures from toggling the preference back and forth.
    preferLocalIcons = !primaryIsLocal;
    image.src = fallback;
  }, { once: true });
  image.src = primary;
}

function renderIcons() {
  if (renderedIcons === drawing.icons) return;
  drawing.icons.forEach((icon, index) => {
    let image = placedLayer.children[index];
    if (!image || image.dataset.iconPath !== icon.path) {
      image = document.createElement('img');
      image.className = 'placed-icon';
      image.dataset.iconPath = icon.path;
      loadIconImage(image, icon);
      const current = placedLayer.children[index];
      if (current) current.replaceWith(image);
      else placedLayer.appendChild(image);
    }
    image.alt = '';
    image.draggable = false;
    image.style.left = `${icon.x}px`;
    image.style.top = `${icon.y}px`;
    image.style.width = `${icon.width}px`;
    image.style.height = `${icon.height}px`;
  });
  while (placedLayer.children.length > drawing.icons.length) {
    placedLayer.lastElementChild.remove();
  }
  renderedIcons = drawing.icons;
}

function renderControls() {
  undoButton.disabled = drawing.past.length === 0;
  redoButton.disabled = drawing.future.length === 0;
  const empty = drawing.draftStrokes.length === 0 && drawing.icons.length === 0;
  clearButton.disabled = empty;
  downloadButton.disabled = empty;
}

function renderScene() {
  renderCanvas();
  renderIcons();
  renderControls();
}

function renderSuggestions() {
  if (suggestions.length === 0) {
    suggestionList.replaceChildren();
    return;
  }
  const fragment = document.createDocumentFragment();
  suggestions.forEach((suggestion, index) => {
    const button = document.createElement('button');
    button.className = 'suggestion-button';
    button.type = 'button';
    button.title = `${suggestion.label}: ${suggestion.name}`;
    button.setAttribute('aria-label', `Use ${suggestion.label} icon`);

    const image = document.createElement('img');
    loadIconImage(image, suggestion);
    image.alt = '';
    image.loading = index < 10 ? 'eager' : 'lazy';
    image.referrerPolicy = 'no-referrer';

    const label = document.createElement('span');
    label.textContent = suggestion.label;
    button.append(image, label);
    button.addEventListener('click', () => selectSuggestion(suggestion));
    fragment.appendChild(button);
  });
  suggestionList.replaceChildren(fragment);
}

function clearSuggestions(message = '') {
  suggestions = [];
  renderSuggestions();
  suggestionState.textContent = message;
}

async function recognizeSketch(version) {
  if (!suggestionSource || drawing.draftStrokes.length === 0) return;
  suggestionState.textContent = 'Recognizing';
  try {
    const found = await suggestionSource.suggest(draftPolylines(drawing), {
      limit: SUGGESTION_LIMIT,
    });
    if (version !== recognitionVersion) return;
    suggestions = found;
    suggestionState.textContent = suggestions.length > 0 ? 'All icons approved' : 'No match';
    renderSuggestions();
  } catch (error) {
    if (version !== recognitionVersion) return;
    clearSuggestions('Recognition failed');
    toast(error.message || String(error));
  }
}

function queueRecognition() {
  recognitionVersion += 1;
  const version = recognitionVersion;
  clearTimeout(recognitionTimer);
  if (drawing.draftStrokes.length === 0) {
    clearSuggestions(suggestionSource ? 'Ready' : 'Model loading');
    return;
  }
  if (!suggestionSource) {
    suggestionState.textContent = 'Model loading';
    return;
  }
  clearSuggestions('Waiting');
  recognitionTimer = setTimeout(() => recognizeSketch(version), RECOGNITION_DELAY);
}

function selectSuggestion(suggestion) {
  const bounds = stage.getBoundingClientRect();
  try {
    drawing = placeSuggestion(drawing, suggestion, {
      stageWidth: bounds.width,
      stageHeight: bounds.height,
    });
    recognitionVersion += 1;
    clearTimeout(recognitionTimer);
    clearSuggestions('Ready');
    renderScene();
  } catch (error) {
    toast(error.message || String(error));
  }
}

function commitActiveStroke() {
  if (!activeStroke) return;
  drawing = commitStroke(drawing, activeStroke);
  activeStroke = undefined;
  activePointerId = undefined;
  renderScene();
  queueRecognition();
}

canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0 || activeStroke) return;
  event.preventDefault();
  activePointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  activeStroke = { points: [stagePoint(event)], color: strokeColor, width: strokeWidth };
  renderCanvas();
});

canvas.addEventListener('pointermove', event => {
  if (!activeStroke || event.pointerId !== activePointerId) return;
  event.preventDefault();
  const events = event.getCoalescedEvents?.() ?? [event];
  events.forEach(item => appendPoint(activeStroke.points, stagePoint(item)));
  renderCanvas();
});

canvas.addEventListener('pointerup', event => {
  if (!activeStroke || event.pointerId !== activePointerId) return;
  event.preventDefault();
  appendPoint(activeStroke.points, stagePoint(event));
  commitActiveStroke();
});

canvas.addEventListener('pointercancel', event => {
  if (event.pointerId !== activePointerId) return;
  activeStroke = undefined;
  activePointerId = undefined;
  renderCanvas();
});

undoButton.addEventListener('click', () => {
  drawing = undoDrawing(drawing);
  renderScene();
  queueRecognition();
});

redoButton.addEventListener('click', () => {
  drawing = redoDrawing(drawing);
  renderScene();
  queueRecognition();
});

clearButton.addEventListener('click', () => {
  drawing = clearDrawing(drawing);
  renderScene();
  queueRecognition();
});

document.querySelectorAll('.swatch').forEach(button => {
  button.addEventListener('click', () => {
    strokeColor = button.dataset.color;
    document.querySelectorAll('.swatch').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
  });
});

strokeWidthInput.addEventListener('input', () => {
  strokeWidth = Number(strokeWidthInput.value);
});

function blobDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('load', () => resolve(reader.result), { once: true });
    reader.addEventListener('error', () => reject(reader.error), { once: true });
    reader.readAsDataURL(blob);
  });
}

async function fetchIcon(icon) {
  const { primary, fallback, primaryIsLocal } = iconSources(icon);
  try {
    const response = await fetch(primary, { cache: 'no-store' });
    if (response.ok) return response;
  } catch {}
  preferLocalIcons = !primaryIsLocal;
  return fetch(fallback, { cache: 'no-store' });
}

async function embeddedIconData(state) {
  const iconDataByPath = new Map();
  for (const icon of state.icons) {
    if (iconDataByPath.has(icon.path)) continue;
    const response = await fetchIcon(icon);
    if (!response.ok) throw new Error(`Could not embed ${icon.label}: HTTP ${response.status}`);
    iconDataByPath.set(icon.path, await blobDataUrl(await response.blob()));
  }
  return iconDataByPath;
}

downloadButton.addEventListener('click', async () => {
  const exportState = drawing;
  const bounds = stage.getBoundingClientRect();
  downloadButton.disabled = true;
  try {
    const iconDataByPath = await embeddedIconData(exportState);
    const svg = createDrawingSvg({
      state: exportState,
      width: bounds.width,
      height: bounds.height,
      iconDataByPath,
    });
    const blobUrl = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = 'autodraw.svg';
    link.click();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 0);
    toast('SVG downloaded');
  } catch (error) {
    toast(error.message || String(error));
  } finally {
    renderControls();
  }
});

document.addEventListener('keydown', event => {
  if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'z') return;
  event.preventDefault();
  drawing = event.shiftKey ? redoDrawing(drawing) : undoDrawing(drawing);
  renderScene();
  queueRecognition();
});

function resizeActiveStroke(scaleX, scaleY) {
  if (!activeStroke) return;
  activeStroke = {
    ...activeStroke,
    points: activeStroke.points.map(point => [point[0] * scaleX, point[1] * scaleY]),
    width: activeStroke.width * Math.sqrt(scaleX * scaleY),
  };
}

function handleStageResize() {
  const bounds = stage.getBoundingClientRect();
  const nextSize = { width: bounds.width, height: bounds.height };
  if (!stageSize) {
    stageSize = nextSize;
    renderScene();
    return;
  }
  if (
    Math.abs(stageSize.width - nextSize.width) < 0.5
    && Math.abs(stageSize.height - nextSize.height) < 0.5
  ) return;

  resizeActiveStroke(
    nextSize.width / stageSize.width,
    nextSize.height / stageSize.height,
  );
  drawing = resizeDrawing(drawing, {
    fromWidth: stageSize.width,
    fromHeight: stageSize.height,
    toWidth: nextSize.width,
    toHeight: nextSize.height,
  });
  stageSize = nextSize;
  renderScene();
  queueRecognition();
}

if ('ResizeObserver' in window) new ResizeObserver(handleStageResize).observe(stage);
window.addEventListener('resize', handleStageResize);
window.addEventListener('beforeunload', () => activeModel?.dispose());

// Retrieval reaches every icon in the pinned corpus; the candidate manifest only covers the
// classifier's classes, so it is kept as a fallback rather than as the primary path.
async function startRetrieval() {
  let embedder;
  try {
    ({ embedder } = await loadSketchEmbedder({
      runtime: ort,
      metadataUrl: EMBEDDER_METADATA_URL,
      modelUrl: EMBEDDER_MODEL_URL,
    }));
    const index = await loadIconRetrievalIndex({ indexUrl: ICON_INDEX_URL, embedder });
    const source = createRetrievalSuggestionSource({ embedder, index });
    activeModel = embedder;
    return source;
  } catch (error) {
    await embedder?.dispose();
    throw error;
  }
}

async function startCandidates() {
  let classifier;
  try {
    const response = await fetch(CANDIDATES_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Candidate request failed: ${response.status}`);
    const manifest = await response.json();
    classifier = await loadQuickDrawClassifier({
      runtime: ort,
      metadataUrl: MODEL_METADATA_URL,
      modelUrl: MODEL_URL,
    });
    const source = createCandidateSuggestionSource({
      classifier,
      index: createApprovedCandidateIndex(manifest, classifier.classes),
    });
    activeModel = classifier;
    return source;
  } catch (error) {
    await classifier?.dispose();
    throw error;
  }
}

// A cold index scores only the probed clusters, so the rest of the corpus is fetched
// after first paint; each loaded shard widens the search on the next stroke.
function warmCorpus(source) {
  if (typeof source.warm !== 'function') return;
  const start = () => {
    source.warm({ concurrency: 4 }).then(coverage => {
      if (coverage.failed > 0 && coverage.vectors === 0) {
        toast('Icon corpus could not be warmed; suggestions stay limited to probed clusters');
      }
    }).catch(() => {});
  };
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(start, { timeout: 3000 });
  } else {
    setTimeout(start, 1500);
  }
}

async function initialize() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = '/vendor/';
  let source;
  try {
    source = await startRetrieval();
  } catch (retrievalError) {
    try {
      source = await startCandidates();
      toast(`Retrieval unavailable, using approved candidates: ${retrievalError.message || retrievalError}`);
    } catch (fallbackError) {
      setRuntimeStatus('Unavailable', 'error');
      clearSuggestions('Unavailable');
      toast(fallbackError.message || String(fallbackError));
      return;
    }
  }

  suggestionSource = source;
  setRuntimeStatus('Ready', 'ready');
  suggestionState.textContent = 'Ready';
  queueRecognition();
  warmCorpus(source);
}

handleStageResize();
initialize();