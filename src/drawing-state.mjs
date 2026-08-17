import { assertApprovedIconSource } from './icon-candidate.mjs';

const ICON_PADDING = 12;
const MINIMUM_ICON_SIZE = 96;

function finitePoint(point) {
  return Array.isArray(point)
    && point.length === 2
    && point.every(value => typeof value === 'number' && Number.isFinite(value));
}

function sceneSnapshot(state) {
  return Object.freeze({
    draftStrokes: state.draftStrokes,
    icons: state.icons,
  });
}

function updatedState(state, scene) {
  return Object.freeze({
    draftStrokes: Object.freeze(scene.draftStrokes),
    icons: Object.freeze(scene.icons),
    past: Object.freeze([...state.past, sceneSnapshot(state)]),
    future: Object.freeze([]),
  });
}

function restoredState(scene, past, future) {
  return Object.freeze({
    draftStrokes: scene.draftStrokes,
    icons: scene.icons,
    past: Object.freeze(past),
    future: Object.freeze(future),
  });
}

export function createDrawingState() {
  return Object.freeze({
    draftStrokes: Object.freeze([]),
    icons: Object.freeze([]),
    past: Object.freeze([]),
    future: Object.freeze([]),
  });
}

export function commitStroke(state, { points, color, width }) {
  if (!Array.isArray(points) || points.length === 0 || !points.every(finitePoint)) {
    throw new Error('Stroke points must contain finite x and y coordinates');
  }
  if (typeof color !== 'string' || color.length === 0) {
    throw new Error('Stroke color must be a non-empty string');
  }
  if (!Number.isFinite(width) || width <= 0) throw new Error('Stroke width must be positive');

  const stroke = Object.freeze({
    points: Object.freeze(points.map(point => Object.freeze([...point]))),
    color,
    width,
  });
  return updatedState(state, {
    draftStrokes: [...state.draftStrokes, stroke],
    icons: state.icons,
  });
}

function fittedAxis(minimum, maximum, stageSize) {
  const center = (minimum + maximum) / 2;
  const size = Math.min(
    stageSize,
    Math.max(maximum - minimum + ICON_PADDING * 2, MINIMUM_ICON_SIZE),
  );
  return {
    start: Math.max(0, Math.min(stageSize - size, center - size / 2)),
    size,
  };
}

export function placeSuggestion(state, suggestion, { stageWidth, stageHeight }) {
  if (!Number.isFinite(stageWidth) || stageWidth <= 0 || !Number.isFinite(stageHeight) || stageHeight <= 0) {
    throw new Error('Drawing stage dimensions must be positive');
  }
  if (
    !suggestion
    || !Number.isInteger(suggestion.id)
    || typeof suggestion.label !== 'string'
    || typeof suggestion.name !== 'string'
  ) {
    throw new Error('Suggestion must contain icon metadata');
  }
  assertApprovedIconSource(suggestion, 'Suggestion');
  if (state.draftStrokes.length === 0) throw new Error('Cannot place a suggestion without an active sketch');

  const points = state.draftStrokes.flatMap(stroke => stroke.points);
  const xAxis = fittedAxis(
    Math.min(...points.map(point => point[0])),
    Math.max(...points.map(point => point[0])),
    stageWidth,
  );
  const yAxis = fittedAxis(
    Math.min(...points.map(point => point[1])),
    Math.max(...points.map(point => point[1])),
    stageHeight,
  );
  const icon = Object.freeze({
    id: suggestion.id,
    label: suggestion.label,
    name: suggestion.name,
    path: suggestion.path,
    url: suggestion.url,
    x: xAxis.start,
    y: yAxis.start,
    width: xAxis.size,
    height: yAxis.size,
  });
  return updatedState(state, {
    draftStrokes: [],
    icons: [...state.icons, icon],
  });
}

export function clearDrawing(state) {
  if (state.draftStrokes.length === 0 && state.icons.length === 0) return state;
  return updatedState(state, { draftStrokes: [], icons: [] });
}

export function undoDrawing(state) {
  if (state.past.length === 0) return state;
  const previous = state.past[state.past.length - 1];
  return restoredState(
    previous,
    state.past.slice(0, -1),
    [sceneSnapshot(state), ...state.future],
  );
}

export function redoDrawing(state) {
  if (state.future.length === 0) return state;
  const [next, ...future] = state.future;
  return restoredState(
    next,
    [...state.past, sceneSnapshot(state)],
    future,
  );
}

export function draftPolylines(state) {
  return state.draftStrokes.map(stroke => stroke.points.map(point => [...point]));
}

function positiveDimensions({ fromWidth, fromHeight, toWidth, toHeight }) {
  return [fromWidth, fromHeight, toWidth, toHeight]
    .every(value => typeof value === 'number' && Number.isFinite(value) && value > 0);
}

function scaledScene(scene, scaleX, scaleY, scaledStrokes, scaledIcons) {
  const lineScale = Math.sqrt(scaleX * scaleY);
  return Object.freeze({
    draftStrokes: Object.freeze(scene.draftStrokes.map(stroke => {
      if (!scaledStrokes.has(stroke)) {
        scaledStrokes.set(stroke, Object.freeze({
          ...stroke,
          points: Object.freeze(stroke.points.map(point => Object.freeze([
            point[0] * scaleX,
            point[1] * scaleY,
          ]))),
          width: stroke.width * lineScale,
        }));
      }
      return scaledStrokes.get(stroke);
    })),
    icons: Object.freeze(scene.icons.map(icon => {
      if (!scaledIcons.has(icon)) {
        scaledIcons.set(icon, Object.freeze({
          ...icon,
          x: icon.x * scaleX,
          y: icon.y * scaleY,
          width: icon.width * scaleX,
          height: icon.height * scaleY,
        }));
      }
      return scaledIcons.get(icon);
    })),
  });
}

export function resizeDrawing(state, dimensions) {
  if (!positiveDimensions(dimensions)) {
    throw new Error('Drawing resize dimensions must be positive finite numbers');
  }
  const scaleX = dimensions.toWidth / dimensions.fromWidth;
  const scaleY = dimensions.toHeight / dimensions.fromHeight;
  const scaledStrokes = new WeakMap();
  const scaledIcons = new WeakMap();
  const scale = scene => scaledScene(scene, scaleX, scaleY, scaledStrokes, scaledIcons);
  const current = scale(state);
  return Object.freeze({
    ...current,
    past: Object.freeze(state.past.map(scale)),
    future: Object.freeze(state.future.map(scale)),
  });
}