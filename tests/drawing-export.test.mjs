import assert from 'node:assert/strict';
import test from 'node:test';

import { createDrawingSvg } from '../src/drawing-export.mjs';
import {
  commitStroke,
  createDrawingState,
  placeSuggestion,
} from '../src/drawing-state.mjs';

const suggestion = {
  id: 10,
  label: 'cat',
  name: 'cat-icon',
  path: 'Animals/line/cat.svg',
  url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/cat.svg',
};

test('SVG export preserves taps, smoothed strokes, and embedded icons', () => {
  const tapped = commitStroke(createDrawingState(), {
    points: [[12, 18]],
    color: '#17212b',
    width: 6,
  });
  const stroked = commitStroke(tapped, {
    points: [[20, 30], [40, 50], [60, 30]],
    color: '#007f7a',
    width: 4,
  });
  const placed = placeSuggestion(stroked, suggestion, {
    stageWidth: 200,
    stageHeight: 100,
  });
  const restored = { ...placed, draftStrokes: stroked.draftStrokes };
  const dataUrl = 'data:image/svg+xml;base64,PHN2Zy8+';
  const svg = createDrawingSvg({
    state: restored,
    width: 200,
    height: 100,
    iconDataByPath: new Map([[suggestion.path, dataUrl]]),
  });

  assert.match(svg, /<circle[^>]+cx="12"[^>]+cy="18"[^>]+r="3"/);
  assert.match(svg, /<path[^>]+d="M 20 30 Q 20 30 30 40 Q 40 50 50 40 L 60 30"/);
  assert.match(svg, /href="data:image\/svg\+xml;base64,PHN2Zy8\+"/);
  assert.doesNotMatch(svg, /https:\/\//);
  assert.doesNotMatch(svg, /<polyline/);
});

test('SVG export rejects missing icon data and invalid dimensions', () => {
  const state = placeSuggestion(
    commitStroke(createDrawingState(), {
      points: [[10, 10]],
      color: '#17212b',
      width: 4,
    }),
    suggestion,
    { stageWidth: 100, stageHeight: 100 },
  );
  assert.throws(
    () => createDrawingSvg({ state, width: 100, height: 100, iconDataByPath: new Map() }),
    /embedded icon data/,
  );
  assert.throws(
    () => createDrawingSvg({ state, width: 0, height: 100, iconDataByPath: new Map() }),
    /dimensions/,
  );
});