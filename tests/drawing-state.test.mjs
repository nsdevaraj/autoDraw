import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clearDrawing,
  commitStroke,
  createDrawingState,
  draftPolylines,
  placeSuggestion,
  redoDrawing,
  resizeDrawing,
  undoDrawing,
} from '../src/drawing-state.mjs';

const catSuggestion = {
  id: 10,
  label: 'cat',
  name: 'cat-icon',
  path: 'Animals/line/cat.svg',
  url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/cat.svg',
};

test('stroke commits support undo and redo without mutating prior state', () => {
  const empty = createDrawingState();
  const first = commitStroke(empty, {
    points: [[10, 10], [30, 20]],
    color: '#17212b',
    width: 4,
  });
  const second = commitStroke(first, {
    points: [[40, 20], [50, 40]],
    color: '#d14c3f',
    width: 6,
  });

  assert.deepEqual(empty.draftStrokes, []);
  assert.deepEqual(draftPolylines(second), [
    [[10, 10], [30, 20]],
    [[40, 20], [50, 40]],
  ]);
  const undone = undoDrawing(second);
  assert.deepEqual(draftPolylines(undone), [[[10, 10], [30, 20]]]);
  assert.deepEqual(draftPolylines(redoDrawing(undone)), draftPolylines(second));
  assert.equal(first.icons, second.icons);
});

test('resize scales the current scene and all history without adding an undo step', () => {
  const first = commitStroke(createDrawingState(), {
    points: [[10, 20], [30, 40]],
    color: '#17212b',
    width: 4,
  });
  const placed = placeSuggestion(first, catSuggestion, {
    stageWidth: 100,
    stageHeight: 100,
  });
  const resized = resizeDrawing(placed, {
    fromWidth: 100,
    fromHeight: 100,
    toWidth: 200,
    toHeight: 50,
  });

  assert.equal(resized.past.length, placed.past.length);
  assert.deepEqual(resized.icons[0], {
    ...placed.icons[0],
    x: placed.icons[0].x * 2,
    y: placed.icons[0].y / 2,
    width: placed.icons[0].width * 2,
    height: placed.icons[0].height / 2,
  });
  assert.deepEqual(draftPolylines(undoDrawing(resized)), [
    [[20, 10], [60, 20]],
  ]);
  assert.equal(
    resized.past[1].draftStrokes[0],
    undoDrawing(resized).draftStrokes[0],
  );
  assert.throws(
    () => resizeDrawing(placed, { fromWidth: 0, fromHeight: 100, toWidth: 200, toHeight: 50 }),
    /dimensions/,
  );
});

test('choosing a suggestion replaces the active sketch and can be undone', () => {
  const sketched = commitStroke(createDrawingState(), {
    points: [[80, 60], [180, 60], [180, 140]],
    color: '#17212b',
    width: 5,
  });
  const placed = placeSuggestion(sketched, catSuggestion, {
    stageWidth: 600,
    stageHeight: 400,
  });

  assert.deepEqual(placed.draftStrokes, []);
  assert.equal(placed.icons.length, 1);
  assert.deepEqual(placed.icons[0], {
    id: 10,
    label: 'cat',
    name: 'cat-icon',
    path: 'Animals/line/cat.svg',
    url: 'https://cdn.jsdelivr.net/gh/example/icons@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/Animals/line/cat.svg',
    x: 68,
    y: 48,
    width: 124,
    height: 104,
  });
  assert.deepEqual(undoDrawing(placed).draftStrokes, sketched.draftStrokes);
  assert.deepEqual(redoDrawing(undoDrawing(placed)).icons, placed.icons);
});

test('clear removes draft strokes and placed icons as one undoable action', () => {
  const withStroke = commitStroke(createDrawingState(), {
    points: [[1, 1]],
    color: '#17212b',
    width: 3,
  });
  const withIcon = placeSuggestion(withStroke, catSuggestion, {
    stageWidth: 200,
    stageHeight: 200,
  });
  const cleared = clearDrawing(withIcon);

  assert.deepEqual(cleared.draftStrokes, []);
  assert.deepEqual(cleared.icons, []);
  assert.deepEqual(undoDrawing(cleared).icons, withIcon.icons);
});

test('drawing state validates strokes, suggestions, and stage bounds', () => {
  assert.throws(
    () => commitStroke(createDrawingState(), { points: [[Number.NaN, 0]], color: '#000', width: 2 }),
    /finite/,
  );
  assert.throws(
    () => placeSuggestion(createDrawingState(), catSuggestion, { stageWidth: 0, stageHeight: 100 }),
    /stage/i,
  );
  assert.throws(
    () => placeSuggestion(createDrawingState(), { ...catSuggestion, url: 'javascript:x' }, { stageWidth: 100, stageHeight: 100 }),
    /commit-pinned jsDelivr URL/,
  );
});