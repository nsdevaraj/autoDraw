import assert from 'node:assert/strict';
import test from 'node:test';

import { SVG_LIMITS, svgToPolylines } from '../src/svg-to-polylines.mjs';

function wrap(body, attributes = '') {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"${attributes}>${body}</svg>`;
}

function bounds(polylines) {
  const points = polylines.flat();
  if (points.length === 0) return null;
  return {
    left: Math.min(...points.map(point => point[0])),
    top: Math.min(...points.map(point => point[1])),
    right: Math.max(...points.map(point => point[0])),
    bottom: Math.max(...points.map(point => point[1])),
  };
}

function closeTo(actual, expected, tolerance = 1e-6) {
  assert.equal(
    Math.abs(actual - expected) <= tolerance,
    true,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

test('straight path commands produce absolute polylines', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<path d="M 10 10 H 30 V 20 L 10 20 Z"/>')),
    [[[10, 10], [30, 10], [30, 20], [10, 20], [10, 10]]],
  );
});

test('relative commands accumulate and implicit repeats become line commands', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<path d="m 5 5 l 10 0 10 0"/>')),
    [[[5, 5], [15, 5], [25, 5]]],
  );
});

test('each move command starts a separate polyline', () => {
  const polylines = svgToPolylines(wrap('<path d="M0 0 L10 0 M20 20 L30 20"/>'));
  assert.equal(polylines.length, 2);
  assert.deepEqual(polylines[0], [[0, 0], [10, 0]]);
  assert.deepEqual(polylines[1], [[20, 20], [30, 20]]);
});

test('curves flatten to polylines that stay within their control hull', () => {
  const polylines = svgToPolylines(wrap('<path d="M0 0 C0 50 100 50 100 0"/>'));
  assert.equal(polylines.length, 1);
  assert.equal(polylines[0].length > 4, true);
  assert.deepEqual(polylines[0][0], [0, 0]);
  assert.deepEqual(polylines[0].at(-1), [100, 0]);

  const box = bounds(polylines);
  assert.equal(box.top >= 0 && box.bottom <= 50, true);
  assert.equal(box.left >= 0 && box.right <= 100, true);
});

test('smooth cubic commands reflect the previous control point', () => {
  const reflected = svgToPolylines(wrap('<path d="M0 0 C0 20 20 20 20 0 S40 -20 40 0"/>'));
  const explicit = svgToPolylines(wrap('<path d="M0 0 C0 20 20 20 20 0 C20 -20 40 -20 40 0"/>'));
  assert.deepEqual(reflected, explicit);
});

test('quadratic commands match their equivalent cubic form', () => {
  const quadratic = svgToPolylines(wrap('<path d="M0 0 Q50 50 100 0"/>'));
  const cubic = svgToPolylines(wrap(
    '<path d="M0 0 C33.333333333333336 33.333333333333336 66.66666666666667 33.333333333333336 100 0"/>',
  ));
  assert.equal(quadratic.length, 1);
  assert.equal(quadratic[0].length, cubic[0].length);
  quadratic[0].forEach((point, index) => {
    closeTo(point[0], cubic[0][index][0], 1e-9);
    closeTo(point[1], cubic[0][index][1], 1e-9);
  });
});

test('arcs sweep through the expected quadrant', () => {
  const polylines = svgToPolylines(wrap('<path d="M50 0 A50 50 0 0 1 100 50"/>'));
  assert.equal(polylines.length, 1);
  closeTo(polylines[0][0][0], 50);
  closeTo(polylines[0][0][1], 0);
  closeTo(polylines[0].at(-1)[0], 100);
  closeTo(polylines[0].at(-1)[1], 50);

  const box = bounds(polylines);
  assert.equal(box.left >= 50 - 1e-6 && box.right <= 100 + 1e-6, true);
  assert.equal(box.top >= -1e-6 && box.bottom <= 50 + 1e-6, true);
});

test('degenerate arc radii collapse to a straight line', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<path d="M0 0 A0 0 0 0 1 40 0"/>')),
    [[[0, 0], [40, 0]]],
  );
});

test('basic shapes convert to closed outlines', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<rect x="10" y="20" width="30" height="40"/>')),
    [[[10, 20], [40, 20], [40, 60], [10, 60], [10, 20]]],
  );
  assert.deepEqual(
    svgToPolylines(wrap('<polygon points="0,0 10,0 10,10"/>')),
    [[[0, 0], [10, 0], [10, 10], [0, 0]]],
  );
  assert.deepEqual(
    svgToPolylines(wrap('<polyline points="0,0 10,0 10,10"/>')),
    [[[0, 0], [10, 0], [10, 10]]],
  );
  assert.deepEqual(
    svgToPolylines(wrap('<line x1="1" y1="2" x2="3" y2="4"/>')),
    [[[1, 2], [3, 4]]],
  );
});

test('circles and ellipses approximate their radii', () => {
  const box = bounds(svgToPolylines(wrap('<circle cx="50" cy="50" r="25"/>')));
  closeTo(box.left, 25, 0.05);
  closeTo(box.right, 75, 0.05);
  closeTo(box.top, 25, 0.05);
  closeTo(box.bottom, 75, 0.05);

  const ellipse = bounds(svgToPolylines(wrap('<ellipse cx="50" cy="50" rx="40" ry="10"/>')));
  closeTo(ellipse.left, 10, 0.05);
  closeTo(ellipse.right, 90, 0.05);
  closeTo(ellipse.top, 40, 0.05);
  closeTo(ellipse.bottom, 60, 0.05);
});

test('rounded rectangles stay inside the plain rectangle bounds', () => {
  const box = bounds(svgToPolylines(wrap('<rect x="0" y="0" width="80" height="40" rx="10"/>')));
  closeTo(box.left, 0, 1e-6);
  closeTo(box.top, 0, 1e-6);
  closeTo(box.right, 80, 1e-6);
  closeTo(box.bottom, 40, 1e-6);
});

test('nested transforms compose from the outside in', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<g transform="translate(10 20)"><path d="M0 0 L5 0" transform="scale(2)"/></g>')),
    [[[10, 20], [20, 20]]],
  );
  assert.deepEqual(
    svgToPolylines(wrap('<path d="M0 0 L10 0" transform="matrix(0 1 -1 0 0 0)"/>')),
    [[[0, 0], [0, 10]]],
  );
});

test('rotation about a point transforms around that point', () => {
  const polylines = svgToPolylines(wrap('<path d="M10 0 L20 0" transform="rotate(90 10 0)"/>'));
  closeTo(polylines[0][0][0], 10);
  closeTo(polylines[0][0][1], 0);
  closeTo(polylines[0][1][0], 10);
  closeTo(polylines[0][1][1], 10);
});

test('invisible and non-rendered content contributes no geometry', () => {
  assert.deepEqual(svgToPolylines(wrap('<rect width="100" height="100" fill="none"/>')), []);
  assert.deepEqual(
    svgToPolylines(wrap('<rect width="100" height="100" style="fill:none;stroke:none"/>')),
    [],
  );
  assert.deepEqual(svgToPolylines(wrap('<path d="M0 0 L10 0" display="none"/>')), []);
  assert.deepEqual(svgToPolylines(wrap('<path d="M0 0 L10 0" opacity="0"/>')), []);
  assert.deepEqual(svgToPolylines(wrap('<g visibility="hidden"><path d="M0 0 L10 0"/></g>')), []);
  assert.deepEqual(svgToPolylines(wrap('<defs><path d="M0 0 L10 0"/></defs>')), []);
  assert.deepEqual(svgToPolylines(wrap('<text x="0" y="0">hello</text>')), []);
});

test('an unpainted rectangle does not suppress its painted siblings', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<rect width="100" height="100" fill="none"/><path d="M0 0 L10 0"/>')),
    [[[0, 0], [10, 0]]],
  );
});

test('stroked outline icons survive the unpainted filter', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<path d="M0 0 L10 0" fill="none" stroke="currentColor"/>')),
    [[[0, 0], [10, 0]]],
  );
});

test('paint inherited from a group hides its children', () => {
  assert.deepEqual(svgToPolylines(wrap('<g fill="none"><rect width="10" height="10"/></g>')), []);
});

test('namespaced element names are recognized', () => {
  assert.deepEqual(
    svgToPolylines('<svg:svg xmlns:svg="http://www.w3.org/2000/svg"><svg:path d="M0 0 L10 0"/></svg:svg>'),
    [[[0, 0], [10, 0]]],
  );
});

test('comments, declarations, and CDATA are ignored', () => {
  assert.deepEqual(
    svgToPolylines(
      '<?xml version="1.0"?><svg><!-- <path d="M0 0 L99 0"/> -->'
      + '<style><![CDATA[ path { fill: none } ]]></style><path d="M0 0 L10 0"/></svg>',
    ),
    [[[0, 0], [10, 0]]],
  );
});

test('greater-than characters inside attribute values do not end the tag', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<path data-note="a > b" d="M0 0 L10 0"/>')),
    [[[0, 0], [10, 0]]],
  );
});

test('entity references in attribute values are decoded', () => {
  assert.deepEqual(
    svgToPolylines(wrap('<path d="M0 0 L10 0" data-name="a &amp; b"/>')),
    [[[0, 0], [10, 0]]],
  );
});

test('an external document type declaration is skipped', () => {
  assert.deepEqual(
    svgToPolylines(
      '<?xml version="1.0" encoding="utf-8"?>'
      + '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
      + '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
      + '<svg viewBox="0 0 100 100"><path d="M0 0 L10 0"/></svg>',
    ),
    [[[0, 0], [10, 0]]],
  );
});

test('entity declarations and internal subsets are rejected', () => {
  assert.throws(
    () => svgToPolylines('<!DOCTYPE svg [<!ENTITY a "b">]><svg><path d="M0 0 L1 0"/></svg>'),
    /internal subsets are not supported/,
  );
  assert.throws(
    () => svgToPolylines('<!ENTITY a "b"><svg><path d="M0 0 L1 0"/></svg>'),
    /entity declarations are not supported/,
  );
  assert.throws(
    () => svgToPolylines('<!DOCTYPE svg PUBLIC "unterminated'),
    /unterminated/,
  );
});

test('external and scripted references are never resolved', () => {
  assert.deepEqual(svgToPolylines(wrap('<use href="#missing"/>')), []);
  assert.deepEqual(svgToPolylines(wrap('<image href="https://example.com/a.png"/>')), []);
  assert.deepEqual(svgToPolylines(wrap('<script>fetch("https://example.com")</script>')), []);
  assert.deepEqual(svgToPolylines(wrap('<foreignObject><div/></foreignObject>')), []);
});

test('oversized sources are refused before parsing', () => {
  assert.throws(
    () => svgToPolylines('x'.repeat(SVG_LIMITS.maxCharacters + 1)),
    /exceeds the maximum supported size/,
  );
});

test('element, segment, and vertex budgets bound the work per icon', () => {
  const many = wrap('<path d="M0 0 L1 0"/>'.repeat(20));
  assert.throws(() => svgToPolylines(many, { maxElements: 8 }), /element budget/);
  assert.throws(
    () => svgToPolylines(wrap('<path d="M0 0 L1 0 L2 0 L3 0 L4 0"/>'), { maxSegments: 2 }),
    /segment budget/,
  );
  assert.throws(
    () => svgToPolylines(wrap('<path d="M0 0 C0 50 100 50 100 0"/>'), {
      maxVertices: 2,
      flattenResolution: 4096,
    }),
    /vertex budget/,
  );
});

test('malformed path data and unsupported commands are rejected', () => {
  assert.throws(() => svgToPolylines(wrap('<path d="L10 10"/>')), /without a move command/);
  assert.throws(() => svgToPolylines(wrap('<path d="10 10"/>')), /must start with a command/);
  assert.throws(() => svgToPolylines(wrap('<path d="M0 0 L abc"/>')), /invalid number/);
  assert.throws(() => svgToPolylines(wrap('<path d="M0 0 A5 5 0 2 1 10 10"/>')), /arc flag must be 0 or 1/);
  assert.throws(() => svgToPolylines(wrap('<path d="M0 0 X10 10"/>')), /Unsupported SVG path command/);
});

test('empty and geometry-free documents produce no polylines', () => {
  assert.deepEqual(svgToPolylines(wrap('')), []);
  assert.deepEqual(svgToPolylines(wrap('<rect width="0" height="10"/>')), []);
  assert.deepEqual(svgToPolylines(wrap('<circle cx="5" cy="5" r="0"/>')), []);
  assert.deepEqual(svgToPolylines(wrap('<polyline points="1,1"/>')), []);
});

test('percentage lengths are skipped rather than guessed', () => {
  assert.deepEqual(svgToPolylines(wrap('<rect width="100%" height="100%"/>')), []);
});

test('flatten resolution controls how finely curves are subdivided', () => {
  const coarse = svgToPolylines(wrap('<path d="M0 0 C0 50 100 50 100 0"/>'), { flattenResolution: 8 });
  const fine = svgToPolylines(wrap('<path d="M0 0 C0 50 100 50 100 0"/>'), { flattenResolution: 1024 });
  assert.equal(fine[0].length > coarse[0].length, true);
});

test('non-string sources and invalid limits are rejected', () => {
  assert.throws(() => svgToPolylines(null), /must be a string/);
  assert.throws(() => svgToPolylines(wrap(''), { maxElements: 0 }), /positive integer/);
  assert.throws(() => svgToPolylines(wrap(''), { flattenResolution: 4 }), /at least 8/);
});
