// Converts SVGDepot icons into the polyline form the sketch rasterizer already consumes.
// Only geometry is read: paint is used solely to drop invisible shapes, so filled
// silhouettes arrive as outlines and land in the same domain as hand-drawn strokes.

const DEFAULT_LIMITS = Object.freeze({
  maxCharacters: 512 * 1024,
  maxElements: 20000,
  maxSegments: 20000,
  maxVertices: 20000,
  flattenResolution: 256,
});

const KAPPA = 0.5522847498307936;
const IDENTITY = Object.freeze([1, 0, 0, 1, 0, 0]);
const MAX_SUBDIVISION_DEPTH = 16;

// Subtrees that define reusable or non-geometric content, plus everything we refuse
// to resolve (external references, scripts, fonts) for safety.
const SKIPPED_ELEMENTS = new Set([
  'clippath', 'defs', 'desc', 'filter', 'font', 'foreignobject', 'image', 'marker',
  'mask', 'metadata', 'pattern', 'script', 'style', 'switch', 'symbol', 'text',
  'title', 'use',
]);
const SHAPE_ELEMENTS = new Set([
  'circle', 'ellipse', 'line', 'path', 'polygon', 'polyline', 'rect',
]);

const NAMED_ENTITIES = Object.freeze({
  amp: '&', apos: "'", gt: '>', lt: '<', quot: '"',
});

function resolvedLimits(options) {
  const limits = { ...DEFAULT_LIMITS, ...options };
  for (const key of ['maxCharacters', 'maxElements', 'maxSegments', 'maxVertices']) {
    if (!Number.isInteger(limits[key]) || limits[key] < 1) {
      throw new Error(`SVG ${key} limit must be a positive integer`);
    }
  }
  if (!Number.isInteger(limits.flattenResolution) || limits.flattenResolution < 8) {
    throw new Error('SVG flatten resolution must be an integer of at least 8');
  }
  return limits;
}

function decodeEntities(value) {
  if (!value.includes('&')) return value;
  return value.replace(/&(#x[0-9a-fA-F]{1,6}|#[0-9]{1,7}|[a-zA-Z]{2,5});/g, (match, body) => {
    if (body[0] !== '#') return NAMED_ENTITIES[body.toLowerCase()] ?? match;
    const codePoint = body[1] === 'x' || body[1] === 'X'
      ? Number.parseInt(body.slice(2), 16)
      : Number.parseInt(body.slice(1), 10);
    if (!Number.isInteger(codePoint) || codePoint < 1 || codePoint > 0x10ffff) return match;
    return String.fromCodePoint(codePoint);
  });
}

function multiply(left, right) {
  return [
    left[0] * right[0] + left[2] * right[1],
    left[1] * right[0] + left[3] * right[1],
    left[0] * right[2] + left[2] * right[3],
    left[1] * right[2] + left[3] * right[3],
    left[0] * right[4] + left[2] * right[5] + left[4],
    left[1] * right[4] + left[3] * right[5] + left[5],
  ];
}

function parseTransform(value) {
  if (typeof value !== 'string' || value.length === 0) return IDENTITY;
  let matrix = IDENTITY;
  const pattern = /([a-zA-Z]+)\s*\(([^)]*)\)/g;
  let match;
  while ((match = pattern.exec(value)) !== null) {
    const numbers = match[2]
      .split(/[\s,]+/)
      .filter(Boolean)
      .map(Number);
    if (numbers.some(number => !Number.isFinite(number))) {
      throw new Error('SVG transform contains a non-finite number');
    }
    matrix = multiply(matrix, transformMatrix(match[1].toLowerCase(), numbers));
  }
  return matrix;
}

function transformMatrix(name, numbers) {
  const radians = degrees => (degrees * Math.PI) / 180;
  if (name === 'matrix' && numbers.length === 6) return numbers;
  if (name === 'translate' && numbers.length >= 1) return [1, 0, 0, 1, numbers[0], numbers[1] ?? 0];
  if (name === 'scale' && numbers.length >= 1) return [numbers[0], 0, 0, numbers[1] ?? numbers[0], 0, 0];
  if (name === 'skewx' && numbers.length === 1) return [1, 0, Math.tan(radians(numbers[0])), 1, 0, 0];
  if (name === 'skewy' && numbers.length === 1) return [1, Math.tan(radians(numbers[0])), 0, 1, 0, 0];
  if (name === 'rotate' && (numbers.length === 1 || numbers.length === 3)) {
    const angle = radians(numbers[0]);
    const rotation = [Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle), 0, 0];
    if (numbers.length === 1) return rotation;
    return multiply(
      multiply([1, 0, 0, 1, numbers[1], numbers[2]], rotation),
      [1, 0, 0, 1, -numbers[1], -numbers[2]],
    );
  }
  return IDENTITY;
}

function parseStyle(value) {
  const declarations = {};
  if (typeof value !== 'string' || value.length === 0) return declarations;
  for (const declaration of value.split(';')) {
    const separator = declaration.indexOf(':');
    if (separator === -1) continue;
    const property = declaration.slice(0, separator).trim().toLowerCase();
    if (property) declarations[property] = declaration.slice(separator + 1).trim().toLowerCase();
  }
  return declarations;
}

function presentation(attributes, inherited) {
  const style = parseStyle(attributes.style);
  const pick = name => style[name] ?? attributes[name]?.trim().toLowerCase();
  return {
    display: pick('display'),
    fill: pick('fill') ?? inherited.fill,
    opacity: pick('opacity'),
    stroke: pick('stroke') ?? inherited.stroke,
    visibility: pick('visibility'),
  };
}

function isHidden(paint) {
  return paint.display === 'none'
    || paint.visibility === 'hidden'
    || Number.parseFloat(paint.opacity) === 0;
}

// A shape painted with neither fill nor stroke contributes no ink; icon packs use these
// as invisible full-canvas bounding rects that would otherwise dominate the raster.
function isUnpainted(paint) {
  return (paint.fill === 'none' || paint.fill === 'transparent')
    && (paint.stroke === undefined || paint.stroke === 'none' || paint.stroke === 'transparent');
}

// A bare external DOCTYPE is inert because nothing here resolves external DTDs, but an
// internal subset can declare entities, which is the expansion attack this parser refuses.
function skipDoctype(text, start) {
  let quote = null;
  for (let index = start + 9; index < text.length; index += 1) {
    const character = text[index];
    if (quote !== null) {
      if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'") quote = character;
    else if (character === '[') throw new Error('SVG document type internal subsets are not supported');
    else if (character === '>') return index + 1;
  }
  throw new Error('SVG document type declaration is unterminated');
}

function findTagEnd(text, start) {
  let quote = null;
  for (let index = start + 1; index < text.length; index += 1) {
    const character = text[index];
    if (quote !== null) {
      if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'") quote = character;
    else if (character === '>') return index;
  }
  return -1;
}

function parseAttributes(body) {
  const attributes = {};
  const pattern = /([:a-zA-Z_][-:.\w]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  let match;
  while ((match = pattern.exec(body)) !== null) {
    attributes[match[1].toLowerCase()] = decodeEntities(match[2] ?? match[3]);
  }
  return attributes;
}

function length(value, fallback = Number.NaN) {
  if (typeof value !== 'string') return fallback;
  const trimmed = value.trim();
  if (trimmed.endsWith('%')) return Number.NaN;
  const number = Number.parseFloat(trimmed);
  return Number.isFinite(number) ? number : fallback;
}

class ContourBuilder {
  constructor(budget) {
    this.contours = [];
    this.budget = budget;
    this.current = null;
  }

  moveTo(x, y) {
    this.current = { start: [x, y], segments: [], closed: false };
    this.contours.push(this.current);
  }

  addSegment(segment) {
    if (this.current === null) throw new Error('SVG geometry started without a move command');
    this.budget.segments -= 1;
    if (this.budget.segments < 0) throw new Error('SVG geometry exceeds the segment budget');
    this.current.segments.push(segment);
  }

  lineTo(x, y) {
    this.addSegment(['L', x, y]);
  }

  cubicTo(x1, y1, x2, y2, x, y) {
    this.addSegment(['C', x1, y1, x2, y2, x, y]);
  }

  close() {
    if (this.current !== null) this.current.closed = true;
  }
}

function angleBetween(ux, uy, vx, vy) {
  const magnitude = Math.hypot(ux, uy) * Math.hypot(vx, vy);
  if (magnitude === 0) return 0;
  const cosine = Math.min(1, Math.max(-1, (ux * vx + uy * vy) / magnitude));
  return Math.sign(ux * vy - uy * vx) * Math.acos(cosine);
}

function appendArc(builder, x1, y1, rx, ry, rotation, largeArc, sweep, x2, y2) {
  if (rx === 0 || ry === 0 || (x1 === x2 && y1 === y2)) {
    builder.lineTo(x2, y2);
    return;
  }

  let radiusX = Math.abs(rx);
  let radiusY = Math.abs(ry);
  const phi = (rotation * Math.PI) / 180;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);
  const midX = (x1 - x2) / 2;
  const midY = (y1 - y2) / 2;
  const rotatedX = cosPhi * midX + sinPhi * midY;
  const rotatedY = -sinPhi * midX + cosPhi * midY;

  const overshoot = (rotatedX * rotatedX) / (radiusX * radiusX)
    + (rotatedY * rotatedY) / (radiusY * radiusY);
  if (overshoot > 1) {
    const scale = Math.sqrt(overshoot);
    radiusX *= scale;
    radiusY *= scale;
  }

  const denominator = radiusX * radiusX * rotatedY * rotatedY
    + radiusY * radiusY * rotatedX * rotatedX;
  const numerator = radiusX * radiusX * radiusY * radiusY - denominator;
  const coefficient = (largeArc === sweep ? -1 : 1)
    * Math.sqrt(Math.max(0, numerator / denominator));
  const centerX = (coefficient * radiusX * rotatedY) / radiusY;
  const centerY = (-coefficient * radiusY * rotatedX) / radiusX;

  const startUnitX = (rotatedX - centerX) / radiusX;
  const startUnitY = (rotatedY - centerY) / radiusY;
  const endUnitX = (-rotatedX - centerX) / radiusX;
  const endUnitY = (-rotatedY - centerY) / radiusY;
  const startAngle = angleBetween(1, 0, startUnitX, startUnitY);
  let sweepAngle = angleBetween(startUnitX, startUnitY, endUnitX, endUnitY);
  if (!sweep && sweepAngle > 0) sweepAngle -= 2 * Math.PI;
  if (sweep && sweepAngle < 0) sweepAngle += 2 * Math.PI;

  const absoluteCenterX = cosPhi * centerX - sinPhi * centerY + (x1 + x2) / 2;
  const absoluteCenterY = sinPhi * centerX + cosPhi * centerY + (y1 + y2) / 2;
  const segmentCount = Math.max(1, Math.ceil(Math.abs(sweepAngle) / (Math.PI / 2)));
  const segmentAngle = sweepAngle / segmentCount;
  const alpha = (4 / 3) * Math.tan(segmentAngle / 4);

  const pointAt = angle => {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    return [
      absoluteCenterX + radiusX * cosine * cosPhi - radiusY * sine * sinPhi,
      absoluteCenterY + radiusX * cosine * sinPhi + radiusY * sine * cosPhi,
    ];
  };
  const derivativeAt = angle => {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    return [
      -radiusX * sine * cosPhi - radiusY * cosine * sinPhi,
      -radiusX * sine * sinPhi + radiusY * cosine * cosPhi,
    ];
  };

  for (let index = 0; index < segmentCount; index += 1) {
    const from = startAngle + index * segmentAngle;
    const to = from + segmentAngle;
    const [fromX, fromY] = pointAt(from);
    const [toX, toY] = pointAt(to);
    const [fromDx, fromDy] = derivativeAt(from);
    const [toDx, toDy] = derivativeAt(to);
    builder.cubicTo(
      fromX + alpha * fromDx,
      fromY + alpha * fromDy,
      toX - alpha * toDx,
      toY - alpha * toDy,
      toX,
      toY,
    );
  }
}

class PathScanner {
  constructor(data) {
    this.data = data;
    this.index = 0;
  }

  skipSeparators() {
    while (this.index < this.data.length) {
      const code = this.data.charCodeAt(this.index);
      if (code !== 32 && code !== 9 && code !== 10 && code !== 13 && code !== 12 && code !== 44) break;
      this.index += 1;
    }
  }

  hasMore() {
    this.skipSeparators();
    return this.index < this.data.length;
  }

  peek() {
    return this.data[this.index];
  }

  readCommand() {
    const character = this.data[this.index];
    this.index += 1;
    return character;
  }

  readNumber() {
    this.skipSeparators();
    const { data } = this;
    const start = this.index;
    let index = this.index;
    const isDigit = position => {
      const code = data.charCodeAt(position);
      return code >= 48 && code <= 57;
    };

    if (data[index] === '+' || data[index] === '-') index += 1;
    while (index < data.length && isDigit(index)) index += 1;
    if (data[index] === '.') {
      index += 1;
      while (index < data.length && isDigit(index)) index += 1;
    }
    if (data[index] === 'e' || data[index] === 'E') {
      let exponent = index + 1;
      if (data[exponent] === '+' || data[exponent] === '-') exponent += 1;
      if (exponent < data.length && isDigit(exponent)) {
        exponent += 1;
        while (exponent < data.length && isDigit(exponent)) exponent += 1;
        index = exponent;
      }
    }

    const value = Number(data.slice(start, index));
    if (index === start || !Number.isFinite(value)) {
      throw new Error('SVG path data contains an invalid number');
    }
    this.index = index;
    return value;
  }

  readFlag() {
    this.skipSeparators();
    const character = this.data[this.index];
    if (character !== '0' && character !== '1') {
      throw new Error('SVG arc flag must be 0 or 1');
    }
    this.index += 1;
    return character === '1';
  }
}

function appendPathData(builder, data) {
  const scanner = new PathScanner(data);
  let x = 0;
  let y = 0;
  let startX = 0;
  let startY = 0;
  let controlX = 0;
  let controlY = 0;
  let previous = null;
  let previousWasCubic = false;
  let previousWasQuadratic = false;

  const quadraticTo = (qx, qy, toX, toY) => {
    builder.cubicTo(
      x + (2 / 3) * (qx - x),
      y + (2 / 3) * (qy - y),
      toX + (2 / 3) * (qx - toX),
      toY + (2 / 3) * (qy - toY),
      toX,
      toY,
    );
  };

  while (scanner.hasMore()) {
    let command;
    if (/[a-zA-Z]/.test(scanner.peek())) {
      command = scanner.readCommand();
    } else if (previous === null) {
      throw new Error('SVG path data must start with a command');
    } else if (previous === 'M' || previous === 'm') {
      command = previous === 'M' ? 'L' : 'l';
    } else {
      command = previous;
    }

    const relative = command === command.toLowerCase();
    const originX = relative ? x : 0;
    const originY = relative ? y : 0;
    const isCubic = 'CcSs'.includes(command);
    const isQuadratic = 'QqTt'.includes(command);

    switch (command.toUpperCase()) {
      case 'M': {
        x = originX + scanner.readNumber();
        y = originY + scanner.readNumber();
        startX = x;
        startY = y;
        builder.moveTo(x, y);
        break;
      }
      case 'L': {
        x = originX + scanner.readNumber();
        y = originY + scanner.readNumber();
        builder.lineTo(x, y);
        break;
      }
      case 'H': {
        x = originX + scanner.readNumber();
        builder.lineTo(x, y);
        break;
      }
      case 'V': {
        y = originY + scanner.readNumber();
        builder.lineTo(x, y);
        break;
      }
      case 'C': {
        const x1 = originX + scanner.readNumber();
        const y1 = originY + scanner.readNumber();
        const x2 = originX + scanner.readNumber();
        const y2 = originY + scanner.readNumber();
        const toX = originX + scanner.readNumber();
        const toY = originY + scanner.readNumber();
        builder.cubicTo(x1, y1, x2, y2, toX, toY);
        controlX = x2;
        controlY = y2;
        x = toX;
        y = toY;
        break;
      }
      case 'S': {
        const x1 = previousWasCubic ? 2 * x - controlX : x;
        const y1 = previousWasCubic ? 2 * y - controlY : y;
        const x2 = originX + scanner.readNumber();
        const y2 = originY + scanner.readNumber();
        const toX = originX + scanner.readNumber();
        const toY = originY + scanner.readNumber();
        builder.cubicTo(x1, y1, x2, y2, toX, toY);
        controlX = x2;
        controlY = y2;
        x = toX;
        y = toY;
        break;
      }
      case 'Q': {
        const qx = originX + scanner.readNumber();
        const qy = originY + scanner.readNumber();
        const toX = originX + scanner.readNumber();
        const toY = originY + scanner.readNumber();
        quadraticTo(qx, qy, toX, toY);
        controlX = qx;
        controlY = qy;
        x = toX;
        y = toY;
        break;
      }
      case 'T': {
        const qx = previousWasQuadratic ? 2 * x - controlX : x;
        const qy = previousWasQuadratic ? 2 * y - controlY : y;
        const toX = originX + scanner.readNumber();
        const toY = originY + scanner.readNumber();
        quadraticTo(qx, qy, toX, toY);
        controlX = qx;
        controlY = qy;
        x = toX;
        y = toY;
        break;
      }
      case 'A': {
        const radiusX = scanner.readNumber();
        const radiusY = scanner.readNumber();
        const rotation = scanner.readNumber();
        const largeArc = scanner.readFlag();
        const sweep = scanner.readFlag();
        const toX = originX + scanner.readNumber();
        const toY = originY + scanner.readNumber();
        appendArc(builder, x, y, radiusX, radiusY, rotation, largeArc, sweep, toX, toY);
        x = toX;
        y = toY;
        break;
      }
      case 'Z': {
        builder.close();
        x = startX;
        y = startY;
        break;
      }
      default:
        throw new Error(`Unsupported SVG path command: ${command}`);
    }

    previous = command;
    previousWasCubic = isCubic;
    previousWasQuadratic = isQuadratic;
  }
}

function appendRect(builder, attributes) {
  const x = length(attributes.x, 0);
  const y = length(attributes.y, 0);
  const width = length(attributes.width);
  const height = length(attributes.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;

  const rawX = length(attributes.rx);
  const rawY = length(attributes.ry);
  const radiusX = Math.min(Number.isFinite(rawX) ? rawX : (rawY || 0), width / 2);
  const radiusY = Math.min(Number.isFinite(rawY) ? rawY : (rawX || 0), height / 2);

  if (radiusX <= 0 || radiusY <= 0) {
    builder.moveTo(x, y);
    builder.lineTo(x + width, y);
    builder.lineTo(x + width, y + height);
    builder.lineTo(x, y + height);
    builder.close();
    return;
  }

  const offsetX = radiusX * KAPPA;
  const offsetY = radiusY * KAPPA;
  builder.moveTo(x + radiusX, y);
  builder.lineTo(x + width - radiusX, y);
  builder.cubicTo(
    x + width - radiusX + offsetX, y,
    x + width, y + radiusY - offsetY,
    x + width, y + radiusY,
  );
  builder.lineTo(x + width, y + height - radiusY);
  builder.cubicTo(
    x + width, y + height - radiusY + offsetY,
    x + width - radiusX + offsetX, y + height,
    x + width - radiusX, y + height,
  );
  builder.lineTo(x + radiusX, y + height);
  builder.cubicTo(
    x + radiusX - offsetX, y + height,
    x, y + height - radiusY + offsetY,
    x, y + height - radiusY,
  );
  builder.lineTo(x, y + radiusY);
  builder.cubicTo(x, y + radiusY - offsetY, x + radiusX - offsetX, y, x + radiusX, y);
  builder.close();
}

function appendEllipse(builder, centerX, centerY, radiusX, radiusY) {
  if (!(radiusX > 0) || !(radiusY > 0)) return;
  const offsetX = radiusX * KAPPA;
  const offsetY = radiusY * KAPPA;
  builder.moveTo(centerX + radiusX, centerY);
  builder.cubicTo(
    centerX + radiusX, centerY + offsetY,
    centerX + offsetX, centerY + radiusY,
    centerX, centerY + radiusY,
  );
  builder.cubicTo(
    centerX - offsetX, centerY + radiusY,
    centerX - radiusX, centerY + offsetY,
    centerX - radiusX, centerY,
  );
  builder.cubicTo(
    centerX - radiusX, centerY - offsetY,
    centerX - offsetX, centerY - radiusY,
    centerX, centerY - radiusY,
  );
  builder.cubicTo(
    centerX + offsetX, centerY - radiusY,
    centerX + radiusX, centerY - offsetY,
    centerX + radiusX, centerY,
  );
  builder.close();
}

function appendPoints(builder, attributes, closed) {
  const numbers = (attributes.points ?? '')
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);
  if (numbers.length < 4 || numbers.some(number => !Number.isFinite(number))) return;

  builder.moveTo(numbers[0], numbers[1]);
  for (let index = 2; index + 1 < numbers.length; index += 2) {
    builder.lineTo(numbers[index], numbers[index + 1]);
  }
  if (closed) builder.close();
}

function appendShape(builder, name, attributes) {
  if (name === 'path') {
    if (typeof attributes.d === 'string' && attributes.d.trim()) appendPathData(builder, attributes.d);
    return;
  }
  if (name === 'rect') {
    appendRect(builder, attributes);
    return;
  }
  if (name === 'circle') {
    const radius = length(attributes.r);
    appendEllipse(builder, length(attributes.cx, 0), length(attributes.cy, 0), radius, radius);
    return;
  }
  if (name === 'ellipse') {
    appendEllipse(
      builder,
      length(attributes.cx, 0),
      length(attributes.cy, 0),
      length(attributes.rx),
      length(attributes.ry),
    );
    return;
  }
  if (name === 'line') {
    const x1 = length(attributes.x1, 0);
    const y1 = length(attributes.y1, 0);
    const x2 = length(attributes.x2, 0);
    const y2 = length(attributes.y2, 0);
    builder.moveTo(x1, y1);
    builder.lineTo(x2, y2);
    return;
  }
  appendPoints(builder, attributes, name === 'polygon');
}

function transformContours(contours, matrix) {
  const apply = (x, y) => [
    matrix[0] * x + matrix[2] * y + matrix[4],
    matrix[1] * x + matrix[3] * y + matrix[5],
  ];
  return contours.map(contour => ({
    start: apply(contour.start[0], contour.start[1]),
    closed: contour.closed,
    segments: contour.segments.map(segment => {
      if (segment[0] === 'L') return ['L', ...apply(segment[1], segment[2])];
      return [
        'C',
        ...apply(segment[1], segment[2]),
        ...apply(segment[3], segment[4]),
        ...apply(segment[5], segment[6]),
      ];
    }),
  }));
}

function parseContours(text, limits) {
  const stack = [{ matrix: IDENTITY, fill: undefined, stroke: undefined }];
  const contours = [];
  const budget = { segments: limits.maxSegments };
  let elementCount = 0;
  let skipDepth = 0;
  let index = 0;

  while (index < text.length) {
    const start = text.indexOf('<', index);
    if (start === -1) break;

    if (text.startsWith('<!--', start)) {
      const end = text.indexOf('-->', start + 4);
      if (end === -1) break;
      index = end + 3;
      continue;
    }
    if (text.startsWith('<![CDATA[', start)) {
      const end = text.indexOf(']]>', start + 9);
      if (end === -1) break;
      index = end + 3;
      continue;
    }
    if (/^<!doctype/i.test(text.slice(start, start + 9))) {
      index = skipDoctype(text, start);
      continue;
    }
    if (/^<!entity/i.test(text.slice(start, start + 8))) {
      throw new Error('SVG entity declarations are not supported');
    }
    if (text.startsWith('<?', start)) {
      const end = text.indexOf('?>', start + 2);
      if (end === -1) break;
      index = end + 2;
      continue;
    }

    const end = findTagEnd(text, start);
    if (end === -1) break;
    const body = text.slice(start + 1, end);
    index = end + 1;

    const nameMatch = /^\/?\s*([^\s/>]+)/.exec(body);
    if (nameMatch === null) continue;
    const name = nameMatch[1].toLowerCase().replace(/^.*:/, '');

    if (body.startsWith('/')) {
      if (skipDepth > 0) skipDepth -= 1;
      if (stack.length > 1) stack.pop();
      continue;
    }

    elementCount += 1;
    if (elementCount > limits.maxElements) {
      throw new Error('SVG document exceeds the element budget');
    }

    const selfClosing = body.endsWith('/');
    const parent = stack[stack.length - 1];

    if (skipDepth > 0) {
      if (!selfClosing) {
        skipDepth += 1;
        stack.push(parent);
      }
      continue;
    }

    const attributes = parseAttributes(body);
    const paint = presentation(attributes, parent);
    const isShape = SHAPE_ELEMENTS.has(name);
    const skipped = SKIPPED_ELEMENTS.has(name)
      || isHidden(paint)
      || (isShape && isUnpainted(paint));

    if (skipped) {
      if (!selfClosing) {
        skipDepth = 1;
        stack.push(parent);
      }
      continue;
    }

    const matrix = multiply(parent.matrix, parseTransform(attributes.transform));
    if (isShape) {
      const builder = new ContourBuilder(budget);
      appendShape(builder, name, attributes);
      contours.push(...transformContours(builder.contours, matrix));
    }
    if (!selfClosing) stack.push({ matrix, fill: paint.fill, stroke: paint.stroke });
  }

  return contours;
}

function contourBounds(contours) {
  let left = Number.POSITIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;

  const include = (x, y) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (x < left) left = x;
    if (x > right) right = x;
    if (y < top) top = y;
    if (y > bottom) bottom = y;
  };

  for (const contour of contours) {
    include(contour.start[0], contour.start[1]);
    for (const segment of contour.segments) {
      for (let offset = 1; offset + 1 < segment.length; offset += 2) {
        include(segment[offset], segment[offset + 1]);
      }
    }
  }

  if (left > right || top > bottom) return null;
  return { left, top, right, bottom };
}

function appendCubic(points, x0, y0, x1, y1, x2, y2, x3, y3, tolerance, depth) {
  const chordX = x3 - x0;
  const chordY = y3 - y0;
  const distance1 = Math.abs((x1 - x3) * chordY - (y1 - y3) * chordX);
  const distance2 = Math.abs((x2 - x3) * chordY - (y2 - y3) * chordX);
  const deviation = (distance1 + distance2) ** 2;

  if (depth >= MAX_SUBDIVISION_DEPTH
    || deviation <= tolerance * (chordX * chordX + chordY * chordY)) {
    points.push([x3, y3]);
    return;
  }

  const x01 = (x0 + x1) / 2;
  const y01 = (y0 + y1) / 2;
  const x12 = (x1 + x2) / 2;
  const y12 = (y1 + y2) / 2;
  const x23 = (x2 + x3) / 2;
  const y23 = (y2 + y3) / 2;
  const x012 = (x01 + x12) / 2;
  const y012 = (y01 + y12) / 2;
  const x123 = (x12 + x23) / 2;
  const y123 = (y12 + y23) / 2;
  const midX = (x012 + x123) / 2;
  const midY = (y012 + y123) / 2;

  appendCubic(points, x0, y0, x01, y01, x012, y012, midX, midY, tolerance, depth + 1);
  appendCubic(points, midX, midY, x123, y123, x23, y23, x3, y3, tolerance, depth + 1);
}

function flattenContours(contours, limits) {
  const bounds = contourBounds(contours);
  if (bounds === null) return [];

  const diagonal = Math.hypot(bounds.right - bounds.left, bounds.bottom - bounds.top);
  const distanceTolerance = Math.max(diagonal / limits.flattenResolution, Number.EPSILON);
  // appendCubic compares squared perpendicular deviation against this scaled chord length.
  const tolerance = distanceTolerance * distanceTolerance;
  const polylines = [];
  let vertexCount = 0;

  for (const contour of contours) {
    const points = [contour.start];
    let x = contour.start[0];
    let y = contour.start[1];

    for (const segment of contour.segments) {
      if (segment[0] === 'L') {
        points.push([segment[1], segment[2]]);
        x = segment[1];
        y = segment[2];
      } else {
        appendCubic(
          points,
          x, y,
          segment[1], segment[2],
          segment[3], segment[4],
          segment[5], segment[6],
          tolerance,
          0,
        );
        x = segment[5];
        y = segment[6];
      }
      vertexCount = points.length;
      if (vertexCount > limits.maxVertices) {
        throw new Error('SVG geometry exceeds the vertex budget');
      }
    }

    if (contour.closed) points.push([contour.start[0], contour.start[1]]);

    const simplified = points.filter((point, position) => position === 0
      || point[0] !== points[position - 1][0]
      || point[1] !== points[position - 1][1]);
    if (simplified.length > 1) polylines.push(simplified);
  }

  return polylines;
}

export function svgToPolylines(svgText, options = {}) {
  if (typeof svgText !== 'string') throw new Error('SVG source must be a string');
  const limits = resolvedLimits(options);
  if (svgText.length > limits.maxCharacters) {
    throw new Error('SVG source exceeds the maximum supported size');
  }
  return flattenContours(parseContours(svgText, limits), limits);
}

export const SVG_LIMITS = DEFAULT_LIMITS;
