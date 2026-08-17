const DEFAULT_OPTIONS = Object.freeze({
  size: 64,
  padding: 4,
  strokeWidth: 2.5,
  supersample: 4,
});

function finiteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${label} coordinates must be finite numbers`);
  }
  return value;
}

function rasterOptions(options = {}) {
  const values = { ...DEFAULT_OPTIONS, ...options };
  if (!Number.isInteger(values.size) || values.size < 2) {
    throw new Error('Raster size must be an integer greater than 1');
  }
  if (!Number.isFinite(values.padding) || values.padding < 0) {
    throw new Error('Raster padding must be non-negative');
  }
  if (!Number.isFinite(values.strokeWidth) || values.strokeWidth <= 0) {
    throw new Error('Raster stroke width must be positive');
  }
  if ((values.padding + values.strokeWidth / 2) * 2 >= values.size) {
    throw new Error('Raster padding and stroke width must leave drawable space');
  }
  if (!Number.isInteger(values.supersample) || values.supersample < 1 || values.supersample > 8) {
    throw new Error('Raster supersample must be an integer from 1 to 8');
  }
  return values;
}

function validatedPolylines(polylines) {
  if (!Array.isArray(polylines)) throw new Error('Polylines must be an array');
  return polylines
    .map((polyline, strokeIndex) => {
      if (!Array.isArray(polyline)) throw new Error(`Polyline ${strokeIndex} must be an array`);
      return polyline.map((point, pointIndex) => {
        if (!Array.isArray(point) || point.length !== 2) {
          throw new Error(`Point ${strokeIndex}/${pointIndex} must contain exactly x and y coordinates`);
        }
        return [
          finiteNumber(point[0], `Point ${strokeIndex}/${pointIndex}`),
          finiteNumber(point[1], `Point ${strokeIndex}/${pointIndex}`),
        ];
      });
    })
    .filter(polyline => polyline.length > 0);
}

export function stroke3ToPolylines(stroke3) {
  if (!Array.isArray(stroke3)) throw new Error('Stroke-3 drawing must be an array');
  if (stroke3.length === 0) throw new Error('Stroke-3 drawing must contain at least one point');
  const polylines = [];
  let currentPolyline;
  let x = 0;
  let y = 0;
  let previousPenLift = 1;

  stroke3.forEach((point, index) => {
    if (!Array.isArray(point) || point.length !== 3) {
      throw new Error(`Stroke-3 point ${index} must contain exactly dx, dy, and pen state`);
    }
    const deltaX = finiteNumber(point[0], `Stroke-3 point ${index}`);
    const deltaY = finiteNumber(point[1], `Stroke-3 point ${index}`);
    const penLift = point[2];
    if (penLift !== 0 && penLift !== 1) {
      throw new Error(`Stroke-3 point ${index} pen state must be 0 or 1`);
    }

    x += deltaX;
    y += deltaY;
    if (previousPenLift === 1) {
      currentPolyline = [[x, y]];
      polylines.push(currentPolyline);
    } else {
      currentPolyline.push([x, y]);
    }
    previousPenLift = penLift;
  });

  if (previousPenLift !== 1) throw new Error('Stroke-3 drawing must end with pen state 1');

  return polylines;
}

export function quickDrawToPolylines(drawing) {
  if (!Array.isArray(drawing)) throw new Error('Quick Draw drawing must be an array');
  return drawing.map((stroke, strokeIndex) => {
    if (
      !Array.isArray(stroke)
      || (stroke.length !== 2 && stroke.length !== 3)
      || !Array.isArray(stroke[0])
      || !Array.isArray(stroke[1])
      || (stroke.length === 3 && !Array.isArray(stroke[2]))
    ) {
      throw new Error(`Quick Draw stroke ${strokeIndex} must contain x and y arrays with optional timing`);
    }
    if (stroke[0].length !== stroke[1].length) {
      throw new Error(`Quick Draw stroke ${strokeIndex} x and y arrays must have equal length`);
    }
    if (stroke.length === 3 && stroke[2].length !== stroke[0].length) {
      throw new Error(`Quick Draw stroke ${strokeIndex} timing array must match coordinate length`);
    }
    return stroke[0].map((x, pointIndex) => [
      finiteNumber(x, `Quick Draw stroke ${strokeIndex}`),
      finiteNumber(stroke[1][pointIndex], `Quick Draw stroke ${strokeIndex}`),
    ]);
  });
}

function normalizedPolylines(polylines, { size, padding, strokeWidth }) {
  const points = polylines.flat();
  if (points.length === 0) return [];

  const xValues = points.map(point => point[0]);
  const yValues = points.map(point => point[1]);
  const left = Math.min(...xValues);
  const top = Math.min(...yValues);
  const drawingWidth = Math.max(...xValues) - left;
  const drawingHeight = Math.max(...yValues) - top;
  if (!Number.isFinite(drawingWidth) || !Number.isFinite(drawingHeight)) {
    throw new Error('Drawing bounds exceed the supported numeric range');
  }
  const drawableSize = size - (padding + strokeWidth / 2) * 2;
  const scales = [drawingWidth, drawingHeight]
    .filter(dimension => dimension > 0)
    .map(dimension => drawableSize / dimension);
  const scale = scales.length > 0 ? Math.min(...scales) : 1;
  const scaledWidth = drawingWidth * scale;
  const scaledHeight = drawingHeight * scale;
  const offsetX = (size - scaledWidth) / 2 - left * scale;
  const offsetY = (size - scaledHeight) / 2 - top * scale;
  if (![scale, offsetX, offsetY].every(Number.isFinite)) {
    throw new Error('Drawing bounds exceed the supported numeric range');
  }

  return polylines.map(polyline => polyline.map(point => {
    const normalizedPoint = [
      point[0] * scale + offsetX,
      point[1] * scale + offsetY,
    ];
    if (!normalizedPoint.every(Number.isFinite)) {
      throw new Error('Drawing bounds exceed the supported numeric range');
    }
    return normalizedPoint;
  }));
}

function squaredDistanceToSegment(x, y, start, end) {
  const segmentX = end[0] - start[0];
  const segmentY = end[1] - start[1];
  const lengthSquared = segmentX * segmentX + segmentY * segmentY;
  if (lengthSquared === 0) {
    const deltaX = x - start[0];
    const deltaY = y - start[1];
    return deltaX * deltaX + deltaY * deltaY;
  }

  const projection = Math.max(0, Math.min(1, (
    (x - start[0]) * segmentX + (y - start[1]) * segmentY
  ) / lengthSquared));
  const closestX = start[0] + projection * segmentX;
  const closestY = start[1] + projection * segmentY;
  const deltaX = x - closestX;
  const deltaY = y - closestY;
  return deltaX * deltaX + deltaY * deltaY;
}

function drawCapsule(highResolution, highSize, supersample, start, end, radius) {
  const left = Math.max(0, Math.floor((Math.min(start[0], end[0]) - radius) * supersample));
  const top = Math.max(0, Math.floor((Math.min(start[1], end[1]) - radius) * supersample));
  const right = Math.min(
    highSize - 1,
    Math.ceil((Math.max(start[0], end[0]) + radius) * supersample) - 1,
  );
  const bottom = Math.min(
    highSize - 1,
    Math.ceil((Math.max(start[1], end[1]) + radius) * supersample) - 1,
  );
  const radiusSquared = radius * radius;

  for (let y = top; y <= bottom; y += 1) {
    const sampleY = (y + 0.5) / supersample;
    for (let x = left; x <= right; x += 1) {
      if (highResolution[y * highSize + x] === 1) continue;
      const sampleX = (x + 0.5) / supersample;
      if (squaredDistanceToSegment(sampleX, sampleY, start, end) <= radiusSquared) {
        highResolution[y * highSize + x] = 1;
      }
    }
  }
}

export function rasterizePolylines(polylines, options = {}) {
  const values = rasterOptions(options);
  const validated = validatedPolylines(polylines);
  const output = new Uint8Array(values.size * values.size);
  if (validated.length === 0) return output;

  const normalized = normalizedPolylines(validated, values);
  const highSize = values.size * values.supersample;
  const highResolution = new Uint8Array(highSize * highSize);
  const radius = values.strokeWidth / 2;

  normalized.forEach(polyline => {
    if (polyline.length === 1) {
      drawCapsule(highResolution, highSize, values.supersample, polyline[0], polyline[0], radius);
      return;
    }
    for (let index = 1; index < polyline.length; index += 1) {
      drawCapsule(
        highResolution,
        highSize,
        values.supersample,
        polyline[index - 1],
        polyline[index],
        radius,
      );
    }
  });

  const samplesPerPixel = values.supersample ** 2;
  for (let outputY = 0; outputY < values.size; outputY += 1) {
    for (let outputX = 0; outputX < values.size; outputX += 1) {
      let inkSamples = 0;
      for (let sampleY = 0; sampleY < values.supersample; sampleY += 1) {
        const highY = outputY * values.supersample + sampleY;
        for (let sampleX = 0; sampleX < values.supersample; sampleX += 1) {
          const highX = outputX * values.supersample + sampleX;
          inkSamples += highResolution[highY * highSize + highX];
        }
      }
      output[outputY * values.size + outputX] = Math.round(inkSamples * 255 / samplesPerPixel);
    }
  }
  return output;
}

export function rasterizeStroke3(stroke3, options = {}) {
  return rasterizePolylines(stroke3ToPolylines(stroke3), options);
}

export function rasterizeQuickDraw(drawing, options = {}) {
  return rasterizePolylines(quickDrawToPolylines(drawing), options);
}
