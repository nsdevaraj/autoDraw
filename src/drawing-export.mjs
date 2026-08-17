function finitePositive(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function attribute(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function number(value) {
  return Number(value.toFixed(4)).toString();
}

function strokeElement(stroke) {
  const [first] = stroke.points;
  if (stroke.points.length === 1) {
    return `<circle cx="${number(first[0])}" cy="${number(first[1])}" r="${number(stroke.width / 2)}" fill="${attribute(stroke.color)}"/>`;
  }

  const commands = [`M ${number(first[0])} ${number(first[1])}`];
  for (let index = 1; index < stroke.points.length; index += 1) {
    const previous = stroke.points[index - 1];
    const current = stroke.points[index];
    commands.push(
      `Q ${number(previous[0])} ${number(previous[1])} ${number((previous[0] + current[0]) / 2)} ${number((previous[1] + current[1]) / 2)}`,
    );
  }
  const last = stroke.points[stroke.points.length - 1];
  commands.push(`L ${number(last[0])} ${number(last[1])}`);
  return `<path d="${commands.join(' ')}" fill="none" stroke="${attribute(stroke.color)}" stroke-width="${number(stroke.width)}" stroke-linecap="round" stroke-linejoin="round"/>`;
}

export function createDrawingSvg({ state, width, height, iconDataByPath }) {
  if (!finitePositive(width) || !finitePositive(height)) {
    throw new Error('SVG export dimensions must be positive finite numbers');
  }
  if (!(iconDataByPath instanceof Map)) throw new Error('SVG export requires embedded icon data');

  const elements = [
    `<rect width="100%" height="100%" fill="#ffffff"/>`,
  ];
  for (const icon of state.icons) {
    const iconData = iconDataByPath.get(icon.path);
    if (typeof iconData !== 'string' || !iconData.startsWith('data:image/')) {
      throw new Error(`SVG export is missing embedded icon data: ${icon.path}`);
    }
    elements.push(
      `<image href="${attribute(iconData)}" x="${number(icon.x)}" y="${number(icon.y)}" width="${number(icon.width)}" height="${number(icon.height)}" preserveAspectRatio="xMidYMid meet"/>`,
    );
  }
  state.draftStrokes.forEach(stroke => elements.push(strokeElement(stroke)));

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${number(width)}" height="${number(height)}" viewBox="0 0 ${number(width)} ${number(height)}">`,
    ...elements,
    '</svg>',
  ].join('\n');
}