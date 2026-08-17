export function parseLineList(text) {
  return text
    .split(/\r?\n/)
    .map(value => value.trim())
    .filter(Boolean);
}