export const TOKENIZER_ID = 'filename-nfkd-lower-ascii-alpha-v1';

export function tokenize(value) {
  return [...new Set(
    value
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase()
      .split(/[^a-z]+/)
      .filter(Boolean),
  )];
}
