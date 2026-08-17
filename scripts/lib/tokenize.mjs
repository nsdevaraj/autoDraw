// The browser retrieval runtime needs the same tokenizer, and it cannot import from
// scripts/, so the implementation lives in src/ and this module re-exports it.
export { TOKENIZER_ID, tokenize } from '../../src/tokenize.mjs';