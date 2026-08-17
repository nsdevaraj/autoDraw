import { readFile } from 'node:fs/promises';

import { createVercelIconHandler } from '../src/vercel-icon-function.mjs';

const manifest = JSON.parse(await readFile(
  new URL('../data/quickdraw-candidates.json', import.meta.url),
  'utf8',
));
const handler = createVercelIconHandler({ manifest });

export const GET = handler;
export const HEAD = handler;