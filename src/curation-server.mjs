import { readFile, realpath, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';

import { isConfinedSvgPath } from './icon-candidate.mjs';
import { createApprovedIconProxy } from './icon-proxy.mjs';

const CONTENT_TYPES = {
  '.bin': 'application/octet-stream',
  '.gz': 'application/gzip',
  '.html': 'text/html; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
};
const CONTENT_SECURITY_POLICY = "default-src 'self'; img-src 'self' https://cdn.jsdelivr.net data:; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'";

const SERVED_PATHS = new Map([
  ['/', 'index.html'],
  ['/index.html', 'index.html'],
  ['/curate.html', 'curate.html'],
  ['/config/mvp-classes.txt', 'config/mvp-classes.txt'],
  ['/data/quickdraw-candidates.json', 'data/quickdraw-candidates.json'],
  ['/src/curation-app.mjs', 'src/curation-app.mjs'],
  ['/src/curation-state.mjs', 'src/curation-state.mjs'],
  ['/src/line-list.mjs', 'src/line-list.mjs'],
  ['/src/autodraw-suggestions.mjs', 'src/autodraw-suggestions.mjs'],
  ['/src/drawing-app.mjs', 'src/drawing-app.mjs'],
  ['/src/drawing-export.mjs', 'src/drawing-export.mjs'],
  ['/src/drawing-state.mjs', 'src/drawing-state.mjs'],
  ['/src/icon-candidate.mjs', 'src/icon-candidate.mjs'],
  ['/src/icon-retrieval.mjs', 'src/icon-retrieval.mjs'],
  ['/src/quickdraw-classifier.mjs', 'src/quickdraw-classifier.mjs'],
  ['/src/sketch-embedder.mjs', 'src/sketch-embedder.mjs'],
  ['/src/sketch-rasterizer.mjs', 'src/sketch-rasterizer.mjs'],
  ['/src/tokenize.mjs', 'src/tokenize.mjs'],
  ['/models/quickdraw-mvp/model.json', 'models/quickdraw-mvp/model.json'],
  ['/models/quickdraw-mvp/quickdraw-mvp.onnx', 'models/quickdraw-mvp/quickdraw-mvp.onnx'],
  ['/models/sketch-embedder/model.json', 'models/sketch-embedder/model.json'],
  ['/models/sketch-embedder/sketch-embedder.onnx', 'models/sketch-embedder/sketch-embedder.onnx'],
  ['/vendor/ort.wasm.min.mjs', 'node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'],
  ['/vendor/ort-wasm-simd-threaded.mjs', 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs'],
  ['/vendor/ort-wasm-simd-threaded.wasm', 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm'],
]);

function pathError(message, statusCode) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

function responseHeaders(contentType) {
  return {
    'Cache-Control': 'no-store',
    'Content-Security-Policy': CONTENT_SECURITY_POLICY,
    'Content-Type': contentType,
    'X-Content-Type-Options': 'nosniff',
  };
}

function isInside(rootPath, filePath) {
  return filePath === rootPath || filePath.startsWith(`${rootPath}${sep}`);
}

function svgDepotPath(pathname) {
  if (!pathname.startsWith('/svgdepot/')) return null;
  const relativePath = pathname.slice('/svgdepot/'.length);
  if (!isConfinedSvgPath(relativePath)) throw pathError('Not found', 404);
  return relativePath;
}

// The icon index is hundreds of generated shards, so the route matches their exact
// generated names rather than listing every file.
const ICON_INDEX_PREFIX = '/data/icon-embeddings/';
const ICON_INDEX_FILE = /^(?:index\.json|shard-\d{1,4}\.(?:bin|meta\.json\.gz))$/;

function iconEmbeddingPath(pathname) {
  if (!pathname.startsWith(ICON_INDEX_PREFIX)) return null;
  const name = pathname.slice(ICON_INDEX_PREFIX.length);
  if (!ICON_INDEX_FILE.test(name)) throw pathError('Not found', 404);
  return `data/icon-embeddings/${name}`;
}

function decodedPathname(requestPathname) {
  try {
    return decodeURIComponent(requestPathname);
  } catch {
    throw pathError('Not found', 404);
  }
}

export async function resolveServedFile(projectRoot, requestPathname) {
  const pathname = decodedPathname(requestPathname);

  const localSvgPath = svgDepotPath(pathname);
  const relativePath = localSvgPath
    ?? iconEmbeddingPath(pathname)
    ?? SERVED_PATHS.get(pathname);
  if (!relativePath) throw pathError('Not found', 404);

  const rootRealPath = await realpath(projectRoot);
  const allowedRootPath = localSvgPath
    ? await realpath(resolve(projectRoot, '.cache', 'svgdepot'))
    : rootRealPath;
  if (!isInside(rootRealPath, allowedRootPath)) {
    throw pathError('Forbidden', 403);
  }
  const fileRealPath = await realpath(resolve(allowedRootPath, relativePath));
  if (!isInside(allowedRootPath, fileRealPath)) throw pathError('Forbidden', 403);
  return fileRealPath;
}

export function createCurationHandler(projectRoot, { fetchImpl = globalThis.fetch } = {}) {
  let proxyPromise;

  async function approvedIconFallback(requestPathname) {
    const localSvgPath = svgDepotPath(decodedPathname(requestPathname));
    if (!localSvgPath) return null;
    proxyPromise ??= readFile(
      resolve(projectRoot, 'data', 'quickdraw-candidates.json'),
      'utf8',
    ).then(JSON.parse).then(manifest => createApprovedIconProxy({ manifest, fetchImpl }));
    return (await proxyPromise).icon(localSvgPath);
  }

  return async function handleCurationRequest(request, response) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, {
        ...responseHeaders('text/plain; charset=utf-8'),
        Allow: 'GET, HEAD',
      });
      response.end('Method not allowed');
      return;
    }

    try {
      const url = new URL(request.url, 'http://localhost');
      const filePath = await resolveServedFile(projectRoot, url.pathname);
      const fileStats = await stat(filePath);
      if (!fileStats.isFile()) throw pathError('Not found', 404);
      const body = await readFile(filePath);
      const contentType = CONTENT_TYPES[extname(filePath).toLowerCase()] ?? 'application/octet-stream';
      response.writeHead(200, responseHeaders(contentType));
      response.end(request.method === 'HEAD' ? undefined : body);
    } catch (error) {
      if (!error.statusCode) {
        try {
          const fallback = await approvedIconFallback(new URL(request.url, 'http://localhost').pathname);
          if (fallback) {
            response.writeHead(200, responseHeaders('image/svg+xml'));
            response.end(request.method === 'HEAD' ? undefined : fallback.body);
            return;
          }
        } catch {}
      }
      const statusCode = error.statusCode ?? 404;
      response.writeHead(statusCode, responseHeaders('text/plain; charset=utf-8'));
      response.end(request.method === 'HEAD' ? undefined : statusCode === 403 ? 'Forbidden' : 'Not found');
    }
  };
}