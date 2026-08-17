import { expectedIconUrl, isApprovedIconUrl, isConfinedSvgPath } from './icon-candidate.mjs';

export const MAXIMUM_PROXIED_ICON_BYTES = 2 * 1024 * 1024;
export const DEFAULT_ICON_CACHE_BYTES = 8 * 1024 * 1024;
export const DEFAULT_MAXIMUM_CONCURRENT_FETCHES = 4;

function approvedIconUrls(manifest) {
  if (
    !manifest
    || manifest.schemaVersion !== 1
    || !manifest.source
    || !Array.isArray(manifest.classes)
  ) {
    throw new Error('Invalid Quick Draw candidate manifest');
  }

  const urls = new Map();
  for (const manifestClass of manifest.classes) {
    for (const candidate of manifestClass.candidates ?? []) {
      if (!isApprovedIconUrl(candidate.url, candidate.path, manifest.source)) continue;
      const existing = urls.get(candidate.path);
      if (existing && existing !== candidate.url) {
        throw new Error(`Conflicting approved icon URL: ${candidate.path}`);
      }
      urls.set(candidate.path, candidate.url);
    }
  }
  return urls;
}

async function boundedBody(response) {
  const contentLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(contentLength) && contentLength > MAXIMUM_PROXIED_ICON_BYTES) {
    throw new Error('Icon response exceeds the maximum size');
  }
  if (!response.body) throw new Error('Icon response has no body');

  const reader = response.body.getReader();
  const chunks = [];
  let byteCount = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      byteCount += value.byteLength;
      if (byteCount > MAXIMUM_PROXIED_ICON_BYTES) {
        await reader.cancel();
        throw new Error('Icon response exceeds the maximum size');
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  if (byteCount === 0) throw new Error('Icon response is empty');
  const body = new Uint8Array(byteCount);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

async function validatedSvgResponse(response) {
  if (!response.ok) throw new Error(`Icon request failed: ${response.status}`);
  const contentType = response.headers.get('content-type')?.split(';')[0].trim();
  if (contentType !== 'image/svg+xml') throw new Error('Icon response is not SVG');
  const body = await boundedBody(response);
  const prefix = new TextDecoder().decode(body.subarray(0, 4096));
  if (!/<svg(?:\s|\/?>)/i.test(prefix)) {
    throw new Error('Icon response does not contain SVG markup');
  }
  return body;
}

export function createApprovedIconProxy({
  manifest,
  fetchImpl = globalThis.fetch,
  maximumCacheBytes = DEFAULT_ICON_CACHE_BYTES,
  maximumConcurrentFetches = DEFAULT_MAXIMUM_CONCURRENT_FETCHES,
}) {
  if (typeof fetchImpl !== 'function') throw new Error('Icon proxy requires fetch');
  if (!Number.isSafeInteger(maximumCacheBytes) || maximumCacheBytes < 0) {
    throw new Error('Icon cache byte limit must be a non-negative safe integer');
  }
  if (!Number.isSafeInteger(maximumConcurrentFetches) || maximumConcurrentFetches < 1) {
    throw new Error('Maximum concurrent icon fetches must be a positive safe integer');
  }
  const urls = approvedIconUrls(manifest);
  // Retrieval reaches every icon in the pinned commit, not just the manifest's candidates,
  // so unlisted paths are resolved to the one URL shape expectedIconUrl() can produce.
  const source = manifest.source;
  const cache = new Map();
  const activeRequests = new Map();
  let cachedBytes = 0;
  let activeFetchCount = 0;
  const fetchWaiters = [];

  function acquireFetchPermit() {
    if (activeFetchCount < maximumConcurrentFetches) {
      activeFetchCount += 1;
      return Promise.resolve();
    }
    return new Promise(resolvePermit => fetchWaiters.push(resolvePermit));
  }

  function releaseFetchPermit() {
    const next = fetchWaiters.shift();
    if (next) next();
    else activeFetchCount -= 1;
  }

  async function fetchIcon(url) {
    await acquireFetchPermit();
    try {
      const response = await fetchImpl(url, { redirect: 'error' });
      return await validatedSvgResponse(response);
    } finally {
      releaseFetchPermit();
    }
  }

  function cachedIcon(path) {
    const value = cache.get(path);
    if (!value) return null;
    cache.delete(path);
    cache.set(path, value);
    return value;
  }

  function cacheIcon(path, value) {
    if (value.body.byteLength > maximumCacheBytes) return;
    cache.set(path, value);
    cachedBytes += value.body.byteLength;
    while (cachedBytes > maximumCacheBytes) {
      const oldestPath = cache.keys().next().value;
      const oldest = cache.get(oldestPath);
      cache.delete(oldestPath);
      cachedBytes -= oldest.body.byteLength;
    }
  }

  async function icon(path) {
    if (!isConfinedSvgPath(path)) return null;
    const url = urls.get(path) ?? expectedIconUrl(source, path);
    if (!url) return null;

    const cached = cachedIcon(path);
    if (cached) return cached;
    if (activeRequests.has(path)) return activeRequests.get(path);

    const request = (async () => {
      return Object.freeze({
        body: await fetchIcon(url),
        path,
        url,
      });
    })();
    activeRequests.set(path, request);
    try {
      const value = await request;
      cacheIcon(path, value);
      return value;
    } catch (error) {
      throw error;
    } finally {
      activeRequests.delete(path);
    }
  }

  return Object.freeze({ icon });
}