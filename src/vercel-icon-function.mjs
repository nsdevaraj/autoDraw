import { createApprovedIconProxy } from './icon-proxy.mjs';

const ICON_CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'unsafe-inline'; sandbox";

function responseHeaders({ cache = false, contentType = 'text/plain; charset=utf-8' } = {}) {
  const headers = {
    'Cache-Control': cache
      ? 'public, max-age=86400, stale-if-error=604800'
      : 'no-store',
    'Content-Security-Policy': ICON_CONTENT_SECURITY_POLICY,
    'Content-Type': contentType,
    'Cross-Origin-Resource-Policy': 'same-origin',
    'X-Content-Type-Options': 'nosniff',
  };
  if (cache) headers['Vercel-CDN-Cache-Control'] = 'max-age=31536000, immutable';
  return headers;
}

function textResponse(message, status, headers = {}) {
  return new Response(message, {
    status,
    headers: { ...responseHeaders(), ...headers },
  });
}

export function createVercelIconHandler({ manifest, fetchImpl = globalThis.fetch }) {
  const proxy = createApprovedIconProxy({ manifest, fetchImpl });

  return async function handleIconRequest(request) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return textResponse('Method not allowed', 405, { Allow: 'GET, HEAD' });
    }

    let path;
    try {
      path = new URL(request.url).searchParams.get('path');
    } catch {
      return textResponse('Not found', 404);
    }

    try {
      const approvedIcon = await proxy.icon(path);
      if (!approvedIcon) return textResponse('Not found', 404);
      return new Response(request.method === 'HEAD' ? null : approvedIcon.body, {
        status: 200,
        headers: responseHeaders({ cache: true, contentType: 'image/svg+xml' }),
      });
    } catch {
      return textResponse('Bad gateway', 502);
    }
  };
}