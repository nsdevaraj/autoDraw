import { createHash } from 'node:crypto';

export const ICON_MANIFEST_SCHEMA_VERSION = 1;
export const CDN_HOSTNAME = 'cdn.jsdelivr.net';
export const UNAVAILABLE_DIGEST = -1;

const INDEX_SCHEMA_VERSION = 1;
const ACCEPTED_CONTENT_TYPES = new Set(['image/svg+xml', 'text/plain']);

// Marks an icon we will never accept, so callers skip it instead of retrying it.
export class IconRejectedError extends Error {
  constructor(message) {
    super(message);
    this.name = 'IconRejectedError';
  }
}

export function iconPathsFromIndex(index) {
  if (index?.schemaVersion !== INDEX_SCHEMA_VERSION) {
    throw new Error(`Unsupported SVGDepot index schema version: ${index?.schemaVersion}`);
  }
  if (!Array.isArray(index.icons) || !Array.isArray(index.packs) || !Array.isArray(index.categories)) {
    throw new Error('SVGDepot index must contain icons, packs, and categories');
  }

  return index.icons.map(([packId, filename], iconId) => {
    const pack = index.packs[packId];
    if (!Array.isArray(pack) || typeof filename !== 'string') {
      throw new Error(`SVGDepot index entry ${iconId} is malformed`);
    }
    const category = index.categories[pack[0]];
    if (typeof category !== 'string') {
      throw new Error(`SVGDepot index entry ${iconId} has an unknown category`);
    }
    return { packId, path: [category, pack[1], filename].filter(Boolean).join('/') };
  });
}

// Deterministically keeps the first icons of every pack so a sample still spans all packs.
export function selectIconIds(icons, limitPerPack = Number.POSITIVE_INFINITY) {
  if (limitPerPack === Number.POSITIVE_INFINITY) return icons.map((icon, iconId) => iconId);
  if (!Number.isInteger(limitPerPack) || limitPerPack < 1) {
    throw new Error('Icons per pack must be a positive integer');
  }

  const takenByPack = new Map();
  const selected = [];
  icons.forEach((icon, iconId) => {
    const taken = takenByPack.get(icon.packId) ?? 0;
    if (taken >= limitPerPack) return;
    takenByPack.set(icon.packId, taken + 1);
    selected.push(iconId);
  });
  return selected;
}

// The supervision subset: only the icons already mapped to a Quick Draw class.
export function candidateIconIds(candidates, iconCount, commit) {
  if (candidates?.schemaVersion !== 1 || !Array.isArray(candidates.classes)) {
    throw new Error('Invalid Quick Draw candidate manifest');
  }
  if (candidates.source?.commit !== commit) {
    throw new Error('Candidate manifest was built from a different SVGDepot commit');
  }

  const iconIds = new Set();
  for (const item of candidates.classes) {
    for (const candidate of item.candidates ?? []) {
      if (!Number.isInteger(candidate.id) || candidate.id < 0 || candidate.id >= iconCount) {
        throw new Error(`Candidate icon id is out of range: ${candidate.id}`);
      }
      iconIds.add(candidate.id);
    }
  }
  return [...iconIds].sort((left, right) => left - right);
}

export function assertTrustedIconResponse(response, maxBytes) {
  let hostname;
  try {
    hostname = new URL(response.url).hostname;
  } catch {
    throw new IconRejectedError(`Icon response has an unusable URL: ${response.url}`);
  }
  if (hostname !== CDN_HOSTNAME) {
    throw new IconRejectedError(
      `Icon response redirected away from ${CDN_HOSTNAME}: ${response.url}`,
    );
  }

  const contentType = (response.headers.get('content-type') ?? '')
    .split(';')[0]
    .trim()
    .toLowerCase();
  if (!ACCEPTED_CONTENT_TYPES.has(contentType)) {
    throw new IconRejectedError(
      `Icon response has an unexpected content type: ${contentType || '(none)'}`,
    );
  }

  const declaredBytes = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredBytes) && declaredBytes > maxBytes) {
    throw new IconRejectedError(
      `Icon response declares ${declaredBytes} bytes, above the ${maxBytes} limit`,
    );
  }
}

export function buildIconManifest({ source, iconCount, fetcher, counts, digests, iconDigests }) {
  const manifest = {
    schemaVersion: ICON_MANIFEST_SCHEMA_VERSION,
    source: {
      repository: source.repository,
      ref: source.ref,
      commit: source.commit,
      iconCount,
    },
    fetcher,
    counts: {
      ...counts,
      available: iconDigests.filter(digestId => digestId !== UNAVAILABLE_DIGEST).length,
      unique: digests.length,
    },
    digests,
    iconDigests,
  };
  return {
    ...manifest,
    fingerprint: createHash('sha256').update(JSON.stringify(manifest)).digest('hex'),
  };
}
