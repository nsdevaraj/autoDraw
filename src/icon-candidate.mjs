export function isConfinedSvgPath(value) {
  if (typeof value !== 'string' || !value.endsWith('.svg') || value.includes('\\')) return false;
  const segments = value.split('/');
  return segments.length > 1
    && segments.every(segment => segment.length > 0 && segment !== '.' && segment !== '..');
}

function repositorySlug(repository) {
  if (typeof repository !== 'string') return null;
  try {
    const url = new URL(repository);
    const segments = url.pathname.replace(/^\/+|\/+$/g, '').split('/');
    if (
      url.protocol !== 'https:'
      || url.hostname !== 'github.com'
      || url.username
      || url.password
      || url.port
      || url.search
      || url.hash
      || segments.length !== 2
      || segments.some(segment => !segment)
    ) return null;
    return `${segments[0]}/${segments[1].replace(/\.git$/, '')}`;
  } catch {
    return null;
  }
}

function encodedPath(path) {
  return path.split('/').map(encodeURIComponent).join('/');
}

export function expectedIconUrl(source, path) {
  const repository = repositorySlug(source?.repository);
  if (!repository || !/^[0-9a-f]{40}$/.test(source?.commit ?? '')) return null;
  return `https://cdn.jsdelivr.net/gh/${repository}@${source.commit}/${encodedPath(path)}`;
}

export function isApprovedIconUrl(value, path, source) {
  if (typeof value !== 'string' || !isConfinedSvgPath(path)) return false;
  const expected = source ? expectedIconUrl(source, path) : null;
  if (source) return expected !== null && value === expected;
  try {
    const url = new URL(value);
    const match = url.pathname.match(/^\/gh\/[^/]+\/[^/@]+@([0-9a-f]{40})\/(.+)$/);
    return url.protocol === 'https:'
      && url.hostname === 'cdn.jsdelivr.net'
      && !url.username
      && !url.password
      && !url.port
      && !url.search
      && !url.hash
      && match !== null
      && decodeURIComponent(match[2]) === path;
  } catch {
    return false;
  }
}

export function assertApprovedIconSource(value, label = 'Icon', source) {
  if (!isConfinedSvgPath(value?.path)) {
    throw new Error(`${label} must contain a confined SVG path`);
  }
  if (!isApprovedIconUrl(value?.url, value.path, source)) {
    const detail = source ? 'commit-pinned jsDelivr URL matching its candidate path' : 'commit-pinned jsDelivr URL';
    throw new Error(`${label} must use a ${detail}`);
  }
}