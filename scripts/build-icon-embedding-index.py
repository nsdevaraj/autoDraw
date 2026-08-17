#!/usr/bin/env python3

"""Encode every SVGDepot icon into the sketch embedding space and shard it for the browser.

The encoder is never retrained per icon. It maps a 64x64 stroke bitmap to a unit vector,
so corpus coverage is a property of this index rather than of the model weights, and the
~206k icons that carry no class supervision are reached zero-shot on shape alone.
"""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

try:
    import numpy as np
except ImportError as error:
    raise SystemExit('NumPy is required: python3 -m pip install -r requirements-data.txt') from error

try:
    import onnxruntime
except ImportError as error:
    raise SystemExit(
        'onnxruntime is required: python3 -m pip install -r requirements-model.txt',
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.integrity import (
    atomic_replace_directory,
    content_fingerprint,
    file_sha256,
    resolve_confined_file,
)

from scripts.lib.svgdepot_index import icon_paths_from_index


DEFAULT_INDEX = PROJECT_ROOT / 'data' / 'svgdepot-index.json.gz'
DEFAULT_ICON_MANIFEST = PROJECT_ROOT / 'data' / 'svgdepot-icon-manifest.json.gz'
DEFAULT_STORE = PROJECT_ROOT / '.cache' / 'svgdepot-icons'
DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / 'models' / 'sketch-embedder'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'icon-embeddings'
DEFAULT_RASTER_CACHE = PROJECT_ROOT / '.cache' / 'icon-rasters'
DEFAULT_WORKER = PROJECT_ROOT / 'scripts' / 'rasterize-svg-worker.mjs'
DEFAULT_PARSER_SOURCE = PROJECT_ROOT / 'src' / 'svg-to-polylines.mjs'
DEFAULT_RASTERIZER_SOURCE = PROJECT_ROOT / 'src' / 'sketch-rasterizer.mjs'

INDEX_SCHEMA_VERSION = 1
UNAVAILABLE_DIGEST = -1
STATUS_OK = 0
STATUS_NAMES = ('ok', 'unreadable', 'empty', 'inkOutOfBand')
QUANTIZED_MAXIMUM = 127.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build the sharded icon embedding index used for browser retrieval.',
    )
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--icon-manifest', type=Path, default=DEFAULT_ICON_MANIFEST)
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--model-directory', type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--raster-cache', type=Path, default=DEFAULT_RASTER_CACHE)
    parser.add_argument('--worker', type=Path, default=DEFAULT_WORKER)
    parser.add_argument('--node', default='node')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--clusters', type=int, default=256)
    parser.add_argument('--kmeans-iterations', type=int, default=25)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260817)
    parser.add_argument('--force-rasters', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 32:
        parser.error('--jobs must be from 1 to 32')
    if args.clusters < 2:
        parser.error('--clusters must be at least 2')
    if args.batch < 1:
        parser.error('--batch must be positive')
    return args


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            return json.load(source)
    return json.loads(path.read_text(encoding='utf-8'))


def write_gzip_json(path, value):
    payload = json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    # mtime=0 keeps the gzip container byte-identical across rebuilds.
    with gzip.GzipFile(filename='', mode='wb', fileobj=path.open('wb'), mtime=0) as target:
        target.write(payload)


def load_icon_manifest(path):
    manifest = read_json(path)
    fingerprint = manifest.pop('fingerprint', None)
    if fingerprint != content_fingerprint(manifest):
        raise ValueError(f'Icon manifest fingerprint mismatch: {path}')
    manifest['fingerprint'] = fingerprint
    return manifest


def load_embedder_metadata(model_directory):
    metadata = read_json(model_directory / 'model.json')
    fingerprint = metadata.pop('fingerprint', None)
    if fingerprint != content_fingerprint(metadata):
        raise ValueError('Embedder metadata fingerprint mismatch')
    metadata['fingerprint'] = fingerprint
    if metadata.get('kind') != 'sketch-embedder':
        raise ValueError('Model metadata is not a sketch embedder')

    model_path = resolve_confined_file(model_directory, metadata['model']['filename'])
    if file_sha256(model_path) != metadata['model']['sha256']:
        raise ValueError('Embedder ONNX file does not match its recorded hash')
    return metadata, model_path


def group_icons_by_digest(icons, icon_manifest):
    """Collapse byte-identical icons so each distinct SVG is rasterized and encoded once."""
    icon_digests = icon_manifest['iconDigests']
    digests = icon_manifest['digests']
    if len(icon_digests) != len(icons):
        raise ValueError('Icon manifest does not cover the index')

    grouped = {}
    available = 0
    for icon_id, digest_id in enumerate(icon_digests):
        if digest_id == UNAVAILABLE_DIGEST:
            continue
        available += 1
        grouped.setdefault(digests[digest_id], []).append(icon_id)
    # Sorting by digest makes the rasterization order independent of manifest ordering.
    return sorted(grouped.items()), available


class SvgRasterWorker:
    """One `rasterize-svg-worker.mjs` subprocess speaking the length-prefixed byte protocol."""

    def __init__(self, args, rasterizer):
        self.bitmap_bytes = rasterizer['size'] * rasterizer['size']
        self.process = subprocess.Popen(
            [
                args.node, str(args.worker),
                '--size', str(rasterizer['size']),
                '--padding', str(rasterizer['padding']),
                '--stroke-width', str(rasterizer['strokeWidth']),
                '--supersample', str(rasterizer['supersample']),
                '--min-ink', str(rasterizer['minInk']),
                '--max-ink', str(rasterizer['maxInk']),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def rasterize(self, source_bytes):
        self.process.stdin.write(len(source_bytes).to_bytes(4, 'little'))
        self.process.stdin.write(source_bytes)
        self.process.stdin.flush()
        status = self._read_exact(1)[0]
        if status != STATUS_OK:
            return status, None
        return status, self._read_exact(self.bitmap_bytes)

    def _read_exact(self, length):
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                stderr = self.process.stderr.read().decode('utf-8', errors='replace')
                raise RuntimeError(f'Raster worker ended early: {stderr.strip()}')
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def close(self):
        self.process.stdin.close()
        status = self.process.wait()
        stderr = self.process.stderr.read().decode('utf-8', errors='replace')
        if status != 0:
            raise RuntimeError(f'Raster worker failed: {stderr.strip()}')

    def terminate(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def rasterize_partition(args, rasterizer, store, entries, part_path, statuses, offset):
    """Rasterize a contiguous slice of digests into its own part file."""
    worker = SvgRasterWorker(args, rasterizer)
    try:
        with part_path.open('wb') as target:
            for position, (digest, _icon_ids) in enumerate(entries):
                blob = resolve_confined_file(store, f'{digest[:2]}/{digest}.svg')
                status, bitmap = worker.rasterize(blob.read_bytes())
                statuses[offset + position] = status
                if bitmap is not None:
                    target.write(bitmap)
        worker.close()
    except BaseException:
        worker.terminate()
        raise


def rasterize_all(args, rasterizer, grouped, raster_cache):
    """Fan out across worker subprocesses, then merge parts in order and dedupe by bitmap."""
    raster_cache.mkdir(parents=True, exist_ok=True)
    bitmap_bytes = rasterizer['size'] * rasterizer['size']
    statuses = [None] * len(grouped)
    jobs = min(args.jobs, max(1, len(grouped)))
    bounds = [len(grouped) * index // jobs for index in range(jobs + 1)]

    part_paths = [raster_cache / f'part-{index}.bin' for index in range(jobs)]
    failures = []

    def run(index):
        try:
            rasterize_partition(
                args, rasterizer, args.store,
                grouped[bounds[index]:bounds[index + 1]],
                part_paths[index], statuses, bounds[index],
            )
        except BaseException as error:  # noqa: BLE001 - surfaced after every thread joins
            failures.append(error)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]

    return merge_rasters(grouped, statuses, part_paths, bounds, raster_cache, bitmap_bytes)


def merge_rasters(grouped, statuses, part_paths, bounds, raster_cache, bitmap_bytes):
    """Stream the parts into one file, keeping the first icon of each identical bitmap."""
    vectors_path = raster_cache / 'rasters.bin'
    status_counts = {name: 0 for name in STATUS_NAMES}
    seen = {}
    members = []

    with vectors_path.open('wb') as target:
        for part_index, part_path in enumerate(part_paths):
            with part_path.open('rb') as source:
                for position in range(bounds[part_index], bounds[part_index + 1]):
                    status = statuses[position]
                    if status is None:
                        raise RuntimeError(f'Digest {position} was never rasterized')
                    status_counts[STATUS_NAMES[status]] += 1
                    if status != STATUS_OK:
                        continue

                    bitmap = source.read(bitmap_bytes)
                    if len(bitmap) != bitmap_bytes:
                        raise RuntimeError(f'Raster part {part_index} ended early')
                    key = hashlib.sha256(bitmap).digest()
                    existing = seen.get(key)
                    if existing is None:
                        seen[key] = len(members)
                        members.append(list(grouped[position][1]))
                        target.write(bitmap)
                    else:
                        members[existing].extend(grouped[position][1])
                if source.read(1):
                    raise RuntimeError(f'Raster part {part_index} has trailing data')

    for part_path in part_paths:
        part_path.unlink(missing_ok=True)
    return vectors_path, members, status_counts


def encode_rasters(model_path, vectors_path, count, rasterizer, batch_size):
    """Run the tracked ONNX encoder over every unique bitmap."""
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=['CPUExecutionProvider'],
    )
    size = rasterizer['size']
    images = np.memmap(vectors_path, dtype=np.uint8, mode='r', shape=(count, 1, size, size))
    embeddings = None
    aux_classes = np.zeros(count, dtype=np.int32)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = np.ascontiguousarray(images[start:stop], dtype=np.float32) / 255.0
        embedding, logits = session.run(['embedding', 'logits'], {'bitmap': batch})
        if embeddings is None:
            embeddings = np.zeros((count, embedding.shape[1]), dtype=np.float32)
        embeddings[start:stop] = embedding
        aux_classes[start:stop] = np.argmax(logits, axis=1)
        if stop % (batch_size * 100) == 0 or stop == count:
            print(f'Encoded {stop:,}/{count:,} icons')

    del images
    return embeddings, aux_classes


def spherical_kmeans(embeddings, clusters, iterations, seed):
    """Deterministic k-means++ on the unit sphere, matching the dot-product metric."""
    random = np.random.default_rng(seed)
    count = embeddings.shape[0]
    centroids = np.zeros((clusters, embeddings.shape[1]), dtype=np.float32)
    centroids[0] = embeddings[random.integers(count)]

    closest = embeddings @ centroids[0]
    for index in range(1, clusters):
        # k-means++ over the chordal distance, which is monotone in the dot product.
        weights = np.maximum(1.0 - closest, 0.0).astype(np.float64)
        total = weights.sum()
        chosen = random.integers(count) if total <= 0 else int(
            np.searchsorted(np.cumsum(weights), random.random() * total),
        )
        centroids[index] = embeddings[min(chosen, count - 1)]
        closest = np.maximum(closest, embeddings @ centroids[index])

    assignments = np.zeros(count, dtype=np.int32)
    for iteration in range(iterations):
        previous = assignments
        assignments = assign_clusters(embeddings, centroids)
        sums = np.zeros_like(centroids)
        np.add.at(sums, assignments, embeddings)
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        # An emptied cluster keeps its previous centroid instead of collapsing to zero.
        centroids = np.where(norms > 0, sums / np.maximum(norms, 1e-12), centroids)
        if iteration > 0 and np.array_equal(previous, assignments):
            break

    return centroids, assign_clusters(embeddings, centroids)


def assign_clusters(embeddings, centroids, chunk=16384):
    assignments = np.zeros(embeddings.shape[0], dtype=np.int32)
    for start in range(0, embeddings.shape[0], chunk):
        stop = min(start + chunk, embeddings.shape[0])
        assignments[start:stop] = np.argmax(embeddings[start:stop] @ centroids.T, axis=1)
    return assignments


def quantize(vectors):
    """int8 codes with a per-vector scale, so short vectors keep their full resolution."""
    peaks = np.max(np.abs(vectors), axis=1)
    scales = np.where(peaks > 0, peaks / QUANTIZED_MAXIMUM, 1.0).astype(np.float32)
    codes = np.rint(vectors / scales[:, None]).clip(-127, 127).astype(np.int8)
    return codes, scales


def build_shards(output_directory, embeddings, aux_classes, assignments, members, icons, clusters):
    shards = []
    for cluster in range(clusters):
        positions = np.flatnonzero(assignments == cluster)
        codes, scales = quantize(embeddings[positions])

        binary_path = output_directory / f'shard-{cluster}.bin'
        with binary_path.open('wb') as target:
            target.write(codes.tobytes())
            target.write(scales.astype('<f4').tobytes())

        entries = []
        for position in positions:
            representative = min(members[position])
            entries.append([
                int(representative),
                icons[representative]['path'],
                len(members[position]),
                int(aux_classes[position]),
            ])

        meta_path = output_directory / f'shard-{cluster}.meta.json.gz'
        write_gzip_json(meta_path, {
            'shard': cluster,
            'count': int(positions.size),
            # Icon URLs are deliberately absent: the browser rebuilds them from the pinned
            # commit through expectedIconUrl(), so an unapproved URL cannot be represented.
            'icons': entries,
        })

        shards.append({
            'id': cluster,
            'count': int(positions.size),
            'bytes': binary_path.stat().st_size,
            'sha256': file_sha256(binary_path),
            'metaBytes': meta_path.stat().st_size,
            'metaSha256': file_sha256(meta_path),
        })
    return shards


def build_provenance(args, node_version):
    def relative(path):
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    builder_path = Path(__file__).resolve()
    return {
        'builder': {
            'path': relative(builder_path),
            'sha256': file_sha256(builder_path),
            'runtime': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'onnxruntime': onnxruntime.__version__,
            },
        },
        'rasterizerFiles': {
            'worker': {'path': relative(args.worker), 'sha256': file_sha256(args.worker)},
            'parser': {
                'path': relative(DEFAULT_PARSER_SOURCE),
                'sha256': file_sha256(DEFAULT_PARSER_SOURCE),
            },
            'implementation': {
                'path': relative(DEFAULT_RASTERIZER_SOURCE),
                'sha256': file_sha256(DEFAULT_RASTERIZER_SOURCE),
            },
        },
        'node': {'version': node_version},
    }


def main():
    args = parse_args()
    index = read_json(args.index)
    icons = icon_paths_from_index(index)
    icon_manifest = load_icon_manifest(args.icon_manifest)
    metadata, model_path = load_embedder_metadata(args.model_directory)

    if icon_manifest['source']['commit'] != index['source']['commit']:
        raise ValueError('Icon manifest and index were built from different commits')

    rasterizer = metadata['rasterizer']
    grouped, available = group_icons_by_digest(icons, icon_manifest)
    print(f'Icons: {len(icons):,} ({available:,} available, {len(grouped):,} unique files)')

    node_version = subprocess.run(
        [args.node, '--version'], capture_output=True, check=True, text=True,
    ).stdout.strip()

    raster_cache = args.raster_cache
    if args.force_rasters and raster_cache.exists():
        shutil.rmtree(raster_cache)
    vectors_path, members, status_counts = rasterize_all(args, rasterizer, grouped, raster_cache)
    print(f'Rasterized: {json.dumps(status_counts)}')
    print(f'Unique bitmaps: {len(members):,}')

    embeddings, aux_classes = encode_rasters(
        model_path, vectors_path, len(members), rasterizer, args.batch,
    )
    clusters = min(args.clusters, len(members))
    centroids, assignments = spherical_kmeans(
        embeddings, clusters, args.kmeans_iterations, args.seed,
    )
    centroid_codes, centroid_scales = quantize(centroids)

    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f'.{args.output.name}.tmp-', dir=args.output.parent,
    ))
    try:
        shards = build_shards(
            temporary_directory, embeddings, aux_classes, assignments, members, icons, clusters,
        )
        document = {
            'schemaVersion': INDEX_SCHEMA_VERSION,
            'kind': 'icon-embedding-index',
            'source': {
                'repository': index['source']['repository'],
                'commit': index['source']['commit'],
            },
            'model': {
                'filename': metadata['model']['filename'],
                'sha256': metadata['model']['sha256'],
                'fingerprint': metadata['fingerprint'],
            },
            'classes': metadata['classes'],
            'rasterizer': rasterizer,
            'embedding': {
                'dim': int(embeddings.shape[1]),
                'dtype': 'int8',
                'scale': 'per-vector-float32',
                'metric': 'dot',
            },
            'counts': {
                'icons': len(icons),
                'available': available,
                'uniqueFiles': len(grouped),
                'vectors': len(members),
                'byStatus': status_counts,
            },
            'clustering': {
                'count': clusters,
                'seed': args.seed,
                'iterations': args.kmeans_iterations,
                'centroidCodes': base64.b64encode(centroid_codes.tobytes()).decode('ascii'),
                'centroidScales': [float(value) for value in centroid_scales],
            },
            'shards': shards,
            'provenance': build_provenance(args, node_version),
            'sourceIconManifestFingerprint': icon_manifest['fingerprint'],
        }
        document['fingerprint'] = content_fingerprint(document)
        (temporary_directory / 'index.json').write_text(
            json.dumps(document, separators=(',', ':'), ensure_ascii=False),
            encoding='utf-8',
        )
        atomic_replace_directory(temporary_directory, args.output)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    index_bytes = (args.output / 'index.json').stat().st_size
    shard_bytes = sum(shard['bytes'] + shard['metaBytes'] for shard in shards)
    print(f'Clusters: {clusters}')
    print(f'index.json: {index_bytes:,} bytes')
    print(f'Shards: {len(shards)} ({shard_bytes:,} bytes total)')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
