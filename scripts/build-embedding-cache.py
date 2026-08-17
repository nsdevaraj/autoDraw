#!/usr/bin/env python3

"""Rasterize the SVGDepot icons that supervise sketch-to-icon retrieval."""

import argparse
import gzip
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ImportError as error:
    raise SystemExit('NumPy is required: python3 -m pip install -r requirements-data.txt') from error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.integrity import (
    atomic_replace_directory,
    content_fingerprint,
    file_sha256,
    resolve_confined_file,
)


DEFAULT_CANDIDATES = PROJECT_ROOT / 'data' / 'quickdraw-candidates.json'
DEFAULT_ICON_MANIFEST = PROJECT_ROOT / 'data' / 'svgdepot-icon-manifest.json.gz'
DEFAULT_STORE = PROJECT_ROOT / '.cache' / 'svgdepot-icons'
DEFAULT_SKETCH_CACHE = PROJECT_ROOT / '.cache' / 'quickdraw-training'
DEFAULT_OUTPUT = PROJECT_ROOT / '.cache' / 'quickdraw-embedding'
DEFAULT_WORKER = PROJECT_ROOT / 'scripts' / 'rasterize-svg-worker.mjs'
DEFAULT_PARSER_SOURCE = PROJECT_ROOT / 'src' / 'svg-to-polylines.mjs'
DEFAULT_RASTERIZER_SOURCE = PROJECT_ROOT / 'src' / 'sketch-rasterizer.mjs'
UNAVAILABLE_DIGEST = -1
STATUS_OK = 0
STATUS_NAMES = ('ok', 'unreadable', 'empty', 'inkOutOfBand')
SPLIT_TRAIN = 0
SPLIT_HOLDOUT = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build the icon-side raster cache for sketch-to-icon retrieval.',
    )
    parser.add_argument('--candidates', type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument('--icon-manifest', type=Path, default=DEFAULT_ICON_MANIFEST)
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--sketch-cache', type=Path, default=DEFAULT_SKETCH_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--worker', type=Path, default=DEFAULT_WORKER)
    parser.add_argument('--node', default='node')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--padding', type=float, default=4)
    parser.add_argument('--stroke-width', type=float, default=2.5)
    parser.add_argument('--supersample', type=int, default=4)
    parser.add_argument('--min-ink', type=float, default=0.01)
    parser.add_argument('--max-ink', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--holdout-per-class', type=int, default=1)
    parser.add_argument('--min-icons-per-class', type=int, default=2)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.size < 8:
        parser.error('--size must be at least 8')
    if args.holdout_per_class < 0:
        parser.error('--holdout-per-class must be non-negative')
    if args.min_icons_per_class < 1:
        parser.error('--min-icons-per-class must be positive')
    return args


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            return json.load(source)
    return json.loads(path.read_text(encoding='utf-8'))


def load_sketch_classes(sketch_cache):
    metadata = read_json(sketch_cache / 'metadata.json')
    fingerprint = metadata.pop('fingerprint', None)
    if fingerprint != content_fingerprint(metadata):
        raise ValueError(f'Sketch cache fingerprint mismatch: {sketch_cache}')
    classes = metadata.get('classes')
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError('Sketch cache must define at least two classes')
    return classes, fingerprint


def load_icon_manifest(path):
    manifest = read_json(path)
    fingerprint = manifest.pop('fingerprint', None)
    if fingerprint != content_fingerprint(manifest):
        raise ValueError(f'Icon manifest fingerprint mismatch: {path}')
    manifest['fingerprint'] = fingerprint
    return manifest


def digest_for_icon(icon_manifest, icon_id):
    icon_digests = icon_manifest['iconDigests']
    if not isinstance(icon_id, int) or icon_id < 0 or icon_id >= len(icon_digests):
        return None
    digest_id = icon_digests[icon_id]
    return None if digest_id == UNAVAILABLE_DIGEST else icon_manifest['digests'][digest_id]


def planned_icons(candidates, icon_manifest, classes):
    """Pair every sketch class with the stored SVG blobs of its candidate icons."""
    if candidates['source']['commit'] != icon_manifest['source']['commit']:
        raise ValueError('Candidate and icon manifests were built from different commits')

    candidates_by_class = {item['name']: item for item in candidates['classes']}
    planned = []
    for class_index, class_name in enumerate(classes):
        entry = candidates_by_class.get(class_name)
        if entry is None:
            raise ValueError(f'Candidate manifest has no class: {class_name}')
        for candidate in entry['candidates']:
            digest = digest_for_icon(icon_manifest, candidate['id'])
            if digest is None:
                continue
            planned.append({
                'classIndex': class_index,
                'className': class_name,
                'iconId': candidate['id'],
                'path': candidate['path'],
                'digest': digest,
            })
    return planned


def assign_splits(icons, seed, holdout_per_class, class_count):
    """Hold out whole icons per class so retrieval can be scored on unseen artwork."""
    splits = np.full(len(icons), SPLIT_TRAIN, dtype=np.int8)
    if holdout_per_class == 0:
        return splits

    random = np.random.default_rng(seed)
    by_class = {}
    for position, icon in enumerate(icons):
        by_class.setdefault(icon['classIndex'], []).append(position)

    for class_index in range(class_count):
        positions = by_class.get(class_index, [])
        # Keep at least two icons in training so every class still forms positive pairs.
        available = len(positions) - 2
        if available < 1:
            continue
        chosen = random.choice(
            np.asarray(positions),
            size=min(holdout_per_class, available),
            replace=False,
        )
        splits[chosen] = SPLIT_HOLDOUT
    return splits


class SvgRasterWorker:
    def __init__(self, args):
        self.bitmap_bytes = args.size * args.size
        self.process = subprocess.Popen(
            [
                args.node, str(args.worker),
                '--size', str(args.size),
                '--padding', str(args.padding),
                '--stroke-width', str(args.stroke_width),
                '--supersample', str(args.supersample),
                '--min-ink', str(args.min_ink),
                '--max-ink', str(args.max_ink),
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


def build_request(args, candidates, icon_manifest, sketch_fingerprint, classes):
    builder_path = Path(__file__).resolve()
    node_executable = shutil.which(args.node)
    if node_executable is None:
        raise ValueError(f'Node executable not found: {args.node}')
    node_path = Path(node_executable).resolve()
    node_version = subprocess.run(
        [str(node_path), '--version'],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    def relative(path):
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    return {
        'classes': classes,
        'sourceSketchCacheFingerprint': sketch_fingerprint,
        'sourceCandidatesFingerprint': candidates['fingerprint'],
        'sourceIconManifestFingerprint': icon_manifest['fingerprint'],
        'source': {
            'repository': icon_manifest['source']['repository'],
            'commit': icon_manifest['source']['commit'],
        },
        'sampling': {'seed': args.seed, 'holdoutPerClass': args.holdout_per_class},
        'rasterizer': {
            'worker': relative(args.worker.resolve()),
            'size': args.size,
            'padding': args.padding,
            'strokeWidth': args.stroke_width,
            'supersample': args.supersample,
            'minInk': args.min_ink,
            'maxInk': args.max_ink,
        },
        'rasterizerFiles': {
            'worker': {
                'path': relative(args.worker.resolve()),
                'sha256': file_sha256(args.worker.resolve()),
            },
            'parser': {
                'path': relative(DEFAULT_PARSER_SOURCE),
                'sha256': file_sha256(DEFAULT_PARSER_SOURCE),
            },
            'implementation': {
                'path': relative(DEFAULT_RASTERIZER_SOURCE),
                'sha256': file_sha256(DEFAULT_RASTERIZER_SOURCE),
            },
        },
        'builder': {
            'path': relative(builder_path),
            'sha256': file_sha256(builder_path),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
        },
        'node': {'version': node_version, 'sha256': file_sha256(node_path)},
    }


def artifact_metadata(path, shape, dtype):
    return {
        'filename': path.name,
        'bytes': path.stat().st_size,
        'sha256': file_sha256(path),
        'shape': list(shape),
        'dtype': np.dtype(dtype).name,
    }


def cache_is_current(output_directory, request):
    metadata_path = output_directory / 'metadata.json'
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('request') != request:
            return False
        fingerprint = metadata.pop('fingerprint', None)
        if fingerprint != content_fingerprint(metadata):
            return False
        for artifact in metadata['artifacts'].values():
            path = resolve_confined_file(output_directory, artifact['filename'])
            if path.stat().st_size != artifact['bytes']:
                return False
            if file_sha256(path) != artifact['sha256']:
                return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def rasterize_icons(args, planned, temporary_directory):
    worker = SvgRasterWorker(args)
    bitmaps = []
    kept = []
    status_counts = {name: 0 for name in STATUS_NAMES}
    try:
        for icon in planned:
            blob = resolve_confined_file(
                args.store,
                f"{icon['digest'][:2]}/{icon['digest']}.svg",
            )
            status, bitmap = worker.rasterize(blob.read_bytes())
            status_counts[STATUS_NAMES[status]] += 1
            if bitmap is None:
                continue
            bitmaps.append(bitmap)
            kept.append(icon)
        worker.close()
    except Exception:
        worker.terminate()
        raise

    images = np.frombuffer(b''.join(bitmaps), dtype=np.uint8).reshape(
        len(kept), args.size, args.size,
    )
    np.save(temporary_directory / 'icon-images.npy', images)
    return kept, status_counts


def build_cache(args, request, planned, class_count, output_directory):
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f'.{output_directory.name}.tmp-',
        dir=output_directory.parent,
    ))
    try:
        kept, status_counts = rasterize_icons(args, planned, temporary_directory)
        if not kept:
            raise ValueError('No icons could be rasterized')

        labels = np.asarray([icon['classIndex'] for icon in kept], dtype=np.int16)
        splits = assign_splits(kept, args.seed, args.holdout_per_class, class_count)
        np.save(temporary_directory / 'icon-labels.npy', labels)
        np.save(temporary_directory / 'icon-splits.npy', splits)
        (temporary_directory / 'icon-paths.json').write_text(
            f"{json.dumps([{'iconId': icon['iconId'], 'path': icon['path']} for icon in kept])}\n",
            encoding='utf-8',
        )

        counts_by_class = np.bincount(labels, minlength=class_count)
        sparse_classes = [
            request['classes'][class_index]
            for class_index, count in enumerate(counts_by_class)
            if count < args.min_icons_per_class
        ]

        artifacts = {
            'images': artifact_metadata(
                temporary_directory / 'icon-images.npy',
                (len(kept), args.size, args.size),
                np.uint8,
            ),
            'labels': artifact_metadata(
                temporary_directory / 'icon-labels.npy', (len(kept),), np.int16,
            ),
            'splits': artifact_metadata(
                temporary_directory / 'icon-splits.npy', (len(kept),), np.int8,
            ),
            'paths': artifact_metadata(
                temporary_directory / 'icon-paths.json', (len(kept),), np.uint8,
            ),
        }
        metadata = {
            'schemaVersion': 1,
            'request': request,
            'classes': request['classes'],
            'counts': {
                'planned': len(planned),
                'kept': len(kept),
                'train': int((splits == SPLIT_TRAIN).sum()),
                'holdout': int((splits == SPLIT_HOLDOUT).sum()),
                'byStatus': status_counts,
                'minPerClass': int(counts_by_class.min()),
                'sparseClasses': sparse_classes,
            },
            'artifacts': artifacts,
        }
        metadata['fingerprint'] = content_fingerprint(metadata)
        (temporary_directory / 'metadata.json').write_text(
            f'{json.dumps(metadata, indent=2)}\n',
            encoding='utf-8',
        )
        atomic_replace_directory(temporary_directory, output_directory)
        return metadata
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def main():
    args = parse_args()
    for name in ('candidates', 'icon_manifest', 'store', 'sketch_cache', 'output', 'worker'):
        setattr(args, name, getattr(args, name).resolve())

    classes, sketch_fingerprint = load_sketch_classes(args.sketch_cache)
    candidates = read_json(args.candidates)
    icon_manifest = load_icon_manifest(args.icon_manifest)
    request = build_request(args, candidates, icon_manifest, sketch_fingerprint, classes)

    if not args.force and cache_is_current(args.output, request):
        print(f'Icon cache is current: {args.output}')
        return

    planned = planned_icons(candidates, icon_manifest, classes)
    if not planned:
        raise ValueError('No candidate icons are present in the local store')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_cache(args, request, planned, len(classes), args.output)
    counts = metadata['counts']

    print(f"Classes: {len(classes)}")
    print(f"Icons planned: {counts['planned']}")
    print(f"Icons kept: {counts['kept']} (train {counts['train']}, holdout {counts['holdout']})")
    print(f"Rasterizer statuses: {counts['byStatus']}")
    print(f"Fewest icons in a class: {counts['minPerClass']}")
    if counts['sparseClasses']:
        print(f"Classes below --min-icons-per-class: {', '.join(counts['sparseClasses'])}")
    print(f"Output: {args.output}")
    print(f"Fingerprint: {metadata['fingerprint']}")


if __name__ == '__main__':
    try:
        main()
    except Exception as error:  # noqa: BLE001 - surfaced as a CLI message
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
