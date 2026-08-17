#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import struct
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
    stream_sha256,
)


DEFAULT_MANIFEST = PROJECT_ROOT / 'data' / 'sketchrnn-mvp-manifest.json'
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / 'datasets' / 'quickdraw-sketchrnn'
DEFAULT_OUTPUT = PROJECT_ROOT / '.cache' / 'quickdraw-training'
DEFAULT_WORKER = PROJECT_ROOT / 'scripts' / 'rasterize-stroke3-worker.mjs'
DEFAULT_RASTERIZER_SOURCE = PROJECT_ROOT / 'src' / 'sketch-rasterizer.mjs'
PRODUCTION_MANIFEST_SHA256 = '98de0a2954e9111d8c1015a3b59afecd2b085a53078d53b13427c1a2e0e79544'
SPLIT_ARGUMENTS = {
    'train': 'train_per_class',
    'valid': 'valid_per_class',
    'test': 'test_per_class',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a deterministic sampled raster cache for Quick Draw training.',
    )
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--worker', type=Path, default=DEFAULT_WORKER)
    parser.add_argument('--rasterizer-source', type=Path, default=DEFAULT_RASTERIZER_SOURCE)
    parser.add_argument('--node', default='node')
    parser.add_argument('--train-per-class', type=int, default=3000)
    parser.add_argument('--valid-per-class', type=int, default=500)
    parser.add_argument('--test-per-class', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--padding', type=float, default=4)
    parser.add_argument('--stroke-width', type=float, default=2.5)
    parser.add_argument('--supersample', type=int, default=4)
    parser.add_argument(
        '--allow-untrusted-archives',
        action='store_true',
        help='UNSAFE: allow pickle-backed archives outside the pinned production dataset',
    )
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    for split_name, argument_name in SPLIT_ARGUMENTS.items():
        if getattr(args, argument_name) < 1:
            parser.error(f'--{split_name}-per-class must be positive')
    return args


def load_manifest(path, expected_sha256):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256 or ''):
        raise ValueError('A valid --manifest-sha256 is required')
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f'Manifest checksum mismatch: expected {expected_sha256}, got {actual_sha256}',
        )
    manifest = json.loads(path.read_text(encoding='utf-8'))
    fingerprint = manifest.pop('fingerprint', None)
    if fingerprint != content_fingerprint(manifest):
        raise ValueError(f'Manifest fingerprint mismatch: {path}')
    classes = manifest.get('classes')
    if not isinstance(classes, list) or not classes:
        raise ValueError('Manifest must contain classes')
    manifest['fingerprint'] = fingerprint
    manifest['fileSha256'] = actual_sha256
    return manifest


def split_counts(args):
    return {
        split_name: getattr(args, argument_name)
        for split_name, argument_name in SPLIT_ARGUMENTS.items()
    }


def build_request(args, manifest, class_names):
    builder_path = Path(__file__).resolve()
    integrity_path = PROJECT_ROOT / 'model' / 'integrity.py'
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
    worker_path = args.worker.resolve()
    rasterizer_source_path = args.rasterizer_source.resolve()
    try:
        worker_name = worker_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        worker_name = worker_path.as_posix()
    try:
        rasterizer_source_name = rasterizer_source_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rasterizer_source_name = rasterizer_source_path.as_posix()
    rasterizer = {
        'worker': worker_name,
        'size': args.size,
        'padding': args.padding,
        'strokeWidth': args.stroke_width,
        'supersample': args.supersample,
    }
    return {
        'sourceManifestFingerprint': manifest['fingerprint'],
        'sourceManifestSha256': manifest['fileSha256'],
        'classes': class_names,
        'sampling': {
            'seed': args.seed,
            'perClass': split_counts(args),
        },
        'rasterizer': rasterizer,
        'rasterizerFiles': {
            'worker': {
                'path': worker_name,
                'sha256': file_sha256(worker_path),
            },
            'implementation': {
                'path': rasterizer_source_name,
                'sha256': file_sha256(rasterizer_source_path),
            },
        },
        'builder': {
            'path': builder_path.relative_to(PROJECT_ROOT).as_posix(),
            'sha256': file_sha256(builder_path),
            'dependencies': {
                integrity_path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(integrity_path),
            },
            'runtime': {
                'python': platform.python_version(),
                'numpy': np.__version__,
            },
        },
        'node': {
            'version': node_version,
            'sha256': file_sha256(node_path),
        },
    }


def validate_class_entries(manifest, requested_counts):
    for entry in manifest['classes']:
        class_name = entry.get('name')
        if not isinstance(class_name, str) or not class_name:
            raise ValueError('Every manifest class must have a name')
        for split_name, requested_count in requested_counts.items():
            available = entry.get('splits', {}).get(split_name, {}).get('count')
            if not isinstance(available, int) or available < requested_count:
                raise ValueError(
                    f'{class_name}/{split_name} has {available} samples; '
                    f'{requested_count} requested',
                )


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
        for split in metadata.get('splits', {}).values():
            for artifact in (split.get('images'), split.get('labels')):
                path = resolve_confined_file(output_directory, artifact['filename'])
                if not path.is_file() or path.stat().st_size != artifact['bytes']:
                    return False
                if file_sha256(path) != artifact['sha256']:
                    return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def sample_seed(seed, class_index, split_name):
    digest = hashlib.sha256(f'{seed}:{class_index}:{split_name}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def selected_samples(samples, count, seed):
    random = np.random.default_rng(seed)
    indices = random.choice(len(samples), size=count, replace=False)
    return [np.asarray(samples[index], dtype=np.int16) for index in indices]


def encode_sample(sample):
    if sample.ndim != 2 or sample.shape[1] != 3 or sample.dtype != np.dtype(np.int16):
        raise ValueError('Training samples must be int16 stroke-3 arrays')
    little_endian = sample.astype('<i2', copy=False)
    return struct.pack('<I', len(little_endian)) + little_endian.tobytes(order='C')


def read_exact(source, length):
    chunks = []
    remaining = length
    while remaining > 0:
        chunk = source.read(remaining)
        if not chunk:
            raise RuntimeError('Raster worker ended before producing a complete bitmap')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


class RasterWorker:
    def __init__(self, args):
        command = [
            args.node,
            str(args.worker),
            '--size', str(args.size),
            '--padding', str(args.padding),
            '--stroke-width', str(args.stroke_width),
            '--supersample', str(args.supersample),
        ]
        self.bitmap_bytes = args.size * args.size
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def rasterize(self, samples):
        bitmaps = []
        for sample in samples:
            self.process.stdin.write(encode_sample(sample))
            self.process.stdin.flush()
            bitmaps.append(read_exact(self.process.stdout, self.bitmap_bytes))
        return bitmaps

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


def open_archive(entry, data_directory):
    filename = entry.get('filename')
    if not isinstance(filename, str) or not filename:
        raise ValueError('Every class entry must contain a filename')
    path = resolve_confined_file(data_directory, filename)
    source = path.open('rb')
    try:
        archive_bytes = os.fstat(source.fileno()).st_size
        archive_sha256 = stream_sha256(source)
        if archive_bytes != entry['bytes'] or archive_sha256 != entry['sha256']:
            raise ValueError(f'Archive trust mismatch: {path}')
        source.seek(0)
        return source, np.load(source, allow_pickle=True, encoding='latin1')
    except Exception:
        source.close()
        raise


def build_cache(args, manifest, request, output_directory):
    counts = request['sampling']['perClass']
    class_count = len(request['classes'])
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f'.{output_directory.name}.tmp-',
        dir=output_directory.parent,
    ))
    image_maps = {}
    label_maps = {}
    worker = None
    try:
        for split_name, count in counts.items():
            total = class_count * count
            image_maps[split_name] = np.lib.format.open_memmap(
                temporary_directory / f'{split_name}-images.npy',
                mode='w+',
                dtype=np.uint8,
                shape=(total, args.size, args.size),
            )
            label_maps[split_name] = np.lib.format.open_memmap(
                temporary_directory / f'{split_name}-labels.npy',
                mode='w+',
                dtype=np.int16,
                shape=(total,),
            )

        worker = RasterWorker(args)
        for class_index, entry in enumerate(manifest['classes']):
            source, archive = open_archive(entry, args.data_dir)
            try:
                for split_name, count in counts.items():
                    samples = selected_samples(
                        archive[split_name],
                        count,
                        sample_seed(args.seed, class_index, split_name),
                    )
                    bitmaps = worker.rasterize(samples)
                    start = class_index * count
                    end = start + count
                    image_maps[split_name][start:end] = np.frombuffer(
                        b''.join(bitmaps),
                        dtype=np.uint8,
                    ).reshape(count, args.size, args.size)
                    label_maps[split_name][start:end] = class_index
            finally:
                archive.close()
                source.close()
            print(f"{entry['name']}: rasterized", flush=True)
        worker.close()

        for split_name in counts:
            image_maps[split_name].flush()
            label_maps[split_name].flush()
        image_maps.clear()
        label_maps.clear()

        split_metadata = {}
        for split_name, count in counts.items():
            total = class_count * count
            image_path = temporary_directory / f'{split_name}-images.npy'
            label_path = temporary_directory / f'{split_name}-labels.npy'
            split_metadata[split_name] = {
                'count': total,
                'images': artifact_metadata(
                    image_path,
                    (total, args.size, args.size),
                    np.uint8,
                ),
                'labels': artifact_metadata(label_path, (total,), np.int16),
            }

        metadata = {
            'schemaVersion': 1,
            'request': request,
            'classes': request['classes'],
            'sampling': request['sampling'],
            'rasterizer': request['rasterizer'],
            'splits': split_metadata,
        }
        metadata['fingerprint'] = content_fingerprint(metadata)
        (temporary_directory / 'metadata.json').write_text(
            f'{json.dumps(metadata, indent=2)}\n',
            encoding='utf-8',
        )

        atomic_replace_directory(temporary_directory, output_directory)
    except Exception:
        if worker is not None:
            worker.terminate()
        image_maps.clear()
        label_maps.clear()
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def main():
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.data_dir = args.data_dir.resolve()
    args.worker = args.worker.resolve()
    args.rasterizer_source = args.rasterizer_source.resolve()
    production_inputs = (
        args.manifest == DEFAULT_MANIFEST.resolve()
        and args.data_dir == DEFAULT_DATA_DIRECTORY.resolve()
        and args.manifest_sha256 in (None, PRODUCTION_MANIFEST_SHA256)
    )
    if args.output == DEFAULT_OUTPUT.resolve():
        if args.manifest != DEFAULT_MANIFEST.resolve() or args.data_dir != DEFAULT_DATA_DIRECTORY.resolve():
            raise ValueError('Production cache requires the production manifest and data directory')
        if args.worker != DEFAULT_WORKER.resolve() or args.rasterizer_source != DEFAULT_RASTERIZER_SOURCE.resolve():
            raise ValueError('Production cache requires the canonical rasterizer worker and source')
        if args.manifest_sha256 not in (None, PRODUCTION_MANIFEST_SHA256):
            raise ValueError('Production cache manifest checksum is pinned')
        if args.allow_untrusted_archives:
            raise ValueError('Production cache cannot allow untrusted archives')
        args.manifest_sha256 = PRODUCTION_MANIFEST_SHA256
    elif args.manifest_sha256 is None:
        raise ValueError('--manifest-sha256 is required for a custom training cache')
    if not production_inputs and not args.allow_untrusted_archives:
        raise ValueError('Custom pickle archives require --allow-untrusted-archives')
    manifest = load_manifest(args.manifest, args.manifest_sha256)
    class_names = [entry['name'] for entry in manifest['classes']]
    counts = split_counts(args)
    validate_class_entries(manifest, counts)
    request = build_request(args, manifest, class_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.force and cache_is_current(args.output, request):
        print(f'Training cache is current: {args.output}')
        return
    build_cache(args, manifest, request, args.output)
    print(f'Built training cache: {args.output}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
