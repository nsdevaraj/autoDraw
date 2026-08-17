#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    import numpy as np
except ImportError as error:
    raise SystemExit('NumPy is required: python3 -m pip install numpy') from error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = PROJECT_ROOT / 'config' / 'mvp-classes.txt'
DEFAULT_OUTPUT = PROJECT_ROOT / 'datasets' / 'quickdraw-sketchrnn'
DEFAULT_MANIFEST = PROJECT_ROOT / 'data' / 'sketchrnn-mvp-manifest.json'
DEFAULT_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn'
EXPECTED_SPLITS = {'train': 70000, 'valid': 2500, 'test': 2500}
PRODUCTION_CLASSES_SHA256 = '03080cf69afaa97d3068628aa1168e38f02ad7fd236f47166cf895e78e6d56d2'
BUFFER_SIZE = 1024 * 1024
MAX_UNVERIFIED_ARCHIVE_BYTES = 128 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download and verify Quick Draw Sketch-RNN stroke-3 archives.',
    )
    class_selectors = parser.add_mutually_exclusive_group()
    class_selectors.add_argument('--classes', type=Path)
    class_selectors.add_argument('--class-name', action='append', dest='class_names')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument(
        '--trust-manifest',
        type=Path,
        help='Manifest providing trusted checksums (defaults to --manifest)',
    )
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--allow-nonstandard-splits', action='store_true')
    parser.add_argument(
        '--allow-unverified-archives',
        action='store_true',
        help='UNSAFE: allow NumPy pickle loading without a trusted manifest checksum',
    )
    args = parser.parse_args()
    args.classes_explicit = args.classes is not None
    if args.classes is None:
        args.classes = DEFAULT_CLASSES
    if args.manifest is None:
        args.manifest = DEFAULT_MANIFEST
    if args.trust_manifest is None:
        args.trust_manifest = args.manifest
    if args.jobs < 1 or args.jobs > 16:
        parser.error('--jobs must be from 1 to 16')
    return args


def parse_classes(args):
    if args.class_names:
        class_names = args.class_names
    else:
        class_names = [
            line.strip()
            for line in args.classes.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    unique_names = []
    seen = set()
    for class_name in class_names:
        if not class_name or class_name in {'.', '..'} or '/' in class_name or '\\' in class_name:
            raise ValueError(f'Invalid class name: {class_name!r}')
        if class_name not in seen:
            seen.add(class_name)
            unique_names.append(class_name)
    if not unique_names:
        raise ValueError('No class names provided')
    return unique_names


def archive_url(base_url, class_name):
    return f"{base_url.rstrip('/')}/{quote(f'{class_name}.npz', safe='')}"


def download_archive(
    base_url,
    class_name,
    output_directory,
    force,
    expected_sha256=None,
    expected_bytes=None,
):
    target = output_directory / f'{class_name}.npz'
    if target.exists() and not force:
        return target, False

    request = Request(
        archive_url(base_url, class_name),
        headers={'User-Agent': 'autoDraw-sketchrnn-fetcher/1'},
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=output_directory,
            prefix=f'.{target.name}.part-',
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with urlopen(request, timeout=120) as response:
                content_length = response.headers.get('Content-Length')
                if expected_bytes is not None and content_length is not None:
                    if int(content_length) != expected_bytes:
                        raise ValueError(
                            f'{class_name}.npz download size mismatch: '
                            f'expected {expected_bytes}, got {content_length}',
                        )
                byte_limit = expected_bytes if expected_bytes is not None else MAX_UNVERIFIED_ARCHIVE_BYTES
                downloaded_bytes = 0
                while True:
                    chunk = response.read(BUFFER_SIZE)
                    if not chunk:
                        break
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > byte_limit:
                        raise ValueError(
                            f'{class_name}.npz download exceeds {byte_limit} bytes',
                        )
                    output.write(chunk)
                if expected_bytes is not None and downloaded_bytes != expected_bytes:
                    raise ValueError(
                        f'{class_name}.npz download size mismatch: '
                        f'expected {expected_bytes}, got {downloaded_bytes}',
                    )
        if expected_sha256 is not None:
            downloaded_sha256 = file_sha256(temporary_path)
            if downloaded_sha256 != expected_sha256:
                raise ValueError(
                    f'{class_name}.npz download checksum mismatch: '
                    f'expected {expected_sha256}, got {downloaded_sha256}',
                )
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return target, True


def stream_sha256(source):
    digest = hashlib.sha256()
    while True:
        chunk = source.read(BUFFER_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    with path.open('rb') as source:
        return stream_sha256(source)


def load_expected_archives(manifest_path, base_url):
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fingerprint = manifest.pop('fingerprint', None)
    fingerprint_source = json.dumps(manifest, separators=(',', ':'), ensure_ascii=False)
    actual_fingerprint = hashlib.sha256(fingerprint_source.encode('utf-8')).hexdigest()
    if fingerprint != actual_fingerprint:
        raise ValueError(f'Trust manifest fingerprint mismatch: {manifest_path}')
    if manifest.get('source', {}).get('baseUrl') != base_url.rstrip('/'):
        return {}
    trusted_archives = {}
    for entry in manifest.get('classes', []):
        class_name = entry.get('name')
        checksum = entry.get('sha256')
        archive_bytes = entry.get('bytes')
        if not isinstance(class_name, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum or ''):
            raise ValueError(f'Trust manifest has invalid SHA-256 for {class_name!s}')
        if not isinstance(archive_bytes, int) or archive_bytes <= 0:
            raise ValueError(f'Trust manifest has invalid byte count for {class_name}')
        trusted_archives[class_name] = {'sha256': checksum, 'bytes': archive_bytes}
    return trusted_archives


def verify_sample(class_name, split_name, sample_index, sample, dtypes, pen_states):
    drawing = np.asarray(sample)
    if drawing.ndim != 2 or drawing.shape[1] != 3:
        raise ValueError(
            f'{class_name} {split_name}[{sample_index}] must have shape (points, 3)',
        )
    if not np.issubdtype(drawing.dtype, np.number):
        raise ValueError(f'{class_name} {split_name}[{sample_index}] must contain numbers')
    if drawing.dtype != np.dtype(np.int16):
        raise ValueError(f'{class_name} {split_name}[{sample_index}] dtype must be int16')
    if drawing.shape[0] == 0:
        raise ValueError(f'{class_name} {split_name}[{sample_index}] must contain at least one point')
    if not np.isfinite(drawing[:, :2]).all():
        raise ValueError(f'{class_name} {split_name}[{sample_index}] has non-finite coordinates')
    if not np.isin(drawing[:, 2], (0, 1)).all():
        raise ValueError(f'{class_name} {split_name}[{sample_index}] pen state must be 0 or 1')
    if int(drawing[-1, 2]) != 1:
        raise ValueError(f'{class_name} {split_name}[{sample_index}] must end with pen state 1')
    dtypes.add(str(drawing.dtype))
    pen_states.update(int(value) for value in np.unique(drawing[:, 2]))
    return int(drawing.shape[0])


def verify_archive(class_name, path, allow_nonstandard_splits, base_url, expected_sha256=None):
    split_stats = {}
    dtypes = set()
    pen_states = set()
    with path.open('rb') as source:
        archive_bytes = os.fstat(source.fileno()).st_size
        archive_sha256 = stream_sha256(source)
        if expected_sha256 is not None and archive_sha256 != expected_sha256:
            raise ValueError(
                f'{class_name}.npz checksum mismatch: expected {expected_sha256}, got {archive_sha256}',
            )
        source.seek(0)
        with np.load(source, allow_pickle=True, encoding='latin1') as archive:
            missing_splits = [name for name in EXPECTED_SPLITS if name not in archive]
            if missing_splits:
                raise ValueError(f"{class_name}.npz is missing splits: {', '.join(missing_splits)}")

            for split_name, expected_count in EXPECTED_SPLITS.items():
                samples = archive[split_name]
                count = len(samples)
                if not allow_nonstandard_splits and count != expected_count:
                    raise ValueError(
                        f'{class_name} {split_name} has {count} samples; expected {expected_count}',
                    )
                lengths = [
                    verify_sample(class_name, split_name, index, sample, dtypes, pen_states)
                    for index, sample in enumerate(samples)
                ]
                split_stats[split_name] = {
                    'count': count,
                    'minSequenceLength': min(lengths, default=0),
                    'maxSequenceLength': max(lengths, default=0),
                }

    return {
        'name': class_name,
        'filename': path.name,
        'url': archive_url(base_url, class_name),
        'bytes': archive_bytes,
        'sha256': archive_sha256,
        'splits': split_stats,
        'dtypes': sorted(dtypes),
        'penStates': sorted(pen_states),
    }


def write_manifest(path, base_url, entries):
    content = {
        'schemaVersion': 1,
        'source': {
            'dataset': 'Quick, Draw! Sketch-RNN',
            'creator': 'Google, Inc.',
            'attribution': 'Quick, Draw! Dataset by Google, Inc., licensed under CC BY 4.0.',
            'baseUrl': base_url.rstrip('/'),
            'format': 'Sketch-RNN stroke-3 NPZ',
            'license': 'CC BY 4.0',
            'licenseUrl': 'https://creativecommons.org/licenses/by/4.0/',
            'documentation': 'https://github.com/googlecreativelab/quickdraw-dataset',
            'expectedSplits': EXPECTED_SPLITS,
        },
        'classes': entries,
    }
    fingerprint_source = json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    manifest = {
        **content,
        'fingerprint': hashlib.sha256(fingerprint_source.encode('utf-8')).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.tmp-',
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(f'{json.dumps(manifest, indent=2)}\n')
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def validate_production_run(args):
    if args.allow_unverified_archives and args.output.resolve() == DEFAULT_OUTPUT.resolve():
        raise ValueError('unsafe archive loading cannot use the production archive cache')
    if args.manifest.resolve() != DEFAULT_MANIFEST.resolve():
        return
    if args.allow_unverified_archives:
        raise ValueError('unsafe archive loading cannot write the production manifest')
    if args.classes_explicit:
        raise ValueError('--classes cannot write the full production manifest')
    if args.class_names:
        raise ValueError('--class-name cannot write the full production manifest')
    if args.base_url.rstrip('/') != DEFAULT_BASE_URL:
        raise ValueError('production manifest requires the official base URL')
    if args.trust_manifest.resolve() != DEFAULT_MANIFEST.resolve():
        raise ValueError('production manifest requires the production trust manifest')
    if args.allow_nonstandard_splits:
        raise ValueError('production manifest requires standard split sizes')
    if file_sha256(args.classes) != PRODUCTION_CLASSES_SHA256:
        raise ValueError('MVP class-list fingerprint mismatch')


def main():
    args = parse_args()
    class_names = parse_classes(args)
    validate_production_run(args)
    trusted_archives = load_expected_archives(args.trust_manifest, args.base_url)
    if not args.allow_unverified_archives:
        untrusted = [name for name in class_names if name not in trusted_archives]
        if untrusted:
            raise ValueError(
                f"No trusted checksum for {untrusted[0]}; provide a trusted manifest or use "
                '--allow-unverified-archives for disposable fixtures only',
            )
    args.output.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        missing = [name for name in class_names if not (args.output / f'{name}.npz').is_file()]
        if missing:
            raise FileNotFoundError(f"Missing archives: {', '.join(missing)}")
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    download_archive,
                    args.base_url,
                    class_name,
                    args.output,
                    args.force,
                    trusted_archives.get(class_name, {}).get('sha256'),
                    trusted_archives.get(class_name, {}).get('bytes'),
                ): class_name
                for class_name in class_names
            }
            for future in as_completed(futures):
                class_name = futures[future]
                _, downloaded = future.result()
                print(f"{class_name}: {'downloaded' if downloaded else 'cached'}", flush=True)

    entries = []
    for class_name in class_names:
        archive_path = args.output / f'{class_name}.npz'
        print(f'{class_name}: verifying', flush=True)
        entry = verify_archive(
            class_name,
            archive_path,
            args.allow_nonstandard_splits,
            args.base_url,
            trusted_archives.get(class_name, {}).get('sha256'),
        )
        entries.append(entry)

    write_manifest(args.manifest, args.base_url, entries)
    total_bytes = sum(entry['bytes'] for entry in entries)
    print(
        f'Verified {len(entries)} classes, {total_bytes} bytes; manifest: {args.manifest}',
        flush=True,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
