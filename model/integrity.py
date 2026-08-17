"""Shared deterministic hashing helpers for training artifacts."""

import hashlib
import json
import os
from pathlib import Path
import shutil


HASH_BUFFER_SIZE = 1024 * 1024


def stream_sha256(source):
    digest = hashlib.sha256()
    while True:
        chunk = source.read(HASH_BUFFER_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    with path.open('rb') as source:
        return stream_sha256(source)


def content_fingerprint(content):
    encoded = json.dumps(content, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def remove_path(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def atomic_replace_directory(temporary_directory, output_directory):
    """Replace a directory while restoring the previous output on failure."""
    temporary_directory = Path(temporary_directory)
    output_directory = Path(output_directory)
    backup = output_directory.with_name(f'.{output_directory.name}.old-{os.getpid()}')
    remove_path(backup)
    if output_directory.exists() or output_directory.is_symlink():
        os.replace(output_directory, backup)
    try:
        os.replace(temporary_directory, output_directory)
    except Exception:
        if backup.exists() and not output_directory.exists():
            os.replace(backup, output_directory)
        raise
    remove_path(backup)


def resolve_confined_file(root, relative_path):
    """Resolve a relative path while rejecting absolute, traversal, and symlink escapes."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError('Artifact path must be a non-empty string')
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f'Artifact path must be relative: {relative_path}')
    if '..' in relative.parts:
        raise ValueError(f'Artifact path must not contain parent traversal: {relative_path}')
    try:
        root = Path(root).resolve(strict=True)
        candidate = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f'Artifact path cannot be resolved: {relative_path}') from error
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f'Artifact path escapes its root: {relative_path}') from error
    return candidate
