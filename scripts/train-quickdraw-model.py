#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.quickdraw_cnn import create_model
from model.integrity import (
    atomic_replace_directory,
    content_fingerprint,
    file_sha256,
    resolve_confined_file,
)


DEFAULT_CACHE = PROJECT_ROOT / '.cache' / 'quickdraw-training'
DEFAULT_OUTPUT = PROJECT_ROOT / 'models' / 'quickdraw-mvp'
MODEL_FILENAME = 'quickdraw-mvp.onnx'
METADATA_FILENAME = 'model.json'
ONNX_OPSET = 17
MINIMUM_RELEASE_ACCURACY = 0.70


def is_release_threshold(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and MINIMUM_RELEASE_ACCURACY <= value <= 1
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and export the MVP Quick Draw CNN as ONNX.',
    )
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--min-test-accuracy', type=float, default=0.70)
    parser.add_argument('--min-class-accuracy', type=float, default=0.70)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be positive')
    if not is_release_threshold(args.min_test_accuracy):
        parser.error('--min-test-accuracy must be from 0.70 to 1')
    if not is_release_threshold(args.min_class_accuracy):
        parser.error('--min-class-accuracy must be from 0.70 to 1')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    return args


def load_cache_metadata(cache_directory):
    path = cache_directory / 'metadata.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    fingerprint = metadata.pop('fingerprint', None)
    if fingerprint != content_fingerprint(metadata):
        raise ValueError(f'Training cache fingerprint mismatch: {path}')
    metadata['fingerprint'] = fingerprint
    classes = metadata.get('classes')
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Training cache must contain at least two unique classes')
    rasterizer = metadata.get('rasterizer', {})
    if rasterizer.get('size') != 64:
        raise ValueError('Training cache raster size must be 64')

    for split_name in ('train', 'valid', 'test'):
        split = metadata.get('splits', {}).get(split_name)
        if not isinstance(split, dict) or split.get('count', 0) < 1:
            raise ValueError(f'Training cache is missing split: {split_name}')
        for artifact_name in ('images', 'labels'):
            artifact = split.get(artifact_name)
            if not isinstance(artifact, dict):
                raise ValueError(f'Training cache is missing {split_name}/{artifact_name}')
            artifact_path = resolve_confined_file(cache_directory, artifact['filename'])
            if artifact_path.stat().st_size != artifact['bytes']:
                raise ValueError(f'Training cache size mismatch: {artifact_path}')
            if file_sha256(artifact_path) != artifact['sha256']:
                raise ValueError(f'Training cache checksum mismatch: {artifact_path}')
    return metadata


class NpyBitmapDataset(Dataset):
    def __init__(self, cache_directory, split_metadata):
        self.images = np.load(
            resolve_confined_file(cache_directory, split_metadata['images']['filename']),
            mmap_mode='r',
        )
        self.labels = np.load(
            resolve_confined_file(cache_directory, split_metadata['labels']['filename']),
            mmap_mode='r',
        )
        if self.images.dtype != np.uint8 or self.images.ndim != 3 or self.images.shape[1:] != (64, 64):
            raise ValueError('Image cache must have uint8 shape [samples, 64, 64]')
        if self.labels.dtype != np.int16 or self.labels.shape != (len(self.images),):
            raise ValueError('Label cache must have int16 shape [samples]')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        label = int(self.labels[index])
        return image.unsqueeze(0).float().div_(255), label


def select_device(requested):
    if requested == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is not available')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available')
    return torch.device(requested)


def set_training_seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def empty_class_metrics(class_count):
    return {
        'correct': np.zeros(class_count, dtype=np.int64),
        'total': np.zeros(class_count, dtype=np.int64),
    }


def finalize_metrics(loss_sum, sample_count, top1_correct, top5_correct, class_metrics):
    per_class = []
    for correct, total in zip(class_metrics['correct'], class_metrics['total']):
        per_class.append(float(correct / total) if total else 0.0)
    return {
        'loss': float(loss_sum / sample_count),
        'accuracy': float(top1_correct / sample_count),
        'top5Accuracy': float(top5_correct / sample_count),
        'perClassAccuracy': per_class,
    }


def evaluate_torch(model, loader, device, class_count):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_sum = 0.0
    sample_count = 0
    top1_correct = 0
    top5_correct = 0
    class_metrics = empty_class_metrics(class_count)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            predictions = logits.argmax(dim=1)
            top_count = min(5, class_count)
            top_predictions = logits.topk(top_count, dim=1).indices
            top1_correct += (predictions == labels).sum().item()
            top5_correct += (top_predictions == labels.unsqueeze(1)).any(dim=1).sum().item()
            sample_count += len(labels)
            labels_cpu = labels.cpu().numpy()
            predictions_cpu = predictions.cpu().numpy()
            np.add.at(class_metrics['total'], labels_cpu, 1)
            np.add.at(class_metrics['correct'], labels_cpu, predictions_cpu == labels_cpu)
    return finalize_metrics(
        loss_sum,
        sample_count,
        top1_correct,
        top5_correct,
        class_metrics,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    sample_count = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        sample_count += len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return {
        'loss': float(loss_sum / sample_count),
        'accuracy': float(correct / sample_count),
        'top5Accuracy': None,
        'perClassAccuracy': None,
    }


def evaluate_onnx(session, images, labels, batch_size, class_count, torch_model):
    loss_sum = 0.0
    sample_count = 0
    top1_correct = 0
    top5_correct = 0
    max_logit_difference = 0.0
    class_metrics = empty_class_metrics(class_count)
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            batch_images = np.asarray(images[start:end], dtype=np.float32)[:, None, :, :] / 255
            batch_labels = np.asarray(labels[start:end], dtype=np.int64)
            logits = session.run(['logits'], {'bitmap': batch_images})[0]
            torch_logits = torch_model(torch.from_numpy(batch_images)).numpy()
            max_logit_difference = max(
                max_logit_difference,
                float(np.max(np.abs(torch_logits - logits))),
            )
            logits_shifted = logits - logits.max(axis=1, keepdims=True)
            log_probabilities = logits_shifted - np.log(np.exp(logits_shifted).sum(axis=1, keepdims=True))
            loss_sum -= log_probabilities[np.arange(len(batch_labels)), batch_labels].sum()
            predictions = logits.argmax(axis=1)
            top_count = min(5, class_count)
            top_predictions = np.argpartition(logits, -top_count, axis=1)[:, -top_count:]
            top1_correct += int((predictions == batch_labels).sum())
            top5_correct += int((top_predictions == batch_labels[:, None]).any(axis=1).sum())
            sample_count += len(batch_labels)
            np.add.at(class_metrics['total'], batch_labels, 1)
            np.add.at(class_metrics['correct'], batch_labels, predictions == batch_labels)
    return (
        finalize_metrics(
            float(loss_sum),
            sample_count,
            top1_correct,
            top5_correct,
            class_metrics,
        ),
        max_logit_difference,
    )


def enforce_accuracy_thresholds(
    runtime_name,
    metrics,
    minimum_accuracy,
    minimum_per_class_accuracy,
    class_names,
):
    if not is_release_threshold(minimum_accuracy):
        raise ValueError('Minimum accuracy cannot be below the 0.70 release floor')
    if not is_release_threshold(minimum_per_class_accuracy):
        raise ValueError('Minimum per-class accuracy cannot be below the 0.70 release floor')
    if metrics['accuracy'] < minimum_accuracy:
        raise ValueError(
            f"{runtime_name} accuracy {metrics['accuracy']:.4f} is below "
            f'{minimum_accuracy:.4f}',
        )
    per_class = metrics['perClassAccuracy']
    if len(per_class) != len(class_names):
        raise ValueError(f'{runtime_name} per-class metric length mismatch')
    lowest_index = min(range(len(per_class)), key=per_class.__getitem__)
    if per_class[lowest_index] < minimum_per_class_accuracy:
        raise ValueError(
            f'{runtime_name} per-class accuracy for {class_names[lowest_index]} '
            f'{per_class[lowest_index]:.4f} is below {minimum_per_class_accuracy:.4f}',
        )


def git_output(arguments):
    result = subprocess.run(
        ['git', *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def source_file_hashes():
    paths = (
        PROJECT_ROOT / 'model' / '__init__.py',
        PROJECT_ROOT / 'model' / 'quickdraw_cnn.py',
        PROJECT_ROOT / 'model' / 'integrity.py',
        PROJECT_ROOT / 'scripts' / 'build-training-cache.py',
        PROJECT_ROOT / 'scripts' / 'rasterize-stroke3-worker.mjs',
        PROJECT_ROOT / 'scripts' / 'train-quickdraw-model.py',
        PROJECT_ROOT / 'src' / 'sketch-rasterizer.mjs',
        PROJECT_ROOT / 'package.json',
        PROJECT_ROOT / 'package-lock.json',
        PROJECT_ROOT / 'requirements-data.txt',
        PROJECT_ROOT / 'requirements-model.txt',
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }


def repository_provenance(source_hashes):
    shallow_output = git_output(['rev-parse', '--is-shallow-repository'])
    is_shallow = shallow_output is not None and shallow_output.strip() == b'true'
    revision_output = None if is_shallow else git_output([
        'log',
        '-1',
        '--format=%H',
        '--',
        *source_hashes,
    ])
    revision = revision_output.decode('utf-8').strip() if revision_output else None
    files_match_revision = revision is not None
    if revision is not None:
        for path, expected_hash in source_hashes.items():
            contents = git_output(['show', f'{revision}:{path}'])
            if contents is None or hashlib.sha256(contents).hexdigest() != expected_hash:
                files_match_revision = False
                break
    if files_match_revision:
        for arguments in (
            ['diff', '--cached', '--quiet', revision, '--', *source_hashes],
            ['diff', '--quiet', '--', *source_hashes],
        ):
            difference = subprocess.run(
                ['git', *arguments],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if difference.returncode != 0:
                files_match_revision = False
                break
    if files_match_revision:
        for path in source_hashes:
            tree_entry = git_output(['ls-tree', revision, '--', path])
            if not tree_entry:
                files_match_revision = False
                break
            expected_mode = tree_entry.split(maxsplit=1)[0]
            if expected_mode not in (b'100644', b'100755'):
                files_match_revision = False
                break
            try:
                actual_mode = (PROJECT_ROOT / path).lstat().st_mode
            except OSError:
                files_match_revision = False
                break
            expected_executable = expected_mode == b'100755'
            actual_executable = bool(actual_mode & 0o111)
            if not stat.S_ISREG(actual_mode) or actual_executable != expected_executable:
                files_match_revision = False
                break
    return {
        'repositoryRevision': revision,
        'filesMatchRevision': files_match_revision,
        'files': source_hashes,
    }


def browser_runtime_metadata():
    package = json.loads((PROJECT_ROOT / 'package.json').read_text(encoding='utf-8'))
    return {
        'package': 'onnxruntime-web',
        'version': package['devDependencies']['onnxruntime-web'],
        'lockfileSha256': file_sha256(PROJECT_ROOT / 'package-lock.json'),
    }


def export_onnx(model, output_path):
    example = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=['bitmap'],
        output_names=['logits'],
        dynamic_axes={'bitmap': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported, full_check=True)


def train(args):
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    source_hashes = source_file_hashes()
    source_provenance = repository_provenance(source_hashes)
    if args.output == DEFAULT_OUTPUT.resolve() and not source_provenance['filesMatchRevision']:
        raise ValueError('Production model source files must match the recorded Git revision')
    cache_metadata = load_cache_metadata(args.cache)
    class_names = cache_metadata['classes']
    class_count = len(class_names)
    device = select_device(args.device)
    set_training_seed(args.seed)

    datasets = {
        split_name: NpyBitmapDataset(args.cache, cache_metadata['splits'][split_name])
        for split_name in ('train', 'valid', 'test')
    }
    loaders = {
        'train': make_loader(
            datasets['train'],
            args.batch_size,
            True,
            args.seed,
            args.num_workers,
        ),
        'valid': make_loader(
            datasets['valid'],
            args.batch_size,
            False,
            args.seed,
            args.num_workers,
        ),
        'test': make_loader(
            datasets['test'],
            args.batch_size,
            False,
            args.seed,
            args.num_workers,
        ),
    }

    model = create_model(num_classes=class_count, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    best_state = None
    best_validation_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        training_metrics = train_epoch(model, loaders['train'], optimizer, device)
        validation_metrics = evaluate_torch(model, loaders['valid'], device, class_count)
        history.append({
            'epoch': epoch,
            'train': training_metrics,
            'valid': validation_metrics,
        })
        print(
            f"epoch={epoch} train_loss={training_metrics['loss']:.4f} "
            f"train_accuracy={training_metrics['accuracy']:.4f} "
            f"valid_accuracy={validation_metrics['accuracy']:.4f}",
            flush=True,
        )
        if validation_metrics['accuracy'] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics['accuracy']
            best_epoch = epoch
            best_state = copy.deepcopy({
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            })

    model.load_state_dict(best_state)
    model = model.to('cpu').eval()
    test_loader = make_loader(
        datasets['test'],
        args.batch_size,
        False,
        args.seed,
        args.num_workers,
    )
    test_metrics = evaluate_torch(model, test_loader, torch.device('cpu'), class_count)
    enforce_accuracy_thresholds(
        'Native test',
        test_metrics,
        args.min_test_accuracy,
        args.min_class_accuracy,
        class_names,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f'.{args.output.name}.tmp-',
        dir=args.output.parent,
    ))
    try:
        model_path = temporary_directory / MODEL_FILENAME
        export_onnx(model, model_path)
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        test_split = cache_metadata['splits']['test']
        test_images = np.load(
            resolve_confined_file(args.cache, test_split['images']['filename']),
            mmap_mode='r',
        )
        test_labels = np.load(
            resolve_confined_file(args.cache, test_split['labels']['filename']),
            mmap_mode='r',
        )
        onnx_test_metrics, max_logit_difference = evaluate_onnx(
            session,
            test_images,
            test_labels,
            args.batch_size,
            class_count,
            model,
        )
        if max_logit_difference >= 1e-4:
            raise ValueError(f'ONNX logit parity failed: {max_logit_difference}')
        enforce_accuracy_thresholds(
            'ONNX test',
            onnx_test_metrics,
            args.min_test_accuracy,
            args.min_class_accuracy,
            class_names,
        )

        model_bytes = model_path.stat().st_size
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        metadata = {
            'schemaVersion': 1,
            'kind': 'quickdraw-classifier',
            'classes': class_names,
            'sourceCacheFingerprint': cache_metadata['fingerprint'],
            'trainingData': {
                'cacheFingerprint': cache_metadata['fingerprint'],
                'sourceManifestFingerprint': cache_metadata['request']['sourceManifestFingerprint'],
                'sourceManifestSha256': cache_metadata['request']['sourceManifestSha256'],
                'sampling': cache_metadata['sampling'],
                'splits': cache_metadata['splits'],
                'rasterizerFiles': cache_metadata['request']['rasterizerFiles'],
                'builder': cache_metadata['request']['builder'],
                'node': cache_metadata['request']['node'],
            },
            'rasterizer': cache_metadata['rasterizer'],
            'architecture': model.config,
            'model': {
                'filename': MODEL_FILENAME,
                'format': 'ONNX',
                'opset': ONNX_OPSET,
                'bytes': model_bytes,
                'sha256': file_sha256(model_path),
                'parameterCount': parameter_count,
                'input': {
                    'name': 'bitmap',
                    'shape': ['batch', 1, 64, 64],
                    'dtype': 'float32',
                    'normalization': 'uint8 / 255',
                },
                'output': {
                    'name': 'logits',
                    'shape': ['batch', class_count],
                    'dtype': 'float32',
                },
            },
            'training': {
                'seed': args.seed,
                'epochs': args.epochs,
                'bestEpoch': best_epoch,
                'batchSize': args.batch_size,
                'optimizer': {
                    'name': 'AdamW',
                    'learningRate': args.learning_rate,
                    'weightDecay': args.weight_decay,
                },
                'numWorkers': args.num_workers,
                'minimumTestAccuracy': args.min_test_accuracy,
                'minimumPerClassAccuracy': args.min_class_accuracy,
                'deterministicAlgorithms': True,
                'device': str(device),
                'history': history,
            },
            'metrics': {
                'test': test_metrics,
                'onnxTest': onnx_test_metrics,
                'maxLogitDifference': max_logit_difference,
            },
            'runtime': {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'machine': platform.machine(),
                'numpy': np.__version__,
                'torch': torch.__version__,
                'onnx': onnx.__version__,
                'onnxruntime': ort.__version__,
            },
            'browserRuntime': browser_runtime_metadata(),
            'source': source_provenance,
        }
        metadata['fingerprint'] = content_fingerprint(metadata)
        (temporary_directory / METADATA_FILENAME).write_text(
            f'{json.dumps(metadata, indent=2)}\n',
            encoding='utf-8',
        )
        atomic_replace_directory(temporary_directory, args.output)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    print(
        f"test_accuracy={test_metrics['accuracy']:.4f} "
        f"onnx_accuracy={onnx_test_metrics['accuracy']:.4f} "
        f'model={args.output / MODEL_FILENAME}',
        flush=True,
    )


if __name__ == '__main__':
    try:
        train(parse_args())
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
