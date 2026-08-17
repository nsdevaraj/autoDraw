#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.integrity import content_fingerprint, file_sha256, resolve_confined_file


DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / 'models' / 'quickdraw-mvp'
DEFAULT_CACHE = PROJECT_ROOT / '.cache' / 'quickdraw-training'
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
        description='Independently verify the tracked Quick Draw ONNX classifier.',
    )
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    return args


def load_json_with_fingerprint(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    fingerprint = value.pop('fingerprint', None)
    if fingerprint != content_fingerprint(value):
        raise ValueError(f'Fingerprint mismatch: {path}')
    value['fingerprint'] = fingerprint
    return value


def validate_cache(cache_directory, metadata):
    for split_name, split in metadata['splits'].items():
        for artifact_name in ('images', 'labels'):
            artifact = split[artifact_name]
            path = resolve_confined_file(cache_directory, artifact['filename'])
            if path.stat().st_size != artifact['bytes']:
                raise ValueError(f'Cache size mismatch: {path}')
            if file_sha256(path) != artifact['sha256']:
                raise ValueError(f'Cache checksum mismatch: {path}')


def validate_cache_provenance(metadata, cache_metadata):
    if metadata.get('sourceCacheFingerprint') != cache_metadata.get('fingerprint'):
        raise ValueError('Model and cache fingerprints do not match')
    if metadata.get('classes') != cache_metadata.get('classes'):
        raise ValueError('Model and cache class order does not match')
    if metadata.get('rasterizer') != cache_metadata.get('rasterizer'):
        raise ValueError('Model and cache rasterizer settings do not match')

    request = cache_metadata['request']
    expected_training_data = {
        'cacheFingerprint': cache_metadata['fingerprint'],
        'sourceManifestFingerprint': request['sourceManifestFingerprint'],
        'sourceManifestSha256': request['sourceManifestSha256'],
        'sampling': cache_metadata['sampling'],
        'splits': cache_metadata['splits'],
        'rasterizerFiles': request['rasterizerFiles'],
        'builder': request['builder'],
        'node': request['node'],
    }
    if metadata.get('trainingData') != expected_training_data:
        raise ValueError('Model training data provenance does not match cache metadata')


class FoldedQuickDrawCNN(nn.Module):
    """PyTorch equivalent of the exported graph with BatchNorm folded into Conv."""

    def __init__(self, initializers):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 192)
        self.fc2 = nn.Linear(192, 40)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self._load_parameter(self.conv1, initializers['onnx::Conv_44'], initializers['onnx::Conv_45'])
        self._load_parameter(self.conv2, initializers['onnx::Conv_47'], initializers['onnx::Conv_48'])
        self._load_parameter(self.conv3, initializers['onnx::Conv_50'], initializers['onnx::Conv_51'])
        self._load_parameter(
            self.fc1,
            initializers['classifier.1.weight'],
            initializers['classifier.1.bias'],
        )
        self._load_parameter(
            self.fc2,
            initializers['classifier.4.weight'],
            initializers['classifier.4.bias'],
        )

    @staticmethod
    def _load_parameter(layer, weight, bias):
        layer.weight.data.copy_(torch.from_numpy(np.array(weight, copy=True)))
        layer.bias.data.copy_(torch.from_numpy(np.array(bias, copy=True)))

    def forward(self, values):
        values = self.pool(torch.relu(self.conv1(values)))
        values = self.pool(torch.relu(self.conv2(values)))
        values = self.pool(torch.relu(self.conv3(values)))
        values = self.adaptive_pool(values)
        values = torch.flatten(values, 1)
        values = torch.relu(self.fc1(values))
        return self.fc2(values)


def metric_accumulators(class_count):
    return {
        'loss': 0.0,
        'count': 0,
        'top1': 0,
        'top5': 0,
        'classCorrect': np.zeros(class_count, dtype=np.int64),
        'classTotal': np.zeros(class_count, dtype=np.int64),
    }


def accumulate_metrics(metrics, logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    metrics['loss'] -= float(log_probabilities[np.arange(len(labels)), labels].sum())
    predictions = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    metrics['count'] += len(labels)
    metrics['top1'] += int((predictions == labels).sum())
    metrics['top5'] += int((top5 == labels[:, None]).any(axis=1).sum())
    np.add.at(metrics['classTotal'], labels, 1)
    np.add.at(metrics['classCorrect'], labels, predictions == labels)


def finish_metrics(metrics):
    return {
        'loss': metrics['loss'] / metrics['count'],
        'accuracy': metrics['top1'] / metrics['count'],
        'top5Accuracy': metrics['top5'] / metrics['count'],
        'perClassAccuracy': (
            metrics['classCorrect'] / metrics['classTotal']
        ).astype(float).tolist(),
    }


def verify(args):
    model_directory = args.model_dir.resolve()
    cache_directory = args.cache.resolve()
    metadata = load_json_with_fingerprint(model_directory / 'model.json')
    cache_metadata = load_json_with_fingerprint(cache_directory / 'metadata.json')
    validate_cache(cache_directory, cache_metadata)
    validate_cache_provenance(metadata, cache_metadata)

    model_path = resolve_confined_file(model_directory, metadata['model']['filename'])
    if model_path.stat().st_size != metadata['model']['bytes']:
        raise ValueError('Model size mismatch')
    if file_sha256(model_path) != metadata['model']['sha256']:
        raise ValueError('Model checksum mismatch')

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in onnx_model.graph.initializer
    }
    torch_model = FoldedQuickDrawCNN(initializers).eval()
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

    test_split = cache_metadata['splits']['test']
    images = np.load(
        resolve_confined_file(cache_directory, test_split['images']['filename']),
        mmap_mode='r',
    )
    labels = np.load(
        resolve_confined_file(cache_directory, test_split['labels']['filename']),
        mmap_mode='r',
    )
    torch_metrics = metric_accumulators(len(metadata['classes']))
    onnx_metrics = metric_accumulators(len(metadata['classes']))
    max_logit_difference = 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), args.batch_size):
            end = min(start + args.batch_size, len(labels))
            batch_images = np.asarray(images[start:end], dtype=np.float32)[:, None, :, :] / 255
            batch_labels = np.asarray(labels[start:end], dtype=np.int64)
            torch_logits = torch_model(torch.from_numpy(batch_images)).numpy()
            onnx_logits = session.run(['logits'], {'bitmap': batch_images})[0]
            max_logit_difference = max(
                max_logit_difference,
                float(np.max(np.abs(torch_logits - onnx_logits))),
            )
            accumulate_metrics(torch_metrics, torch_logits, batch_labels)
            accumulate_metrics(onnx_metrics, onnx_logits, batch_labels)

    result = {
        'cacheFingerprint': cache_metadata['fingerprint'],
        'modelSha256': metadata['model']['sha256'],
        'torch': finish_metrics(torch_metrics),
        'onnx': finish_metrics(onnx_metrics),
        'maxLogitDifference': max_logit_difference,
    }
    return result, metadata


def close_enough(actual, expected, tolerance=1e-6):
    return abs(actual - expected) <= tolerance


def assert_matches_metadata(result, metadata):
    for runtime_name, metadata_name in (('torch', 'test'), ('onnx', 'onnxTest')):
        actual = result[runtime_name]
        expected = metadata['metrics'][metadata_name]
        for metric_name in ('loss', 'accuracy', 'top5Accuracy'):
            if not close_enough(actual[metric_name], expected[metric_name]):
                raise ValueError(f'{runtime_name} {metric_name} does not match metadata')
        if len(actual['perClassAccuracy']) != len(expected['perClassAccuracy']):
            raise ValueError(f'{runtime_name} per-class metric length mismatch')
        for actual_value, expected_value in zip(
            actual['perClassAccuracy'],
            expected['perClassAccuracy'],
        ):
            if not close_enough(actual_value, expected_value):
                raise ValueError(f'{runtime_name} per-class accuracy does not match metadata')
    if result['maxLogitDifference'] >= 1e-4:
        raise ValueError('Recomputed logit parity exceeds the release threshold')
    if metadata['metrics']['maxLogitDifference'] >= 1e-4:
        raise ValueError('Recorded logit parity exceeds the release threshold')
    minimum_accuracy = metadata['training']['minimumTestAccuracy']
    minimum_per_class_accuracy = metadata['training'].get(
        'minimumPerClassAccuracy',
        MINIMUM_RELEASE_ACCURACY,
    )
    if not is_release_threshold(minimum_accuracy) or not is_release_threshold(
        minimum_per_class_accuracy,
    ):
        raise ValueError('Recorded accuracy thresholds are below the 0.70 release floor')
    for runtime_name in ('torch', 'onnx'):
        actual = result[runtime_name]
        if actual['accuracy'] < minimum_accuracy:
            raise ValueError(f'Recomputed {runtime_name} accuracy is below the release threshold')
        if min(actual['perClassAccuracy']) < minimum_per_class_accuracy:
            raise ValueError(
                f'Recomputed {runtime_name} per-class accuracy is below the release threshold',
            )


if __name__ == '__main__':
    try:
        result, metadata = verify(parse_args())
        assert_matches_metadata(result, metadata)
        print(json.dumps(result))
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
