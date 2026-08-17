#!/usr/bin/env python3

"""Train the shared sketch/icon encoder and export it as ONNX."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.sketch_embedder import DEFAULT_EMBEDDING_DIM, create_embedder
from model.integrity import (
    atomic_replace_directory,
    content_fingerprint,
    file_sha256,
    resolve_confined_file,
)


DEFAULT_SKETCH_CACHE = PROJECT_ROOT / '.cache' / 'quickdraw-training'
DEFAULT_ICON_CACHE = PROJECT_ROOT / '.cache' / 'quickdraw-embedding'
DEFAULT_OUTPUT = PROJECT_ROOT / 'models' / 'sketch-embedder'
MODEL_FILENAME = 'sketch-embedder.onnx'
METADATA_FILENAME = 'model.json'
ONNX_OPSET = 17
SPLIT_TRAIN = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and export the sketch-to-icon embedding encoder.',
    )
    parser.add_argument('--sketch-cache', type=Path, default=DEFAULT_SKETCH_CACHE)
    parser.add_argument('--icon-cache', type=Path, default=DEFAULT_ICON_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--classes',
        type=Path,
        help='Line-separated subset of the cached classes to keep',
    )
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--embedding-dim', type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--classification-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    parser.add_argument('--min-recall-at-10', type=float, default=0.70)
    parser.add_argument('--min-class-recall-at-10', type=float, default=0.70)
    parser.add_argument('--min-class-accuracy', type=float, default=0.70)
    parser.add_argument(
        '--allow-weak-classes',
        action='store_true',
        help='Report classes below the floor instead of failing the run',
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.batch_size < 2:
        parser.error('--batch-size must be at least 2')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be positive')
    if not 0 < args.temperature <= 1:
        parser.error('--temperature must be in (0, 1]')
    if not 0 <= args.min_class_accuracy <= 1:
        parser.error('--min-class-accuracy must be from 0 to 1')
    if not 0 <= args.min_recall_at_10 <= 1:
        parser.error('--min-recall-at-10 must be from 0 to 1')
    if not 0 <= args.min_class_recall_at_10 <= 1:
        parser.error('--min-class-recall-at-10 must be from 0 to 1')
    return args


def load_fingerprinted_metadata(directory):
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    fingerprint = metadata.pop('fingerprint', None)
    if fingerprint != content_fingerprint(metadata):
        raise ValueError(f'Cache fingerprint mismatch: {directory}')
    metadata['fingerprint'] = fingerprint
    return metadata


def load_array(directory, artifact, dtype, dimensions):
    path = resolve_confined_file(directory, artifact['filename'])
    if path.stat().st_size != artifact['bytes']:
        raise ValueError(f'Cache size mismatch: {path}')
    if file_sha256(path) != artifact['sha256']:
        raise ValueError(f'Cache checksum mismatch: {path}')
    array = np.load(path, mmap_mode='r')
    if array.dtype != dtype or array.ndim != dimensions:
        raise ValueError(f'Cache array has an unexpected layout: {path}')
    return array


def load_sketches(cache_directory):
    metadata = load_fingerprinted_metadata(cache_directory)
    splits = {}
    for split_name in ('train', 'valid', 'test'):
        split = metadata['splits'][split_name]
        splits[split_name] = (
            load_array(cache_directory, split['images'], np.uint8, 3),
            load_array(cache_directory, split['labels'], np.int16, 1),
        )
    return metadata, splits


def load_icons(cache_directory, classes):
    metadata = load_fingerprinted_metadata(cache_directory)
    if metadata['classes'] != classes:
        raise ValueError('Icon cache classes do not match the sketch cache')
    artifacts = metadata['artifacts']
    return (
        metadata,
        load_array(cache_directory, artifacts['images'], np.uint8, 3),
        load_array(cache_directory, artifacts['labels'], np.int16, 1),
        load_array(cache_directory, artifacts['splits'], np.int8, 1),
    )


def selected_classes(cache_classes, class_list_path):
    """Map cached class indices onto a kept subset so weak classes can be pruned."""
    if class_list_path is None:
        return cache_classes, {index: index for index in range(len(cache_classes))}

    requested = [
        line.strip()
        for line in class_list_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(set(requested)) != len(requested):
        raise ValueError('Class list contains duplicate names')
    if len(requested) < 2:
        raise ValueError('Class list must keep at least two classes')
    unknown = [name for name in requested if name not in cache_classes]
    if unknown:
        raise ValueError(f"Class list names are absent from the cache: {', '.join(unknown)}")
    return requested, {
        cache_classes.index(name): index for index, name in enumerate(requested)
    }


def restrict_to_classes(labels, class_map):
    labels = np.asarray(labels, dtype=np.int64)
    positions = np.flatnonzero(np.isin(labels, np.asarray(sorted(class_map))))
    mapped = np.asarray([class_map[label] for label in labels[positions]], dtype=np.int64)
    return positions, mapped


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


def to_batch(images, indices, device):
    batch = np.asarray(images[indices], dtype=np.float32) / 255
    return torch.from_numpy(batch).unsqueeze(1).to(device)


def icon_positions_by_class(labels, splits, positions, class_count):
    trainable = {}
    for class_index in range(class_count):
        chosen = positions[np.flatnonzero((labels == class_index) & (splits == SPLIT_TRAIN))]
        if chosen.size > 0:
            trainable[class_index] = chosen
    return trainable


def sample_class_icons(trainable, class_count, generator):
    """Pick one training icon per class so every step sees all classes as candidates."""
    chosen = np.zeros(class_count, dtype=np.int64)
    covered = np.zeros(class_count, dtype=bool)
    for class_index in range(class_count):
        positions = trainable.get(class_index)
        if positions is None:
            continue
        chosen[class_index] = positions[generator.integers(len(positions))]
        covered[class_index] = True
    return chosen, covered


def contrastive_loss(sketch_embeddings, icon_embeddings, labels, temperature, covered):
    logits = sketch_embeddings @ icon_embeddings.T / temperature
    logits = logits.masked_fill(~covered.unsqueeze(0), float('-inf'))
    sketch_to_icon = nn.functional.cross_entropy(logits, labels)

    # The transpose scores each icon against the batch, which stops the icon side from
    # collapsing onto whatever the sketch side happens to produce. Targets are the sketches
    # of that class, spread evenly, so classes absent from the batch must be dropped.
    one_hot = nn.functional.one_hot(labels, icon_embeddings.shape[0]).float()
    per_class_counts = one_hot.sum(dim=0)
    present = covered & (per_class_counts > 0)
    targets = (one_hot / per_class_counts.clamp(min=1)).T
    icon_to_sketch = nn.functional.cross_entropy(logits.T[present], targets[present])
    return sketch_to_icon + icon_to_sketch


def train_epoch(model, images, positions, labels, icon_images, trainable, args, device, generator):
    model.train()
    class_count = args.class_count
    covered_classes = np.zeros(class_count, dtype=bool)
    for class_index in trainable:
        covered_classes[class_index] = True
    order = generator.permutation(len(positions))
    loss_sum = 0.0
    sample_count = 0

    for start in range(0, len(order), args.batch_size):
        batch = np.sort(order[start:start + args.batch_size])
        batch = batch[covered_classes[labels[batch]]]
        if len(batch) < 2:
            continue
        batch_images = to_batch(images, positions[batch], device)
        batch_labels = torch.from_numpy(labels[batch]).to(device)

        icon_indices, covered_mask = sample_class_icons(trainable, class_count, generator)
        icon_batch = to_batch(icon_images, icon_indices, device)
        covered = torch.from_numpy(covered_mask).to(device)

        args.optimizer.zero_grad(set_to_none=True)
        sketch_embeddings, sketch_logits = model(batch_images)
        icon_embeddings = model.embed(icon_batch)
        loss = contrastive_loss(
            sketch_embeddings, icon_embeddings, batch_labels, args.temperature, covered,
        )
        loss = loss + args.classification_weight * nn.functional.cross_entropy(
            sketch_logits, batch_labels,
        )
        loss.backward()
        args.optimizer.step()

        loss_sum += loss.item() * len(batch)
        sample_count += len(batch)

    return {'loss': float(loss_sum / max(sample_count, 1))}


@torch.inference_mode()
def encode_all(model, images, positions, device, batch_size):
    model.eval()
    embeddings = []
    logits = []
    for start in range(0, len(positions), batch_size):
        embedding, logit = model(
            to_batch(images, positions[start:start + batch_size], device),
        )
        embeddings.append(embedding.cpu())
        logits.append(logit.cpu())
    return torch.cat(embeddings), torch.cat(logits)


def classification_metrics(logits, labels, class_count):
    predictions = logits.argmax(dim=1).numpy()
    correct = np.zeros(class_count, dtype=np.int64)
    total = np.zeros(class_count, dtype=np.int64)
    np.add.at(total, labels, 1)
    np.add.at(correct, labels, predictions == labels)
    per_class = [float(c / t) if t else 0.0 for c, t in zip(correct, total)]
    top_count = min(5, class_count)
    top = logits.topk(top_count, dim=1).indices.numpy()
    return {
        'accuracy': float((predictions == labels).mean()),
        'top5Accuracy': float((top == labels[:, None]).any(axis=1).mean()),
        'perClassAccuracy': per_class,
    }


def retrieval_metrics(sketch_embeddings, sketch_labels, icon_embeddings, icon_labels, ranks, class_count):
    """Score how often an icon of the true class appears in the top-k retrieved icons."""
    if len(icon_labels) == 0:
        return {
            **{f'recallAt{rank}': 0.0 for rank in ranks},
            'perClassRecallAt10': [0.0] * class_count,
        }
    similarity = sketch_embeddings @ icon_embeddings.T
    largest = min(max(ranks), similarity.shape[1])
    ordered = similarity.topk(largest, dim=1).indices.numpy()
    retrieved = icon_labels[ordered]
    matches = retrieved == sketch_labels[:, None]

    hits = matches[:, :min(10, largest)].any(axis=1)
    correct = np.zeros(class_count, dtype=np.int64)
    total = np.zeros(class_count, dtype=np.int64)
    np.add.at(total, sketch_labels, 1)
    np.add.at(correct, sketch_labels, hits)
    return {
        **{
            f'recallAt{rank}': float(matches[:, :min(rank, largest)].any(axis=1).mean())
            for rank in ranks
        },
        'perClassRecallAt10': [
            float(c / t) if t else 0.0 for c, t in zip(correct, total)
        ],
    }


def export_onnx(model, output_path, class_count, embedding_dim):
    model.eval()
    example = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=['bitmap'],
        output_names=['embedding', 'logits'],
        dynamic_axes={
            'bitmap': {0: 'batch'},
            'embedding': {0: 'batch'},
            'logits': {0: 'batch'},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output_path)), full_check=True)

    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    probe = np.random.default_rng(7).random((3, 1, 64, 64), dtype=np.float32)
    embedding, logits = session.run(['embedding', 'logits'], {'bitmap': probe})
    if embedding.shape != (3, embedding_dim) or logits.shape != (3, class_count):
        raise ValueError('Exported ONNX produced unexpected output shapes')
    with torch.inference_mode():
        torch_embedding, torch_logits = model(torch.from_numpy(probe))
    return max(
        float(np.max(np.abs(torch_embedding.numpy() - embedding))),
        float(np.max(np.abs(torch_logits.numpy() - logits))),
    )


def source_file_hashes():
    paths = (
        PROJECT_ROOT / 'model' / 'sketch_embedder.py',
        PROJECT_ROOT / 'model' / 'quickdraw_cnn.py',
        PROJECT_ROOT / 'model' / 'integrity.py',
        PROJECT_ROOT / 'scripts' / 'build-embedding-cache.py',
        PROJECT_ROOT / 'scripts' / 'rasterize-svg-worker.mjs',
        PROJECT_ROOT / 'scripts' / 'train-sketch-embedder.py',
        PROJECT_ROOT / 'src' / 'svg-to-polylines.mjs',
        PROJECT_ROOT / 'src' / 'sketch-rasterizer.mjs',
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in paths}


def enforce_release_gates(args, metrics, classes):
    """Gate on retrieval, which is what a user sees; the aux head is only a diagnostic."""
    retrieval = metrics['retrieval']['all']
    weak = {
        'byRecall': [
            classes[index]
            for index, recall in enumerate(retrieval['perClassRecallAt10'])
            if recall < args.min_class_recall_at_10
        ],
        'byClassification': [
            classes[index]
            for index, accuracy in enumerate(metrics['classification']['perClassAccuracy'])
            if accuracy < args.min_class_accuracy
        ],
    }

    problems = []
    if retrieval['recallAt10'] < args.min_recall_at_10:
        problems.append(
            f"recall@10 {retrieval['recallAt10']:.4f} is below {args.min_recall_at_10:.4f}",
        )
    if not args.allow_weak_classes:
        if weak['byRecall']:
            problems.append(
                f"{len(weak['byRecall'])} classes are below the "
                f'{args.min_class_recall_at_10:.2f} per-class recall@10 floor: '
                f"{', '.join(weak['byRecall'])}",
            )
        if weak['byClassification']:
            problems.append(
                f"{len(weak['byClassification'])} classes are below the "
                f'{args.min_class_accuracy:.2f} per-class accuracy floor: '
                f"{', '.join(weak['byClassification'])}",
            )
    if problems:
        raise ValueError(
            '; '.join(problems)
            + '. Remove the listed classes from --classes and re-run, '
            + 'or use --allow-weak-classes to inspect the result.',
        )
    return weak


def write_output(args, model_state, metrics, sketch_metadata, icon_metadata, classes, weak_classes):
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f'.{args.output.name}.tmp-',
        dir=args.output.parent,
    ))
    try:
        model_path = temporary_directory / MODEL_FILENAME
        max_difference = export_onnx(
            model_state, model_path, len(classes), args.embedding_dim,
        )
        model_bytes = model_path.read_bytes()

        metadata = {
            'schemaVersion': 1,
            'kind': 'sketch-embedder',
            'classes': classes,
            'trainingData': {
                'sketchCacheFingerprint': sketch_metadata['fingerprint'],
                'iconCacheFingerprint': icon_metadata['fingerprint'],
                'iconCounts': icon_metadata['counts'],
                'source': icon_metadata['request']['source'],
            },
            'model': {
                'filename': MODEL_FILENAME,
                'format': 'ONNX',
                'opset': ONNX_OPSET,
                'bytes': len(model_bytes),
                'sha256': file_sha256(model_path),
                'parameterCount': sum(p.numel() for p in model_state.parameters()),
                'embeddingDim': args.embedding_dim,
                'input': {
                    'name': 'bitmap',
                    'dtype': 'float32',
                    'shape': ['batch', 1, 64, 64],
                    'normalization': 'uint8 / 255',
                },
                'outputs': {
                    'embedding': {
                        'name': 'embedding',
                        'dtype': 'float32',
                        'shape': ['batch', args.embedding_dim],
                        'normalization': 'L2',
                    },
                    'logits': {
                        'name': 'logits',
                        'dtype': 'float32',
                        'shape': ['batch', len(classes)],
                    },
                },
            },
            'rasterizer': icon_metadata['request']['rasterizer'],
            'metrics': {**metrics, 'maxLogitDifference': max_difference},
            'training': {
                'epochs': args.epochs,
                'batchSize': args.batch_size,
                'optimizer': {
                    'name': 'AdamW',
                    'learningRate': args.learning_rate,
                    'weightDecay': args.weight_decay,
                },
                'temperature': args.temperature,
                'classificationWeight': args.classification_weight,
                'seed': args.seed,
                'deterministicAlgorithms': True,
                'minClassAccuracy': args.min_class_accuracy,
                'minRecallAt10': args.min_recall_at_10,
                'minClassRecallAt10': args.min_class_recall_at_10,
                'weakClasses': weak_classes,
            },
            'runtime': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'torch': torch.__version__,
                'onnx': onnx.__version__,
                'onnxruntime': ort.__version__,
            },
            'source': {'files': source_file_hashes()},
        }
        metadata['fingerprint'] = content_fingerprint(metadata)
        (temporary_directory / METADATA_FILENAME).write_text(
            f'{json.dumps(metadata, indent=2)}\n',
            encoding='utf-8',
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_directory(temporary_directory, args.output)
        return metadata
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def main():
    args = parse_args()
    args.sketch_cache = args.sketch_cache.resolve()
    args.icon_cache = args.icon_cache.resolve()
    args.output = args.output.resolve()

    sketch_metadata, sketch_splits = load_sketches(args.sketch_cache)
    cache_classes = sketch_metadata['classes']
    icon_metadata, icon_images, icon_labels, icon_splits = load_icons(
        args.icon_cache, cache_classes,
    )
    classes, class_map = selected_classes(cache_classes, args.classes)

    args.class_count = len(classes)
    set_training_seed(args.seed)
    device = select_device(args.device)
    model = create_embedder(args.class_count, args.seed, args.embedding_dim).to(device)
    args.optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    icon_positions, icon_labels_array = restrict_to_classes(icon_labels, class_map)
    icon_splits_array = np.asarray(icon_splits)[icon_positions]
    trainable = icon_positions_by_class(
        icon_labels_array, icon_splits_array, icon_positions, args.class_count,
    )
    if not trainable:
        raise ValueError('The icon cache has no trainable icons for the selected classes')

    generator = np.random.default_rng(args.seed)
    train_images, train_labels = sketch_splits['train']
    train_positions, train_labels_array = restrict_to_classes(train_labels, class_map)
    for epoch in range(args.epochs):
        stats = train_epoch(
            model, train_images, train_positions, train_labels_array, icon_images,
            trainable, args, device, generator,
        )
        print(f"epoch {epoch + 1}/{args.epochs} loss {stats['loss']:.4f}", flush=True)

    test_images, test_labels = sketch_splits['test']
    test_positions, test_labels_array = restrict_to_classes(test_labels, class_map)
    sketch_embeddings, sketch_logits = encode_all(
        model, test_images, test_positions, device, args.batch_size,
    )
    icon_embeddings, _ = encode_all(model, icon_images, icon_positions, device, args.batch_size)
    holdout = icon_splits_array != SPLIT_TRAIN

    metrics = {
        'classification': classification_metrics(
            sketch_logits, test_labels_array, args.class_count,
        ),
        'retrieval': {
            'all': retrieval_metrics(
                sketch_embeddings, test_labels_array,
                icon_embeddings, icon_labels_array, (1, 5, 10), args.class_count,
            ),
            'holdoutIcons': retrieval_metrics(
                sketch_embeddings, test_labels_array,
                icon_embeddings[holdout], icon_labels_array[holdout], (1, 5, 10),
                args.class_count,
            ),
        },
    }

    weak_classes = enforce_release_gates(args, metrics, classes)
    metadata = write_output(
        args, model.cpu(), metrics, sketch_metadata, icon_metadata, classes, weak_classes,
    )

    print(f"Classes: {args.class_count}")
    print(f"Test accuracy: {metrics['classification']['accuracy']:.4f}")
    print(f"Test top-5 accuracy: {metrics['classification']['top5Accuracy']:.4f}")
    print(f"Lowest per-class accuracy: {min(metrics['classification']['perClassAccuracy']):.4f}")
    for name, scores in metrics['retrieval'].items():
        formatted = ' '.join(
            f'{key} {value:.4f}' for key, value in scores.items() if key.startswith('recall')
        )
        print(f'Retrieval ({name}): {formatted}')
        print(f"  lowest per-class recall@10: {min(scores['perClassRecallAt10']):.4f}")
    if weak_classes['byRecall']:
        print(f"Below the recall floor: {', '.join(weak_classes['byRecall'])}")
    if weak_classes['byClassification']:
        print(f"Below the accuracy floor: {', '.join(weak_classes['byClassification'])}")
    print(f"Model: {args.output / MODEL_FILENAME} ({metadata['model']['bytes']:,} bytes)")
    print(f"Fingerprint: {metadata['fingerprint']}")


if __name__ == '__main__':
    try:
        main()
    except Exception as error:  # noqa: BLE001 - surfaced as a CLI message
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
