import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const PYTHON = '.venv-model/bin/python3';

test('Quick Draw CNN has deterministic initialization, bounded size, and logits contract', () => {
  const source = `
import hashlib
import json
import torch

from model.quickdraw_cnn import QuickDrawCNN, create_model

first = create_model(num_classes=40, seed=123)
second = create_model(num_classes=40, seed=123)
different = create_model(num_classes=40, seed=124)
first.eval()
second.eval()
different.eval()

def state_hash(model):
  digest = hashlib.sha256()
  for name, tensor in model.state_dict().items():
    digest.update(name.encode('utf-8'))
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()

with torch.inference_mode():
    logits = first(torch.zeros(2, 1, 64, 64))

errors = []
for shape in ((2, 64, 64), (2, 3, 64, 64), (2, 1, 32, 32)):
    try:
        first(torch.zeros(*shape))
    except ValueError as error:
        errors.append(str(error))

print(json.dumps({
    'logitsShape': list(logits.shape),
    'parameterCount': sum(parameter.numel() for parameter in first.parameters()),
    'firstHash': state_hash(first),
    'secondHash': state_hash(second),
    'differentHash': state_hash(different),
    'errors': errors,
    'config': first.config,
}))
`;
  const result = spawnSync(PYTHON, ['-c', source], { encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
  const details = JSON.parse(result.stdout);
  assert.deepEqual(details.logitsShape, [2, 40]);
  assert.equal(details.parameterCount > 200000, true);
  assert.equal(details.parameterCount < 750000, true);
  assert.equal(details.firstHash, details.secondHash);
  assert.notEqual(details.firstHash, details.differentHash);
  assert.equal(details.errors.length, 3);
  assert.deepEqual(details.config, {
    inputShape: [1, 64, 64],
    numClasses: 40,
    channels: [32, 64, 128],
    embeddingSize: 192,
    dropout: 0.2,
  });
});

test('Quick Draw CNN rejects fewer than two classes', () => {
  const result = spawnSync(PYTHON, [
    '-c',
    'from model.quickdraw_cnn import QuickDrawCNN; QuickDrawCNN(num_classes=1)',
  ], { encoding: 'utf8' });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /num_classes must be at least 2/);
});

test('artifact path resolver rejects traversal, absolute paths, and symlink escapes', () => {
  const result = spawnSync(PYTHON, ['-c', `
import os
from pathlib import Path
import tempfile
from model.integrity import resolve_confined_file

with tempfile.TemporaryDirectory() as root_name, tempfile.TemporaryDirectory() as outside_name:
    root = Path(root_name)
    outside = Path(outside_name)
    (root / 'ok.npy').write_bytes(b'ok')
    (outside / 'escape.npy').write_bytes(b'escape')
    os.symlink(outside / 'escape.npy', root / 'link.npy')
    os.symlink(root / 'loop.npy', root / 'loop.npy')
    assert resolve_confined_file(root, 'ok.npy') == (root / 'ok.npy').resolve()
    errors = []
    for value in (
      '../escape.npy',
      'sub/../ok.npy',
      str(outside / 'escape.npy'),
      'link.npy',
      'loop.npy',
    ):
        try:
            resolve_confined_file(root, value)
        except ValueError as error:
            errors.append(str(error))
    assert len(errors) == 5
`], { encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
});

test('accuracy gates reject a low class even when aggregate accuracy passes', () => {
  const result = spawnSync(PYTHON, ['-c', `
import copy
import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

trainer = load('trainer', 'scripts/train-quickdraw-model.py')
verifier = load('verifier', 'scripts/verify-quickdraw-model.py')
metrics = {
    'loss': 0.1,
    'accuracy': 0.9,
    'top5Accuracy': 1.0,
    'perClassAccuracy': [1.0, 0.69],
}

try:
    trainer.enforce_accuracy_thresholds(
      'Native', metrics, 0.7, 0.7, ['strong', 'weak'],
    )
except ValueError as error:
    assert 'weak' in str(error)
else:
    raise AssertionError('Trainer accepted a class below the floor')

result = {
    'torch': dict(metrics),
    'onnx': dict(metrics),
    'maxLogitDifference': 0.0,
}
metadata = {
    'classes': ['strong', 'weak'],
    'training': {
      'minimumTestAccuracy': 0.7,
      'minimumPerClassAccuracy': 0.7,
    },
    'metrics': {
      'test': dict(metrics),
      'onnxTest': dict(metrics),
      'maxLogitDifference': 0.0,
    },
}
try:
    verifier.assert_matches_metadata(result, metadata)
except ValueError as error:
    assert 'per-class accuracy' in str(error)
else:
    raise AssertionError('Verifier accepted a class below the floor')

legacy_metadata = {
    'schemaVersion': 1,
    'classes': ['strong', 'weak'],
    'training': {'minimumTestAccuracy': 0.7},
    'metrics': {
        'test': dict(metrics),
        'onnxTest': dict(metrics),
        'maxLogitDifference': 0.0,
    },
}
try:
    verifier.assert_matches_metadata(result, legacy_metadata)
except ValueError as error:
    assert 'per-class accuracy' in str(error)
else:
    raise AssertionError('Legacy verifier fallback accepted a class below 70%')

legacy_metrics = dict(metrics, perClassAccuracy=[0.8, 0.8])
legacy_result = {
    'torch': dict(legacy_metrics),
    'onnx': dict(legacy_metrics),
    'maxLogitDifference': 0.0,
}
legacy_metadata['metrics']['test'] = dict(legacy_metrics)
legacy_metadata['metrics']['onnxTest'] = dict(legacy_metrics)
verifier.assert_matches_metadata(legacy_result, legacy_metadata)

release_metrics = dict(metrics, accuracy=0.8, perClassAccuracy=[0.8, 0.8])
release_result = {
  'torch': dict(release_metrics),
  'onnx': dict(release_metrics),
  'maxLogitDifference': 0.0,
}
release_metadata = {
  'schemaVersion': 1,
  'classes': ['strong', 'weak'],
  'training': {
    'minimumTestAccuracy': 0.7,
    'minimumPerClassAccuracy': 0.7,
  },
  'metrics': {
    'test': dict(release_metrics),
    'onnxTest': dict(release_metrics),
    'maxLogitDifference': 0.0,
  },
}
for field in ('minimumTestAccuracy', 'minimumPerClassAccuracy'):
  for value in (0.0, float('nan')):
    invalid_metadata = copy.deepcopy(release_metadata)
    invalid_metadata['training'][field] = value
    try:
      verifier.assert_matches_metadata(release_result, invalid_metadata)
    except ValueError as error:
      assert 'release floor' in str(error)
    else:
      raise AssertionError(f'Verifier accepted invalid {field}')

for minimum_accuracy, minimum_per_class_accuracy in (
  (float('nan'), 0.7),
  (0.7, float('nan')),
):
  try:
    trainer.enforce_accuracy_thresholds(
      'Native',
      release_metrics,
      minimum_accuracy,
      minimum_per_class_accuracy,
      ['strong', 'weak'],
    )
  except ValueError as error:
    assert 'release floor' in str(error)
  else:
    raise AssertionError('Trainer accepted a non-finite release threshold')
`], { encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
});

test('trainer release thresholds cannot be configured below 70%', () => {
  const result = spawnSync(PYTHON, ['-c', `
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('trainer', 'scripts/train-quickdraw-model.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

for option in ('--min-test-accuracy', '--min-class-accuracy'):
  for value in ('0.69', 'nan', 'inf'):
    sys.argv = ['train-quickdraw-model.py', option, value]
    try:
      trainer.parse_args()
    except SystemExit as error:
      assert error.code == 2
    else:
      raise AssertionError(f'{option} accepted invalid value {value}')
`], { encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
});

test('model provenance is stable across artifact-only commits', () => {
  const result = spawnSync(PYTHON, ['-c', `
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location('trainer', 'scripts/train-quickdraw-model.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

def git(repository, *arguments):
  return subprocess.check_output(['git', *arguments], cwd=repository).decode().strip()

with tempfile.TemporaryDirectory() as temporary_name:
  repository = Path(temporary_name) / 'repository'
  repository.mkdir()
  subprocess.run(['git', 'init', '-q'], cwd=repository, check=True)
  subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repository, check=True)
  subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repository, check=True)
  (repository / 'source.txt').write_text('source-v1')
  subprocess.run(['git', 'add', 'source.txt'], cwd=repository, check=True)
  subprocess.run(['git', 'commit', '-qm', 'source'], cwd=repository, check=True)
  source_revision = git(repository, 'rev-parse', 'HEAD')
  source_hash = hashlib.sha256((repository / 'source.txt').read_bytes()).hexdigest()

  (repository / 'artifact.txt').write_text('artifact-v1')
  subprocess.run(['git', 'add', 'artifact.txt'], cwd=repository, check=True)
  subprocess.run(['git', 'commit', '-qm', 'artifact'], cwd=repository, check=True)
  assert git(repository, 'rev-parse', 'HEAD') != source_revision

  trainer.PROJECT_ROOT = repository
  provenance = trainer.repository_provenance({'source.txt': source_hash})
  assert provenance['repositoryRevision'] == source_revision, provenance
  assert provenance['filesMatchRevision'] is True

  (repository / 'source.txt').write_text('dirty-source')
  dirty_hash = hashlib.sha256((repository / 'source.txt').read_bytes()).hexdigest()
  dirty = trainer.repository_provenance({'source.txt': dirty_hash})
  assert dirty['repositoryRevision'] == source_revision
  assert dirty['filesMatchRevision'] is False

  (repository / 'source.txt').unlink()
  (repository / 'same-content.txt').write_text('source-v1')
  (repository / 'source.txt').symlink_to('same-content.txt')
  symlinked = trainer.repository_provenance({'source.txt': source_hash})
  assert symlinked['repositoryRevision'] == source_revision
  assert symlinked['filesMatchRevision'] is False
  (repository / 'source.txt').unlink()
  (repository / 'source.txt').write_text('source-v1')

  (repository / 'source.txt').write_text('staged-source')
  subprocess.run(['git', 'add', 'source.txt'], cwd=repository, check=True)
  (repository / 'source.txt').write_text('source-v1')
  masked_index = trainer.repository_provenance({'source.txt': source_hash})
  assert masked_index['repositoryRevision'] == source_revision
  assert masked_index['filesMatchRevision'] is False
  subprocess.run(['git', 'restore', '--staged', 'source.txt'], cwd=repository, check=True)

  subprocess.run(['git', 'config', 'core.fileMode', 'false'], cwd=repository, check=True)
  (repository / 'source.txt').chmod(0o755)
  mode_changed = trainer.repository_provenance({'source.txt': source_hash})
  assert mode_changed['repositoryRevision'] == source_revision
  assert mode_changed['filesMatchRevision'] is False
  (repository / 'source.txt').chmod(0o644)

  subprocess.run(['git', 'config', 'core.fileMode', 'true'], cwd=repository, check=True)
  (repository / 'source.txt').chmod(0o755)
  subprocess.run(['git', 'add', 'source.txt'], cwd=repository, check=True)
  subprocess.run(['git', 'commit', '-qm', 'executable source'], cwd=repository, check=True)
  executable_revision = git(repository, 'rev-parse', 'HEAD')
  subprocess.run(['git', 'config', 'core.fileMode', 'false'], cwd=repository, check=True)
  (repository / 'source.txt').chmod(0o644)
  mode_removed = trainer.repository_provenance({'source.txt': source_hash})
  assert mode_removed['repositoryRevision'] == executable_revision
  assert mode_removed['filesMatchRevision'] is False
  (repository / 'source.txt').chmod(0o755)

  shallow = Path(temporary_name) / 'shallow'
  subprocess.run(
    ['git', 'clone', '-q', '--depth', '1', repository.as_uri(), str(shallow)],
    check=True,
  )
  trainer.PROJECT_ROOT = shallow
  incomplete = trainer.repository_provenance({'source.txt': source_hash})
  assert incomplete['repositoryRevision'] is None
  assert incomplete['filesMatchRevision'] is False

  symlink_repository = Path(temporary_name) / 'symlink-repository'
  symlink_repository.mkdir()
  subprocess.run(['git', 'init', '-q'], cwd=symlink_repository, check=True)
  subprocess.run(
    ['git', 'config', 'user.email', 'test@example.com'],
    cwd=symlink_repository,
    check=True,
  )
  subprocess.run(
    ['git', 'config', 'user.name', 'Test'],
    cwd=symlink_repository,
    check=True,
  )
  (symlink_repository / 'target.txt').write_text('payload')
  (symlink_repository / 'source.txt').symlink_to('target.txt')
  subprocess.run(['git', 'add', '.'], cwd=symlink_repository, check=True)
  subprocess.run(['git', 'commit', '-qm', 'symlink source'], cwd=symlink_repository, check=True)
  trainer.PROJECT_ROOT = symlink_repository
  symlink_blob_hash = hashlib.sha256(b'target.txt').hexdigest()
  non_regular_source = trainer.repository_provenance({'source.txt': symlink_blob_hash})
  assert non_regular_source['repositoryRevision'] == git(
    symlink_repository,
    'rev-parse',
    'HEAD',
  )
  assert non_regular_source['filesMatchRevision'] is False
`], { encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
});

test('model verifier rejects reordered classes and altered cache provenance', () => {
  const result = spawnSync(PYTHON, ['-c', `
import copy
import importlib.util

spec = importlib.util.spec_from_file_location('verifier', 'scripts/verify-quickdraw-model.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

cache = {
    'fingerprint': 'cache-fingerprint',
    'classes': ['alpha', 'beta'],
    'sampling': {'seed': 1},
    'splits': {'test': {'count': 2}},
    'rasterizer': {'size': 64},
    'request': {
        'sourceManifestFingerprint': 'manifest-fingerprint',
        'sourceManifestSha256': 'a' * 64,
        'rasterizerFiles': {'implementation': {'sha256': 'b' * 64}},
        'builder': {'sha256': 'c' * 64},
        'node': {'sha256': 'd' * 64},
    },
}
metadata = {
    'sourceCacheFingerprint': cache['fingerprint'],
    'classes': list(cache['classes']),
    'rasterizer': dict(cache['rasterizer']),
    'trainingData': {
        'cacheFingerprint': cache['fingerprint'],
        'sourceManifestFingerprint': cache['request']['sourceManifestFingerprint'],
        'sourceManifestSha256': cache['request']['sourceManifestSha256'],
        'sampling': cache['sampling'],
        'splits': cache['splits'],
        'rasterizerFiles': cache['request']['rasterizerFiles'],
        'builder': cache['request']['builder'],
        'node': cache['request']['node'],
    },
}
verifier.validate_cache_provenance(metadata, cache)

reordered = copy.deepcopy(metadata)
reordered['classes'].reverse()
try:
    verifier.validate_cache_provenance(reordered, cache)
except ValueError as error:
    assert 'class order' in str(error)
else:
    raise AssertionError('Verifier accepted reordered classes')

altered = copy.deepcopy(metadata)
altered['trainingData']['builder']['sha256'] = 'e' * 64
try:
    verifier.validate_cache_provenance(altered, cache)
except ValueError as error:
    assert 'provenance' in str(error)
else:
    raise AssertionError('Verifier accepted altered cache provenance')

fingerprint_mismatch = copy.deepcopy(metadata)
fingerprint_mismatch['sourceCacheFingerprint'] = 'other-cache'
try:
  verifier.validate_cache_provenance(fingerprint_mismatch, cache)
except ValueError as error:
  assert 'fingerprints' in str(error)
else:
  raise AssertionError('Verifier accepted a different cache fingerprint')

rasterizer_mismatch = copy.deepcopy(metadata)
rasterizer_mismatch['rasterizer']['size'] = 32
try:
  verifier.validate_cache_provenance(rasterizer_mismatch, cache)
except ValueError as error:
  assert 'rasterizer' in str(error)
else:
  raise AssertionError('Verifier accepted different rasterizer settings')

for field in (
  'cacheFingerprint',
  'sourceManifestFingerprint',
  'sourceManifestSha256',
  'sampling',
  'splits',
  'rasterizerFiles',
  'builder',
  'node',
):
  altered = copy.deepcopy(metadata)
  altered['trainingData'][field] = {'tampered': field}
  try:
    verifier.validate_cache_provenance(altered, cache)
  except ValueError as error:
    assert 'provenance' in str(error)
  else:
    raise AssertionError(f'Verifier accepted altered {field}')
`], { encoding: 'utf8' });

  assert.equal(result.status, 0, result.stderr);
});
