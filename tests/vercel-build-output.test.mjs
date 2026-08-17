import assert from 'node:assert/strict';
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { mkdir, readdir, rename, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';

import {
  assertSafeBuildOutput,
  cleanupFailedBuildOutput,
  publishBuildOutput,
} from '../src/vercel-build-output.mjs';

test('build output rejects the project root, ancestors, and source directories', () => {
  const parent = mkdtempSync(join(tmpdir(), 'autodraw-output-safety-'));
  const root = join(parent, 'autodraw-project');
  const defaultOutput = join(root, 'dist');
  const alias = join(parent, 'project-alias');
  mkdirSync(root);
  mkdirSync(join(root, 'src'));
  symlinkSync(root, alias);

  try {
    assert.equal(assertSafeBuildOutput({ projectRoot: root, defaultOutput, output: defaultOutput }), defaultOutput);
    for (const output of [root, dirname(root), join(root, 'src'), join(alias, 'src')]) {
      assert.throws(
        () => assertSafeBuildOutput({ projectRoot: root, defaultOutput, output }),
        /unsafe Vercel output/,
      );
    }
    const sibling = join(parent, 'autodraw-project-output');
    assert.equal(
      assertSafeBuildOutput({ projectRoot: root, defaultOutput, output: sibling }),
      sibling,
    );
    const externalCustomTarget = join(parent, 'external-custom-output');
    mkdirSync(externalCustomTarget);
    const inProjectAlias = join(root, 'custom-output');
    symlinkSync(externalCustomTarget, inProjectAlias);
    assert.throws(
      () => assertSafeBuildOutput({
        projectRoot: root,
        defaultOutput,
        output: inProjectAlias,
      }),
      /unsafe Vercel output/,
    );

    symlinkSync(join(root, 'src'), defaultOutput);
    assert.throws(
      () => assertSafeBuildOutput({ projectRoot: root, defaultOutput, output: defaultOutput }),
      /unsafe Vercel output/,
    );
    rmSync(defaultOutput);
    const externalTarget = join(parent, 'external-dist');
    mkdirSync(externalTarget);
    symlinkSync(externalTarget, defaultOutput);
    assert.throws(
      () => assertSafeBuildOutput({ projectRoot: root, defaultOutput, output: defaultOutput }),
      /unsafe Vercel output/,
    );
    rmSync(defaultOutput);
    symlinkSync(join(parent, 'missing-dist'), defaultOutput);
    assert.throws(
      () => assertSafeBuildOutput({ projectRoot: root, defaultOutput, output: defaultOutput }),
      /unsafe Vercel output/,
    );
  } finally {
    rmSync(parent, { recursive: true, force: true });
  }
});

test('build output publication restores the previous directory when publish fails', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  await mkdir(output);
  await mkdir(temporaryOutput);
  writeFileSync(join(output, 'old.txt'), 'old');
  writeFileSync(join(temporaryOutput, 'new.txt'), 'new');
  let renameCount = 0;

  try {
    await assert.rejects(
      () => publishBuildOutput({
        output,
        temporaryOutput,
        renameImpl: async (source, destination) => {
          renameCount += 1;
          if (renameCount === 2) throw new Error('publish failed');
          await rename(source, destination);
        },
      }),
      /publish failed/,
    );
    assert.equal(readFileSync(join(output, 'old.txt'), 'utf8'), 'old');
    assert.equal(existsSync(join(output, 'new.txt')), false);
    assert.deepEqual((await readdir(root)).sort(), ['.dist.tmp', 'dist']);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('build output publication atomically replaces and removes its backup', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-success-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  await mkdir(output);
  await mkdir(temporaryOutput);
  writeFileSync(join(output, 'old.txt'), 'old');
  writeFileSync(join(temporaryOutput, 'new.txt'), 'new');

  try {
    await publishBuildOutput({ output, temporaryOutput });
    assert.equal(readFileSync(join(output, 'new.txt'), 'utf8'), 'new');
    assert.equal(existsSync(join(output, 'old.txt')), false);
    assert.deepEqual((await readdir(root)).sort(), ['dist']);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('build output publication recovers an interrupted backup before replacing it', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-recovery-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  const backup = join(root, '.dist.old');
  await mkdir(backup);
  await mkdir(temporaryOutput);
  writeFileSync(join(backup, 'old.txt'), 'old');
  writeFileSync(join(temporaryOutput, 'new.txt'), 'new');
  const renames = [];

  try {
    await publishBuildOutput({
      output,
      temporaryOutput,
      renameImpl: async (source, destination) => {
        renames.push([source, destination]);
        await rename(source, destination);
      },
    });
    assert.deepEqual(renames, [
      [backup, output],
      [output, backup],
      [temporaryOutput, output],
    ]);
    assert.equal(readFileSync(join(output, 'new.txt'), 'utf8'), 'new');
    assert.deepEqual((await readdir(root)).sort(), ['dist']);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('build output publication preserves publish and rollback errors', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-double-failure-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  await mkdir(output);
  await mkdir(temporaryOutput);
  let renameCount = 0;

  try {
    await assert.rejects(
      () => publishBuildOutput({
        output,
        temporaryOutput,
        renameImpl: async (source, destination) => {
          renameCount += 1;
          if (renameCount === 2) throw new Error('publish failed');
          if (renameCount === 3) throw new Error('rollback failed');
          await rename(source, destination);
        },
      }),
      error => {
        assert.equal(error instanceof AggregateError, true);
        assert.deepEqual(error.errors.map(item => item.message), [
          'publish failed',
          'rollback failed',
        ]);
        return true;
      },
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('build output publication rejects overlapping publishers without touching output', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-overlap-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  const lock = join(root, '.dist.lock');
  await mkdir(output);
  await mkdir(temporaryOutput);
  symlinkSync(`owner-${process.pid}-active`, lock);
  writeFileSync(join(output, 'old.txt'), 'old');
  writeFileSync(join(temporaryOutput, 'new.txt'), 'new');

  try {
    await assert.rejects(
      () => publishBuildOutput({ output, temporaryOutput }),
      /publication lock exists/,
    );
    assert.equal(readFileSync(join(output, 'old.txt'), 'utf8'), 'old');
    assert.equal(readFileSync(join(temporaryOutput, 'new.txt'), 'utf8'), 'new');
    assert.equal(lstatSync(lock).isSymbolicLink(), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('build output publication rejects a stale lock without deleting it', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-stale-lock-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  const lock = join(root, '.dist.lock');
  await mkdir(output);
  await mkdir(temporaryOutput);
  symlinkSync('owner-2147483647-stale', lock);
  writeFileSync(join(output, 'old.txt'), 'old');
  writeFileSync(join(temporaryOutput, 'new.txt'), 'new');

  try {
    await assert.rejects(
      () => publishBuildOutput({ output, temporaryOutput }),
      error => {
        assert.match(error.message, /publication lock exists/);
        assert.match(error.message, new RegExp(lock.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
        assert.match(error.message, /confirming no build is running/);
        return true;
      },
    );
    assert.equal(readFileSync(join(output, 'old.txt'), 'utf8'), 'old');
    assert.equal(readFileSync(join(temporaryOutput, 'new.txt'), 'utf8'), 'new');
    assert.equal(lstatSync(lock).isSymbolicLink(), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('build output publication preserves primary and lock cleanup errors', async () => {
  const root = mkdtempSync(join(tmpdir(), 'autodraw-publish-cleanup-failure-'));
  const output = join(root, 'dist');
  const temporaryOutput = join(root, '.dist.tmp');
  await mkdir(output);
  await mkdir(temporaryOutput);
  let renameCount = 0;

  try {
    await assert.rejects(
      () => publishBuildOutput({
        output,
        temporaryOutput,
        renameImpl: async (source, destination) => {
          renameCount += 1;
          if (renameCount === 2) throw new Error('publish failed');
          await rename(source, destination);
        },
        unlinkImpl: async () => {
          throw new Error('lock cleanup failed');
        },
      }),
      error => {
        assert.equal(error instanceof AggregateError, true);
        assert.deepEqual(error.errors.map(item => item.message), [
          'publish failed',
          'lock cleanup failed',
        ]);
        return true;
      },
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('failed build cleanup preserves both the primary and cleanup errors', async () => {
  const primaryError = new Error('asset copy failed');
  await assert.rejects(
    () => cleanupFailedBuildOutput({
      temporaryOutput: '/tmp/autodraw-cleanup-failure',
      primaryError,
      rmImpl: async () => {
        throw new Error('temporary cleanup failed');
      },
    }),
    error => {
      assert.equal(error instanceof AggregateError, true);
      assert.deepEqual(error.errors.map(item => item.message), [
        'asset copy failed',
        'temporary cleanup failed',
      ]);
      return true;
    },
  );
});