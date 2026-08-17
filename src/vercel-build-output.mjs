import { existsSync, lstatSync, realpathSync } from 'node:fs';
import { lstat, readlink, rename, rm, symlink, unlink } from 'node:fs/promises';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';

function containsPath(parent, child) {
  const relationship = relative(parent, child);
  return relationship === ''
    || (
      relationship !== '..'
      && !relationship.startsWith(`..${sep}`)
      && !isAbsolute(relationship)
    );
}

function canonicalPath(path) {
  let existingPath = resolve(path);
  const missingSegments = [];
  while (!existsSync(existingPath)) {
    const parent = dirname(existingPath);
    if (parent === existingPath) throw new Error(`Cannot resolve Vercel output path: ${path}`);
    missingSegments.unshift(basename(existingPath));
    existingPath = parent;
  }
  return resolve(realpathSync.native(existingPath), ...missingSegments);
}

function isSymbolicLink(path) {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function pathExists(path) {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function lockTarget(path, readlinkImpl = readlink) {
  try {
    return await readlinkImpl(path);
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    if (error?.code === 'EINVAL') return 'invalid';
    throw error;
  }
}

async function acquirePublicationLock({
  lock,
  ownerTarget,
  symlinkImpl,
}) {
  try {
    await symlinkImpl(ownerTarget, lock);
  } catch (error) {
    if (error?.code === 'EEXIST') {
      throw new Error(
        `Vercel output publication lock exists: ${lock}. `
        + 'Remove it only after confirming no build is running.',
      );
    }
    throw error;
  }
}

async function releasePublicationLock({ lock, ownerTarget, unlinkImpl }) {
  const existingTarget = await lockTarget(lock);
  if (existingTarget === null) return;
  if (existingTarget !== ownerTarget) {
    throw new Error('Vercel output publication lock ownership changed');
  }
  await unlinkImpl(lock);
}

function combinedError(primaryError, cleanupError) {
  if (!primaryError) return cleanupError;
  const errors = primaryError instanceof AggregateError
    ? [...primaryError.errors, cleanupError]
    : [primaryError, cleanupError];
  return new AggregateError(errors, 'Vercel output publication and cleanup failed');
}

function publicationOwnerTarget() {
  return `owner-${process.pid}-${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
}

async function transactionError(operation) {
  try {
    await operation();
    return null;
  } catch (error) {
    return error;
  }
}

export function assertSafeBuildOutput({ projectRoot, defaultOutput, output }) {
  const lexicalRoot = resolve(projectRoot);
  const root = canonicalPath(lexicalRoot);
  const allowedOutput = resolve(defaultOutput);
  const resolvedOutput = resolve(output);
  const canonicalOutput = canonicalPath(resolvedOutput);
  const expectedDefaultOutput = resolve(root, relative(lexicalRoot, allowedOutput));
  if (resolvedOutput === allowedOutput) {
    if (
      !isSymbolicLink(resolvedOutput)
      && canonicalOutput === expectedDefaultOutput
      && containsPath(root, canonicalOutput)
    ) return resolvedOutput;
    throw new Error(`Refusing unsafe Vercel output path: ${resolvedOutput}`);
  }
  if (
    containsPath(lexicalRoot, resolvedOutput)
    || containsPath(resolvedOutput, lexicalRoot)
    || containsPath(root, canonicalOutput)
    || containsPath(canonicalOutput, root)
  ) {
    throw new Error(`Refusing unsafe Vercel output path: ${resolvedOutput}`);
  }
  return resolvedOutput;
}

export async function cleanupFailedBuildOutput({
  temporaryOutput,
  primaryError,
  rmImpl = rm,
}) {
  const cleanupError = await transactionError(() => rmImpl(
    resolve(temporaryOutput),
    { recursive: true, force: true },
  ));
  if (cleanupError) throw combinedError(primaryError, cleanupError);
  throw primaryError;
}

export async function publishBuildOutput({
  output,
  temporaryOutput,
  renameImpl = rename,
  rmImpl = rm,
  symlinkImpl = symlink,
  unlinkImpl = unlink,
}) {
  const resolvedOutput = resolve(output);
  const resolvedTemporaryOutput = resolve(temporaryOutput);
  const backup = resolve(
    dirname(resolvedOutput),
    `.${basename(resolvedOutput)}.old`,
  );
  const lock = resolve(
    dirname(resolvedOutput),
    `.${basename(resolvedOutput)}.lock`,
  );
  const ownerTarget = publicationOwnerTarget();
  await acquirePublicationLock({
    lock,
    ownerTarget,
    symlinkImpl,
  });

  const primaryError = await transactionError(async () => {
    const hadBackup = await pathExists(backup);
    const outputBeforeRecovery = await pathExists(resolvedOutput);
    if (hadBackup && outputBeforeRecovery) {
      await rmImpl(backup, { recursive: true, force: true });
    } else if (hadBackup) {
      await renameImpl(backup, resolvedOutput);
    }
    const hadOutput = await pathExists(resolvedOutput);
    if (hadOutput) await renameImpl(resolvedOutput, backup);

    try {
      await renameImpl(resolvedTemporaryOutput, resolvedOutput);
    } catch (error) {
      if (hadOutput && !await pathExists(resolvedOutput)) {
        try {
          await renameImpl(backup, resolvedOutput);
        } catch (rollbackError) {
          throw new AggregateError(
            [error, rollbackError],
            'Vercel output publication and rollback failed',
          );
        }
      }
      throw error;
    }
    if (hadOutput) await rmImpl(backup, { recursive: true, force: true });
  });
  const cleanupError = await transactionError(() => releasePublicationLock({
    lock,
    ownerTarget,
    unlinkImpl,
  }));
  if (cleanupError) throw combinedError(primaryError, cleanupError);
  if (primaryError) throw primaryError;
}