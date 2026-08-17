const CURATION_SCHEMA_VERSION = 1;
const PROGRESS_KIND = 'quickdraw-curation-progress';
const CURATED_KIND = 'quickdraw-curated-icons';
export const CANDIDATE_DECISIONS = Object.freeze({
  ACCEPTED: 'accepted',
  REJECTED: 'rejected',
});
const DECISIONS = new Set(Object.values(CANDIDATE_DECISIONS));

function assertManifest(manifest) {
  if (
    !manifest
    || manifest.schemaVersion !== 1
    || !manifest.source?.commit
    || typeof manifest.fingerprint !== 'string'
    || !Array.isArray(manifest.classes)
  ) {
    throw new Error('Invalid Quick Draw candidate manifest');
  }
}

function parseStoredState(storedState) {
  if (typeof storedState === 'string') {
    try {
      return JSON.parse(storedState);
    } catch {
      throw new Error('Curation state is not valid JSON');
    }
  }
  return storedState;
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function createCurationState(manifest) {
  assertManifest(manifest);
  return {
    kind: PROGRESS_KIND,
    schemaVersion: CURATION_SCHEMA_VERSION,
    sourceCommit: manifest.source.commit,
    manifestFingerprint: manifest.fingerprint,
    decisions: {},
    reviewedClasses: [],
  };
}

export function restoreCurationState(manifest, storedState) {
  assertManifest(manifest);
  const value = parseStoredState(storedState);
  if (!value || value.schemaVersion !== CURATION_SCHEMA_VERSION) {
    throw new Error(`Unsupported curation schema version: ${value?.schemaVersion}`);
  }
  if (value.kind !== PROGRESS_KIND) {
    throw new Error('Imported file is not Quick Draw curation progress');
  }
  if (value.sourceCommit !== manifest.source.commit) {
    throw new Error('Curation state belongs to a different SVGDepot commit');
  }
  if (value.manifestFingerprint !== manifest.fingerprint) {
    throw new Error('Curation state belongs to a different candidate manifest');
  }
  if (!isRecord(value.decisions)) throw new Error('Curation decisions must be an object');
  if (!Array.isArray(value.reviewedClasses) || value.reviewedClasses.some(name => typeof name !== 'string')) {
    throw new Error('Reviewed classes must be an array of strings');
  }

  const classes = new Map(manifest.classes.map(item => [item.name, item]));
  const decisions = {};
  for (const [className, classDecisions] of Object.entries(value.decisions)) {
    const manifestClass = classes.get(className);
    if (!manifestClass) throw new Error(`Unknown Quick Draw class: ${className}`);
    if (!isRecord(classDecisions)) throw new Error(`Decisions for ${className} must be an object`);
    const candidateIds = new Set(manifestClass.candidates.map(candidate => candidate.id));
    decisions[className] = {};

    for (const [candidateId, decision] of Object.entries(classDecisions ?? {})) {
      const numericCandidateId = Number(candidateId);
      if (!Number.isInteger(numericCandidateId) || String(numericCandidateId) !== candidateId) {
        throw new Error(`Invalid candidate ID for ${className}: ${candidateId}`);
      }
      if (!candidateIds.has(numericCandidateId)) {
        throw new Error(`Unknown candidate ${candidateId} for ${className}`);
      }
      if (!DECISIONS.has(decision)) {
        throw new Error(`Invalid decision for ${className}/${candidateId}: ${decision}`);
      }
      decisions[className][candidateId] = decision;
    }
  }

  const reviewedClasses = [...new Set(value.reviewedClasses)];
  for (const className of reviewedClasses) {
    if (!classes.has(className)) throw new Error(`Unknown reviewed class: ${className}`);
  }

  return {
    kind: PROGRESS_KIND,
    schemaVersion: CURATION_SCHEMA_VERSION,
    sourceCommit: manifest.source.commit,
    manifestFingerprint: manifest.fingerprint,
    decisions,
    reviewedClasses: reviewedClasses.sort((left, right) => left.localeCompare(right)),
  };
}

export function setCandidateDecision(state, className, candidateId, decision) {
  if (decision !== null && !DECISIONS.has(decision)) {
    throw new Error(`Invalid candidate decision: ${decision}`);
  }

  const decisions = { ...state.decisions };
  const classDecisions = { ...(decisions[className] ?? {}) };
  if (decision === null) delete classDecisions[candidateId];
  else classDecisions[candidateId] = decision;

  if (Object.keys(classDecisions).length === 0) delete decisions[className];
  else decisions[className] = classDecisions;

  return { ...state, decisions };
}

export function setClassReviewed(state, className, reviewed) {
  const reviewedClasses = new Set(state.reviewedClasses);
  if (reviewed) reviewedClasses.add(className);
  else reviewedClasses.delete(className);
  return {
    ...state,
    reviewedClasses: [...reviewedClasses].sort((left, right) => left.localeCompare(right)),
  };
}

export function nextClassAfterRemoval(classNames, removedClassName) {
  const removedIndex = classNames.indexOf(removedClassName);
  if (removedIndex < 0) return classNames[0] ?? null;
  const remainingClassNames = classNames.filter(className => className !== removedClassName);
  return remainingClassNames[Math.min(removedIndex, remainingClassNames.length - 1)] ?? null;
}

export function getCurationStats(manifest, state) {
  assertManifest(manifest);
  let acceptedCandidateCount = 0;
  let rejectedCandidateCount = 0;

  for (const classDecisions of Object.values(state.decisions)) {
    for (const decision of Object.values(classDecisions)) {
      if (decision === CANDIDATE_DECISIONS.ACCEPTED) acceptedCandidateCount += 1;
      if (decision === CANDIDATE_DECISIONS.REJECTED) rejectedCandidateCount += 1;
    }
  }

  return {
    classCount: manifest.classes.length,
    reviewedClassCount: state.reviewedClasses.length,
    acceptedCandidateCount,
    rejectedCandidateCount,
  };
}

export function buildCuratedManifest(manifest, state) {
  const restored = restoreCurationState(manifest, state);
  const reviewedClasses = new Set(restored.reviewedClasses);

  return {
    kind: CURATED_KIND,
    schemaVersion: CURATION_SCHEMA_VERSION,
    source: manifest.source,
    candidateManifestFingerprint: manifest.fingerprint,
    classes: manifest.classes
      .map(item => ({
        name: item.name,
        candidates: item.candidates.filter(candidate => (
          restored.decisions[item.name]?.[candidate.id] === CANDIDATE_DECISIONS.ACCEPTED
        )),
      }))
      .filter(item => reviewedClasses.has(item.name) || item.candidates.length > 0),
  };
}