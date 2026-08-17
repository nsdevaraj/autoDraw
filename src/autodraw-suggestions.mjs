import { assertApprovedIconSource } from './icon-candidate.mjs';
import { retrieveIcons } from './icon-retrieval.mjs';

const APPROVED_INDEX = Symbol('approved-candidate-index');

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function validatedCandidate(candidate, className, source) {
  if (
    !isRecord(candidate)
    || !Number.isInteger(candidate.id)
    || typeof candidate.name !== 'string'
    || candidate.name.length === 0
    || typeof candidate.category !== 'string'
    || typeof candidate.pack !== 'string'
    || !Number.isFinite(candidate.score)
  ) {
    throw new Error(`Invalid candidate for ${className}`);
  }
  assertApprovedIconSource(candidate, `Candidate ${className}/${candidate.id}`, source);
  return Object.freeze({
    id: candidate.id,
    name: candidate.name,
    category: candidate.category,
    pack: candidate.pack,
    path: candidate.path,
    score: candidate.score,
    url: candidate.url,
  });
}

export function createApprovedCandidateIndex(manifest, classLabels) {
  if (
    !isRecord(manifest)
    || manifest.schemaVersion !== 1
    || typeof manifest.fingerprint !== 'string'
    || manifest.fingerprint.length === 0
    || !Array.isArray(manifest.classes)
  ) {
    throw new Error('Invalid Quick Draw candidate manifest');
  }
  if (
    !Array.isArray(classLabels)
    || classLabels.length < 1
    || classLabels.some(label => typeof label !== 'string' || label.length === 0)
    || new Set(classLabels).size !== classLabels.length
  ) {
    throw new Error('Classifier labels must be unique non-empty strings');
  }

  const manifestClasses = new Map();
  for (const manifestClass of manifest.classes) {
    if (!isRecord(manifestClass) || typeof manifestClass.name !== 'string') {
      throw new Error('Candidate manifest contains an invalid class');
    }
    if (manifestClasses.has(manifestClass.name)) {
      throw new Error(`Candidate manifest contains duplicate class: ${manifestClass.name}`);
    }
    manifestClasses.set(manifestClass.name, manifestClass);
  }

  const candidatesByLabel = {};
  for (const label of classLabels) {
    const manifestClass = manifestClasses.get(label);
    if (!manifestClass || !Array.isArray(manifestClass.candidates) || manifestClass.candidates.length === 0) {
      throw new Error(`Classifier class ${label} has no candidate icons`);
    }
    const candidateIds = new Set();
    candidatesByLabel[label] = Object.freeze(manifestClass.candidates.map(candidate => {
      const validated = validatedCandidate(candidate, label, manifest.source);
      if (candidateIds.has(validated.id)) {
        throw new Error(`Duplicate candidate ${label}/${validated.id}`);
      }
      candidateIds.add(validated.id);
      return validated;
    }));
  }

  return Object.freeze({
    [APPROVED_INDEX]: true,
    manifestFingerprint: manifest.fingerprint,
    classLabels: Object.freeze([...classLabels]),
    candidatesByLabel: Object.freeze(candidatesByLabel),
  });
}

export function suggestApprovedCandidates(index, predictions, {
  limit = Number.POSITIVE_INFINITY,
  perClass = Number.POSITIVE_INFINITY,
} = {}) {
  if (!index?.[APPROVED_INDEX]) throw new Error('Invalid approved candidate index');
  if (limit !== Number.POSITIVE_INFINITY && (!Number.isInteger(limit) || limit < 1)) {
    throw new Error('Suggestion limit must be positive');
  }
  if (
    perClass !== Number.POSITIVE_INFINITY
    && (!Number.isInteger(perClass) || perClass < 1)
  ) {
    throw new Error('Suggestions per class must be positive');
  }
  if (!Array.isArray(predictions)) throw new Error('Predictions must be an array');

  const seenLabels = new Set();
  const validatedPredictions = predictions.map(prediction => {
    if (
      !isRecord(prediction)
      || typeof prediction.label !== 'string'
      || !Number.isFinite(prediction.probability)
      || prediction.probability < 0
      || prediction.probability > 1
    ) {
      throw new Error('Prediction probability must be a finite value from 0 to 1');
    }
    if (!index.candidatesByLabel[prediction.label]) {
      throw new Error(`No approved candidates for prediction: ${prediction.label}`);
    }
    return prediction;
  }).filter(prediction => {
    if (seenLabels.has(prediction.label)) return false;
    seenLabels.add(prediction.label);
    return true;
  });

  const suggestions = [];
  const seenCandidates = new Set();
  const maximumCandidates = Math.min(
    perClass,
    Math.max(
      0,
      ...validatedPredictions.map(prediction => index.candidatesByLabel[prediction.label].length),
    ),
  );
  for (let candidateRank = 0; candidateRank < maximumCandidates; candidateRank += 1) {
    for (const prediction of validatedPredictions) {
      const candidate = index.candidatesByLabel[prediction.label][candidateRank];
      if (!candidate || seenCandidates.has(candidate.id)) continue;
      seenCandidates.add(candidate.id);
      suggestions.push(Object.freeze({
        ...candidate,
        label: prediction.label,
        probability: prediction.probability,
        approved: true,
      }));
      if (suggestions.length === limit) return suggestions;
    }
  }
  return suggestions;
}

// Both suggestion sources answer the same question — which approved icons match this
// sketch — so the editor can degrade from retrieval to the manifest without branching.
export function createRetrievalSuggestionSource({ embedder, index, ...defaults }) {
  if (typeof embedder?.embed !== 'function') throw new Error('A sketch embedder is required');
  if (!index) throw new Error('An icon retrieval index is required');

  return Object.freeze({
    kind: 'retrieval',
    coverage: index.counts?.vectors ?? 0,
    async suggest(polylines, options = {}) {
      const embedded = await embedder.embed(polylines, { limit: 5 });
      if (embedded === null) return [];
      return retrieveIcons(index, {
        ...defaults,
        ...options,
        embedding: embedded.embedding,
        predictions: embedded.predictions,
      });
    },
  });
}

export function createCandidateSuggestionSource({ classifier, index }) {
  if (typeof classifier?.classify !== 'function') throw new Error('A classifier is required');
  if (!index?.[APPROVED_INDEX]) throw new Error('Invalid approved candidate index');

  return Object.freeze({
    kind: 'candidates',
    coverage: Object.values(index.candidatesByLabel)
      .reduce((total, candidates) => total + candidates.length, 0),
    async suggest(polylines, options = {}) {
      const predictions = await classifier.classify(polylines, { limit: 5 });
      return suggestApprovedCandidates(index, predictions, options);
    },
  });
}