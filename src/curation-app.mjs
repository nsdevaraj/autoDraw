import {
  buildCuratedManifest,
  CANDIDATE_DECISIONS,
  createCurationState,
  getCurationStats,
  nextClassAfterRemoval,
  restoreCurationState,
  setCandidateDecision,
  setClassReviewed,
} from './curation-state.mjs';
import { parseLineList } from './line-list.mjs';

const MANIFEST_URL = './data/quickdraw-candidates.json';
const MVP_CLASSES_URL = './config/mvp-classes.txt';
const DECISION_CONTROLS = Object.freeze([
  Object.freeze({ value: CANDIDATE_DECISIONS.ACCEPTED, label: 'Accept', buttonClass: 'accept' }),
  Object.freeze({ value: CANDIDATE_DECISIONS.REJECTED, label: 'Reject', buttonClass: 'reject' }),
]);

const elements = {
  candidateGrid: document.getElementById('candidateGrid'),
  candidateScroll: document.getElementById('candidateScroll'),
  classList: document.getElementById('classList'),
  classMeta: document.getElementById('classMeta'),
  className: document.getElementById('className'),
  classSearch: document.getElementById('classSearch'),
  curationPanel: document.getElementById('curationPanel'),
  decisionStats: document.getElementById('decisionStats'),
  exportCuratedButton: document.getElementById('exportCuratedButton'),
  exportProgressButton: document.getElementById('exportProgressButton'),
  importButton: document.getElementById('importButton'),
  importFile: document.getElementById('importFile'),
  nextButton: document.getElementById('nextButton'),
  previousButton: document.getElementById('previousButton'),
  progressFill: document.getElementById('progressFill'),
  progressText: document.getElementById('progressText'),
  queueLabel: document.getElementById('queueLabel'),
  resetButton: document.getElementById('resetButton'),
  reviewButton: document.getElementById('reviewButton'),
  selectionSummary: document.getElementById('selectionSummary'),
  sourceLabel: document.getElementById('sourceLabel'),
  stateMessage: document.getElementById('stateMessage'),
  statePanel: document.getElementById('statePanel'),
  toastRegion: document.getElementById('toastRegion'),
};

let manifest;
let mvpClasses;
let curationState;
let storageKey;
let activeClassName;
let scope = 'mvp';
let statusFilter = 'all';
let storageAvailable = true;
let storageWarningShown = false;

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast${type === 'error' ? ' error' : ''}`;
  toast.textContent = message;
  elements.toastRegion.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}

async function fetchText(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.text();
}

function persistState() {
  if (!storageAvailable) return;
  try {
    localStorage.setItem(storageKey, JSON.stringify(curationState));
    localStorage.setItem(`${storageKey}:active`, activeClassName ?? '');
  } catch {
    storageAvailable = false;
    showStorageWarning('Progress could not be saved locally');
  }
}

function showStorageWarning(message) {
  if (storageWarningShown) return;
  storageWarningShown = true;
  showToast(message, 'error');
}

function readLocalValue(key) {
  if (!storageAvailable) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    storageAvailable = false;
    showStorageWarning('Saved progress could not be read; local persistence is disabled');
    return null;
  }
}

function restoreLocalState() {
  const stored = readLocalValue(storageKey);
  if (!stored) return createCurationState(manifest);

  try {
    return restoreCurationState(manifest, stored);
  } catch (error) {
    showToast(error.message, 'error');
    return createCurationState(manifest);
  }
}

function visibleClasses() {
  const reviewed = new Set(curationState.reviewedClasses);
  const query = elements.classSearch.value.trim().toLowerCase();
  return manifest.classes.filter(item => {
    if (scope === 'mvp' && !mvpClasses.has(item.name)) return false;
    if (statusFilter === 'pending' && reviewed.has(item.name)) return false;
    if (statusFilter === 'reviewed' && !reviewed.has(item.name)) return false;
    return !query || item.name.toLowerCase().includes(query) || item.aliases.some(alias => alias.toLowerCase().includes(query));
  });
}

function activeClass() {
  return manifest.classes.find(item => item.name === activeClassName);
}

function classDecisionCounts(item) {
  const decisions = Object.values(curationState.decisions[item.name] ?? {});
  return {
    accepted: decisions.filter(decision => decision === CANDIDATE_DECISIONS.ACCEPTED).length,
    decided: decisions.length,
  };
}

function findClassRow(className) {
  return [...elements.classList.querySelectorAll('[data-class-name]')]
    .find(row => row.dataset.className === className);
}

function selectClass(className, { restoreFocus = false } = {}) {
  activeClassName = className;
  persistState();
  render();
  elements.candidateScroll.scrollTop = 0;
  if (className === null) elements.classList.querySelector('.empty-list')?.focus();
  else if (restoreFocus) findClassRow(className)?.focus();
}

function renderProgress() {
  const stats = getCurationStats(manifest, curationState);
  const percent = stats.classCount === 0 ? 0 : stats.reviewedClassCount / stats.classCount * 100;
  elements.progressText.textContent = `${stats.reviewedClassCount} of ${stats.classCount} classes reviewed`;
  elements.decisionStats.textContent = `${stats.acceptedCandidateCount} accepted · ${stats.rejectedCandidateCount} rejected`;
  elements.progressFill.style.width = `${percent}%`;
}

function renderClassList() {
  const items = visibleClasses();
  const reviewed = new Set(curationState.reviewedClasses);
  elements.classList.replaceChildren();
  elements.queueLabel.textContent = `${items.length} ${items.length === 1 ? 'class' : 'classes'}`;

  if (items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-list';
    empty.tabIndex = -1;
    empty.setAttribute('role', 'status');
    empty.textContent = 'No classes match';
    elements.classList.appendChild(empty);
    return;
  }

  for (const item of items) {
    const counts = classDecisionCounts(item);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `class-row${item.name === activeClassName ? ' active' : ''}${reviewed.has(item.name) ? ' reviewed' : ''}`;
    button.dataset.className = item.name;
    button.setAttribute('aria-current', item.name === activeClassName ? 'true' : 'false');
    button.setAttribute(
      'aria-label',
      `${item.name}, ${reviewed.has(item.name) ? 'reviewed' : 'pending'}, ${counts.accepted} accepted of ${item.candidates.length}`,
    );

    const dot = document.createElement('span');
    dot.className = 'class-dot';
    dot.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span');
    name.className = 'class-name';
    name.textContent = item.name;
    const count = document.createElement('span');
    count.className = 'class-count';
    count.textContent = `${counts.accepted}/${item.candidates.length}`;
    button.append(dot, name, count);
    elements.classList.appendChild(button);
  }
}

function candidateSourceLabel(candidate) {
  return `${candidate.name} from ${candidate.category}, ${candidate.pack}`;
}

function makeDecisionButton(definition, currentDecision, candidate) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `decision-button ${definition.buttonClass}`;
  button.dataset.candidateId = String(candidate.id);
  button.dataset.decision = definition.value;
  button.setAttribute('aria-pressed', String(currentDecision === definition.value));
  button.setAttribute('aria-label', `${definition.label} ${candidateSourceLabel(candidate)}`);
  button.textContent = definition.label;
  return button;
}

function makeCandidateCard(item, candidate) {
  const decision = curationState.decisions[item.name]?.[candidate.id] ?? null;
  const card = document.createElement('article');
  card.className = `candidate-card${decision ? ` ${decision}` : ''}`;

  const preview = document.createElement('a');
  preview.className = 'candidate-preview';
  preview.href = candidate.url;
  preview.target = '_blank';
  preview.rel = 'noreferrer';
  preview.title = `Open ${candidate.name}`;
  preview.setAttribute('aria-label', `Open ${candidateSourceLabel(candidate)}`);
  const image = document.createElement('img');
  image.src = candidate.url;
  image.alt = candidate.name;
  image.loading = 'lazy';
  image.addEventListener('error', () => image.classList.add('failed'));
  const failure = document.createElement('div');
  failure.className = 'image-failure';
  failure.textContent = 'Preview unavailable';
  preview.append(image, failure);

  const body = document.createElement('div');
  body.className = 'candidate-body';
  const name = document.createElement('div');
  name.className = 'candidate-name';
  name.title = candidate.name;
  name.textContent = candidate.name;
  const source = document.createElement('div');
  source.className = 'candidate-source';
  source.title = `${candidate.category} / ${candidate.pack}`;
  source.textContent = `${candidate.category} / ${candidate.pack}`;
  const score = document.createElement('div');
  score.className = 'candidate-score';
  score.textContent = `Match ${candidate.score}`;
  body.append(name, source, score);

  const decisions = document.createElement('div');
  decisions.className = 'decision-row';
  decisions.setAttribute('role', 'group');
  decisions.setAttribute('aria-label', `Decision for ${candidateSourceLabel(candidate)}`);
  decisions.append(...DECISION_CONTROLS.map(definition => (
    makeDecisionButton(definition, decision, candidate)
  )));
  card.append(preview, body, decisions);
  return card;
}

function renderActiveClass() {
  const item = activeClass();
  if (!item) {
    elements.curationPanel.classList.add('hidden');
    elements.statePanel.classList.remove('hidden');
    elements.stateMessage.textContent = 'Select a class';
    return;
  }

  elements.statePanel.classList.add('hidden');
  elements.curationPanel.classList.remove('hidden');
  elements.className.textContent = item.name;
  elements.classMeta.textContent = `${item.candidates.length} candidates${item.aliases.length ? ` · aliases: ${item.aliases.join(', ')}` : ''}`;
  const isReviewed = curationState.reviewedClasses.includes(item.name);
  elements.reviewButton.classList.toggle('reviewed', isReviewed);
  elements.reviewButton.textContent = isReviewed ? 'Reviewed' : 'Mark reviewed';
  elements.reviewButton.setAttribute('aria-pressed', String(isReviewed));

  elements.candidateGrid.replaceChildren(...item.candidates.map(candidate => makeCandidateCard(item, candidate)));
  const counts = classDecisionCounts(item);
  elements.selectionSummary.textContent = `${counts.accepted} accepted · ${counts.decided} of ${item.candidates.length} decided`;

  const items = visibleClasses();
  const index = items.findIndex(candidate => candidate.name === item.name);
  elements.previousButton.disabled = index <= 0;
  elements.nextButton.disabled = index < 0 || index >= items.length - 1;
}

function renderFilters() {
  document.querySelectorAll('[data-scope]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.scope === scope));
  });
  document.querySelectorAll('[data-status]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.status === statusFilter));
  });
}

function ensureVisibleActiveClass(items) {
  if (activeClassName && items.some(item => item.name === activeClassName)) return false;
  activeClassName = items[0]?.name ?? null;
  persistState();
  return true;
}

function render() {
  const items = visibleClasses();
  const activeClassChanged = ensureVisibleActiveClass(items);
  renderFilters();
  renderProgress();
  renderClassList();
  renderActiveClass();
  if (activeClassChanged) elements.candidateScroll.scrollTop = 0;
}

function moveActive(offset) {
  const items = visibleClasses();
  const index = items.findIndex(item => item.name === activeClassName);
  const next = items[index + offset];
  if (next) selectClass(next.name);
}

function downloadJson(filename, value) {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.hidden = true;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function wireEvents() {
  elements.classList.addEventListener('click', event => {
    const row = event.target.closest('[data-class-name]');
    if (row) selectClass(row.dataset.className, { restoreFocus: true });
  });

  elements.candidateGrid.addEventListener('click', event => {
    const button = event.target.closest('[data-candidate-id][data-decision]');
    const item = activeClass();
    if (!button || !item) return;
    const candidateId = Number(button.dataset.candidateId);
    const decision = button.dataset.decision;
    const current = curationState.decisions[item.name]?.[candidateId] ?? null;
    curationState = setCandidateDecision(
      curationState,
      item.name,
      candidateId,
      current === decision ? null : decision,
    );
    persistState();
    render();
    elements.candidateGrid
      .querySelector(`[data-candidate-id="${candidateId}"][data-decision="${decision}"]`)
      ?.focus();
  });

  document.querySelectorAll('[data-scope]').forEach(button => {
    button.addEventListener('click', () => {
      scope = button.dataset.scope;
      render();
    });
  });
  document.querySelectorAll('[data-status]').forEach(button => {
    button.addEventListener('click', () => {
      statusFilter = button.dataset.status;
      render();
    });
  });
  elements.classSearch.addEventListener('input', render);

  elements.reviewButton.addEventListener('click', () => {
    const item = activeClass();
    if (!item) return;
    const classNamesBeforeReview = visibleClasses().map(candidate => candidate.name);
    const reviewed = !curationState.reviewedClasses.includes(item.name);
    curationState = setClassReviewed(
      curationState,
      item.name,
      reviewed,
    );
    const leavesCurrentFilter = (statusFilter === 'pending' && reviewed)
      || (statusFilter === 'reviewed' && !reviewed);
    if (leavesCurrentFilter) {
      selectClass(nextClassAfterRemoval(classNamesBeforeReview, item.name));
      return;
    }
    persistState();
    render();
  });

  elements.previousButton.addEventListener('click', () => moveActive(-1));
  elements.nextButton.addEventListener('click', () => moveActive(1));

  elements.importButton.addEventListener('click', () => elements.importFile.click());
  elements.importFile.addEventListener('change', async () => {
    const file = elements.importFile.files[0];
    if (!file) return;
    try {
      curationState = restoreCurationState(manifest, await file.text());
      persistState();
      render();
      showToast('Progress imported');
    } catch (error) {
      showToast(error.message, 'error');
    } finally {
      elements.importFile.value = '';
    }
  });

  elements.exportProgressButton.addEventListener('click', () => {
    downloadJson(`quickdraw-curation-${manifest.fingerprint.slice(0, 7)}.json`, curationState);
  });
  elements.exportCuratedButton.addEventListener('click', () => {
    downloadJson(`quickdraw-icons-${manifest.fingerprint.slice(0, 7)}.json`, buildCuratedManifest(manifest, curationState));
  });

  elements.resetButton.addEventListener('click', () => {
    if (!window.confirm('Reset all curation progress for this manifest?')) return;
    curationState = createCurationState(manifest);
    persistState();
    render();
    showToast('Progress reset');
  });
}

async function boot() {
  try {
    const [manifestText, mvpText] = await Promise.all([
      fetchText(MANIFEST_URL),
      fetchText(MVP_CLASSES_URL),
    ]);
    manifest = JSON.parse(manifestText);
    mvpClasses = new Set(parseLineList(mvpText));
    storageKey = `autodraw-curation-v1:${manifest.fingerprint}`;
    curationState = restoreLocalState();
    activeClassName = readLocalValue(`${storageKey}:active`) || null;
    elements.sourceLabel.textContent = manifest.source.commit.slice(0, 7);
    wireEvents();
    [
      elements.importButton,
      elements.exportProgressButton,
      elements.exportCuratedButton,
      elements.resetButton,
    ].forEach(button => { button.disabled = false; });
    render();
  } catch (error) {
    elements.stateMessage.textContent = `Could not load curation data: ${error.message}`;
    document.querySelector('.spinner')?.remove();
  }
}

boot();