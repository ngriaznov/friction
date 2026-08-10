// Playground UI wiring: loads the engine via loader.js, then drives
// Fix/Check/Explain against whatever's in the textarea.
//
// XSS hygiene: every piece of user text or engine output that reaches the
// DOM goes in via textContent/createTextNode. Nothing here ever assigns
// innerHTML with interpolated content.

import { loadFriction } from "./loader.js";

const els = {
  statusLine: document.getElementById("status-line"),
  versionBadge: document.getElementById("version-badge"),
  progressList: document.getElementById("progress-list"),
  progressWrap: document.getElementById("progress-wrap"),
  editorWrap: document.getElementById("editor-wrap"),
  input: document.getElementById("input"),
  btnFix: document.getElementById("btn-fix"),
  btnCheck: document.getElementById("btn-check"),
  btnExplain: document.getElementById("btn-explain"),
  results: document.getElementById("results"),
};

const progressRows = new Map();

let engine = null;

function setStatus(text, isError) {
  els.statusLine.textContent = text;
  els.statusLine.classList.toggle("status-error", Boolean(isError));
}

function formatBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function ensureProgressRow(key, label) {
  let row = progressRows.get(key);
  if (row) return row;

  const wrap = document.createElement("div");
  wrap.className = "progress-row";

  const name = document.createElement("span");
  name.className = "progress-label";
  name.textContent = label;

  const track = document.createElement("div");
  track.className = "progress-track";
  const bar = document.createElement("div");
  bar.className = "progress-bar";
  track.appendChild(bar);

  const stat = document.createElement("span");
  stat.className = "progress-stat";

  wrap.append(name, track, stat);
  els.progressList.appendChild(wrap);

  row = { bar, stat };
  progressRows.set(key, row);
  return row;
}

// Renders the wrapper's lifecycle events (this page is just the first
// consumer of loader.js's public onEvent stream — a hosting site
// subscribes the same way, or via the window "friction:*" CustomEvents).
function onEngineEvent(event) {
  switch (event.type) {
    case "asset:start":
      ensureProgressRow(event.key, event.label);
      break;
    case "asset:progress": {
      const row = ensureProgressRow(event.key, event.label);
      const pct = event.total > 0 ? Math.min(100, Math.round((event.loaded / event.total) * 100)) : 0;
      row.bar.style.width = `${pct}%`;
      row.stat.textContent = event.fromCache
        ? "cached"
        : `${formatBytes(event.loaded)} / ${formatBytes(event.total)}`;
      break;
    }
    case "asset:done": {
      const row = ensureProgressRow(event.key, event.label);
      row.bar.style.width = "100%";
      row.stat.textContent = event.fromCache ? "cached" : `${formatBytes(event.bytes)} done`;
      break;
    }
    case "engine:init":
      setStatus("initializing engine…");
      break;
    default:
      // "ready"/"error" are handled by boot()'s own control flow.
      break;
  }
}

function clearResults() {
  els.results.textContent = "";
}

// --- Fix: side-by-side original/fixed plus the wrapper's structured
// diff and fired-rule tally (engine.fix()'s own result shape) ---

function renderDiff(ops) {
  const pre = document.createElement("pre");
  pre.className = "diff-view";
  for (const op of ops) {
    const line = document.createElement("div");
    const prefix = op.type === "add" ? "+ " : op.type === "del" ? "- " : "  ";
    line.className = `diff-line diff-${op.type}`;
    line.textContent = prefix + op.line;
    pre.appendChild(line);
  }
  return pre;
}

function renderFixResult(result) {
  clearResults();

  const columns = document.createElement("div");
  columns.className = "fix-columns";

  const originalCol = document.createElement("div");
  originalCol.className = "fix-column";
  const originalHeading = document.createElement("h3");
  originalHeading.textContent = "Original";
  const originalPre = document.createElement("pre");
  originalPre.textContent = result.input;
  originalCol.append(originalHeading, originalPre);

  const fixedCol = document.createElement("div");
  fixedCol.className = "fix-column";
  const fixedHeading = document.createElement("h3");
  fixedHeading.textContent = "Fixed";
  const fixedPre = document.createElement("pre");
  fixedPre.textContent = result.output;
  fixedCol.append(fixedHeading, fixedPre);

  columns.append(originalCol, fixedCol);
  els.results.appendChild(columns);

  if (result.fired.length > 0) {
    const firedHeading = document.createElement("h3");
    firedHeading.textContent = "Rules fired";
    const firedList = document.createElement("ul");
    firedList.className = "fired-list";
    for (const entry of result.fired) {
      const item = document.createElement("li");
      item.textContent =
        entry.count > 1
          ? `pass ${entry.pass}: ${entry.rule} ×${entry.count}`
          : `pass ${entry.pass}: ${entry.rule}`;
      firedList.appendChild(item);
    }
    els.results.append(firedHeading, firedList);
  }

  const diffHeading = document.createElement("h3");
  diffHeading.textContent = "Diff";
  const diffView = renderDiff(result.diff);

  els.results.append(diffHeading, diffView);
}

// --- Check: findings list (rule id, message, span excerpt) ---

function excerptFor(sourceBytes, start, end) {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  const slice = sourceBytes.subarray(Math.max(0, start), Math.max(0, end));
  const text = decoder.decode(slice);
  return text.length > 160 ? `${text.slice(0, 160)}…` : text;
}

function renderCheckResult(source, report) {
  clearResults();

  const sourceBytes = new TextEncoder().encode(source);

  const summary = document.createElement("p");
  summary.className = "check-summary";
  summary.textContent = `genre: ${report.genre} — ${report.spans.length} finding(s)`;
  els.results.appendChild(summary);

  if (report.spans.length === 0) {
    const empty = document.createElement("p");
    empty.textContent = "No findings.";
    els.results.appendChild(empty);
    return;
  }

  const list = document.createElement("ul");
  list.className = "findings-list";

  for (const span of report.spans) {
    const item = document.createElement("li");
    item.className = "finding";

    const head = document.createElement("div");
    head.className = "finding-head";
    const rule = document.createElement("span");
    rule.className = "finding-rule";
    rule.textContent = `${span.channel}: ${span.frame_id}`;
    head.appendChild(rule);
    if (typeof span.score === "number") {
      const score = document.createElement("span");
      score.className = "finding-score";
      score.textContent = `score ${span.score}`;
      head.appendChild(score);
    }
    item.appendChild(head);

    if (span.message) {
      const message = document.createElement("div");
      message.className = "finding-message";
      message.textContent = span.message;
      item.appendChild(message);
    }

    const excerpt = document.createElement("code");
    excerpt.className = "finding-excerpt";
    excerpt.textContent = excerptFor(sourceBytes, span.start, span.end);
    item.appendChild(excerpt);

    const location = document.createElement("div");
    location.className = "finding-location";
    location.textContent = `line ${span.line}, column ${span.column}`;
    item.appendChild(location);

    list.appendChild(item);
  }

  els.results.appendChild(list);
}

// --- Explain: per-pass JSON in a collapsible <pre> ---

function renderExplainResult(report) {
  clearResults();

  const summary = document.createElement("p");
  summary.className = "check-summary";
  summary.textContent = `${report.passes.length} pass(es) — ${report.patches_applied} patch(es) applied`;
  els.results.appendChild(summary);

  for (const pass of report.passes) {
    const details = document.createElement("details");
    details.open = pass.fired.length > 0;
    const summaryEl = document.createElement("summary");
    summaryEl.textContent = `pass ${pass.pass} — fired ${pass.fired.length}, held ${pass.held.length}`;
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(pass, null, 2);
    details.append(summaryEl, pre);
    els.results.appendChild(details);
  }
}

// --- Actions ---

function setButtonsEnabled(enabled) {
  els.btnFix.disabled = !enabled;
  els.btnCheck.disabled = !enabled;
  els.btnExplain.disabled = !enabled;
}

async function runFix() {
  const original = els.input.value;
  setButtonsEnabled(false);
  setStatus("running fix…");
  try {
    const result = engine.fix(original);
    renderFixResult(result);
    setStatus(result.changed ? "fix complete" : "fix complete — no changes");
  } catch (err) {
    setStatus(`fix failed: ${err.message ?? err}`, true);
  } finally {
    setButtonsEnabled(true);
  }
}

async function runCheck() {
  const source = els.input.value;
  setButtonsEnabled(false);
  setStatus("running check…");
  try {
    renderCheckResult(source, engine.check(source));
    setStatus("check complete");
  } catch (err) {
    setStatus(`check failed: ${err.message ?? err}`, true);
  } finally {
    setButtonsEnabled(true);
  }
}

async function runExplain() {
  const source = els.input.value;
  setButtonsEnabled(false);
  setStatus("running explain…");
  try {
    renderExplainResult(engine.explain(source));
    setStatus("explain complete");
  } catch (err) {
    setStatus(`explain failed: ${err.message ?? err}`, true);
  } finally {
    setButtonsEnabled(true);
  }
}

els.btnFix.addEventListener("click", runFix);
els.btnCheck.addEventListener("click", runCheck);
els.btnExplain.addEventListener("click", runExplain);

async function boot() {
  setButtonsEnabled(false);
  setStatus("loading engine…");
  try {
    engine = await loadFriction({ onEvent: onEngineEvent });
    els.progressWrap.hidden = true;
    els.editorWrap.hidden = false;
    els.versionBadge.textContent = `v${engine.version}`;
    setStatus(`engine ready (${engine.engineVersion()})`);
    setButtonsEnabled(true);
    els.input.focus();
  } catch (err) {
    setStatus(`engine load failed: ${err.message ?? err}`, true);
  }
}

boot();
