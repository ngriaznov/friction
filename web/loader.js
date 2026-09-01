// The embeddable friction wasm wrapper: fetches the three heavy NLP
// assets (tagger weights, parser weights, DMS index), reports loading
// progress to the hosting page, initializes the engine, and returns a
// structured API over fix/check/explain.
//
// A hosting site consumes this module two ways, both public:
//   const engine = await loadFriction({ onEvent: (e) => ... });
// or, without holding the promise, by listening on `window`:
//   window.addEventListener("friction:progress", (e) => ...e.detail...);
//   window.addEventListener("friction:ready", ...);
//   window.addEventListener("friction:error", ...);
//
// Byte-fence note: bytes are handed to init_engine() RAW — gzip-vs-.bin
// format detection happens engine-side (see friction-nlp's
// weights_install.rs::is_gzip). Decompressing here would silently change
// what the engine ends up loading, so this module never touches asset
// bytes beyond concatenating the fetched chunks.

// `expectedBytes` seeds overall-progress totals until a real
// content-length arrives; each is replaced by the true size the moment
// the response headers land, so it only has to be the right magnitude.
const ASSETS = [
  {
    key: "tagger",
    label: "tagger weights",
    path: "crates/friction-nlp/weights/perceptron_en.json.gz",
    filename: "perceptron_en.json.gz",
    expectedBytes: 1_400_000,
  },
  {
    key: "parser",
    label: "parser weights",
    path: "crates/friction-nlp/weights/parser_en.json.gz",
    filename: "parser_en.json.gz",
    expectedBytes: 6_400_000,
  },
  {
    key: "dms",
    label: "DMS index",
    path: "crates/friction-packs/packs/dms-index-en-v1.toml",
    filename: "dms-index-en-v1.toml",
    expectedBytes: 2_600_000,
  },
];

const GITHUB_RAW_TEMPLATE =
  "https://raw.githubusercontent.com/ngriaznov/friction/v{version}/";

const LOCAL_PACKS_BASE = "./packs/";
const LOCAL_PROBE_URL = "./packs/dms-index-en-v1.toml";

// Resolves which base URL to fetch the three assets from, and whether that
// base uses the flat local-dev filename convention or the nested
// repo-relative path convention (GitHub raw, an explicit `baseUrl` option,
// and any ?packs= override — all are assumed to mirror the repo layout,
// since an override is pointing at an alternate mirror of it).
async function resolvePackBase(version, explicitBase) {
  if (explicitBase) {
    const base = explicitBase.endsWith("/") ? explicitBase : `${explicitBase}/`;
    return { base, flat: false };
  }
  const params = new URLSearchParams(window.location.search);
  const override = params.get("packs");
  if (override) {
    const base = override.endsWith("/") ? override : `${override}/`;
    return { base, flat: false };
  }
  try {
    const res = await fetch(LOCAL_PROBE_URL, { method: "HEAD" });
    if (res.ok) {
      return { base: LOCAL_PACKS_BASE, flat: true };
    }
  } catch {
    // No local dev server, or no packs staged there — fall through to the
    // GitHub default below.
  }
  return { base: GITHUB_RAW_TEMPLATE.replace("{version}", version), flat: false };
}

function assetUrl(base, flat, asset) {
  return base + (flat ? asset.filename : asset.path);
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch failed (${res.status} ${res.statusText}) for ${url}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress(buf.length, buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

// Opens (creating if needed) this version's asset cache, and evicts every
// other version's cache — a stale cached pack must never outlive the
// engine build it was fetched for.
async function openAssetCache(version) {
  if (!("caches" in window)) return null;
  const name = `friction-packs-v${version}`;
  const keys = await caches.keys();
  await Promise.all(
    keys
      .filter((key) => key.startsWith("friction-packs-") && key !== name)
      .map((key) => caches.delete(key)),
  );
  return caches.open(name);
}

async function loadAsset(cache, url, onProgress) {
  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      const buf = new Uint8Array(await cached.arrayBuffer());
      onProgress(buf.length, buf.length, true);
      return { bytes: buf, fromCache: true };
    }
  }
  const bytes = await fetchWithProgress(url, (loaded, total) =>
    onProgress(loaded, total, false),
  );
  if (cache) {
    await cache.put(url, new Response(bytes));
  }
  return { bytes, fromCache: false };
}

// --- Line diff (public: the `diff` field of `engine.fix()`'s result) ---

/**
 * Line-level diff between two texts as a flat edit script:
 * `[{ type: "equal" | "del" | "add", line }]` in output order. Standard
 * LCS, O(lines²) — sized for playground documents, not multi-MB inputs.
 */
export function diffLines(a, b) {
  if (a === b) {
    return a.split("\n").map((line) => ({ type: "equal", line }));
  }
  const aLines = a.split("\n");
  const bLines = b.split("\n");
  const n = aLines.length;
  const m = bLines.length;

  // Standard LCS DP table, walked backward to recover the edit script.
  const dp = new Array(n + 1);
  for (let i = 0; i <= n; i++) dp[i] = new Uint32Array(m + 1);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = aLines[i] === bLines[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (aLines[i] === bLines[j]) {
      ops.push({ type: "equal", line: aLines[i] });
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", line: aLines[i] });
      i += 1;
    } else {
      ops.push({ type: "add", line: bLines[j] });
      j += 1;
    }
  }
  while (i < n) {
    ops.push({ type: "del", line: aLines[i] });
    i += 1;
  }
  while (j < m) {
    ops.push({ type: "add", line: bLines[j] });
    j += 1;
  }
  return ops;
}

// --- Loading events ---

// Every lifecycle event goes to the caller's onEvent AND out on `window`
// as a CustomEvent, so a hosting page can drive a splash screen or
// progress bar from a script that never touches this module's promise.
// DOM event names: asset/engine progress -> "friction:progress",
// terminal states -> "friction:ready" / "friction:error".
function makeEmitter(onEvent) {
  return (event) => {
    if (onEvent) onEvent(event);
    const domName =
      event.type === "ready"
        ? "friction:ready"
        : event.type === "error"
          ? "friction:error"
          : "friction:progress";
    window.dispatchEvent(new CustomEvent(domName, { detail: event }));
  };
}

// --- Structured fix ---

// Aggregates explain()'s per-pass fired operations into a caller-friendly
// rule tally. The raw per-pass byte offsets stay behind `explain()` on
// purpose: they index each PASS'S OWN input, not the original document —
// a coordinate system that has burned every consumer that treated it as
// document offsets. The `diff` field is the "where"; this is the "why".
function aggregateFired(report) {
  const counts = new Map();
  for (const pass of report.passes) {
    for (const op of pass.fired) {
      const key = `${pass.pass} ${op.rule}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([key, count]) => {
      const split = key.indexOf(" ");
      return { pass: Number(key.slice(0, split)), rule: key.slice(split + 1), count };
    })
    .sort((x, y) => x.pass - y.pass || x.rule.localeCompare(y.rule));
}

function bindEngine(wasm, base, version) {
  return {
    /**
     * Structured fix: `{ input, output, changed, diff, fired, suggestions }`.
     * `diff` is a line-level edit script in final-document coordinates
     * (see `diffLines`); `fired` is `[{ pass, rule, count }]` — which
     * rules produced the change, from the engine's own explain report.
     * `suggestions` is `[{ rule, start, end, line, column, message }]` —
     * every held candidate still present in the fixed output (the same
     * rows `friction fix --suggest` lists), positioned against `output`,
     * present whether or not anything changed.
     */
    fix(input) {
      const { output, suggestions } = JSON.parse(wasm.suggest_text(input));
      const changed = output !== input;
      const diff = diffLines(input, output);
      const fired = changed
        ? aggregateFired(JSON.parse(wasm.explain_text(input)))
        : [];
      return { input, output, changed, diff, fired, suggestions };
    },
    /** `friction fix --suggest`'s fixed output + suggestion rows, parsed. */
    suggest(input) {
      return JSON.parse(wasm.suggest_text(input));
    },
    /** `friction check --format json`, parsed. */
    check(input) {
      return JSON.parse(wasm.check_text(input));
    },
    /** `friction explain --format json`, parsed (per-pass fired/held). */
    explain(input) {
      return JSON.parse(wasm.explain_text(input));
    },
    // The raw string-in/string-out exports, for callers that want the
    // engine's exact JSON text (hashing, piping) rather than objects.
    fixText: wasm.fix_text,
    suggestText: wasm.suggest_text,
    checkText: wasm.check_text,
    explainText: wasm.explain_text,
    engineVersion: wasm.engine_version,
    base,
    version,
  };
}

/**
 * Loads the wasm module, fetches the three pack assets (cache-first),
 * initializes the engine, and resolves to the bound engine API.
 *
 * Options:
 * - `baseUrl`: fetch assets from this base (a mirror of the repo layout)
 *   instead of the default resolution (?packs= query, then local
 *   ./packs/, then raw.githubusercontent.com at the engine's version tag).
 * - `onEvent`: called with every lifecycle event. Event shapes:
 *   `{ type: "asset:start", key, label, url }`
 *   `{ type: "asset:progress", key, label, loaded, total, fromCache,
 *      overallLoaded, overallTotal }` — totals fall back to documented
 *      approximate sizes until a content-length arrives, so overall
 *      progress is always renderable
 *   `{ type: "asset:done", key, label, bytes, fromCache }`
 *   `{ type: "engine:init" }` — all bytes down, engine building
 *   `{ type: "ready", version, engineVersion, base }`
 *   `{ type: "error", message, key? }`
 * Every event is also dispatched on `window` (see makeEmitter above), so
 * hosting pages can subscribe without importing this module.
 *
 * Throws (after emitting the `error` event) on any failed fetch or
 * engine-init failure, naming the failing asset and base.
 */
export async function loadFriction({ baseUrl, onEvent } = {}) {
  const emit = makeEmitter(onEvent);

  try {
    const wasm = await import("./pkg/friction_wasm.js");
    await wasm.default();

    const pkg = await fetch("./pkg/package.json").then((res) => res.json());
    const version = pkg.version;
    const { base, flat } = await resolvePackBase(version, baseUrl);
    const cache = await openAssetCache(version);

    const totals = new Map(
      ASSETS.map((asset) => [asset.key, asset.expectedBytes]),
    );
    const loadedByKey = new Map(ASSETS.map((asset) => [asset.key, 0]));
    const overall = () => {
      let loaded = 0;
      let total = 0;
      for (const asset of ASSETS) {
        loaded += loadedByKey.get(asset.key);
        total += totals.get(asset.key);
      }
      return { loaded, total };
    };

    const bytes = {};
    for (const asset of ASSETS) {
      const url = assetUrl(base, flat, asset);
      emit({ type: "asset:start", key: asset.key, label: asset.label, url });
      let result;
      try {
        result = await loadAsset(cache, url, (loaded, total, fromCache) => {
          if (total > 0) totals.set(asset.key, total);
          loadedByKey.set(asset.key, loaded);
          const sum = overall();
          emit({
            type: "asset:progress",
            key: asset.key,
            label: asset.label,
            loaded,
            total: totals.get(asset.key),
            fromCache,
            overallLoaded: sum.loaded,
            overallTotal: sum.total,
          });
        });
      } catch (err) {
        throw Object.assign(
          new Error(`could not load ${asset.label} from ${url}: ${err.message}`),
          { assetKey: asset.key },
        );
      }
      bytes[asset.key] = result.bytes;
      totals.set(asset.key, result.bytes.length);
      loadedByKey.set(asset.key, result.bytes.length);
      emit({
        type: "asset:done",
        key: asset.key,
        label: asset.label,
        bytes: result.bytes.length,
        fromCache: result.fromCache,
      });
    }

    emit({ type: "engine:init" });
    wasm.init_engine(bytes.tagger, bytes.parser, bytes.dms);

    const engine = bindEngine(wasm, base, version);
    emit({
      type: "ready",
      version,
      engineVersion: engine.engineVersion(),
      base,
    });
    return engine;
  } catch (err) {
    emit({
      type: "error",
      message: err.message ?? String(err),
      key: err.assetKey,
    });
    throw err;
  }
}
