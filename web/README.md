# friction playground

A static browser page that runs the friction engine entirely client-side,
via WebAssembly. No server, no build step, no dependency beyond a static
file host. Open `index.html`, type or paste text, and run Fix, Check, or
Explain.

Everything under `web/` is plain HTML, CSS, and JavaScript (ES modules).
There is no bundler and nothing to `npm install`.

## Running it locally

From the repository root:

```sh
scripts/stage-local-packs.sh
python3 -m http.server 8080 --directory web
```

Then open `http://localhost:8080`. The wasm build itself (`web/pkg/`) is
produced separately by `wasm-pack`; this page expects it to already exist
at `web/pkg/friction_wasm.js`.

`scripts/stage-local-packs.sh` copies the three engine assets — tagger
weights, parser weights, and the DMS index — into `web/packs/`, so the page
loads them from your checkout instead of over the network. `web/packs/` is
git-ignored; re-run the script whenever those source files change.

## How the pack download works

On first load, the engine needs three files:

| asset | size |
| --- | --- |
| tagger weights | ~1.4 MB |
| parser weights | ~6.4 MB |
| DMS index | ~2.6 MB |

`web/loader.js` resolves where to fetch them from, in this order:

1. A `?packs=<base-url>` query parameter, if present.
2. `web/packs/`, if a local dev server has it staged (see above).
3. Otherwise, `raw.githubusercontent.com`, pinned to the tag matching the
   wasm build's own version (`v{version}` from `web/pkg/package.json`).

GitHub's release-asset URLs are deliberately never used here: they don't
send CORS headers, so a browser page can't fetch them cross-origin. Raw
file URLs do.

Each asset is fetched once and stored in the browser's Cache API, keyed to
the engine version, so a repeat visit skips the network entirely. Caches
from older versions are cleared automatically. A progress bar for each
asset is shown while it downloads, and the editor stays locked until all
three are in place and the engine has initialized.

## Embedding the wrapper

`web/loader.js` is the reusable piece: a hosting site (friction-cli.dev,
for example) imports it, and the demo page here is only its first
consumer.

```js
import { loadFriction } from "./loader.js";

const engine = await loadFriction({
  baseUrl: undefined,      // optional pack mirror; defaults to GitHub raw at the version tag
  onEvent: (e) => { /* loading lifecycle, see below */ },
});

const result = engine.fix(text);
// { input, output, changed,
//   diff:  [{ type: "equal" | "del" | "add", line }],
//   fired: [{ pass, rule, count }] }

engine.check(text);   // `friction check --format json`, parsed
engine.explain(text); // `friction explain --format json`, parsed
```

Every loading event also fires on `window` as a CustomEvent, so a page
can drive a splash screen or progress bar without importing the module:
`friction:progress` (asset downloads and engine init, with per-asset and
overall byte counts), `friction:ready`, and `friction:error`.

`diff` speaks final-document coordinates and is safe to render directly.
`fired` says which rules produced the change. The per-pass byte offsets
in `explain()` index each pass's own input, not the original document —
use them only if you replay passes the way the engine does.

## What the buttons do

- **Fix** runs the full repair engine and shows the original text, the
  fixed text, and a line-level diff between them.
- **Check** runs detection only and lists every flagged span (rule,
  message, and the matched text) without changing anything.
- **Explain** runs the same repair engine as Fix but reports, pass by
  pass, every edit that fired and every candidate a gate held back,
  instead of the fixed text.

All three call straight into the wasm engine: the same one the CLI runs
on, compiled to WebAssembly rather than a native binary. Nothing here
talks to a server.
