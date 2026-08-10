# The WebAssembly playground

This is an architecture record for `crates/friction-wasm` and the browser
build it produces: what the crate exposes, how it gets its weight and pack
data without the native binary's `include_bytes!` path, why that data comes
from `raw.githubusercontent.com` specifically, and how the release pipeline
and a future `friction-cli.dev` deployment fit together. For the engine
itself, see the [README](../README.md) and [docs/OPERATIONS.md](OPERATIONS.md).

## Why a separate crate

The native binary embeds every pack and weight file with `include_bytes!`,
which is why it is deterministic and needs no network at run time (about
59 MB uncompiled, per the README). A wasm module cannot afford the same
choice: `include_bytes!`-ing the parser weights and the DMS index into the
`.wasm` binary would put roughly 30 MB in front of every page load, before
a user has typed anything. `crates/friction-wasm` exists to draw that line
at the crate boundary instead of scattering `#[cfg(target_arch = "wasm32")]`
through the engine crates: it is a thin `wasm-bindgen` wrapper, compiled
with `wasm-pack build crates/friction-wasm --release --target web`, that
turns off the embedding features on `friction-nlp` and `friction-packs` and
installs their weights and index from bytes fetched at page load instead.
Everything downstream of that install call (tagging, parsing, matching,
editing) runs unmodified; the wasm crate's own surface is `fix`/`check`/
`explain` entry points over the same pipeline the CLI drives, plus the
install calls that must run before any of them.

## The byte-installation seams

Two features gate the embedding, one per crate, each off by default only
for the wasm build:

- `friction-nlp`'s `embedded-weights` (on by default) controls whether the
  tagger and parser weights are `include_bytes!`'d. With it off,
  [`install_tagger_weights(bytes: Vec<u8>)`](../crates/friction-nlp/src/tag_perceptron.rs)
  and [`install_parser_weights(bytes: Vec<u8>)`](../crates/friction-nlp/src/dep_perceptron.rs)
  (both in `friction-nlp`, backed by the shared
  [`weights_install`](../crates/friction-nlp/src/weights_install.rs) module)
  supply them instead.
- `friction-packs`'s `embedded-dms` (on by default) controls whether the
  27 MB DMS index binary is embedded. With it off,
  [`install_dms_index_bin(bytes: Vec<u8>)`](../crates/friction-packs/src/registry.rs)
  and `install_dms_index_toml(bytes: Vec<u8>)` (which rebuilds the binary
  view from the TOML source at install time) supply it instead.

All four functions return a `Result`, erroring rather than panicking on a
redundant or too-late install: calling one after the corresponding
tagger, parser, or DMS index has already been built (the embedded default,
or an earlier install) is a caller bug, not a runtime condition to recover
from. With the feature off and no install ever made, the accessor panics
naming the missing call, so a misconfigured wasm build fails loudly on
first use instead of running with an empty model. Each `install_*` must run
before anything in the process touches the corresponding tagger, parser, or
`friction_packs::DMS`, including a caller's own cached singleton, which
is why the wasm crate performs all four installs during its own
initialization, before exposing `fix`, `check`, or `explain` to the page.

Two packs stay embedded regardless: `attestation-en-v1.bin` and
`jargon-attest-en-v1.bin` are `include_bytes!`'d unconditionally in
`friction-packs`, both well under the size that made the tagger, parser,
and DMS index worth gating. The wasm bundle carries them either way.

## Why the models come from raw.githubusercontent.com

The playground fetches the tagger, parser, and DMS artifacts from
`raw.githubusercontent.com` at the release tag rather than shipping them as
release assets or standing up a server, and that choice comes down to
CORS, measured directly against the alternatives:

- A GitHub release-download URL sends no `access-control-allow-origin`
  header, so a browser fetch against it fails outright.
- The `api.github.com` release-asset endpoint does send CORS headers, but
  only on its own `302` response. The redirect target,
  `release-assets.githubusercontent.com`, strips them, so the browser
  still blocks the followed request.
- `raw.githubusercontent.com` sends `access-control-allow-origin: *` on
  the file itself and supports range requests, which is what makes it
  fetchable from a page on any origin and resumable or partially
  cacheable if needed.

That makes a tagged commit on GitHub the artifact host: no server code, no
CDN account, and the same commit the release tag already points at. It
also means the browser build depends on the pack files staying committed
to the repository at each release tag, which they already are as the
source the native build embeds from.

`raw.githubusercontent.com` is the fallback, not the only source:
`web/loader.js` honors an explicit `baseUrl` option first, then a
`?packs=<base-url>` query parameter, then a locally staged `web/packs/`
(see [Local development](#local-development) below), and only then falls
back to GitHub raw, pinned to the tag matching the wasm build's own
version.

## The JS wrapper API

`web/loader.js` is the embeddable half of the wrapper: `loadFriction({
baseUrl, onEvent })` fetches the assets, initializes the engine, and
resolves to the bound API. While it loads, it reports every lifecycle
step — per-asset and overall byte progress, cache hits, engine
construction, ready, error — both to the `onEvent` callback and as
`friction:progress` / `friction:ready` / `friction:error` CustomEvents on
`window`, so a hosting page can drive its own loading UI from either
side. The bound API returns structured results: `fix()` yields `{ input,
output, changed, diff, fired }` — a line-level diff in final-document
coordinates plus a tally of the rules that fired — and
`check()`/`explain()` return the CLI's own JSON shapes, parsed.
`web/README.md` documents the exact event and result shapes.

## Assets and sizes

The playground fetches the smallest form of each artifact that the
matching `install_*` function accepts:

| artifact | source format | size | installed via |
|---|---|---|---|
| tagger weights | `perceptron_en.json.gz` | 1.4 MB | `install_tagger_weights` |
| parser weights | `parser_en.json.gz` | 6.4 MB | `install_parser_weights` |
| DMS index | `dms-index-en-v1.toml` | 2.6 MB | `install_dms_index_toml` |

The tagger and parser installs auto-detect `json.gz` vs. the derived
`.bin` view format by gzip magic, so fetching the audited `json.gz`
interchange format (the same one `crates/friction-nlp/weights/NOTICE.md`
documents provenance for) avoids shipping the larger pre-built `.bin`
artifacts over the network. The DMS index is the reverse trade: its TOML
source is a tenth of the derived `.bin`'s 27 MB, and `install_dms_index_toml`
rebuilds the binary view from it at install time, so fetching TOML and
paying a one-time rebuild beats fetching the prebuilt binary.

## Browser caching

Every fetch is keyed by the release tag it targets (`raw.githubusercontent.com`
serves the pack files as committed at that tag), so the three artifacts
above are immutable for a given version. The playground stores them in the
Cache API under a key that includes the version, so a returning visitor on
the same version pays no network cost after the first load, and a new
release naturally invalidates by fetching a different key rather than by
any explicit cache-busting logic.

## Deployment

The release pipeline (`.github/workflows/release.yml`) builds
`friction-playground-{version}.tar.gz` in its `wasm` job and publishes it
as a release asset alongside the native binaries: `index.html`, `app.js`,
`loader.js`, `style.css`, and the `wasm-pack` output in `pkg/`. Serving it
is deploying a static site: any static host works, `friction-cli.dev`
among them (Cloudflare Pages, at the time of writing). The pack files still
come from `raw.githubusercontent.com` regardless of where `web/` itself is
hosted: there is no server component to deploy alongside it, and the host
never sees or proxies the model data.

## Local development

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack   # or the installer script release.yml uses

wasm-pack build crates/friction-wasm --release --target web --out-dir ../../web/pkg
scripts/stage-local-packs.sh
python3 -m http.server 8080 --directory web
```

A plain HTTP server is enough: nothing in `web/` needs a build step of its
own, since `wasm-pack --target web` already produces a `pkg/` that a
`<script type="module">` can import directly. Serving over `file://` will
not work: the `fetch` calls both to the pack assets and to instantiate the
wasm module itself need a real origin.

`scripts/stage-local-packs.sh` copies the three assets from `crates/`
straight into `web/packs/` (git-ignored), which `web/loader.js` prefers
over `raw.githubusercontent.com` when present, so local iteration never
depends on the current commit already being pushed and tagged. Re-run it
whenever the source weights or DMS TOML change.
