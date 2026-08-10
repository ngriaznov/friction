#!/usr/bin/env bash
# Copies the three heavy assets web/loader.js fetches at runtime into
# web/packs/ (flat — matches loader.js's local-dev URL convention, which is
# deliberately simpler than the nested repo-relative paths its GitHub-raw
# fallback uses), so a local `python3 -m http.server --directory web` can
# serve them from ./packs/ instead of hitting raw.githubusercontent.com.
# Plain cp — the assets are fetched and installed raw (gzip vs .bin
# auto-detected engine-side), so staging them gzipped-as-is is required,
# not incidental.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$repo_root/web/packs"

mkdir -p "$dest"

cp "$repo_root/crates/friction-nlp/weights/perceptron_en.json.gz" "$dest/perceptron_en.json.gz"
cp "$repo_root/crates/friction-nlp/weights/parser_en.json.gz" "$dest/parser_en.json.gz"
cp "$repo_root/crates/friction-packs/packs/dms-index-en-v1.toml" "$dest/dms-index-en-v1.toml"

echo "staged local packs into $dest"
