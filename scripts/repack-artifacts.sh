#!/usr/bin/env bash
# Regenerates every derived binary artifact that is a pure function of
# committed repository bytes, in dependency order:
#
#   weights-pack   json.gz weight artifacts  -> friction-nlp/weights/*.bin
#   dms-pack       dms-index-v1.toml         -> dms-index-v1.bin
#   attest-pack    attestation-v1.toml       -> attestation-v1.bin
#   frame-pack     frame-rules-v1.toml (+ DMS TOML + evidence bins)
#                                            -> frame-pack-v1.bin
#
# Deliberately NOT covered: jargon-attest-v1.bin (rebuilt from external
# source dumps by .github/workflows/attest-refresh.yml) and the
# human/machine evidence bins (rebuilt from locally staged external
# corpora by `corpus-tool human-evidence`): neither is a function of
# committed bytes alone.
#
# Run this after editing any source pack TOML or weight artifact; the
# friction-packs drift tests (and CI's artifact-freshness job) fail
# until the derived artifacts match the sources again. Re-running on
# unchanged sources is byte-idempotent. Every builder is deterministic.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo run --release -p corpus-tool -- weights-pack
cargo run --release -p corpus-tool -- dms-pack
cargo run --release -p corpus-tool -- attest-pack
cargo run --release -p corpus-tool -- frame-pack

echo "repack-artifacts: all derived artifacts regenerated"
