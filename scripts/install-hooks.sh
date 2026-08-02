#!/usr/bin/env bash
# Points this clone's git hooks at the committed scripts/git-hooks/
# directory (currently: a pre-commit hook that auto-regenerates derived
# binary artifacts when their sources are staged — see that hook's own
# header). Opt-in, once per clone:
#
#   ./scripts/install-hooks.sh
#
# Undo with: git config --unset core.hooksPath
set -euo pipefail
cd "$(dirname "$0")/.."

git config core.hooksPath scripts/git-hooks
echo "install-hooks: core.hooksPath -> scripts/git-hooks"
