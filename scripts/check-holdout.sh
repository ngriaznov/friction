#!/usr/bin/env bash
# check-holdout.sh — verify the sealed holdout split hasn't drifted.
#
# corpus/holdout.lock is written once when the holdout split is frozen.
# Each line is tab-separated: id, sha256, relative path
# (relative to the repository root, e.g. "corpus/docs/readme-0042.md").
#
# If corpus/holdout.lock is absent (repo has no corpus yet), this is a no-op
# success so CI stays green before the corpus generation work lands.
#
# Usage: scripts/check-holdout.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
lock_file="${repo_root}/corpus/holdout.lock"

if [[ ! -f "${lock_file}" ]]; then
	echo "holdout lock absent - skipping"
	exit 0
fi

if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
	sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
	echo "error: neither sha256sum nor shasum is available" >&2
	exit 1
fi

failures=0
line_no=0
verified=0

while IFS=$'\t' read -r id expected_sha rel_path || [[ -n "${id}" ]]; do
	line_no=$((line_no + 1))

	# Skip blank lines and comments.
	[[ -z "${id}" || "${id}" == \#* ]] && continue

	if [[ -z "${expected_sha}" || -z "${rel_path}" ]]; then
		echo "error: ${lock_file}:${line_no}: malformed line (expected id<TAB>sha256<TAB>path)" >&2
		failures=$((failures + 1))
		continue
	fi

	target="${repo_root}/${rel_path}"

	if [[ ! -f "${target}" ]]; then
		echo "error: holdout doc missing: id=${id} path=${rel_path}" >&2
		failures=$((failures + 1))
		continue
	fi

	actual_sha="$(sha256 "${target}")"

	if [[ "${actual_sha}" != "${expected_sha}" ]]; then
		echo "error: holdout hash mismatch: id=${id} path=${rel_path}" >&2
		echo "  expected: ${expected_sha}" >&2
		echo "  actual:   ${actual_sha}" >&2
		failures=$((failures + 1))
		continue
	fi

	verified=$((verified + 1))
done <"${lock_file}"

if ((failures > 0)); then
	echo "holdout check failed: ${failures} mismatch(es)" >&2
	exit 1
fi

echo "holdout check passed: ${verified} line(s) verified"
