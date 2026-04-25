#!/usr/bin/env bash
# tg33 — golden-snapshot regression gate for compliance matrix generators.
#
# Runs the three *.generated.md producers and diffs their outputs
# against the committed snapshots. Exits 0 when nothing changed,
# 1 with a unified diff dump when ANY matrix drifted.
#
# Engineers regenerate the snapshots intentionally:
#   python3 scripts/fnp_python_compliance_matrix.py
#   python3 scripts/fnp_random_compliance_matrix.py
#   python3 scripts/fnp_polynomial_compliance_matrix.py
#   git add crates/fnp-python/*.generated.md
#   git commit -m "docs(compliance): regen matrices for <reason>"
#
# CI invokes this script in PR builds. A surface drop (e.g. accidental
# removal of a #[pyfunction]) flags as a diff and fails the gate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GENERATORS=(
    "scripts/fnp_python_compliance_matrix.py"
    "scripts/fnp_random_compliance_matrix.py"
    "scripts/fnp_polynomial_compliance_matrix.py"
)

OUTPUTS=(
    "crates/fnp-python/COMPLIANCE.generated.md"
    "crates/fnp-python/RANDOM_COMPLIANCE.generated.md"
    "crates/fnp-python/POLYNOMIAL_COMPLIANCE.generated.md"
)

# Snapshot every output BEFORE running the generators so we can compare
# against the committed state, not the agent's intermediate state.
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for output in "${OUTPUTS[@]}"; do
    if [[ -f "$output" ]]; then
        cp "$output" "$TMPDIR/$(basename "$output").snapshot"
    fi
done

# 14rm: run all three generators regardless of any one's exit code so
# engineers see the full surface picture in a single run. Per-generator
# gate failures (MUST < 95%) and drift events both contribute to a
# unified non-zero exit at the end.
generator_failures=()
for gen in "${GENERATORS[@]}"; do
    if ! python3 "$gen"; then
        generator_failures+=("$gen")
        echo "FAIL: $gen exited non-zero (gate or runtime error)" >&2
    fi
done

drift=0
for output in "${OUTPUTS[@]}"; do
    snapshot="$TMPDIR/$(basename "$output").snapshot"
    if [[ ! -f "$snapshot" ]]; then
        echo "WARN: no committed snapshot for $output (first generation)" >&2
        drift=1
        continue
    fi
    if ! diff -u "$snapshot" "$output" > "$TMPDIR/diff.out"; then
        echo "DRIFT in $output:" >&2
        cat "$TMPDIR/diff.out" >&2
        drift=1
    fi
done

if [[ ${#generator_failures[@]} -gt 0 ]]; then
    echo "" >&2
    echo "Generator(s) exited non-zero (likely MUST < 95% gate):" >&2
    for gen in "${generator_failures[@]}"; do
        echo "  $gen" >&2
    done
fi

if [[ ${#generator_failures[@]} -gt 0 || $drift -ne 0 ]]; then
    echo "" >&2
    echo "Compliance matrix regression detected. If the surface change is" >&2
    echo "intentional, regenerate and commit the snapshot:" >&2
    for output in "${OUTPUTS[@]}"; do
        echo "  $output" >&2
    done
    echo "  git add crates/fnp-python/*.generated.md" >&2
    echo "  git commit -m \"docs(compliance): regen matrices for <reason>\"" >&2
    exit 1
fi

echo "All 3 compliance matrices match committed snapshot."
