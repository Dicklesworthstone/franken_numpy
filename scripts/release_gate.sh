#!/usr/bin/env bash
# Release gate (bead deadlock-audit-3ltbd.2): exit 0 only when <sha> may be tagged <version> and
# published.
#
# v0.4.0 was tagged at e84c5267c and all ten library crates were published to crates.io while NO
# CI run existed for that commit; the last fully green run was 64 commits earlier. This is the
# check that refuses that. For exactly <sha> it requires:
#   1. a completed "CI Gate Topology" run on <sha> whose nine gates G1..G9 all concluded
#      `success` - a G8 skipped after a failed G7 fails here, it does not pass by default;
#   2. the workspace version at <sha> to equal <version>, and CHANGELOG.md at <sha> to carry a
#      "## [<version>]" section;
#   3. numpy's own test suite through an fnp_python cdylib BUILT HERE FROM <sha> (git archive):
#      scripts/run_numpy_dropin_suite.sh over its default module list, with aa_failed == 0,
#      unowned == 0 and no module missing (crash, timeout or collection error).
# Push runs on main are serialised latest-wins from G2 on (.github/workflows/ci.yml), so a release
# commit gets its own full run with `gh workflow run ci.yml --ref <branch at sha>`.
# Publishing stays a manual owner action; paste this script's output into the release notes.
#
# Usage: scripts/release_gate.sh <sha> <version> [--ci-only]
#   --ci-only   checks 1 and 2 only and exits 2 ("not releasable: drop-in not run") when they pass
#   PYTHON      interpreter with numpy 2.4.x + pytest + hypothesis for check 3 (default python3)
#   PYO3_PYTHON the interpreter the cdylib links against (default: PYTHON)
# Deletion condition: a CI release workflow that enforces the same three checks before publishing.
set -uo pipefail
SHA_ARG="${1:?usage: release_gate.sh <sha> <version> [--ci-only]}"
VERSION="${2:?usage: release_gate.sh <sha> <version> [--ci-only]}"
MODE="${3:-full}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 70
PYTHON="${PYTHON:-python3}"
SHA="$(git rev-parse --verify "${SHA_ARG}^{commit}" 2>/dev/null)" || {
  echo "FAIL: $SHA_ARG is not a commit in this repository"; exit 1; }
failures=()
echo "release gate: sha=$SHA version=$VERSION mode=$MODE host=$(hostname) at $(date -u +%FT%TZ)"

# 1. CI: some completed run on exactly this sha with all nine gates `success`.
GATES=(G1 G2 G3 G4 G5 G6 G7 G8 G9)
runs="$(gh run list --workflow ci.yml --commit "$SHA" --limit 50 \
  --json databaseId,status,conclusion,event --jq '.[] | "\(.databaseId) \(.status) \(.conclusion) \(.event)"')"
green_run=""
if [[ -z "$runs" ]]; then
  failures+=("CI: no CI Gate Topology run exists for $SHA")
else
  while read -r run_id status conclusion event; do
    echo "  CI run $run_id event=$event status=$status conclusion=$conclusion"
    [[ "$status" == "completed" ]] || continue
    jobs="$(gh run view "$run_id" --json jobs --jq '.jobs[] | "\(.name | split(":")[0])=\(.conclusion)"')"
    missing=()
    for gate in "${GATES[@]}"; do
      verdict="$(grep -E "^$gate=" <<<"$jobs" | head -1 | cut -d= -f2)"
      echo "    $gate ${verdict:-absent}"
      [[ "$verdict" == "success" ]] || missing+=("$gate ${verdict:-absent}")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
      green_run="$run_id"
      break
    fi
    echo "    run $run_id is not green: ${missing[*]}"
  done <<<"$runs"
  [[ -n "$green_run" ]] || failures+=("CI: no run on $SHA has G1..G9 all success (see the per-gate lines above)")
fi
[[ -n "$green_run" ]] && echo "  CI: run $green_run on $SHA has G1..G9 success"

# 2. Version and changelog at the sha.
tree_version="$(git show "$SHA:Cargo.toml" | awk '/^\[workspace.package\]/{p=1} p && /^version *=/{gsub(/[" ]/,"",$0); split($0,a,"="); print a[2]; exit}')"
echo "  workspace version at $SHA: ${tree_version:-<none>}"
[[ "$tree_version" == "$VERSION" ]] || failures+=("version: Cargo.toml at $SHA says '${tree_version}', not '$VERSION'")
# Read the file first: `git show | grep -q` fails under pipefail when grep's early exit SIGPIPEs
# git show - on exactly the files that DO match.
changelog="$(git show "$SHA:CHANGELOG.md" 2>/dev/null)"
if grep -qF "## [$VERSION]" <<<"$changelog"; then
  echo "  CHANGELOG.md at $SHA has a [$VERSION] section"
else
  failures+=("changelog: CHANGELOG.md at $SHA has no '## [$VERSION]' section")
fi

# 3. numpy's own suite through a cdylib built from exactly this tree.
if [[ "$MODE" == "--ci-only" ]]; then
  echo "drop-in suite: NOT RUN (--ci-only)"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/fnp-release-gate-XXXXXX")"
  echo "  building fnp_python at $SHA in $WORK"
  git archive "$SHA" | tar -x -C "$WORK"
  # A target directory of its own, always: `git archive` stamps every file with the COMMIT time,
  # and cargo identifies workspace units by their path relative to the workspace root, so in a
  # shared CARGO_TARGET_DIR it judged an artifact built from another tree "fresh" and the gate
  # tested that tree's library instead of <sha>'s (caught on its first full run, 2026-10-08).
  target="$WORK/target"
  if ( cd "$WORK" && CARGO_TARGET_DIR="$target" \
        PYO3_PYTHON="${PYO3_PYTHON:-$(command -v "$PYTHON")}" \
        cargo build --release -p fnp-python --lib >"$WORK/build.log" 2>&1 ); then
    so="$WORK/fnp_python.so"
    cp "$target/release/libfnp_python.so" "$so"
    echo "  cdylib sha256 $(sha256sum "$so" | cut -d' ' -f1) (built from $SHA)"
    PYTHON="$PYTHON" "$WORK/scripts/run_numpy_dropin_suite.sh" "$so" "$WORK/dropin" >"$WORK/dropin.log" 2>&1
    report="$WORK/dropin/report.json"
    if [[ -f "$report" ]]; then
      read -r modules aa_failed unowned divergences < <(jq -r '"\(.modules) \(.aa_failed) \(.unowned) \(.divergences)"' "$report")
      missing_modules="$(grep -c 'MISSING REPORT' "$WORK/dropin.log")"
      echo "  drop-in: report $report modules=$modules aa_failed=$aa_failed unowned=$unowned divergences=$divergences missing=$missing_modules"
      [[ "$aa_failed" == "0" ]] || failures+=("drop-in: aa_failed=$aa_failed (the A/A lane itself fails; the interpreter is not a valid oracle)")
      [[ "$unowned" == "0" ]] || failures+=("drop-in: $unowned divergence(s) owned by no bead")
      [[ "$missing_modules" == "0" ]] || failures+=("drop-in: $missing_modules module(s) produced no report")
    else
      failures+=("drop-in: no report.json (see $WORK/dropin.log)")
    fi
  else
    failures+=("drop-in: cdylib build failed at $SHA (see $WORK/build.log)")
  fi
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "NOT RELEASABLE: $SHA"
  printf '  FAIL: %s\n' "${failures[@]}"
  exit 1
fi
if [[ "$MODE" == "--ci-only" ]]; then
  echo "NOT RELEASABLE YET: CI and version checks pass at $SHA; run without --ci-only for the drop-in suite"
  exit 2
fi
echo "RELEASABLE: $SHA as $VERSION"
