#!/usr/bin/env bash
# Run CI's nine gates (G1..G9 of .github/workflows/ci.yml) on exactly <sha> WITHOUT GitHub
# Actions (#2: Actions are off and releases never depend on them), and write the receipt that
# `FNP_GATE_RECEIPT=<receipt> scripts/release_gate.sh` accepts in place of a CI run.
#
# Usage: scripts/run_release_gates.sh <sha> <perf-reference> <workdir>
#   <perf-reference>  G7's A/B reference commit-ish (for a release: the previous release tag)
#   <workdir>         absent or empty directory; receives a fresh clone of this repository at
#                     <sha> (src/), the
#                     target dir, one log per gate and receipt.tsv
#   PYTHON            interpreter with numpy<2.5 (G2/G3 oracles) that can create a venv for the
#                     G9 wheel check (default python3)
#
# Same commands and gate-level env as ci.yml. Two deliberate differences:
#   - every gate runs even after an earlier one fails (ci.yml's `needs:` chain skips them), so one
#     run reports all nine; release_gate.sh still requires all nine at rc 0;
#   - Cargo is plain `cargo` on this host. Run it ON the build host (a worker or any Linux x86_64
#     box with the pinned toolchain), so no shim can fall back to building somewhere else.
# Receipt format: "# key=value" header lines (sha, perf_reference, host, rustc, python), then one
# tab-separated line per gate: <G1..G9> <rc> <seconds> <log path>.
set -uo pipefail
usage="usage: run_release_gates.sh <sha> <perf-reference> <workdir>"
SHA_ARG="${1:?$usage}"
REFERENCE_ARG="${2:?$usage}"
WORK="${3:?$usage}"
PYTHON="${PYTHON:-python3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)" || exit 70

mkdir -p "$WORK" || exit 70
WORK="$(cd "$WORK" && pwd)"
if [[ -n "$(ls -A "$WORK")" ]]; then
  echo "FAIL: $WORK is not empty (a receipt must come from one fresh run)"
  exit 70
fi
SRC="$WORK/src"
# A fresh clone of this repository, not its working tree: the receipt must describe exactly <sha>
# (an unpushed release commit is fine), and G7 needs the history to build its reference.
git clone -q --no-local "$REPO" "$SRC" || exit 70
git -C "$SRC" -c advice.detachedHead=false checkout -q --detach "$SHA_ARG" || exit 70
SHA="$(git -C "$SRC" rev-parse HEAD)"
REFERENCE="$(git -C "$SRC" rev-parse --verify "${REFERENCE_ARG}^{commit}")" || exit 70
cd "$SRC" || exit 70

# ci.yml's workflow-level env.
export RUSTFLAGS="-D warnings -C target-feature=+avx2" CARGO_TERM_COLOR=never
export CARGO_TARGET_DIR="$WORK/target"

RECEIPT="$WORK/receipt.tsv"
{
  echo "# sha=$SHA"
  echo "# perf_reference=$REFERENCE"
  echo "# host=$(hostname) arch=$(uname -m) started=$(date -u +%FT%TZ)"
  echo "# rustc=$(rustc --version)"
  echo "# python=$("$PYTHON" -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)' 2>&1)"
} >"$RECEIPT"

gate() { # gate <G#> <function>: run it in a subshell, log it, append its receipt line
  local id=$1 fn=$2 t0=$SECONDS rc
  echo "== $id start $(date -u +%T)"
  ( "$fn" ) >"$WORK/$id.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\t%s\n' "$id" "$rc" "$((SECONDS - t0))" "$WORK/$id.log" >>"$RECEIPT"
  echo "== $id rc=$rc $((SECONDS - t0))s"
}

g1() {
  cargo fmt --check && cargo check --workspace --all-targets &&
    cargo clippy --workspace --all-targets -- -D warnings
}
g2() { cargo test --workspace --no-fail-fast; }
g3() {
  export FNP_REQUIRE_REAL_NUMPY_ORACLE=1 FNP_ORACLE_PYTHON="$PYTHON"
  cargo run -p fnp-conformance --bin capture_numpy_oracle || return
  local source
  source="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("oracle_source", ""))' \
    crates/fnp-conformance/fixtures/oracle_outputs/ufunc_oracle_output.json)" || return
  echo "G3 oracle_source=$source"
  [[ "$source" != "pure_python_fallback" ]] || { echo "G3 failure: oracle used pure_python_fallback"; return 1; }
  cargo run -p fnp-conformance --bin run_ufunc_differential
}
g4() { bash scripts/e2e/run_security_policy_gate.sh; }
g5() { bash scripts/e2e/run_test_contract_gate.sh; }
g6() {
  FNP_WORKFLOW_RETRIES=0 FNP_WORKFLOW_FLAKE_BUDGET=0 FNP_WORKFLOW_COVERAGE_FLOOR=1.0 \
    bash scripts/e2e/run_workflow_scenario_gate.sh
}
g7() {
  FNP_PERF_AB_REFERENCE="$REFERENCE" FNP_PERF_AB_ROUNDS=9 FNP_PERF_MAX_MEDIAN_REGRESSION_RATIO=0.07 \
    FNP_PERF_COVERAGE_FLOOR=1.0 FNP_PERF_REPORT_DIR="$WORK/perf_ab" \
    bash scripts/e2e/run_performance_budget_gate.sh
}
g8() {
  cargo run -p fnp-conformance --bin run_raptorq_gate -- --retries 1 --flake-budget 0 --coverage-floor 1.0 || return
  local packet
  for packet in FNP-P2C-001 FNP-P2C-002 FNP-P2C-003 FNP-P2C-004 FNP-P2C-005 FNP-P2C-006 FNP-P2C-007 FNP-P2C-008 FNP-P2C-009; do
    cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id "$packet" || return
  done
}
g9() {
  "$PYTHON" -m venv "$WORK/venv-g9" || return
  # shellcheck disable=SC1091
  source "$WORK/venv-g9/bin/activate" || return
  python -m pip install -q --upgrade pip maturin "numpy<2.5" || return
  maturin build --release --out "$WORK/dist" || return
  python -m pip install -q "$WORK"/dist/*.whl || return
  cd "$WORK" || return
  python - <<'PY'
import fnp_python as pkg
import fnp_python.fnp_python as ext
import fnp_python.linalg as la
import numpy as np
missing = sorted(n for n in dir(ext) if not hasattr(pkg, n))
assert not missing, f"{len(missing)} extension names missing from the package: {missing[:10]}"
assert la is pkg.linalg, "dotted submodule import does not resolve to the package attribute"
assert pkg.__version__ and pkg.__numpy_version__ == np.__version__
assert (pkg.random.default_rng(12345).standard_normal(3) == np.random.default_rng(12345).standard_normal(3)).all()
assert isinstance(pkg.allclose(np.arange(3.0), np.arange(3.0)), bool)
print("wheel ok:", pkg.__version__, "numpy", np.__version__)
PY
}

for id in G1 G2 G3 G4 G5 G6 G7 G8 G9; do
  gate "$id" "$(tr '[:upper:]' '[:lower:]' <<<"$id")"
done
echo "# finished=$(date -u +%FT%TZ)" >>"$RECEIPT"
cat "$RECEIPT"
awk -F'\t' '/^G[1-9]\t/ && $2 != 0 {bad=1} END {exit bad}' "$RECEIPT"
