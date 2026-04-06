#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
REFERENCE_PATH="${1:-$ROOT_DIR/artifacts/baselines/ufunc_benchmark_baseline.json}"
CANDIDATE_PATH="${2:-${FNP_PERF_CANDIDATE_BASELINE:-$ROOT_DIR/artifacts/logs/ufunc_benchmark_baseline_candidate_${TS}.json}}"
REPORT_PATH="${3:-${FNP_PERF_BUDGET_REPORT:-$ROOT_DIR/artifacts/logs/performance_budget_delta_${TS}.json}}"
REFERENCE_SNAPSHOT_PATH="${FNP_PERF_REFERENCE_SNAPSHOT:-$ROOT_DIR/artifacts/logs/ufunc_benchmark_baseline_reference_${TS}.json}"
MAX_P99_REGRESSION_RATIO="${FNP_PERF_MAX_P99_REGRESSION_RATIO:-0.07}"
COVERAGE_FLOOR="${FNP_PERF_COVERAGE_FLOOR:-1.0}"

cd "$ROOT_DIR"

run_cargo() {
  if command -v rch >/dev/null 2>&1; then
    echo "[performance-budget-gate] executor=rch"
    rch exec -- cargo "$@"
  else
    echo "[performance-budget-gate] executor=cargo"
    cargo "$@"
  fi
}

if [[ ! -f "$REFERENCE_PATH" ]]; then
  echo "[performance-budget-gate] missing reference baseline: $REFERENCE_PATH" >&2
  exit 1
fi

echo "[performance-budget-gate] root=$ROOT_DIR"
echo "[performance-budget-gate] reference=$REFERENCE_PATH"
echo "[performance-budget-gate] reference_snapshot=$REFERENCE_SNAPSHOT_PATH"
echo "[performance-budget-gate] candidate=$CANDIDATE_PATH"
echo "[performance-budget-gate] report=$REPORT_PATH"
echo "[performance-budget-gate] max_p99_regression_ratio=$MAX_P99_REGRESSION_RATIO coverage_floor=$COVERAGE_FLOOR"

mkdir -p "$(dirname "$REFERENCE_SNAPSHOT_PATH")"
mkdir -p "$(dirname "$CANDIDATE_PATH")"
cp "$REFERENCE_PATH" "$REFERENCE_SNAPSHOT_PATH"

# Generate candidate into the tracked baseline path so rch artifact retrieval
# reliably materializes it locally, then copy to timestamped candidate output.
run_cargo run -p fnp-conformance --bin generate_benchmark_baseline -- \
  --output-path "$REFERENCE_PATH"

cp "$REFERENCE_PATH" "$CANDIDATE_PATH"
cp "$REFERENCE_SNAPSHOT_PATH" "$REFERENCE_PATH"

run_cargo run -p fnp-conformance --bin run_performance_budget_gate -- \
  --reference-path "$REFERENCE_SNAPSHOT_PATH" \
  --candidate-path "$CANDIDATE_PATH" \
  --report-path "$REPORT_PATH" \
  --max-p99-regression-ratio "$MAX_P99_REGRESSION_RATIO" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[performance-budget-gate] completed"
