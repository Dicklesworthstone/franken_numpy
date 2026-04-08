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

run_rch_build_and_select_worker() {
  local output=""
  local status=0
  local worker=""

  set +e
  output="$(
    rch exec -- cargo build -p fnp-conformance \
      --bin generate_benchmark_baseline \
      --bin run_performance_budget_gate 2>&1
  )"
  status=$?
  set -e

  printf '%s\n' "$output"
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi

  worker="$(
    printf '%s\n' "$output" \
      | sed -n 's/.*Selected worker: \([^ ]*\) at .*/\1/p' \
      | head -n 1
  )"
  if [[ -z "$worker" ]]; then
    echo "[performance-budget-gate] unable to determine rch worker from build output" >&2
    return 1
  fi

  RCH_SELECTED_WORKER="$worker"
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
mkdir -p "$(dirname "$REPORT_PATH")"

cp "$REFERENCE_PATH" "$REFERENCE_SNAPSHOT_PATH"

if command -v rch >/dev/null 2>&1; then
  run_rch_build_and_select_worker
  echo "[performance-budget-gate] worker=$RCH_SELECTED_WORKER"

  ssh "$RCH_SELECTED_WORKER" \
    "cd '$ROOT_DIR' && \
     /data/tmp/cargo-target/debug/generate_benchmark_baseline --output-path '$CANDIDATE_PATH'"
  scp -q "$RCH_SELECTED_WORKER:$CANDIDATE_PATH" "$CANDIDATE_PATH"

  ssh "$RCH_SELECTED_WORKER" \
    "cd '$ROOT_DIR' && \
     /data/tmp/cargo-target/debug/run_performance_budget_gate \
       --reference-path '$REFERENCE_SNAPSHOT_PATH' \
       --candidate-path '$CANDIDATE_PATH' \
       --report-path '$REPORT_PATH' \
       --max-p99-regression-ratio '$MAX_P99_REGRESSION_RATIO' \
       --coverage-floor '$COVERAGE_FLOOR'"
  if [[ ! -f "$CANDIDATE_PATH" ]]; then
    echo "[performance-budget-gate] missing copied candidate: $CANDIDATE_PATH" >&2
    exit 1
  fi
  scp -q "$RCH_SELECTED_WORKER:$REPORT_PATH" "$REPORT_PATH"
  if [[ ! -f "$REPORT_PATH" ]]; then
    echo "[performance-budget-gate] missing copied report: $REPORT_PATH" >&2
    exit 1
  fi
else
  run_cargo run -p fnp-conformance --bin generate_benchmark_baseline -- \
    --output-path "$CANDIDATE_PATH"
  run_cargo run -p fnp-conformance --bin run_performance_budget_gate -- \
    --reference-path "$REFERENCE_SNAPSHOT_PATH" \
    --candidate-path "$CANDIDATE_PATH" \
    --report-path "$REPORT_PATH" \
    --max-p99-regression-ratio "$MAX_P99_REGRESSION_RATIO" \
    --coverage-floor "$COVERAGE_FLOOR"
fi

echo "[performance-budget-gate] completed"
