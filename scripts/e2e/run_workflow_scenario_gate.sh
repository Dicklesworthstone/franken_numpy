#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
LOG_PATH="${1:-$ROOT_DIR/artifacts/logs/workflow_scenario_e2e_${TS}.jsonl}"
REPORT_PATH="${2:-${FNP_WORKFLOW_RELIABILITY_REPORT:-$ROOT_DIR/artifacts/logs/workflow_scenario_reliability_${TS}.json}}"
ARTIFACT_INDEX_PATH="${3:-${FNP_WORKFLOW_ARTIFACT_INDEX:-$ROOT_DIR/artifacts/logs/workflow_scenario_artifact_index_${TS}.json}}"
RETRIES="${FNP_WORKFLOW_RETRIES:-1}"
FLAKE_BUDGET="${FNP_WORKFLOW_FLAKE_BUDGET:-0}"
COVERAGE_FLOOR="${FNP_WORKFLOW_COVERAGE_FLOOR:-1.0}"

cd "$ROOT_DIR"

run_cargo() {
  if command -v rch >/dev/null 2>&1; then
    echo "[workflow-scenario-gate] executor=rch"
    rch exec -- cargo "$@"
  else
    echo "[workflow-scenario-gate] executor=cargo"
    cargo "$@"
  fi
}

echo "[workflow-scenario-gate] root=$ROOT_DIR"
echo "[workflow-scenario-gate] workflow_log=$LOG_PATH"
echo "[workflow-scenario-gate] reliability_report=$REPORT_PATH"
echo "[workflow-scenario-gate] artifact_index=$ARTIFACT_INDEX_PATH"
echo "[workflow-scenario-gate] retries=$RETRIES flake_budget=$FLAKE_BUDGET coverage_floor=$COVERAGE_FLOOR"

run_cargo run -p fnp-conformance --bin run_workflow_scenario_gate -- \
  --log-path "$LOG_PATH" \
  --artifact-index-path "$ARTIFACT_INDEX_PATH" \
  --report-path "$REPORT_PATH" \
  --retries "$RETRIES" \
  --flake-budget "$FLAKE_BUDGET" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[workflow-scenario-gate] completed"
