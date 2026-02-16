#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
LOG_PATH="${1:-$ROOT_DIR/artifacts/logs/test_contract_e2e_${TS}.jsonl}"
REPORT_PATH="${2:-${FNP_TEST_CONTRACT_RELIABILITY_REPORT:-$ROOT_DIR/artifacts/logs/test_contract_gate_reliability_${TS}.json}}"
RETRIES="${FNP_TEST_CONTRACT_RETRIES:-1}"
FLAKE_BUDGET="${FNP_TEST_CONTRACT_FLAKE_BUDGET:-0}"
COVERAGE_FLOOR="${FNP_TEST_CONTRACT_COVERAGE_FLOOR:-1.0}"

cd "$ROOT_DIR"

echo "[test-contract-gate] root=$ROOT_DIR"
echo "[test-contract-gate] runtime_policy_log=$LOG_PATH"
echo "[test-contract-gate] reliability_report=$REPORT_PATH"
echo "[test-contract-gate] retries=$RETRIES flake_budget=$FLAKE_BUDGET coverage_floor=$COVERAGE_FLOOR"

rch exec -- cargo run -p fnp-conformance --bin run_test_contract_gate -- \
  --log-path "$LOG_PATH" \
  --report-path "$REPORT_PATH" \
  --retries "$RETRIES" \
  --flake-budget "$FLAKE_BUDGET" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[test-contract-gate] completed"
