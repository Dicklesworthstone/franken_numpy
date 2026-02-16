#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
LOG_PATH="${1:-$ROOT_DIR/artifacts/logs/runtime_policy_e2e_${TS}.jsonl}"
REPORT_PATH="${2:-${FNP_SECURITY_RELIABILITY_REPORT:-$ROOT_DIR/artifacts/logs/security_gate_reliability_${TS}.json}}"
RETRIES="${FNP_SECURITY_RETRIES:-1}"
FLAKE_BUDGET="${FNP_SECURITY_FLAKE_BUDGET:-0}"
COVERAGE_FLOOR="${FNP_SECURITY_COVERAGE_FLOOR:-1.0}"

cd "$ROOT_DIR"

echo "[security-gate] root=$ROOT_DIR"
echo "[security-gate] runtime_policy_log=$LOG_PATH"
echo "[security-gate] reliability_report=$REPORT_PATH"
echo "[security-gate] retries=$RETRIES flake_budget=$FLAKE_BUDGET coverage_floor=$COVERAGE_FLOOR"

rch exec -- cargo run -p fnp-conformance --bin run_security_gate -- \
  --log-path "$LOG_PATH" \
  --report-path "$REPORT_PATH" \
  --retries "$RETRIES" \
  --flake-budget "$FLAKE_BUDGET" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[security-gate] completed"
