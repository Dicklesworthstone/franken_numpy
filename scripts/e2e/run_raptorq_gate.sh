#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
REPORT_PATH="${1:-${FNP_RAPTORQ_RELIABILITY_REPORT:-$ROOT_DIR/artifacts/logs/raptorq_gate_reliability_${TS}.json}}"
RETRIES="${FNP_RAPTORQ_RETRIES:-1}"
FLAKE_BUDGET="${FNP_RAPTORQ_FLAKE_BUDGET:-0}"
COVERAGE_FLOOR="${FNP_RAPTORQ_COVERAGE_FLOOR:-1.0}"

cd "$ROOT_DIR"

echo "[raptorq-gate] root=$ROOT_DIR"
echo "[raptorq-gate] reliability_report=$REPORT_PATH"
echo "[raptorq-gate] retries=$RETRIES flake_budget=$FLAKE_BUDGET coverage_floor=$COVERAGE_FLOOR"

rch exec -- cargo run -p fnp-conformance --bin run_raptorq_gate -- \
  --report-path "$REPORT_PATH" \
  --retries "$RETRIES" \
  --flake-budget "$FLAKE_BUDGET" \
  --coverage-floor "$COVERAGE_FLOOR"

echo "[raptorq-gate] completed"
