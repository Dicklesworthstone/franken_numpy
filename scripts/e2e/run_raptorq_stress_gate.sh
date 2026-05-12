#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
MODE="${FNP_RAPTORQ_STRESS_MODE:-local}"
REPORT_PATH="${FNP_RAPTORQ_STRESS_REPORT:-$ROOT_DIR/target/raptorq_stress_report_${TS}.json}"
OUTPUT_DIR="${FNP_RAPTORQ_STRESS_OUTPUT_DIR:-}"
SOURCE_BYTES="${FNP_RAPTORQ_STRESS_SOURCE_BYTES:-}"
PARALLELISM="${FNP_RAPTORQ_PARALLELISM:-1}"
STRESS_PARALLELISM="${FNP_RAPTORQ_STRESS_PARALLELISM:-$PARALLELISM}"
COVERAGE_FLOOR="${FNP_RAPTORQ_COVERAGE_FLOOR:-1.0}"
RETRIES="${FNP_RAPTORQ_RETRIES:-0}"
FLAKE_BUDGET="${FNP_RAPTORQ_FLAKE_BUDGET:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      if [[ -z "$SOURCE_BYTES" ]]; then
        SOURCE_BYTES="65536"
      fi
      shift
      ;;
    --local)
      MODE="local"
      shift
      ;;
    --report-path)
      REPORT_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --source-bytes)
      SOURCE_BYTES="$2"
      shift 2
      ;;
    --parallelism)
      PARALLELISM="$2"
      shift 2
      ;;
    --stress-parallelism)
      STRESS_PARALLELISM="$2"
      shift 2
      ;;
    --coverage-floor)
      COVERAGE_FLOOR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: scripts/e2e/run_raptorq_stress_gate.sh [--smoke|--local] [--report-path <path>] [--output-dir <path>] [--source-bytes <n>] [--parallelism <n>] [--stress-parallelism <n>] [--coverage-floor <ratio>]"
      exit 0
      ;;
    *)
      echo "[raptorq-stress-gate] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_BYTES" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    SOURCE_BYTES="65536"
  else
    SOURCE_BYTES="4194304"
  fi
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/target/raptorq_stress_gate/${MODE}_${TS}"
fi

cd "$ROOT_DIR"
mkdir -p "$(dirname "$REPORT_PATH")" "$OUTPUT_DIR"
RCH_LOG="$OUTPUT_DIR/rch_output.log"

echo "[raptorq-stress-gate] root=$ROOT_DIR"
echo "[raptorq-stress-gate] mode=$MODE"
echo "[raptorq-stress-gate] report=$REPORT_PATH"
echo "[raptorq-stress-gate] output_dir=$OUTPUT_DIR"
echo "[raptorq-stress-gate] source_bytes=$SOURCE_BYTES parallelism=$PARALLELISM stress_parallelism=$STRESS_PARALLELISM coverage_floor=$COVERAGE_FLOOR"

set +e
rch exec -- cargo run -p fnp-conformance --bin run_raptorq_gate -- \
  --report-path "$REPORT_PATH" \
  --retries "$RETRIES" \
  --flake-budget "$FLAKE_BUDGET" \
  --coverage-floor "$COVERAGE_FLOOR" \
  --parallelism "$PARALLELISM" \
  --stress-mode "$MODE" \
  --stress-output-dir "$OUTPUT_DIR" \
  --stress-source-bytes "$SOURCE_BYTES" \
  --stress-parallelism "$STRESS_PARALLELISM" >"$RCH_LOG" 2>&1 &
RCH_PID=$!
RCH_STATUS=""
while kill -0 "$RCH_PID" 2>/dev/null; do
  if grep -q "Remote command finished: exit=0" "$RCH_LOG"; then
    kill "$RCH_PID" 2>/dev/null || true
    wait "$RCH_PID" 2>/dev/null || true
    RCH_STATUS=0
    break
  fi
  if grep -q "Remote command finished: exit=" "$RCH_LOG"; then
    wait "$RCH_PID" 2>/dev/null
    RCH_STATUS=$?
    break
  fi
  sleep 1
done
if [[ -z "$RCH_STATUS" ]]; then
  wait "$RCH_PID" 2>/dev/null
  RCH_STATUS=$?
fi
set -e
cat "$RCH_LOG"
if [[ "$RCH_STATUS" -ne 0 ]]; then
  exit "$RCH_STATUS"
fi

python3 - "$REPORT_PATH" "$RCH_LOG" <<'PY'
import json
import os
import sys

report_path = sys.argv[1]
log_path = sys.argv[2]
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
else:
    with open(log_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    start = text.find('{\n  "status"')
    if start < 0:
        raise SystemExit("invalid raptorq stress report: no JSON object in RCH log")
    report, _ = json.JSONDecoder().raw_decode(text[start:])

stress = report.get("stress")
required = [
    "worker_count",
    "input_hash",
    "recovered_hash",
    "elapsed_ms",
    "source_symbols",
    "repair_symbols",
    "total_symbols",
    "dropped_symbol_scenario",
    "recovery_matrix",
    "replay_command",
]
if not isinstance(stress, dict):
    raise SystemExit("invalid raptorq stress report: missing stress object")
missing = [field for field in required if field not in stress]
if report.get("status") != "pass" or missing:
    raise SystemExit(f"invalid raptorq stress report: status={report.get('status')} missing={missing}")
if stress["input_hash"] != stress["recovered_hash"]:
    raise SystemExit("raptorq stress report hash mismatch")
if not stress["dropped_symbol_scenario"]:
    raise SystemExit("raptorq stress report missing dropped-symbol scenario")
matrix = stress.get("recovery_matrix")
if not isinstance(matrix, list) or not matrix:
    raise SystemExit("raptorq stress report missing recovery matrix")
if stress.get("source_symbols", 0) >= 2 and stress.get("repair_symbols", 0) >= 2:
    if not any(item.get("dropped_count", 0) >= 2 and item.get("recovery_success") for item in matrix):
        raise SystemExit("raptorq stress report missing successful two-symbol recovery scenario")
for item in matrix:
    if item.get("recovery_success") is not True:
        raise SystemExit(f"raptorq stress recovery scenario failed: {item}")
    if item.get("recovered_hash") != stress["input_hash"]:
        raise SystemExit(f"raptorq stress recovery scenario hash mismatch: {item}")
PY

echo "[raptorq-stress-gate] completed"
