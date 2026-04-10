#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
QUICK=0

if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
  shift
fi

if [[ $QUICK -eq 1 ]]; then
  DEFAULT_OUTPUT_PATH="$ROOT_DIR/artifacts/logs/cross_engine_benchmark_quick_${TS}.json"
else
  DEFAULT_OUTPUT_PATH="$ROOT_DIR/artifacts/baselines/cross_engine_benchmark_v1.json"
fi

OUTPUT_PATH="${1:-${FNP_CROSS_ENGINE_OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}}"
MANIFEST_PATH="${2:-${FNP_CROSS_ENGINE_MANIFEST_PATH:-$ROOT_DIR/artifacts/contracts/cross_engine_benchmark_workloads_v1.yaml}}"
ORACLE_PYTHON="${FNP_ORACLE_PYTHON:-}"

report_path() {
  local input_path="$1"
  local base="${input_path%.json}"
  printf '%s.report.md' "$base"
}

sidecar_path() {
  local input_path="$1"
  local base="${input_path%.json}"
  printf '%s.raptorq.json' "$base"
}

scrub_path() {
  local input_path="$1"
  local base="${input_path%.json}"
  printf '%s.scrub_report.json' "$base"
}

decode_path() {
  local input_path="$1"
  local base="${input_path%.json}"
  printf '%s.decode_proof.json' "$base"
}

REPORT_PATH="$(report_path "$OUTPUT_PATH")"
SIDECAR_PATH="$(sidecar_path "$OUTPUT_PATH")"
SCRUB_PATH="$(scrub_path "$OUTPUT_PATH")"
DECODE_PATH="$(decode_path "$OUTPUT_PATH")"

run_rch_build_and_select_worker() {
  local output=""
  local status=0
  local worker=""

  set +e
  output="$(
    rch exec -- cargo build -p fnp-conformance \
      --bin run_cross_engine_benchmark 2>&1
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
    echo "[cross-engine-benchmark] unable to determine rch worker from build output" >&2
    return 1
  fi

  RCH_SELECTED_WORKER="$worker"
}

validate_artifacts() {
  python3 - "$OUTPUT_PATH" "$REPORT_PATH" "$SIDECAR_PATH" "$SCRUB_PATH" "$DECODE_PATH" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
sidecar_path = Path(sys.argv[3])
scrub_path = Path(sys.argv[4])
decode_path = Path(sys.argv[5])

for path in (output_path, report_path, sidecar_path, scrub_path, decode_path):
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}")

report = json.loads(output_path.read_text(encoding="utf-8"))
if "env_fingerprint" not in report or "workloads" not in report:
    raise SystemExit("cross-engine benchmark report missing required top-level fields")
workloads = report["workloads"]
if not workloads:
    raise SystemExit("cross-engine benchmark report contains zero workloads")
for workload in workloads:
    for required in ("name", "family", "fnp", "numpy", "ratio", "band"):
        if required not in workload:
            raise SystemExit(f"workload missing field: {required}")

scrub = json.loads(scrub_path.read_text(encoding="utf-8"))
if scrub.get("status") != "ok":
    raise SystemExit(f"scrub report status is not ok: {scrub.get('status')}")

decode = json.loads(decode_path.read_text(encoding="utf-8"))
if decode.get("recovery_success") is not True:
    raise SystemExit("decode proof did not report recovery_success=true")

report_text = report_path.read_text(encoding="utf-8").strip()
if not report_text:
    raise SystemExit("markdown report is empty")

ratios = sorted((float(workload["ratio"]), workload["name"]) for workload in workloads)
best_ratio, best_name = ratios[0]
worst_ratio, worst_name = ratios[-1]
print(
    json.dumps(
        {
            "workload_count": len(workloads),
            "best_ratio": best_ratio,
            "best_name": best_name,
            "worst_ratio": worst_ratio,
            "worst_name": worst_name,
        },
        indent=2,
    )
)
PY
}

echo "[cross-engine-benchmark] root=$ROOT_DIR"
echo "[cross-engine-benchmark] manifest=$MANIFEST_PATH"
echo "[cross-engine-benchmark] output=$OUTPUT_PATH"
echo "[cross-engine-benchmark] report=$REPORT_PATH"
echo "[cross-engine-benchmark] quick=$QUICK"

mkdir -p "$(dirname "$OUTPUT_PATH")"

if command -v rch >/dev/null 2>&1; then
  run_rch_build_and_select_worker
  echo "[cross-engine-benchmark] worker=$RCH_SELECTED_WORKER"

  SSH_CMD=(
    ssh "$RCH_SELECTED_WORKER"
    "cd '$ROOT_DIR' && export FNP_REQUIRE_REAL_NUMPY_ORACLE=1 && export CARGO_INCREMENTAL=0 && /data/tmp/cargo-target/debug/run_cross_engine_benchmark --manifest-path '$MANIFEST_PATH' --output-path '$OUTPUT_PATH' $( [[ $QUICK -eq 1 ]] && printf '%s' '--quick' )"
  )
  if [[ -n "$ORACLE_PYTHON" ]]; then
    SSH_CMD=(
      ssh "$RCH_SELECTED_WORKER"
      "cd '$ROOT_DIR' && export FNP_REQUIRE_REAL_NUMPY_ORACLE=1 && export CARGO_INCREMENTAL=0 && export FNP_ORACLE_PYTHON='$ORACLE_PYTHON' && /data/tmp/cargo-target/debug/run_cross_engine_benchmark --manifest-path '$MANIFEST_PATH' --output-path '$OUTPUT_PATH' $( [[ $QUICK -eq 1 ]] && printf '%s' '--quick' )"
    )
  fi
  "${SSH_CMD[@]}"

  scp -q "$RCH_SELECTED_WORKER:$OUTPUT_PATH" "$OUTPUT_PATH"
  scp -q "$RCH_SELECTED_WORKER:$REPORT_PATH" "$REPORT_PATH"
  scp -q "$RCH_SELECTED_WORKER:$SIDECAR_PATH" "$SIDECAR_PATH"
  scp -q "$RCH_SELECTED_WORKER:$SCRUB_PATH" "$SCRUB_PATH"
  scp -q "$RCH_SELECTED_WORKER:$DECODE_PATH" "$DECODE_PATH"
else
  echo "[cross-engine-benchmark] executor=cargo"
  export FNP_REQUIRE_REAL_NUMPY_ORACLE=1
  export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
  cargo run -p fnp-conformance --bin run_cross_engine_benchmark -- \
    --manifest-path "$MANIFEST_PATH" \
    --output-path "$OUTPUT_PATH" \
    $( [[ $QUICK -eq 1 ]] && printf '%s ' '--quick' )
fi

echo "[cross-engine-benchmark] validating artifacts"
validate_artifacts
echo "[cross-engine-benchmark] completed"
