#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%s)"
SHARD="${FNP_PY_CONFORMANCE_SHARD:-fnp-python-smoke}"
REPORT_PATH="${FNP_PY_CONFORMANCE_SHARD_REPORT:-$ROOT_DIR/artifacts/logs/fnp_python_conformance_shards_${TS}.jsonl}"

cd "$ROOT_DIR"

run_cargo() {
  if command -v rch >/dev/null 2>&1; then
    echo "[fnp-python-conformance-shards] executor=rch"
    rch exec -- cargo "$@"
  else
    echo "[fnp-python-conformance-shards] executor=cargo"
    cargo "$@"
  fi
}

echo "[fnp-python-conformance-shards] root=$ROOT_DIR"
echo "[fnp-python-conformance-shards] shard=$SHARD"
echo "[fnp-python-conformance-shards] report=$REPORT_PATH"

run_cargo run -p fnp-conformance --bin run_fnp_python_conformance_shards -- \
  --repo-root "$ROOT_DIR" \
  --shard "$SHARD" \
  --report-path "$REPORT_PATH" \
  "$@"

echo "[fnp-python-conformance-shards] completed"
