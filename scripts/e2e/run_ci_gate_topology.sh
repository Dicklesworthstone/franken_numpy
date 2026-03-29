#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORACLE_OUTPUT_PATH="$ROOT_DIR/crates/fnp-conformance/fixtures/oracle_outputs/ufunc_oracle_output.json"
readonly PHASE2C_PACKETS=(
  "FNP-P2C-001"
  "FNP-P2C-002"
  "FNP-P2C-003"
  "FNP-P2C-004"
  "FNP-P2C-005"
  "FNP-P2C-006"
  "FNP-P2C-007"
  "FNP-P2C-008"
  "FNP-P2C-009"
)

cd "$ROOT_DIR"

# Remote artifact retrieval can race on Cargo's incremental cache directories.
# The topology gate does not require incremental compilation, so default it off
# unless the caller explicitly opts back in.
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

resolve_require_real_oracle() {
  if [[ -n "${FNP_REQUIRE_REAL_NUMPY_ORACLE:-}" ]]; then
    printf '%s' "$FNP_REQUIRE_REAL_NUMPY_ORACLE"
    return
  fi

  if [[ "${CI:-}" == "1" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
    printf '1'
  else
    printf '0'
  fi
}

read_oracle_source() {
  python3 - "$ORACLE_OUTPUT_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)
print(payload.get("oracle_source", ""))
PY
}

validate_phase2c_packets() {
  local packet_id
  for packet_id in "${PHASE2C_PACKETS[@]}"; do
    echo "[ci-topology] validating packet readiness for $packet_id"
    rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id "$packet_id"
  done
}

echo "[ci-topology] root=$ROOT_DIR"
echo "[ci-topology] cargo_incremental=$CARGO_INCREMENTAL"

echo "[ci-topology] G1 fmt+lint"
rch exec -- cargo fmt --check
rch exec -- cargo clippy --workspace --all-targets -- -D warnings

echo "[ci-topology] G2 unit+property"
rch exec -- cargo test --workspace --lib

echo "[ci-topology] G3 differential"
REQUIRE_REAL_ORACLE="$(resolve_require_real_oracle)"
export FNP_REQUIRE_REAL_NUMPY_ORACLE="$REQUIRE_REAL_ORACLE"
rch exec -- cargo run -p fnp-conformance --bin capture_numpy_oracle
rch exec -- cargo run -p fnp-conformance --bin run_ufunc_differential
ORACLE_SOURCE="$(read_oracle_source)"
echo "[ci-topology] G3 oracle_source=$ORACLE_SOURCE require_real_oracle=$REQUIRE_REAL_ORACLE"
if [[ "$REQUIRE_REAL_ORACLE" == "1" && "$ORACLE_SOURCE" == "pure_python_fallback" ]]; then
  echo "[ci-topology] G3 failure: oracle capture used pure_python_fallback; configure NumPy-backed interpreter via FNP_ORACLE_PYTHON" >&2
  exit 1
fi

echo "[ci-topology] G4 adversarial+security"
scripts/e2e/run_security_policy_gate.sh

echo "[ci-topology] G5 test-contract"
scripts/e2e/run_test_contract_gate.sh

echo "[ci-topology] G6 workflow+forensics"
scripts/e2e/run_workflow_scenario_gate.sh

echo "[ci-topology] G7 performance-budget"
scripts/e2e/run_performance_budget_gate.sh

echo "[ci-topology] G8 durability+decode-proof"
scripts/e2e/run_raptorq_gate.sh
validate_phase2c_packets

echo "[ci-topology] completed"
