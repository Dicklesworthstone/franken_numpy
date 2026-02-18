# FrankenNumPy

<div align="center">
  <img src="franken_numpy_illustration.webp" alt="FrankenNumPy - memory-safe clean-room NumPy reimplementation in Rust">
</div>

FrankenNumPy is a memory-safe, clean-room Rust reimplementation of NumPy with two simultaneous goals:

1. ABSOLUTELY COMPLETE and TOTAL drop-in behavioral compatibility with legacy NumPy (no reduced-scope acceptance).
2. A more rigorous architecture for reliability, performance, and explainability.

## Core Identity

**Stride Calculus Engine (SCE):** deterministic shape/stride/broadcast legality and zero-copy view guarantees.

This is the primary non-regression contract.

## Method Stack

1. `alien-artifact-coding`: explicit decision contracts and evidence ledgers.
2. `extreme-software-optimization`: profile-first + one-lever + isomorphism proofs.
3. `RaptorQ-everywhere`: durable artifact sidecars and recovery proofs.
4. `frankenlibc/frankenfs doctrine`: strict/hardened mode split + fail-closed compatibility gates.

## Canonical Documents

- `AGENTS.md`
- `COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md`
- `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` (copied exemplar)
- `PLAN_TO_PORT_NUMPY_TO_RUST.md`
- `EXISTING_NUMPY_STRUCTURE.md`
- `PROPOSED_ARCHITECTURE.md`
- `FEATURE_PARITY.md`

## Current Implementation Snapshot

Implemented first major vertical slices:

- `fnp-dtype`: deterministic promotion/cast foundations (current wave)
- `fnp-ndarray`: broadcast legality, reshape `-1` inference, contiguous stride calculus (C/F)
- `fnp-runtime`: strict/hardened decision policy + fail-closed wire decoding + audited override gate + evidence ledger
- `fnp-ufunc`: broadcasted binary ops + reduction sum (axis/keepdims)
- `fnp-random`: deterministic RNG stream contracts + bounded guardrail checks
- `fnp-linalg`: first-wave solver/factorization/spectral contract surfaces with fixture-driven conformance checks
- `fnp-io`: first-wave NPY/NPZ parser/writer contract surfaces with hardened adversarial boundary checks
- `fnp-conformance`: fixture suites + adversarial security suites + oracle capture + differential report + benchmark baseline + RaptorQ sidecars/scrub/decode proofs
  + workflow scenario corpus + golden journey gate with strict/hardened replay logs

Program doctrine:
- Current implementation is an in-progress parity wave, not a reduced-scope endpoint.
- Remaining behavior gaps are parity debt to be closed, not accepted feature cuts.

## Conformance and Artifact Commands

```bash
cargo run -p fnp-conformance --bin capture_numpy_oracle
cargo run -p fnp-conformance --bin run_ufunc_differential
cargo run -p fnp-conformance --bin generate_benchmark_baseline
cargo run -p fnp-conformance --bin run_performance_budget_gate
cargo run -p fnp-conformance --bin generate_raptorq_sidecars
cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001
cargo run -p fnp-conformance --bin run_security_gate
cargo run -p fnp-conformance --bin run_test_contract_gate
cargo run -p fnp-conformance --bin run_workflow_scenario_gate
scripts/e2e/run_performance_budget_gate.sh
scripts/e2e/run_security_policy_gate.sh
scripts/e2e/run_test_contract_gate.sh
scripts/e2e/run_workflow_scenario_gate.sh
scripts/e2e/run_ci_gate_topology.sh
```

Example: use `uv` Python 3.14 + NumPy for capture:

```bash
uv venv --python 3.14 .venv-numpy314
uv pip install --python .venv-numpy314/bin/python numpy
FNP_ORACLE_PYTHON="$(pwd)/.venv-numpy314/bin/python3" cargo run -p fnp-conformance --bin capture_numpy_oracle
```

Notes:
- Oracle capture prefers legacy NumPy import, then system NumPy.
- Oracle interpreter is configurable with `FNP_ORACLE_PYTHON` (defaults to `python3`).
- If neither is importable in the environment, it uses `pure_python_fallback` and records this in the oracle artifact.
- `scripts/e2e/run_ci_gate_topology.sh` runs oracle capture before G3 differential and fails in CI when fallback is used (`FNP_REQUIRE_REAL_NUMPY_ORACLE=1`, auto-enabled for `CI=1`/`GITHUB_ACTIONS=true`).
- Packet contract schema lock is versioned as `phase2c-contract-v1` in `artifacts/contracts/`.
- Security threat controls are machine-mapped in `artifacts/contracts/security_control_checks_v1.yaml`.
- Runtime-policy e2e logs are emitted as JSONL under `artifacts/logs/` by `run_security_gate` / `scripts/e2e/run_security_policy_gate.sh`.
- Workflow-scenario e2e logs are emitted as JSONL under `artifacts/logs/` by `run_workflow_scenario_gate` / `scripts/e2e/run_workflow_scenario_gate.sh`.
- Performance budget baseline-delta reports are emitted under `artifacts/logs/` by `run_performance_budget_gate` / `scripts/e2e/run_performance_budget_gate.sh`.
- Test/logging conventions are locked in `artifacts/contracts/test_logging_contract_v1.json` and `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md`.

## CI Gate Topology (G1..G8)

Canonical ordered gate runner:

```bash
scripts/e2e/run_ci_gate_topology.sh
```

Ordered gates (`fast -> heavy`):

1. `G1` fmt + lint: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`
2. `G2` unit/property: `cargo test --workspace --lib`
3. `G3` oracle capture + differential: `cargo run -p fnp-conformance --bin capture_numpy_oracle` then `cargo run -p fnp-conformance --bin run_ufunc_differential`
4. `G4` adversarial/security: `scripts/e2e/run_security_policy_gate.sh`
5. `G5` test/logging contract: `scripts/e2e/run_test_contract_gate.sh`
6. `G6` workflow e2e + forensics: `scripts/e2e/run_workflow_scenario_gate.sh`
7. `G7` performance budget: `scripts/e2e/run_performance_budget_gate.sh`
8. `G8` durability/decode proof: `scripts/e2e/run_raptorq_gate.sh`

Topology contract and policy lock:

- Machine contract: `artifacts/contracts/ci_gate_topology_v1.json`
- Contract enforcement suite: `ci_gate_topology_contract` in `run_test_contract_gate`
- G6 forensics output requirements: `artifact_index_path` + `replay_command`

Branch-protection policy (required checks + merge blockers) is captured in
`artifacts/contracts/ci_gate_topology_v1.json` and blocks merges on:

- compatibility drift
- missing required artifacts
- budget breach

`run_test_contract_gate` enforces that G7 (`performance-budget`) and G8 (`durability-decode`) remain in required checks and keep their mandatory output fields.

## Repository Layout

- `crates/fnp-dtype`
- `crates/fnp-ndarray`
- `crates/fnp-iter`
- `crates/fnp-ufunc`
- `crates/fnp-linalg`
- `crates/fnp-random`
- `crates/fnp-io`
- `crates/fnp-conformance`
- `crates/fnp-runtime`
- `legacy_numpy_code/numpy` (behavioral oracle)

## Required Checks

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --workspace
```

Additional when available:

```bash
cargo test -p fnp-conformance -- --nocapture
cargo bench
```

## Next Work

1. Expand `FNP-P2C-005` corpus to adversarial and high-dimensional edge cases.
2. Move oracle capture to true legacy/system NumPy path in CI images.
3. Expand `FNP-P2C-007` (RNG), `FNP-P2C-008` (linalg), and `FNP-P2C-009` (npy/npz) from first-wave coverage to full legacy-matrix parity breadth.
4. Promote benchmark + sidecar checks to mandatory CI gates.
5. Publish and continuously burn down full legacy NumPy parity debt until zero.
