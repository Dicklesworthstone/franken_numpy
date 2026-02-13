# FrankenNumPy

<div align="center">
  <img src="franken_numpy_illustration.webp" alt="FrankenNumPy - memory-safe clean-room NumPy reimplementation in Rust">
</div>

FrankenNumPy is a memory-safe, clean-room Rust reimplementation of NumPy with two simultaneous goals:

1. Drop-in behavioral compatibility for scoped APIs.
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

- `fnp-dtype`: scoped deterministic promotion/cast foundations
- `fnp-ndarray`: broadcast legality, reshape `-1` inference, contiguous stride calculus (C/F)
- `fnp-runtime`: strict/hardened decision policy + fail-closed wire decoding + audited override gate + evidence ledger
- `fnp-ufunc`: broadcasted binary ops + reduction sum (axis/keepdims)
- `fnp-conformance`: fixture suites + adversarial security suites + oracle capture + differential report + benchmark baseline + RaptorQ sidecars/scrub/decode proofs

## Conformance and Artifact Commands

```bash
cargo run -p fnp-conformance --bin capture_numpy_oracle
cargo run -p fnp-conformance --bin run_ufunc_differential
cargo run -p fnp-conformance --bin generate_benchmark_baseline
cargo run -p fnp-conformance --bin generate_raptorq_sidecars
cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001
cargo run -p fnp-conformance --bin run_security_gate
scripts/e2e/run_security_policy_gate.sh
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
- Packet contract schema lock is versioned as `phase2c-contract-v1` in `artifacts/contracts/`.
- Security threat controls are machine-mapped in `artifacts/contracts/security_control_checks_v1.yaml`.
- Runtime-policy e2e logs are emitted as JSONL under `artifacts/logs/` by `run_security_gate` / `scripts/e2e/run_security_policy_gate.sh`.

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
3. Implement `FNP-P2C-007` (RNG), `FNP-P2C-008` (linalg), and `FNP-P2C-009` (npy/npz).
4. Promote benchmark + sidecar checks to mandatory CI gates.
