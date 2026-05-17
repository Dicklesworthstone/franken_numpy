# PROPOSED_ARCHITECTURE

> **Status note (2026-05-16):** This document is the early architecture plan written when the codebase was at the FNP-P2C-005 slice (Section 6 below still describes that snapshot). The plan's principles, layering, SCE contract, runtime mode matrix, and optimization governance remain accurate. The "Current implementation" line items in §6 are now severely understated — fnp-ufunc has grown from 4 elementwise ops + 1 reduction to 35 binary / 43 unary / 30+ reduction methods + FFT + einsum + masked / string / datetime arrays + polynomials. For the live per-crate inventory see README's "Workspace and Crate Map" section, and for the 100% numpy.__all__ surface achievement see `audit_numpy_reality.md`.

## 1. Architecture Principles

1. Spec-first implementation from extraction packets.
2. Strict/hardened compatibility mode split is mandatory.
3. Fail-closed behavior for unknown or incompatible semantics.
4. Profile-first optimization with one-lever, proof-backed changes.
5. Durable evidence artifacts with RaptorQ sidecar contracts.

## 2. Layering

`array API -> shape/stride engine (SCE) -> dispatcher -> kernels -> IO`

The `fnp-python` PyO3 bindings sit above this chain as the Python-facing surface, exposing the array API to NumPy callers without altering the canonical Rust layering below it.

## 3. Crate Map

- `fnp-dtype`: dtype taxonomy, promotion table, cast policy primitives.
- `fnp-ndarray`: shape legality, stride calculus, reshape/broadcast contracts.
- `fnp-iter`: nditer-like traversal and overlap-safe iteration contracts.
- `fnp-ufunc`: ufunc dispatch, broadcasting execution, reductions.
- `fnp-linalg`: linear algebra adapters and scoped solver contracts.
- `fnp-random`: deterministic RNG streams and state schemas.
- `fnp-io`: npy/npz parser + writer with hardened boundary checks.
- `fnp-python`: PyO3 bindings exposing 100% of `numpy.__all__` (499/499 names) with structural CI lock-in.
- `fnp-conformance`: differential harness, adversarial policy harness, security-contract validator, oracle capture, benchmark + RaptorQ artifact tooling.
- `fnp-runtime`: mode split, fail-closed wire decoding, explicit override-audit gate, decision/evidence ledger, policy gate orchestration.

## 4. Stride Calculus Engine (SCE) Contract

SCE owns deterministic legality and transformation rules:

1. `shape -> element_count` with overflow checks.
2. `shape + order + item_size -> strides` (C/F contiguous baselines).
3. `lhs_shape + rhs_shape -> broadcast_shape` deterministically.
4. `old_count + reshape_spec -> resolved_shape` with NumPy-style `-1` semantics.
5. Alias-sensitive transitions rejected when invariants are violated.

SCE is the non-negotiable compatibility kernel.

## 5. Runtime Mode Matrix

| Input Class | Strict Mode | Hardened Mode |
|---|---|---|
| Known compatible + low risk | allow | allow |
| Known compatible + high risk | allow | full_validate |
| Unknown semantics | fail_closed | fail_closed |
| Known incompatible semantics | fail_closed | fail_closed |

All decisions are recorded in an evidence ledger.
Unknown wire mode/class inputs are fail-closed.

## 6. Implemented `FNP-P2C-005` Slice

Current implementation in `fnp-ufunc`:

- broadcasted binary elementwise ops: add/sub/mul/div
- reduction: `sum` with `axis` + `keepdims` support
- shape/value/dtype checks through fixture-driven differential suites

Current differential harness in `fnp-conformance`:

- fixture schema for ufunc/reduction inputs
- oracle capture binary (`capture_numpy_oracle`)
- comparator + machine-readable differential report (`run_ufunc_differential`)
- fallback source tagging (`legacy`, `system`, `pure_python_fallback`)

## 7. Integration Hooks (asupersync + frankentui)

### asupersync usage plan

- async orchestration for conformance capture and artifact pipelines
- cancellation-safe long-running benchmark/conformance jobs
- structured telemetry channels for evidence/event streams

Current state:
- `fnp-runtime` exposes typed optional capability snapshots for deterministic lab runtime, explicit capability contexts, RaptorQ encoding pipelines, and structured outcomes
- `fnp-conformance` uses asupersync RaptorQ primitives for sidecar generation and scrub/recovery drills

### frankentui usage plan

- terminal-native observability dashboards for parity drift and performance deltas
- interactive incident/recovery views for hardened-mode decisions

Current state:
- `fnp-runtime` exposes typed optional `frankentui` capability snapshots for render buffers, frames, themes, and terminal capability discovery

## 8. Performance/Optimization Governance

Every optimization follows:

1. baseline (`p50/p95/p99`, memory),
2. profile hotspot,
3. score opportunity (`impact * confidence / effort`),
4. implement one lever,
5. prove isomorphism,
6. re-baseline.

Implemented baseline generator:
- `cargo run -p fnp-conformance --bin generate_benchmark_baseline`

## 9. Conformance/Artifact Pipeline

For each feature family:

1. input fixtures,
2. oracle capture,
3. target execution,
4. parity comparison report,
5. durability sidecars + scrub + decode proof.

Implemented commands:

```bash
cargo run -p fnp-conformance --bin capture_numpy_oracle
cargo run -p fnp-conformance --bin run_ufunc_differential
cargo run -p fnp-conformance --bin generate_benchmark_baseline
cargo run -p fnp-conformance --bin generate_raptorq_sidecars
cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001
cargo run -p fnp-conformance --bin run_security_gate
scripts/e2e/run_security_policy_gate.sh
```

Operational detail:
- capture uses configurable interpreter `FNP_ORACLE_PYTHON` (fallback `python3`).
- packet readiness uses `phase2c-contract-v1` mandatory-field validation and emits `not_ready` when required fields/files are missing.
- security threat controls are machine-validated against `SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md`, `hardened_mode_allowlist_v1.yaml`, and `security_control_checks_v1.yaml`.
- runtime/adversarial policy gates emit deterministic JSONL evidence with `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`.

## 10. Security and Compatibility Boundaries

- parser/IO boundaries hardened and fuzzed first.
- shape/cast transitions are explicit, audited state transitions.
- unknown metadata or unsupported protocol fields fail closed.
- strict/hardened divergence is explicitly reported.
