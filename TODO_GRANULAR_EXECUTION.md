# TODO_GRANULAR_EXECUTION

Status key:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked/deferred with explicit reason

## 0. Session Control

- [x] Confirm `AGENTS.md` constraints and no-destructive policy
- [x] Confirm requested workstreams:
  - `FNP-P2C-005` ufunc/reduction core
  - legacy oracle capture + differential pipeline
  - RaptorQ sidecar generation + scrub + decode proof for conformance + benchmark bundles
- [x] Keep this tracker updated after each meaningful subtask

## 1. Workstream A — `FNP-P2C-005` Ufunc/Reduction Core

### 1.1 API and data-model design
- [x] Define `UFuncArray` data structure (shape + contiguous values + dtype tag)
- [x] Define `UFuncError` contract
- [x] Define `BinaryOp` enum and op dispatch contract
- [x] Define reduction contract (`axis`, `keepdims`)

### 1.2 Broadcasted binary execution
- [x] Implement output-shape derivation via `fnp-ndarray::broadcast_shape`
- [x] Implement contiguous C-order index mapping helper
- [x] Implement broadcasted source-index resolution for lhs/rhs
- [x] Implement elementwise `add`
- [x] Implement elementwise `sub`
- [x] Implement elementwise `mul`
- [x] Implement elementwise `div`
- [x] Validate divide-by-zero behavior contract for floating outputs

### 1.3 Reduction execution
- [x] Implement full-array sum (axis None)
- [x] Implement axis-specific sum for N-D arrays
- [x] Implement `keepdims=true` output-shape policy
- [x] Implement `keepdims=false` output-shape policy
- [x] Implement axis-bound checks and errors

### 1.4 Tests for workstream A
- [x] Unit tests for binary broadcasting legality
- [x] Unit tests for binary value correctness
- [x] Unit tests for reduction correctness (axis None)
- [x] Unit tests for reduction correctness (axis set + keepdims variants)
- [x] Unit tests for error paths (shape mismatch, axis OOB)

### 1.5 Integration wiring
- [x] Add crate dependencies (`fnp-ufunc` -> `fnp-ndarray`, `fnp-dtype`)
- [x] Add conformance crate dependency on `fnp-ufunc`
- [x] Replace placeholder tests in `fnp-ufunc`

## 2. Workstream B — Legacy Oracle Capture + Differential Pipeline

### 2.1 Fixture schemas
- [x] Define normalized ufunc input-case schema
- [x] Define normalized oracle-output schema
- [x] Define differential-report schema

### 2.2 Capture pipeline
- [x] Implement conformance binary: `capture_numpy_oracle`
- [x] Add robust python invocation (`python3` + embedded script)
- [x] Attempt legacy-path import (`legacy_numpy_code/numpy`) first
- [x] Add fallback to system NumPy when legacy import is unavailable
- [x] Emit source indicator (`legacy` vs `system`) in output artifact
- [x] Write deterministic JSON output ordering
- [x] Add pure-Python fallback oracle path when both legacy and system NumPy imports are unavailable

### 2.3 Differential comparator
- [x] Implement conformance suite loading input + oracle output
- [x] Execute `fnp-ufunc` for each case
- [x] Compare shapes exactly
- [x] Compare values with explicit tolerance policy
- [x] Emit machine-readable parity report artifact

### 2.4 Fixtures and reports
- [x] Add first ufunc/reduction input corpus JSON
- [x] Generate first oracle output JSON via capture binary
- [x] Generate first differential report JSON via comparator
- [x] Wire suite into `run_all_core_suites`

### 2.5 Tests and CLI coverage
- [x] Add integration test covering differential suite pass path
- [x] Add integration test for missing oracle artifact path
- [x] Add documentation for capture + compare command sequence

## 3. Workstream C — RaptorQ Sidecars, Scrub, Decode Proof

### 3.1 Sidecar artifact contract
- [x] Define sidecar JSON schema for encoded symbols + metadata
- [x] Define scrub report JSON schema
- [x] Define decode proof JSON schema

### 3.2 asupersync integration
- [x] Add `asupersync` dependency in `fnp-conformance`
- [x] Implement sidecar generation using `asupersync` encoding primitives
- [x] Include deterministic `object_id`, symbol params, and source hash
- [x] Persist encoded symbol records in sidecar artifact

### 3.3 Scrub and decode proof
- [x] Implement decode-from-sidecar validation path
- [x] Verify decoded bytes hash against source hash
- [x] Emit scrub status (`ok`/`failed`) report
- [x] Emit decode proof artifact with deterministic fields

### 3.4 Bundle targets
- [x] Conformance bundle target: fixture JSON set + differential report
- [x] Benchmark bundle target: baseline report JSON
- [x] Generate sidecars for both targets
- [x] Generate scrub + decode proof for both targets

### 3.5 Tests
- [x] Unit/integration test for sidecar encode->decode roundtrip
- [x] Integration test for scrub failure on tampered symbol payload
- [x] Stabilize recovery-drill selection to choose a deterministic recoverable dropped-symbol candidate when available

## 4. Workstream D — Bench Baseline Artifacts

### 4.1 Benchmark harness
- [x] Implement benchmark baseline generator binary (JSON output)
- [x] Record p50/p95/p99 for ufunc and reduction sentinel workloads
- [x] Record metadata: timestamp, commit hash (if available), workload sizes

### 4.2 Artifact wiring
- [x] Write benchmark baseline artifact under `artifacts/baselines`
- [x] Ensure RaptorQ sidecar generation consumes this artifact

## 5. Documentation and Tracking Updates

- [x] Update `FEATURE_PARITY.md` statuses for new suites/artifacts
- [x] Update `PROPOSED_ARCHITECTURE.md` with implemented pipeline details
- [x] Update `README.md` commands for capture/diff/sidecar/scrub
- [x] Add/refresh optimization matrix and isomorphism proof docs if behavior changes

## 6. Validation and Quality Gates

- [x] `cargo fmt --check`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p fnp-conformance -- --nocapture`
- [x] `cargo bench`

### 6.1 Validation Notes
- [x] Initial `cargo test --workspace` run caught one failing test:
  - `raptorq_artifacts::tests::sidecar_roundtrip_scrub_is_ok`
- [x] Applied fix in `crates/fnp-conformance/src/raptorq_artifacts.rs`:
  - recovery drill now scans deterministic drop candidates and prefers the first candidate that proves successful recovery
- [x] Re-ran full gate sequence after fix; all commands passed
- [x] `cargo bench` completed successfully; no dedicated benchmark targets are yet defined, so unit tests were run in bench profile with all tests ignored

## 7. Landing-The-Plane Checklist

- [x] Confirm no destructive operations were executed
- [x] Summarize all changes and rationale
- [x] List residual risks and highest-value next steps
- [x] Confirm method-stack artifacts produced or explicitly deferred with reason

### 7.1 Residual Risks (Captured)
- [x] Oracle source now uses `system` NumPy (via local `uv` Python 3.14 venv), but parity is not yet anchored to the vendored legacy NumPy runtime path.
- [x] `cargo bench` gate passes but workspace does not yet expose dedicated `[[bench]]` targets; benchmark evidence currently comes from `generate_benchmark_baseline` artifact pipeline.
- [x] Ufunc coverage currently scoped to arithmetic binary ops + `sum` reductions; broader NumPy API surface remains open for subsequent parity waves.

### 7.2 Highest-Value Next Steps (Captured)
- [x] Add dedicated criterion benchmark targets and wire regression thresholds.
- [x] Expand oracle corpus to larger shape, dtype-promotion, NaN/Inf, and aliasing/view edge cases.
- [x] Run capture pipeline against legacy NumPy runtime once local environment can import `legacy_numpy_code/numpy`.
- [x] Continue port wave into planned crates (`fnp-fft`, `fnp-random` full API, `fnp-linalg` parity expansion) with the same differential + durability artifact pattern.

## 8. Post-Landing Follow-On (This Session)

### 8.1 Real NumPy environment enablement
- [x] Create local `uv` venv with Python 3.14 (`.venv-numpy314`)
- [x] Install NumPy in local venv
- [x] Re-run oracle capture against real NumPy runtime (`oracle_source=system`)

### 8.2 Oracle runtime robustness
- [x] Fix dtype alias mismatch in capture script (`f64` -> `float64`, etc.) so real NumPy capture succeeds
- [x] Add explicit interpreter override env var support (`FNP_ORACLE_PYTHON`) with `python3` fallback
- [x] Improve capture failure diagnostics to include selected interpreter path/name

### 8.3 Validation after follow-on changes
- [x] Re-run `capture_numpy_oracle` with explicit venv interpreter override
- [x] Re-run `run_ufunc_differential` (green: 7/7)
- [x] Re-run `cargo test -p fnp-conformance` (green)
- [x] Re-run `cargo fmt --check` (green)

## 9. Bead `bd-23m.3` — Contract Schema + Artifact Topology Lock

### 9.1 Triage and ownership
- [x] Select top-impact bead via `bv --robot-next` (`bd-23m.3`)
- [x] Mark bead `in_progress` via `br update`
- [x] Reserve edit paths via Agent Mail reservations (`PlumCarp`)

### 9.2 Implementation
- [x] Add versioned contract schema lock artifacts under `artifacts/contracts/`
- [x] Add machine-checkable packet validator module in `fnp-conformance`
- [x] Add binary command for packet readiness validation (`validate_phase2c_packet`)
- [x] Enforce missing-file and missing-field => `not_ready`
- [x] Add unit tests for ready, missing-file, and missing-field paths

### 9.3 Documentation and tracking
- [x] Update `README.md` with packet-validator command
- [x] Update `PROPOSED_ARCHITECTURE.md` with readiness-validator details
- [x] Update `FEATURE_PARITY.md` foundation status for contract-schema lock
- [x] Run formatter/check/tests for this bead and record outcomes
- [x] Post completion update in Agent Mail thread `bd-23m.3`
- [x] Close `bd-23m.3` if validation passes

### 9.4 Validation outcomes
- [x] `cargo fmt --check`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p fnp-conformance -- --nocapture`
- [x] `cargo bench`

## 10. Bead `bd-23m.2` — Security/Compatibility Threat Matrix (Started)

### 10.1 Triage and state
- [x] Re-run `bv --robot-next` after closing `bd-23m.3` and select next top-impact bead (`bd-23m.2`)
- [x] Mark `bd-23m.2` as `in_progress`
- [x] Send kickoff update in Agent Mail thread `bd-23m.2`

### 10.2 Initial coding started
- [x] Add versioned threat matrix artifact: `artifacts/contracts/SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md`
- [x] Add versioned hardened-mode allowlist artifact: `artifacts/contracts/hardened_mode_allowlist_v1.yaml`
- [ ] Wire machine validation/gating for threat-matrix + allowlist artifacts
- [ ] Add conformance/adversarial fixture hooks tied to the threat classes
