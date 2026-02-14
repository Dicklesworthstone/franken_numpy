# TODO_GRANULAR_EXECUTION

Status key:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked/deferred with explicit reason

## 0. Session Control

- [x] Confirm `AGENTS.md` constraints and no-destructive policy
- [x] Reconfirm absolute parity doctrine: complete drop-in legacy NumPy overlap is mandatory; reduced-scope completion is forbidden.
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
- [x] Ufunc coverage is currently limited to arithmetic binary ops + `sum` reductions; remaining NumPy API behavior is explicit parity debt that blocks any "drop-in complete" claim until closed.

### 7.2 Highest-Value Next Steps (Captured)
- [x] Add dedicated criterion benchmark targets and wire regression thresholds.
- [x] Expand oracle corpus to larger shape, dtype-promotion, NaN/Inf, and aliasing/view edge cases.
- [x] Run capture pipeline against legacy NumPy runtime once local environment can import `legacy_numpy_code/numpy`.
- [x] Continue port wave into planned crates (`fnp-fft`, `fnp-random` full API, `fnp-linalg` parity expansion) with the same differential + durability artifact pattern.
- [x] Keep full legacy feature/functionality parity matrix current and require each session to reduce owned parity debt.

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
- [x] Wire machine validation/gating for threat-matrix + allowlist artifacts
- [x] Add conformance/adversarial fixture hooks tied to the threat classes

### 10.3 Follow-on implementation (current pass)
- [x] Add executable security control map: `artifacts/contracts/security_control_checks_v1.yaml`
- [x] Add machine-check suite: `crates/fnp-conformance/src/security_contracts.rs`
- [x] Extend runtime policy fixtures with structured log context fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`)
- [x] Add adversarial runtime-policy fixture corpus: `runtime_policy_adversarial_cases.json`
- [x] Add fail-closed wire-decoding path for unknown runtime mode/class in `fnp-runtime`
- [x] Add explicit audited override gate evaluator in `fnp-runtime`
- [x] Add runtime policy JSONL logging path support in `fnp-conformance`
- [x] Add e2e security gate binary: `run_security_gate`
- [x] Add e2e wrapper script: `scripts/e2e/run_security_policy_gate.sh`

### 10.4 Validation outcomes (current pass)
- [x] `cargo fmt --all`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p fnp-conformance -- --nocapture`
- [x] `cargo bench`
- [x] `scripts/e2e/run_security_policy_gate.sh`

## 11. Round-3 Extreme Optimization + Alien Uplift (Current Pass)

### 11.1 Profile-first baseline capture
- [x] Build release benchmark binary (`generate_benchmark_baseline`)
- [x] Capture pre-change hyperfine artifact (`round3_before`)
- [x] Capture pre-change baseline snapshot (`ufunc_benchmark_baseline_round3_before.json`)
- [x] Attempt `perf` profiling and record environment restriction (`perf_event_paranoid=4`)
- [x] Capture fallback syscall profile (`strace -c`) for pre-change run

### 11.2 One-lever implementation
- [x] Replace per-element axis reduction reindexing with contiguous kernel in `crates/fnp-ufunc/src/lib.rs`
- [x] Keep API and dtype semantics unchanged
- [x] Add non-last-axis ordering regression test (`axis=0`, 3D array)
- [x] Add empty-axis zero-initialized output regression test

### 11.3 Rebaseline and quantify
- [x] Capture post-change hyperfine artifact (`round3_after`)
- [x] Capture post-change baseline snapshot (`ufunc_benchmark_baseline_round3_after.json`)
- [x] Capture fallback syscall profile (`strace -c`) for post-change run
- [x] Confirm command-level mean latency delta (`22.924 ms -> 10.070 ms`, `-56.07%`)
- [x] Confirm targeted reduction percentile deltas (`~ -90%` p50/p95/p99 on axis reduction workload)

### 11.4 Proof and decision artifacts
- [x] Generate round-3 golden checksum manifest (`artifacts/proofs/golden_checksums_round3.txt`)
- [x] Verify checksum manifest (`sha256sum -c`)
- [x] Add round-3 isomorphism proof (`artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md`)
- [x] Add round-3 opportunity matrix (`artifacts/optimization/ROUND3_OPPORTUNITY_MATRIX.md`)
- [x] Add round-3 alien recommendation cards (`artifacts/decisions/ALIEN_GRAVEYARD_RECOMMENDATION_CARDS_ROUND3.md`)

### 11.5 Open-bead triage + diagnostics
- [x] Run `bv --robot-next`, `bv --robot-priority`, `bv --robot-triage`
- [x] Run `bv --robot-alerts` (no active critical alerts)
- [x] Run `bv --robot-insights` for critical path and bottleneck update
- [x] Run `bv --robot-suggest` and defer low-signal dependency suggestions pending manual validation

## 12. Bead `bd-23m.5` — Unit/Property Test Conventions + Structured Log Contract (Current Pass)

### 12.1 Triage and impact confirmation
- [x] Re-run `bv --robot-next` and confirm `bd-23m.5` remains top-impact work item
- [x] Verify `bd-23m.5` is still `in_progress`
- [x] Re-check dependency fan-out and unblock counts before implementation

### 12.2 Machine contract completion
- [x] Add versioned machine contract artifact: `artifacts/contracts/test_logging_contract_v1.json`
- [x] Define required structured log fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`)
- [x] Define mandatory invariant families for unit/property/adversarial suites
- [x] Define shrink requirements (`deterministic_seed_replay`, bounded `max_shrink_steps`, shrink reason-code requirement)
- [x] Define required test helper API anchors
- [x] Define required gate script list for enforcement
- [x] Define required fixture collection list for contract coverage

### 12.3 Human-readable conventions companion
- [x] Add conventions companion doc: `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md`
- [x] Document invariant families and suite anchors
- [x] Document shrink strategy and deterministic replay rules
- [x] Document fixture ID policy and examples
- [x] Document gate commands and fail criteria
- [x] Map bead decisions to alien graveyard + FrankenSuite summary section IDs
- [x] Document RaptorQ applicability boundary for durable vs ephemeral artifacts

### 12.4 Conformance enforcement wiring
- [x] Add contract validation module: `crates/fnp-conformance/src/test_contracts.rs`
- [x] Validate schema version and contract version IDs
- [x] Validate required structured log fields, invariant families, shrink settings, helper APIs, gate script declarations
- [x] Validate gate script existence from repo root
- [x] Validate fixture-level requirements (naming policy, seeds, reason codes, artifact refs)
- [x] Add module tests (`fixture_id_policy_enforced`, `test_contract_suite_is_green`)
- [x] Wire `run_test_contract_suite` into `run_all_core_suites`

### 12.5 Gate tooling and e2e enforcement
- [x] Add gate binary: `crates/fnp-conformance/src/bin/run_test_contract_gate.rs`
- [x] Execute contract + runtime policy suites from one gate command
- [x] Parse emitted JSONL runtime-policy logs and enforce required fields per entry
- [x] Add wrapper script: `scripts/e2e/run_test_contract_gate.sh`
- [x] Mark wrapper script executable

### 12.6 Documentation updates
- [x] Update `artifacts/contracts/README.md` with test/logging contract artifacts
- [x] Update `README.md` conformance command list with test-contract gate commands
- [x] Update broader spec docs (`COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md`, `PLAN_TO_PORT_NUMPY_TO_RUST.md`) with finalized test-contract references

### 12.7 Validation and closure
- [x] `cargo fmt --check`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p fnp-conformance -- --nocapture`
- [x] `scripts/e2e/run_test_contract_gate.sh`
- [x] `cargo bench`
- [x] Add bead completion notes + evidence links via `br comments add`
- [x] Close `bd-23m.5` with `br close` once all acceptance evidence is green

## 13. Bead `bd-23m.24.1` — DOC-PASS-00 Baseline Gap Matrix + Quantitative Expansion Targets (Current Pass)

### 13.1 Triage and claim
- [x] Re-run `bv --robot-next` after closing `bd-23m.5`
- [x] Confirm top actionable bead is `bd-23m.24.1`
- [x] Mark `bd-23m.24.1` as `in_progress`

### 13.2 Baseline measurement
- [x] Measure baseline line counts for target docs (`EXHAUSTIVE_LEGACY_ANALYSIS.md`, `EXISTING_NUMPY_STRUCTURE.md`)
- [x] Define explicit target line counts and expansion multipliers (12.0x and 16.0x)
- [x] Add stable `DOC-PASS-00` section anchors in both target docs

### 13.3 Pass-1 gap matrix integration
- [x] Add per-domain gap matrix to `EXHAUSTIVE_LEGACY_ANALYSIS.md` with legacy anchors
- [x] Add pass-1 gap matrix to `EXISTING_NUMPY_STRUCTURE.md` with traceability anchors
- [x] Record explicit unit/property, differential, e2e, and structured logging implication status
- [x] Link assertions to executable evidence artifacts where available (`fnp-conformance` fixtures/contracts/gates)

### 13.4 Contradictions and unknowns
- [x] Add contradiction register with owner, risk, and closure criteria in both target docs
- [x] Flag reduced-scope V1 language as deprecated and closure-required (`DOC-C001`)
- [ ] Reconcile remaining doc contradictions in downstream passes (`bd-23m.24.2`, `.24.3`, `.24.4`, `.24.10`)

### 13.5 Bead closure steps
- [x] Add `br comments add` completion note with matrix/evidence links
- [x] Close `bd-23m.24.1` after acceptance criteria verification

## 14. Packet `FNP-P2C-005` Kickoff (`bd-23m.16` / `bd-23m.16.1`) (Current Pass)

### 14.1 Triage and claim
- [x] Re-run `bv --robot-next` after closing `bd-23m.24.1`
- [x] Mark packet feature `bd-23m.16` as `in_progress`
- [x] Mark packet sub-bead `bd-23m.16.1` as `in_progress`

### 14.2 Legacy anchor extraction
- [x] Extract concrete symbol anchors from `ufunc_object.c` (`_parse_signature`, `_ufunc_setup_flags`, `convert_ufunc_arguments`, `ufunc_get_name_cstr`, signature normalization/resolution calls)
- [x] Extract dispatch and override anchors from `dispatching.h`, `override.c`, `reduction.c`, `ufunc_type_resolution.c`
- [x] Extract oracle test anchors from `test_ufunc.py`, `test_umath.py`, `test_overrides.py`

### 14.3 Artifact authoring for `FNP-P2C-005-A`
- [x] Create `artifacts/phase2c/FNP-P2C-005/legacy_anchor_map.md`
- [x] Create `artifacts/phase2c/FNP-P2C-005/behavior_extraction_ledger.md`
- [x] Map legacy anchors to planned Rust module boundaries
- [x] Record strict/hardened observable contracts and compatibility invariants
- [x] Record unknown/under-specified edges with owner, risk, and closure criteria
- [x] Record planned unit/property, differential/metamorphic/adversarial, and e2e verification hooks
- [x] Include structured logging contract linkage (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`)

### 14.4 Packet follow-on
- [x] Add bead notes/evidence links via `br comments add` for `bd-23m.16.1`
- [x] Close `bd-23m.16.1` if acceptance criteria are fully satisfied

## 15. Packet `FNP-P2C-005-B` Contract Table (`bd-23m.16.2`) (Kickoff)

### 15.1 Claim and handoff
- [x] Confirm `bd-23m.16.2` is ready after closing `bd-23m.16.1`
- [x] Mark `bd-23m.16.2` as `in_progress`
- [ ] Draft strict/hardened contract table artifact for packet boundary
- [ ] Attach unit/property + e2e scenario mapping and reason-code requirements
