# EXHAUSTIVE_LEGACY_ANALYSIS.md — FrankenNumPy

Date: 2026-02-13  
Method stack: `$porting-to-rust` Phase-2 Deep Extraction + `$alien-artifact-coding` + `$extreme-software-optimization` + RaptorQ durability + frankenlibc/frankenfs strict/hardened doctrine.

## 0. Mission and Completion Criteria

This document defines exhaustive legacy extraction for FrankenNumPy. Phase-2 is complete only when each scoped subsystem has:
1. explicit invariants,
2. explicit crate ownership,
3. explicit oracle families,
4. explicit strict/hardened policy behavior,
5. explicit performance and durability gates.

## 1. Source-of-Truth Crosswalk

Legacy corpus:
- `/data/projects/franken_numpy/legacy_numpy_code/numpy`
- Upstream oracle: `numpy/numpy`

Project contracts:
- `/data/projects/franken_numpy/COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md` (sections 14-21)
- `/data/projects/franken_numpy/EXISTING_NUMPY_STRUCTURE.md`
- `/data/projects/franken_numpy/PLAN_TO_PORT_NUMPY_TO_RUST.md`
- `/data/projects/franken_numpy/PROPOSED_ARCHITECTURE.md`
- `/data/projects/franken_numpy/FEATURE_PARITY.md`

## DOC-PASS-00 Baseline Gap Matrix + Quantitative Expansion Targets

Snapshot baseline (2026-02-14):

| Document | Baseline lines | Target lines | Expansion multiplier | Anchor requirement |
|---|---:|---:|---:|---|
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 275 | 3300 | 12.0x | Every new assertion links to legacy path and/or executable artifact |
| `EXISTING_NUMPY_STRUCTURE.md` | 62 | 992 | 16.0x | Every new subsystem map row links to crate owner + conformance anchor |

Pass-1 domain gap matrix (traceable to legacy anchors and executable evidence):

| Domain | Legacy anchors | Current depth (0-5) | Target multiplier | Unit/Property implications | Differential implications | E2E implications | Structured logging implications | Evidence anchors |
|---|---|---:|---:|---|---|---|---|---|
| Shape/stride legality | `numpy/_core/src/multiarray/shape.c`, `shape.h` | 3 | 3.0x | partial (`shape_stride_cases.json`) | partial (`run_ufunc_differential`) | missing dedicated journey | partial (runtime policy only) | `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `crates/fnp-conformance/src/lib.rs` |
| Dtype promotion/casting | `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | 2 | 4.0x | partial (`dtype_promotion_cases.json`) | missing cast-matrix differential | missing dedicated journey | missing per-cast reason codes | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` |
| Transfer/alias semantics | `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | 1 | 6.0x | missing overlap property suite | missing overlap differential | missing overlap journey | missing overlap log taxonomy | legacy anchors only (no Rust artifact yet) |
| Ufunc dispatch/override | `umath/ufunc_object.c`, dispatch loops | 3 | 3.0x | partial (metamorphic/adversarial corpus) | partial (`ufunc_differential`) | missing override journey | partial (`fixture_id`,`seed` in fixtures) | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_*` |
| NDIter traversal | `multiarray/nditer*` | 1 | 6.0x | missing | missing | missing | missing | `artifacts/phase2c/FNP-P2C-006/legacy_anchor_map.md`, `artifacts/phase2c/FNP-P2C-006/behavior_extraction_ledger.md` |
| Random subsystem | `numpy/random/*.pyx`, `random/src/*` | 1 | 5.0x | missing deterministic stream properties | missing RNG differential | missing seed/state journey | missing RNG log taxonomy | `crates/fnp-random/src/lib.rs` (stub) |
| Linalg subsystem | `numpy/linalg/lapack_lite/*` | 1 | 5.0x | missing solver invariants | missing solver differential | missing solver journey | missing linalg reason codes | `crates/fnp-linalg/src/lib.rs` (stub) |
| IO subsystem | `numpy/lib/format.py`, npy/npz paths | 2 | 4.0x | missing malformed-header property coverage | missing io round-trip differential | missing io journey | missing parser reason-code families | `crates/fnp-io/src/lib.rs` (stub) |

Contradictions/unknowns register (must be closed during doc-overhaul passes):

| ID | Contradiction or unknown | Source/evidence anchor | Owner | Risk | Closure criteria |
|---|---|---|---|---|---|
| `DOC-C001` | `EXISTING_NUMPY_STRUCTURE.md` still states a reduced-scope “V1 extraction boundary” which conflicts with full drop-in parity doctrine. | `EXISTING_NUMPY_STRUCTURE.md` section 6 vs `COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md` sections 0 and 2 | `bd-23m.24.2` | critical | Replace scope-cut language with parity-debt sequencing language and explicit ownership table. |
| `DOC-C002` | Verification/logging implication mapping is now documented in DOC-PASS-03, but multiple packet domains remain missing/deferred. | `EXHAUSTIVE_LEGACY_ANALYSIS.md` DOC-PASS-03.4 + `EXISTING_NUMPY_STRUCTURE.md` DOC-PASS-03.4 | `bd-23m.24.4` | high | Keep pass-03 matrices synchronized with packet execution until unresolved domains are implementation-backed and evidence-complete. |
| `DOC-C003` | Packet ownership crosswalk for unresolved domains is now documented in DOC-PASS-02.4, but executable crate boundaries remain unimplemented for those domains. | `EXHAUSTIVE_LEGACY_ANALYSIS.md` DOC-PASS-02.4 + crate stubs (`fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) | `bd-23m.24.3` | high | Keep crosswalk synchronized with packet execution and close only when packet owners replace stubs with production surfaces and full E/F/G evidence. |
| `DOC-C004` | Structured logging contract was added but not yet threaded into all subsystem extraction sections. | `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md` | `bd-23m.24.10` | medium | Each packet section names required log fields and reason-code taxonomy coverage status. |

## DOC-PASS-01 Full Module/Package Cartography with Ownership and Boundaries

### DOC-PASS-01.1 Workspace ownership map (crate -> behavior contract)

| Crate | Ownership boundary (Rust anchors) | Legacy anchor families | Executable evidence anchors | Boundary contract |
|---|---|---|---|---|
| `fnp-dtype` | `crates/fnp-dtype/src/lib.rs` (`DType`, `promote`, `can_cast_lossless`) | `numpy/_core/src/multiarray/dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json`, `crates/fnp-conformance/src/lib.rs` (`run_dtype_promotion_suite`) | Owns deterministic dtype identity/promotion/cast primitives only; does not own array layout, iteration, or execution kernels. |
| `fnp-ndarray` | `crates/fnp-ndarray/src/lib.rs` (`broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout`) | `numpy/_core/src/multiarray/shape.c`, `shape.h`, stride legality loci | `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `run_shape_stride_suite` | Owns shape/stride legality and contiguous layout calculus; does not own dtype promotion policy or numeric kernels. |
| `fnp-ufunc` | `crates/fnp-ufunc/src/lib.rs` (`UFuncArray`, `BinaryOp`, `elementwise_binary`, `reduce_sum`) | `numpy/_core/src/umath/ufunc_object.c`, reduction/dispatch loop families | `crates/fnp-conformance/src/ufunc_differential.rs`, `ufunc_*` fixtures, `run_ufunc_*` suites | Owns broadcasted elementwise/reduction execution semantics for scoped ops; relies on `fnp-dtype` + `fnp-ndarray` for promotion/layout decisions. |
| `fnp-runtime` | `crates/fnp-runtime/src/lib.rs` (`decide_compatibility*`, `evaluate_policy_override`, `EvidenceLedger`) | Runtime policy doctrine from strict/hardened contract set (fail-closed matrix) | `runtime_policy_cases.json`, `runtime_policy_adversarial_cases.json`, workflow scenario suite | Owns strict/hardened compatibility decisions, fail-closed wire decoding, and policy/audit evidence records; does not own numerical semantics. |
| `fnp-conformance` | `crates/fnp-conformance/src/lib.rs` + submodules (`contract_schema`, `security_contracts`, `test_contracts`, `workflow_scenarios`, `benchmark`, `raptorq_artifacts`, `ufunc_differential`) | Oracle/tests from `legacy_numpy_code/numpy`, packet artifact topology contracts | `run_all_core_suites`, contract/test/security/workflow suites, RaptorQ sidecar/scrub/decode proof tooling | Owns verification and evidence production/validation pipeline; must not redefine core semantics owned by execution crates. |
| `fnp-iter` | `crates/fnp-iter/src/lib.rs` (template placeholder) | `numpy/_core/src/multiarray/nditer*` | Packet-A docs in `artifacts/phase2c/FNP-P2C-006/*` | Planned owner for NDIter traversal invariants; executable ownership is not yet implemented (parity debt). |
| `fnp-random` | `crates/fnp-random/src/lib.rs` (template placeholder) | `numpy/random/*.pyx`, `numpy/random/src/*` | none yet beyond crate stub | Planned owner for deterministic RNG streams/state schemas; currently unimplemented parity debt. |
| `fnp-linalg` | `crates/fnp-linalg/src/lib.rs` (template placeholder) | `numpy/linalg/lapack_lite/*` | none yet beyond crate stub | Planned owner for linear algebra adapter contracts; currently unimplemented parity debt. |
| `fnp-io` | `crates/fnp-io/src/lib.rs` (template placeholder) | `numpy/lib/format.py`, npy/npz parser/writer paths | none yet beyond crate stub | Planned owner for NPY/NPZ parsing/writing hardening; currently unimplemented parity debt. |

### DOC-PASS-01.2 Dependency direction and layering constraints

Workspace dependency graph (from `Cargo.toml` manifests):

1. Foundation leaves: `fnp-dtype`, `fnp-ndarray`, `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`.
2. Execution layer: `fnp-ufunc` -> (`fnp-dtype`, `fnp-ndarray`).
3. Policy layer: `fnp-runtime` (optional feature links: `asupersync`, `ftui`).
4. Verification layer: `fnp-conformance` -> (`fnp-dtype`, `fnp-ndarray`, `fnp-ufunc`, `fnp-runtime`) + evidence dependencies (`serde*`, `sha2`, `base64`, `asupersync`).

Layering law (non-negotiable for this pass):

- Core semantics flow upward: dtype/shape -> execution -> runtime policy -> conformance/evidence.
- Verification modules may call execution/policy crates, but execution crates must not depend on conformance crates.
- Stub crates (`fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) are reserved ownership slots; promotion to active owners requires packet A->I evidence artifacts.

Potentially invalid edges to reject:

| Forbidden edge | Why forbidden | Detection/closure |
|---|---|---|
| `fnp-dtype` -> `fnp-ufunc`/`fnp-conformance` | Inverts semantic hierarchy and couples primitives to higher layers | Enforce via manifest review in packet D-stage and contract-schema checks. |
| `fnp-ndarray` -> `fnp-runtime` | Layout legality must remain policy-neutral | Reject in implementation plans (`artifacts/phase2c/*/implementation_plan.md`). |
| Any crate -> `fnp-conformance` (except tests/tools) | Would mix production semantics with oracle/harness code | Gate via Cargo review + `cargo check --workspace --all-targets`. |

### DOC-PASS-01.3 Module hot paths, adapters, and compatibility pockets

| Path | Role type | Why it is a topology hotspot |
|---|---|---|
| `crates/fnp-ndarray/src/lib.rs` (`broadcast_shape`, `fix_unknown_dimension`, `contiguous_strides`) | Hot path | Central SCE legality kernel; every broadcast/reshape legality decision funnels here. |
| `crates/fnp-ufunc/src/lib.rs` (`UFuncArray::elementwise_binary`, `reduce_sum`) | Hot path | Numeric execution path that combines broadcast mapping and reduction index traversal. |
| `crates/fnp-runtime/src/lib.rs` (`decide_compatibility_from_wire`, `decide_and_record_with_context`) | Policy adapter | Converts wire metadata into strict/hardened decisions and audit ledger events. |
| `crates/fnp-conformance/src/ufunc_differential.rs` (`PY_CAPTURE_SCRIPT`, `capture_numpy_oracle`, `compare_against_oracle`) | Oracle adapter/shim | Bridges Rust execution to legacy/system NumPy (or pure fallback) for parity comparisons. |
| `crates/fnp-conformance/src/contract_schema.rs` | Artifact contract validator | Enforces packet artifact topology (`phase2c-contract-v1`) before packet promotion. |
| `crates/fnp-conformance/src/workflow_scenarios.rs` | Integration shim | Links fixture-level checks to E2E scenario corpus and replay logging expectations. |
| `crates/fnp-conformance/src/raptorq_artifacts.rs` | Durability adapter | Generates/validates sidecars, scrub reports, and decode proofs for long-lived evidence bundles. |

### DOC-PASS-01.4 Verification implications by ownership boundary (covered/missing/deferred)

| Ownership boundary | Unit/Property | Differential | E2E | Structured logging | Status + owner |
|---|---|---|---|---|---|
| `fnp-ndarray` shape/stride legality | covered (crate tests + shape/stride fixtures) | covered (indirect via ufunc differential shape checks), still partial | deferred to packet scenario layers | partial (runtime/workflow logs exist; shape-specific reason taxonomy incomplete) | partial; packet owner `bd-23m.12` |
| `fnp-dtype` promotion/cast primitives | covered (crate tests + promotion fixture suite) | missing full cast-matrix differential | missing packet-level journey | missing per-cast reason-code taxonomy | missing/partial; packet owner `bd-23m.13` |
| `fnp-ufunc` scoped ops (`add/sub/mul/div/sum`) | covered (crate tests + metamorphic/adversarial suites) | covered for scoped fixtures; broader NumPy surface missing | deferred (workflow corpus links exist, packet-local journeys incomplete) | covered at fixture/scenario level | partial; packet owner `bd-23m.17` + `bd-23m.19` |
| `fnp-runtime` strict/hardened policy | covered (crate tests + policy suites) | not applicable as numerical diff; policy-wire adversarial coverage is covered | covered via workflow scenario suite | covered (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) | covered for current surface; continuing under foundation beads `bd-23m.6`, `bd-23m.23` |
| `fnp-iter` NDIter traversal ownership | missing | missing | missing | missing | missing; packet owner `bd-23m.17` |
| `fnp-random` deterministic stream/state | missing | missing | missing | missing | missing; packet owner `bd-23m.18` |
| `fnp-linalg` solver contracts | missing | missing | missing | missing | missing; packet owner `bd-23m.19` |
| `fnp-io` npy/npz contracts | missing | missing | missing | missing | missing; packet owner `bd-23m.20` |

### DOC-PASS-01.5 Contradictions and unknowns register (topology-specific)

| ID | Contradiction / unknown | Evidence anchors | Risk | Owner | Closure criteria |
|---|---|---|---|---|---|
| `TOPO-C001` | Four ownership crates (`fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) are still template stubs, so topology ownership exists but implementation boundaries are not executable yet. | `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | critical | `bd-23m.17`, `bd-23m.18`, `bd-23m.19`, `bd-23m.20` | Replace template modules with packet D-stage boundaries + passing unit/property suites and contract artifacts. |
| `TOPO-C002` | Differential harness is currently concentrated in ufunc/policy surfaces; packet-specific differential lanes for transfer, RNG, linalg, and IO are not wired. | `crates/fnp-conformance/src/lib.rs`, fixture inventory under `crates/fnp-conformance/fixtures` | high | packet F-stage beads (`bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`) | Add packet-local fixture manifests, oracle capture paths, and differential report outputs with reason codes. |
| `TOPO-C003` | Layering constraints are documented but not machine-enforced as a workspace contract test. | Root `Cargo.toml` + crate manifests | medium | foundation gate `bd-23m.23` | Add automated dependency-direction check to reliability gate outputs and fail CI on forbidden edges. |
| `TOPO-C004` | Runtime/workflow structured logs are optional via path configuration, allowing silent no-op when env/config is absent. | `set_runtime_policy_log_path`, `set_workflow_scenario_log_path`, `maybe_append_runtime_policy_log` | medium | foundation orchestration `bd-23m.6` | Gate runs must set explicit log paths and fail when required log artifacts are absent. |

## DOC-PASS-02 Symbol/API Census and Surface Classification

### DOC-PASS-02.1 Crate-level public symbol census

| Crate | Public symbol families (Rust anchors) | Classification | Legacy correspondence |
|---|---|---|---|
| `fnp-dtype` | `DType` enum; `promote`; `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | Compatibility-kernel API (Tier A) | `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` promotion/cast families |
| `fnp-ndarray` | `MemoryOrder`, `ShapeError`, `broadcast_*`, `element_count`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout` (`crates/fnp-ndarray/src/lib.rs`) | Compatibility-kernel API (Tier A) | `shape.c`, `shape.h`, stride legality/broadcast shape semantics |
| `fnp-ufunc` | `BinaryOp`, `UFuncArray` constructors/accessors/ops, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | Execution API (Tier A) | `umath/ufunc_object.c` dispatch/reduction semantics |
| `fnp-runtime` | mode/class/action enums; decision and ledger APIs; override gate; posterior/expected-loss helpers (`crates/fnp-runtime/src/lib.rs`) | Policy/audit API (Tier A) | Strict/hardened compatibility doctrine and fail-closed runtime matrix |
| `fnp-conformance` | `HarnessConfig`, `SuiteReport`, all `run_*` suites + module exports in `benchmark`, `contract_schema`, `raptorq_artifacts`, `security_contracts`, `test_contracts`, `ufunc_differential`, `workflow_scenarios` | Verification/tooling API (Tier B) | Legacy oracle capture/comparison and packet artifact contract validation |
| `fnp-iter` | `add(left, right)` placeholder (`crates/fnp-iter/src/lib.rs`) | Reserved ownership slot (Tier C) | Intended NDIter mapping not yet implemented |
| `fnp-random` | `add(left, right)` placeholder (`crates/fnp-random/src/lib.rs`) | Reserved ownership slot (Tier C) | Intended RNG mapping not yet implemented |
| `fnp-linalg` | `add(left, right)` placeholder (`crates/fnp-linalg/src/lib.rs`) | Reserved ownership slot (Tier C) | Intended linalg mapping not yet implemented |
| `fnp-io` | `add(left, right)` placeholder (`crates/fnp-io/src/lib.rs`) | Reserved ownership slot (Tier C) | Intended NPY/NPZ mapping not yet implemented |

### DOC-PASS-02.2 Call/usage context graph (operator paths -> API surfaces)

| Entry path | Immediate API calls | Downstream semantic surfaces |
|---|---|---|
| `capture_numpy_oracle` bin | `ufunc_differential::capture_numpy_oracle` | Legacy/system NumPy capture into fixture oracle JSON |
| `run_ufunc_differential` bin | `compare_against_oracle`, `write_differential_report` | `execute_input_case` -> `UFuncArray::{new,elementwise_binary,reduce_sum}` + dtype parsing |
| `run_security_gate` bin | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts::run_security_contract_suite` | `decide_and_record_with_context`, `decide_compatibility_from_wire`, runtime ledger/log contract checks |
| `run_test_contract_gate` bin | `test_contracts::run_test_contract_suite` + runtime suites + log schema validation | fixture-shape/metadata contracts and structured logging field enforcement |
| `run_workflow_scenario_gate` bin | `run_user_workflow_scenario_suite` | mixed pipeline: ufunc execution + runtime policy + scenario log emission (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) |
| `generate_benchmark_baseline` bin | `benchmark::generate_benchmark_baseline` | controlled workload timing and baseline artifact emission |
| `generate_raptorq_sidecars` bin | `raptorq_artifacts::generate_bundle_sidecar_and_reports` | sidecar generation, scrub report, decode-proof workflow |
| `validate_phase2c_packet` bin | `contract_schema::validate_phase2c_packet`, `write_packet_readiness_report` | packet artifact completeness and schema/token checks |

### DOC-PASS-02.3 Stability and user-visibility classification

| Tier | Definition | Current symbol groups | Risk if treated as stable too early |
|---|---|---|---|
| Tier A | Implemented compatibility-kernel/public semantics used by harness execution | `fnp-dtype`, `fnp-ndarray`, `fnp-ufunc`, `fnp-runtime` public APIs | Medium: behavior is first-wave complete, not full-NumPy complete; parity drift remains possible outside scoped fixtures. |
| Tier B | Operational tooling APIs for CI/gates/artifact lifecycle | `fnp-conformance` public modules and binaries | Medium: interfaces are workflow-stable but still evolving with packet schema/logging requirements. |
| Tier C | Placeholder ownership surfaces only | `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` `add()` stubs | Critical: presenting as production APIs would create false parity claims and hide major debt. |

### DOC-PASS-02.4 Packet ownership index for unresolved behavior domains (`DOC-C003`)

| Behavior domain | Placeholder surface now | Packet owner | Required closure gates (minimum) | Evidence anchors |
|---|---|---|---|---|
| NDIter traversal/index semantics | `fnp-iter::add` only | `bd-23m.17` / `FNP-P2C-006` | Replace stub with traversal API boundaries; close E/F/G lanes (`bd-23m.17.5`, `bd-23m.17.6`, `bd-23m.17.7`) + strict/hardened threat closure (`bd-23m.17.3`) | `artifacts/phase2c/FNP-P2C-006/*`, future packet readiness report |
| RNG streams + state schema | `fnp-random::add` only | `bd-23m.18` / `FNP-P2C-007` | Implement deterministic RNG/state APIs and close packet E/F/G + logging coverage | `artifacts/phase2c/FNP-P2C-007/*` |
| Linalg adapters/solver contracts | `fnp-linalg::add` only | `bd-23m.19` / `FNP-P2C-008` | Replace stub with scoped linalg API and close packet E/F/G + performance/isomorphism reports | `artifacts/phase2c/FNP-P2C-008/*` |
| NPY/NPZ parser-writer contracts | `fnp-io::add` only | `bd-23m.20` / `FNP-P2C-009` | Replace stub with parser/writer boundary APIs; close packet E/F/G + security/logging + sidecar proof requirements | `artifacts/phase2c/FNP-P2C-009/*` |

### DOC-PASS-02.5 Verification implications by symbol family

| Symbol family | Unit/Property | Differential | E2E | Structured logging | Assessment |
|---|---|---|---|---|---|
| Shape/stride symbols (`fnp-ndarray`) | covered (crate tests + `shape_stride_cases.json`) | partial through ufunc differential shape validation | deferred packet-local scenarios | partial | parity debt remains for richer alias/transfer semantics |
| Dtype symbols (`fnp-dtype`) | covered for promotion/cast primitives | partial (promotion diff exists, full cast matrix absent) | deferred | missing cast taxonomy | medium risk |
| Ufunc symbols (`fnp-ufunc`) | covered for scoped ops | covered for scoped fixture corpus | partial via workflow suite | covered at fixture/scenario level | medium risk |
| Runtime policy symbols (`fnp-runtime`) | covered (unit + policy fixtures) | adversarial wire-class differential equivalent covered | covered in workflow scenarios | covered with required fields | low-medium risk |
| Verification/tooling symbols (`fnp-conformance`) | covered by module tests and gate execution | N/A | covered by gate binaries | covered by contract suites | medium operational risk (schema evolution) |
| Placeholder symbols (`fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`) | missing | missing | missing | missing | critical parity debt |

### DOC-PASS-02.6 Symbol-layer contradictions and unknowns

| ID | Contradiction/unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `SYM-C001` | Tier-C placeholder crates still export callable public functions (`add`) that could be misconstrued as parity surfaces. | high | `bd-23m.17`/`18`/`19`/`20` | Replace stubs with explicit packet-scoped APIs or mark crate surfaces as intentionally non-parity in crate docs/tests. |
| `SYM-C002` | Runtime and workflow log appenders are path-config driven; absent config can silently skip log emission. | medium | `bd-23m.6` | Require explicit log path in orchestrator/gates and fail when expected logs are absent. |
| `SYM-C003` | Differential harness is strongest for ufunc/runtime but weaker for cast matrix, transfer/alias, RNG, linalg, and IO. | high | packet F-stage beads (`bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`) | Add packet-specific fixture and oracle diff lanes with failure taxonomy and reproducible artifacts. |

## DOC-PASS-03 Data Model, State, and Invariant Mapping

### DOC-PASS-03.1 Canonical data model map (state-bearing types)

| Model family | Concrete types (anchors) | State schema | Ownership |
|---|---|---|---|
| Dtype identity/promotion | `DType` + `promote` + `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | finite scalar domain + deterministic promotion/cast relation | `fnp-dtype` |
| Shape/stride legality | `MemoryOrder`, `ShapeError`, `NdLayout` + legality functions (`crates/fnp-ndarray/src/lib.rs`) | shape vectors, stride vectors, item-size arithmetic, legality error taxonomy | `fnp-ndarray` |
| Numeric execution payload | `UFuncArray`, `BinaryOp`, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | shape/value/dtype tuple with broadcasted binary and reduction transitions | `fnp-ufunc` |
| Runtime policy/audit | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger` (`crates/fnp-runtime/src/lib.rs`) | mode/class/risk/threshold input state -> action + posterior/loss + audit metadata and append-only ledger | `fnp-runtime` |
| Conformance fixtures/reports | `HarnessConfig`, `SuiteReport`, `UFuncInputCase`, `UFuncDifferentialReport`, workflow scenario structs (`crates/fnp-conformance/src/lib.rs`, `ufunc_differential.rs`, `workflow_scenarios.rs`) | corpus-driven execution state, case counters, failure diagnostics | `fnp-conformance` |
| Artifact contract/durability | `PacketReadinessReport`, `MissingField` (`contract_schema.rs`), `RaptorQSidecar`, `ScrubReport`, `DecodeProofArtifact` (`raptorq_artifacts.rs`) | packet readiness state + integrity/durability metadata | `fnp-conformance` |
| Reserved ownership placeholders | `add` stubs in `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` | scalar arithmetic placeholder only; no domain state machine | packet owners `bd-23m.17`/`18`/`19`/`20` |

### DOC-PASS-03.2 State transition ledger (input class -> transition -> terminal)

| Transition class | Entry state | Transition function(s) | Success terminal | Failure terminal |
|---|---|---|---|---|
| Broadcast merge | `(lhs_shape, rhs_shape)` | `broadcast_shape`, `broadcast_shapes` | deterministic merged shape | `ShapeError::IncompatibleBroadcast` |
| Reshape inference | `(new_shape spec, old_count)` | `fix_unknown_dimension` | resolved concrete shape | `MultipleUnknownDimensions` / `InvalidDimension` / `IncompatibleElementCount` / `Overflow` |
| Layout derivation | `(shape, item_size, order)` | `contiguous_strides`, `NdLayout::contiguous` | legal contiguous layout | `InvalidItemSize` / `Overflow` |
| Ufunc binary op | `(lhs UFuncArray, rhs UFuncArray, op)` | `elementwise_binary` | new output `UFuncArray` with promoted dtype | shape or input-length error via `UFuncError` |
| Ufunc reduction | `(UFuncArray, axis, keepdims)` | `reduce_sum` | reduced output `UFuncArray` | `AxisOutOfBounds` / shape-derived errors |
| Runtime policy decision | `(mode,class,risk,threshold)` | `decide_compatibility`, `decide_compatibility_from_wire` | `allow` or `full_validate` or `fail_closed` | unknown mode/class intentionally collapses to `fail_closed` |
| Runtime event capture | `(ledger, decision context)` | `decide_and_record_with_context` | appended `DecisionEvent` | normalized default tokens when context fields are empty |
| Workflow scenario execution | `(scenario fixture, mode)` | `run_user_workflow_scenario_suite`, `execute_mode` | pass/fail status with step-level evidence | deterministic failure records for missing fixtures/scripts or expectation mismatch |
| Packet readiness gate | `(packet artifact dir)` | `validate_phase2c_packet` | ready report | non-ready report with missing artifact/field/parse diagnostics |
| Durability pipeline | `(bundle files)` | `generate_bundle_sidecar_and_reports`, scrub/decode helpers | sidecar + scrub + decode proof | integrity mismatch/decode failure surfaces |

### DOC-PASS-03.3 Mutability boundaries and trust zones

| Zone | Mutable surface | Guard rail | Residual risk |
|---|---|---|---|
| Compatibility kernel data | derived vectors/buffers during computation | constructor + precondition checks (shape length, axis bounds, overflow guards) | public struct fields in some models can be mutated after construction if misused |
| Runtime policy ledger | `EvidenceLedger.events` append path | append-only API (`record`) + context normalization | downstream consumers must treat missing logs as gate failure, not optional telemetry |
| File-backed audit logs | runtime/workflow JSONL append | contract suites validate required fields | if log path not configured outside gates, append may no-op |
| Artifact outputs | readiness reports + RaptorQ artifacts | schema/token/path validation + hash checks | stale artifacts can drift without periodic gate enforcement |
| Placeholder crates | trivial function bodies | explicit documentation + packet ownership mapping | accidental reliance on placeholders as real APIs |

### DOC-PASS-03.4 Invariant obligations with evidence status

| Invariant | Expression | Evidence status | Logging/e2e implication |
|---|---|---|---|
| `INV-SHAPE-001` | element counts and stride arithmetic never overflow into unchecked state | covered by `fnp-ndarray` unit tests + shape fixtures | errors should map to stable reason taxonomy in packet logs (partial today) |
| `INV-SHAPE-002` | broadcast legality is deterministic and fail-fast on incompatible dimensions | covered for current shape corpus | dedicated broadcast-focused e2e journey still deferred |
| `INV-RESHAPE-001` | reshape `-1` inference is unique and count-preserving | covered by unit tests | packet-local reason-code coverage pending |
| `INV-UFUNC-001` | ufunc construction enforces value length == shape element count | covered by unit tests + differential path | workflow logs capture case/step context for failures |
| `INV-UFUNC-002` | reduction axis must be in bounds and output shape semantics respect `keepdims` | covered by unit tests + adversarial fixtures | scenario logs include pass/fail + detail per step |
| `INV-POLICY-001` | unknown/incompatible wire semantics must fail closed | covered (runtime adversarial suite + workflow scenarios) | required structured log fields are enforced by test/security contract gates |
| `INV-AUDIT-001` | decision events always carry non-empty audit identity fields | covered via context normalization + log validators | missing field is a contract/gate failure |
| `INV-CONTRACT-001` | packet readiness requires mandatory files/tokens/paths | covered by `validate_phase2c_packet` | readiness reports are explicit triage artifacts |
| `INV-DURABILITY-001` | sidecar/scrub/decode artifacts preserve payload integrity | covered for generated bundles | decode proof artifacts are mandatory for durable claim |
| `INV-OWNERSHIP-001` | unresolved domains must expose real packet-scoped models before parity claims | missing (stubs remain) | packet owners must add unit/differential/e2e/logging trails before closure |

### DOC-PASS-03.5 Data-model contradictions and closure backlog

| ID | Contradiction/unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `MODEL-C001` | `NdLayout` exposes public fields, so post-construction mutation can violate assumptions outside constructor checks. | medium | `bd-23m.12` | Decide encapsulation policy (public-by-contract vs encapsulated mutators) and enforce consistently. |
| `MODEL-C002` | Runtime/workflow logging optionality can weaken forensic guarantees when not running gates. | medium | `bd-23m.6` | Require explicit log path and gate-level assertion of produced log artifacts in orchestrator flows. |
| `MODEL-C003` | Placeholder crates do not implement their promised domain state machines/invariants. | critical | `bd-23m.17`/`18`/`19`/`20` | Replace stubs with packet-scoped state models + invariant tests + differential/e2e evidence. |
| `MODEL-C004` | Cast-matrix and transfer/alias invariants are not yet represented as first-class state contracts in code. | high | `bd-23m.13`/`bd-23m.14` | Add explicit state/invariant tables and executable fixtures for cast matrix and overlap/alias transitions. |

## 2. Quantitative Legacy Inventory (Measured)

- Total files: `2326`
- Python: `493`
- Native: `c=180`, `cpp=29`, `h=227`, `hpp=16`
- Cython: `pyx=13`, `pxd=6`
- Test-like files: `530`

High-density zones:
- `numpy/_core/src` (402 files)
- `numpy/typing/tests` (149)
- `numpy/f2py/tests` (132)
- `numpy/_core/tests` (100)
- `numpy/linalg/lapack_lite` (26)
- `numpy/random/src` (37)

## 3. Subsystem Extraction Matrix (Legacy -> Rust)

| Legacy locus | Non-negotiable behavior to preserve | Target crates | Primary oracles | Phase-2 extraction deliverables |
|---|---|---|---|---|
| `numpy/_core/src/multiarray/shape.c` + `shape.h` | shape legality, stride legality, reshape semantics | `fnp-ndarray` | `_core/tests/test_shape_base.py`, `test_nditer.py` | shape/stride state machine, invalid-transition matrix |
| `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | promotion and cast determinism | `fnp-dtype` | `_core/tests/test_dtype.py`, `test_casting*` | promotion matrix, cast allow/deny table |
| `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | assignment loop correctness under strides | `fnp-iter`, `fnp-ufunc` | `_core/tests/test_mem_overlap.py` | alias/overlap safety matrix, loop invariants |
| `umath/ufunc_object.c`, dispatch loop sources | ufunc dispatch and broadcasting precedence | `fnp-ufunc` | `_core/tests/test_ufunc.py`, `test_overrides.py` | dispatch precedence graph, override semantics spec |
| `multiarray/nditer*` | iterator consistency across order/layout | `fnp-iter` | `_core/tests/test_nditer.py` | traversal equivalence ledger |
| `numpy/random/*.pyx` + `random/src/*` | RNG determinism and state serialization | `fnp-random` | `random/tests/*` | RNG stream fixture corpus + state schema |
| `numpy/linalg/lapack_lite/*` | solver shape/tolerance/return semantics | `fnp-linalg` | `linalg/tests/test_linalg.py` | BLAS/LAPACK boundary contract table |
| `numpy/lib/stride_tricks.py` + IO paths | view tricks and npy/npz metadata semantics | `fnp-ndarray`, `fnp-io` | `lib/tests/test_stride_tricks.py`, IO tests | metadata parser/state machine spec |

## 4. Alien-Artifact Invariant Ledger (Formal Obligations)

- `FNP-I1` Shape legality closure: all produced array states satisfy dimension/stride legality.
- `FNP-I2` Promotion determinism: scoped dtype pairs map to a single canonical result dtype.
- `FNP-I3` Broadcast determinism: result shape is deterministic and independent of iteration schedule.
- `FNP-I4` Alias safety: write operations do not violate overlap semantics.
- `FNP-I5` RNG reproducibility: fixed seed and algorithm produce stable output stream.

Required proof artifacts per implemented slice:
1. invariant statement,
2. executable witness fixtures,
3. failing counterexample archive (if any),
4. remediation proof.

## 5. Native/C/Fortran Boundary Register

| Boundary | Files | Risk | Mandatory mitigation |
|---|---|---|---|
| ndarray metadata core | `multiarray/*.c` | critical | legality checks first, then loop execution |
| ufunc dispatch core | `umath/ufunc_object.c` | high | precedence fixtures + override differential tests |
| lapack-lite bridge | `linalg/lapack_lite/*` | high | solver IO/tolerance conformance fixtures |
| RNG backend | `random/src/*`, `random/*.pyx` | high | deterministic stream and state round-trip checks |

## 6. Compatibility and Security Doctrine (Mode-Split)

Decision law (runtime):
`mode + metadata_contract + risk_score + budget -> allow | full_validate | fail_closed`

| Threat | Strict mode | Hardened mode | Required ledger artifact |
|---|---|---|---|
| malformed npy/npz metadata | fail-closed | fail-closed with bounded diagnostics | parser incident ledger |
| shape bomb payload | execute in scoped limits | tighter admission caps | admission guard report |
| unsafe cast attempt | reject unscoped cast | allow only policy-allowlisted cast | cast decision ledger |
| stride alias confusion | reject invalid transitions | reject invalid transitions | stride invariant report |
| unknown incompatible metadata version | fail-closed | fail-closed | compatibility drift report |

## 7. Conformance Program (Exhaustive First Wave)

### 7.1 Fixture families

1. Shape/reshape fixtures
2. Stride/view and overlap fixtures
3. Broadcast fixtures
4. Ufunc arithmetic/reduction fixtures
5. Dtype promotion fixtures
6. RNG determinism fixtures
7. NPY/NPZ IO fixtures

### 7.2 Differential harness outputs (fnp-conformance)

Each run emits:
- machine-readable parity report,
- mismatch class histogram,
- minimized repro fixture bundle,
- strict/hardened divergence report.

Release gate rule: critical-family drift => hard fail.

## 8. Extreme Optimization Program

Primary hotspots:
- ufunc dispatch path
- non-contiguous iteration loops
- dtype conversion hot loops
- reduction kernels

Budgets (from spec section 17):
- broadcast p95 <= 180 ms
- reductions p95 <= 210 ms
- reshape/view p95 <= 40 ms
- npy parse throughput >= 400 MB/s
- p99 regression <= +7%, peak RSS regression <= +8%

Optimization governance:
1. baseline,
2. profile,
3. single optimization lever,
4. conformance proof,
5. budget gate,
6. evidence commit.

## 9. RaptorQ-Everywhere Artifact Contract

Durable artifacts requiring RaptorQ sidecars:
- conformance fixture bundles,
- benchmark baselines,
- promotion/cast ledgers,
- risk/proof ledgers.

Required envelope fields:
- source hash,
- symbol manifest,
- scrub status,
- decode proof chain.

## 10. Phase-2 Execution Backlog (Concrete)

1. Extract shape/stride legality rules from `shape.c`.
2. Extract promotion/cast tables from dtype sources.
3. Extract overlap/alias rules from low-level loop code.
4. Extract ufunc dispatch precedence and override rules.
5. Extract NDIter traversal invariants.
6. Extract RNG state schema and deterministic stream behavior.
7. Extract linalg solver input/output and tolerance behavior.
8. Extract npy/npz metadata parsing state machine.
9. Build first differential fixture corpus for items 1-8.
10. Implement mismatch taxonomy in `fnp-conformance`.
11. Add strict/hardened divergence reporting.
12. Attach RaptorQ sidecar generation and decode-proof validation.

Definition of done for Phase-2:
- each row in section 3 has extraction artifacts,
- all seven fixture families are runnable,
- G1-G6 gates from comprehensive spec map to concrete harness outputs.

## 11. Residual Gaps and Risks

- `PROPOSED_ARCHITECTURE.md` crate list contains literal `\n` separators; normalize before automation.
- high-risk native boundaries (multiarray/umath/lapack-lite) require wider differential corpus before aggressive optimization.
- RNG and cast semantics are common silent-regression vectors and require explicit release blockers.

## 12. Deep-Pass Hotspot Inventory (Measured)

Measured from `/data/projects/franken_numpy/legacy_numpy_code/numpy`:
- file count: `2326`
- highest concentration: `numpy/_core` (`604` files), `numpy/f2py` (`177`), `numpy/typing` (`152`), `numpy/random` (`101`), `numpy/lib` (`97`)

Top source hotspots by line count (first-wave extraction anchors):
1. `numpy/linalg/lapack_lite/f2c_d_lapack.c` (`41864`)
2. `numpy/linalg/lapack_lite/f2c_s_lapack.c` (`41691`)
3. `numpy/linalg/lapack_lite/f2c_z_lapack.c` (`29996`)
4. `numpy/linalg/lapack_lite/f2c_c_lapack.c` (`29861`)
5. `numpy/linalg/lapack_lite/f2c_blas.c` (`21603`)
6. `numpy/_core/src/umath/ufunc_object.c` (`6796`)

Interpretation:
- low-level shape/ufunc/cast paths must be extracted before high-level wrappers,
- lapack-lite bridge dominates numerical backend complexity,
- iterator/transfer semantics remain highest silent-regression risk.

## 13. Phase-2C Extraction Payload Contract (Per Ticket)

Each `FNP-P2C-*` ticket MUST produce:
1. type inventory (dtype/array/iterator structs and fields),
2. shape-stride legality rules with exact conditions,
3. cast/promotion rule tables (including defaults),
4. error/diagnostic contract map,
5. alias/overlap/memory-order behavior ledger,
6. strict/hardened mode split,
7. explicit exclusion ledger,
8. fixture mapping manifest,
9. optimization candidate notes + isomorphism risk,
10. RaptorQ artifact declarations.

Artifact location (normative):
- `artifacts/phase2c/FNP-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FNP-P2C-00X/contract_table.md`
- `artifacts/phase2c/FNP-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FNP-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FNP-P2C-00X/risk_note.md`

### 13.1 Packet `FNP-P2C-006` extraction status (`A` stage)

Completed in this pass:
- `artifacts/phase2c/FNP-P2C-006/legacy_anchor_map.md`
- `artifacts/phase2c/FNP-P2C-006/behavior_extraction_ledger.md`
- `artifacts/phase2c/FNP-P2C-006/contract_table.md`
- `artifacts/phase2c/FNP-P2C-006/risk_note.md`
- `artifacts/phase2c/FNP-P2C-006/implementation_plan.md`

Anchor families locked for packet-A:
- `numpy/lib/_stride_tricks_impl.py` (`as_strided`, `_broadcast_to`, `broadcast_to`, `_broadcast_shape`, `broadcast_shapes`, `broadcast_arrays`)
- `numpy/_core/src/multiarray/nditer_constr.c` (`npyiter_fill_axisdata`, `broadcast_error`, `operand_different_than_broadcast`)
- `numpy/_core/src/multiarray/nditer_api.c` (`NpyIter_GetShape`, `NpyIter_CreateCompatibleStrides`)
- oracle tests: `numpy/lib/tests/test_stride_tricks.py`, `numpy/_core/tests/test_nditer.py`

Verification hooks registered:
- unit/property expansion target: `bd-23m.17.5`
- differential/metamorphic/adversarial expansion target: `bd-23m.17.6`
- e2e replay/logging target: `bd-23m.17.7`
- strict/hardened threat closure target: `bd-23m.17.3`

Contract-table status:
- strict/hardened invariants, pre/postconditions, failure semantics, and packet reason-code vocabulary are defined in `artifacts/phase2c/FNP-P2C-006/contract_table.md`
- every contract row links planned unit/property IDs, differential IDs, and e2e IDs for packet follow-on stages

Threat-model status:
- hostile input classes, fail-closed envelope, hardened recovery constraints, adversarial fixture lanes, and residual-risk closures are documented in `artifacts/phase2c/FNP-P2C-006/risk_note.md`

Implementation-plan status:
- crate/module boundary skeleton, packet D->I implementation sequence, instrumentation insertion points, and compile-safe validation path are documented in `artifacts/phase2c/FNP-P2C-006/implementation_plan.md`

## 14. Strict/Hardened Compatibility Drift Budgets

Packet acceptance budgets:
- strict mode critical drift budget: `0`
- strict mode non-critical drift budget: `<= 0.10%`
- hardened mode divergence budget: `<= 1.00%` and only allowlisted defensive cases
- unknown dtype/layout/metadata behavior: fail-closed

Required per-packet report fields:
- `strict_parity`,
- `hardened_parity`,
- `divergence_classes`,
- `cast_drift_summary`,
- `compatibility_drift_hash`.

## 15. Extreme-Software-Optimization Execution Law

Mandatory loop:
1. baseline,
2. profile,
3. one lever,
4. conformance + invariant proof,
5. re-baseline.

Primary sentinel workloads:
- reshape/broadcast-heavy traces (`FNP-P2C-001`, `FNP-P2C-006`),
- mixed-dtype cast pipelines (`FNP-P2C-002`, `FNP-P2C-003`),
- ufunc dispatch stress (`FNP-P2C-005`),
- RNG reproducibility workloads (`FNP-P2C-007`).

Optimization scoring gate:
`score = (impact * confidence) / effort`, merge only if `score >= 2.0`.

## 16. RaptorQ Evidence Topology and Recovery Drills

Durable artifacts requiring sidecars:
- parity reports,
- fixture corpora,
- cast/promotion ledgers,
- benchmark baselines,
- strict/hardened divergence logs.

Naming convention:
- payload: `packet_<id>_<artifact>.json`
- sidecar: `packet_<id>_<artifact>.raptorq.json`
- proof: `packet_<id>_<artifact>.decode_proof.json`

Decode-proof failure policy: hard blocker for packet promotion.

## 17. Phase-2C Exit Checklist (Operational)

Phase-2C is complete only when:
1. `FNP-P2C-001..009` artifacts exist and pass schema validation.
2. Every packet has strict and hardened fixture coverage.
3. Drift budgets from section 14 are met.
4. At least one hotspot optimization proof exists for each high-risk packet.
5. RaptorQ sidecars + decode proofs are scrub-clean.
6. Remaining risks are explicit with owners and next actions.
