# EXISTING_NUMPY_STRUCTURE

## DOC-PASS-00 Baseline Gap Matrix + Quantitative Expansion Targets

Snapshot baseline (2026-02-14):

| Document | Baseline lines | Target lines | Expansion multiplier | Status |
|---|---:|---:|---:|---|
| `EXISTING_NUMPY_STRUCTURE.md` | 62 | 992 | 16.0x | baseline pass complete (`bd-23m.24.1` closed) |
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 275 | 3300 | 12.0x | baseline pass complete (`bd-23m.24.1` closed) |

Gap matrix for this document (pass-1 planning):

| Area | Current state | Missing for parity-grade docs | Coverage implications to record |
|---|---|---|---|
| Subsystem map detail | coarse bullets | per-package ownership + boundary invariants + packet IDs | unit/property + differential + e2e + structured logging per subsystem |
| Semantic hotspots | short list | explicit legality formulas and tie-break rules | metamorphic/property families and adversarial classes |
| Compatibility-critical behavior | high-level statements | strict/hardened split with drift budget and fail-closed triggers | runtime-policy reason-code matrix and replay fields |
| Risk areas | broad categories | threat-to-control-to-test mappings | parser fuzz corpus, crash triage IDs, forensic log fields |
| Conformance families | list only | fixture IDs, oracle anchors, and closure gates | coverage ledger with covered/missing/deferred status |

Pass-1 explicit coverage/traceability matrix (covered, missing, deferred):

| Subsystem | Legacy anchors | Executable evidence anchors | Unit/Property | Differential | E2E | Structured logging |
|---|---|---|---|---|---|---|
| Shape/stride | `numpy/_core/src/multiarray/shape.c`, `shape.h` | `crates/fnp-conformance/fixtures/shape_stride_cases.json` | covered (partial) | covered (partial via ufunc diff) | missing | covered (partial via runtime-policy logs) |
| Dtype promotion/cast | `numpy/_core/src/multiarray/dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `crates/fnp-conformance/fixtures/dtype_promotion_cases.json` | covered (partial) | missing (cast-matrix diff missing) | missing | missing |
| Ufunc dispatch | `numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_*` | covered (partial metamorphic/adversarial) | covered (partial) | missing | covered (fixture-level fields) |
| Transfer/alias | `numpy/_core/src/multiarray/dtype_transfer.c`, `lowlevel_strided_loops.c.src` | none yet (legacy only) | missing | missing | missing | missing |
| NDIter | `numpy/_core/src/multiarray/nditer*` | `artifacts/phase2c/FNP-P2C-006/legacy_anchor_map.md`, `artifacts/phase2c/FNP-P2C-006/behavior_extraction_ledger.md`, `artifacts/phase2c/FNP-P2C-006/contract_table.md` | missing | missing | missing | missing |
| Random | `numpy/random/*.pyx`, `numpy/random/src/*` | `crates/fnp-random/src/lib.rs` (stub) | missing | missing | missing | missing |
| Linalg | `numpy/linalg/lapack_lite/*` | `crates/fnp-linalg/src/lib.rs` (stub) | missing | missing | missing | missing |
| IO | `numpy/lib/format.py`, npy/npz handling paths | `crates/fnp-io/src/lib.rs` (stub) | missing | missing | missing | missing |

Traceability anchors:

- Legacy source roots: `/data/projects/franken_numpy/legacy_numpy_code/numpy`
- Machine contracts: `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/phase2c_contract_schema_v1.json`
- Gate artifacts: `crates/fnp-conformance/src/test_contracts.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `scripts/e2e/run_test_contract_gate.sh`

Contradictions/unknowns register (for closure in doc passes 01-10):

| ID | Item | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `DOC-C001` | Section 6 carries historical reduced-scope language incompatible with full drop-in parity doctrine. | critical | `bd-23m.24.2` | Replace with parity-debt sequencing model and explicit owner/blocker table. |
| `DOC-C002` | Verification/logging implications are now mapped in DOC-PASS-03.4, but several packet domains remain explicitly missing/deferred. | high | `bd-23m.24.4` | Keep DOC-PASS-03.4 synchronized with packet implementation progress until missing lanes are closed. |
| `DOC-C003` | Ownership for unresolved NDIter/RNG/linalg/IO extraction is now mapped in DOC-PASS-02.5, but executable crate boundaries are still stubs. | high | `bd-23m.24.3` | Keep packet ownership/gates synchronized with crate implementation state until `bd-23m.17`/`bd-23m.18`/`bd-23m.19`/`bd-23m.20` are closed. |

## DOC-PASS-02 Symbol/API Census and Surface Classification

### DOC-PASS-02.1 Public symbol surface by crate

| Crate | Public symbol/API surface (Rust anchors) | Visibility class | Primary usage contexts |
|---|---|---|---|
| `fnp-dtype` | `DType::{name,item_size,parse}`, `promote`, `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | core semantic contract | Called by `fnp-ufunc` execution (`crates/fnp-ufunc/src/lib.rs`) and conformance promotion suite (`run_dtype_promotion_suite` in `crates/fnp-conformance/src/lib.rs`). |
| `fnp-ndarray` | `MemoryOrder`, `ShapeError`, `can_broadcast`, `broadcast_shape`, `broadcast_shapes`, `element_count`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout::{contiguous,nbytes}` (`crates/fnp-ndarray/src/lib.rs`) | core semantic contract | Used by ufunc execution shape checks and by conformance shape/stride suite (`run_shape_stride_suite`). |
| `fnp-ufunc` | `BinaryOp`, `UFuncArray::{new,scalar,shape,values,dtype,elementwise_binary,reduce_sum}`, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | execution contract | Called from differential/metamorphic/adversarial harness via `execute_input_case` (`crates/fnp-conformance/src/ufunc_differential.rs`). |
| `fnp-runtime` | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionLossModel`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger`, `decide_*`, `evaluate_policy_override`, `posterior_incompatibility`, `expected_loss_for_action` (`crates/fnp-runtime/src/lib.rs`) | policy/audit contract | Used by runtime policy suites and workflow scenario execution in `fnp-conformance`. |
| `fnp-conformance` | `HarnessConfig`, `HarnessReport`, `SuiteReport`, `run_*` suites, plus module exports (`benchmark`, `contract_schema`, `raptorq_artifacts`, `security_contracts`, `test_contracts`, `ufunc_differential`, `workflow_scenarios`) (`crates/fnp-conformance/src/lib.rs`) | verification/tooling contract | Entry point for all gate binaries and packet-readiness tooling. |
| `fnp-iter` | `add(left, right)` placeholder (`crates/fnp-iter/src/lib.rs`) | reserved ownership slot (not parity-usable) | No production call sites; placeholder only. |
| `fnp-random` | `add(left, right)` placeholder (`crates/fnp-random/src/lib.rs`) | reserved ownership slot (not parity-usable) | No production call sites; placeholder only. |
| `fnp-linalg` | `add(left, right)` placeholder (`crates/fnp-linalg/src/lib.rs`) | reserved ownership slot (not parity-usable) | No production call sites; placeholder only. |
| `fnp-io` | `add(left, right)` placeholder (`crates/fnp-io/src/lib.rs`) | reserved ownership slot (not parity-usable) | No production call sites; placeholder only. |

### DOC-PASS-02.2 Operator entry points and call graph anchors

| Operator entry | Calls into | Main outputs/artifacts |
|---|---|---|
| `capture_numpy_oracle` (`crates/fnp-conformance/src/bin/capture_numpy_oracle.rs`) | `ufunc_differential::capture_numpy_oracle` | `crates/fnp-conformance/fixtures/oracle_outputs/ufunc_oracle_output.json` |
| `run_ufunc_differential` (`crates/fnp-conformance/src/bin/run_ufunc_differential.rs`) | `compare_against_oracle`, `write_differential_report` | `crates/fnp-conformance/fixtures/oracle_outputs/ufunc_differential_report.json` |
| `run_security_gate` (`crates/fnp-conformance/src/bin/run_security_gate.rs`) | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts::run_security_contract_suite` | `artifacts/logs/runtime_policy_e2e_*.jsonl` + gate summary JSON |
| `run_test_contract_gate` (`crates/fnp-conformance/src/bin/run_test_contract_gate.rs`) | `test_contracts::run_test_contract_suite` + runtime suites + runtime-log validation | `artifacts/logs/test_contract_e2e_*.jsonl` + gate summary JSON |
| `run_workflow_scenario_gate` (`crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs`) | `workflow_scenarios::run_user_workflow_scenario_suite` | `artifacts/logs/workflow_scenario_e2e_*.jsonl` |
| `generate_benchmark_baseline` (`crates/fnp-conformance/src/bin/generate_benchmark_baseline.rs`) | `benchmark::generate_benchmark_baseline` | `artifacts/baselines/ufunc_benchmark_baseline.json` |
| `generate_raptorq_sidecars` (`crates/fnp-conformance/src/bin/generate_raptorq_sidecars.rs`) | `raptorq_artifacts::generate_bundle_sidecar_and_reports` | `artifacts/raptorq/*.sidecar.json`, `*.scrub_report.json`, `*.decode_proof.json` |
| `validate_phase2c_packet` (`crates/fnp-conformance/src/bin/validate_phase2c_packet.rs`) | `contract_schema::validate_phase2c_packet`, `write_packet_readiness_report` | `artifacts/phase2c/<packet>/packet_readiness_report.json` |

### DOC-PASS-02.3 Stability and user-visibility tiers

| Symbol tier | Included surfaces | Stability assessment | User visibility |
|---|---|---|---|
| Tier A: compatibility-kernel contracts | `fnp-dtype`, `fnp-ndarray`, `fnp-ufunc`, `fnp-runtime` public APIs | medium: implemented and tested for first-wave scope, still parity-incomplete vs full NumPy | direct to internal consumers and conformance harnesses; not yet published as stable external API |
| Tier B: verification/tooling contracts | `fnp-conformance` suite/module public functions and binaries | medium: operationally stable for current packet workflow | primarily developer/CI/operator-facing |
| Tier C: reserved ownership placeholders | `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` `add()` stubs | low: placeholders only; incompatible with parity claims | should be treated as non-contractual internals until packet implementation lands |

### DOC-PASS-02.4 Verification and logging implications by symbol family

| Symbol family | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `fnp-dtype` promotion/cast APIs | covered (crate tests + promotion fixtures) | partial (promotion only; cast-matrix diff missing) | missing | missing dedicated cast reason-code taxonomy | partial |
| `fnp-ndarray` shape/stride APIs | covered (crate tests + shape_stride fixtures) | partial via ufunc differential shape validation | deferred | partial via runtime/workflow logs | partial |
| `fnp-ufunc` execution APIs | covered (crate tests + metamorphic/adversarial fixtures) | covered for scoped ops (`add/sub/mul/div/sum`) | deferred packet scenarios | covered at scenario/log-entry level | partial |
| `fnp-runtime` policy APIs | covered (crate tests + policy suites) | policy-wire adversarial coverage (non-numeric) | covered via workflow scenario suite | covered (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) | covered for current scope |
| `fnp-conformance` contract/sidecar APIs | covered via module tests and gate binaries | N/A | covered through gate binaries | covered through gate log validation | covered/ongoing |
| Stub crate symbols (`add`) | trivial | none | none | none | missing (intentional parity debt) |

### DOC-PASS-02.5 Packet ownership and closure gates for unresolved domains (`DOC-C003`)

| Unresolved behavior family | Current placeholder crate API | Packet owner bead | Required closure gates |
|---|---|---|---|
| NDIter traversal/index semantics | `fnp-iter::add` placeholder only | `bd-23m.17` (`FNP-P2C-006`) | Replace placeholder with iterator contracts + packet D->I artifacts + unit/property (`bd-23m.17.5`), differential/adversarial (`bd-23m.17.6`), e2e/logging (`bd-23m.17.7`). |
| RNG deterministic streams/state schema | `fnp-random::add` placeholder only | `bd-23m.18` (`FNP-P2C-007`) | Implement deterministic RNG APIs and state schemas, then close packet E/F/G evidence lanes and structured logging coverage. |
| Linalg adapter semantics | `fnp-linalg::add` placeholder only | `bd-23m.19` (`FNP-P2C-008`) | Implement linalg boundary contracts + packet evidence chain (unit/property, differential, e2e, sidecar proof). |
| NPY/NPZ parser/writer hardening | `fnp-io::add` placeholder only | `bd-23m.20` (`FNP-P2C-009`) | Implement parser/writer APIs with hardened boundary checks + packet E/F/G evidence + durability artifacts. |

## DOC-PASS-03 Data Model, State, and Invariant Mapping

### DOC-PASS-03.1 Canonical data models (state-carrying structs/enums)

| Model | Rust anchor | State carried | Mutability boundary |
|---|---|---|---|
| DType taxonomy | `fnp_dtype::DType` (`crates/fnp-dtype/src/lib.rs`) | discrete scalar domain (`Bool`, `I32`, `I64`, `F32`, `F64`) | immutable value enum; transitions only via pure functions (`parse`, `promote`, cast checks). |
| Shape/stride layout model | `fnp_ndarray::NdLayout` + `MemoryOrder` + `ShapeError` (`crates/fnp-ndarray/src/lib.rs`) | `shape`, `strides`, `item_size`, legality error class | constructor enforces legality (`NdLayout::contiguous`); consumers can read fields but validity hinges on checked constructors/functions. |
| Ufunc execution payload | `fnp_ufunc::UFuncArray` + `BinaryOp` + `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | logical shape, value buffer, dtype identity | creation gated by `UFuncArray::new`; operations produce new arrays (non-in-place state transitions). |
| Runtime policy event model | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger` (`crates/fnp-runtime/src/lib.rs`) | decision inputs, posterior/expected-loss outputs, audit metadata, append-only event sequence | state mutation isolated to `EvidenceLedger::record` and audit-context normalization. |
| Conformance fixture/report models | `HarnessConfig`, `SuiteReport`, `UFunc*Case`/`UFuncDifferentialReport` (`crates/fnp-conformance/src/lib.rs`, `ufunc_differential.rs`) | fixture inputs, pass/fail counters, mismatch diagnostics | suites are functional over fixture files; report structs are produced as outputs. |
| Artifact durability/contract models | `PacketReadinessReport`, `RaptorQSidecar`, `ScrubReport`, `DecodeProofArtifact` (`contract_schema.rs`, `raptorq_artifacts.rs`) | packet completeness state, symbol/hash metadata, scrub/decode status | generated artifacts are immutable outputs of validator/generator pipelines. |

### DOC-PASS-03.2 State transitions and lifecycle edges

| Transition | Source state | Operation edge | Target state | Fail-closed/error path |
|---|---|---|---|---|
| reshape inference | `(new_shape spec, old element count)` | `fix_unknown_dimension` | resolved `Vec<usize>` | `MultipleUnknownDimensions`, `InvalidDimension`, `IncompatibleElementCount`, `Overflow` |
| broadcast legality | `(lhs shape, rhs shape)` | `broadcast_shape` / `broadcast_shapes` | merged output shape | `ShapeError::IncompatibleBroadcast` |
| contiguous layout derivation | `(shape, item_size, order)` | `contiguous_strides` / `NdLayout::contiguous` | legal stride vector/layout | `InvalidItemSize`, `Overflow` |
| ufunc binary execution | `(lhs UFuncArray, rhs UFuncArray, BinaryOp)` | `elementwise_binary` | new `UFuncArray` output | shape error or input-length rejection via `UFuncError` |
| ufunc reduction | `(UFuncArray, axis, keepdims)` | `reduce_sum` | reduced-shape `UFuncArray` | axis bounds and shape-derived failures (`AxisOutOfBounds`, shape errors) |
| runtime policy decision | `(mode,class,risk,threshold)` | `decide_compatibility*` | `DecisionAction` | unknown wire mode/class => `FailClosed` |
| runtime audit append | `(ledger, decision context)` | `decide_and_record_with_context` | appended `DecisionEvent` | context normalization to non-empty defaults for required audit fields |
| workflow scenario replay | `(scenario corpus + fixtures + mode)` | `run_user_workflow_scenario_suite` | per-scenario pass/fail report + JSONL entries | missing fixtures/scripts/required fields produce deterministic failure records |
| packet readiness validation | `(packet dir artifacts)` | `validate_phase2c_packet` | `PacketReadinessReport` | missing files/fields parse errors force non-ready status |

### DOC-PASS-03.3 Mutability boundaries and write surfaces

| Boundary | Writable state | Protection mechanism |
|---|---|---|
| Numeric core objects (`DType`, `UFuncArray` outputs) | only through constructors/ops that enforce invariants | pure/persistent-style API: operations return new values instead of mutating shared global state |
| Layout legality (`fnp-ndarray`) | derived vectors (`shape`, `strides`) | overflow and legality checks before any layout object is returned |
| Runtime policy ledger | append-only event vector (`EvidenceLedger`) | controlled via `record`; audit context normalized to required default tokens |
| Gate/workflow logs | filesystem JSONL append (`runtime_policy` and `workflow_scenarios`) | explicit schema field checks in contract suites and gate binaries |
| Artifact contracts | packet readiness and sidecar outputs | schema/token/path validation and hash-based scrub/decode proof generation |

### DOC-PASS-03.4 Invariant obligations and verification/logging implications

| Invariant family | Contract expression | Evidence (covered/missing/deferred) | Logging implication |
|---|---|---|---|
| Shape arithmetic safety | `element_count` and stride derivation must not overflow | covered (unit + shape fixtures) | failures should carry stable reason taxonomy in packet logs (partial) |
| Broadcast determinism | broadcast merge independent of operand order alignment rules | covered (unit + shape fixtures), broader high-arity differentials deferred | scenario logs carry fixture IDs but no dedicated broadcast reason-code matrix yet |
| Reshape `-1` legality | at most one unknown dim and exact element-count preservation | covered (unit tests) | error family present; packet-level reason-code mapping still partial |
| Ufunc input/output consistency | input length == element count, axis bounds legal, dtype promotion deterministic | covered for scoped ops (unit + metamorphic/adversarial + differential) | workflow logs include required fields and per-step pass/fail detail |
| Runtime fail-closed doctrine | unknown/incompatible wire semantics must map to `fail_closed` | covered (policy/adversarial suites + workflow scenarios) | explicitly logged via required fields (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) |
| Packet artifact completeness | required files/tokens/JSON/YAML paths must be present | covered by contract schema validator | readiness report captures missing artifacts/fields/parse errors |
| Durability integrity | sidecar symbols/hash scrub/decode proof must be coherent | covered for generated bundles | scrub/decode artifacts become mandatory audit trail pointers |
| Unresolved domain executability | NDIter/RNG/linalg/IO crates must expose real domain APIs, not stubs | missing (currently placeholders) | requires packet-local reason-code vocab once implemented |

### DOC-PASS-03.5 Data/invariant contradictions and unknowns

| ID | Contradiction or unknown | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `STATE-C001` | `NdLayout` fields are public, so post-construction mutation can bypass constructor-time legality assumptions. | medium | `bd-23m.12` | Decide/encode whether layout fields remain public by design or become encapsulated with invariant-preserving mutators only. |
| `STATE-C002` | Runtime/workflow logging can no-op when no path is configured, weakening forensic guarantees outside gate runs. | medium | `bd-23m.6` | Require explicit log paths in orchestrator contexts and fail on missing required logs. |
| `STATE-C003` | Tier-C placeholder crates do not yet encode their promised domain state machines. | critical | `bd-23m.17`/`bd-23m.18`/`bd-23m.19`/`bd-23m.20` | Replace stubs with packet-scoped data models + invariant tests + differential/e2e/logging artifacts. |

## 1. Legacy Oracle

- Root: /dp/franken_numpy/legacy_numpy_code/numpy
- Upstream: numpy/numpy

## 2. Subsystem Map

- numpy/_core/src/multiarray: ndarray construction, shape/stride logic, assignment, nditer, text parsing.
- numpy/_core/src/umath: ufunc machinery and reduction kernels.
- numpy/_core/include/numpy: public ndarray/dtype APIs and ABI contracts.
- numpy/_core/src/_simd: CPU feature detection and SIMD dispatch checks.
- numpy/random and random/src: BitGenerator implementations and distribution paths.
- numpy/lib, numpy/linalg, numpy/fft, numpy/matrixlib, numpy/ma: higher-level Python semantics.
- numpy/tests and package-specific test folders: regression and parity baseline.

## 3. Semantic Hotspots (Must Preserve)

1. shape.c/shape.h dimensional and broadcast legality.
2. descriptor/dtypemeta and casting tables.
3. lowlevel_strided_loops and dtype_transfer assignment semantics.
4. nditer behavior across memory order/stride/layout combinations.
5. stride_tricks behavior for views and broadcasted representations.
6. ufunc override and dispatch selection semantics.
7. dtype promotion matrix determinism.

## 4. Compatibility-Critical Behaviors

- Array API and array-function overrides.
- Public PyArrayObject layout expectations for downstream interop.
- Runtime SIMD capability detection affecting chosen kernels.
- ufunc reduction vs elementwise override precedence.

## 5. Security and Stability Risk Areas

- textreading parser paths and malformed input handling.
- stringdtype conversion routines and buffer bounds.
- stride arithmetic UB risk in low-level loops.
- random generator state determinism and serialization paths.
- external data and dlpack interoperability boundaries.

## 6. Historical V1 Extraction Boundary (Deprecated; convert to parity-debt ledger)

Include now:
- ndarray/dtype core, ufunc/reduction core, promotion/casting contracts, basic npy/npz and RNG parity.

Exclude for V1:
- f2py, packaging/docs/tooling stacks, non-critical module breadth.

## 7. High-Value Conformance Fixture Families

- _core/tests for dtype, array interface, overlap, simd, string dtype.
- random/tests for deterministic generator streams.
- lib/linalg/fft/polynomial/matrixlib/ma tests for scoped behavioral parity.
- tests/test_public_api and testing/tests for API surface stability.

## 8. Extraction Notes for Rust Spec

- Start with shape/stride/dtype model before any heavy optimization.
- Keep promotion matrix and casting behavior explicit and versioned.
- Use parity-by-op-family reports as release gates.

## 9. Packet `FNP-P2C-006` Legacy Anchor + Behavior Ledger (A-stage)

Packet focus: stride-tricks and broadcasting API.

Packet-B output status: `artifacts/phase2c/FNP-P2C-006/contract_table.md` now defines strict/hardened invariant rows and failure reason-code vocabulary for this subsystem.

### 9.1 Legacy anchor -> Rust boundary map

| Legacy anchors | Observable behavior family | Planned Rust boundary |
|---|---|---|
| `numpy/lib/_stride_tricks_impl.py` (`as_strided`, `_broadcast_to`, `broadcast_to`) | stride-view construction, read-only/writeable semantics, shape validation | `crates/fnp-ndarray` public API + layout core (`broadcast_shape`, `broadcast_shapes`, `NdLayout`) |
| `numpy/lib/_stride_tricks_impl.py` (`_broadcast_shape`, `broadcast_arrays`) | N-ary broadcast merge, high-arity behavior, output view semantics | `crates/fnp-ndarray` + packet-specific conformance fixtures (`bd-23m.17.5`, `bd-23m.17.6`) |
| `numpy/_core/src/multiarray/nditer_constr.c` (`npyiter_fill_axisdata`, `broadcast_error`, `operand_different_than_broadcast`) | zero-stride propagation, no-broadcast rejection, mismatch diagnostics | `crates/fnp-iter` iterator semantics layer (planned under `bd-23m.17.4`) |
| `numpy/_core/src/multiarray/nditer_api.c` (`NpyIter_GetShape`, `NpyIter_CreateCompatibleStrides`) | iterator-shape exposure and compatible-stride derivation | `crates/fnp-iter` traversal/introspection contracts + `fnp-ndarray` layout integration |

### 9.2 Verification hooks recorded by packet-A ledger

| Verification lane | Current status | Next owner bead |
|---|---|---|
| Unit/property | Anchor ledger complete, executable packet-specific tests not yet implemented | `bd-23m.17.5` |
| Differential/metamorphic/adversarial | Anchor ledger complete, packet-specific differential corpus not yet implemented | `bd-23m.17.6` |
| E2E replay/forensics | Anchor ledger complete, packet-specific workflow scenario not yet implemented | `bd-23m.17.7` |
| Structured logging | Contract fields known; packet-local enforcement hooks still pending implementation | `bd-23m.17.5`, `bd-23m.17.7` |
