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
| NDIter / transfer semantics | `numpy/_core/src/multiarray/nditer*`, `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | `crates/fnp-iter/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-003/*` and `FNP-P2C-006/*` | covered (unit/property contract tests) | missing packet-local differential lane | missing packet-local journey | covered at record-schema level (not yet gate-enforced) |
| Random | `numpy/random/*.pyx`, `numpy/random/src/*` | `crates/fnp-random/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-007/*` | covered (deterministic/state/bounded tests) | missing packet-local differential lane | missing seed/state journey | covered at record-schema level (not yet gate-enforced) |
| Linalg | `numpy/linalg/lapack_lite/*` | `crates/fnp-linalg/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-008/*` | covered (shape/solver/policy contract tests) | missing packet-local differential lane | missing solver journey | covered at record-schema level (not yet gate-enforced) |
| IO | `numpy/lib/format.py`, npy/npz handling paths | `crates/fnp-io/src/lib.rs`, packet docs in `artifacts/phase2c/FNP-P2C-009/*` | covered (parser/budget/policy contract tests) | missing packet-local differential lane | missing io journey | covered at record-schema level (not yet gate-enforced) |

Traceability anchors:

- Legacy source roots: `/data/projects/franken_numpy/legacy_numpy_code/numpy`
- Machine contracts: `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/phase2c_contract_schema_v1.json`
- Gate artifacts: `crates/fnp-conformance/src/test_contracts.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `scripts/e2e/run_test_contract_gate.sh`

Contradictions/unknowns register (for closure in doc passes 01-10):

| ID | Item | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `DOC-C001` | Section 6 carries historical reduced-scope language incompatible with full drop-in parity doctrine. | critical | `bd-23m.24.2` | Replace with parity-debt sequencing model and explicit owner/blocker table. |
| `DOC-C002` | Verification/logging implications are now mapped in DOC-PASS-03.4, but several packet domains remain explicitly missing/deferred. | high | `bd-23m.24.4` | Keep DOC-PASS-03.4 synchronized with packet implementation progress until missing lanes are closed. |
| `DOC-C003` | Ownership for NDIter/RNG/linalg/IO extraction is mapped and packet-local APIs now exist, but those APIs are underrepresented in conformance gate execution and packet F/G evidence. | high | `bd-23m.24.3` | Keep ownership/gate mapping synchronized until packet-local differential + replay suites are integrated and packet closure artifacts are complete. |

## DOC-PASS-01 Full Module/Package Cartography with Ownership and Boundaries

### DOC-PASS-01.1 Workspace ownership map (crate -> behavior contract)

| Crate | Ownership boundary (Rust anchors) | Legacy anchor families | Executable evidence anchors | Boundary contract |
|---|---|---|---|---|
| `fnp-dtype` | `DType`, `promote`, `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | `dtypemeta.c`, `descriptor.c`, `can_cast_table.h` | `dtype_promotion_cases.json`, `run_dtype_promotion_suite` | Owns deterministic dtype identity/promotion/cast primitives only. |
| `fnp-ndarray` | `broadcast_shape`, `broadcast_shapes`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout` (`crates/fnp-ndarray/src/lib.rs`) | `shape.c`, `shape.h`, stride/broadcast legality paths | `shape_stride_cases.json`, `run_shape_stride_suite` | Owns shape/stride legality and layout calculus only. |
| `fnp-ufunc` | `UFuncArray`, `BinaryOp`, `elementwise_binary`, `reduce_sum` (`crates/fnp-ufunc/src/lib.rs`) | `umath/ufunc_object.c` dispatch/reduction families | `ufunc_differential.rs`, `run_ufunc_*` suites | Owns scoped ufunc execution semantics. |
| `fnp-runtime` | strict/hardened decision APIs + `EvidenceLedger` (`crates/fnp-runtime/src/lib.rs`) | strict/hardened policy doctrine and fail-closed matrix | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, workflow scenarios | Owns compatibility policy decisions and audit records. |
| `fnp-conformance` | suite runners + artifact/security/workflow modules (`crates/fnp-conformance/src/lib.rs`, `src/bin/*.rs`) | legacy oracle tests and packet artifact topology contracts | gate binaries + packet readiness reports + sidecar/scrub/decode proof artifacts | Owns verification/evidence orchestration and contract enforcement. |
| `fnp-iter` | transfer/index APIs (`select_transfer_class`, overlap/flatiter validators, `TransferLogRecord`) (`crates/fnp-iter/src/lib.rs`) | `dtype_transfer.c`, `lowlevel_strided_loops.c.src`, `nditer*` | crate unit/property suites + `artifacts/phase2c/FNP-P2C-003/*` | Owns first-wave transfer/overlap/flatiter contract checks. |
| `fnp-random` | `DeterministicRng`, bounded/float/state APIs, `RandomLogRecord` (`crates/fnp-random/src/lib.rs`) | `numpy/random/_generator.pyx`, `numpy/random/src/*` | crate unit/property suites + `artifacts/phase2c/FNP-P2C-007/*` | Owns first-wave deterministic stream/state contract checks. |
| `fnp-linalg` | shape/solver/qr/svd/lstsq/policy validators + `LinAlgLogRecord` (`crates/fnp-linalg/src/lib.rs`) | `numpy/linalg/lapack_lite/*`, wrappers in `numpy/linalg/*` | crate unit/property suites + `artifacts/phase2c/FNP-P2C-008/*` | Owns first-wave linalg boundary and policy contract checks. |
| `fnp-io` | magic/header/descriptor/memmap/load/npz validators + `IOLogRecord` (`crates/fnp-io/src/lib.rs`) | `numpy/lib/format.py`, npy/npz paths | crate unit/property suites + `artifacts/phase2c/FNP-P2C-009/*` | Owns first-wave NPY/NPZ parser/writer boundary and policy checks. |

### DOC-PASS-01.2 Dependency direction and layering constraints

Dependency direction (from Cargo manifests):

1. Foundation semantics: `fnp-dtype`, `fnp-ndarray`.
2. Packet-local contract layers: `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`.
3. Execution layer: `fnp-ufunc` depends on `fnp-dtype` + `fnp-ndarray`.
4. Policy layer: `fnp-runtime` (optional runtime integrations).
5. Verification/gate layer: `fnp-conformance` depends on semantic/execution/policy crates and owns binaries.

Layering constraints for this pass:

- Semantic crates must not depend on conformance/tooling crates.
- Packet-local crates must remain policy-neutral (no runtime decision coupling in core contract logic).
- Conformance binaries are orchestration-only and must not redefine numerical semantics.
- Any new dependency edge that inverts this direction is a gate-level defect.

### DOC-PASS-01.3 Module and binary entrypoint cartography

| Path | Role class | Topology implication |
|---|---|---|
| `crates/fnp-ndarray/src/lib.rs` | SCE legality kernel | Broadcast/reshape/stride legality funnel. |
| `crates/fnp-ufunc/src/lib.rs` | Execution kernel | Broadcasted binary/reduction operation semantics. |
| `crates/fnp-runtime/src/lib.rs` | Policy kernel | strict/hardened fail-closed decisioning and ledger recording. |
| `crates/fnp-iter/src/lib.rs` | Transfer/index contract kernel | overlap/broadcast/flatiter legality for transfer semantics. |
| `crates/fnp-random/src/lib.rs` | RNG contract kernel | deterministic constructor/state/stream/bounded sampling semantics. |
| `crates/fnp-linalg/src/lib.rs` | Linalg contract kernel | shape/solver/mode/backend policy contract enforcement. |
| `crates/fnp-io/src/lib.rs` | IO boundary kernel | NPY/NPZ parser/dispatch/budget and fail-closed policy checks. |
| `crates/fnp-conformance/src/bin/run_security_gate.rs` | Security gate orchestrator | Security + runtime policy contract enforcement path. |
| `crates/fnp-conformance/src/bin/run_test_contract_gate.rs` | Logging contract gate | Structured logging schema and runtime suite coupling. |
| `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` | E2E replay gate | Scenario-level replay/logging checks. |
| `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs` | Packet readiness gate | Packet artifact closure and schema validation. |

### DOC-PASS-01.4 Verification and logging implications by ownership boundary

| Ownership boundary | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `fnp-dtype` | covered | partial (promotion-focused) | missing packet-local | partial | partial |
| `fnp-ndarray` | covered | partial | deferred packet-local | partial | partial |
| `fnp-ufunc` | covered | covered for scoped ops | deferred packet-local | covered | partial |
| `fnp-runtime` | covered | covered for policy-wire classes | covered via workflow suite | covered | covered for scoped surface |
| `fnp-iter` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |
| `fnp-random` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |
| `fnp-linalg` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |
| `fnp-io` | covered | missing packet-local lane | missing packet-local | covered at schema level | partial |

### DOC-PASS-01.5 Topology contradictions and closure backlog

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `TOPO-C001` | Packet-local crates (`fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`) now implement first-wave contract APIs, but conformance gate orchestration still prioritizes ufunc/runtime lanes. | high | `bd-23m.17`/`bd-23m.18`/`bd-23m.19`/`bd-23m.20` | Integrate packet-local differential + replay suites into operational gate flows. |
| `TOPO-C002` | Dependency direction is documented, but no automated check currently enforces forbidden edges. | medium | `bd-23m.23` | Add machine-enforced dependency-direction contract check and fail on violations. |
| `TOPO-C003` | Logging contracts exist for packet-local crates, but enforcement is mostly schema-level tests and not fully represented in gate outputs. | medium | packet G/I beads | Promote packet-local structured-log checks into gate assertions and readiness reports. |

## DOC-PASS-02 Symbol/API Census and Surface Classification

### DOC-PASS-02.1 Public symbol surface by crate

| Crate | Public symbol/API surface (Rust anchors) | Visibility class | Primary usage contexts |
|---|---|---|---|
| `fnp-dtype` | `DType::{name,item_size,parse}`, `promote`, `can_cast_lossless` (`crates/fnp-dtype/src/lib.rs`) | core semantic contract | Called by `fnp-ufunc` execution (`crates/fnp-ufunc/src/lib.rs`) and conformance promotion suite (`run_dtype_promotion_suite` in `crates/fnp-conformance/src/lib.rs`). |
| `fnp-ndarray` | `MemoryOrder`, `ShapeError`, `can_broadcast`, `broadcast_shape`, `broadcast_shapes`, `element_count`, `fix_unknown_dimension`, `contiguous_strides`, `NdLayout::{contiguous,nbytes}` (`crates/fnp-ndarray/src/lib.rs`) | core semantic contract | Used by ufunc execution shape checks and by conformance shape/stride suite (`run_shape_stride_suite`). |
| `fnp-ufunc` | `BinaryOp`, `UFuncArray::{new,scalar,shape,values,dtype,elementwise_binary,reduce_sum}`, `UFuncError` (`crates/fnp-ufunc/src/lib.rs`) | execution contract | Called from differential/metamorphic/adversarial harness via `execute_input_case` (`crates/fnp-conformance/src/ufunc_differential.rs`). |
| `fnp-runtime` | `RuntimeMode`, `CompatibilityClass`, `DecisionAction`, `DecisionLossModel`, `DecisionAuditContext`, `DecisionEvent`, `OverrideAuditEvent`, `EvidenceLedger`, `decide_*`, `evaluate_policy_override`, `posterior_incompatibility`, `expected_loss_for_action` (`crates/fnp-runtime/src/lib.rs`) | policy/audit contract | Used by runtime policy suites and workflow scenario execution in `fnp-conformance`. |
| `fnp-conformance` | `HarnessConfig`, `HarnessReport`, `SuiteReport`, `run_*` suites, plus module exports (`benchmark`, `contract_schema`, `raptorq_artifacts`, `security_contracts`, `test_contracts`, `ufunc_differential`, `workflow_scenarios`) (`crates/fnp-conformance/src/lib.rs`) | verification/tooling contract | Entry point for all gate binaries and packet-readiness tooling. |
| `fnp-iter` | `TransferClass`, `OverlapAction`, `TransferSelectorInput`, `FlatIterIndex`, `TransferError`, transfer/overlap/flatiter validators, `TransferLogRecord` (`crates/fnp-iter/src/lib.rs`) | packet-local first-wave contract | Transfer semantics foundation for packet-003/006 domains; currently unit/property-backed only. |
| `fnp-random` | `DeterministicRng`, `RandomError`, `RandomRuntimeMode`, `RandomLogRecord`, stream/state/bounded APIs (`crates/fnp-random/src/lib.rs`) | packet-local first-wave contract | Deterministic RNG boundary for packet-007; currently unit/property-backed only. |
| `fnp-linalg` | `LinAlgError`, `QrMode`, shape/solver/spectral/lstsq/policy validators, `LinAlgLogRecord` (`crates/fnp-linalg/src/lib.rs`) | packet-local first-wave contract | Linalg boundary for packet-008; currently unit/property-backed only. |
| `fnp-io` | `IOError`, `IOSupportedDType`, `MemmapMode`, `LoadDispatch`, NPY/NPZ validators, `IOLogRecord` (`crates/fnp-io/src/lib.rs`) | packet-local first-wave contract | IO boundary for packet-009; currently unit/property-backed only. |

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
| Tier C: packet-local first-wave contract APIs | `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` contract surfaces | medium: implemented at crate level but under-integrated in packet-local differential/e2e gates | internal contractual surfaces with explicit parity debt until lane-complete evidence lands |

### DOC-PASS-02.4 Verification and logging implications by symbol family

| Symbol family | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `fnp-dtype` promotion/cast APIs | covered (crate tests + promotion fixtures) | partial (promotion only; cast-matrix diff missing) | missing | missing dedicated cast reason-code taxonomy | partial |
| `fnp-ndarray` shape/stride APIs | covered (crate tests + shape_stride fixtures) | partial via ufunc differential shape validation | deferred | partial via runtime/workflow logs | partial |
| `fnp-ufunc` execution APIs | covered (crate tests + metamorphic/adversarial fixtures) | covered for scoped ops (`add/sub/mul/div/sum`) | deferred packet scenarios | covered at scenario/log-entry level | partial |
| `fnp-runtime` policy APIs | covered (crate tests + policy suites) | policy-wire adversarial coverage (non-numeric) | covered via workflow scenario suite | covered (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) | covered for current scope |
| `fnp-conformance` contract/sidecar APIs | covered via module tests and gate binaries | N/A | covered through gate binaries | covered through gate log validation | covered/ongoing |
| Packet-local first-wave symbols (`fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`) | covered (crate unit/property + reason-code/log schema checks) | missing packet-local differential lanes | missing packet-local replay lanes | covered at schema level; not fully gate-enforced | partial/open |

### DOC-PASS-02.5 Packet ownership and closure gates for unresolved domains (`DOC-C003`)

| Unresolved behavior family | Current packet-local contract API | Packet owner bead | Required closure gates |
|---|---|---|---|
| NDIter traversal/index semantics | transfer/index contract APIs in `fnp-iter` | `bd-23m.17` (`FNP-P2C-006`) | Expand into packet F/G/H/I lanes and wire into conformance gates. |
| RNG deterministic streams/state schema | deterministic stream/state APIs in `fnp-random` | `bd-23m.18` (`FNP-P2C-007`) | Add oracle differential + replay lanes and promote deterministic-seed witness artifacts into packet gates. |
| Linalg adapter semantics | shape/solver/policy contract APIs in `fnp-linalg` | `bd-23m.19` (`FNP-P2C-008`) | Add linalg differential + replay + optimization evidence lanes. |
| NPY/NPZ parser/writer hardening | parser/writer boundary validators in `fnp-io` | `bd-23m.20` (`FNP-P2C-009`) | Add IO differential + replay lanes and gate-enforced durability/logging links. |

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
| Unresolved domain executability | NDIter/RNG/linalg/IO crates must be lane-complete in conformance gates (not crate-only complete) | partial (crate APIs implemented; differential/e2e integration pending) | requires packet-local differential/replay gate integration and readiness artifacts |

### DOC-PASS-03.5 Data/invariant contradictions and unknowns

| ID | Contradiction or unknown | Risk | Owner bead | Closure criteria |
|---|---|---|---|---|
| `STATE-C001` | `NdLayout` fields are public, so post-construction mutation can bypass constructor-time legality assumptions. | medium | `bd-23m.12` | Decide/encode whether layout fields remain public by design or become encapsulated with invariant-preserving mutators only. |
| `STATE-C002` | Runtime/workflow logging can no-op when no path is configured, weakening forensic guarantees outside gate runs. | medium | `bd-23m.6` | Require explicit log paths in orchestrator contexts and fail on missing required logs. |
| `STATE-C003` | Packet-local data models now exist for NDIter/RNG/linalg/IO, but their packet-local differential/e2e evidence lanes are not fully integrated into gate workflows. | high | `bd-23m.17`/`bd-23m.18`/`bd-23m.19`/`bd-23m.20` | Integrate packet-local differential/replay suites into gate outputs and packet readiness closure artifacts. |

## DOC-PASS-04 Execution-Path Tracing and Control-Flow Narratives

### DOC-PASS-04.1 Major workflow narratives (entrypoint -> branch -> terminal)

| Workflow ID | Entry anchor | Ordered branch decisions | Terminal outcomes | Evidence anchors |
|---|---|---|---|---|
| `EP-001` Ufunc differential replay | `crates/fnp-conformance/src/bin/run_ufunc_differential.rs` | fixture/oracle load -> case parse -> `UFuncArray` shape/value admission -> op dispatch -> compare | pass report / mismatch report / parse or admission failure | `crates/fnp-conformance/src/ufunc_differential.rs`, `fixtures/ufunc_*` |
| `EP-002` Security gate | `crates/fnp-conformance/src/bin/run_security_gate.rs` | runtime policy suite -> adversarial policy suite -> security-contract suite -> aggregate gate result | gate pass / gate fail with reason taxonomy | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts` |
| `EP-003` Workflow scenario gate | `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` | fixture parse -> strict/hardened execution -> expectation compare -> log append | scenario pass / deterministic failure record | `workflow_scenarios::run_user_workflow_scenario_suite`, `artifacts/logs/workflow_scenario_e2e_*.jsonl` |
| `EP-004` Packet readiness validator | `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs` | required file check -> schema/token check -> parse check -> readiness summarization | ready / non-ready diagnostic report | `contract_schema::validate_phase2c_packet`, `packet_readiness_report.json` |
| `EP-005` IO dispatch and validation | `crates/fnp-io/src/lib.rs` (`classify_load_dispatch`, header/payload validators) | NPZ magic -> NPY magic -> pickle allowed? -> fail-closed invalid prefix | dispatch branch / stable IO error reason code | `IO_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-009/*` |
| `EP-006` RNG deterministic generation | `crates/fnp-random/src/lib.rs` (`DeterministicRng`, `bounded_u64`) | upper-bound gate -> rejection sampling threshold loop -> modulo projection | bounded value / stable RNG error reason code | `RANDOM_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-007/*` |
| `EP-007` Linalg branch family | `crates/fnp-linalg/src/lib.rs` (shape preflight, QR/SVD/LSTSQ/policy validators) | rank/shape preflight -> mode token parse -> convergence gate -> policy metadata gate | output-shape contracts / stable linalg reason codes | `LINALG_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-008/*` |
| `EP-008` Transfer/index legality | `crates/fnp-iter/src/lib.rs` (transfer selector, overlap policy, flatiter validators) | context checks -> stride multiple checks -> overlap direction -> index/value arity checks | transfer class/copy action / stable transfer reason codes | `TRANSFER_PACKET_REASON_CODES`, `artifacts/phase2c/FNP-P2C-003/*` |

### DOC-PASS-04.2 Control-flow ordering and fallback law

| Branch family | Ordering rule | Fallback semantics |
|---|---|---|
| Dispatch classification (`EP-005`) | NPZ prefix first, then NPY prefix, then pickle only with explicit policy allow | Unknown/forbidden prefixes fail closed (`io_load_dispatch_invalid`) |
| Runtime mode/class decode (`EP-002`/`EP-003`) | normalize + classify known tokens before action selection | unknown metadata resolves to fail-closed and logged reason code |
| Ufunc admission (`EP-001`) | validate fixtures and shape/value/dtype preconditions before numeric dispatch | precondition failures short-circuit with deterministic diagnostics |
| RNG bounded sampling (`EP-006`) | reject `upper_bound == 0` before sampling loop | invalid bound hard-fails with stable reason code |
| Linalg mode/convergence (`EP-007`) | shape/rank gate precedes mode/convergence checks | invalid mode/non-convergence/unknown metadata fail closed |
| Transfer overlap/indexing (`EP-008`) | context sanity then overlap/index checks before allowing lane operations | contract violations emit stable transfer reason codes and reject operation |

### DOC-PASS-04.3 Verification/logging implications by execution path

| Workflow ID | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| `EP-001` | covered | covered for scoped ufunc corpus | partial | covered | partial parity debt outside scoped op set |
| `EP-002` | covered | covered for policy-wire classes | covered | covered | strong for current policy scope |
| `EP-003` | covered | N/A | covered | covered when path configured | medium operational risk |
| `EP-004` | covered | N/A | covered | structured readiness reports | strong artifact-topology coverage |
| `EP-005` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |
| `EP-006` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |
| `EP-007` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |
| `EP-008` | covered | missing packet-local differential lane | missing packet-local replay lane | schema-level coverage exists | partial |

### DOC-PASS-04.4 Control-flow contradictions and closure backlog

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `FLOW-C001` | Packet-local paths (`EP-005`..`EP-008`) are implemented and tested but not yet promoted to first-class suites in core gate orchestration. | high | `bd-23m.14`/`17`/`18`/`19`/`20` | Integrate packet-local differential/replay suites into gate binaries and `run_all_core_suites`. |
| `FLOW-C002` | Runtime/workflow log append behavior is configuration-dependent, creating potential forensic blind spots. | medium | `bd-23m.6` | Require explicit log path configuration and fail when expected logs are absent. |
| `FLOW-C003` | Packet readiness reports validate artifact topology but do not yet encode branch-reachability proof links for packet-local paths. | medium | packet I beads + DOC pass-10 | Add branch-level replay evidence references into readiness report conventions. |

## DOC-PASS-05 Complexity, Performance, and Memory Characterization

### DOC-PASS-05.1 Complexity map by workflow

| Workflow ID | Dominant operation class | Time complexity (current wave) | Space complexity (current wave) | Anchors |
|---|---|---|---|---|
| `EP-001` | fixture replay + ufunc execution + oracle compare | `O(C * N)` (cases × per-case element count) | `O(N)` per-case buffers + report growth | `ufunc_differential.rs`, `fnp-ufunc/src/lib.rs` |
| `EP-002` | runtime/security suite execution | `O(C)` over policy/security fixtures | `O(C)` reports/log metadata | `run_security_gate.rs`, runtime suites |
| `EP-003` | scenario replay engine | `O(S * K)` (scenarios × steps) | `O(S + log_entries)` | `workflow_scenarios.rs`, workflow gate |
| `EP-004` | packet readiness contract scan | `O(F)` required artifact checks | `O(F)` diagnostics | `contract_schema.rs`, readiness validator |
| `EP-005` | IO dispatch + header/shape budget checks | dispatch `O(1)`; schema checks `O(rank)` | `O(rank)` metadata structures | `fnp-io/src/lib.rs` |
| `EP-006` | deterministic RNG + bounded sampling | `O(1)` per draw expected; `fill_u64(len)` is `O(len)` | `O(1)` per draw; `O(len)` for fills | `fnp-random/src/lib.rs` |
| `EP-007` | linalg contract validators + small helper kernels | mostly `O(rank)`/`O(1)` (2x2 solve is constant) | mostly `O(1)` plus output-shape vectors | `fnp-linalg/src/lib.rs` |
| `EP-008` | transfer selector + overlap/index checks | selector/overlap `O(1)`; index validation `O(len)` | `O(1)` plus caller index storage | `fnp-iter/src/lib.rs` |

### DOC-PASS-05.2 Memory-growth characterization and boundedness controls

| Domain | Growth driver | Current boundedness controls | Residual risk |
|---|---|---|---|
| Ufunc execution | broadcasted output element count and dtype width | shape legality + element-count admission | large broadcast expansions can still pressure memory |
| Conformance/gate reports | fixture corpus size and run count | harness config and packet contract checks | longitudinal report/log accumulation can grow quickly |
| RNG fill workflows | requested output length | caller-controlled `len` and packet fixture contracts | large fill requests can dominate memory |
| Linalg contract workflows | batch-shape metadata and output tuples | `MAX_BATCH_SHAPE_CHECKS`, policy bounds | future full kernels will change memory profile materially |
| IO parser/workflow | shape rank, archive member count, payload footprint | `MAX_HEADER_BYTES`, `MAX_ARCHIVE_MEMBERS`, byte budget constants | decoded payload memory envelopes need cross-gate enforcement |
| Transfer/index checks | mask/fancy index lengths | fail-fast validation paths | large caller-provided index vectors still expensive |

### DOC-PASS-05.3 Hotspot family inventory and optimization-governance readiness

| Hotspot family | Current evidence | Governance gap | Owner |
|---|---|---|---|
| Ufunc loops | baseline artifact exists (`artifacts/baselines/ufunc_benchmark_baseline.json`) | continue per-lever profile/isomorphism deltas | `bd-23m.16` |
| Conformance orchestration | differential/gate reports exist | standardized profiler artifact convention missing | packet H/I + foundation |
| Runtime/workflow gates | gate summaries + logs exist | longitudinal cost tracking not yet formalized | `bd-23m.8` |
| Packet-local iter/random/linalg/io paths | unit/property evidence exists | dedicated perf baselines/profiles mostly missing | `bd-23m.14`/`18`/`19`/`20` |
| Durability sidecar pipeline | sidecar/scrub/decode artifacts generated | throughput/storage trend baselines missing | packet I + durability owners |

### DOC-PASS-05.4 Verification/logging implications for optimization work

| Domain | Unit/Property | Differential | E2E | Logging | Status |
|---|---|---|---|---|---|
| Ufunc optimization | covered | covered (scoped corpus) | partial | covered | partial outside scoped op set |
| Runtime/security optimization | covered | covered (policy-wire) | covered | covered | strong |
| Iter/random/linalg/io optimization | covered | missing packet-local lanes | missing packet-local lanes | schema-level only | missing/partial |
| Durability pipeline optimization | covered by contract suites | N/A | gate path exists | artifact metadata logged | partial trend visibility |

### DOC-PASS-05.5 Performance/memory contradiction register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `PERF-C001` | EV-gated optimization policy is explicit, but non-ufunc packet paths lack standardized baseline/profile artifacts. | high | `bd-23m.8`, packet H beads | Require path-specific baseline/profile attachments in packet optimization reports. |
| `PERF-C002` | RNG rejection sampling (`bounded_u64`) has expected constant behavior but no explicit retry cap in adversarial worst cases. | medium | `bd-23m.18` | Add bounded retry policy or explicit guardrail rationale + adversarial witness tests. |
| `PERF-C003` | Gate-cost regressions are not tracked longitudinally as first-class artifacts. | medium | `bd-23m.8` | Add periodic gate-cost trend artifacts with alert thresholds. |
| `PERF-C004` | Crate-local memory bounds exist, but a cross-packet aggregate memory envelope is undocumented. | medium | `bd-23m.24.6` follow-on + packet owners | Publish aggregate memory budget matrix with enforcement hooks in gate scripts. |

## DOC-PASS-06 Concurrency/Lifecycle Semantics and Ordering Guarantees

### DOC-PASS-06.1 Shared-state and lifecycle ownership map

| Surface | Anchor | Mutation path | Synchronization/ownership | Ordering semantics |
|---|---|---|---|---|
| Runtime decision events | `fnp-runtime::EvidenceLedger` | `record` during `decide_and_record_with_context` | per-instance mutable ownership (`&mut self`) | append order equals decision call order per ledger instance |
| Runtime/shape/dtype log path config | `crates/fnp-conformance/src/lib.rs` statics | `set_*_log_path` setters | `OnceLock<Mutex<Option<PathBuf>>>` | last successful setter wins |
| Workflow log path + required flag | `crates/fnp-conformance/src/workflow_scenarios.rs` statics | `set_workflow_scenario_log_path`, `set_workflow_scenario_log_required` | `OnceLock<Mutex<...>>` + drop-guard reset in workflow gate | required-flag lifecycle is deterministic per workflow gate run |
| Gate attempt lifecycle state | gate binaries (`run_security_gate`, `run_test_contract_gate`, `run_workflow_scenario_gate`) | monotonic attempt loop with retry budget | local vectors/structs owned per process run | attempt order is strictly `0..=retries`, break on first pass |

### DOC-PASS-06.2 Ordering guarantees and branch sequencing

| Domain | Guarantee | Anchor |
|---|---|---|
| Scenario and step execution | scenario corpus order is preserved; steps run in declared order | `for scenario in &scenarios`, `for step in &scenario.steps` in `workflow_scenarios.rs` |
| Runtime action evaluation | compatibility decision computed before event append; event includes normalized context | `decide_compatibility*` + `normalize_audit_context` + `ledger.record` in `fnp-runtime` |
| Log append sequence | each helper appends serialized JSON line in append mode after directory/create/open checks | `maybe_append_*log` helpers in conformance/workflow modules |
| Workflow required-log behavior | missing workflow log path is fatal only when required flag is true | `maybe_append_workflow_log` + `workflow_scenario_log_required()` |

### DOC-PASS-06.3 Concurrency/lifecycle risk register

| ID | Risk | Owner | Closure criteria |
|---|---|---|---|
| `CONC-C001` | Global mutable log-path config can be overwritten by concurrent in-process gate runs. | foundation `bd-23m.6`/`bd-23m.23` | move to scoped per-run log handles or enforce single-run isolation, plus regression tests |
| `CONC-C002` | Runtime/shape/dtype appenders allow silent no-op when log path is unset. | foundation `bd-23m.6` | add mandatory-log mode or explicit gate-level missing-log failures |
| `CONC-C003` | No explicit stress tests for cross-process log append contention/order stability. | `bd-23m.7` + follow-on docs/testing passes | add concurrency stress harness and artifactized ordering checks |
| `CONC-C004` | Ordering guarantees documented but not uniformly encoded as formal contract tests across packet-local logging pipelines. | packet G/I beads | add ordering contract tests and require them in readiness evidence |

### DOC-PASS-06.4 Verification/logging implications

| Area | Unit/Property | Differential | E2E | Structured logging | Status |
|---|---|---|---|---|---|
| Runtime lifecycle/event ordering | covered | N/A | covered via gates | covered | strong in single-process scope |
| Workflow lifecycle and required logging | covered | N/A | covered | covered | medium (global config contention remains) |
| Cross-process append behavior | missing | missing | missing | partial visibility only | open |

## DOC-PASS-07 Error Taxonomy, Failure Modes, and Recovery Semantics

### DOC-PASS-07.1 Canonical error taxonomy (code-level anchors)

| Error family | Primary type/registry | Failure class | Recovery semantics |
|---|---|---|---|
| Shape/stride legality | `ShapeError` (`crates/fnp-ndarray/src/lib.rs`) | contract/input validation failure | fail-fast `Result::Err`; caller must correct input shape/stride spec. |
| Ufunc execution | `UFuncError` + `UFUNC_PACKET_REASON_CODES` (`crates/fnp-ufunc/src/lib.rs`) | shape/input/axis contract failure | deterministic rejection; reason code emitted via `UFuncError::reason_code`. |
| Runtime policy | `DecisionAction::FailClosed` (`crates/fnp-runtime/src/lib.rs`) | compatibility fail-closed rejection | deterministic fail-closed action plus evidence event record. |
| Transfer/index semantics | `TransferError` + `TRANSFER_PACKET_REASON_CODES` (`crates/fnp-iter/src/lib.rs`) | selector/overlap/index contract failure | immediate rejection with stable transfer reason code. |
| RNG contracts | `RandomError` + `RANDOM_PACKET_REASON_CODES` (`crates/fnp-random/src/lib.rs`) | bounded-generation/state validation failure | deterministic error return (no automatic repair). |
| Linalg contracts | `LinAlgError` + `LINALG_PACKET_REASON_CODES` (`crates/fnp-linalg/src/lib.rs`) | shape/mode/convergence/policy failure | fail-fast plus packet-local reason code. |
| IO contracts | `IOError` + `IO_PACKET_REASON_CODES` (`crates/fnp-io/src/lib.rs`) | parser/dispatch/policy failure | fail-fast and fail-closed on unknown/incompatible metadata. |
| Gate reliability diagnostics | `deterministic_failure`, `flake_budget_exceeded`, `coverage_floor_breach` (gate bins) | orchestration reliability failure | non-zero gate exit + structured diagnostics with evidence refs. |

### DOC-PASS-07.2 Failure mode matrix (detect -> classify -> act)

| Failure mode | Detection point | Classification channel | Action | Escalation path |
|---|---|---|---|---|
| Shape/ufunc/transfer contract mismatch | crate validators/constructors | typed error enum + reason code | immediate `Err` | fixture/gate failure output with case metadata |
| Unknown compatibility metadata | runtime/linalg/io policy checks | fail-closed decision or policy reason code | reject operation | scenario/gate diagnostics + audit trails |
| Workflow log path missing (required mode) | `maybe_append_workflow_log` | explicit error string | fail scenario step | retry loop then deterministic failure diagnostics |
| Flake budget exceeded | gate reliability checks | `flake_budget_exceeded` | mark gate failed | operator triage with attempt logs |
| Coverage floor breach | gate reliability checks | `coverage_floor_breach` | mark gate failed | expand/repair coverage before promotion |

### DOC-PASS-07.3 Recovery semantics by layer

| Layer | Recovery strategy implemented | Explicit debt |
|---|---|---|
| Core semantic crates | deterministic fail-fast typed errors | no auto semantic repair (intentional parity-observability constraint) |
| Runtime policy layer | fail-closed defaults + optional hardened full-validate path | no probabilistic auto-override policies |
| Conformance suites | accumulate case failures with explicit diagnostics | packet-local differential/replay lanes incomplete in some domains |
| Gate binaries | bounded retry loops + reliability diagnostics | no adaptive/decision-theoretic retry policies |
| Logging/forensics | structured JSONL append + schema/field contract checks | global-path contention hardening and strict mandatory logging still open |

### DOC-PASS-07.4 User-visible vs internal error channels

| Channel | Audience | Current form |
|---|---|---|
| Error enum `Display` output | humans (dev/operators) | readable text with contextual fields |
| Reason-code registries (`*_PACKET_REASON_CODES`) | machine analyzers/gates/replay tools | stable tokenized identifiers |
| Gate diagnostics payloads | CI/operator automation | structured JSON diagnostics (`subsystem`, `reason_code`, `message`, `evidence_refs`) |
| Runtime/workflow log records | forensic replay and audit tooling | JSONL entries keyed by fixture/test/mode/env/artifact/reason fields |

### DOC-PASS-07.5 Verification implications for error/recovery semantics

| Domain | Unit/Property | Differential | E2E | Logging-contract coverage | Status |
|---|---|---|---|---|---|
| Shape + ufunc + runtime lanes | covered | covered (scoped) | covered in current gates | covered | strong for current scope |
| Transfer/random/linalg/io lanes | covered at crate level | missing packet-local differential lanes | missing packet-local replay lanes | schema-level coverage only | partial/open |
| Gate reliability diagnostics | covered | N/A | covered | covered | strong |

### DOC-PASS-07.6 Error/recovery contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `ERR-C001` | Packet-local error taxonomies exist, but gate orchestration still prioritizes ufunc/runtime lanes. | high | `bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6` | integrate packet-local differential/replay suites into gate summaries with reason-code outputs |
| `ERR-C002` | Runtime/shape/dtype log appenders can no-op when path unset, reducing recovery forensics. | medium | `bd-23m.6` | add mandatory logging mode or gate failure on missing expected logs |
| `ERR-C003` | Retry logic is fixed-budget and not decision-theoretic/adaptive. | medium | `bd-23m.8` | add explicit expected-loss retry/escalation policy and artifacts |
| `ERR-C004` | Packet readiness does not yet require exhaustive reason-code coverage attestations. | medium | packet I beads + docs pass-10 | require reason-code coverage proofs in readiness contracts |

## DOC-PASS-08 Security/Compatibility Edge Cases and Undefined Zones

### DOC-PASS-08.1 High-risk edge-zone map

| Edge zone | Legacy anchor(s) | Rust anchor(s) | Strict expectation | Hardened expectation |
|---|---|---|---|---|
| Unknown wire metadata | no canonical NumPy strict/hardened wire token contract | `decide_compatibility_from_wire`, `evaluate_policy_override` (`crates/fnp-runtime/src/lib.rs`) | fail-closed on unknown mode/class | fail-closed on unknowns; override requires hardened + allowlist + known-compatible |
| NPY/NPZ parser boundaries | `numpy/lib/format.py`, `numpy/lib/npyio.py` | `validate_magic_version`, `validate_header_schema`, `validate_npz_archive_budget` (`crates/fnp-io/src/lib.rs`) | reject malformed/overflowed inputs | same, with bounded retry/size controls |
| Object/pickle policy edges | `numpy.load(... allow_pickle=...)` behavior families | `enforce_pickle_policy`, `classify_load_dispatch`, `validate_memmap_contract` (`crates/fnp-io/src/lib.rs`) | reject object payload when policy disallows | same unless explicit policy allow path |
| Reshape/stride undefined edges | `numpy/_core/src/multiarray/shape.c`, `numpy/lib/_stride_tricks_impl.py` | `fix_unknown_dimension`, `NdLayout::as_strided`, `broadcast_strides` (`crates/fnp-ndarray/src/lib.rs`) | deterministic contract rejection on invalid transitions | same |
| NDIter overlap and index policy | `numpy/_core/src/multiarray/nditer*` | `overlap_copy_policy`, `validate_nditer_flags`, `validate_flatiter_*` (`crates/fnp-iter/src/lib.rs`) | reject violating overlap/broadcast/index rules | same |
| Linalg policy metadata/backend boundary | `numpy/linalg/*` | `validate_policy_metadata`, `validate_backend_bridge` (`crates/fnp-linalg/src/lib.rs`) | fail unknown metadata/backend unsupported paths | same |

### DOC-PASS-08.2 Undefined-zone stance and open risk

| Zone | Current deterministic stance | Risk |
|---|---|---|
| `unknown_semantics` token handling across crates | runtime layer remains fail-closed authority for unknown/incompatible semantics | high |
| Negative-stride view breadth parity | ndarray currently rejects negative strides in `as_strided`-related validation | high |
| Object dtype + memmap/pickle combinations | explicit policy gates decide branch; invalid combinations reject | high |
| Unknown gate reason-code families | workflow gate maps unknown reason codes to generic `scenario_assertion` class | medium |
| Optional logging paths outside required workflow mode | some appenders can no-op when path unset | medium |

### DOC-PASS-08.3 Verification/logging implications

| Edge family | Unit/Property | Differential | E2E | Logging coverage | Status |
|---|---|---|---|---|---|
| Runtime metadata fail-closed paths | covered | covered (runtime suites) | covered in security/test/workflow gates | covered | strong |
| IO parser/policy boundedness | covered | partial packet-local integration | partial | covered | medium/open |
| Stride/reshape/alias edges | covered | partial | partial | covered | medium with parity debt |
| Iterator/linalg/random packet-local security edges | covered | missing packet-local differential lanes | missing packet-local replay lanes | schema-level coverage | partial/open |

### DOC-PASS-08.4 Security/compatibility contradiction register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `SEC-C001` | Packet-local metadata token acceptance and runtime unknown-semantics fail-closed doctrine are not yet unified as one explicit cross-crate rule. | high | foundation `bd-23m.8` + packet F/G owners | codify one end-to-end unknown-semantics contract with tests |
| `SEC-C002` | Negative-stride parity remains open because ndarray currently rejects negative strides in view validation. | high | `bd-23m.17` | implement/verify parity matrix for legacy-compatible negative-stride cases |
| `SEC-C003` | Hostile object/pickle IO paths lack full packet-local differential + replay evidence despite policy guards. | high | `bd-23m.20.6`, `bd-23m.20.7` | add adversarial differential/replay suites and readiness artifacts |
| `SEC-C004` | Optional-path logging for runtime/shape/dtype appenders can reduce forensic completeness during edge failures. | medium | `bd-23m.6` | enforce mandatory logging or fail gates when expected logs are absent |
| `SEC-C005` | Workflow gate currently coarsens unknown reason codes into generic classification, reducing triage precision. | medium | `bd-23m.23` | expand and test explicit reason-code class mapping |

## DOC-PASS-09 Unit/E2E Test Corpus and Logging Evidence Crosswalk

### DOC-PASS-09.1 Behavior evidence matrix (machine-parseable rows)

| row_id | behavior_family | unit_property_lane | differential_lane | e2e_gate_lane | logging_contract_lane | status | risk | owner_beads |
|---|---|---|---|---|---|---|---|---|
| `XW-001` | shape/stride legality + views | `fnp-ndarray` tests + `run_shape_stride_suite` | no dedicated legacy-oracle diff lane yet | indirect via workflow/runtime gates | shape-stride log entries with normalized replay fields | partial | medium | `bd-23m.12.6`, `bd-23m.17.6` |
| `XW-002` | dtype promotion determinism | dtype tests + `run_dtype_promotion_suite` | no dedicated oracle diff lane yet | aggregate only | dtype-promotion log entries with normalized replay fields | partial | medium | `bd-23m.13.6` |
| `XW-003` | runtime strict/hardened/fail-closed policy | runtime tests + runtime conformance + adversarial suite | adversarial fixture lane present | dedicated security/test/workflow gates + `scripts/e2e/run_*` wrappers | runtime ledger/log validation + required log-field gate | strong | low | `bd-23m.5`, `bd-23m.23` |
| `XW-004` | ufunc parity (differential/metamorphic/adversarial) | `fnp-ufunc` tests + conformance suites | dedicated oracle differential lane present | workflow-linked, but no dedicated standalone ufunc e2e gate | `UFuncLogRecord` field/registry validation | strong (core) | medium | `bd-23m.16.6`, `bd-23m.16.7` |
| `XW-005` | iter/random/linalg/io packet-local contracts | strong crate-level unit/property coverage per packet | packet-local differential lanes not yet integrated | packet-local replay lanes not yet integrated | packet-specific log-record validators and reason-code registries | partial/open | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` |

### DOC-PASS-09.2 Mandatory replay/log field contract

`run_test_contract_gate` enforces the canonical required fields for replay-complete logs:
- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Packet-local log-record structs (`UFuncLogRecord`, `TransferLogRecord`, `RandomLogRecord`, `LinAlgLogRecord`, `IOLogRecord`) and conformance normalizers must remain aligned to this field contract.

### DOC-PASS-09.3 Coverage gap ledger

| gap_id | Missing lane | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `GAP-XW-001` | Differential lanes for iter/random/linalg/io packet-local domains | high | `bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6` | add packet-local differential harnesses + reports |
| `GAP-XW-002` | E2E replay/gate lanes for iter/random/linalg/io packet-local domains | high | `bd-23m.14.7`, `bd-23m.18.7`, `bd-23m.19.7`, `bd-23m.20.7` | add packet-local workflow scenarios + gate integration |
| `GAP-XW-003` | Dedicated shape/dtype gate wrappers | medium | `bd-23m.23` + packet owners | add subsystem-focused gate partitioning or wrappers |
| `GAP-XW-004` | Optional-path logging still possible on some appenders | medium | `bd-23m.6` | enforce required logging or fail on missing expected logs |

### DOC-PASS-09.4 Crosswalk contradiction register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `XW-C001` | Replay-complete log contracts exist across packet-local crates, but cross-lane differential/e2e evidence is not uniformly integrated. | high | packet F/G owners | require four-lane evidence (unit/property + differential + e2e + log contract) for packet readiness |
| `XW-C002` | Reliability envelopes are mature for runtime/workflow gates but not yet generalized to packet-local lanes. | medium | `bd-23m.23` + packet owners | extend reliability summary/failure-envelope pattern to packet-local gates |
| `XW-C003` | Crosswalk is doc-embedded; no CI-emitted machine snapshot artifact currently guards drift. | medium | docs `bd-23m.24.10` + automation follow-up | emit versioned crosswalk JSON artifact and gate schema drift |

## DOC-PASS-10 Expansion Draft (Pass A): Subsystem Topology and Boundary Narratives

### DOC-PASS-10.1 Canonical subsystem topology map

| layer_id | Subsystem slice | Responsibilities | Source anchors | Dependency law |
|---|---|---|---|---|
| `A1` | Semantic kernel | dtype promotion/casting, shape/stride legality, broadcast/view contracts | `crates/fnp-dtype/src/lib.rs`, `crates/fnp-ndarray/src/lib.rs` | foundational; no dependence on conformance/gate bins |
| `A2` | Execution kernel set | ufunc compute paths, transfer/overlap/index contracts, packet-local RNG/linalg/IO contracts | `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | depends on `A1`; must emit typed errors/reason codes |
| `A3` | Runtime policy kernel | strict/hardened routing, fail-closed doctrine, override audit events | `crates/fnp-runtime/src/lib.rs` | evaluates compatibility metadata for `A1/A2` behavior requests |
| `A4` | Conformance orchestrator | fixture loading, suite execution, differential/metamorphic/adversarial reports, normalized logging | `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/src/workflow_scenarios.rs` | orchestrates `A1/A2/A3`; emits suite summaries |
| `A5` | Reliability/durability envelope | retry/flake/coverage gating, report diagnostics, workflow forensics index, RaptorQ artifact checks | `crates/fnp-conformance/src/bin/run_*_gate.rs`, `crates/fnp-conformance/src/raptorq_artifacts.rs`, `scripts/e2e/run_*_gate.sh` | wraps `A4`; policy/governance only, no semantic rewriting |

### DOC-PASS-10.2 Boundary narratives (call-flow law)

| Boundary | Allowed flow | Disallowed flow | Why |
|---|---|---|---|
| `A1 -> A2` | execution consumes legality/promotion outputs from semantic kernel | execution defining alternate legality rules ad hoc | preserves deterministic SCE + dtype contract as single source of truth |
| `A2/A1 -> A3` | runtime policy receives mode/class/risk metadata and returns action (`allow`, `full_validate`, `fail_closed`) | execution bypassing policy layer on compatibility-sensitive paths | enforces strict/hardened doctrine and fail-closed guarantees |
| `A3/A2/A1 -> A4` | conformance invokes kernels and policy via fixture suites and records normalized metadata | direct gate-bin invocation of semantic internals without suite accounting | maintains reproducible fixture/result accounting |
| `A4 -> A5` | gate bins consume suite summaries, apply reliability budgets, and emit deterministic diagnostics | gate bins mutating suite outcomes to force pass | keeps governance auditable and non-destructive |

### DOC-PASS-10.3 Verification/logging implications by subsystem slice

| layer_id | Unit/Property | Differential | E2E | Logging contract | Current status |
|---|---|---|---|---|---|
| `A1` | strong for shape/stride/dtype primitives | partial (dedicated shape/dtype diff lanes pending) | partial (indirect) | shape/dtype suite logs with normalized fields | medium/open |
| `A2` | strong crate-level for all packet-local kernels | strong for ufunc, partial/open for iter/random/linalg/io | partial/open packet-local e2e debt | packet log-record schemas enforce replay fields | mixed/open |
| `A3` | strong runtime suite + adversarial suite | strong policy differential (fixture-driven) | strong via security/test/workflow gates | runtime ledger/log validation + required field checks | strong |
| `A4` | strong suite-level tests | strong in integrated lanes | strong in workflow scenario suite | `normalize_*` helpers + append logs | strong with optional-path caveat |
| `A5` | strong gate-bin tests | N/A | strong via scripts and gate binaries (`rch exec -- cargo run ...`) | reliability summaries, diagnostics, artifact indices | strong |

### DOC-PASS-10.4 Topology contradiction register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `ARCH-C001` | Topology layering is documented but not yet mechanically enforced in CI import/dependency policy checks. | medium | foundation automation follow-up | add crate-layer policy checks and fail CI on forbidden cross-layer coupling |
| `ARCH-C002` | Packet-local `A2` domains remain unevenly connected into `A4/A5` differential/e2e governance lanes. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | close packet-local F/G lanes and require lane-complete readiness evidence |
| `ARCH-C003` | Optional-path logging in portions of `A4` weakens the observability assumptions of `A5` reliability governance. | medium | `bd-23m.6`, `bd-23m.23` | enforce required logging for gate-critical appenders or gate fail on missing logs |

## DOC-PASS-11 Expansion Draft (Pass B): Behavioral/Risk Synthesis

### DOC-PASS-11.1 Concentrated risk map by behavior family

| domain_id | Contract core | Current strongest evidence lane | Highest residual risk |
|---|---|---|---|
| `BR-001` | shape/stride legality | unit/property + shape-stride conformance suite | full legacy differential/e2e breadth still open |
| `BR-002` | dtype promotion determinism | dtype suite + unit checks | dedicated dtype differential/e2e gate lane missing |
| `BR-003` | runtime strict/hardened fail-closed policy | runtime + adversarial suites + security/test/workflow gates | optional-path logging caveat in some appenders |
| `BR-004` | ufunc parity | oracle differential + metamorphic + adversarial suites | dedicated standalone ufunc e2e gate wrapper absent |
| `BR-005` | iter/random/linalg/io packet-local policies | crate-level unit/property + log-schema validators | packet-local differential + replay lane integration debt |

### DOC-PASS-11.2 Drift sentinel summary

| sentinel_id | Drift signal | Detection anchor | Escalation |
|---|---|---|---|
| `SEN-001` | fail-closed violation | runtime ledger validation + gate diagnostics | hard gate fail and incident bead |
| `SEN-002` | ufunc oracle mismatch | ufunc differential report | fail lane + repro fixture capture |
| `SEN-003` | flake/coverage reliability regression | gate diagnostics (`flake_budget_exceeded`, `coverage_floor_breach`) | fail gate + replay workflow |
| `SEN-004` | replay log field incompleteness | test-contract gate required field checks | fail contract gate until log schema restored |
| `SEN-005` | packet lane incompleteness | crosswalk gap ledger (`GAP-XW-*`) at readiness time | block packet closure |

### DOC-PASS-11.3 Verification/logging implications

| synthesis_area | Unit/Property | Differential | E2E | Logging | Status |
|---|---|---|---|---|---|
| Core semantic/runtime/ufunc synthesis | strong | strong for runtime/ufunc, partial for shape/dtype | strong for runtime/workflow | strong (with optional-path caveat) | medium-strong |
| Packet-local synthesis (iter/random/linalg/io) | strong | missing packet-local differential lanes | missing packet-local replay lanes | strong schema-level only | partial/open |

### DOC-PASS-11.4 Synthesis contradiction register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `SYN-C001` | Packet-local risk concentration is known but lane-complete evidence remains inconsistent at readiness closure. | high | packet F/G owners | enforce lane-complete readiness checklist in gate path |
| `SYN-C002` | Runtime events include expected-loss signals, but gate escalation remains mostly threshold-based. | medium | `bd-23m.8`, `bd-23m.23` | define/implement expected-loss escalation policy for reliability gates |
| `SYN-C003` | Sentinel/drift status is documented but not emitted as a dedicated machine snapshot artifact per run. | medium | docs automation follow-up | emit versioned sentinel snapshot artifact in CI |

## DOC-PASS-12 Independent Red-Team Contradiction and Completeness Review

### DOC-PASS-12.1 Adversarial finding ledger (severity + evidence anchors)

| finding_id | Severity | Finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `RT12-F001` | high | Historical placeholder/stub framing conflicted with current packet-local crate reality. | `DOC-PASS-02.3`, `DOC-PASS-02.4`, `DOC-PASS-02.5`; `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | corrected in current doc revision; contradiction shifted to integration debt |
| `RT12-F002` | high | Packet-local subsystems still lack lane-complete differential and replay/e2e integration at readiness boundaries. | `DOC-PASS-09.3`, `DOC-PASS-09.4`, `DOC-PASS-11.3`; `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/workflow_scenarios.rs` | open; packet F/G/H/I lane closure remains mandatory |
| `RT12-F003` | medium | Optional-path logging in conformance/workflow appenders can undermine replay-complete gate evidence. | `DOC-PASS-10.4` (`ARCH-C003`); `crates/fnp-conformance/src/lib.rs` (`maybe_append_*`), `crates/fnp-conformance/src/workflow_scenarios.rs` (`maybe_append_workflow_log`) | open; require deterministic required-log enforcement for gate-critical paths |
| `RT12-F004` | medium | Expected-loss model is implemented in runtime decisions but not yet first-class in gate escalation policy. | `DOC-PASS-11.3`, `DOC-PASS-11.4`; `crates/fnp-runtime/src/lib.rs`, `crates/fnp-conformance/src/bin/run_*_gate.rs` | open; unify escalation policy around expected-loss + deterministic fallback trigger |
| `RT12-F005` | medium | Contradiction tracking is distributed across pass-local tables without a unified machine artifact. | `DOC-PASS-01.5`, `03.5`, `04.4`, `05.5`, `07.6`, `08.4`, `09.4`, `10.4`, `11.4` | open; generate contradiction/sentinel snapshot artifact per gate cycle |

### DOC-PASS-12.2 Assertion traceability map (legacy anchors and executable evidence)

| assertion_id | Assertion under review | Legacy anchor(s) | Executable evidence anchor(s) | Traceability status |
|---|---|---|---|---|
| `RT12-A001` | Shape/stride legality and broadcast calculus are deterministic under SCE contracts. | `legacy_numpy_code/numpy/_core/src/multiarray/shape.c`, `legacy_numpy_code/numpy/_core/src/multiarray/shape.h` | `crates/fnp-ndarray/src/lib.rs`, `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `crates/fnp-conformance/src/lib.rs` | traceable; broad differential/e2e matrix still incomplete |
| `RT12-A002` | Dtype promotion/cast semantics are deterministic and explicit. | `legacy_numpy_code/numpy/_core/src/multiarray/dtypemeta.c`, `legacy_numpy_code/numpy/_core/src/multiarray/descriptor.c`, `legacy_numpy_code/numpy/_core/src/multiarray/can_cast_table.h` | `crates/fnp-dtype/src/lib.rs`, `crates/fnp-conformance/fixtures/dtype_promotion_cases.json`, `crates/fnp-conformance/src/lib.rs` | traceable; dedicated dtype parity lane remains open |
| `RT12-A003` | Strict/hardened runtime policy is fail-closed on unknown/incompatible semantics with auditable reasons. | clean-room policy extension (no direct legacy equivalent) | `crates/fnp-runtime/src/lib.rs`, `crates/fnp-conformance/fixtures/runtime_policy_cases.json`, `crates/fnp-conformance/fixtures/runtime_policy_adversarial_cases.json`, `crates/fnp-conformance/src/test_contracts.rs` | traceable via executable fixture + contract checks |
| `RT12-A004` | Ufunc parity is checked against oracle outputs with deterministic mismatch reporting. | `legacy_numpy_code/numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/src/bin/run_ufunc_differential.rs` | traceable; standalone ufunc e2e gate artifact remains pending |
| `RT12-A005` | Durability obligations require sidecar/scrub/decode-proof linkage for governed artifacts. | contract-level durability doctrine | `crates/fnp-conformance/src/raptorq_artifacts.rs`, `crates/fnp-conformance/src/bin/run_raptorq_gate.rs`, `artifacts/raptorq/conformance_bundle_v1.sidecar.json`, `artifacts/raptorq/conformance_bundle_v1.scrub_report.json`, `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` | traceable and currently strong |

### DOC-PASS-12.3 Unit/property, differential, e2e, and logging implications

| implication_id | Area | Unit/Property | Differential | E2E/replay | Logging/forensics | Status |
|---|---|---|---|---|---|---|
| `RT12-I001` | packet-local iter/random/linalg/io parity evidence chain | covered at crate scope | missing/partial packet-local differential lanes | missing/partial packet-local replay scenarios | schema checks covered; gate propagation partial | open/high |
| `RT12-I002` | gate-critical log completeness enforcement | partial | N/A | partial (gate runs tolerate optional-path appenders) | missing fail-closed required-log policy | open/medium |
| `RT12-I003` | expected-loss-informed gate escalation | covered in runtime tests | N/A | partial (threshold-led governance dominates) | partial diagnostics, no unified expected-loss artifact | open/medium |
| `RT12-I004` | contradiction/sentinel trend observability | N/A | N/A | deferred pending snapshot generation | missing single machine-readable trend artifact | open/medium |

### DOC-PASS-12.4 Contradiction register with owner/risk/closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `RT12-C001` | Packet-local contract APIs exist, but lane-complete differential + replay evidence is not uniformly present at readiness time. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet readiness reports show closed differential + replay lanes across all packet-local domains |
| `RT12-C002` | Optional-path logging persists in gate-critical flows, weakening replay guarantees. | medium | `bd-23m.6`, `bd-23m.23` | gate-critical appenders become required-path or fail-closed and are enforced by test-contract gate |
| `RT12-C003` | Expected-loss calculus remains runtime-local instead of governing reliability gate escalation policy. | medium | `bd-23m.8`, `bd-23m.23` | gate policy emits expected-loss-based escalation artifact with deterministic fallback trigger |
| `RT12-C004` | No unified contradiction/sentinel machine artifact exists for CI trend and diff auditing. | medium | docs automation follow-up + `bd-23m.23` | publish versioned contradiction/sentinel snapshot artifact each gate cycle and diff it in CI |

## DOC-PASS-14 Full-Agent Deep Dive Pass A (Structure Specialist)

### DOC-PASS-14.1 Canonical topology alias map (cross-doc coherence)

| Canonical layer | Local alias (this doc) | Alias in `EXHAUSTIVE_LEGACY_ANALYSIS.md` | Ownership anchor(s) | Structural contract |
|---|---|---|---|---|
| semantic kernel | `A1` | `L1` | `crates/fnp-dtype/src/lib.rs`, `crates/fnp-ndarray/src/lib.rs` | single source of truth for legality/promotion; execution may not redefine semantics |
| execution kernel set | `A2` | `L2` | `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | consumes semantic outputs and emits typed errors/reason codes |
| runtime policy kernel | `A3` | `L3` | `crates/fnp-runtime/src/lib.rs` | mode/class/risk arbitration (`allow`/`full_validate`/`fail_closed`) with audit evidence |
| conformance orchestrator | `A4` | `L4` | `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/src/workflow_scenarios.rs` | fixture-driven execution/reporting; no semantic mutation |
| reliability/durability envelope | `A5` | `L5` | `crates/fnp-conformance/src/bin/run_*_gate.rs`, `crates/fnp-conformance/src/raptorq_artifacts.rs`, `scripts/e2e/run_*_gate.sh` | governance and durability checks only; cannot rewrite suite outcomes |

### DOC-PASS-14.2 Structure-specialist findings ledger (adversarial)

| finding_id | Severity | Structural finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `ST14-F001` | medium | Two equivalent layer ID vocabularies (`A*` vs `L*`) were previously unmapped, increasing long-term contradiction risk. | `DOC-PASS-10.1` in both target docs | resolved in this pass with explicit alias map |
| `ST14-F002` | high | Dependency-direction law remains documentation-only and is not yet enforced as a machine gate. | `DOC-PASS-10.4` (`ARCH-C001`), `DOC-PASS-01.5` (`TOPO-C002`), root `Cargo.toml` workspace graph | open; requires CI dependency-direction policy checks |
| `ST14-F003` | high | Packet-local `A2/L2` domains are structurally under-integrated into `A4/A5` governance lanes. | `DOC-PASS-09.3`, `DOC-PASS-11.3`, `DOC-PASS-12.3` | open; must close packet-local differential/replay/gate lanes |
| `ST14-F004` | medium | No single structure-level checklist proves all gate paths are `rch`-offloaded and replay-deterministic. | `DOC-PASS-10.3`, `scripts/e2e/run_*_gate.sh`, `crates/fnp-conformance/src/bin/run_*_gate.rs` | open; add explicit gate topology checklist artifact |

### DOC-PASS-14.3 Boundary-law coherence audit

| boundary_id | Canonical boundary law | Evidence of enforcement today | Gap | Owner |
|---|---|---|---|---|
| `ST14-B001` | semantic kernel -> execution kernel only (`A1 -> A2`) | crate API boundaries and typed error contracts | no automated forbidden-edge workspace check | foundation automation follow-up |
| `ST14-B002` | execution/semantic -> runtime policy (`A2/A1 -> A3`) | runtime decision API + policy fixture suites | packet-local branches not uniformly represented in gate scenarios | packet F/G owners |
| `ST14-B003` | runtime/execution/semantic -> conformance (`A3/A2/A1 -> A4`) | conformance orchestrator and fixture contracts | lane completeness differs by packet domain | packet + conformance owners |
| `ST14-B004` | conformance -> reliability envelope (`A4 -> A5`) | gate bins consume suite reports and publish diagnostics | missing unified topology checklist artifact for invocation/replay linkage | `bd-23m.10`, `bd-23m.23` |

### DOC-PASS-14.4 Unit/property, differential, e2e, and logging implications (structure lens)

| implication_id | Structure concern | Unit/Property | Differential | E2E/replay | Logging/forensics | Status |
|---|---|---|---|---|---|---|
| `ST14-I001` | alias-map drift prevention (`A*`/`L*`) | covered by canonical map | N/A | N/A | N/A | resolved |
| `ST14-I002` | dependency-direction enforcement | missing automated structural test | N/A | partial indirect detection | partial diagnostics only | open/high |
| `ST14-I003` | packet-local topology integration (`A2/L2 -> A4/A5`) | covered at crate level | missing/partial packet-local lanes | missing/partial packet-local workflow replay | partial schema coverage, incomplete gate propagation | open/high |
| `ST14-I004` | gate invocation topology determinism (`rch` + replay contract) | partial gate-bin test coverage | N/A | partial scripted wrappers, no single checklist | partial diagnostics without topology checklist artifact | open/medium |

### DOC-PASS-14.5 Structure contradiction register with closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `ST14-C001` | Equivalent layer IDs across docs were not explicitly mapped, risking cross-pass reference drift. | medium | `bd-23m.24.15` | canonical alias table exists in both docs and is retained in later passes |
| `ST14-C002` | Dependency-direction law is prose-only and can drift from real workspace graph edges. | high | `bd-23m.23` + docs automation follow-up | gate emits and enforces dependency-edge policy checks |
| `ST14-C003` | Packet-local topology ownership is declared but lane completeness is not yet uniformly proven at gate level. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet-local differential + replay lanes are required and green in readiness outputs |
| `ST14-C004` | Missing unified gate topology artifact linking `rch` invocation path, replay contract, and diagnostics references. | medium | `bd-23m.10`, `bd-23m.23` | publish and enforce a gate topology checklist artifact in CI outputs |

## DOC-PASS-15 Full-Agent Deep Dive Pass B (Behavior Specialist)

### DOC-PASS-15.1 Behavior findings ledger (adversarial)

| finding_id | Severity | Behavioral finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `BH15-F001` | high | Runtime policy computes action selection (`allow`/`full_validate`/`fail_closed`) using risk thresholding, while packet-local policy validators currently enforce metadata token legality only; these behaviors must remain explicitly separated in docs. | `crates/fnp-runtime/src/lib.rs` (`decide_compatibility`), `crates/fnp-io/src/lib.rs` (`validate_io_policy_metadata`), `crates/fnp-linalg/src/lib.rs` (`validate_policy_metadata`) | resolved in doc language; implementation integration remains open |
| `BH15-F002` | medium | Deterministic invariant coverage is strong in unit/property lanes, but confidence remains uneven without closed packet-local differential/replay lanes. | `crates/fnp-ndarray/src/lib.rs`, `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`; `DOC-PASS-12.3` | open; packet-local F/G lane closure required |
| `BH15-F003` | medium | Packet reason-code registries are stable locally, but no global ontology artifact guarantees cross-packet semantic normalization. | reason-code registries in packet crates; `crates/fnp-conformance/src/test_contracts.rs` | open; add ontology artifact + gate policy checks |
| `BH15-F004` | low | Unknown/incompatible metadata handling is consistently fail-closed across runtime and packet-local validators. | runtime + `fnp-linalg` + `fnp-io` policy metadata validators | confirmed/retained |

### DOC-PASS-15.2 Behavior invariant traceability map

| assertion_id | Behavioral assertion | Legacy anchor(s) | Executable evidence anchor(s) | Status |
|---|---|---|---|---|
| `BH15-A001` | Shape/broadcast/reshape semantics are deterministic with overflow/invalid-shape rejection. | `legacy_numpy_code/numpy/_core/src/multiarray/shape.c`, `legacy_numpy_code/numpy/_core/src/multiarray/shape.h` | `crates/fnp-ndarray/src/lib.rs` (`broadcast_shape`, `fix_unknown_dimension`, overflow tests) | traceable; broad parity matrix still expanding |
| `BH15-A002` | Ufunc behavior is deterministic across elementwise/reduction paths with stable error vocabularies. | `legacy_numpy_code/numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-ufunc/src/lib.rs` property grids + reason-code tests, `crates/fnp-conformance/src/ufunc_differential.rs` | traceable; standalone ufunc e2e gate artifact pending |
| `BH15-A003` | Transfer/flatiter/overlap semantics are deterministic and policy-checked. | `legacy_numpy_code/numpy/_core/src/multiarray/dtype_transfer.c`, `legacy_numpy_code/numpy/_core/src/multiarray/nditer*` | `crates/fnp-iter/src/lib.rs` selector/overlap/flags tests + replay-log checks | traceable; packet-local differential/replay integration pending |
| `BH15-A004` | RNG seed/state/jump contracts produce deterministic reproducible streams. | `legacy_numpy_code/numpy/random/src/*`, `legacy_numpy_code/numpy/random/*.pyx` | `crates/fnp-random/src/lib.rs` deterministic stream/state tests + replay-log checks | traceable; packet-local differential/replay integration pending |
| `BH15-A005` | Runtime strict/hardened routing is fail-closed on unknown/incompatible classes with expected-loss evidence fields. | policy extension (clean-room) | `crates/fnp-runtime/src/lib.rs`, `runtime_policy_cases.json`, `runtime_policy_adversarial_cases.json` | traceable; gate-level expected-loss escalation policy remains open |

### DOC-PASS-15.3 Strict vs hardened edge-case coherence matrix

| behavior_surface | Strict mode expectation | Hardened mode expectation | Evidence anchors | Coherence status |
|---|---|---|---|---|
| known-compatible low-risk class | `allow` | `allow` | runtime `decide_compatibility` | coherent |
| known-compatible high-risk class | `allow` | `full_validate` when threshold exceeded | runtime threshold logic | coherent at runtime; docs now explicitly separate packet-local validator scope |
| known-incompatible / unknown class | `fail_closed` | `fail_closed` | runtime + packet-local metadata validators | coherent |
| packet-local metadata token validation | known mode/class tokens accepted, unknown rejected | same | `validate_policy_metadata` / `validate_io_policy_metadata` | coherent; intentionally narrower than runtime routing |

### DOC-PASS-15.4 Unit/property, differential, e2e, and logging implications (behavior lens)

| implication_id | Behavior concern | Unit/Property | Differential | E2E/replay | Logging/forensics | Status |
|---|---|---|---|---|---|---|
| `BH15-I001` | runtime-vs-packet policy boundary clarity | covered by runtime + packet-local tests | partial (runtime strong, packet-local action-routing differential absent) | partial workflow coverage | strong reason-code/log-field checks | open/medium |
| `BH15-I002` | packet-local behavior parity confidence | strong crate-level invariants | missing/partial packet-local lanes | missing/partial packet-local replay scenarios | schema-level checks present | open/high |
| `BH15-I003` | cross-packet reason-code semantic normalization | covered packet-local registry checks | N/A | N/A | partial (presence checks, no ontology-level semantics) | open/medium |
| `BH15-I004` | fail-closed unknown/incompatible handling | strong runtime + packet-local tests | strong runtime policy fixture evidence | strong in security/workflow gate paths | strong with optional-path caveat | medium-strong |

### DOC-PASS-15.5 Behavior contradiction register with closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `BH15-C001` | Potential conflation of runtime action-routing behavior with packet-local metadata validator behavior. | high | `bd-23m.24.16` + docs automation follow-up | final integration pass preserves explicit layer separation language |
| `BH15-C002` | Behavior invariants are unit/property-strong but lack uniform packet-local differential/replay proof closure. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet-local differential + replay lanes are green and referenced in readiness outputs |
| `BH15-C003` | Missing global reason-code ontology artifact across packet registries. | medium | `bd-23m.23` + test-contract follow-up | publish ontology artifact and enforce mapping checks in gate outputs |
| `BH15-C004` | Expected-loss values are emitted by runtime but not yet first-class in reliability escalation policy. | medium | `bd-23m.8`, `bd-23m.23` | gate policy consumes expected-loss artifact with deterministic fallback trigger |

## DOC-PASS-16 Full-Agent Deep Dive Pass C (Risk/Perf/Test Specialist)

### DOC-PASS-16.1 Risk/perf/test findings ledger (adversarial)

| finding_id | Severity | Finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `RPT16-F001` | high | Reliability gates expose explicit controls (`retries`, `flake_budget`, `coverage_floor`, diagnostics), but packet-local differential/replay incompleteness still drives the largest residual risk. | `run_security_gate.rs`, `run_test_contract_gate.rs`, `run_workflow_scenario_gate.rs`; prior pass gap ledgers | open; packet-local F/G lane closure required |
| `RPT16-F002` | medium | Runtime expected-loss evidence exists, but gate escalation policy does not yet consume expected-loss as a first-class signal. | `crates/fnp-runtime/src/lib.rs`, gate bins in `crates/fnp-conformance/src/bin/run_*_gate.rs` | open |
| `RPT16-F003` | medium | Performance evidence chain is well-artifacted (baseline/profile/isomorphism/rollback trigger), but EV gate automation remains pending. | `crates/fnp-conformance/src/benchmark.rs`, `artifacts/optimization/ROUND3_OPPORTUNITY_MATRIX.md`, `artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md` | open |
| `RPT16-F004` | medium | Logging contracts enforce replay-critical fields strongly, yet optional-path appenders still create known observability caveats. | `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/test_contracts.rs` | open |
| `RPT16-F005` | low | Durability controls (sidecar/scrub/decode proof + staleness checks) are executable and robust. | `crates/fnp-conformance/src/raptorq_artifacts.rs`, `run_raptorq_gate.rs` | confirmed/retained |

### DOC-PASS-16.2 Coverage and gate topology matrix

| lane_id | Risk/test lane | Primary executors | Current strength | Dominant gap | Owner bead(s) |
|---|---|---|---|---|---|
| `RPT16-L001` | Unit/property invariants | packet crate tests | strong | cross-packet ontology + CI synthesis | `bd-23m.23` |
| `RPT16-L002` | Differential/metamorphic/adversarial | conformance suites (`run_ufunc_differential`, etc.) | strong for integrated lanes | packet-local differential lane incompleteness | `bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6` |
| `RPT16-L003` | E2E workflow/replay | workflow gate + e2e scripts | medium-strong | packet-local replay incompleteness | `bd-23m.14.7`, `bd-23m.18.7`, `bd-23m.19.7`, `bd-23m.20.7` |
| `RPT16-L004` | Reliability governance | security/test/workflow gate bins | strong | expected-loss policy integration missing | `bd-23m.8`, `bd-23m.23` |
| `RPT16-L005` | Durability integrity | RaptorQ suite + gate | strong | no critical blocker; freshness discipline ongoing | `bd-23m.21`, `bd-23m.22` |

### DOC-PASS-16.3 Performance governance evidence chain

| stage_id | Governance stage | Evidence anchor(s) | Enforced invariant | Residual risk |
|---|---|---|---|---|
| `RPT16-P001` | baseline telemetry | `benchmark.rs`, `artifacts/baselines/ufunc_benchmark_baseline*.json` | p50/p95/p99 + throughput/bandwidth reproducibly captured | workload scope still selective |
| `RPT16-P002` | hotspot profiling fallback | `artifacts/optimization/ROUND3_OPPORTUNITY_MATRIX.md` | profiling fallback documented when `perf` unavailable | lower-fidelity attribution under fallback |
| `RPT16-P003` | isomorphism proof chain | `artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md`, `golden_checksums_round3.txt` | behavior equivalence required for perf levers | not yet single-step gate-automated |
| `RPT16-P004` | rollback trigger governance | `ISOMORPHISM_PROOF_ROUND3.md` | sustained p99 regression rollback condition documented | policy/manual enforcement today |

### DOC-PASS-16.4 Logging and durability contract checklist

| checklist_id | Contract requirement | Evidence anchor(s) | Status |
|---|---|---|---|
| `RPT16-K001` | replay-critical fields are mandatory (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) | `test_contracts.rs`, `lib.rs` normalization and validations | strong |
| `RPT16-K002` | workflow logs must cover scenario IDs and linked references | `run_workflow_scenario_gate.rs` (`validate_workflow_log_coverage`) | strong |
| `RPT16-K003` | durability bundles require sidecar + scrub + decode proof, schema-valid and hash-consistent | `raptorq_artifacts.rs`, `run_raptorq_gate.rs` | strong |
| `RPT16-K004` | stale durability artifacts are detected and rejected | `raptorq_artifacts.rs` mtime/generated_at checks | strong |
| `RPT16-K005` | gate-critical appenders should be fail-closed when logging cannot be written | prior contradiction registers (`ARCH-C003`, `RT12-C002`) | open |

### DOC-PASS-16.5 Risk/perf/test contradiction register with closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `RPT16-C001` | Gate reliability is mature, but packet-local differential/replay lane incompleteness remains the dominant readiness blocker. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet-local F/G lanes are green and referenced in readiness artifacts |
| `RPT16-C002` | Expected-loss stays runtime-local; gate escalation remains threshold-led. | medium | `bd-23m.8`, `bd-23m.23` | gate diagnostics include expected-loss artifact + fallback trigger |
| `RPT16-C003` | Performance governance lacks machine-enforced EV + rollback automation despite strong artifact chain. | medium | `bd-23m.8`, `bd-23m.10` | CI enforces EV and rollback conditions |
| `RPT16-C004` | Optional-path gate appenders still permit observability degradation. | medium | `bd-23m.6`, `bd-23m.23` | required-log policy enforced fail-closed for gate-critical appenders |

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

## 6. Parity-Debt Sequencing Ledger (Replaces Historical V1 Scope-Cut Framing)

This program does not accept reduced-scope V1 completion. Remaining gaps are tracked as explicit parity debt with owners and closure gates.

| Domain | Current implementation state | Remaining parity debt | Owner packet |
|---|---|---|---|
| Shape/stride + ufunc + runtime policy | first-wave core implemented and gate-backed | broaden fixture/oracle coverage and close packet-local F/G/H/I lanes | `bd-23m.12`, `bd-23m.16`, `bd-23m.17` |
| Transfer / NDIter semantics | packet-local transfer/index contracts implemented | integrate full NDIter traversal behavior and gate-backed differential/replay evidence | `bd-23m.14`, `bd-23m.17` |
| RNG | deterministic stream/state contracts implemented | oracle differential + workflow replay + final evidence pack | `bd-23m.18` |
| Linalg | shape/solver policy contracts implemented | differential/replay/optimization proof chain to packet closure | `bd-23m.19` |
| IO | NPY/NPZ boundary validation contracts implemented | round-trip/adversarial differential and full gate-integrated evidence chain | `bd-23m.20` |
| Non-core ecosystem breadth | not release-cut criteria | remains explicit parity debt backlog until full drop-in behavior matrix reaches zero | master program `bd-23m` |

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
