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
| Transfer/alias semantics | `dtype_transfer.c`, `lowlevel_strided_loops.c.src` | 2 | 4.0x | partial (packet-local unit/property coverage exists) | missing overlap differential | missing overlap journey | partial (`TransferLogRecord` schema/reason codes) | `crates/fnp-iter/src/lib.rs`, `artifacts/phase2c/FNP-P2C-003/*` |
| Ufunc dispatch/override | `umath/ufunc_object.c`, dispatch loops | 3 | 3.0x | partial (metamorphic/adversarial corpus) | partial (`ufunc_differential`) | missing override journey | partial (`fixture_id`,`seed` in fixtures) | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/fixtures/ufunc_*` |
| NDIter traversal | `multiarray/nditer*` | 2 | 4.0x | partial (transfer/flatiter contract tests exist) | missing | missing | partial (transfer reason-code/log schema checks) | `crates/fnp-iter/src/lib.rs`, `artifacts/phase2c/FNP-P2C-006/*` |
| Random subsystem | `numpy/random/*.pyx`, `random/src/*` | 2 | 3.0x | covered deterministic stream/state/bounded contracts | missing RNG differential | missing seed/state journey | partial (`RandomLogRecord` schema/reason codes) | `crates/fnp-random/src/lib.rs`, `artifacts/phase2c/FNP-P2C-007/*` |
| Linalg subsystem | `numpy/linalg/lapack_lite/*` | 2 | 3.0x | covered solver/shape/policy contracts | missing solver differential | missing solver journey | partial (`LinAlgLogRecord` schema/reason codes) | `crates/fnp-linalg/src/lib.rs`, `artifacts/phase2c/FNP-P2C-008/*` |
| IO subsystem | `numpy/lib/format.py`, npy/npz paths | 2 | 3.0x | covered parser/budget/policy contract tests | missing io round-trip differential | missing io journey | partial (`IOLogRecord` schema/reason codes) | `crates/fnp-io/src/lib.rs`, `artifacts/phase2c/FNP-P2C-009/*` |

Contradictions/unknowns register (must be closed during doc-overhaul passes):

| ID | Contradiction or unknown | Source/evidence anchor | Owner | Risk | Closure criteria |
|---|---|---|---|---|---|
| `DOC-C001` | `EXISTING_NUMPY_STRUCTURE.md` still states a reduced-scope “V1 extraction boundary” which conflicts with full drop-in parity doctrine. | `EXISTING_NUMPY_STRUCTURE.md` section 6 vs `COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md` sections 0 and 2 | `bd-23m.24.2` | critical | Replace scope-cut language with parity-debt sequencing language and explicit ownership table. |
| `DOC-C002` | Verification/logging implication mapping is now documented in DOC-PASS-03, but multiple packet domains remain missing/deferred. | `EXHAUSTIVE_LEGACY_ANALYSIS.md` DOC-PASS-03.4 + `EXISTING_NUMPY_STRUCTURE.md` DOC-PASS-03.4 | `bd-23m.24.4` | high | Keep pass-03 matrices synchronized with packet execution until unresolved domains are implementation-backed and evidence-complete. |
| `DOC-C003` | Packet ownership crosswalk is documented and packet-local crates are implemented, but those domains remain under-integrated in conformance differential/e2e gate lanes. | `EXHAUSTIVE_LEGACY_ANALYSIS.md` DOC-PASS-02.4 + packet-local crates (`fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) + `run_all_core_suites` | `bd-23m.24.3` | high | Keep crosswalk synchronized with packet execution and close only when packet-local F/G lanes are gate-integrated with complete evidence artifacts. |
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
| `fnp-iter` | `crates/fnp-iter/src/lib.rs` (`TransferSelectorInput`, `select_transfer_class`, `overlap_copy_policy`, `validate_nditer_flags`, `FlatIterIndex`, `TransferLogRecord`) | `numpy/_core/src/multiarray/dtype_transfer.c`, `lowlevel_strided_loops.c.src`, `numpy/_core/src/multiarray/nditer*` | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-003/fixture_manifest.json`, `parity_gate.yaml`) | Owns transfer selector/overlap policy/flatiter contract checks and packet-003 reason-code taxonomy; packet-local differential/e2e gates are still pending. |
| `fnp-random` | `crates/fnp-random/src/lib.rs` (`DeterministicRng`, `RandomError`, `RandomLogRecord`, bounded/float/state APIs) | `numpy/random/_generator.pyx`, `numpy/random/src/*` | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-007/fixture_manifest.json`, `parity_gate.yaml`) | Owns deterministic stream/state constructor contracts and replay-log schema for packet-007 first wave; oracle differential and workflow lanes are still pending. |
| `fnp-linalg` | `crates/fnp-linalg/src/lib.rs` (`LinAlgError`, `QrMode`, `solve_2x2`, `svd_output_shapes`, `lstsq_output_shapes`, policy validators, `LinAlgLogRecord`) | `numpy/linalg/lapack_lite/*`, high-level wrappers in `numpy/linalg/*` | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-008/contract_table.md`, `unit_property_evidence.json`) | Owns linalg shape/solver/mode/tolerance/backend fail-closed contracts for packet-008 first wave; broader parity matrix and differential/e2e lanes remain open. |
| `fnp-io` | `crates/fnp-io/src/lib.rs` (`NpyHeader`, `IOError`, magic/header/descriptor/memmap/load/npz validators, `IOLogRecord`) | `numpy/lib/format.py`, npy/npz IO paths | crate unit/property tests + packet refs (`artifacts/phase2c/FNP-P2C-009/contract_table.md`, `unit_property_evidence.json`) | Owns NPY/NPZ contract validation, bounded parser budgets, and fail-closed metadata policy for packet-009 first wave; full oracle/e2e parity lanes remain open. |

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
| `crates/fnp-iter/src/lib.rs` (`select_transfer_class`, `overlap_copy_policy`, flatiter validators) | Transfer/index hotspot | Encodes packet-003 transfer legality and overlap/broadcast policy checks that future NDIter and stride-tricks paths depend on. |
| `crates/fnp-random/src/lib.rs` (`DeterministicRng::{next_u64,next_f64,bounded_u64,state}`) | RNG hotspot | Encodes deterministic-seed/state semantics and bounded-generation policy for packet-007. |
| `crates/fnp-linalg/src/lib.rs` (shape/solver/qr/svd/lstsq/policy validators) | Linalg contract hotspot | Captures packet-008 fail-closed shape and backend policy semantics that gate solver behavior. |
| `crates/fnp-io/src/lib.rs` (magic/header/descriptor/memmap/load/npz validators) | IO security hotspot | Captures packet-009 parser-budget and contract checks for NPY/NPZ boundary hardening. |
| `crates/fnp-conformance/src/bin/*.rs` gate binaries | Workflow orchestration layer | Operational entrypoints (`run_security_gate`, `run_test_contract_gate`, `run_workflow_scenario_gate`, etc.) that decide what evidence actually ships in CI/gate runs. |

### DOC-PASS-01.4 Verification implications by ownership boundary (covered/missing/deferred)

| Ownership boundary | Unit/Property | Differential | E2E | Structured logging | Status + owner |
|---|---|---|---|---|---|
| `fnp-ndarray` shape/stride legality | covered (crate tests + shape/stride fixtures) | covered (indirect via ufunc differential shape checks), still partial | deferred to packet scenario layers | partial (runtime/workflow logs exist; shape-specific reason taxonomy incomplete) | partial; packet owner `bd-23m.12` |
| `fnp-dtype` promotion/cast primitives | covered (crate tests + promotion fixture suite) | missing full cast-matrix differential | missing packet-level journey | missing per-cast reason-code taxonomy | missing/partial; packet owner `bd-23m.13` |
| `fnp-ufunc` scoped ops (`add/sub/mul/div/sum`) | covered (crate tests + metamorphic/adversarial suites) | covered for scoped fixtures; broader NumPy surface missing | deferred (workflow corpus links exist, packet-local journeys incomplete) | covered at fixture/scenario level | partial; packet owner `bd-23m.17` + `bd-23m.19` |
| `fnp-runtime` strict/hardened policy | covered (crate tests + policy suites) | not applicable as numerical diff; policy-wire adversarial coverage is covered | covered via workflow scenario suite | covered (`fixture_id`,`seed`,`mode`,`env_fingerprint`,`artifact_refs`,`reason_code`) | covered for current surface; continuing under foundation beads `bd-23m.6`, `bd-23m.23` |
| `fnp-iter` transfer/index contracts | covered (unit + property grids in crate tests) | missing packet-local oracle differential lane | missing packet-local journey | covered at record-schema level; not yet gate-enforced | partial; packet owners `bd-23m.14` + `bd-23m.17` |
| `fnp-random` deterministic stream/state | covered (determinism/state/bounded/range tests) | missing dedicated RNG oracle differential lane | missing seed/state replay journey | covered by `RandomLogRecord` schema tests; not yet gate-enforced | partial; packet owner `bd-23m.18` |
| `fnp-linalg` solver/shape/backend contracts | covered (unit/property contract tests) | missing dedicated linalg oracle differential lane | missing solver workflow replay journey | covered by `LinAlgLogRecord` schema checks; not yet gate-enforced | partial; packet owner `bd-23m.19` |
| `fnp-io` NPY/NPZ parser/writer contracts | covered (unit/property parser/budget/policy tests) | missing dedicated IO round-trip differential lane | missing IO replay journey | covered by `IOLogRecord` schema checks; not yet gate-enforced | partial; packet owner `bd-23m.20` |

### DOC-PASS-01.5 Contradictions and unknowns register (topology-specific)

| ID | Contradiction / unknown | Evidence anchors | Risk | Owner | Closure criteria |
|---|---|---|---|---|---|
| `TOPO-C001` | Packet-D boundaries are implemented for `fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`, but `run_all_core_suites` does not yet execute packet-local F/G/H/I lanes for those domains, so topology ownership is only partially exercised in gates. | `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs`, `crates/fnp-conformance/src/lib.rs` (`run_all_core_suites`) | high | `bd-23m.17`, `bd-23m.18`, `bd-23m.19`, `bd-23m.20` | Add packet-local differential/e2e/gate wiring (F/G) and include those suites in operational gate paths before claiming full boundary closure. |
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
| `fnp-iter` | transfer/index APIs (`TransferSelectorInput`, `TransferError`, `select_transfer_class`, overlap/flatiter validators, `TransferLogRecord`) (`crates/fnp-iter/src/lib.rs`) | Packet-local first-wave API (Tier C) | `dtype_transfer.c`/`nditer*` transfer and overlap policy families |
| `fnp-random` | `DeterministicRng`, `RandomError`, `RandomLogRecord`, stream/state/bounded sampling APIs (`crates/fnp-random/src/lib.rs`) | Packet-local first-wave API (Tier C) | `numpy/random/_generator.pyx` constructor/stream semantics |
| `fnp-linalg` | `LinAlgError`, `QrMode`, solver/shape/policy validators, `LinAlgLogRecord` (`crates/fnp-linalg/src/lib.rs`) | Packet-local first-wave API (Tier C) | `numpy/linalg/lapack_lite/*` and wrapper behavior contracts |
| `fnp-io` | `NpyHeader`, `IOError`, NPY/NPZ boundary validators, `IOLogRecord` (`crates/fnp-io/src/lib.rs`) | Packet-local first-wave API (Tier C) | `numpy/lib/format.py` and npy/npz dispatch/parser contracts |

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
| Tier C | Packet-local first-wave boundaries with incomplete gate integration | `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io` public contract APIs | High: APIs exist, but full packet-level differential/e2e/optimization evidence is incomplete, so production parity claims remain invalid. |

### DOC-PASS-02.4 Packet ownership index for unresolved behavior domains (`DOC-C003`)

| Behavior domain | Placeholder surface now | Packet owner | Required closure gates (minimum) | Evidence anchors |
|---|---|---|---|---|
| NDIter traversal/index semantics | transfer/index boundary APIs in `fnp-iter` (selector, overlap, flatiter validators, logs) | `bd-23m.17` / `FNP-P2C-006` | Extend from packet-D boundary into packet-F/G oracle + replay lanes and integrate into conformance gate workflows | `artifacts/phase2c/FNP-P2C-006/*`, `crates/fnp-iter/src/lib.rs` |
| RNG streams + state schema | deterministic RNG/state APIs in `fnp-random` | `bd-23m.18` / `FNP-P2C-007` | Add packet-F/G oracle differential + replay suites and gate wiring; keep deterministic-seed witness artifacts mandatory | `artifacts/phase2c/FNP-P2C-007/*`, `crates/fnp-random/src/lib.rs` |
| Linalg adapters/solver contracts | shape/solver/policy contracts in `fnp-linalg` | `bd-23m.19` / `FNP-P2C-008` | Add packet-F/G linalg differential + replay lanes and packet-H optimization/isomorphism evidence | `artifacts/phase2c/FNP-P2C-008/*`, `crates/fnp-linalg/src/lib.rs` |
| NPY/NPZ parser-writer contracts | parser/writer boundary validators in `fnp-io` | `bd-23m.20` / `FNP-P2C-009` | Add packet-F/G IO round-trip/adversarial lanes and gate-enforced logging/durability links | `artifacts/phase2c/FNP-P2C-009/*`, `crates/fnp-io/src/lib.rs` |

### DOC-PASS-02.5 Verification implications by symbol family

| Symbol family | Unit/Property | Differential | E2E | Structured logging | Assessment |
|---|---|---|---|---|---|
| Shape/stride symbols (`fnp-ndarray`) | covered (crate tests + `shape_stride_cases.json`) | partial through ufunc differential shape validation | deferred packet-local scenarios | partial | parity debt remains for richer alias/transfer semantics |
| Dtype symbols (`fnp-dtype`) | covered for promotion/cast primitives | partial (promotion diff exists, full cast matrix absent) | deferred | missing cast taxonomy | medium risk |
| Ufunc symbols (`fnp-ufunc`) | covered for scoped ops | covered for scoped fixture corpus | partial via workflow suite | covered at fixture/scenario level | medium risk |
| Runtime policy symbols (`fnp-runtime`) | covered (unit + policy fixtures) | adversarial wire-class differential equivalent covered | covered in workflow scenarios | covered with required fields | low-medium risk |
| Verification/tooling symbols (`fnp-conformance`) | covered by module tests and gate execution | N/A | covered by gate binaries | covered by contract suites | medium operational risk (schema evolution) |
| Packet-local first-wave symbols (`fnp-iter`/`fnp-random`/`fnp-linalg`/`fnp-io`) | covered (crate unit/property contract suites) | missing packet-local oracle differential lanes | missing packet-local workflow replay lanes | covered at record-schema level; not yet enforced in gate outputs | high parity debt (integration/evidence gap) |

### DOC-PASS-02.6 Symbol-layer contradictions and unknowns

| ID | Contradiction/unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `SYM-C001` | Tier-C packet-local APIs now exist for iter/random/linalg/io, but gate orchestration still under-represents them compared with ufunc/runtime lanes. | high | `bd-23m.17`/`18`/`19`/`20` | Add packet-local suites to conformance gate execution paths and require parity artifacts before promoting these APIs beyond packet scope. |
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
| Packet-local first-wave ownership models | transfer/index contracts (`fnp-iter`), deterministic RNG/state (`fnp-random`), linalg shape/solver policies (`fnp-linalg`), NPY/NPZ boundary validators (`fnp-io`) | domain-specific contract models with reason-code vocabularies and replay-log records; not yet fully gate-integrated | packet owners `bd-23m.17`/`18`/`19`/`20` |

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
| Packet-local first-wave crates | contract-scoped APIs with partial gate integration | explicit packet ownership + contradiction/gap ledgers | overestimating readiness before differential/e2e lanes are integrated |

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
| `INV-OWNERSHIP-001` | unresolved domains must have lane-complete packet evidence (unit/property + differential + e2e + logging) before parity claims | partial (models exist; packet-local differential/e2e lanes remain open) | packet owners must close F/G lanes and include evidence in readiness closure artifacts |

### DOC-PASS-03.5 Data-model contradictions and closure backlog

| ID | Contradiction/unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `MODEL-C001` | `NdLayout` exposes public fields, so post-construction mutation can violate assumptions outside constructor checks. | medium | `bd-23m.12` | Decide encapsulation policy (public-by-contract vs encapsulated mutators) and enforce consistently. |
| `MODEL-C002` | Runtime/workflow logging optionality can weaken forensic guarantees when not running gates. | medium | `bd-23m.6` | Require explicit log path and gate-level assertion of produced log artifacts in orchestrator flows. |
| `MODEL-C003` | Packet-local state models exist for iter/random/linalg/io, but those models are not yet fully exercised by packet-local differential and workflow replay gates. | high | `bd-23m.17`/`18`/`19`/`20` | Wire packet-local differential/e2e suites into gates and require packet-level readiness artifacts before treating models as parity-complete. |
| `MODEL-C004` | Cast-matrix and transfer/alias invariants are not yet represented as first-class state contracts in code. | high | `bd-23m.13`/`bd-23m.14` | Add explicit state/invariant tables and executable fixtures for cast matrix and overlap/alias transitions. |

## DOC-PASS-04 Execution-Path Tracing and Control-Flow Narratives

### DOC-PASS-04.1 Canonical execution workflows (entry -> branch -> terminal)

| Workflow ID | Entry anchor | Primary branch points (ordered) | Terminal outcomes | Legacy/evidence anchors | Ownership |
|---|---|---|---|---|---|
| `EP-001` ufunc differential | `crates/fnp-conformance/src/bin/run_ufunc_differential.rs` | (1) fixture/oracle load success; (2) dtype parse success; (3) `UFuncArray::new` shape/value gate; (4) op dispatch (`add/sub/mul/div/sum`) | pass report / mismatch report / fixture parse failure | `crates/fnp-conformance/src/ufunc_differential.rs`, `fixtures/ufunc_input_cases.json`, `fixtures/oracle_outputs/ufunc_oracle_output.json` | `bd-23m.16`, `bd-23m.17` |
| `EP-002` security gate | `crates/fnp-conformance/src/bin/run_security_gate.rs` | (1) runtime policy suite; (2) adversarial policy suite; (3) security contract suite; (4) gate aggregate pass/fail | gate pass / gate fail with reason taxonomy | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`, `security_contracts::run_security_contract_suite` | foundation `bd-23m.6`, packet owners |
| `EP-003` workflow scenario replay | `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` | (1) scenario fixture parse; (2) strict/hardened mode execution; (3) expectation comparison; (4) scenario log append | scenario pass / deterministic failure record | `workflow_scenarios::run_user_workflow_scenario_suite`, `artifacts/logs/workflow_scenario_e2e_*.jsonl` | foundation + packet G beads |
| `EP-004` packet readiness contract | `crates/fnp-conformance/src/bin/validate_phase2c_packet.rs` | (1) required file presence; (2) schema token checks; (3) JSON/YAML parse checks; (4) readiness summarization | ready / non-ready with missing-field diagnostics | `contract_schema::validate_phase2c_packet`, `packet_readiness_report.json` | packet I beads |
| `EP-005` IO load dispatch | `crates/fnp-io/src/lib.rs` (`classify_load_dispatch`) | (1) NPZ magic check; (2) NPY magic check; (3) pickle branch only if `allow_pickle`; else fail-closed | `LoadDispatch::{Npz,Npy,Pickle}` / `IOError::LoadDispatchInvalid` | `numpy/lib/format.py`, `IO_PACKET_REASON_CODES` | `bd-23m.20` |
| `EP-006` RNG bounded generation | `crates/fnp-random/src/lib.rs` (`bounded_u64`) | (1) upper-bound non-zero; (2) rejection-threshold loop; (3) modulo projection | bounded sample / `RandomError::InvalidUpperBound` | `numpy/random/_generator.pyx`, `RANDOM_PACKET_REASON_CODES` | `bd-23m.18` |
| `EP-007` linalg branch family | `crates/fnp-linalg/src/lib.rs` (`qr_output_shapes`, `svd_output_shapes`, `validate_policy_metadata`) | (1) rank/shape preflight; (2) mode token parse; (3) convergence gate; (4) fail-closed unknown metadata | shape tuple outputs / `LinAlgError::*` fail-closed terminals | `numpy/linalg/lapack_lite/*`, `LINALG_PACKET_REASON_CODES` | `bd-23m.19` |
| `EP-008` transfer selector and overlap policy | `crates/fnp-iter/src/lib.rs` (`select_transfer_class`, `overlap_copy_policy`, flatiter validators) | (1) context sanity checks; (2) stride-multiple checks; (3) overlap direction choice; (4) read/write arity checks | transfer class + copy action / `TransferError::*` | `dtype_transfer.c`, `lowlevel_strided_loops.c.src`, `TRANSFER_PACKET_REASON_CODES` | `bd-23m.14`, `bd-23m.17` |

### DOC-PASS-04.2 Branch ordering and fallback law (deterministic precedence)

| Branch family | Deterministic ordering rule | Fallback / fail-closed behavior | Why ordering matters |
|---|---|---|---|
| IO payload dispatch (`EP-005`) | Evaluate NPZ magic first, then NPY magic, then pickle (only with explicit allow), else reject | Unknown/forbidden prefixes always map to `io_load_dispatch_invalid` | Prevent ambiguous decode paths and policy bypass via permissive probing. |
| Runtime mode/class decode (`EP-002`, `EP-003`) | Normalize wire mode/class; unknown tokens do not branch into permissive defaults | Unknown metadata maps to fail-closed action and logged reason code | Preserves strict/hardened doctrine under malformed metadata. |
| Ufunc case execution (`EP-001`) | Validate fixture + shape/value consistency before numeric op dispatch | Any precondition failure short-circuits with deterministic mismatch/error output | Avoids undefined behavior from malformed fixtures and preserves oracle comparability. |
| RNG bounded sampling (`EP-006`) | Reject invalid bound first; only then run rejection sampling loop | `upper_bound == 0` hard-fails with stable reason code | Prevents silent modulo-by-zero style divergence and keeps replay deterministic. |
| Linalg mode/convergence (`EP-007`) | Shape/rank preflight -> mode token parse -> convergence predicate | Non-convergence/invalid mode/unknown metadata fail with explicit packet reason codes | Prevents accidental fallback to undefined solver branches. |
| Transfer overlap and flatiter writes (`EP-008`) | Validate context -> determine overlap direction -> enforce index/value arity | Contract violation yields stable transfer reason code and no write admission | Protects alias-sensitive transfer semantics and write-lane safety. |

### DOC-PASS-04.3 Coverage implications by workflow

| Workflow ID | Unit/Property | Differential | E2E | Structured logging | Assessment |
|---|---|---|---|---|---|
| `EP-001` | covered (ufunc unit/metamorphic/adversarial suites) | covered for scoped ops | partial via workflow corpus | covered at fixture/scenario level | partial parity debt outside scoped op families |
| `EP-002` | covered (runtime policy tests) | covered (policy-wire adversarial classes) | covered in gate execution | covered (required contract fields) | strong for policy surface |
| `EP-003` | covered by scenario tests | N/A (orchestration path) | covered | covered when log path configured | medium operational risk due optional log path configuration |
| `EP-004` | covered by contract-schema tests | N/A | covered via packet validator binary | readiness reports are structured outputs | strong for artifact topology checks |
| `EP-005` | covered (io contract tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `IOLogRecord` schema tests | partial, gate integration still pending |
| `EP-006` | covered (determinism/bounded/range/state tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `RandomLogRecord` schema tests | partial, gate integration still pending |
| `EP-007` | covered (linalg contract tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `LinAlgLogRecord` schema tests | partial, gate integration still pending |
| `EP-008` | covered (transfer/index contract tests) | missing packet-local oracle differential | missing packet-local replay lane | covered by `TransferLogRecord` schema tests | partial, gate integration still pending |

### DOC-PASS-04.4 Control-flow contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `FLOW-C001` | Packet-local execution paths (`EP-005`..`EP-008`) have deterministic contract tests but are not yet first-class suites in `run_all_core_suites`. | high | `bd-23m.18`, `bd-23m.19`, `bd-23m.20`, `bd-23m.14`/`17` | Add packet-local suites and include them in gate orchestrators. |
| `FLOW-C002` | Workflow/security log appenders are path-config dependent; absent paths can downgrade forensic coverage despite successful test execution. | medium | foundation `bd-23m.6` | Make log path configuration mandatory in gate invocations and fail when expected logs are absent. |
| `FLOW-C003` | Differential harness precedence is ufunc-centric; equivalent branch-level oracle replay is missing for random/linalg/io/transfer paths. | high | packet F-stage beads (`bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`, `bd-23m.14.6`) | Implement packet-local oracle differentials with per-branch reason-code reporting. |
| `FLOW-C004` | Packet readiness validation (`EP-004`) checks artifact presence/schema but cannot prove runtime branch reachability without correlated replay logs. | medium | packet I beads + doc pass-10 | Add branch-reachability evidence links (`fixture_id`,`reason_code`) to readiness report conventions. |

## DOC-PASS-05 Complexity, Performance, and Memory Characterization

### DOC-PASS-05.1 Complexity classes by canonical execution path

| Workflow ID | Dominant operations | Time complexity (current implementation) | Space complexity (current implementation) | Anchors |
|---|---|---|---|---|
| `EP-001` ufunc differential | fixture parse + per-case execution + oracle compare | `O(C * N)` where `C` is case count and `N` is per-case element count | `O(N)` per case for output buffers + report accumulation | `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-ufunc/src/lib.rs` |
| `EP-002` security gate | finite policy/security suite execution | `O(C)` over configured security/policy fixtures | `O(C)` for suite report/log aggregation | `run_security_gate.rs`, runtime suite functions in `crates/fnp-conformance/src/lib.rs` |
| `EP-003` workflow scenario replay | scenario script execution and expectation checks | `O(S * K)` where `S` is scenario count and `K` is steps per scenario | `O(S + log_entries)` | `workflow_scenarios.rs`, `run_workflow_scenario_gate.rs` |
| `EP-004` packet readiness validation | file existence/schema parse checks | `O(F)` where `F` is required artifact count | `O(F)` for missing-field/readiness structures | `contract_schema.rs`, `validate_phase2c_packet.rs` |
| `EP-005` IO dispatch/validation | magic checks, header/shape validation, budget checks | dispatch is `O(1)`; shape/header checks are `O(rank)`; payload footprint checks are `O(1)` arithmetic | `O(rank)` for copied shape/header descriptors | `crates/fnp-io/src/lib.rs` |
| `EP-006` RNG deterministic generation | splitmix step + optional rejection loop | `next_u64`/`next_f64` are `O(1)`; `bounded_u64` has expected `O(1)` with probabilistic retries | `O(1)` per sample; `fill_u64(len)` is `O(len)` | `crates/fnp-random/src/lib.rs` |
| `EP-007` linalg branch family | shape/mode validation + small-kernel helpers | validators are `O(rank)` or `O(1)`; `solve_2x2` is `O(1)` | mostly `O(1)`; shape outputs are `O(rank)` | `crates/fnp-linalg/src/lib.rs` |
| `EP-008` transfer/index legality | selector/overlap checks + index scan | selector/overlap are `O(1)`; fancy/mask index validation is `O(index_len)`/`O(len)` | `O(1)` except `Fancy`/`BoolMask` index storage already provided by caller | `crates/fnp-iter/src/lib.rs` |

### DOC-PASS-05.2 Memory-growth and allocation behavior

| Path family | Allocation model | Growth driver | Boundedness controls | Residual risk |
|---|---|---|---|---|
| `fnp-ufunc` elementwise/reduction | allocates output value vectors; keeps input vectors immutable | output element count and dtype width | shape legality and element-count admission checks | high memory pressure for very large broadcasted outputs if fixture caps are weak |
| `fnp-ndarray` layout calculus | shape/stride vectors and small metadata structs | rank and reshape/broadcast rank expansions | overflow checks + rank-aware validation | large-rank adversarial inputs still require explicit fixture budgets |
| `fnp-conformance` reports/artifacts | suite reports + mismatch vectors + artifact JSON outputs | fixture corpus size and number of gate runs | harness config + packet contract validation | artifact accumulation can grow quickly without retention policy |
| `fnp-random` stream generation | constant-state RNG; vector allocation only in `fill_u64` | requested fill length | caller-controlled `len`; deterministic state model | large requested fills can dominate memory unless bounded by caller/gate fixture contracts |
| `fnp-linalg` contract checks | mostly scalar validation and small shape vectors | matrix rank metadata and output-shape tuples | bounded shape checks (`MAX_BATCH_SHAPE_CHECKS`, etc.) | full-matrix numerical kernels (future packet stages) will alter memory profile materially |
| `fnp-io` parser/dispatch contracts | shape/header copies, archive-member name vectors | shape rank, archive member count, metadata size | `MAX_HEADER_BYTES`, `MAX_ARCHIVE_MEMBERS`, byte-budget constants | decoded payload memory growth requires additional end-to-end budget enforcement in replay gates |
| Transfer/index validation (`fnp-iter`) | mostly zero-allocation checks, except caller-provided index structures | mask/fancy index lengths | strict contract validation and fail-fast checks | large index vectors still impose caller-side memory pressure |

### DOC-PASS-05.3 Hotspot inventory and optimization-governance readiness

| Hotspot family | Why hotspot | Baseline/profile artifact status | EV/one-lever governance status | Owner |
|---|---|---|---|---|
| Broadcasted ufunc loops | `O(N)` numeric kernel path across many fixtures | baseline present (`artifacts/baselines/ufunc_benchmark_baseline.json`) | governance scaffold present; continued per-lever isomorphism artifacts required | `bd-23m.16` |
| Differential harness orchestration | multiplies execution cost by fixture corpus size and oracle bridge overhead | partial (differential report artifacts present; profiler artifacts not standardized) | needs standardized profile capture attachment in packet H/I outputs | packet F/H beads |
| Runtime/security/workflow gates | repeated suite execution with log + summary generation | partial (gate summaries/logs exist; cross-gate cost dashboards absent) | add periodic gate-cost snapshots and budget alarms | foundation `bd-23m.8` |
| RNG/linalg/io/transfer packet-local paths | emerging hotspots as packet-local suites scale | missing dedicated benchmark baselines | EV gate cannot be applied consistently until per-path baselines/profiles exist | `bd-23m.14`/`18`/`19`/`20` |
| Artifact durability pipeline | sidecar/scrub/decode proofs scale with bundle size/count | partial (sidecar/scrub/decode artifacts generated; throughput trends not tracked) | needs throughput and storage-growth baselines in optimization reports | packet I + durability owners |

### DOC-PASS-05.4 Verification and logging implications for performance changes

| Domain | Unit/Property guardrails | Differential guardrails | E2E guardrails | Logging guardrails | Coverage status |
|---|---|---|---|---|---|
| Ufunc hotspot optimization | covered (ufunc unit/metamorphic/adversarial) | covered for scoped corpus | partial | covered | partial outside scoped ops |
| Runtime gate optimization | covered | policy differential equivalent covered | covered | covered | strong |
| RNG/linalg/io/transfer optimization | covered at unit/property level | missing packet-local differentials | missing packet-local replay lanes | schema-level only | missing/partial |
| Artifact pipeline optimization | contract and durability checks covered | N/A | gate runs available | artifact metadata logged | partial (throughput trend logs missing) |

### DOC-PASS-05.5 Complexity/performance contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `PERF-C001` | EV-gated optimization doctrine is documented, but non-ufunc packet paths lack standardized benchmark baselines/profiler artifacts. | high | `bd-23m.8`, packet H beads | Add baseline/profile artifacts for iter/random/linalg/io paths and require them in packet optimization reports. |
| `PERF-C002` | `bounded_u64` uses rejection sampling with no explicit retry cap; expected `O(1)` but worst-case retries are unbounded. | medium | `bd-23m.18` | Document/implement bounded retry policy (or explicit rationale) and add adversarial witness tests. |
| `PERF-C003` | Gate/logging costs are observable per run but not tracked as longitudinal metrics, making regressions hard to detect early. | medium | foundation `bd-23m.8` | Emit periodic gate-cost trend artifacts with environment fingerprints and threshold alerts. |
| `PERF-C004` | Packet-local memory budgets are encoded in crate constants, but cross-packet aggregate memory envelope is undocumented. | medium | `bd-23m.24.6` follow-on + packet owners | Define and publish cross-packet memory budget matrix with enforcement hooks in gate scripts. |

## DOC-PASS-06 Concurrency/Lifecycle Semantics and Ordering Guarantees

### DOC-PASS-06.1 Shared-state ownership map (who can mutate what)

| Shared state surface | Owner module | Mutation API | Synchronization model | Ordering guarantee |
|---|---|---|---|---|
| Runtime decision ledger (`EvidenceLedger.events`) | `crates/fnp-runtime/src/lib.rs` | `EvidenceLedger::record` via `&mut self` | Rust exclusive-borrow discipline (`&mut`) | insertion order equals call order within a single ledger instance. |
| Runtime policy log path config | `crates/fnp-conformance/src/lib.rs` | `set_runtime_policy_log_path` | `OnceLock<Mutex<Option<PathBuf>>>` | last successful setter call wins for subsequent append attempts. |
| Shape/stride log path config | `crates/fnp-conformance/src/lib.rs` | `set_shape_stride_log_path` | `OnceLock<Mutex<Option<PathBuf>>>` | last successful setter call wins for subsequent append attempts. |
| Dtype-promotion log path config | `crates/fnp-conformance/src/lib.rs` | `set_dtype_promotion_log_path` | `OnceLock<Mutex<Option<PathBuf>>>` | last successful setter call wins for subsequent append attempts. |
| Workflow scenario log path config | `crates/fnp-conformance/src/workflow_scenarios.rs` | `set_workflow_scenario_log_path` | `OnceLock<Mutex<Option<PathBuf>>>` | last successful setter call wins for subsequent append attempts. |
| Workflow log required-flag | `crates/fnp-conformance/src/workflow_scenarios.rs` | `set_workflow_scenario_log_required` | `OnceLock<Mutex<bool>>` | gate sets true before attempts; guard resets to false on drop. |

### DOC-PASS-06.2 Lifecycle traces (init -> execute -> finalize)

| Lifecycle ID | Initialization | Active phase | Finalization | Failure terminal |
|---|---|---|---|---|
| `LC-001` runtime decision event lifecycle | construct `EvidenceLedger::new` | `decide_and_record_with_context` computes action/posterior/loss and appends event | caller inspects `events()` / `last()` | fail-closed decisions still produce events; malformed context normalized with default tokens. |
| `LC-002` security/test gate attempt lifecycle | parse options, build default harness config, set attempt log path | run suites, summarize attempt, evaluate pass/fail + diagnostics, possibly retry | emit final summary/artifact index; stop on pass or budget exhaustion | deterministic failure/flake-budget/coverage-floor diagnostics in summary artifacts. |
| `LC-003` workflow scenario gate lifecycle | set `workflow_log_required=true`, install drop guard, set attempt log path | run scenario suite; each step emits workflow log entry via append helper; optional retries | drop guard forces `workflow_log_required=false`; summary and diagnostics emitted | if required log path is unset, append helper fails immediately and gate attempt fails deterministically. |
| `LC-004` log append lifecycle (runtime/shape/dtype/workflow) | resolve configured path or env fallback | ensure parent directory exists, open file in append mode, serialize entry, append newline payload | return `Ok(())` after append | path open/create/write errors are surfaced as gate/suite failures. |

### DOC-PASS-06.3 Ordering guarantees (explicitly documented)

| Ordering domain | Guarantee | Anchor |
|---|---|---|
| Decision events | `events` vector preserves append order; `last()` is the most recent recorded decision in that ledger instance. | `EvidenceLedger::record`, `events`, `last` in `crates/fnp-runtime/src/lib.rs` |
| Workflow scenario evaluation | Scenario list is processed in corpus order; each scenario executes `steps` in declared order (`for step in &scenario.steps`). | `run_user_workflow_scenario_suite` and `execute_mode` in `crates/fnp-conformance/src/workflow_scenarios.rs` |
| Gate attempts | Attempt indices are monotonic from `0..=retries` and first successful attempt breaks the loop. | `run_security_gate.rs`, `run_test_contract_gate.rs`, `run_workflow_scenario_gate.rs` |
| Log entries per step | Within one process, log emission follows step execution order and append call sequence for that attempt. | `maybe_append_*log` helpers and per-step calls in conformance/workflow modules |
| Required-log lifecycle | `set_workflow_scenario_log_required(true)` is active during gate run and is reset by drop guard after run exits. | `WorkflowLogRequirementGuard` in `run_workflow_scenario_gate.rs` |

### DOC-PASS-06.4 Race-sensitive and lifecycle-sensitive zones

| Zone | Sensitivity | Current mitigation | Residual risk |
|---|---|---|---|
| Global log-path setters (`OnceLock<Mutex<...>>`) | process-global mutable config can be overwritten by later setter calls | mutex-protected writes/reads avoid data races | semantic interference if multiple gate flows run concurrently in one process and reconfigure shared path slots. |
| Optional runtime/shape/dtype log path behavior | append helper returns `Ok(())` when no path is set | explicit path setting in security/test gate binaries | non-gate or custom invocations may silently skip logs unless enforced externally. |
| Workflow required-log policy | required-flag guards against silent skip in workflow gate | guard + explicit error on missing path | if other callers invoke suite without required flag, missing path can still be treated as non-fatal. |
| File append semantics | append mode preserves payload-at-end writes per call | each append writes serialized line + newline in one `write_all` call | no explicit cross-process file lock; interleaving across separate processes is possible at OS scheduling granularity. |
| Ledger ownership | per-instance mutable ledger avoids shared mutable globals | `&mut` API enforces single mutable accessor in safe Rust | callers needing cross-thread shared ledgers must introduce their own synchronization policy. |

### DOC-PASS-06.5 Verification and logging implications

| Lifecycle area | Unit/Property | Differential | E2E | Structured logging | Assessment |
|---|---|---|---|---|---|
| Runtime decision ordering | covered by runtime unit tests (event recording + context normalization) | N/A numerical diff; policy-wire behavior covered | covered indirectly via security/workflow gates | covered via runtime policy log entries | strong for single-process sequencing semantics |
| Workflow step ordering and required logs | covered by workflow scenario tests and gate checks | N/A | covered via workflow gate retries and coverage checks | covered with required fields per step | medium risk due shared global path config |
| Gate retry lifecycle | covered in gate binaries through deterministic attempt summary logic | N/A | covered by gate scripts and summaries | attempts include log-path evidence refs | strong for deterministic retry accounting |
| Cross-process append behavior | missing explicit stress/property tests | missing | missing | partial (log lines exist, but interleaving behavior not asserted) | open lifecycle reliability gap |

### DOC-PASS-06.6 Concurrency/lifecycle contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `CONC-C001` | Conformance log path config is global and mutable (`OnceLock<Mutex<Option<PathBuf>>>`), so concurrent in-process gate runs can override each other's destinations. | high | foundation `bd-23m.6`/`bd-23m.23` | move to scoped/log-handle passing or isolate config per run context; add regression test for concurrent invocations. |
| `CONC-C002` | Runtime/shape/dtype append helpers allow silent success when log path is unset, unlike workflow gate required-path mode. | medium | foundation `bd-23m.6` | add mandatory-log mode for all gate-critical appenders or fail gate when expected logs are missing. |
| `CONC-C003` | No explicit end-to-end tests exercise cross-process append contention and ordering stability for JSONL logs. | medium | `bd-23m.7` + docs pass-10 | add stress harness and retention/ordering validation artifacts for log append concurrency. |
| `CONC-C004` | Ordering guarantees are strong inside one process/ledger but not codified as formal contract tests across all packet-local log pipelines. | medium | packet G/I beads | add ordering contract tests and include results in packet readiness evidence packs. |

## DOC-PASS-07 Error Taxonomy, Failure Modes, and Recovery Semantics

### DOC-PASS-07.1 Canonical error taxonomy (code-level anchors)

| Error family | Primary type/registry | Failure class | Typical trigger | Recovery semantics |
|---|---|---|---|---|
| Shape/stride legality | `ShapeError` (`fnp-ndarray`) | input validation / contract violation | invalid dimension, overflow, incompatible broadcast, out-of-bounds view | fail-fast `Result::Err`; caller must adjust shape/stride inputs. |
| Ufunc execution | `UFuncError` + `UFUNC_PACKET_REASON_CODES` (`fnp-ufunc`) | shape/input/axis contract violation | invalid input length, axis out of bounds, shape merge failures | fail-fast; reason code mapped via `UFuncError::reason_code`. |
| Runtime policy | `DecisionAction::FailClosed` (`fnp-runtime`) | compatibility rejection / fail-closed guard | unknown wire mode/class, known incompatible class, high-risk hardened path | deterministic fail-closed decision and evidence event emission. |
| Transfer/index semantics | `TransferError` + `TRANSFER_PACKET_REASON_CODES` (`fnp-iter`) | selector/overlap/index contract violation | invalid context, lossy same-value cast, overlap policy violations, index mismatch | fail-fast rejection with stable transfer reason code. |
| RNG contract | `RandomError` + `RANDOM_PACKET_REASON_CODES` (`fnp-random`) | bounded-generation contract violation | invalid upper bound (`0`) | deterministic error return; caller must provide valid bound. |
| Linalg contract | `LinAlgError` + `LINALG_PACKET_REASON_CODES` (`fnp-linalg`) | shape/mode/convergence/policy failure | singular solves, invalid qr mode, non-convergence, unknown metadata | fail-fast with packet-specific reason codes; unknown metadata fails closed. |
| IO contract | `IOError` + `IO_PACKET_REASON_CODES` (`fnp-io`) | parser/dispatch/budget/policy failure | bad magic/version, invalid schema/descriptor, memmap/pickle policy violations | fail-fast; unknown metadata and invalid dispatch are rejected fail-closed. |
| Gate reliability diagnostics | `deterministic_failure` / `flake_budget_exceeded` / `coverage_floor_breach` (gate binaries) | orchestration reliability failure | retries exhausted, flake budget exceeded, coverage ratio below floor | gate exits non-zero and emits diagnostics with evidence references. |

### DOC-PASS-07.2 Failure mode matrix (detect -> classify -> act)

| Failure mode | Detection point | Classification channel | Action | Escalation path |
|---|---|---|---|---|
| Semantic contract mismatch (shape/ufunc/transfer) | crate-level validators and constructors | typed error enum + packet reason code | immediate `Err`, no fallback mutation | fixture/gate failure with case-id context. |
| Unknown compatibility metadata | runtime/linalg/io policy metadata checks | fail-closed action or policy error reason | reject operation (`fail_closed`) | scenario/gate diagnostics + audit/event trails. |
| Log path unset in workflow gate | `maybe_append_workflow_log` with required flag | explicit string error | fail scenario step and gate attempt | retry (if configured), then deterministic_failure diagnostic. |
| Gate suite mismatch | attempt summarization in gate binaries | suite failures + reliability diagnostics | continue until retry budget exhausted or pass | report JSON and exit code `2` on failure. |
| Flaky behavior beyond budget | gate reliability checks (`flake_budget`) | diagnostic reason code `flake_budget_exceeded` | mark gate failed | operator triage with per-attempt log evidence. |
| Coverage below floor | gate reliability checks (`coverage_floor`) | diagnostic reason code `coverage_floor_breach` | mark gate failed | rerun/expand fixture coverage before promotion. |

### DOC-PASS-07.3 Recovery semantics by layer

| Layer | Recovery strategy implemented today | Strategy not implemented (explicit debt) |
|---|---|---|
| Core semantic crates (`fnp-ndarray`, `fnp-ufunc`, `fnp-iter`, `fnp-random`, `fnp-linalg`, `fnp-io`) | deterministic fail-fast errors with typed enums and stable reason codes | no automatic semantic repair/rewrite; intentional to preserve parity observability |
| Runtime policy layer (`fnp-runtime`) | fail-closed defaults on unknown/incompatible semantics; hardened path can escalate to `full_validate` | no probabilistic auto-override without allowlist; intentional safety constraint |
| Conformance suites | per-case failure accumulation with explicit failure messages | packet-local differential/replay lanes still incomplete for some domains |
| Gate binaries | bounded retry loops + flake/coverage diagnostics + evidence refs | adaptive retry policies beyond fixed budgets not implemented |
| Logging/forensics | structured JSONL append with required fields and contract validation | cross-process append contention hardening and strict global log-path isolation pending |

### DOC-PASS-07.4 User-visible vs internal error channels

| Channel | Intended audience | Current format |
|---|---|---|
| `Display` implementations on error enums | developers/operators reading test output and logs | human-readable explanatory strings with key context (axis, expected/actual, policy text) |
| `reason_code` fields and packet reason-code registries | machine gates, analytics, replay tooling | stable tokenized identifiers (`*_contract_violation`, `*_invalid`, `*_fail_closed` families) |
| Gate diagnostics objects | CI/operator pipelines | structured JSON with subsystem, reason_code, message, evidence_refs |
| Workflow/runtime log entries | replay/forensics tooling | structured JSONL rows with fixture/test/mode/env/artifact/reason fields |

### DOC-PASS-07.5 Verification implications for error/recovery semantics

| Domain | Unit/Property | Differential | E2E | Logging-contract coverage | Status |
|---|---|---|---|---|---|
| Shape + ufunc + runtime errors | covered | covered for scoped ufunc/runtime lanes | covered in workflow/security/test gates | covered | strong in currently scoped domains |
| Transfer/random/linalg/io errors | covered at crate level | missing packet-local differential lanes | missing packet-local replay lanes | schema-level coverage exists | partial/open integration debt |
| Gate reliability diagnostics | covered by gate logic/tests and execution paths | N/A | covered (gates emit diagnostics) | covered by report JSON + attempt logs | strong |

### DOC-PASS-07.6 Error/recovery contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `ERR-C001` | Error taxonomies exist across packet-local crates, but conformance gate orchestration still emphasizes ufunc/runtime lanes. | high | packet F/G owners (`bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6`) | integrate packet-local differential/replay suites and include their reason-code outcomes in gate summaries. |
| `ERR-C002` | Runtime/shape/dtype log appenders can no-op when path unset, weakening recovery forensics despite errors being generated. | medium | foundation `bd-23m.6` | require mandatory logging mode or explicit gate failure on missing expected logs. |
| `ERR-C003` | Gate retries are fixed-budget and non-adaptive; there is no dynamic policy for repeated transient failures. | medium | foundation `bd-23m.8` | add decision-theoretic retry/escalation strategy with explicit expected-loss justification. |
| `ERR-C004` | Packet readiness reports validate artifact presence but do not yet require exhaustive reason-code coverage proofs for packet-local failure families. | medium | packet I beads + docs pass-10 | add reason-code coverage attestations to packet readiness contracts. |

## DOC-PASS-08 Security/Compatibility Edge Cases and Undefined Zones

### DOC-PASS-08.1 Edge-case boundary matrix (legacy anchor -> Rust boundary -> mode behavior)

| Edge zone | Legacy anchor(s) | Rust anchor(s) | Strict mode expectation | Hardened mode expectation | Evidence status |
|---|---|---|---|---|---|
| Unknown runtime wire metadata (`mode`/`class`) | no canonical NumPy wire token schema for strict-vs-hardened routing (FrankenNumPy-added contract surface) | `decide_compatibility_from_wire`, `decide_compatibility`, `evaluate_policy_override` (`crates/fnp-runtime/src/lib.rs`) | fail-closed on unknown `mode`/class | fail-closed on unknown `mode`/class; allow override only for known-compatible + allowlisted + hardened path | covered by runtime unit tests + security/test-contract/workflow suites |
| NPY magic/version and header schema ambiguity | `numpy/lib/format.py` and npy/npz readers in `numpy/lib/npyio.py` | `validate_magic_version`, `validate_header_schema`, `validate_read_payload`, `validate_npz_archive_budget` (`crates/fnp-io/src/lib.rs`) | reject invalid magic/version/header and overflowed budgets | same semantic rejection with bounded retry/size controls | covered in `fnp-io` unit tests; gate integration partial |
| Object dtype/pickle dispatch ambiguity | `numpy.load(... allow_pickle=...)` behavior families | `enforce_pickle_policy`, `classify_load_dispatch`, `validate_memmap_contract` (`crates/fnp-io/src/lib.rs`) | reject object payload when pickle policy disallows it | same rejection unless explicit allow path is provided | unit coverage present; packet-local differential/replay still open |
| Reshape `-1`, overflow, and incompatible element-count edge cases | `numpy/_core/src/multiarray/shape.c` contracts | `fix_unknown_dimension`, `element_count`, `contiguous_strides` (`crates/fnp-ndarray/src/lib.rs`) | deterministic error for multiple `-1`, invalid negatives, overflow, or incompatible counts | same | strong unit/property coverage; broad legacy differential expansion still pending |
| Stride-tricks out-of-bounds / negative-stride edge behavior | `numpy/lib/_stride_tricks_impl.py` (`as_strided`, `broadcast_to`) | `NdLayout::as_strided`, `required_view_nbytes`, `NdLayout::broadcast_to` (`crates/fnp-ndarray/src/lib.rs`) | reject out-of-bounds views and negative strides | same | unit coverage present; parity gap remains for negative-stride-compatible legacy cases |
| Iterator overlap/broadcast/write edge behavior | `numpy/_core/src/multiarray/nditer_constr.c`, `nditer_api.c` | `overlap_copy_policy`, `validate_nditer_flags`, `validate_flatiter_*` (`crates/fnp-iter/src/lib.rs`) | reject policy-violating overlap/broadcast/index requests | same | strong unit/property coverage; packet-local differential lane pending |
| Linalg policy metadata/backend bridge boundary | `numpy/linalg/*` operational behavior families | `validate_policy_metadata`, `validate_backend_bridge`, `validate_tolerance_policy` (`crates/fnp-linalg/src/lib.rs`) | fail on unknown metadata/backend unsupported/budget overflow | same; bounded revalidation/tolerance budgets | unit coverage present; packet-local differential/replay lane pending |

### DOC-PASS-08.2 Undefined or ambiguous zones and deterministic FrankenNumPy stance

| Undefined/ambiguous zone | Why this is ambiguous at parity boundary | Current deterministic stance | Risk |
|---|---|---|---|
| Tokenized `unknown_semantics` values in packet-local metadata validators | token is syntactically accepted in IO/linalg metadata validators, but semantic meaning is not a legacy-stable allow signal | runtime decision layer remains authoritative and fail-closes unknown/incompatible semantic classes | high |
| Negative-stride `as_strided` behavior breadth | legacy permits nuanced negative-stride view behavior in some pathways | `required_view_nbytes` rejects negative strides (`ShapeError::NegativeStride`) pending explicit parity expansion | high |
| Object-dtype load/memmap policy interaction | legacy behavior depends on payload kind, policy flags, and path | object dtype requires explicit pickle path; object memmap is rejected; unknown dispatch prefixes reject | high |
| NDIter overlap/broadcast interactions under complex flags | legacy C machinery contains many branch-specific diagnostics and transitions | current validator focuses on explicit flag contracts (`no_broadcast`, `copy_if_overlap`) and rejects violations | medium |
| Workflow/gate failure-class taxonomy for novel reason codes | new reason codes may emerge before classifier updates | unknown reason codes currently collapse to generic `scenario_assertion` in workflow gate classification | medium |
| Logging-path optionality outside required workflow mode | some appenders can no-op when path is unset | workflow gate can force required logging; runtime/shape/dtype appenders still have optional-path behavior | medium |

### DOC-PASS-08.3 Security/fault scenario matrix (detect -> contain -> audit)

| Scenario | Primary detection guard | Deterministic containment action | Audit / replay artifact path |
|---|---|---|---|
| Header bomb / malformed NPY payload | `validate_magic_version`, `validate_header_schema`, `validate_read_payload` | fail-fast parse rejection | IO reason codes + packet logs (`io_magic_invalid`, `io_header_schema_invalid`, `io_read_payload_incomplete`) |
| Archive amplification attempt | `validate_npz_archive_budget` member/decoded-size/retry caps | fail-fast budget rejection | IO logs + gate diagnostics |
| Pickle smuggling via ambiguous payload prefix | `classify_load_dispatch` + `enforce_pickle_policy` | reject dispatch unless explicit pickle policy allows | IO reason codes (`io_pickle_policy_violation`, `io_load_dispatch_invalid`) |
| Metadata spoofing / unknown control tokens | `decide_compatibility_from_wire`, `validate_policy_metadata`, `validate_io_policy_metadata` | fail-closed action/rejection | runtime evidence ledger + packet-local policy reason codes |
| Stride alias abuse / view overflow | `required_view_nbytes`, `NdLayout::as_strided`, `broadcast_strides` | reject out-of-bounds or incompatible layouts | ndarray reason-path logs and fixture failures |
| Retry-budget gaming / flaky masking | gate retry loops + `flake_budget` + `coverage_floor` checks | non-zero gate exit with explicit diagnostic reason code | gate report JSON + per-attempt logs + replay command guidance |

### DOC-PASS-08.4 Verification/logging implications for security/compatibility edge zones

| Edge family | Unit/Property | Differential | E2E | Logging/forensics | Status |
|---|---|---|---|---|---|
| Runtime fail-closed metadata handling | covered | covered for runtime suites | covered in security/test/workflow gates | covered via evidence ledger + runtime policy logs | strong |
| IO parser/policy boundedness | covered | partial packet-local differential integration | partial | covered via IO log schemas/reason codes | medium/open integration debt |
| Stride-tricks and reshape undefined zones | covered | partial | partial | covered at test/log level | medium with parity debt |
| Iterator/linalg/random packet-local policy edges | covered | missing packet-local differential lanes | missing packet-local replay lanes | schema-level coverage present | partial/open |
| Gate failure taxonomy and replay guidance | covered | N/A | covered | covered, but unknown reasons currently coarsened | medium |

### DOC-PASS-08.5 Security/compatibility contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `SEC-C001` | Packet-local metadata validators accept tokenized `unknown_semantics`, while runtime doctrine requires semantic fail-closed handling for unknowns. | high | packet F/G owners + foundation `bd-23m.8` | document/encode a single cross-crate rule: unknown semantic class must deterministically map to fail-closed behavior end-to-end. |
| `SEC-C002` | Negative-stride view behavior is currently rejected in ndarray, leaving parity debt against legacy stride-tricks breadth. | high | packet `bd-23m.17` | add explicit legacy parity matrix for negative strides and implement/test contract-compliant branches. |
| `SEC-C003` | Object/pickle load policy is bounded but packet-local differential/replay evidence for hostile payload classes is incomplete. | high | packet `bd-23m.20.6`, `bd-23m.20.7` | add adversarial corpus + workflow replay artifacts proving deterministic policy handling. |
| `SEC-C004` | Optional-path logging for runtime/shape/dtype appenders can weaken forensic completeness in edge failures. | medium | foundation `bd-23m.6` | enforce mandatory logging mode or gate-level missing-log failure checks across all critical appenders. |
| `SEC-C005` | Workflow gate collapses unknown reason-code classes into a generic bucket, reducing triage precision for new failure families. | medium | foundation `bd-23m.23` | expand failure-class taxonomy and require explicit mapping tests for newly introduced reason codes. |
| `SEC-C006` | Iterator/linalg/random packet-local edges are unit/property-covered but not yet gate-backed by full differential/replay suites. | medium | packets `bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6` | complete packet-local differential + replay gates and include outputs in readiness summaries. |

## DOC-PASS-09 Unit/E2E Test Corpus and Logging Evidence Crosswalk

### DOC-PASS-09.1 Crosswalk schema (machine-parseable row contract)

Each row in the matrices below is intended to be parseable with stable keys:
- `row_id` (stable identifier)
- `behavior_family`
- `unit_property_lane`
- `differential_lane`
- `e2e_gate_lane`
- `logging_contract_lane`
- `status`
- `risk`
- `owner_beads`

### DOC-PASS-09.2 Behavior-to-evidence matrix

| row_id | behavior_family | unit_property_lane | differential_lane | e2e_gate_lane | logging_contract_lane | status | risk | owner_beads |
|---|---|---|---|---|---|---|---|---|
| `XW-001` | shape/stride legality + view transforms (`fnp-ndarray` + conformance shape suite) | `fnp-ndarray` unit/property tests (`fix_unknown_dimension`, `broadcast_*`, `as_strided`, `sliding_window`) + `run_shape_stride_suite` | no dedicated legacy oracle differential lane yet | workflow/scenario gates indirectly exercise shape/runtime paths, not a dedicated shape gate | `ShapeStrideLogEntry` append path + normalized `fixture_id/seed/mode/env_fingerprint/artifact_refs/reason_code` in conformance (`run_shape_stride_suite`) | partial | medium | `bd-23m.12.6`, `bd-23m.17.6`, docs `bd-23m.24.10` |
| `XW-002` | dtype promotion determinism (`fnp-dtype`) | `fnp-dtype` unit tests + `run_dtype_promotion_suite` fixture corpus | no standalone legacy-oracle diff lane yet | covered only through aggregate gates; no dedicated dtype e2e runner | `DTypePromotionLogEntry` with normalized replay fields (`run_dtype_promotion_suite`) | partial | medium | `bd-23m.13.6`, docs `bd-23m.24.10` |
| `XW-003` | runtime compatibility policy (strict/hardened/fail-closed) | runtime unit tests + `run_runtime_policy_suite` + `run_runtime_policy_adversarial_suite` | adversarial lane present for wire/metadata hostile cases (fixture-driven, not NumPy oracle) | `run_security_gate`, `run_test_contract_gate`, `run_workflow_scenario_gate` (+ `scripts/e2e/run_*`) | `RuntimePolicyLogEntry`, runtime ledger validation (`validate_runtime_policy_log_fields`), required log fields in test-contract gate | strong | low | foundation `bd-23m.5`, `bd-23m.23` |
| `XW-004` | ufunc execution parity (broadcast, reductions, dispatch) | `fnp-ufunc` unit tests + conformance metamorphic/adversarial suites | dedicated oracle differential lane: `run_ufunc_differential_suite` + `run_ufunc_metamorphic_suite` + `run_ufunc_adversarial_suite` | workflow gate artifact linkage includes `ufunc_input_cases`; no dedicated standalone ufunc e2e gate binary | `UFuncLogRecord` replay-complete validation fields and reason-code registry checks | strong (core lane) | medium | `bd-23m.16.6`, `bd-23m.16.7` |
| `XW-005` | transfer/iterator overlap + flatiter contracts (`fnp-iter`) | strong unit/property coverage in `fnp-iter` + conformance packet-003 log-contract tests | packet-local differential lane not yet integrated | packet-local workflow replay lane missing | `TransferLogRecord` validation + reason-code vocabulary checks in conformance | partial/open | high | `bd-23m.14.6`, `bd-23m.14.7` |
| `XW-006` | RNG deterministic streams/state (`fnp-random`) | strong crate-level unit/property coverage (`same_seed`, state round-trip, bounds) | packet-local differential lane not yet integrated | packet-local workflow replay lane missing | `RandomLogRecord` replay-complete validation + reason-code roundtrip tests | partial/open | high | `bd-23m.18.6`, `bd-23m.18.7` |
| `XW-007` | linalg policy/shape/solver contracts (`fnp-linalg`) | strong crate-level unit/property coverage | packet-local differential lane not yet integrated | packet-local workflow replay lane missing | `LinAlgLogRecord` replay-complete validation + reason-code roundtrip tests | partial/open | high | `bd-23m.19.6`, `bd-23m.19.7` |
| `XW-008` | IO parser/dispatch/policy contracts (`fnp-io`) | strong crate-level unit/property coverage | packet-local differential lane not yet integrated | packet-local workflow replay lane missing | `IOLogRecord` replay-complete validation + reason-code roundtrip tests | partial/open | high | `bd-23m.20.6`, `bd-23m.20.7` |
| `XW-009` | reliability/durability gates (retry/flake/coverage, artifact envelopes) | gate-bin unit coverage + conformance core-suite checks | N/A (governance lane) | dedicated gate binaries + e2e scripts (`run_security_policy_gate.sh`, `run_test_contract_gate.sh`, `run_workflow_scenario_gate.sh`, `run_raptorq_gate.sh`) | reliability diagnostics + workflow forensics artifact index + failure envelopes | strong | medium | foundation `bd-23m.23`, `bd-23m.7` |

### DOC-PASS-09.3 Mandatory replay/logging field contract crosswalk

| Field | Contract meaning | Validation anchors |
|---|---|---|
| `fixture_id` | unique fixture identity for replay/triage lineage | packet log-record validators (`UFuncLogRecord`, `TransferLogRecord`, `RandomLogRecord`, `LinAlgLogRecord`, `IOLogRecord`) and `REQUIRED_LOG_FIELDS` in `run_test_contract_gate` |
| `seed` | deterministic replay handle | required in conformance log-contract gate (`seed` must be u64) and packet log structs |
| `mode` | strict/hardened execution mode channel | required by log-contract gate + packet runtime-mode enums |
| `env_fingerprint` | environment identity for reproducibility | normalized by conformance helpers (`normalize_env_fingerprint`) and required by log validators |
| `artifact_refs` | evidence pointers to fixtures/reports/contracts | normalized (`normalize_artifact_refs`) and required non-empty by log validators + contract gate |
| `reason_code` | stable machine-level failure/success taxonomy token | normalized (`normalize_reason_code`) and validated against packet reason-code registries |

### DOC-PASS-09.4 Gap backlog derived from crosswalk

| gap_id | Missing lane | Impact | Risk | Owner | Closure criteria |
|---|---|---|---|---|---|
| `GAP-XW-001` | No packet-local differential lane for iter/random/linalg/io packets | cannot prove end-to-end parity drift on these lanes | high | `bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6` | add fixture-driven differential harnesses + reports per packet |
| `GAP-XW-002` | No packet-local e2e workflow replay scripts for iter/random/linalg/io packets | weakens reproducible incident forensics outside runtime/ufunc path | high | `bd-23m.14.7`, `bd-23m.18.7`, `bd-23m.19.7`, `bd-23m.20.7` | add workflow corpus + gate integration + replay commands |
| `GAP-XW-003` | Shape/dtype lanes lack dedicated gate binaries despite suite-level coverage | harder to isolate regressions in CI by subsystem | medium | foundation `bd-23m.23` + packet owners | add subsystem-focused gate wrappers or explicit gate-suite partitioning |
| `GAP-XW-004` | Runtime/shape/dtype appenders can no-op if log path unset | potential forensic blind spots | medium | `bd-23m.6` | enforce required-log policy or fail gate on missing expected logs |

### DOC-PASS-09.5 Crosswalk contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `XW-C001` | Packet-local crates expose replay-complete log contracts, but corresponding differential/e2e conformance lanes are unevenly integrated. | high | packet F/G owners | require each packet readiness report to include all four lanes: unit/property, differential, e2e, log contract validation. |
| `XW-C002` | Gate reliability tooling is mature for runtime/workflow lanes, but equivalent reliability envelopes are not yet emitted for packet-local lanes. | medium | foundation `bd-23m.23` + packet owners | extend reliability-report pattern to packet-local gates as they are added. |
| `XW-C003` | Crosswalk evidence is currently embedded in docs only; there is no generated machine artifact that CI can diff directly for drift. | medium | docs `bd-23m.24.10` + foundation automation bead | emit crosswalk snapshot JSON as CI artifact and gate schema/version drift. |

## DOC-PASS-10 Expansion Draft (Pass A): Topology and Boundary Narratives

### DOC-PASS-10.1 Layered topology contract (project-level)

| layer_id | Layer | Owned responsibilities | Primary anchors | Allowed dependency direction |
|---|---|---|---|---|
| `L1` | API semantics kernel | dtype/shape/stride/broadcast legality and deterministic view contracts | `crates/fnp-dtype/src/lib.rs`, `crates/fnp-ndarray/src/lib.rs` | inward to pure semantic primitives only |
| `L2` | Execution kernels | transfer/iterator execution classes, ufunc kernels/reductions, packet-local numeric contracts | `crates/fnp-iter/src/lib.rs`, `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | may depend on `L1`; must not mutate conformance/gate state directly |
| `L3` | Runtime policy/audit | strict/hardened decision law, fail-closed semantics, override audit, expected-loss event capture | `crates/fnp-runtime/src/lib.rs` | consumes outputs from `L1/L2`; exposes decision events |
| `L4` | Conformance orchestration | fixture ingestion, suite execution, parity comparison, log normalization/append | `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/src/workflow_scenarios.rs` | orchestrates `L1/L2/L3`; owns suite reports |
| `L5` | Gate/reliability + durability artifacts | retry/flake/coverage governance, forensics index, RaptorQ sidecars/scrub/decode proofs | `crates/fnp-conformance/src/bin/run_*_gate.rs`, `crates/fnp-conformance/src/raptorq_artifacts.rs`, `scripts/e2e/run_*_gate.sh` | wraps `L4`; must remain policy-only (no semantic mutation) |

### DOC-PASS-10.2 Boundary narratives (who may call whom, and why)

| Boundary | Narrative | Contract |
|---|---|---|
| `L1 -> L2` | semantic legality feeds execution but execution does not redefine legality | execution kernels must treat shape/dtype legality as preconditions and surface typed errors, not reinterpret rules ad hoc |
| `L2/L1 -> L3` | runtime policy evaluates compatibility/risk metadata before action | unknown/incompatible semantics map to fail-closed decisions and auditable events |
| `L3/L2/L1 -> L4` | conformance runs fixtures against semantic + execution + policy layers and emits structured reports | suites normalize `fixture_id/seed/mode/env_fingerprint/artifact_refs/reason_code` for replay |
| `L4 -> L5` | gates consume suite summaries and reliability budgets, then publish deterministic status + diagnostics | retry/flake/coverage controls may classify outcomes but cannot alter semantic verdicts |

### DOC-PASS-10.3 Verification/logging implications by topology layer

| layer_id | Unit/Property | Differential | E2E | Logging contract | Status |
|---|---|---|---|---|---|
| `L1` | strong for ndarray/dtype legality invariants | partial (shape/dtype differential expansion pending) | partial (indirect via workflow gates) | shape/dtype suite logs exist; required-path enforcement not universal | medium |
| `L2` | strong crate-level for ufunc/iter/random/linalg/io | strong for ufunc only; packet-local differential debt in iter/random/linalg/io | partial (runtime/workflow-centric) | packet log-record schemas enforce replay fields | mixed/open |
| `L3` | strong runtime unit + adversarial suites | N/A numeric diff; strong policy fixture differential | strong gate integration | runtime ledger + log-field checks enforced | strong |
| `L4` | strong suite-level validation | strong on integrated suites | strong via workflow scenarios | normalized fields + append helpers | strong with optional-path caveat |
| `L5` | strong gate-bin coverage | N/A | strong scripted replay wrappers (`rch exec -- cargo run ...`) | reliability reports/diagnostics/artifact indices | strong |

### DOC-PASS-10.4 Topology contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `ARCH-C001` | Layer topology is now explicit, but import/build-time guardrails do not yet automatically enforce forbidden cross-layer coupling. | medium | foundation automation follow-up | add static dependency checks (crate-level policy) and gate violations in CI. |
| `ARCH-C002` | `L2` packet-local domains expose log schemas but remain unevenly integrated into `L4/L5` differential and e2e governance lanes. | high | packet F/G owners | promote packet-local lanes into gate suite roster and readiness reports. |
| `ARCH-C003` | Optional-path logging in parts of `L4` can reduce observability guarantees expected by `L5` reliability governance. | medium | `bd-23m.6` + `bd-23m.23` | enforce required-log policy for all gate-critical appenders or fail gate when logs are missing. |

## DOC-PASS-11 Expansion Draft (Pass B): Behavioral and Risk Synthesis

### DOC-PASS-11.1 Behavioral risk concentration map

| domain_id | Behavioral contract core | Dominant failure modes | Blast radius if regressed | Current controls | Residual debt |
|---|---|---|---|---|---|
| `BR-001` | shape/stride/broadcast legality | overflow, incompatible broadcast, invalid reshape/view transitions | corrupts nearly all array operations upstream of kernel execution | `ShapeError`-backed checks, shape-stride suite, unit/property invariants | dedicated differential/e2e breadth for full legacy matrix still incomplete |
| `BR-002` | dtype promotion/cast determinism | wrong promotion table resolution, cast policy drift | silent numeric/type drift across all ufunc/linalg/IO paths | dtype promotion suite + registry checks | dedicated dtype-focused differential/e2e gate lane pending |
| `BR-003` | runtime strict/hardened compatibility doctrine | unknown/incompatible metadata not failing closed, override misuse | policy bypass can invalidate security and compatibility guarantees globally | fail-closed decision law + adversarial/runtime suites + gate diagnostics | optional-path logging caveat remains in some appenders |
| `BR-004` | ufunc dispatch/reduction parity | shape/dtype mismatch, reduction axis drift, broadcast dispatch divergence | high user-visible parity break on core numerical path | oracle differential + metamorphic + adversarial suites | standalone ufunc e2e gate wrapper not yet isolated |
| `BR-005` | iterator/transfer/flatiter overlap semantics | overlap policy misclassification, index/write contract breaks | aliasing/assignment corruption in non-contiguous and iterator-heavy paths | packet-local unit/property + reason-code/log schema checks | packet-local differential and replay lanes still open |
| `BR-006` | RNG deterministic stream/state schema | seed/state drift, bounded generation policy violation | reproducibility and scientific repeatability failures | deterministic/state tests + log schema checks | packet-local differential/replay gate integration pending |
| `BR-007` | linalg policy/shape/convergence contracts | mode-policy mismatch, solver/shape contract violations | high-severity numerical correctness and stability risk | packet-local unit/property + reason-code/log schema checks | packet-local differential/replay gate integration pending |
| `BR-008` | IO parser/dispatch/policy boundaries | malformed payload acceptance, pickle/memmap policy drift, budget bypass | security and data-integrity risk at ingestion boundary | bounded validators + reason-code/log schema checks | packet-local differential/replay gate integration pending |

### DOC-PASS-11.2 Compatibility drift sentinel matrix

| sentinel_id | Drift signal | Detection lane | Trigger threshold | Escalation action |
|---|---|---|---|---|
| `SEN-001` | fail-closed violations on unknown/incompatible semantics | runtime ledger checks (`validate_runtime_policy_log_fields`) + gate diagnostics | any non-zero occurrence | immediate gate fail; block promotion and open high-priority incident bead |
| `SEN-002` | ufunc parity mismatches vs oracle | `run_ufunc_differential_suite` report failures | any mismatch in required family | fail build lane; require minimized repro and reason-code mapping update |
| `SEN-003` | flake/coverage reliability drift | gate reliability diagnostics (`flake_budget_exceeded`, `coverage_floor_breach`) | over configured budget/floor | block merge; replay via emitted command and update corpus/fixtures |
| `SEN-004` | missing replay-complete log fields | `run_test_contract_gate` log validator + packet log-record validators | any required-field miss | fail test-contract gate and require logging contract remediation |
| `SEN-005` | packet-local lane incompleteness at readiness | packet readiness reports vs crosswalk `GAP-XW-*` | any missing required lane at closure | disallow packet closure until lane-evidence checklist is complete |

### DOC-PASS-11.3 Budgeted-control and expected-loss integration status

| control_surface | Budgeted behavior implemented | Expected-loss / decision-theoretic status | Gap owner |
|---|---|---|---|
| runtime policy decisions | yes (`hardened_validation_threshold`, fail-closed routing, expected-loss fields in events) | implemented in runtime decision events | foundation maintained |
| gate retry/flake/coverage governance | yes (explicit retries, flake budget, coverage floor) | heuristic thresholds currently dominate; expected-loss escalation policy not unified | `bd-23m.8`, `bd-23m.23` |
| packet-local parser/solver budgets | partial (IO/linalg bounded retries/search depth/cap checks) | mostly threshold-based guards; unified loss-based escalation not yet encoded | packet owners + foundation policy |

### DOC-PASS-11.4 Verification/logging implications for pass-B synthesis

| synthesis_area | Unit/Property | Differential | E2E | Logging/forensics | Status |
|---|---|---|---|---|---|
| Core semantic/kernel concentration (`BR-001..BR-004`) | strong | strong for ufunc/runtime, partial for shape/dtype | strong for runtime/workflow lanes | strong with known optional-path caveat | medium-strong |
| Packet-local risk concentration (`BR-005..BR-008`) | strong crate-level | missing packet-local differential lanes | missing packet-local replay lanes | strong schema-level, weaker gate integration | partial/open |
| Reliability sentinel framework (`SEN-*`) | strong gate-bin/test coverage | N/A | strong for security/test/workflow/raptorq gate runners | strong diagnostics + replay-command linkage | strong |

### DOC-PASS-11.5 Behavioral/risk synthesis contradictions and closure register

| ID | Contradiction / unknown | Risk | Owner | Closure criteria |
|---|---|---|---|---|
| `SYN-C001` | Risk concentration is known for packet-local domains, but lane completeness remains inconsistent at readiness time. | high | packet F/G owners | require lane-complete checklist enforcement in packet readiness gate. |
| `SYN-C002` | Runtime uses expected-loss annotations, but gate escalation strategy is still primarily threshold-driven. | medium | `bd-23m.8` + `bd-23m.23` | add explicit expected-loss-based escalation policy for reliability gates. |
| `SYN-C003` | Cross-cut sentinel set is documented, but no single generated artifact currently tracks sentinel status over time. | medium | docs automation follow-up | emit sentinel snapshot artifact per gate run and trend it in CI. |

## DOC-PASS-12 Independent Red-Team Contradiction and Completeness Review

### DOC-PASS-12.1 Adversarial findings ledger (severity + evidence anchors)

| finding_id | Severity | Finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `RT12-F001` | high | Earlier doc framing treated packet-local crates as placeholders/stubs despite implemented packet-local contract APIs. | `DOC-PASS-02.1`, `DOC-PASS-02.4`, `DOC-PASS-02.5`; `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | corrected language in pass-12 precursor edits; contradiction now reframed as lane-integration debt, not implementation absence |
| `RT12-F002` | high | Packet-local domains remain under-integrated into differential + replay/e2e governance despite strong crate-level unit/property coverage. | `DOC-PASS-09.2`, `DOC-PASS-09.4`, `DOC-PASS-11.4`; `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/workflow_scenarios.rs` | open; must be closed by packet F/G/H/I lane completion beads before readiness sign-off |
| `RT12-F003` | medium | Optional-path log appenders can weaken replay-complete evidence guarantees on gate-critical paths. | `DOC-PASS-10.4` (`ARCH-C003`), `DOC-PASS-11.4`; `crates/fnp-conformance/src/lib.rs` (`maybe_append_*`), `crates/fnp-conformance/src/workflow_scenarios.rs` (`maybe_append_workflow_log`) | open; require fail-closed required-log policy for gate-critical appenders |
| `RT12-F004` | medium | Runtime uses expected-loss calculus, but reliability gate escalation remains mostly threshold-based. | `DOC-PASS-11.3`, `DOC-PASS-11.5` (`SYN-C002`); `crates/fnp-runtime/src/lib.rs` (`expected_loss_*` fields), `crates/fnp-conformance/src/bin/run_*_gate.rs` | open; unify expected-loss escalation policy across gate binaries |
| `RT12-F005` | medium | Contradiction tracking is distributed across many pass sections without a single machine snapshot artifact. | `DOC-PASS-02.6`, `03.5`, `04.4`, `05.5`, `06.6`, `07.6`, `08.5`, `09.5`, `10.4`, `11.5` | open; add generated contradiction/sentinel snapshot artifact per gate cycle |

### DOC-PASS-12.2 Assertion traceability map (legacy anchors and executable evidence)

| assertion_id | Assertion under review | Legacy anchor(s) | Executable evidence anchor(s) | Traceability status |
|---|---|---|---|---|
| `RT12-A001` | SCE shape/stride/broadcast legality is deterministic and fail-fast on invalid contracts. | `legacy_numpy_code/numpy/_core/src/multiarray/shape.c`, `legacy_numpy_code/numpy/_core/src/multiarray/shape.h` | `crates/fnp-ndarray/src/lib.rs`, `crates/fnp-conformance/fixtures/shape_stride_cases.json`, `crates/fnp-conformance/src/lib.rs` | traceable; differential breadth remains partial |
| `RT12-A002` | Dtype promotion behavior is deterministic under declared cast policy. | `legacy_numpy_code/numpy/_core/src/multiarray/dtypemeta.c`, `legacy_numpy_code/numpy/_core/src/multiarray/descriptor.c`, `legacy_numpy_code/numpy/_core/src/multiarray/can_cast_table.h` | `crates/fnp-dtype/src/lib.rs`, `crates/fnp-conformance/fixtures/dtype_promotion_cases.json`, `crates/fnp-conformance/src/lib.rs` | traceable; dedicated dtype gate lane still pending |
| `RT12-A003` | Strict/hardened policy decisions are fail-closed for unknown/incompatible semantics with auditable reasoning. | clean-room policy extension (no single legacy equivalent) | `crates/fnp-runtime/src/lib.rs`, `crates/fnp-conformance/fixtures/runtime_policy_cases.json`, `crates/fnp-conformance/fixtures/runtime_policy_adversarial_cases.json`, `crates/fnp-conformance/src/test_contracts.rs` | traceable via executable policy fixtures and schema checks |
| `RT12-A004` | Ufunc dispatch/reduction behavior is oracle-checked and mismatch-reporting is deterministic. | `legacy_numpy_code/numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/src/bin/run_ufunc_differential.rs` | traceable; standalone ufunc e2e wrapper still open |
| `RT12-A005` | Durable artifact obligations include sidecar/scrub/decode-proof linkage. | contract-level requirement (durability doctrine) | `crates/fnp-conformance/src/raptorq_artifacts.rs`, `crates/fnp-conformance/src/bin/run_raptorq_gate.rs`, `artifacts/raptorq/conformance_bundle_v1.sidecar.json`, `artifacts/raptorq/conformance_bundle_v1.scrub_report.json`, `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` | traceable; gating breadth is strong |

### DOC-PASS-12.3 Unit/property, differential, e2e, logging implication matrix

| implication_id | Area | Unit/Property | Differential | E2E/replay | Logging/forensics | Status |
|---|---|---|---|---|---|---|
| `RT12-I001` | packet-local iter/random/linalg/io parity proof chain | covered at crate level | missing/partial packet-local differential lanes | missing/partial packet-local replay scenarios | schema-level checks covered; gate-level propagation partial | open/high |
| `RT12-I002` | required-log enforcement on gate-critical appenders | partial (`#[cfg(test)]` validation exists) | N/A | partial (gates run even when some appenders are optional-path) | missing fail-closed enforcement for missing gate-critical logs | open/medium |
| `RT12-I003` | expected-loss-informed escalation in reliability envelope | covered in runtime decision unit tests | N/A | partial (gate binaries use threshold controls) | partial (diagnostics emitted, but expected-loss policy not unified) | open/medium |
| `RT12-I004` | contradiction/sentinel drift observability over time | N/A | N/A | deferred pending snapshot emission | missing single machine artifact for contradiction trend tracking | open/medium |

### DOC-PASS-12.4 Contradiction/unknown closure register

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `RT12-C001` | Packet-local contract APIs are implemented, but readiness still lacks lane-complete differential + replay evidence for all packet-local domains. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | each packet ships closed differential + workflow/e2e evidence and appears as green in readiness reports |
| `RT12-C002` | Gate-critical logging remains partially optional-path and can silently weaken replay completeness guarantees. | medium | `bd-23m.6`, `bd-23m.23` | enforce required logging policy (or explicit fail) in all gate-critical appenders and validate in test-contract gate |
| `RT12-C003` | Expected-loss model is runtime-local; reliability governance does not yet consume it as primary escalation policy. | medium | `bd-23m.8`, `bd-23m.23` | gate retries/flake/coverage escalation includes expected-loss policy artifact and deterministic fallback trigger |
| `RT12-C004` | Contradiction registers are comprehensive but fragmented, preventing single-artifact trend/diff auditing. | medium | docs automation follow-up + `bd-23m.23` | emit versioned contradiction/sentinel snapshot artifact and wire it into CI/gate reports |

## DOC-PASS-14 Full-Agent Deep Dive Pass A (Structure Specialist)

### DOC-PASS-14.1 Canonical topology alias map (cross-doc coherence)

| Canonical layer | Local alias (this doc) | Alias in `EXISTING_NUMPY_STRUCTURE.md` | Ownership anchor(s) | Structural contract |
|---|---|---|---|---|
| semantic kernel | `L1` | `A1` | `crates/fnp-dtype/src/lib.rs`, `crates/fnp-ndarray/src/lib.rs` | single source of truth for legality/promotion; execution may not redefine semantics |
| execution kernel set | `L2` | `A2` | `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | consumes semantic outputs and emits typed errors/reason codes |
| runtime policy kernel | `L3` | `A3` | `crates/fnp-runtime/src/lib.rs` | mode/class/risk arbitration (`allow`/`full_validate`/`fail_closed`) with audit evidence |
| conformance orchestrator | `L4` | `A4` | `crates/fnp-conformance/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, `crates/fnp-conformance/src/workflow_scenarios.rs` | fixture-driven execution/reporting; no semantic mutation |
| reliability/durability envelope | `L5` | `A5` | `crates/fnp-conformance/src/bin/run_*_gate.rs`, `crates/fnp-conformance/src/raptorq_artifacts.rs`, `scripts/e2e/run_*_gate.sh` | governance and durability checks only; cannot rewrite suite outcomes |

### DOC-PASS-14.2 Structure-specialist findings ledger (adversarial)

| finding_id | Severity | Structural finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `ST14-F001` | medium | Two equivalent layer ID vocabularies (`L*` vs `A*`) were undocumented as aliases, creating avoidable drift risk in future pass references. | `DOC-PASS-10.1` in both target docs | resolved in this pass via explicit alias map |
| `ST14-F002` | high | Layering/dependency law is documented but still lacks machine enforcement, so cross-layer violations can regress silently. | `DOC-PASS-10.4` (`ARCH-C001`), `DOC-PASS-01.5` (`TOPO-C003`), root `Cargo.toml` workspace graph | open; requires CI-level dependency-direction contract check |
| `ST14-F003` | high | Packet-local `L2/A2` domains remain structurally under-wired into `L4/L5` gate lanes despite clear ownership boundaries. | `DOC-PASS-09.2`, `DOC-PASS-11.4`, `DOC-PASS-12.3` | open; packet-local F/G/H/I lane closure required |
| `ST14-F004` | medium | Gate/replay command conventions are documented, but there is no single structure-level checklist proving each gate path is `rch`-offloaded and replay-deterministic. | `DOC-PASS-10.3`, `scripts/e2e/run_*_gate.sh`, `crates/fnp-conformance/src/bin/run_*_gate.rs` | open; add gate invocation topology checklist artifact |

### DOC-PASS-14.3 Boundary-law coherence audit

| boundary_id | Canonical boundary law | Evidence of enforcement today | Gap | Owner |
|---|---|---|---|---|
| `ST14-B001` | semantic kernel -> execution kernel only (`L1 -> L2`) | crate-level API boundaries and typed error vocabularies | no automated forbidden-edge CI check | foundation automation follow-up |
| `ST14-B002` | execution/semantic -> runtime policy (`L2/L1 -> L3`) | runtime decision API and policy fixture suites | packet-local paths are not uniformly represented in gate-level scenarios | packet F/G owners |
| `ST14-B003` | runtime/execution/semantic -> conformance (`L3/L2/L1 -> L4`) | conformance orchestrator suite wiring and fixture contracts | lane completeness varies by packet domain | packet + conformance owners |
| `ST14-B004` | conformance -> reliability envelope (`L4 -> L5`) | gate binaries consume suite summaries + diagnostics | no single generated topology checklist proving every gate path and replay contract | `bd-23m.10`, `bd-23m.23` |

### DOC-PASS-14.4 Unit/property, differential, e2e, logging implications (structure lens)

| implication_id | Structure concern | Unit/Property | Differential | E2E/replay | Logging/forensics | Status |
|---|---|---|---|---|---|---|
| `ST14-I001` | alias-map drift prevention (`L*`/`A*`) | covered by doc-level canonical map | N/A | N/A | N/A | resolved |
| `ST14-I002` | dependency-direction enforcement | missing automated structural test | N/A | partial (detected indirectly through failures) | partial (failure diagnostics exist, no direct edge-policy artifact) | open/high |
| `ST14-I003` | packet-local topology integration (`L2/A2 -> L4/L5`) | covered at crate level | missing/partial packet-local lanes | missing/partial packet-local workflow replay | partial schema coverage, incomplete gate-level propagation | open/high |
| `ST14-I004` | gate invocation topology determinism (`rch` + replay completeness) | partial via gate-bin tests | N/A | partial scripted wrappers, no single checklist artifact | partial diagnostics without unified topology checklist | open/medium |

### DOC-PASS-14.5 Structure contradiction register with closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `ST14-C001` | Equivalent layer IDs across docs were not explicitly mapped, risking reference drift and false contradictions. | medium | `bd-23m.24.15` | maintained canonical alias table present in both docs and used by subsequent passes |
| `ST14-C002` | Dependency-direction contract remains prose-only and can drift from actual workspace edges. | high | `bd-23m.23` + docs automation follow-up | CI emits explicit dependency-edge policy check and fails forbidden edges |
| `ST14-C003` | Packet-local layer ownership is declared but not lane-complete in conformance/gate topology. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet-local differential + replay lanes are required and green in readiness outputs |
| `ST14-C004` | No unified gate topology artifact ties `rch` command path, replay contract, and diagnostics references together. | medium | `bd-23m.10`, `bd-23m.23` | publish gate topology checklist artifact linking invocation, replay fields, and diagnostics paths |

## DOC-PASS-15 Full-Agent Deep Dive Pass B (Behavior Specialist)

### DOC-PASS-15.1 Behavior findings ledger (adversarial)

| finding_id | Severity | Behavioral finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `BH15-F001` | high | Runtime decision semantics include risk-threshold action routing, but packet-local policy validators (`fnp-io`, `fnp-linalg`) currently validate token legality only, not action selection; docs must keep these layers distinct. | `crates/fnp-runtime/src/lib.rs` (`decide_compatibility`), `crates/fnp-io/src/lib.rs` (`validate_io_policy_metadata`), `crates/fnp-linalg/src/lib.rs` (`validate_policy_metadata`) | resolved in docs by explicit layer separation; implementation integration remains open |
| `BH15-F002` | medium | Deterministic invariants are strong at crate level (SCE/ufunc/iter/random), but cross-packet differential/replay evidence remains uneven, so behavior confidence is non-uniform by domain. | `crates/fnp-ndarray/src/lib.rs`, `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-iter/src/lib.rs`, `crates/fnp-random/src/lib.rs`; `DOC-PASS-12.3` | open; packet-local F/G lane completion required |
| `BH15-F003` | medium | Reason-code stability is tested packet-by-packet, but no global reason-code ontology artifact yet enforces cross-packet semantic normalization. | packet reason-code registries in `fnp-iter`/`fnp-ufunc`/`fnp-random`/`fnp-linalg`/`fnp-io`; `crates/fnp-conformance/src/test_contracts.rs` | open; needs cross-packet taxonomy artifact/policy gate |
| `BH15-F004` | low | Fail-closed handling for unknown metadata is consistently encoded across runtime and packet-local validators, reducing ambiguity for unknown/incompatible behavior classes. | `crates/fnp-runtime/src/lib.rs`, `crates/fnp-linalg/src/lib.rs`, `crates/fnp-io/src/lib.rs` | confirmed/retained |

### DOC-PASS-15.2 Behavior invariant traceability map

| assertion_id | Behavioral assertion | Legacy anchor(s) | Executable evidence anchor(s) | Status |
|---|---|---|---|---|
| `BH15-A001` | Broadcast legality and reshape `-1` inference are deterministic and overflow-checked. | `legacy_numpy_code/numpy/_core/src/multiarray/shape.c`, `legacy_numpy_code/numpy/_core/src/multiarray/shape.h` | `crates/fnp-ndarray/src/lib.rs` (`broadcast_shape`, `fix_unknown_dimension`, overflow tests) | traceable; broad parity matrix still expanding |
| `BH15-A002` | Ufunc broadcasted elementwise/reduction behavior is deterministic with stable error semantics. | `legacy_numpy_code/numpy/_core/src/umath/ufunc_object.c` | `crates/fnp-ufunc/src/lib.rs` (property grids + reason-code tests), `crates/fnp-conformance/src/ufunc_differential.rs` | traceable; standalone ufunc e2e gate artifact pending |
| `BH15-A003` | Transfer/flatiter/overlap behavior follows deterministic policy with explicit rejection reasons. | `legacy_numpy_code/numpy/_core/src/multiarray/dtype_transfer.c`, `legacy_numpy_code/numpy/_core/src/multiarray/nditer*` | `crates/fnp-iter/src/lib.rs` (selector/overlap/flags property grids + log checks) | traceable; packet-local differential/replay integration pending |
| `BH15-A004` | RNG streams are deterministic under seed/state/jump contracts and bounded output rules. | `legacy_numpy_code/numpy/random/src/*`, `legacy_numpy_code/numpy/random/*.pyx` | `crates/fnp-random/src/lib.rs` (seed/state/jump/fill tests + log checks) | traceable; packet-local differential/replay integration pending |
| `BH15-A005` | Strict/hardened compatibility routing is fail-closed for unknown/incompatible semantics with auditable loss annotations. | policy extension (clean-room) | `crates/fnp-runtime/src/lib.rs` (`decide_compatibility`, `expected_loss_for_action`, ledger events), `runtime_policy_*` fixtures | traceable; gate-level expected-loss escalation policy remains open |

### DOC-PASS-15.3 Strict vs hardened edge-case coherence matrix

| behavior_surface | Strict mode expectation | Hardened mode expectation | Evidence anchors | Coherence status |
|---|---|---|---|---|
| known-compatible low-risk class | `allow` | `allow` | `crates/fnp-runtime/src/lib.rs` (`decide_compatibility`) | coherent |
| known-compatible high-risk class | `allow` (compatibility-first) | `full_validate` when `risk_score >= threshold` | `crates/fnp-runtime/src/lib.rs` (`hardened_validation_threshold`) | coherent at runtime; packet-local docs must not over-claim this logic locally |
| known-incompatible / unknown class | `fail_closed` | `fail_closed` | runtime + packet-local metadata validators | coherent |
| packet-local metadata token validation | accepts known mode/class tokens, rejects unknown tokens | same | `validate_policy_metadata` in `fnp-linalg`/`fnp-io` | coherent but intentionally narrower than runtime action routing |

### DOC-PASS-15.4 Unit/property, differential, e2e, logging implications (behavior lens)

| implication_id | Behavior concern | Unit/Property | Differential | E2E/replay | Logging/forensics | Status |
|---|---|---|---|---|---|---|
| `BH15-I001` | runtime-vs-packet policy layer separation | covered by runtime and packet-local unit tests | partial (policy differential strong; packet-local action-routing differential absent) | partial (workflow covers runtime routes more than packet-local policy branches) | strong reason-code/log-field checks | open/medium |
| `BH15-I002` | packet-local deterministic behavior parity (iter/random/linalg/io) | strong crate-level invariants | missing/partial packet-local differential lanes | missing/partial packet-local replay lanes | schema-level log completeness checks exist | open/high |
| `BH15-I003` | cross-packet reason-code semantic normalization | covered packet-local registry tests | N/A | N/A | partial (test-contract gate checks presence, not ontology-level semantics) | open/medium |
| `BH15-I004` | fail-closed unknown/incompatible handling | strong runtime + packet-local validator tests | strong on runtime policy fixtures | strong in security/workflow gate paths | strong with known optional-path caveat in appenders | medium-strong |

### DOC-PASS-15.5 Behavior contradiction register with closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `BH15-C001` | Docs can blur runtime action-routing logic with packet-local token validators unless explicitly separated. | high | `bd-23m.24.16` + docs automation follow-up | maintain explicit runtime-vs-packet policy boundary language in final integration pass |
| `BH15-C002` | Behavior invariants are strong in unit/property lanes but not uniformly proven in packet-local differential/replay lanes. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet-local differential and replay lanes are green and referenced in readiness artifacts |
| `BH15-C003` | No global reason-code ontology artifact ensures consistent semantics across packet registries. | medium | `bd-23m.23` + test-contract policy follow-up | publish cross-packet reason-code taxonomy artifact and enforce mapping checks in gate path |
| `BH15-C004` | Expected-loss is computed at runtime, but reliability escalation remains mostly threshold-based. | medium | `bd-23m.8`, `bd-23m.23` | gate policy consumes expected-loss artifact with deterministic fallback trigger and audit trace |

## DOC-PASS-16 Full-Agent Deep Dive Pass C (Risk/Perf/Test Specialist)

### DOC-PASS-16.1 Risk/perf/test findings ledger (adversarial)

| finding_id | Severity | Finding | Evidence anchors | Disposition |
|---|---|---|---|---|
| `RPT16-F001` | high | Reliability gate controls are explicit (`retries`, `flake_budget`, `coverage_floor`, diagnostics reason codes), but packet-local differential/replay incompleteness still dominates residual risk. | `crates/fnp-conformance/src/bin/run_security_gate.rs`, `run_test_contract_gate.rs`, `run_workflow_scenario_gate.rs`; `DOC-PASS-12.3`, `DOC-PASS-15.4` | open; packet-local F/G/H/I lane closure required |
| `RPT16-F002` | medium | Runtime emits expected-loss evidence, but gate escalation logic remains threshold-driven and does not yet consume expected-loss as first-class policy input. | `crates/fnp-runtime/src/lib.rs` (`expected_loss_*`), gate bins in `crates/fnp-conformance/src/bin/run_*_gate.rs` | open; expected-loss integration policy required |
| `RPT16-F003` | medium | Performance evidence chain is strong (p50/p95/p99 baselines, hyperfine/strace artifacts, isomorphism proof, rollback trigger), but EV-gate automation remains documentation-level. | `crates/fnp-conformance/src/benchmark.rs`, `artifacts/optimization/ROUND3_OPPORTUNITY_MATRIX.md`, `artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md` | open; automate EV gate calculation/reporting |
| `RPT16-F004` | medium | Logging field contracts are strongly enforced (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`), yet optional-path appenders still leave a known observability caveat. | `crates/fnp-conformance/src/lib.rs` (`normalize_*`, log appenders), `crates/fnp-conformance/src/test_contracts.rs` | open; required-log enforcement remains pending |
| `RPT16-F005` | low | Durability controls are robust: sidecar/scrub/decode-proof existence, schema checks, hash match, and stale-artifact detection are all executable. | `crates/fnp-conformance/src/raptorq_artifacts.rs`, `crates/fnp-conformance/src/bin/run_raptorq_gate.rs` | confirmed/retained |

### DOC-PASS-16.2 Coverage and gate topology matrix

| lane_id | Risk/test lane | Primary executors | Current strength | Dominant gap | Owner bead(s) |
|---|---|---|---|---|---|
| `RPT16-L001` | Unit/property invariants | packet crates + `#[cfg(test)]` suites | strong | cross-packet ontology and CI policy synthesis | `bd-23m.23` |
| `RPT16-L002` | Differential/metamorphic/adversarial | `run_ufunc_differential`, conformance suites | strong for ufunc/runtime, partial for packet-local iter/random/linalg/io | packet-local differential lane incompleteness | `bd-23m.14.6`, `bd-23m.18.6`, `bd-23m.19.6`, `bd-23m.20.6` |
| `RPT16-L003` | E2E workflow/replay | `run_workflow_scenario_gate` + `scripts/e2e/run_*_gate.sh` | medium-strong for integrated scenarios | packet-local replay lane incompleteness | `bd-23m.14.7`, `bd-23m.18.7`, `bd-23m.19.7`, `bd-23m.20.7` |
| `RPT16-L004` | Reliability governance | security/test/workflow gate binaries | strong retries/flake/coverage + diagnostics | expected-loss not yet policy input | `bd-23m.8`, `bd-23m.23` |
| `RPT16-L005` | Durability integrity | `run_raptorq_gate`, `raptorq_artifacts` suite | strong | none critical; maintain freshness discipline | `bd-23m.21`, `bd-23m.22` |

### DOC-PASS-16.3 Performance governance evidence chain

| stage_id | Governance stage | Evidence anchor(s) | Enforced invariant | Residual risk |
|---|---|---|---|---|
| `RPT16-P001` | baseline telemetry generation | `crates/fnp-conformance/src/benchmark.rs`, `artifacts/baselines/ufunc_benchmark_baseline*.json` | p50/p95/p99 + throughput/bandwidth metrics are reproducibly emitted | benchmark scope breadth still limited to selected workloads |
| `RPT16-P002` | hotspot profiling fallback chain | `artifacts/optimization/ROUND3_OPPORTUNITY_MATRIX.md` | profile evidence exists even when `perf` unavailable (`strace -c` fallback) | lower-fidelity hotspot attribution under fallback |
| `RPT16-P003` | behavior-isomorphism proof | `artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md`, `artifacts/proofs/golden_checksums_round3.txt` | optimization lever changes must preserve behavior identity | proof automation not yet embedded as one-shot gate |
| `RPT16-P004` | rollback trigger governance | `artifacts/proofs/ISOMORPHISM_PROOF_ROUND3.md` (p99 regression trigger) | sustained tail regression trigger documented with rollback condition | trigger enforcement is policy/manual, not machine gate |

### DOC-PASS-16.4 Logging and durability contract checklist

| checklist_id | Contract requirement | Evidence anchor(s) | Status |
|---|---|---|---|
| `RPT16-K001` | replay-critical fields (`fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`) are mandatory for policy/adversarial/workflow fixtures | `crates/fnp-conformance/src/test_contracts.rs`, `crates/fnp-conformance/src/lib.rs` (`normalize_*`) | strong |
| `RPT16-K002` | workflow log coverage must include all scenario IDs and required metadata links | `crates/fnp-conformance/src/bin/run_workflow_scenario_gate.rs` (`validate_workflow_log_coverage`) | strong |
| `RPT16-K003` | durability bundles require sidecar + scrub report + decode proof, schema-valid and hash-consistent | `crates/fnp-conformance/src/raptorq_artifacts.rs`, `run_raptorq_gate.rs` | strong |
| `RPT16-K004` | stale artifact detection blocks outdated durability bundles | `crates/fnp-conformance/src/raptorq_artifacts.rs` (mtime + generated_at checks) | strong |
| `RPT16-K005` | gate-critical appenders must not silently skip log emission | `DOC-PASS-12/14/15` contradiction registers (`ARCH-C003`, `RT12-C002`) | open |

### DOC-PASS-16.5 Risk/perf/test contradiction register with closure criteria

| ID | Contradiction / unknown | Risk | Owner bead(s) | Closure criteria |
|---|---|---|---|---|
| `RPT16-C001` | Reliability gates are mature, but packet-local differential/replay lanes remain incomplete and dominate residual release risk. | high | `bd-23m.14.6/.7`, `bd-23m.18.6/.7`, `bd-23m.19.6/.7`, `bd-23m.20.6/.7` | packet-local F/G lanes are green and referenced in readiness reports |
| `RPT16-C002` | Expected-loss remains runtime-local; gate escalation is still threshold-led. | medium | `bd-23m.8`, `bd-23m.23` | gate diagnostics include expected-loss policy artifact and deterministic fallback trigger |
| `RPT16-C003` | Performance governance has strong artifacts but lacks machine-enforced EV gate + rollback automation. | medium | `bd-23m.8`, `bd-23m.10` | CI emits EV score and auto-fails when EV/rollback conditions are violated |
| `RPT16-C004` | Logging schema is strict, yet optional-path gate appenders still leave observability holes. | medium | `bd-23m.6`, `bd-23m.23` | required-log policy enforced fail-closed for all gate-critical appenders |

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
