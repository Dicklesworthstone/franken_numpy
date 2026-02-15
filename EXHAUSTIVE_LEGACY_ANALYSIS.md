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
| `DOC-C002` | Existing structure doc does not map unit/property, differential, e2e, and structured logging implications per subsystem. | `EXISTING_NUMPY_STRUCTURE.md` sections 2-8 | `bd-23m.24.4` | high | Add per-subsystem verification matrix with covered/missing/deferred status and artifact links. |
| `DOC-C003` | No explicit packet-to-artifact ownership crosswalk for unresolved domains (NDIter, RNG, linalg, IO). | `EXHAUSTIVE_LEGACY_ANALYSIS.md` sections 10-13 | `bd-23m.24.3` | high | Add packet index that binds each unresolved behavior class to packet bead IDs and closure gates. |
| `DOC-C004` | Structured logging contract was added but not yet threaded into all subsystem extraction sections. | `artifacts/contracts/test_logging_contract_v1.json`, `artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md` | `bd-23m.24.10` | medium | Each packet section names required log fields and reason-code taxonomy coverage status. |

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
