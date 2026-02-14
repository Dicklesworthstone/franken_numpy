# PORTING_ESSENCE_EXTRACTION_LEDGER_V1

Historical filename note: `_V1` is retained for artifact lineage only; this ledger enforces full legacy NumPy parity as the only acceptable end state.

## 0. Purpose

This ledger is the semantic control plane for clean-room implementation. It enumerates behavior contracts and invariants, maps each contract to planned verification assets, and tracks parity debt with explicit ownership.

## 1. Non-Negotiable Program Doctrine

1. Full drop-in parity is mandatory: no reduced-scope acceptance.
2. Any uncovered behavior is parity debt and must carry owner, blocker, risk tier, and closure gate.
3. Clean-room implementation is mandatory: extract behavior contracts, then implement independently in Rust.
4. Every packet row must map to unit/property checks, differential fixtures, and e2e scenario IDs.

## 2. Global Logging Schema Commitments

All packet evidence workflows commit to structured logging fields:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Runtime policy logs currently include additional fields used for forensic replay:

- `expected_action`
- `actual_action`
- `passed`
- `class`
- `risk_score`
- `action`
- `evidence_terms`

## 3. Packet Contract Rows

| packet_id | subsystem | legacy anchors (paths + symbols) | observable compatibility contracts | strict mode contract | hardened mode contract | non-goals (allowed) | unit/property assertions (planned IDs) | differential fixtures (planned IDs) | e2e scenarios (planned IDs) | current evidence refs | parity debt status | owner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FNP-P2C-001` | shape/reshape legality | `legacy_numpy_code/numpy/_core/src/multiarray/shape.c`: `_fix_unknown_dimension`, `_attempt_nocopy_reshape`, `PyArray_Newshape`, `PyArray_Reshape` | reshape legality, `-1` inference, broadcast shape determinism, contiguous stride derivation | exact legacy-observable shape/stride outcomes | same API semantics + bounded defensive validation | no line-by-line C translation | `UP-001-reshape-unknown-dim`, `UP-001-broadcast-shape`, `UP-001-contig-strides` | `DF-001-reshape-oracle-core`, `DF-001-broadcast-shape` | `E2E-001-io-broadcast-reduce`, `E2E-001-reshape-chain` | `artifacts/phase2c/FNP-P2C-001/legacy_anchor_map.md`, `artifacts/phase2c/FNP-P2C-001/behavior_extraction_ledger.md`, `artifacts/phase2c/FNP-P2C-001/contract_table.md`, `artifacts/phase2c/FNP-P2C-001/risk_note.md`, `artifacts/phase2c/FNP-P2C-001/implementation_plan.md`, `crates/fnp-ndarray/src/lib.rs`, `crates/fnp-conformance/src/lib.rs` (`run_shape_stride_suite`) | partial: A/B/C/D-stage artifacts landed; packet E-I deliverables still pending | `bd-23m.12.1` |
| `FNP-P2C-002` | dtype descriptors and promotion | `legacy_numpy_code/numpy/_core/src/multiarray/descriptor.c`, `dtypemeta.c`, `legacy_numpy_code/numpy/_core/include/numpy/can_cast_table.h` | dtype descriptor normalization, promotion table determinism, cast admissibility semantics | exact promotion/cast decisions for legacy matrix | same outward decisions + guarded rejection on malformed/unknown metadata | no legacy ABI binding duplication in Rust internals | `UP-002-promote-commutative`, `UP-002-cast-lossless`, `UP-002-alias-normalization` | `DF-002-promotion-matrix`, `DF-002-mixed-dtype-ufunc` | `E2E-002-mixed-dtype-pipeline` | `artifacts/phase2c/FNP-P2C-002/legacy_anchor_map.md`, `artifacts/phase2c/FNP-P2C-002/behavior_extraction_ledger.md`, `artifacts/phase2c/FNP-P2C-002/contract_table.md`, `crates/fnp-dtype/src/lib.rs`, `run_dtype_promotion_suite` | partial: A/B-stage artifacts landed; packet C-I deliverables still pending | `bd-23m.13.1` |
| `FNP-P2C-003` | strided transfer semantics | `legacy_numpy_code/numpy/_core/src/multiarray/dtype_transfer.c`: `_strided_to_strided_*`, `PyArray_GetStridedZeroPadCopyFn` | overlap-safe transfer behavior, alias-preserving semantics, assignment/astype stride correctness | exact transfer semantics for supported ops | same external behavior + fail-closed on incompatible transfer metadata | no unsafe shortcut that changes alias semantics | `UP-003-overlap-witness`, `UP-003-strided-copy-equivalence` | `DF-003-transfer-overlap-oracle`, `DF-003-astype-strided` | `E2E-003-overlap-assign-pipeline` | anchor-only today (`PHASE2C_EXTRACTION_PACKET.md`) | open: implementation + packet artifacts absent | `bd-23m.14.1` |
| `FNP-P2C-004` | NDIter traversal/index semantics | `legacy_numpy_code/numpy/_core/src/multiarray/nditer_api.c`: `NpyIter_Reset`, `NpyIter_GotoMultiIndex`, `NpyIter_GotoIndex`, `NpyIter_GetTransferFlags` | deterministic traversal order, multi-index correctness, transfer-flag behavior | exact iteration state transitions | same traversal semantics + bounded guard rails on malformed iterator state | no direct C iterator state machine copy | `UP-004-reset-state`, `UP-004-goto-index`, `UP-004-transfer-flags` | `DF-004-nditer-order`, `DF-004-nditer-indexing` | `E2E-004-iterator-kernel-chain` | anchor-only today (`PHASE2C_EXTRACTION_PACKET.md`) | open: `fnp-iter` packet not implemented | `bd-23m.15.1` |
| `FNP-P2C-005` | ufunc dispatch + gufunc signature | `legacy_numpy_code/numpy/_core/src/multiarray/umath/ufunc_object.c`: `_parse_signature`, `_ufunc_setup_flags`, `convert_ufunc_arguments`, `ufunc_get_name_cstr` | dispatch precedence, broadcasted elementwise semantics, reduction axis/keepdims behavior | exact dispatch and reduction outputs | same API + bounded policy checks; unknown incompatible features fail-closed | no heuristic behavior drift without proof | `UP-005-binary-broadcast`, `UP-005-reduce-axis`, `UP-005-div-zero-contract` | `DF-005-ufunc-oracle-suite`, `DF-005-reduction-oracle` | `E2E-005-ufunc-replay`, `E2E-005-policy-forensics` | `crates/fnp-ufunc/src/lib.rs`, `crates/fnp-conformance/src/ufunc_differential.rs`, fixtures under `crates/fnp-conformance/fixtures/` | partial: core differential exists, packet artifact directory still missing | `bd-23m.16.1` |
| `FNP-P2C-006` | stride tricks + broadcasting API | `legacy_numpy_code/numpy/lib/_stride_tricks_impl.py`: `as_strided`, `sliding_window_view`, `broadcast_to`, `broadcast_shapes`, `broadcast_arrays` | broadcast API legality and view semantics without alias corruption | exact observable behavior for scoped APIs | same API with bounded defensive checks for hostile edge cases | no undefined-stride permissiveness beyond explicit policy | `UP-006-broadcast-api-legality`, `UP-006-view-alias-contract` | `DF-006-stride-tricks-oracle`, `DF-006-broadcast-api` | `E2E-006-broadcast-views-replay` | anchor-only today plus partial broadcast internals in `fnp-ndarray` | open: packet artifacts and full API surface missing | `bd-23m.17.1` |
| `FNP-P2C-007` | RNG core and constructor contract | `legacy_numpy_code/numpy/random/_generator.pyx`: `Generator`, `default_rng` | deterministic seed replay, constructor contract compatibility | exact deterministic results for scoped generators | same constructor/output contract + fail-closed on incompatible RNG metadata | no hidden seed mutation behavior | `UP-007-seed-replay`, `UP-007-constructor-contract` | `DF-007-generator-seed-oracle`, `DF-007-distribution-sanity` | `E2E-007-rng-replay-pipeline` | anchor-only today (`PHASE2C_EXTRACTION_PACKET.md`) | open: implementation and fixtures absent | `bd-23m.18.1` |
| `FNP-P2C-008` | linalg bridge first wave | `legacy_numpy_code/numpy/linalg/lapack_lite/*`, wrappers in `legacy_numpy_code/numpy/linalg/*` | shape/tolerance semantics for first-wave linalg ops | exact result-shape/tolerance contract for supported ops | same API contract + bounded defensive checks | no direct lapack shim copy without Rust boundary design | `UP-008-decomp-shape`, `UP-008-residual-bounds` | `DF-008-linalg-oracle-wave1` | `E2E-008-linalg-regression-chain` | anchor-only today (`PHASE2C_EXTRACTION_PACKET.md`) | open: packet not yet implemented | `bd-23m.19.1` |
| `FNP-P2C-009` | NPY/NPZ IO contract | `legacy_numpy_code/numpy/lib/format.py`, `legacy_numpy_code/numpy/lib/npyio.py` | IO metadata compatibility, round-trip behavior, malformed-header handling | exact parse/write outcomes for supported cases | same observable IO behavior + fail-closed parser hardening | no permissive unknown-metadata fallback | `UP-009-header-parse`, `UP-009-roundtrip`, `UP-009-malformed-reject` | `DF-009-io-roundtrip-oracle`, `DF-009-header-adversarial` | `E2E-009-io-ufunc-reduce` | security control anchors in `artifacts/contracts/security_control_checks_v1.yaml`; packet directory absent | open: packet artifacts + full IO parity fixtures missing | `bd-23m.20.1` |

## 4. Open Ambiguities and Risk Ownership

| ambiguity_id | packet_id | ambiguity | risk_tier | owner | closure condition |
|---|---|---|---|---|---|
| `AMB-001` | `FNP-P2C-001` | exact compatibility boundaries for zero-copy view guarantees across all reshape edge cases | critical | `bd-23m.12.2` | contract table finalized + differential fixture pass with strict drift `0.0` |
| `AMB-002` | `FNP-P2C-002` | full legacy cast-table coverage and alias canonicalization for complete dtype matrix | critical | `bd-23m.13.2` | promotion matrix fixture corpus complete + comparator green |
| `AMB-003` | `FNP-P2C-003` | overlap/alias transfer semantics for hostile stride layouts | critical | `bd-23m.14.3` | overlap witness suite + adversarial differential fixtures pass |
| `AMB-004` | `FNP-P2C-004` | NDIter traversal ordering and index/state transition parity | high | `bd-23m.15.2` | iterator state trace fixtures + differential replay pass |
| `AMB-005` | `FNP-P2C-005` | gufunc signature parsing precedence and dispatch conflict tie-breaks | critical | `bd-23m.16.2` | signature contract table + differential tie-break fixtures pass |
| `AMB-006` | `FNP-P2C-006` | safe/hardened handling of hostile stride-trick constructions while preserving API contract | high | `bd-23m.17.3` | threat model closure + strict/hardened replay logs pass |
| `AMB-007` | `FNP-P2C-007` | generator constructor compatibility across full seed/material combinations | high | `bd-23m.18.2` | deterministic seed replay matrix verified against legacy oracle |
| `AMB-008` | `FNP-P2C-008` | linalg tolerance envelopes and fallback policy in strict/hardened modes | medium | `bd-23m.19.2` | residual/tolerance contract locked with differential residual checks |
| `AMB-009` | `FNP-P2C-009` | parser acceptance boundaries for malformed and future-version metadata | high | `bd-23m.20.3` | threat controls + adversarial parser fixtures produce fail-closed evidence |

## 5. Required Follow-Through Artifacts

For every packet, populate:

1. `artifacts/phase2c/<packet_id>/legacy_anchor_map.md`
2. `artifacts/phase2c/<packet_id>/contract_table.md`
3. `artifacts/phase2c/<packet_id>/fixture_manifest.json`
4. `artifacts/phase2c/<packet_id>/parity_gate.yaml`
5. `artifacts/phase2c/<packet_id>/risk_note.md`
6. `artifacts/phase2c/<packet_id>/parity_report.json`
7. `artifacts/phase2c/<packet_id>/parity_report.raptorq.json`
8. `artifacts/phase2c/<packet_id>/parity_report.decode_proof.json`

## 6. Release Constraint

This ledger cannot be interpreted as a reduced-scope acceptance list. It is a burn-down ledger for full legacy parity.
