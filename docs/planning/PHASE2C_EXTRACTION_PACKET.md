# PHASE2C_EXTRACTION_PACKET.md — FrankenNumPy

Date: 2026-02-13

Purpose: convert Phase-2 analysis into direct implementation tickets with concrete legacy anchors, target crates, and oracle tests.

## 1. Ticket Packets

| Ticket ID | Subsystem | Legacy anchors (classes/functions) | Target crates | Oracle tests |
|---|---|---|---|---|
| `FNP-P2C-001` | Shape/reshape legality | `_fix_unknown_dimension`, `_attempt_nocopy_reshape`, `PyArray_Newshape`, `PyArray_Reshape` in `numpy/_core/src/multiarray/shape.c` | `fnp-ndarray` | `numpy/_core/tests/test_shape_base.py`, `numpy/lib/tests/test_shape_base.py` |
| `FNP-P2C-002` | Dtype descriptors and promotion | `_convert_from_type` in `descriptor.c`; `dtypemeta_dealloc`, `initialize_legacy_dtypemeta_aliases` in `dtypemeta.c`; cast-table extraction from `can_cast_table.h` | `fnp-dtype` | `numpy/_core/tests/test_dtype.py`, casting tests in `_core/tests` |
| `FNP-P2C-003` | Strided transfer semantics | `_strided_to_strided_*` family, `PyArray_GetStridedZeroPadCopyFn` in `dtype_transfer.c` | `fnp-iter`, `fnp-ufunc` | `numpy/_core/tests/test_mem_overlap.py`, assignment/astype tests |
| `FNP-P2C-004` | NDIter traversal/index semantics | `NpyIter_Reset`, `NpyIter_GotoMultiIndex`, `NpyIter_GotoIndex`, `NpyIter_GetTransferFlags` in `nditer_api.c` | `fnp-iter` | `numpy/_core/tests/test_nditer.py` |
| `FNP-P2C-005` | Ufunc dispatch + gufunc signature | `_parse_signature`, `_ufunc_setup_flags`, `convert_ufunc_arguments`, `ufunc_get_name_cstr` in `ufunc_object.c` | `fnp-ufunc` | `numpy/_core/tests/test_ufunc.py`, `numpy/_core/tests/test_overrides.py`, `test_umath.py` |
| `FNP-P2C-006` | Stride-tricks and broadcasting API | `as_strided`, `sliding_window_view`, `broadcast_to`, `broadcast_shapes`, `broadcast_arrays` in `numpy/lib/_stride_tricks_impl.py` | `fnp-ndarray`, `fnp-iter` | `numpy/lib/tests/test_stride_tricks.py`, broadcast tests in `_core/tests` |
| `FNP-P2C-007` | RNG core and constructor contract | `cdef class Generator`, `default_rng` in `numpy/random/_generator.pyx` | `fnp-random` | `numpy/random/tests/test_generator_mt19937.py`, `test_seed_sequence.py`, `test_random.py` |
| `FNP-P2C-008` | Linalg bridge first wave | `lapack_lite` bridge contract in `numpy/linalg/lapack_lite/*` and high-level wrappers in `numpy/linalg/*` | `fnp-linalg` | `numpy/linalg/tests/test_linalg.py`, `test_regression.py` |
| `FNP-P2C-009` | NPY/NPZ IO contract | format loaders/writers in `numpy/lib/format.py`, `npyio.py` (scoped) | `fnp-io` | `numpy/lib/tests/test_io.py`, round-trip tests |

## 2. Packet Definition Template

For each ticket above, deliver all artifacts in the same PR:

1. `legacy_anchor_map.md`: path + line anchors + extracted behavior.
2. `contract_table.md`: input/output/error + dtype/shape semantics.
3. `fixture_manifest.json`: oracle mapping and fixture IDs.
4. `parity_gate.yaml`: strict + hardened pass criteria.
5. `risk_note.md`: boundary risks and mitigations.

## 3. Strict/Hardened Expectations per Packet

- Strict mode: exact scoped NumPy observable behavior.
- Hardened mode: same external contract with bounded defensive validation.
- Unknown incompatible metadata/cast/layout: fail-closed in both modes.

## 4. Immediate Execution Order

1. `FNP-P2C-001`
2. `FNP-P2C-002`
3. `FNP-P2C-004`
4. `FNP-P2C-003`
5. `FNP-P2C-005`
6. `FNP-P2C-006`
7. `FNP-P2C-007`
8. `FNP-P2C-008`
9. `FNP-P2C-009`

## 5. Done Criteria (Phase-2C)

- All 9 packets have extracted anchor maps and contract tables.
- At least one runnable fixture family exists per packet in `fnp-conformance`.
- Packet-level parity report schema is produced for every packet.
- RaptorQ sidecars are generated for fixture bundles and parity reports.

## 6. Per-Ticket Extraction Schema (Mandatory Fields)

Every `FNP-P2C-*` packet MUST include:
1. `packet_id`
2. `legacy_paths`
3. `legacy_symbols`
4. `shape_stride_contract`
5. `dtype_cast_contract`
6. `error_contract`
7. `memory_alias_contract`
8. `strict_mode_policy`
9. `hardened_mode_policy`
10. `excluded_scope`
11. `oracle_tests`
12. `performance_sentinels`
13. `compatibility_risks`
14. `raptorq_artifacts`

Missing any field => packet state `NOT READY`.

## 7. Risk Tiering and Gate Escalation

| Ticket | Risk tier | Why | Extra gate |
|---|---|---|---|
| `FNP-P2C-001` | Critical | reshape legality is foundational | shape-law replay |
| `FNP-P2C-002` | Critical | dtype/cast drift contaminates all ops | cast-table lockstep |
| `FNP-P2C-003` | Critical | transfer semantics can silently corrupt data | overlap/alias witness suite |
| `FNP-P2C-004` | High | iterator traversal drift affects kernel semantics | iterator-state trace checks |
| `FNP-P2C-005` | Critical | ufunc dispatch governs broad API behavior | dispatch precedence fixtures |
| `FNP-P2C-007` | High | RNG reproducibility is externally visible | deterministic-seed witness |
| `FNP-P2C-009` | High | npy/npz metadata handling has compatibility risk | malformed-header adversarial gate |

Critical tickets must pass strict drift `0`.

## 8. Packet Artifact Topology (Normative)

Directory template:
- `artifacts/phase2c/FNP-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FNP-P2C-00X/contract_table.md`
- `artifacts/phase2c/FNP-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FNP-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FNP-P2C-00X/risk_note.md`
- `artifacts/phase2c/FNP-P2C-00X/parity_report.json`
- `artifacts/phase2c/FNP-P2C-00X/parity_report.raptorq.json`
- `artifacts/phase2c/FNP-P2C-00X/parity_report.decode_proof.json`

## 9. Optimization and Isomorphism Proof Hooks

Optimization allowed only after first strict parity pass.

Mandatory proof block:
- shape semantics preserved
- dtype/cast semantics preserved
- memory-order/alias semantics preserved
- fixture checksum verification pass/fail

## 10. Packet Readiness Rubric

Packet is `READY_FOR_IMPL` only when:
1. extraction schema complete,
2. fixture manifest includes happy/edge/adversarial paths,
3. strict/hardened gates are machine-checkable,
4. risk note enumerates compatibility + security mitigations,
5. parity report has RaptorQ sidecar + decode proof.
