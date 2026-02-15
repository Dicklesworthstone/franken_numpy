# FNP-P2C-003 Legacy Anchor Map

Packet: `FNP-P2C-003`  
Subsystem: `strided transfer semantics`

## Scope

This map captures concrete legacy NumPy anchors for strided transfer loop selection, cast pipeline composition, overlap handling, where-masked assignment paths, and flatiter transfer behavior. It binds those anchors to current and planned Rust module boundaries for FrankenNumPy packet `FNP-P2C-003`.

## packet_id

`FNP-P2C-003`

## legacy_paths

- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_array.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_scalar.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/mapping.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/convert_datatype.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_mem_overlap.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_unittests.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_regression.py`

## legacy_symbols

- `PyArray_GetDTypeTransferFunction`
- `PyArray_GetStridedZeroPadCopyFn`
- `_strided_to_strided_multistep_cast`
- `_aligned_strided_to_strided_cast`
- `_strided_to_strided_1_to_1`
- `_strided_to_strided_n_to_n`
- `_strided_to_strided_subarray_broadcast`
- `raw_array_assign_array`
- `raw_array_wheremasked_assign_array`
- `raw_array_assign_scalar`
- `iter_subscript`
- `iter_ass_subscript`

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:3089` | `PyArray_GetDTypeTransferFunction` | canonical transfer-loop resolver keyed by dtype pair, alignment, stride, and move-reference semantics | packet-D transfer-loop selector boundary in `crates/fnp-iter` + `crates/fnp-ufunc` (`bd-23m.14.4`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:2746` | `_strided_to_strided_multistep_cast` | chained cast pipeline for wrapped/legacy multi-step transfer paths | packet-D transfer pipeline state model with explicit reason-code taxonomy |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:460` | `PyArray_GetStridedZeroPadCopyFn` | fixed-width string/unicode transfer specialization (zero-pad/truncate/copyswap paths) | packet-D scoped string-transfer policy boundary + packet-F differential fixtures |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:1597` | `_strided_to_strided_1_to_1` | direct 1:1 strided transfer primitive | packet-D baseline transfer kernel contract row |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:1608` | `_strided_to_strided_n_to_n` | repeated/block transfer for n:n element group semantics | packet-D grouped transfer kernel contract row |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:1798` | `_strided_to_strided_subarray_broadcast` | subarray broadcast transfer semantics in cast/assignment pipelines | packet-D broadcasted transfer behavior boundary |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_array.c:124` | `raw_array_assign_array` + transfer resolution | array-to-array assignment with overlap-direction handling and transfer-call dispatch | packet-D/E transfer assignment skeleton + overlap witness hooks |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_array.c:190` | `raw_array_wheremasked_assign_array` | where-masked assignment path with transfer dispatch and mask-stride iteration | packet-D/E where-mask transfer contract boundary |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_scalar.c:70` | `raw_array_assign_scalar` | scalar-to-array transfer semantics including same-value casting context | packet-D scalar transfer contract boundary |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:574` | `iter_subscript` transfer dispatch | flatiter read path copies through dtype transfer loop for slice/fancy/bool indices | packet-D flatiter transfer compatibility boundary |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:854` | `iter_ass_subscript` transfer dispatch | flatiter assignment path routes writes through dtype transfer loops with cast handling | packet-D flatiter write-transfer contract boundary |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/mapping.c:1019` | mapping/index assignment transfer calls | advanced indexing assignment path pulls dtype transfer loops repeatedly | packet-D indexing-transfer bridge in iterator/ufunc integration |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c:3450` | nditer constructor transfer hookup | nditer buffered operation planning requests dtype transfer functions for operand loops | packet-D iterator planner + packet-G replay scenario linkage |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/tests/test_mem_overlap.py:67` | `test_overlapping_assignments` | overlap-safe assignment behavior under aliasing |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_mem_overlap.py:514` | `test_internal_overlap_fuzz` | adversarial internal-overlap detection corpus |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_mem_overlap.py:600` | `class TestUFunc` | ufunc overlap handling for unary/binary/reduce/reduceat paths |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_mem_overlap.py:869` | `test_unary_ufunc_where_same` | where-mask overlap behavior and copy-equivalence |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:1325` | `test_iter_copy_if_overlap` | nditer overlap-copy policy semantics |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:2603` | `test_iter_no_broadcast` | no-broadcast operand transfer constraints |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_unittests.py:726` | `test_void_and_structured_with_subarray` | structured/subarray cast transfer safety classes |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_unittests.py:836` | `test_same_value_overflow` | same-value cast rejection on lossy conversions |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_regression.py:1673` | `test_ufunc_casting_out` | output-casting behavior in ufunc transfer paths |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_regression.py:2401` | `test_eff1d_casting` | 1-D effective casting edge behavior |

## Current Rust Anchor Evidence

| Rust path | Anchor | Coverage note |
|---|---|---|
| `crates/fnp-iter/src/lib.rs:1` | placeholder `add` function | iterator crate is still a stub; no transfer planner/state machine exists yet |
| `crates/fnp-ufunc/src/lib.rs:73` | `UFuncArray::elementwise_binary` | current transfer/traversal behavior is manual broadcast-odometer execution |
| `crates/fnp-ufunc/src/lib.rs:184` | `contiguous_strides_elems` | local stride synthesis for transfer traversal |
| `crates/fnp-ufunc/src/lib.rs:199` | `aligned_broadcast_axis_steps` | axis-step mapping approximates transfer-loop stepping |
| `crates/fnp-ufunc/src/lib.rs:218` | `reduce_sum_axis_contiguous` | contiguous reduction traversal path with stride arithmetic |
| `crates/fnp-dtype/src/lib.rs:49` | `promote` | dtype promotion policy used by current ufunc execution path |
| `crates/fnp-dtype/src/lib.rs:65` | `can_cast_lossless` | scoped cast-safety primitive for transfer-policy planning |
| `crates/fnp-conformance/src/lib.rs:485` | `run_ufunc_differential_suite` | fixture-driven differential lane for current ufunc transfer-adjacent behavior |
| `crates/fnp-conformance/src/lib.rs:723` | `run_ufunc_adversarial_suite` | adversarial lane for broadcast/cast/input failure classes |
| `crates/fnp-conformance/src/workflow_scenarios.rs:134` | workflow fixture map + step runner | e2e replay scaffold exists; no packet-003-specific transfer scenarios yet |
| `crates/fnp-runtime/src/lib.rs:243` | `decide_and_record_with_context` | strict/hardened policy/audit mechanism for reason-code and artifact linkage |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Packet B (`bd-23m.14.2`) must codify strict/hardened invariants for transfer loop selection, overlap policy, where-mask semantics, and same-value casting failures.
- Packet C (`bd-23m.14.3`) must define packet-specific threat controls for alias abuse, malformed transfer metadata, and override misuse.
- Packet D (`bd-23m.14.4`) must introduce a real transfer/module skeleton in `fnp-iter` and explicit migration seams from `fnp-ufunc`.
- Packet E/F (`bd-23m.14.5`, `bd-23m.14.6`) must add unit/property + differential/adversarial suites for overlap, where-mask, same-value cast, and subarray-transfer families.
- Packet G (`bd-23m.14.7`) must add replay-forensics scenarios that emit required structured fields for transfer decisions.
