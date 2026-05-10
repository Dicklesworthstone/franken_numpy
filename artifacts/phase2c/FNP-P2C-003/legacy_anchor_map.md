# FNP-P2C-003 Legacy Anchor Map

Packet: `FNP-P2C-003`  
Subsystem: `strided transfer semantics`

## Scope

This map captures concrete legacy NumPy anchors for strided transfer loop selection, cast pipeline composition, overlap handling, where-masked assignment paths, and flatiter transfer behavior. It binds those anchors to current Rust module boundaries, packet evidence artifacts, and remaining breadth debt for FrankenNumPy packet `FNP-P2C-003`.

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

| Legacy path | Symbol/anchor | Role in observable behavior | Rust boundary and evidence |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:3089` | `PyArray_GetDTypeTransferFunction` | canonical transfer-loop resolver keyed by dtype pair, alignment, stride, and move-reference semantics | `crates/fnp-iter` transfer selector via `TransferSelectorInput`, `TransferContext`, and `select_transfer_loop`; packet-E/F evidence covers the current matrix |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:2746` | `_strided_to_strided_multistep_cast` | chained cast pipeline for wrapped/legacy multi-step transfer paths | transfer pipeline state and reason-code taxonomy in contract/risk rows; broader wrapped-cast breadth remains residual |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:460` | `PyArray_GetStridedZeroPadCopyFn` | fixed-width string/unicode transfer specialization (zero-pad/truncate/copyswap paths) | initial selector coverage exists; full fixed-width string/unicode oracle breadth remains packet-E/F residual debt |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:1597` | `_strided_to_strided_1_to_1` | direct 1:1 strided transfer primitive | packet transfer-loop decision core and unit/property evidence |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:1608` | `_strided_to_strided_n_to_n` | repeated/block transfer for n:n element group semantics | current grouped-transfer contract rows plus residual fixture expansion target |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtype_transfer.c:1798` | `_strided_to_strided_subarray_broadcast` | subarray broadcast transfer semantics in cast/assignment pipelines | current subarray contract rows plus residual fixture expansion target |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_array.c:124` | `raw_array_assign_array` + transfer resolution | array-to-array assignment with overlap-direction handling and transfer-call dispatch | overlap policy and packet-E/F overlap evidence |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_array.c:190` | `raw_array_wheremasked_assign_array` | where-masked assignment path with transfer dispatch and mask-stride iteration | where-mask invariant evidence and packet-G replay linkage |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/array_assign_scalar.c:70` | `raw_array_assign_scalar` | scalar-to-array transfer semantics including same-value casting context | same-value cast reason-code coverage and packet-E/F fixture evidence |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:574` | `iter_subscript` transfer dispatch | flatiter read path copies through dtype transfer loop for slice/fancy/bool indices | `FlatIterIndex`, `resolve_flatiter_indices`, and read validator evidence |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:854` | `iter_ass_subscript` transfer dispatch | flatiter assignment path routes writes through dtype transfer loops with cast handling | `validate_flatiter_write` evidence and packet-F flatiter cases |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/mapping.c:1019` | mapping/index assignment transfer calls | advanced indexing assignment path pulls dtype transfer loops repeatedly | iterator/ufunc integration remains residual breadth debt beyond current packet evidence |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c:3450` | nditer constructor transfer hookup | nditer buffered operation planning requests dtype transfer functions for operand loops | `NditerTransferFlags`, overlap policy evidence, and packet-G replay scenarios |

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
| `crates/fnp-iter/src/lib.rs:61` | `TransferSelectorInput` / `select_transfer_class` | implemented transfer-class selection for dtype/alignment/stride context |
| `crates/fnp-iter/src/lib.rs:80` | `FlatIterIndex` / flatiter validators | implemented flatiter read/write indexing contract with stable error taxonomy |
| `crates/fnp-iter/src/lib.rs:1684` | `TransferContext` / `select_transfer_loop` | implemented packet transfer-loop decision core with packet-E/F/G/I evidence; broader legacy transfer breadth remains residual debt |
| `crates/fnp-ufunc/src/lib.rs:73` | `UFuncArray::elementwise_binary` | current transfer/traversal behavior is manual broadcast-odometer execution |
| `crates/fnp-ufunc/src/lib.rs:184` | `contiguous_strides_elems` | local stride synthesis for transfer traversal |
| `crates/fnp-ufunc/src/lib.rs:199` | `aligned_broadcast_axis_steps` | axis-step mapping approximates transfer-loop stepping |
| `crates/fnp-ufunc/src/lib.rs:218` | `reduce_sum_axis_contiguous` | contiguous reduction traversal path with stride arithmetic |
| `crates/fnp-dtype/src/lib.rs:49` | `promote` | dtype promotion policy used by current ufunc execution path |
| `crates/fnp-dtype/src/lib.rs:65` | `can_cast_lossless` | scoped cast-safety primitive for transfer-policy planning |
| `crates/fnp-conformance/src/lib.rs:485` | `run_ufunc_differential_suite` | fixture-driven differential lane for current ufunc transfer-adjacent behavior |
| `crates/fnp-conformance/src/lib.rs:723` | `run_ufunc_adversarial_suite` | adversarial lane for broadcast/cast/input failure classes |
| `crates/fnp-conformance/src/workflow_scenarios.rs:134` | workflow fixture map + step runner | packet-003 transfer journeys are linked by `workflow_scenario_packet003_*` artifacts; additional journey breadth remains residual |
| `crates/fnp-runtime/src/lib.rs:243` | `decide_and_record_with_context` | strict/hardened policy/audit mechanism for reason-code and artifact linkage |

## Packet Evidence Artifacts

- `artifacts/phase2c/FNP-P2C-003/unit_property_evidence.json`
- `artifacts/phase2c/FNP-P2C-003/differential_metamorphic_adversarial_evidence.json`
- `artifacts/phase2c/FNP-P2C-003/e2e_replay_forensics_evidence.json`
- `artifacts/phase2c/FNP-P2C-003/optimization_profile_report.json`
- `artifacts/phase2c/FNP-P2C-003/optimization_profile_isomorphism_evidence.json`
- `artifacts/phase2c/FNP-P2C-003/parity_report.raptorq.json`
- `artifacts/phase2c/FNP-P2C-003/parity_report.scrub_report.json`
- `artifacts/phase2c/FNP-P2C-003/parity_report.decode_proof.json`
- `artifacts/phase2c/FNP-P2C-003/final_evidence_pack.json`

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Breadth

- Current packet evidence is ready and covers the landed transfer selector, overlap policy, flatiter validators, packet-E/F fixture matrix, packet-G replay linkage, packet-H optimization evidence, and packet-I durability artifacts.
- Broader parity debt remains for grouped/subarray coverage, fixed-width string/unicode zero-pad/truncate/copyswap breadth, deeper advanced-indexing transfer paths, and full `fnp-ufunc` migration onto the reusable selector stack.
- Future behavior-affecting work must attach fresh differential evidence, replay logs, and isomorphism proof before changing transfer traversal or policy behavior.
