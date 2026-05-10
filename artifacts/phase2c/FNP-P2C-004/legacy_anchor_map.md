# FNP-P2C-004 Legacy Anchor Map

Packet: `FNP-P2C-004`  
Subsystem: `NDIter traversal/index semantics`

## Scope

This map captures concrete legacy NumPy anchors for iterator construction, traversal order, index/multi-index controls, external-loop behavior, overlap safety, and flat iterator indexing/assignment semantics. It binds those anchors to current and planned Rust boundaries in FrankenNumPy.

## packet_id

`FNP-P2C-004`

## legacy_paths

- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_pywrap.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c`
- `legacy_numpy_code/numpy/numpy/lib/_index_tricks_impl.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_indexing.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_index_tricks.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_multiarray.py`

## legacy_symbols

- `NpyIter_AdvancedNew`
- `NpyIter_Reset`
- `NpyIter_GotoMultiIndex`
- `NpyIter_GotoIndex`
- `NpyIter_GotoIterIndex`
- `NpyIter_GetShape`
- `NpyIter_CreateCompatibleStrides`
- `NpyIter_RemoveAxis`
- `NpyIter_RemoveMultiIndex`
- `NpyIter_EnableExternalLoop`
- `PyArray_IterNew`
- `PyArray_BroadcastToShape`
- `iter_subscript`
- `iter_ass_subscript`
- `ndindex`
- `ndenumerate`

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c:111` | `NpyIter_AdvancedNew` | canonical iterator constructor: operands/flags/op_axes/itershape normalization, dtype/casting checks, axis coalescing, buffering setup | `crates/fnp-iter/src/lib.rs` iterator constructor + planner skeleton (`bd-23m.15.4`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_pywrap.c:707` | Python `nditer(...)` arg parsing | converts Python surface (`flags`, `op_flags`, `op_axes`, `itershape`) into iterator-core inputs | `fnp-iter` API facade + strict/hardened argument validation |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_pywrap.c:330` | `no_broadcast` op-flag handling | per-operand broadcast prohibition semantics | `fnp-iter` operand policy table and error taxonomy |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:246` | `NpyIter_Reset` | iterator reset semantics, buffer lifecycle interactions | `fnp-iter` reset/state-machine contract |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:440` | `NpyIter_GotoMultiIndex` | bounds-checked multi-index seek semantics | `fnp-iter` multi-index cursor API |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:530` | `NpyIter_GotoIndex` | C/F index tracking and seek semantics | `fnp-iter` flat-index cursor API |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:618` | `NpyIter_GotoIterIndex` | traversal-order index seek/range control | `fnp-iter` iteration-index/range API |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:1001` | `NpyIter_GetShape` | iterator shape exposure, including multi-index permutation-aware behavior | `fnp-iter` shape introspection |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:1059` | `NpyIter_CreateCompatibleStrides` | constructs traversal-compatible output strides for allocated outputs | `fnp-iter` compatible-stride planner |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:30` | `NpyIter_RemoveAxis` | axis removal constraints (`multi_index` required; no buffering/index tracking) | `fnp-iter` axis-manipulation API |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:160` | `NpyIter_RemoveMultiIndex` | transition from multi-index to coalesced traversal | `fnp-iter` index-mode transition policy |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:187` | `NpyIter_EnableExternalLoop` | inner-loop/external-loop mode switch with flag compatibility checks | `fnp-iter` external-loop mode controls |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:110` | `PyArray_IterNew` | baseline flat iterator creation from ndarray | `fnp-iter` flat iterator constructor |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:147` | `PyArray_BroadcastToShape` | broadcasted iterator view with zero-stride dimensions | `fnp-iter` broadcasted iterator adapter |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:491` | `iter_subscript` | flatiter indexing surface (`int`, `slice`, `bool`, `fancy`) and error classes | `fnp-iter` flat indexing contract |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/iterators.c:746` | `iter_ass_subscript` | flatiter assignment semantics including slice/fancy/bool assignment rules | `fnp-iter` flat assignment contract |
| `legacy_numpy_code/numpy/numpy/lib/_index_tricks_impl.py:591` | `class ndenumerate` | yields `(coords, value)` from `array.flat` | `fnp-iter` `ndenumerate` convenience wrapper |
| `legacy_numpy_code/numpy/numpy/lib/_index_tricks_impl.py:641` | `class ndindex` | cartesian N-D index generator with validation | `fnp-iter` `ndindex` iterator utility |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:197` | `test_nditer_multi_index_set` | multi-index seek semantics |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:231` | `test_iter_best_order_multi_index_2d` | permutation-aware multi-index traversal |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:434` | `test_iter_best_order_c_index_2d` | C-index tracking under reordering |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:511` | `test_iter_best_order_f_index_2d` | Fortran-index tracking under reordering |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:1325` | `test_iter_copy_if_overlap` | overlap-safe copy policy |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:1421` | `test_iter_op_axes` | custom axis mapping and broadcast semantics |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:1459` | `test_iter_op_axes_errors` | op_axes validation/error classes |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:2603` | `test_iter_no_broadcast` | no-broadcast operand enforcement |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:3338` | `test_invalid_call_of_enable_external_loop` | incompatible flag-mode rejection |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_indexing.py:1450` | `test_flatiter_indexing_single_integer` | flatiter integer indexing behavior |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_indexing.py:1602` | `test_flatiter_indexing_fancy_assign` | flatiter fancy assignment behavior |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_multiarray.py:10031` | `test_flatiter__array__` | flatiter array conversion semantics |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_index_tricks.py:366` | `TestNdenumerate.test_basic` | coordinate/value iterator parity |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_index_tricks.py:552` | `test_ndindex` | ndindex basic parity vs ndenumerate |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_index_tricks.py:629` | `test_ndindex_against_ndenumerate_compatibility` | index stream equivalence |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_index_tricks.py:678` | `test_ndindex_negative_dimensions` | negative-dimension failure semantics |

## Current Rust Anchor Evidence

| Rust path | Anchor | Coverage note |
|---|---|---|
| `crates/fnp-iter/src/lib.rs:576` | `NditerPlan` | packet-D iterator planning core is landed, including shape/order/external-loop planning and deterministic step construction |
| `crates/fnp-iter/src/lib.rs:941` | `Nditer` | landed iterator state machine covers reset, seek-by-linear-index, seek-by-multi-index, `iterindex`, `multi_index`, and external-loop chunk emission |
| `crates/fnp-iter/src/lib.rs:80` | `FlatIterIndex` | flatiter single, slice, fancy, and boolean-mask contracts are modeled with stable read/write validation helpers |
| `crates/fnp-iter/src/lib.rs:389` | `ndindex` / `ndenumerate` | index-stream helpers provide deterministic coordinate and coordinate/value traversal surfaces |
| `crates/fnp-ufunc/src/lib.rs:73` | `UFuncArray::elementwise_binary` | current broadcast traversal is hand-rolled odometer logic, not reusable iterator API |
| `crates/fnp-ufunc/src/lib.rs:184` | `contiguous_strides_elems` | local stride helper used by ufunc execution path |
| `crates/fnp-ufunc/src/lib.rs:199` | `aligned_broadcast_axis_steps` | local broadcast-step synthesis for two-operand traversal |
| `crates/fnp-ufunc/src/lib.rs:218` | `reduce_sum_axis_contiguous` | contiguous reduction traversal kernel |
| `crates/fnp-conformance/tests/smoke.rs:243` | NumPy-backed NDIter smoke checks | strict iterator progression and seek behavior are compared against live NumPy states |
| `crates/fnp-conformance/fixtures/workflow_scenario_corpus.json:939` | `nditer_packet_replay` | packet-G workflow coverage now links iterator/flatiter fixtures to strict/hardened replay evidence |
| `crates/fnp-conformance/fixtures/workflow_scenario_corpus.json:1011` | `nditer_packet_hostile_guardrails` | hostile iterator guardrail workflow keeps unknown wire metadata fail-closed with packet artifact references |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Packet B (`bd-23m.15.2`) must lock strict/hardened invariant rows for index modes (`multi_index`, `c_index`, `f_index`), `external_loop`, `no_broadcast`, and overlap policy.
- Packet D (`bd-23m.15.4`) landed the `fnp-iter` planning/state core. Remaining migration work is to move ad-hoc `fnp-ufunc` traversal toward shared iterator cursors where it can be proved behavior-preserving.
- Packet E/F (`bd-23m.15.5`, `bd-23m.15.6`) now have baseline unit/property and NumPy smoke coverage; continued breadth expansion should target `op_axes`, `itershape`, overlap, hostile flatiter assignment, and version-specific diagnostic edges.
- Packet G (`bd-23m.15.7`) must attach replay-forensics scenario logs keyed by `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`.
