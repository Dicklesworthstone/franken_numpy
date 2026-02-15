# FNP-P2C-006 Legacy Anchor Map

Packet: `FNP-P2C-006`  
Subsystem: `Stride-tricks and broadcasting API`

## Scope

This map captures concrete legacy NumPy anchors for stride-tricks and broadcast behavior, then binds those anchors to FrankenNumPy module boundaries for clean-room implementation and verification planning.

## packet_id

`FNP-P2C-006`

## legacy_paths

- `legacy_numpy_code/numpy/numpy/lib/stride_tricks.py`
- `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`

## legacy_symbols

- `as_strided`
- `_broadcast_to`
- `broadcast_to`
- `_broadcast_shape`
- `broadcast_shapes`
- `broadcast_arrays`
- `npyiter_fill_axisdata`
- `NpyIter_GetShape`
- `NpyIter_CreateCompatibleStrides`

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/lib/stride_tricks.py:1` | module re-export | public API exposure for `as_strided` and `sliding_window_view` | `crates/fnp-ndarray` public stride-tricks entry points (planned under `bd-23m.17.4`) |
| `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py:38` | `as_strided` | arbitrary shape/stride view construction, explicit dtype preservation, optional writeability downgrade | `crates/fnp-ndarray` layout/view builder + `crates/fnp-runtime` hardened guard hooks |
| `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py:373` | `_broadcast_to` | iterator-backed broadcast materialization with shape validation and read-only/writeable split | `crates/fnp-ndarray` broadcast execution path + future `crates/fnp-iter` iterator adapter |
| `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py:401` | `broadcast_to` | public read-only broadcast view contract | `crates/fnp-ndarray` broadcast API (`broadcast_to`) |
| `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py:446` | `_broadcast_shape` | canonical multi-input broadcast shape resolution with large-arity chunking (`>64` args) | `crates/fnp-ndarray::broadcast_shapes` extension for high-arity parity |
| `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py:467` | `broadcast_shapes` | shape-only broadcast contract | `crates/fnp-ndarray::broadcast_shape` / `broadcast_shapes` |
| `legacy_numpy_code/numpy/numpy/lib/_stride_tricks_impl.py:515` | `broadcast_arrays` | N-array broadcasting contract with backward-compatible writeability semantics | `crates/fnp-ndarray` multi-array broadcast helpers + conformance mismatch taxonomy |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c:1463` | `npyiter_fill_axisdata` | broadcast shape synthesis, zero-stride propagation, reduction-axis detection | `crates/fnp-iter` nditer-like axisdata planner (currently stub crate) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c:1714` | `broadcast_error` | shape mismatch diagnostics when operands/requested shape cannot broadcast | `crates/fnp-iter` + `crates/fnp-runtime` normalized error-family mapping |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_constr.c:1830` | `operand_different_than_broadcast` | `NPY_ITER_NO_BROADCAST` contract and output-vs-input mismatch diagnostics | `crates/fnp-iter` non-broadcastable operand contract table |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:1001` | `NpyIter_GetShape` | iterator shape exposure semantics (multi-index aware) | `crates/fnp-iter` iterator shape introspection API |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/nditer_api.c:1059` | `NpyIter_CreateCompatibleStrides` | stride derivation aligned to iterator traversal order | `crates/fnp-iter` compatible-stride constructor boundary |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:242` | `test_broadcast_to_succeeds` | success-path broadcast-to shapes including zero-length edge cases |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:268` | `test_broadcast_to_raises` | invalid/negative broadcast target validation |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:304` | `test_broadcast_shapes_succeeds` | shape-only broadcast law including high-arity cases |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:345` | `test_broadcast_shapes_raises` | incompatible broadcast shape failures |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:362` | `test_as_strided` | dtype-preservation and custom-stride view behavior |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:580` | `test_writeable` | `broadcast_to` readonly and `broadcast_arrays` compatibility writeability behavior |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_stride_tricks.py:629` | `test_writeable_memoryview` | memoryview read-only contract for broadcasted outputs |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:2544` | `as_strided` integration case | iterator behavior with synthetic strided inputs |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:2586` | `broadcast_to` integration case | iterator traversal against broadcasted shapes |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py:3295` | zero-stride path | stride-trick edge behavior in iterator workflows |

## Current Rust Anchor Evidence

| Rust path | Anchor | Coverage note |
|---|---|---|
| `crates/fnp-ndarray/src/lib.rs:43` | `broadcast_shape` | deterministic binary broadcast shape baseline |
| `crates/fnp-ndarray/src/lib.rs:71` | `broadcast_shapes` | multi-shape merge baseline |
| `crates/fnp-ndarray/src/lib.rs:133` | `contiguous_strides` | contiguous stride construction (C/F) |
| `crates/fnp-ndarray/src/lib.rs:167` | `NdLayout` | shape+stride+item-size ownership boundary |
| `crates/fnp-conformance/src/lib.rs:207` | `run_shape_stride_suite` | fixture-driven shape/stride checks |
| `crates/fnp-conformance/fixtures/shape_stride_cases.json:1` | shape/stride fixture corpus | currently narrow; no stride-tricks API cases yet |
| `crates/fnp-iter/src/lib.rs:1` | stub crate | iterator contract implementation remains parity debt for this packet |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Contract-table stage (`bd-23m.17.2`) must lock strict/hardened behavior for writeability, broadcast mismatches, and no-broadcast operands.
- Implementation stage (`bd-23m.17.4`) must split responsibilities between `fnp-ndarray` (public API + shape law) and `fnp-iter` (iterator axisdata semantics).
- Differential stage (`bd-23m.17.6`) should prioritize broadcast edge classes: zero-size axes, high-arity broadcast-shapes, and non-broadcastable outputs.
