# FNP-P2C-001 Legacy Anchor Map

Packet: `FNP-P2C-001`  
Subsystem: `Shape/reshape legality`

## Scope

This map captures concrete legacy NumPy anchors for reshape legality, `-1` inference, broadcast shape rules, and contiguous stride derivation, then binds them to current/planned Rust boundaries in the clean-room implementation.

## packet_id

`FNP-P2C-001`

## legacy_paths

- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_shape_base.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_shape_base.py`

## legacy_symbols

- `_fix_unknown_dimension`
- `_attempt_nocopy_reshape`
- `PyArray_Newshape`
- `PyArray_Reshape`

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c:200` | `PyArray_Newshape` | Public reshape entrypoint; enforces reshape order policy and delegates legality checks | `crates/fnp-ndarray/src/lib.rs` reshape legality kernel (`fix_unknown_dimension`, `element_count`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c:246` | `_fix_unknown_dimension` callsite | Applies single-unknown-dimension inference and element-count consistency gate | `crates/fnp-ndarray/src/lib.rs:85` (`fix_unknown_dimension`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c:271` | `_attempt_nocopy_reshape` callsite | Determines whether reshape can preserve buffer aliasing without copy | Packet-D implementation boundary (`bd-23m.12.4`) in `fnp-ndarray` |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c:325` | `PyArray_Reshape` | Backward-compatible reshape API wrapper over `PyArray_Newshape` | `fnp-ndarray` public reshape adapter (planned in packet D) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c:360` | `_attempt_nocopy_reshape` | Alias-sensitive stride recomputation logic for no-copy transitions | `fnp-ndarray` alias/view legality module (planned in packet D/E) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/shape.c:467` | `_fix_unknown_dimension` | Canonical `-1` inference + size mismatch rejection (`cannot reshape ...`) | `crates/fnp-ndarray/src/lib.rs:85` with `ShapeError::IncompatibleElementCount` |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/lib/tests/test_shape_base.py:86` | `TestTakeAlongAxis.test_broadcast` | right-aligned broadcast expansion |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_shape_base.py:111` | `TestPutAlongAxis.test_broadcast` | bidirectional broadcast with indexed writes |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_shape_base.py:450` | `test_integer_split_2D_rows_greater_max_int32` | high-cardinality broadcast shape handling |
| `legacy_numpy_code/numpy/numpy/lib/tests/test_shape_base.py:657` | `TestSqueeze.test_basic` | reshape/squeeze shape-equivalence expectations |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_shape_base.py:311` | `TestConcatenate.test_concatenate_axis_None` | flatten/reshape axis semantics across concatenate path |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_shape_base.py:344` | `TestConcatenate.test_concatenate` | multi-dimensional reshape + concatenate shape consistency |

## Current Rust Anchor Evidence

| Rust path | Anchor | Coverage note |
|---|---|---|
| `crates/fnp-ndarray/src/lib.rs:43` | `broadcast_shape` / `broadcast_shapes` | deterministic broadcast legality and incompatibility rejection |
| `crates/fnp-ndarray/src/lib.rs:85` | `fix_unknown_dimension` | single `-1` inference and element-count compatibility checks |
| `crates/fnp-ndarray/src/lib.rs:133` | `contiguous_strides` | C/F contiguous stride derivation with overflow checks |
| `crates/fnp-ndarray/src/lib.rs:223` | stride + `-1` inference unit tests | packet-local happy/edge behavior witnesses |
| `crates/fnp-conformance/src/lib.rs:207` | `run_shape_stride_suite` | fixture-driven conformance check for broadcast + stride cases |
| `crates/fnp-conformance/fixtures/shape_stride_cases.json:1` | `shape_stride` fixture corpus | initial differential-like corpus for packet family |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Packet B must formalize strict/hardened invariant tables for alias-sensitive reshape transitions currently tracked as open ambiguity.
- Packet E/F must expand beyond current fixture slices to full differential/metamorphic/adversarial coverage for reshape legality and broadcast edge matrices.
- Packet G must attach replay-ready scenario traces linking shape-law failures to fixture IDs and reason codes.
