# FNP-P2C-002 Legacy Anchor Map

Packet: `FNP-P2C-002`  
Subsystem: `Dtype descriptors and promotion`

## Scope

This map captures concrete legacy NumPy dtype-descriptor and cast-safety anchors, then binds them to the clean-room Rust dtype boundary in FrankenNumPy.

## packet_id

`FNP-P2C-002`

## legacy_paths

- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/descriptor.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtypemeta.c`
- `legacy_numpy_code/numpy/numpy/_core/src/multiarray/can_cast_table.h`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_dtype.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_floatingpoint_errors.py`

## legacy_symbols

- `_convert_from_type`
- `dtypemeta_dealloc`
- `initialize_legacy_dtypemeta_aliases`
- `_npy_can_cast_safely_table`

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/descriptor.c:1550` | `_convert_from_type` | maps Python type objects to canonical dtype descriptors/defaults | `crates/fnp-dtype/src/lib.rs` dtype parse/normalization boundary (`DType::parse`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/descriptor.c:1626` | `_convert_from_any` | multi-input dispatch for dtype conversion (`None`, descriptor, type, string, tuple) | packet-D descriptor normalization plan (`bd-23m.13.4`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtypemeta.c:37` | `dtypemeta_dealloc` | lifecycle safety for dtype meta state and cast implementation storage | `fnp-dtype` metadata ownership rules + hardened reject paths |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/dtypemeta.c:1332` | `initialize_legacy_dtypemeta_aliases` | binds built-in alias dtypes and canonical numeric families | `fnp-dtype` canonical dtype taxonomy and alias policy |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/can_cast_table.h:35` | `CASTS_SAFELY_FROM_UINT` / `CASTS_SAFELY_FROM_INT` | compile-time safe-cast policy matrix primitives | `crates/fnp-dtype/src/lib.rs:65` (`can_cast_lossless`) |
| `legacy_numpy_code/numpy/numpy/_core/src/multiarray/can_cast_table.h:66` | `_npy_can_cast_safely_table` | canonical legacy safe-cast matrix used by promotion/cast logic | packet-D/E cast matrix expansion boundary (`bd-23m.13.4` / `bd-23m.13.5`) |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/tests/test_dtype.py:214` | `test_field_order_equality` (`np.can_cast(..., casting=\"safe\")`) | structured dtype cast safety semantics |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_dtype.py:1420` | `test_complex_pyscalar_promote_rational` | promotion failure class (`no common DType`) |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_dtype.py:1430` | `test_python_integer_promotion` | Python-scalar promotion default integer policy |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_dtype.py:1462` | `test_permutations_do_not_influence_result` | promotion determinism under permutation |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_floatingpoint_errors.py:143` | `test_floatingpoint_errors_casting` | cast error-class behavior under overflow/invalid operations |

## Current Rust Anchor Evidence

| Rust path | Anchor | Coverage note |
|---|---|---|
| `crates/fnp-dtype/src/lib.rs:35` | `DType::parse` | canonical scoped dtype parsing |
| `crates/fnp-dtype/src/lib.rs:49` | `promote` | deterministic promotion table for current dtype wave |
| `crates/fnp-dtype/src/lib.rs:65` | `can_cast_lossless` | deterministic lossless-cast policy subset |
| `crates/fnp-dtype/src/lib.rs:84` | promotion/cast unit tests | commutativity + matrix-smoke baseline |
| `crates/fnp-conformance/src/lib.rs:290` | `run_dtype_promotion_suite` | fixture-driven promotion conformance lane |
| `crates/fnp-conformance/fixtures/dtype_promotion_cases.json:1` | promotion fixture corpus | initial packet-family fixture baseline |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Packet B must lock strict/hardened invariant rows for descriptor normalization, promotion order, and cast-failure taxonomy.
- Packet E/F must expand from current scoped dtype subset to broader matrix coverage with adversarial and metamorphic fixtures.
- Packet G must tie dtype promotion/cast failures to replay-forensics reason codes in workflow scenarios.
