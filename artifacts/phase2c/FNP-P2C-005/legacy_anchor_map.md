# FNP-P2C-005 Legacy Anchor Map

Packet: `FNP-P2C-005`  
Subsystem: `Ufunc dispatch + gufunc signature`

## Validator Token Fields

packet_id: `FNP-P2C-005`

legacy_paths:
- `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c`
- `legacy_numpy_code/numpy/numpy/_core/src/umath/override.c`
- `legacy_numpy_code/numpy/numpy/_core/src/umath/reduction.c`
- `legacy_numpy_code/numpy/numpy/_core/src/umath/dispatching.h`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_umath.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_overrides.py`

legacy_symbols:
- `_parse_signature`
- `convert_ufunc_arguments`
- `_check_and_copy_sig_to_signature`
- `_get_fixed_signature`
- `promote_and_get_ufuncimpl`
- `normalize_signature_keyword`
- `PyUFunc_CheckOverride`
- `PyUFunc_ReduceWrapper`
- `PyUFunc_DefaultTypeResolver`
- `PyUFunc_RegisterLoopForType`

## Scope

This map captures concrete legacy NumPy anchors for ufunc/gufunc dispatch and signature handling, and binds each anchor to planned Rust module boundaries for clean-room implementation.

## Legacy-to-Rust Anchor Table

| Legacy path | Symbol/anchor | Role in observable behavior | Planned Rust boundary |
|---|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:187` | `_ufunc_setup_flags` | Iterator/op-flag setup that affects iteration semantics and buffering behavior | `crates/fnp-ufunc` dispatch policy layer (`dispatch_flags`) |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:307` | `_parse_signature` | gufunc signature grammar parse and core-dim bookkeeping | `crates/fnp-ufunc` gufunc signature parser (`signature_parser`) |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:543` | `ufunc_get_name_cstr` | stable ufunc naming for diagnostics and error reporting | `crates/fnp-ufunc` diagnostic helpers |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:613` | `convert_ufunc_arguments` | positional/keyword operand normalization before dispatch | `crates/fnp-ufunc` call normalization (`argument_normalizer`) |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:3782` | `_check_and_copy_sig_to_signature` | conflict rules for `sig`, `signature`, and `dtype` parameters | `crates/fnp-ufunc` signature conflict validator |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:3873` | `_get_fixed_signature` | parse/normalize concrete DType signature tuple/string forms | `crates/fnp-ufunc` fixed signature resolver |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:2339` | `promote_and_get_ufuncimpl` call site | dispatch method selection from dtype promotion + signature constraints | `crates/fnp-ufunc` method selection planner |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:4688` | `promote_and_get_ufuncimpl` call site | core `__call__` dispatch path for ufunc execution | `crates/fnp-ufunc` execution planner |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/dispatching.h:23` | `promote_and_get_ufuncimpl` declaration | canonical dispatch API shape and contract | `crates/fnp-ufunc` trait-based kernel resolver |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/override.c:143` | `normalize_signature_keyword` | `sig` to `signature` canonicalization (public API compatibility) | `crates/fnp-ufunc` keyword normalization stage |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/override.c:206` | `PyUFunc_CheckOverride` | `__array_ufunc__` override dispatch, including method variants | `crates/fnp-runtime` override policy + `crates/fnp-ufunc` override hook |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/reduction.c:178` | `PyUFunc_ReduceWrapper` | reduction wrapper semantics (axis/keepdims/wheremask pipeline) | `crates/fnp-ufunc` reduction adapter |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:4831` | `PyUFunc_FromFuncAndDataAndSignatureAndIdentity` | constructor for ufunc objects with signature+identity metadata | `crates/fnp-ufunc` ufunc metadata registry |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:4890` | `PyUFunc_DefaultTypeResolver` binding | default type resolution policy assignment | `crates/fnp-dtype` + `crates/fnp-ufunc` type resolution bridge |
| `legacy_numpy_code/numpy/numpy/_core/src/umath/ufunc_object.c:5172` | `PyUFunc_RegisterLoopForType` | loop registration semantics for custom/user dtypes | `crates/fnp-ufunc` loop registry API |

## Oracle Test Anchors

| Test path | Anchor | Behavior family |
|---|---|---|
| `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py:51` | `test_sig_signature` | `sig`/`signature` argument compatibility and override behavior |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py:426` | `test_signature_failure_*` block | signature grammar parse failures and diagnostics |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py:553` | `test_signature_errors` | invalid signature type/value error contracts |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py:3393` | `test_params_common_gufunc` | gufunc inspect/signature parameter model |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_umath.py:3447` | `test_ufunc_override` family | override dispatch semantics (`__array_ufunc__`) |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_umath.py:3973` | `test_gufunc_override` | gufunc override path consistency |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_umath.py:5094` | `test_bad_legacy_gufunc_silent_errors` | gufunc loop exception propagation |
| `legacy_numpy_code/numpy/numpy/_core/tests/test_overrides.py:271` | `test_verify_matching_signatures` | override signature matching contract |

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Notes for Follow-on Packet Steps

- Strict/hardened split must remain explicit at override and signature conflict boundaries.
- Dispatch choices must be evidence-ledgered for replayable failure forensics where policy decisions occur.
- Differential corpus should prioritize signature parse failures, override precedence, and gufunc reduction edge cases before optimization work.
