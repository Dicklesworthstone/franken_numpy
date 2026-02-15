# FNP-P2C-006 Contract Table

packet_id: `FNP-P2C-006`  
subsystem: `stride-tricks and broadcasting API`

## strict_mode_policy

Strict mode must preserve legacy-observable stride-tricks and broadcasting outcomes (including failure classes and readonly/writeable behavior) for the scoped packet surface.

## hardened_mode_policy

Hardened mode must preserve the same public contract while adding bounded validation and auditability; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full `nditer` parity across all advanced buffering and per-op flag combinations is deferred to later packet stages (`D`/`E`/`F`).
- Exact byte-for-byte warning text parity for legacy writeability deprecation warnings is deferred; class/family parity is the immediate contract.
- APIs outside packet `FNP-P2C-006` are excluded from this contract table.

## performance_sentinels

- `broadcast_shapes` high-arity merge latency and allocation behavior.
- `broadcast_to`/`broadcast_arrays` view construction overhead under rank-mismatched operands.
- iterator-compatible stride planning cost for multi-axis broadcast layouts.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C006-R01` | input array + requested target shape supplied to `broadcast_to` | requested shape must be non-negative and broadcast-compatible with operand | dtype identity preserved | result is a broadcast view (not materialized copy by contract) | same accept/reject class as legacy for valid/invalid target shapes | same accept/reject class with bounded diagnostics | accepted output shape equals requested shape | reject negative/incompatible/non-scalar-to-scalar targets | `broadcast_to_shape_invalid` | `UP-006-broadcast-to-shape-validation` | `DF-006-broadcast-to-oracle` | `E2E-006-stride-tricks-journey` |
| `P2C006-R02` | one or more shapes supplied to `broadcast_shapes` | right-aligned merge law: `(a==b) || (a==1) || (b==1)`; supports high-arity input tuples | dtype not involved | no alias transition (shape-only contract) | deterministic merged shape or deterministic failure | same result/failure; unknown malformed shape payloads fail-closed | merged shape unique for fixed ordered input tuple | reject incompatible shape tuples | `broadcast_shapes_incompatible` | `UP-006-broadcast-shapes-determinism` | `DF-006-broadcast-shapes-high-arity` | `E2E-006-stride-tricks-journey` |
| `P2C006-R03` | `broadcast_to` returns a view for compatible target | shape contract as in `R01` | dtype preserved | returned view is readonly in strict mode | readonly flag preserved and assignment attempts fail | same readonly semantics; hostile write attempts logged | no accidental writeable upgrade for broadcasted view | assignment to result must fail with stable error class | `broadcast_to_readonly_violation` | `UP-006-broadcast-to-readonly` | `DF-006-broadcast-to-oracle` | `E2E-006-stride-tricks-journey` |
| `P2C006-R04` | `broadcast_arrays` invoked on one or more operands | each operand is expanded to merged broadcast shape if needed | dtype preserved per operand | broadcasted outputs may be writeable for compatibility path; memoryview contract remains conservative | compatibility writeability behavior follows legacy class | same observable behavior; hardened logs reason code for unsafe-write path selection | output tuple shape aligns across all returned arrays | reject incompatible operands; preserve warning/error class families | `broadcast_arrays_writeability_contract` | `UP-006-broadcast-arrays-writeability` | `DF-006-broadcast-arrays-compat` | `E2E-006-stride-tricks-journey` |
| `P2C006-R05` | `as_strided` called with explicit/implicit shape+strides | provided shape/stride metadata defines resulting logical layout | dtype must remain unchanged (including structured/custom dtypes) | overlapping/self-aliasing views are permitted by API contract but must not silently relax safeguards | preserve dtype/subclass/writeability semantics for scoped cases | preserve semantics; overlap-risk operations trigger bounded audit path in hardened mode | resulting view shape/strides exactly match supplied metadata | reject invalid metadata combinations per scoped contract class | `as_strided_contract_violation` | `UP-006-as-strided-dtype-subclass` | `DF-006-as-strided-oracle` | `E2E-006-stride-tricks-journey` |
| `P2C006-R06` | iterator constructed over broadcasted operands (`nditer`-class path) | broadcast dimensions imply zero-stride iteration where valid; `NPY_ITER_NO_BROADCAST` forbids implicit expansion | dtype unchanged by iteration planning | non-broadcastable operands must not be remapped silently | preserve broadcast/no-broadcast and reduction-axis class behavior | same behavior with fail-closed handling for malformed remap metadata | axisdata and iterator shape are internally consistent | emit non-broadcastable operand mismatch failures deterministically | `iterator_non_broadcastable_operand` | `UP-006-iterator-axisdata-law` | `DF-006-iterator-broadcast-mismatch` | `E2E-006-iterator-broadcast-replay` |
| `P2C006-R07` | iterator introspection requested (`shape`/compatible strides) | exposed iteration shape and compatible strides must match traversal semantics | dtype unaffected | no alias policy mutation in introspection path | introspection outputs follow legacy traversal law | same law; unsupported mode/class combinations fail-closed | compatible stride output is deterministic for fixed iterator state | reject invalid introspection preconditions with stable class | `iterator_shape_stride_invariant_failed` | `UP-006-iterator-shape-compatible-strides` | `DF-006-iterator-shape-introspection` | `E2E-006-iterator-broadcast-replay` |

## Logging and Failure Semantics

All packet validations must emit structured fields:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `broadcast_to_shape_invalid`
- `broadcast_shapes_incompatible`
- `broadcast_to_readonly_violation`
- `broadcast_arrays_writeability_contract`
- `as_strided_contract_violation`
- `iterator_non_broadcastable_operand`
- `iterator_shape_stride_invariant_failed`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened execution enforces bounded validation on shape/remap metadata and dangerous writeability pathways with deterministic fail-closed exhaustion behavior.
- Expected-loss model: policy-mediated decisions on overlap-risk/writeability paths must record rationale terms in audit logs.
- Calibration trigger: if drift in packet reason-code distribution or strict/hardened mismatch rate exceeds gate thresholds, fallback to conservative deterministic policy path.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If stride-tricks/broadcasting contract drift is detected, revert `artifacts/phase2c/FNP-P2C-006/contract_table.md` and restore the last green packet boundary contract baseline.
