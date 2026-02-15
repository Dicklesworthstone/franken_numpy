# FNP-P2C-005 Contract Table

packet_id: `FNP-P2C-005`  
subsystem: `ufunc dispatch + gufunc signature`

## strict_mode_policy

Strict mode preserves legacy-observable ufunc/gufunc signature parsing, override dispatch precedence, type-resolution and loop-selection determinism, and reduction wrapper behavior for the scoped packet surface.

## hardened_mode_policy

Hardened mode preserves the same public dispatch outcomes while adding bounded validation and deterministic audit semantics; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full parity for every custom-loop ecosystem edge is deferred to packet-E/F closure.
- Exact warning/error text parity is deferred; class/family parity is required now.
- APIs outside packet `FNP-P2C-005` are excluded from this table.

## performance_sentinels

- signature parse/normalization overhead for nested gufunc core-dimension expressions.
- override dispatch branching overhead in mixed `__array_ufunc__` operand sets.
- method-selection/type-resolution overhead under high-rank broadcasted operand matrices.
- reduction wrapper overhead for adversarial axis/keepdims/where combinations.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C005-R01` | ufunc call receives `sig`, `signature`, and/or `dtype` keywords | signature-conflict gate runs before dispatch planning | no cast planner side-effects when conflict is detected | conflict path cannot mutate operand buffers | dual-specification conflict classes match legacy (`sig` vs `signature`) | same class; malformed keyword payloads fail closed with audited reason code | accepted keyword state is canonicalized; rejected state terminates before dispatch | reject mutually-exclusive signature keyword combinations | `ufunc_signature_conflict` | `UP-005-signature-conflict` | `DF-005-signature-conflict-oracle` | `E2E-005-signature-dispatch-journey` |
| `P2C005-R02` | gufunc signature string/tuple provided | core-dimension grammar parse is deterministic for fixed input | parsed dtype lanes remain tied to normalized signature form | parse failure path cannot trigger alias-visible side effects | grammar accept/reject classes match legacy parser family | same class with bounded diagnostic addenda only | parse result yields deterministic normalized signature AST | reject malformed signature grammar/type classes | `ufunc_signature_parse_failed` | `UP-005-signature-grammar` | `DF-005-signature-grammar-oracle` | `E2E-005-signature-dispatch-journey` |
| `P2C005-R03` | fixed-signature resolver invoked from normalized keyword state | normalized signature representation is unique for equivalent inputs | fixed signature constrains downstream cast/type-resolution lanes deterministically | normalization cannot alter operand alias/writeability state | canonicalization outcomes match legacy behavior class | same outcomes; unknown metadata classes fail closed | equivalent user inputs map to one canonical signature form | reject unsupported fixed-signature payload classes | `ufunc_fixed_signature_invalid` | `UP-005-signature-normalization` | `DF-005-fixed-signature-oracle` | `E2E-005-signature-dispatch-journey` |
| `P2C005-R04` | one or more operands expose `__array_ufunc__` hooks | override ordering evaluation runs before built-in loop execution | override payload must preserve method/dtype contract class | override path cannot silently bypass alias/writeability preconditions | override precedence ordering matches legacy resolution order | same ordering; malformed override responses fail closed | selected path (override or built-in) is deterministic for fixed operand set | reject invalid override payload/return classes | `ufunc_override_precedence_violation` | `UP-005-override-precedence` | `DF-005-override-matrix-oracle` | `E2E-005-override-replay` |
| `P2C005-R05` | built-in dispatch path active with fixed operand dtypes + method + signature | broadcast/core-dim compatibility checks are deterministic | dispatch method selection uses deterministic promotion + resolver chain | dispatch planner cannot silently relax overlap/writeability constraints | selected loop/type-resolver class matches legacy for fixed inputs | same loop class; unknown incompatibilities fail closed | selected kernel/loop class is deterministic and replayable | reject unsatisfied dispatch constraints | `ufunc_dispatch_resolution_failed` | `UP-005-dispatch-determinism` | `DF-005-dispatch-selection-oracle` | `E2E-005-dispatch-replay` |
| `P2C005-R06` | default type resolver path engaged | shape contract from broadcast/core-dim planner remains unchanged | resolver/promotion outcome class is deterministic and class-stable | resolver path cannot mutate alias configuration | type resolver acceptance/rejection classes match legacy family | same classes with bounded policy diagnostics | resolved output dtype class matches deterministic witness | reject incompatible dtype promotion/resolution contexts | `ufunc_type_resolution_invalid` | `UP-005-type-resolution` | `DF-005-type-resolution-oracle` | `E2E-005-dispatch-replay` |
| `P2C005-R07` | gufunc loop executes for validated signature and operands | core-dimension/index traversal order follows parsed signature contract | cast behavior inside loop remains class-stable | loop exception path cannot silently commit partial invalid writes | gufunc exceptions propagate with legacy-visible failure class | same class with deterministic audit reason-code emission | successful loop execution preserves deterministic output class | reject loop/runtime exceptions per legacy error class family | `gufunc_loop_exception_propagated` | `UP-005-gufunc-exception-propagation` | `DF-005-gufunc-exception-oracle` | `E2E-005-gufunc-failure-replay` |
| `P2C005-R08` | reduction wrapper path called with `axis`, `keepdims`, optional `where` | reduced shape contract follows deterministic axis/keepdims law | reduction cast lane follows selected ufunc/reducer policy | reduction path enforces alias/writeability checks for in-place/`out` scenarios | reduction wrapper success/failure classes match legacy | same classes with bounded guardrails for hostile axis payloads | reduction output shape/value class is deterministic for fixed input | reject invalid axis/where/keepdims configuration classes | `ufunc_reduction_contract_violation` | `UP-005-reduction-wrapper` | `DF-005-reduction-wrapper-oracle` | `E2E-005-reduction-journey` |
| `P2C005-R09` | loop registration/custom-dtype pathway requested | registration metadata must satisfy dispatch planner invariants | registered loop dtype contract must be internally consistent | registration cannot bypass alias safety/writeability policy | supported registration classes behave as legacy for scoped subset | unknown/incompatible registration semantics fail closed | accepted registrations produce deterministic dispatch class for matching inputs | reject malformed/unsupported loop registration payloads | `ufunc_loop_registry_invalid` | `UP-005-loop-registry-contract` | `DF-005-loop-registry-adversarial` | `E2E-005-dispatch-replay` |
| `P2C005-R10` | runtime policy mediation receives mode/class metadata for ufunc boundary | unknown shape/signature semantics are non-admissible | unknown dtype/dispatch semantics are non-admissible | overrides cannot bypass incompatible-class policy gate | unknown/incompatible metadata fails closed | allowlisted compatible overrides may be audited in hardened mode only | decision action/reason-code is deterministic and replay-complete | reject unknown metadata and incompatible policy classes | `ufunc_policy_unknown_metadata` | `UP-005-policy-fail-closed` | `DF-005-policy-adversarial-oracle` | `E2E-005-policy-replay` |

## Logging and Failure Semantics

All packet ufunc/gufunc validations must emit:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `ufunc_signature_conflict`
- `ufunc_signature_parse_failed`
- `ufunc_fixed_signature_invalid`
- `ufunc_override_precedence_violation`
- `ufunc_dispatch_resolution_failed`
- `ufunc_type_resolution_invalid`
- `gufunc_loop_exception_propagated`
- `ufunc_reduction_contract_violation`
- `ufunc_loop_registry_invalid`
- `ufunc_policy_unknown_metadata`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened dispatch execution enforces explicit caps on signature parse complexity, override recursion depth, dispatch retry counts, and reduction validation retries with deterministic exhaustion behavior.
- Expected-loss model: override and policy-mediation decisions record state/action/loss rationale for replayable audits.
- Calibration trigger: if strict/hardened failure-class drift exceeds packet threshold, fallback to conservative deterministic behavior (`full_validate` or `fail_closed`) and emit audited reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If ufunc dispatch contract drift is detected, revert `artifacts/phase2c/FNP-P2C-005/contract_table.md` and restore the last green packet boundary contract baseline.
