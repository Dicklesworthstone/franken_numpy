# FNP-P2C-001 Contract Table

packet_id: `FNP-P2C-001`  
subsystem: `shape/reshape legality`

## strict_mode_policy

Strict mode must preserve legacy-observable reshape/broadcast outcomes and failure classes for the scoped packet surface with no behavior-altering repairs.

## hardened_mode_policy

Hardened mode must preserve the same public contract while adding bounded defensive validation; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full user-visible no-copy alias/view transition parity for all reshape edge cases remains deferred to packet D/E implementation closure.
- Exact full error-string textual parity (beyond class/family parity) remains deferred to packet F differential closure.
- Non-scoped NumPy APIs outside packet `FNP-P2C-001` are explicitly out of scope for this contract table.

## performance_sentinels

- `broadcast_shape` merge latency under high-rank compatible inputs.
- `contiguous_strides` throughput for large-shape C/F order computations.
- reshape legality checks under overflow-adjacent dimension products.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C001-R01` | exactly one `-1` and non-negative remaining dimensions | infer unknown dimension from `old_count / known_product` | preserve input dtype metadata (shape transform only) | no alias mutation during legality check | same infer/reject behavior as legacy | same behavior plus bounded malformed-input guards | resolved shape has no unknown dimensions and preserves element count | reject when `old_count % known_product != 0` | `reshape_unknown_dim_incompatible` | `UP-001-reshape-unknown-dim` | `DF-001-reshape-oracle-core` | `E2E-001-reshape-chain` |
| `P2C001-R02` | zero or more dimensions provided | enforce single-unknown-dimension rule | no dtype policy mutation | no alias transition | deterministic rejection on multiple unknowns | same rejection class with audit context | at most one unknown dimension accepted | fail if more than one `-1` appears | `reshape_multiple_unknown_dimensions` | `UP-001-reshape-unknown-dim` | `DF-001-reshape-oracle-core` | `E2E-001-reshape-chain` |
| `P2C001-R03` | finite dimension product and known source element count | reshape legality requires exact element-count conservation | no implicit cast during reshape legality evaluation | no-copy/copy decision point remains explicit | reject mismatched element counts | same rejection; fail-closed for overflow/malformed arithmetic | legal reshapes conserve logical element cardinality | raise mismatch failure for incompatible counts | `reshape_element_count_mismatch` | `UP-001-reshape-unknown-dim` | `DF-001-reshape-oracle-core` | `E2E-001-reshape-chain` |
| `P2C001-R04` | two valid shape vectors | right-aligned broadcast merge; dimension compatibility `(a==b) || (a==1) || (b==1)` | dtype unchanged by shape merge | broadcast legality does not alter alias ownership | deterministic output shape or deterministic incompatibility failure | same output/failure; hardened mode may add bounded validation metadata only | merged shape deterministic for fixed inputs | reject incompatible dimension pairs | `broadcast_incompatible_shapes` | `UP-001-broadcast-shape` | `DF-001-broadcast-shape` | `E2E-001-io-broadcast-reduce` |
| `P2C001-R05` | valid positive `item_size`, finite shape | C/F contiguous stride derivation with overflow checks | dtype item-size drives stride units only | stride derivation does not imply alias relaxation | deterministic C/F stride outputs for same input | identical outputs; reject hostile/overflow payloads fail-closed | stride vector length equals rank and respects order contract | fail on `item_size == 0` or overflow | `stride_invalid_item_or_overflow` | `UP-001-contig-strides` | `DF-001-broadcast-shape` | `E2E-001-reshape-chain` |
| `P2C001-R06` | legacy reshape wrapper inputs | wrapper delegates to canonical reshape legality pipeline | no cast-policy divergence in wrapper path | wrapper does not bypass alias-safety gates | wrapper behavior equivalent to core reshape path | same equivalence with hardened validation context | wrapper and canonical path produce same success/failure class | wrapper must surface canonical failure classes | `reshape_wrapper_canonical_mismatch` | `UP-001-reshape-unknown-dim` | `DF-001-reshape-oracle-core` | `E2E-001-reshape-chain` |
| `P2C001-R07` | large broadcastable shapes under platform/resource limits | preserve broadcast legality contract for high-cardinality inputs | dtype preserved | no unexpected alias/copy side effects in legality phase | output shape parity for supported high-cardinality cases | same output contract with bounded resource fail-closed behavior | deterministic shape outcome or deterministic bounded rejection | reject when bounded resource policy is exceeded | `broadcast_resource_guard_triggered` | `UP-001-broadcast-shape` | `DF-001-broadcast-shape` | `E2E-001-io-broadcast-reduce` |

## Logging and Failure Semantics

All packet validations must emit structured fields:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `reshape_unknown_dim_incompatible`
- `reshape_multiple_unknown_dimensions`
- `reshape_element_count_mismatch`
- `broadcast_incompatible_shapes`
- `stride_invalid_item_or_overflow`
- `reshape_wrapper_canonical_mismatch`
- `broadcast_resource_guard_triggered`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened execution must enforce explicit size/overflow/resource caps with deterministic fail-closed behavior.
- Expected-loss model: when hardened policy mediation occurs, decision logs must capture policy rationale and expected-loss terms.
- Calibration trigger: if drift in reshape/broadcast failure classes exceeds packet gate thresholds, automatically fallback to conservative deterministic behavior and emit explicit audit reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If contract drift is detected, revert `artifacts/phase2c/FNP-P2C-001/contract_table.md` and restore the previous packet boundary contract baseline tied to the last green differential report.
