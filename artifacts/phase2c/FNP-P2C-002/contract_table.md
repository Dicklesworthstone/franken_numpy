# FNP-P2C-002 Contract Table

packet_id: `FNP-P2C-002`  
subsystem: `dtype descriptors and promotion`

## strict_mode_policy

Strict mode must preserve legacy-observable dtype normalization, promotion, and cast-admissibility outcomes for the scoped packet surface.

## hardened_mode_policy

Hardened mode must preserve the same public dtype/promotion contract while adding bounded defensive validation; unknown/incompatible semantics remain fail-closed.

## excluded_scope

- Full legacy-wide cast table coverage beyond current scoped dtype set is deferred to packet D/E/F closure.
- Full textual-equivalence of all dtype error messages is deferred; class/family parity remains the immediate contract.
- Non-packet APIs outside `FNP-P2C-002` are excluded from this table.

## performance_sentinels

- promotion-table lookup latency for high-frequency mixed dtype operations.
- cast-admissibility checks under repeated scalar/array coercion paths.
- dtype parse/normalization throughput for representative coercion workloads.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C002-R01` | scoped dtype identifier or type object provided | not primary for this packet; no shape mutation | descriptor normalization resolves to canonical scoped dtype | no alias transition | deterministic normalization output | same normalization + malformed metadata fail-closed | normalized dtype is canonical for supported inputs | reject unsupported/ambiguous dtype identifiers | `dtype_normalization_failed` | `UP-002-alias-normalization` | `DF-002-promotion-matrix` | `E2E-002-mixed-dtype-pipeline` |
| `P2C002-R02` | two scoped dtypes supplied for promotion | shape unchanged by promotion selection | promotion table chooses deterministic output dtype | no alias transition | deterministic promotion result | same output; unknown pairs fail-closed | promoted dtype is stable for identical input pair | reject pairs with no supported promotion path | `dtype_no_common_promotion` | `UP-002-promote-commutative` | `DF-002-promotion-matrix` | `E2E-002-mixed-dtype-pipeline` |
| `P2C002-R03` | cast-admissibility query for scoped src/dst | shape unchanged by cast policy check | lossless-cast matrix defines admissibility | no alias transition | matrix-true casts cannot be rejected | same admissibility with bounded metadata guards | cast decision matches deterministic matrix | reject unsupported cast pairs | `dtype_cast_not_lossless` | `UP-002-cast-lossless` | `DF-002-mixed-dtype-ufunc` | `E2E-002-mixed-dtype-pipeline` |
| `P2C002-R04` | descriptor metadata from known legacy alias set | shape not involved | alias descriptors resolve to canonical dtype family | no alias transition | alias mapping is deterministic and stable | same mapping; unknown aliases fail-closed | alias normalization converges to canonical family | reject malformed alias metadata | `dtype_alias_resolution_failed` | `UP-002-alias-normalization` | `DF-002-promotion-matrix` | `E2E-002-mixed-dtype-pipeline` |
| `P2C002-R05` | incompatible promotion inputs including edge scalar types | no shape mutation | failure class for unsupported promotion is stable | no alias transition | deterministic failure class | same failure class + deterministic audit context | incompatible inputs never silently coerce to invalid dtype | emit stable incompatible-promotion failure | `dtype_incompatible_promotion` | `UP-002-promote-commutative` | `DF-002-promotion-matrix` | `E2E-002-mixed-dtype-pipeline` |
| `P2C002-R06` | float/int cast paths that may overflow/underflow | shape unchanged | cast failure class remains stable under overflow invalid operations | no alias transition | cast failures are class-stable | same class with bounded diagnostics | no silent success for invalid cast class | cast failure reported deterministically | `dtype_cast_overflow_or_invalid` | `UP-002-cast-lossless` | `DF-002-mixed-dtype-ufunc` | `E2E-002-mixed-dtype-pipeline` |

## Logging and Failure Semantics

All packet validations must emit structured fields:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `dtype_normalization_failed`
- `dtype_no_common_promotion`
- `dtype_cast_not_lossless`
- `dtype_alias_resolution_failed`
- `dtype_incompatible_promotion`
- `dtype_cast_overflow_or_invalid`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened execution enforces bounded descriptor/promotion validation limits with deterministic fail-closed behavior.
- Expected-loss model: policy mediation at dtype boundaries must record rationale terms in audit logs.
- Calibration trigger: if promotion/cast drift exceeds packet thresholds, fallback to conservative deterministic policy path.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If dtype contract drift is detected, revert `artifacts/phase2c/FNP-P2C-002/contract_table.md` and restore the last green packet boundary contract baseline.
