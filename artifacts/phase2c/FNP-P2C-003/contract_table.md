# FNP-P2C-003 Contract Table

packet_id: `FNP-P2C-003`  
subsystem: `strided transfer semantics`

## strict_mode_policy

Strict mode preserves legacy-observable strided transfer behavior classes for loop selection, overlap/where handling, cast gating, and flatiter transfer semantics.

## hardened_mode_policy

Hardened mode preserves the same public transfer outcomes while adding bounded validation and deterministic audit logging; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full closure for every legacy string/datetime transfer edge is deferred to packet-E/F fixture expansion.
- Message-text parity is deferred; failure class and reason-code family parity is required now.
- APIs outside packet `FNP-P2C-003` are out of scope for this table.

## performance_sentinels

- transfer-loop dispatch overhead under high-rank broadcasted assignments.
- overlap mitigation/copy-path overhead in alias-heavy workloads.
- where-mask transfer throughput and mask-density sensitivity.
- flatiter slice/fancy transfer cost under non-contiguous layouts.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C003-R01` | transfer request includes `(src_dtype, dst_dtype, aligned, src_stride, dst_stride, move_refs)` | identical transfer context resolves to deterministic transfer loop class | transfer-loop selection reflects cast compatibility class | no silent alias relaxation in selector phase | deterministic accept/reject and loop-class parity | same loop class; unknown context fails closed with audit | selected transfer class is stable for fixed input context | reject unsupported transfer context | `transfer_selector_invalid_context` | `UP-003-transfer-selector-determinism` | `DF-003-transfer-selector-matrix` | `E2E-003-transfer-selector-replay` |
| `P2C003-R02` | array-to-array assignment path with potentially overlapping operands | overlap direction/copy mediation must preserve value-equivalent outcome | dtype cast still follows selected transfer path | read/write overlap cannot silently corrupt destination | overlap-sensitive operations preserve legacy behavior class | same outward result with bounded overlap-policy logging | destination matches copy-equivalent result | reject unsupported overlap configurations | `transfer_overlap_policy_triggered` | `UP-003-overlap-law` | `DF-003-overlap-oracle` | `E2E-003-overlap-replay` |
| `P2C003-R03` | where-mask transfer assignment with broadcasted mask/operands | writes occur only on true mask lanes; false mask lanes unchanged | cast semantics applied only to selected lanes | mask path must not bypass alias/overlap safety checks | deterministic masked-write parity | same behavior with deterministic diagnostics | masked output cells match expected selective update | reject invalid mask metadata/shape | `transfer_where_mask_contract_violation` | `UP-003-where-mask-isolation` | `DF-003-where-mask-oracle` | `E2E-003-where-mask-replay` |
| `P2C003-R04` | same-value cast context requested for transfer | shape/stride legality unchanged by same-value gate | lossy cast attempts are rejected; non-lossy casts pass | no alias-policy bypass via same-value path | same-value failure/success classes match legacy | same class with bounded audit and reason code | cast result/value class is deterministic for fixed input | reject lossy same-value casts | `transfer_same_value_cast_rejected` | `UP-003-same-value-cast` | `DF-003-same-value-oracle` | `E2E-003-same-value-replay` |
| `P2C003-R05` | fixed-width string/unicode transfer requires pad/truncate/copyswap path | shape/stride iteration remains deterministic | fixed-width cast behavior follows selected pad/truncate branch | no alias-policy bypass in specialized transfer path | branch selection and output class match legacy | same output class with fail-closed unknown descriptor handling | transferred payload width semantics are deterministic | reject malformed fixed-width descriptors | `transfer_string_width_mismatch` | `UP-003-string-transfer-width` | `DF-003-string-transfer-oracle` | `E2E-003-string-transfer-replay` |
| `P2C003-R06` | grouped/subarray transfer (`1->1`, `n->n`, subarray broadcast`) | grouped stride traversal preserves deterministic index mapping | grouped cast path remains class-stable | no unsafe alias transition in grouped transfer path | grouped transfer outcomes match legacy class | same outcomes; malformed grouped metadata fails closed | grouped transfer outputs remain deterministic | reject invalid grouped/subarray transfer specs | `transfer_subarray_broadcast_contract_violation` | `UP-003-grouped-transfer-laws` | `DF-003-grouped-transfer-oracle` | `E2E-003-grouped-transfer-replay` |
| `P2C003-R07` | flatiter read indexing (`int`/`slice`/`fancy`/`bool`) | indexing resolves deterministic flattened transfer order | extracted values preserve cast/transfer class behavior | read path must not mutate source alias state | supported index forms preserve legacy class behavior | same outward behavior; malformed indices fail closed | read result class is deterministic for fixed index input | reject unsupported read index forms | `flatiter_transfer_read_violation` | `UP-003-flatiter-read` | `DF-003-flatiter-read-oracle` | `E2E-003-flatiter-journey` |
| `P2C003-R08` | flatiter write assignment (`int`/`slice`/`fancy`/`bool`) | flattened write order and bounds behavior are deterministic | assignment casts follow transfer-class contract | write path enforces writeability + alias safety checks | supported assignments preserve legacy class behavior | same behavior with bounded diagnostics and policy audit | writes are deterministic for fixed index/value inputs | reject unsupported write index/value shapes | `flatiter_transfer_write_violation` | `UP-003-flatiter-write` | `DF-003-flatiter-write-oracle` | `E2E-003-flatiter-journey` |
| `P2C003-R09` | nditer path with `copy_if_overlap` and `no_broadcast` flags | iterator transfer traversal enforces documented mode constraints | cast behavior class remains stable under iterator transfer path | copy/no-copy decisions preserve overlap safety invariant | copy/no-copy behavior classes match legacy | same outcome classes with decision logging | operand outputs satisfy overlap and no-broadcast invariants | reject incompatible nditer flag combinations | `transfer_nditer_overlap_policy` | `UP-003-nditer-transfer-flags` | `DF-003-nditer-transfer-oracle` | `E2E-003-nditer-transfer-replay` |
| `P2C003-R10` | transfer loops execute under floating-point status boundaries | shape/stride traversal unaffected by FPE reporting path | cast loop FPE behavior retains class stability | alias policy unaffected by FPE checks | FPE-related transfer failure class parity retained | same class with deterministic reason-code emission | post-loop FPE reporting is deterministic | reject FPE violation path per policy | `transfer_fpe_cast_error` | `UP-003-transfer-fpe` | `DF-003-transfer-fpe-oracle` | `E2E-003-transfer-fpe-replay` |

## Logging and Failure Semantics

All packet transfer validations must emit:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `transfer_selector_invalid_context`
- `transfer_overlap_policy_triggered`
- `transfer_where_mask_contract_violation`
- `transfer_same_value_cast_rejected`
- `transfer_string_width_mismatch`
- `transfer_subarray_broadcast_contract_violation`
- `flatiter_transfer_read_violation`
- `flatiter_transfer_write_violation`
- `transfer_nditer_overlap_policy`
- `transfer_fpe_cast_error`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened transfer execution enforces explicit caps on transfer-loop resolution retries, overlap remediations, and policy overrides with deterministic exhaustion behavior.
- Expected-loss model: transfer policy decisions (especially overlap and same-value cast mediation) must record state/action/loss rationale.
- Calibration trigger: if strict/hardened transfer failure-class drift exceeds packet threshold, fallback to conservative deterministic behavior (`full_validate` or `fail_closed`) and emit audited reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If transfer contract drift is detected, revert `artifacts/phase2c/FNP-P2C-003/contract_table.md` and restore the last green packet boundary contract baseline.
