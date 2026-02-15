# FNP-P2C-007 Contract Table

packet_id: `FNP-P2C-007`  
subsystem: `RNG core and constructor contract`

## strict_mode_policy

Strict mode preserves legacy-observable constructor, seed-sequence, state, spawn, and jump behavior classes for the scoped RNG surface.

## hardened_mode_policy

Hardened mode preserves the same public RNG outcomes while adding bounded validation and deterministic audit semantics; unknown or incompatible semantics remain fail-closed.

## excluded_scope

- Full algorithm-by-algorithm distribution kernel parity is deferred to packet-E/F closure.
- Exact warning/error text parity is deferred; class/family parity is required now.
- APIs outside packet `FNP-P2C-007` are excluded from this table.

## performance_sentinels

- constructor normalization overhead for mixed seed input classes.
- SeedSequence `generate_state` throughput for large state word counts.
- spawn/jump throughput and state-copy overhead under high fan-out.
- state serialization/deserialization latency for replay workflows.

## Machine-Checkable Contract Rows

| contract_id | preconditions | shape_stride_contract | dtype_cast_contract | memory_alias_contract | strict_mode_policy | hardened_mode_policy | postconditions | error_contract | failure_reason_code | unit_property_ids | differential_ids | e2e_ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P2C007-R01` | constructor request via `default_rng` with supported seed classes | N/A (RNG constructor surface) | seed normalization preserves deterministic constructor class outcomes | no alias-policy bypass in constructor wiring | accepted seed classes map to stable generator construction behavior | same classes; malformed/unknown seed metadata fails closed | constructed generator class and normalized seed context are deterministic | reject unsupported/invalid seed input class | `rng_constructor_seed_invalid` | `UP-007-default-rng-constructor` | `DF-007-constructor-oracle` | `E2E-007-constructor-replay` |
| `P2C007-R02` | `Generator(bit_generator)` binding + reduce/setstate paths | N/A | constructor/state payload class handling remains deterministic | generator/bit-generator binding cannot silently diverge from contract | binding and serialization class behavior matches legacy | same class with bounded diagnostics | generator facade maintains deterministic bit-generator identity semantics | reject malformed generator state payloads | `rng_generator_binding_invalid` | `UP-007-generator-binding` | `DF-007-generator-binding-oracle` | `E2E-007-generator-lifecycle` |
| `P2C007-R03` | SeedSequence created with fixed entropy/spawn-key/pool size | N/A | `generate_state` output class deterministic for fixed inputs and dtype lane | no state-alias mutation across repeated calls for same inputs | deterministic uint32/uint64 state-word outputs | same outputs; malformed metadata fails closed | generated state words match deterministic witness for fixed input | reject invalid seed-sequence context | `rng_seedsequence_generate_state_failed` | `UP-007-seedsequence-generate-state` | `DF-007-seedsequence-reference-oracle` | `E2E-007-seedsequence-replay` |
| `P2C007-R04` | `SeedSequence.spawn(n_children)` call on spawnable sequence | N/A | spawned child sequence classes preserve deterministic lineage | child derivation must not alias parent counters incorrectly | deterministic spawn-key/counter progression class | same class with deterministic audit fields | child sequence lineage is deterministic and monotonic | reject invalid spawn request/context | `rng_seedsequence_spawn_contract_violation` | `UP-007-seedsequence-spawn-lineage` | `DF-007-seedsequence-spawn-oracle` | `E2E-007-seed-lineage-replay` |
| `P2C007-R05` | BitGenerator constructor (`MT19937`/`PCG64`/`Philox`/`SFC64`) with supported seed classes | N/A | algorithm constructor/state init class deterministic for fixed seed | no silent constructor fallback to incompatible algorithm state | constructor success/failure classes match legacy | same class with bounded validation | initialized state schema class is deterministic per algorithm | reject malformed seed/state initialization context | `rng_bitgenerator_init_failed` | `UP-007-bitgenerator-constructors` | `DF-007-bitgenerator-init-oracle` | `E2E-007-bitgenerator-init-replay` |
| `P2C007-R06` | `jumped(jumps)` invoked on jump-capable bit generators | N/A | jumped state class deterministic for fixed source state and jump count | jump partitioning must not alias source stream unexpectedly | jump behavior classes match legacy | same class with deterministic rejection for invalid jumps | jumped generator/state class is deterministic and replayable | reject invalid jump input/context | `rng_jump_contract_violation` | `UP-007-jump-determinism` | `DF-007-jump-witness-oracle` | `E2E-007-jump-replay` |
| `P2C007-R07` | state getter/setter for bit generators | N/A | valid schema roundtrip succeeds; invalid schema rejects with stable class | state application cannot silently mutate incompatible fields | deterministic state roundtrip and rejection classes | same classes with bounded diagnostics/audit linkage | valid state payload rehydrates deterministic generator state class | reject invalid state schema/class payloads | `rng_state_schema_invalid` | `UP-007-state-roundtrip` | `DF-007-state-schema-oracle` | `E2E-007-state-replay` |
| `P2C007-R08` | pickle/unpickle lifecycle for generator + seed sequence state | N/A | serialized/restored state class remains deterministic | no silent loss of seed lineage fields across serialization | serialization behavior class matches legacy | same class with fail-closed malformed payload handling | restored generator preserves seed/state class invariants | reject malformed/legacy-incompatible payload classes | `rng_pickle_state_mismatch` | `UP-007-pickle-state-preservation` | `DF-007-pickle-oracle` | `E2E-007-pickle-replay` |
| `P2C007-R09` | runtime policy mediation for RNG mode/class metadata | N/A | unknown/incompatible policy metadata remains non-admissible | no override bypass for incompatible semantics | fail-closed on unknown/incompatible semantics | audited-only override path for allowlisted compatible cases | decision action and reason code are deterministic | reject unknown metadata and incompatible policy class | `rng_policy_unknown_metadata` | `UP-007-policy-fail-closed` | `DF-007-policy-adversarial-oracle` | `E2E-007-policy-replay` |
| `P2C007-R10` | deterministic-seed witness execution for packet fixtures | N/A | fixed `(seed, mode, fixture)` yields deterministic output class | replay evidence must not alias/mismatch artifact refs | deterministic witness class required before parity promotion | same witness class + bounded diagnostics | structured logs provide replay-complete seed/artifact linkage | reject witness mismatch/drift beyond threshold | `rng_reproducibility_witness_failed` | `UP-007-seed-witness` | `DF-007-seed-witness-oracle` | `E2E-007-seed-witness-replay` |

## Logging and Failure Semantics

All packet RNG validations must emit:

- `fixture_id`
- `seed`
- `mode`
- `env_fingerprint`
- `artifact_refs`
- `reason_code`

Reason-code vocabulary for this packet:

- `rng_constructor_seed_invalid`
- `rng_generator_binding_invalid`
- `rng_seedsequence_generate_state_failed`
- `rng_seedsequence_spawn_contract_violation`
- `rng_bitgenerator_init_failed`
- `rng_jump_contract_violation`
- `rng_state_schema_invalid`
- `rng_pickle_state_mismatch`
- `rng_policy_unknown_metadata`
- `rng_reproducibility_witness_failed`

## Budgeted Mode and Expected-Loss Notes

- Budgeted mode: hardened RNG execution enforces explicit caps for constructor normalization complexity, spawn fan-out, jump operations, and policy retries with deterministic exhaustion behavior.
- Expected-loss model: constructor/state/spawn/jump policy mediation records state/action/loss rationale.
- Calibration trigger: if strict/hardened RNG failure-class drift exceeds packet threshold, fallback to conservative deterministic behavior and emit audited reason codes.

## Graveyard and FrankenSuite Mapping

- `alien_cs_graveyard.md` §0.19 (evidence ledger schema)
- `alien_cs_graveyard.md` §0.4 (decision-theoretic runtime contracts)
- `alien_cs_graveyard.md` §6.12 (property-based testing with shrinking)
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19

## Rollback Handle

If RNG contract drift is detected, revert `artifacts/phase2c/FNP-P2C-007/contract_table.md` and restore the last green packet boundary contract baseline.
