# FNP-P2C-007 Risk Note

packet_id: `FNP-P2C-007`  
subsystem: `RNG core and constructor contract`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C007-RISK-01` | `malformed_seed_input` | Constructor seeds crafted with unsupported type/range/schema combinations to force ambiguous initialization paths. | Deterministic reject with stable failure class. | Same class plus bounded diagnostics/audit context. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C007-R01`. | packet-E constructor tests + packet-F constructor oracle matrix. | `rng_constructor_seed_invalid` |
| `P2C007-RISK-02` | `seedsequence_state_abuse` | Entropy/spawn-key/pool payload abuse intended to break deterministic `generate_state` output contracts. | Reject malformed seed-sequence context; deterministic output for valid input. | Same class with bounded validation and reason-code logging. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract row `P2C007-R03`. | packet-E SeedSequence laws + packet-F reference-vector differential suite. | `rng_seedsequence_generate_state_failed` |
| `P2C007-RISK-03` | `spawn_lineage_abuse` | Spawn-key/counter manipulation attempts to collide child streams or desynchronize lineage accounting. | Enforce deterministic spawn lineage and reject invalid requests. | Same lineage behavior with bounded audit records. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`; contract row `P2C007-R04`. | packet-E spawn-lineage properties + packet-F spawn oracle fixtures. | `rng_seedsequence_spawn_contract_violation` |
| `P2C007-RISK-04` | `jump_partition_abuse` | Malformed jump counts/state payloads intended to violate deterministic stream partition boundaries. | Deterministic reject of invalid jumps; stable class for valid jumps. | Same class with bounded diagnostics and replay linkage. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract row `P2C007-R06`. | jumped witness tests + packet-F jump differential lanes. | `rng_jump_contract_violation` |
| `P2C007-RISK-05` | `state_schema_tampering` | Invalid or tampered state payload schemas injected via state setter or constructor restore flows. | Reject invalid schema classes with stable failure taxonomy. | Same class with fail-closed handling and deterministic reason code. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C007-R07`. | packet-E state-roundtrip tests + packet-F schema differential fixtures. | `rng_state_schema_invalid` |
| `P2C007-RISK-06` | `pickle_payload_abuse` | Malicious pickle payloads targeting seed/state restoration pathways to bypass deterministic initialization contracts. | Reject malformed payload classes; stable restore behavior for valid payloads. | Same behavior with bounded payload validation and audit linkage. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract row `P2C007-R08`. | pickle preservation tests + packet-G replay scenarios. | `rng_pickle_state_mismatch` |
| `P2C007-RISK-07` | `reproducibility_drift` | Output drift for identical `(seed, fixture, mode)` due to hidden nondeterminism or state leakage. | Deterministic witness mismatch is treated as failure. | Same mismatch class with deterministic fallback to conservative mode. | contract row `P2C007-R10`; reproducibility policy controls. | packet-E seed witness tests + packet-F oracle + packet-G replay checks. | `rng_reproducibility_witness_failed` |
| `P2C007-RISK-08` | `policy_override_abuse` | Attempts to force unknown/incompatible RNG semantics through override channels. | No override for incompatible/unknown semantics; fail-closed. | Audited-only override for explicitly allowlisted compatible cases; fail-closed otherwise. | `fnp_runtime::evaluate_policy_override` + security gate controls. | runtime-policy adversarial suite + security gate. | `override_*` |
| `P2C007-RISK-09` | `unknown_metadata_version` | Unknown wire mode/class metadata at RNG policy boundaries. | Fail-closed. | Fail-closed with deterministic reason-code emission. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`; contract row `P2C007-R09`. | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`. | `wire_unknown_*` |
| `P2C007-RISK-10` | `adversarial_fixture` | Poisoned RNG fixture metadata intended to hide constructor/state/spawn/jump parity drift. | Reject malformed fixture payloads. | Reject + quarantine/audit linkage with no silent recovery. | `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-F fixture schema validation + test-contract gate. | `fixture_contract_violation` |
| `P2C007-RISK-11` | `corrupt_durable_artifact` | Tampered packet evidence (sidecar/scrub/decode-proof mismatch) for RNG parity claims. | Fail validation gate on integrity mismatch. | Deterministic bounded recovery only with successful decode/hash proof. | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact`. | `validate_phase2c_packet --packet-id FNP-P2C-007`. | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible RNG semantics are fail-closed in strict and hardened modes.
- Hardened mode may add bounded validation/audit detail, but cannot change packet-visible success/failure class for covered contracts.
- Recovery behavior remains explicit and deterministic: `allow`, `full_validate`, or `fail_closed`.
- Policy/replay records must include `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, and `reason_code`.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E RNG invariant corpus (planned) | detect constructor/state/spawn/jump determinism and schema edge violations | `bd-23m.18.5` |
| Differential/metamorphic | packet-F RNG oracle corpus (planned) | enforce strict parity for constructor/state/jump witness classes | `bd-23m.18.6` |
| E2E/replay | packet-G RNG workflow scenarios (planned) | verify strict/hardened replay traceability and seed lineage forensics | `bd-23m.18.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C007-RES-01` | `fnp-random` has no concrete constructor/state/spawn/jump implementation yet. | keep unsupported RNG semantics fail-closed and block promotion until packet-D lands | `bd-23m.18.4` + packet-E baseline tests |
| `P2C007-RES-02` | deterministic-seed witness suites are not yet encoded in packet-scoped tests. | require packet-E witness suites before parity claims | `bd-23m.18.5` |
| `P2C007-RES-03` | RNG differential oracle coverage for state schema and jump semantics is incomplete. | add packet-F fixture families and strict/hardened differential gates | `bd-23m.18.6` |
| `P2C007-RES-04` | replay forensics for seed lineage is not yet packet-specific. | enforce packet-G scenario logging with required structured fields | `bd-23m.18.7` |
| `P2C007-RES-05` | hardened budget/calibration thresholds are defined but not empirically tuned for full RNG corpus scale. | trigger conservative fallback on drift; recalibrate using packet-H evidence | `bd-23m.18.8` + packet-I closure |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| constructor normalization attempts | `<= 256` normalization evaluations per request | `fail_closed` with `rng_constructor_seed_invalid` |
| spawn fan-out | `<= 4096` child sequences per spawn request | reject with `rng_seedsequence_spawn_contract_violation` |
| jump operations | `<= 1024` jump operations per fixture replay | abort replay with `rng_jump_contract_violation` |
| state schema validation fields | `<= 4096` schema entries per state payload | reject with `rng_state_schema_invalid` |
| policy override evaluations | `<= 16` override checks per request | fallback to conservative default (`fail_closed`) with audited reason code |
| packet-local audit payload | `<= 64 MiB` structured event buffer | truncate optional diagnostics, preserve mandatory fields |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| ambiguous constructor seed class | `{reject, full_validate}` | nondeterministic initialization admitted as valid constructor path |
| seed sequence lineage mutation | `{accept_spawn, reject_spawn}` | child stream collision / lineage corruption |
| jump request on current state | `{accept_jump, reject_jump}` | stream partition drift and reproducibility loss |
| unknown metadata class | `{fail_closed}` | undefined semantics admitted into RNG policy path |

### Calibration and fallback trigger

- Trigger fallback when either condition is true:
  - strict vs hardened RNG failure-class drift rate exceeds `0.1%`, or
  - unknown/uncategorized RNG reason-code rate exceeds `0.01%`.
- Fallback action: force conservative deterministic path (`full_validate` or `fail_closed`) until recalibration artifacts are produced and validated.

## Alien Recommendation Contract Mapping

- Graveyard mappings: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite mappings: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate: policy/optimization levers ship only if `EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction) >= 2.0`; otherwise explicit deferred parity debt remains.
- Isomorphism proof artifacts required for behavior-affecting changes:
  - ordering/tie-break note,
  - before/after golden parity checks,
  - reproducible benchmark delta artifact.
- Hotspot evidence requirement for non-doc follow-on work: attach baseline/profile artifacts before changing RNG policy/optimization behavior (or document profiler-unavailable fallback).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/random/tests/test_seed_sequence.py`
- `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937.py`
- `legacy_numpy_code/numpy/numpy/random/tests/test_random.py`
- `legacy_numpy_code/numpy/numpy/random/tests/test_generator_mt19937_regressions.py`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-007/parity_report.raptorq.json` (planned at packet-I)
- `artifacts/phase2c/FNP-P2C-007/parity_report.decode_proof.json` (planned at packet-I)
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-007/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-007` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-007`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
