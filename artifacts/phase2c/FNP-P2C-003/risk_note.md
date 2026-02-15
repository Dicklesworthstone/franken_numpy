# FNP-P2C-003 Risk Note

packet_id: `FNP-P2C-003`  
subsystem: `strided transfer semantics`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C003-RISK-01` | `malformed_transfer_context` | Invalid/contradictory transfer context tuples (`dtype`, `aligned`, `src_stride`, `dst_stride`, `move_refs`) intended to select wrong transfer loop class. | Deterministic reject with stable failure class. | Same class plus bounded diagnostics and audit context. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C003-R01`. | packet-E selector tests + packet-F selector oracle matrix. | `transfer_selector_invalid_context` |
| `P2C003-RISK-02` | `overlap_alias_abuse` | Crafted overlapping read/write views intended to bypass copy-direction or overlap mediation and silently corrupt destination values. | Enforce overlap-safe result class; reject unsupported overlap patterns. | Same outward result with overlap-policy audit trail; fail-closed on unknown overlap class. | `security_control_checks_v1.yaml` -> `malicious_stride_alias`; contract row `P2C003-R02`. | `test_mem_overlap` anchors + packet-E/F overlap suites + packet-G replay. | `transfer_overlap_policy_triggered` |
| `P2C003-RISK-03` | `where_mask_misuse` | Malformed/abusive where-mask shapes/strides intended to mutate masked-off lanes or bypass transfer checks. | Deterministic masked-write contract enforcement. | Same mask-visible outcome; malformed metadata fails closed with reason code. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract row `P2C003-R03`. | packet-E mask-isolation properties + packet-F where-mask differential fixtures. | `transfer_where_mask_contract_violation` |
| `P2C003-RISK-04` | `same_value_cast_bypass` | Attempts to force lossy conversion through same-value transfer mode. | Reject lossy same-value conversions with stable class. | Same class with deterministic reason-code logging and bounded diagnostics. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C003-R04`. | `test_casting_unittests` same-value anchors + packet-E/F cast suites. | `transfer_same_value_cast_rejected` |
| `P2C003-RISK-05` | `string_width_transfer_abuse` | Fixed-width string/unicode transfer payloads crafted to exploit pad/truncate/copyswap branch selection. | Deterministic transfer branch behavior with stable class outcomes. | Same semantics; unknown descriptor classes fail closed. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract row `P2C003-R05`. | packet-E string-width invariants + packet-F oracle comparisons. | `transfer_string_width_mismatch` |
| `P2C003-RISK-06` | `grouped_transfer_abuse` | Invalid grouped/subarray transfer metadata (`1->1`, `n->n`, broadcast-subarray) to desynchronize stride traversal. | Reject invalid grouped transfer context with stable class. | Same class with bounded validation and audit linkage. | `security_control_checks_v1.yaml` -> `malformed_shape`; contract row `P2C003-R06`. | packet-E grouped transfer laws + packet-F grouped transfer oracle. | `transfer_subarray_broadcast_contract_violation` |
| `P2C003-RISK-07` | `flatiter_transfer_abuse` | Malformed flatiter read/write indices and assignment payloads exploiting transfer-copy paths. | Stable success/failure classes for supported/unsupported index forms. | Same behavior with deterministic fail-closed handling for malformed forms. | `security_control_checks_v1.yaml` -> `adversarial_fixture`; contract rows `P2C003-R07`/`R08`. | packet-E flatiter unit/property + packet-F flatiter differential cases. | `flatiter_transfer_read_violation`, `flatiter_transfer_write_violation` |
| `P2C003-RISK-08` | `policy_override_abuse` | Attempts to bypass transfer fail-closed boundaries through policy override channels. | Unknown/incompatible semantics remain non-overridable and fail-closed. | Audited-only overrides for explicitly allowlisted cases; fail-closed otherwise. | `fnp_runtime::evaluate_policy_override` + security gate controls. | runtime-policy adversarial suite + security gate. | `override_*` |
| `P2C003-RISK-09` | `unknown_metadata_version` | Unknown wire mode/class metadata entering transfer policy boundaries. | Fail-closed. | Fail-closed with deterministic reason-code emission. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`. | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`. | `wire_unknown_*` |
| `P2C003-RISK-10` | `adversarial_fixture` | Poisoned packet fixtures/log metadata designed to hide transfer parity drift. | Reject malformed fixture payloads. | Reject + quarantine/audit linkage; no silent repair. | `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-F fixture schema validation + test-contract gate. | `fixture_contract_violation` |
| `P2C003-RISK-11` | `corrupt_durable_artifact` | Tampered packet evidence bundle (sidecar/scrub/decode-proof mismatch). | Fail validation gate on integrity mismatch. | Bounded deterministic recovery only with successful decode/hash proof. | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact`. | `validate_phase2c_packet --packet-id FNP-P2C-003`. | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible transfer semantics are fail-closed in strict and hardened modes.
- Hardened mode may add bounded validation and policy audit enrichment, but cannot change packet-visible success/failure class for covered contracts.
- Recovery behavior is deterministic and explicit: `allow`, `full_validate`, or `fail_closed`.
- Policy and replay records must include `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, and `reason_code`.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E transfer invariant corpus (planned) | find overlap/mask/cast/flatiter counterexamples with shrinkable traces | `bd-23m.14.5` |
| Differential/metamorphic | packet-F transfer oracle corpus (planned) | enforce strict parity for transfer success/failure classes and copy-equivalent outcomes | `bd-23m.14.6` |
| E2E/replay | packet-G transfer workflow scenarios (planned) | verify strict/hardened replay traceability and policy-forensics linkage | `bd-23m.14.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C003-RES-01` | No dedicated transfer-loop selector exists yet in `fnp-iter`; transfer behavior remains partially implicit. | keep unsupported transfer classes fail-closed and block promotion until packet-D skeleton lands | `bd-23m.14.4` + packet-E baseline tests |
| `P2C003-RES-02` | Overlap/where/same-value invariants are not yet encoded in packet-scoped unit/property suites. | require packet-E suites before claiming stable transfer parity | `bd-23m.14.5` |
| `P2C003-RES-03` | Differential oracle coverage for grouped/subarray/string-width transfer families is incomplete. | add packet-F fixture families and strict/hardened differential gates | `bd-23m.14.6` |
| `P2C003-RES-04` | Replay forensics for transfer policy decisions is not yet packet-specific. | enforce packet-G scenario logging with required structured fields | `bd-23m.14.7` |
| `P2C003-RES-05` | Hardened budget/calibration thresholds are defined but not empirically tuned to full transfer corpus scale. | trigger conservative fallback on drift; recalibrate from packet-H evidence artifacts | `bd-23m.14.8` + packet-I closure |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| transfer-loop resolution attempts | `<= 256` selector evaluations per request | `fail_closed` with `transfer_selector_invalid_context` |
| overlap remediations | `<= 128` copy-direction/remediation decisions per request | stop remediation and `fail_closed` with `transfer_overlap_policy_triggered` |
| where-mask lane validations | `<= 2_000_000` mask-lane decisions per fixture replay | abort fixture with `transfer_where_mask_contract_violation` |
| policy override evaluations | `<= 16` override checks per request | fallback to conservative default (`fail_closed`) with audited reason code |
| packet-local audit payload | `<= 64 MiB` structured event buffer | truncate optional diagnostics, preserve mandatory fields |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| ambiguous transfer context | `{reject, full_validate}` | silent selection of incompatible transfer loop |
| overlap-risk detected | `{copy_path, reject}` | destination corruption from unsafe alias writes |
| same-value cast request | `{accept_cast, reject_cast}` | lossy conversion admitted as same-value success |
| unknown metadata class | `{fail_closed}` | undefined semantics admitted into transfer path |

### Calibration and fallback trigger

- Trigger fallback when either condition is true:
  - strict vs hardened transfer failure-class drift rate exceeds `0.1%`, or
  - unknown/uncategorized transfer reason-code rate exceeds `0.01%`.
- Fallback action: force conservative deterministic path (`full_validate` or `fail_closed`) until recalibration artifacts are produced and validated.

## Alien Recommendation Contract Mapping

- Graveyard mappings: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite mappings: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate: policy/optimization levers ship only if `EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction) >= 2.0`; otherwise they remain explicit deferred parity debt.
- Isomorphism proof artifacts required for policy/optimization changes:
  - ordering/tie-break note,
  - before/after golden parity checks,
  - reproducible benchmark delta artifact.
- Hotspot evidence requirement for non-doc follow-on work: attach baseline/profile artifacts before changing transfer policy/optimization behavior (or include documented profiler-unavailable fallback).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/_core/tests/test_mem_overlap.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_nditer.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_casting_unittests.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_regression.py`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-003/parity_report.raptorq.json` (planned at packet-I)
- `artifacts/phase2c/FNP-P2C-003/parity_report.decode_proof.json` (planned at packet-I)
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-003/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-003` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-003`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
