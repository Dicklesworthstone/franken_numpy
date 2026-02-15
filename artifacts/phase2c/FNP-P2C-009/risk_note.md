# FNP-P2C-009 Risk Note

packet_id: `FNP-P2C-009`  
subsystem: `NPY/NPZ IO contract`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C009-RISK-01` | `malformed_magic_or_version` | Corrupted `.npy`/`.npz` prefix or unsupported version tuple crafted to bypass parser routing. | Deterministic reject with stable class. | Same class with bounded diagnostics. | contract row `P2C009-R01`; `security_control_checks_v1.yaml` -> `unknown_metadata_version`. | packet-E magic/version tests + packet-F header adversarial corpus. | `io_magic_invalid` |
| `P2C009-RISK-02` | `malformed_header_schema` | Header key/type abuse (`shape`, `fortran_order`, `descr`) including oversized header payloads. | Deterministic schema rejection with stable class. | Same class with bounded header-size controls and audit linkage. | contract row `P2C009-R02`; `security_control_checks_v1.yaml` -> `malformed_shape`. | packet-E header schema properties + packet-F malformed-header differential lane. | `io_header_schema_invalid` |
| `P2C009-RISK-03` | `descriptor_tampering` | Crafted dtype descriptors intended to coerce unsupported dtype reconstruction paths. | Reject invalid/unsupported descriptor classes deterministically. | Same class with fail-closed unknown descriptor handling. | contract row `P2C009-R03`; `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-E descriptor roundtrip laws + packet-F descriptor oracle. | `io_dtype_descriptor_invalid` |
| `P2C009-RISK-04` | `payload_truncation_or_count_drift` | Truncated payloads or inconsistent element counts intended to produce silent partial reads. | Reject incomplete payloads with stable class. | Same class with bounded read retries and deterministic reason-code emission. | contract row `P2C009-R05`; `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-E read-count invariants + packet-F truncation oracle fixtures. | `io_read_payload_incomplete` |
| `P2C009-RISK-05` | `pickle_policy_bypass` | Object-array/pickle payloads routed through unsafe paths when `allow_pickle=False`. | Strict policy reject for disallowed pickle/object classes. | Same reject class; no silent policy widening. | contract row `P2C009-R06`; runtime policy controls. | packet-E pickle-policy tests + packet-F policy differential lane. | `io_pickle_policy_violation` |
| `P2C009-RISK-06` | `memmap_contract_abuse` | Invalid memmap handles/modes/object dtypes attempting unsafe file-backed mapping behavior. | Deterministic reject of invalid memmap classes. | Same class with bounded mode/handle validation and fail-closed unknown metadata handling. | contract row `P2C009-R07`; `security_control_checks_v1.yaml` -> `malformed_shape`. | packet-E memmap safety laws + packet-F memmap oracle fixtures. | `io_memmap_contract_violation` |
| `P2C009-RISK-07` | `dispatch_confusion` | Byte-prefix manipulation to force incorrect branch among `.npz`, `.npy`, and pickle loaders. | Deterministic branch selection and reject classes. | Same dispatch classes with no widening of unsafe branch acceptance. | contract row `P2C009-R08`; runtime policy controls. | packet-E dispatch matrix + packet-F dispatch adversarial suite. | `io_load_dispatch_invalid` |
| `P2C009-RISK-08` | `archive_member_tampering` | `.npz` key collision/order abuse, malformed member payloads, or lazy-load path manipulation. | Deterministic reject/success classes for archive contracts. | Same classes with bounded member-validation/audit semantics. | contract row `P2C009-R09`; `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-E archive key tests + packet-F archive roundtrip/adversarial fixtures. | `io_npz_archive_contract_violation` |
| `P2C009-RISK-09` | `zip_resource_amplification` | Archive payloads engineered for extreme expansion or member-count amplification. | Reject unsupported/unsafe envelope classes with stable failures. | Same class with explicit budget caps and deterministic exhaustion behavior. | budget controls + security gate policy controls. | packet-F adversarial archive fixtures + workflow replay checks. | `io_npz_archive_contract_violation` |
| `P2C009-RISK-10` | `policy_override_abuse` | Attempts to force incompatible IO semantics via runtime policy override channels. | Fail-closed for unknown/incompatible semantics. | Audited-only override for allowlisted compatible cases; fail-closed otherwise. | runtime override gate + contract row `P2C009-R10`. | runtime-policy adversarial suite + security gate. | `override_*` |
| `P2C009-RISK-11` | `unknown_metadata_version` | Unknown wire mode/class metadata at IO packet boundaries. | Fail-closed. | Fail-closed with deterministic reason-code emission. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`; contract row `P2C009-R10`. | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`. | `wire_unknown_*` |
| `P2C009-RISK-12` | `corrupt_durable_artifact` | Tampered packet durability evidence (sidecar/scrub/decode-proof mismatch) for IO parity claims. | Fail validation gate on integrity mismatch. | Deterministic bounded recovery only with successful decode/hash proof. | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact`. | `validate_phase2c_packet --packet-id FNP-P2C-009`. | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible IO semantics are fail-closed in strict and hardened modes.
- Hardened mode may add bounded validation and audit enrichment, but cannot alter API-visible success/failure classes for covered contract rows.
- Recovery behavior remains explicit and deterministic: `allow`, `full_validate`, or `fail_closed`.
- Policy/replay records must include `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, and `reason_code`.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E IO parser/roundtrip corpus (planned) | detect header/schema/pickle/memmap/archive contract violations with shrinkable counterexamples | `bd-23m.20.5` |
| Differential/metamorphic | packet-F IO oracle/adversarial corpus (planned) | enforce strict parity for parse/write/dispatch/archive failure-class behavior | `bd-23m.20.6` |
| E2E/replay | packet-G IO workflow scenarios (planned) | verify strict/hardened replay traceability and policy-forensics linkage | `bd-23m.20.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C009-RES-01` | `fnp-io` remains a stub and does not yet implement packet IO boundaries. | keep unsupported IO semantics fail-closed and block parity promotion until packet-D boundary work lands | `bd-23m.20.4` + packet-E baseline tests |
| `P2C009-RES-02` | packet-scoped unit/property parser/roundtrip corpus is incomplete. | require packet-E suite with deterministic shrink/replay logging before parity claims | `bd-23m.20.5` |
| `P2C009-RES-03` | differential/adversarial coverage for malformed metadata and archive edges remains incomplete. | enforce packet-F fixture lanes and differential gate coverage | `bd-23m.20.6` |
| `P2C009-RES-04` | packet-scoped e2e replay for IO ingest/save workflows is incomplete. | add packet-G scenario logs with required structured fields and reason-code linkage | `bd-23m.20.7` |
| `P2C009-RES-05` | hardened budget/calibration thresholds are defined but not tuned for full adversarial corpus scale. | trigger conservative fallback on drift and recalibrate from packet-H evidence before packet-I sign-off | `bd-23m.20.8` + packet-I closure |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| header parse size | `<= 65_536` bytes unless explicit trusted override lane | `fail_closed` with `io_header_schema_invalid` |
| archive member count | `<= 4_096` members per `.npz` load path | `fail_closed` with `io_npz_archive_contract_violation` |
| archive uncompressed byte budget | `<= 2 GiB` decoded payload per request | abort decode with `io_npz_archive_contract_violation` |
| dispatch retry attempts | `<= 8` branch/policy retries per request | `fail_closed` with `io_load_dispatch_invalid` |
| memmap validation retries | `<= 64` validation checks per request | reject with `io_memmap_contract_violation` |
| packet-local audit payload | `<= 64 MiB` structured event payload | truncate optional diagnostics while preserving mandatory fields |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| ambiguous magic/version prefix | `{reject, full_validate}` | unsupported format admitted into parser route |
| uncertain header schema validity | `{accept_header, reject_header}` | malformed metadata accepted as valid array contract |
| pickle policy conflict | `{allow_pickle_path, reject_pickle_path}` | unsafe object payload execution under disallowed policy |
| archive member anomaly | `{load_member, reject_archive}` | silent archive corruption or key mismatch |
| unknown metadata class | `{fail_closed}` | undefined semantics admitted into runtime path |

### Calibration and fallback trigger

- Trigger fallback when either condition is true:
  - strict vs hardened packet failure-class drift exceeds `0.1%`, or
  - unknown/uncategorized reason-code rate exceeds `0.01%`.
- Fallback action: force conservative deterministic behavior (`full_validate` or `fail_closed`) until recalibration artifacts are produced and validated.

## Alien Recommendation Contract Mapping

- Graveyard mappings: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite mappings: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate: policy/optimization levers ship only if `EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction) >= 2.0`; otherwise explicit deferred parity debt remains.
- Isomorphism proof artifacts required for behavior-affecting changes:
  - ordering/tie-break note,
  - before/after golden parity checks,
  - reproducible benchmark delta artifact.
- Hotspot evidence requirement for non-doc follow-on work: attach baseline/profile artifacts before changing IO policy/optimization behavior (or document profiler-unavailable fallback).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/lib/tests/test_format.py`
- `legacy_numpy_code/numpy/numpy/lib/tests/test_io.py`
- `legacy_numpy_code/numpy/numpy/lib/_format_impl.py`
- `legacy_numpy_code/numpy/numpy/lib/_npyio_impl.py`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-009/parity_report.raptorq.json` (planned at packet-I)
- `artifacts/phase2c/FNP-P2C-009/parity_report.decode_proof.json` (planned at packet-I)
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-009/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-009` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-009`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
