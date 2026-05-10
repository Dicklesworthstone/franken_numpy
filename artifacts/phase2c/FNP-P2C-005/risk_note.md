# FNP-P2C-005 Risk Note

packet_id: `FNP-P2C-005`  
subsystem: `ufunc dispatch + gufunc signature`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C005-RISK-01` | `signature_conflict_injection` | Crafted calls provide conflicting `sig` and `signature` (and/or incompatible `dtype`) to force ambiguous dispatch state. | Deterministic reject with legacy-compatible failure class. | Same failure class with bounded diagnostics and audited context. | contract row `P2C005-R01`; `security_control_checks_v1.yaml` -> `malformed_shape`. | packet-E signature conflict tests + packet-F conflict differential corpus. | `ufunc_signature_conflict` |
| `P2C005-RISK-02` | `signature_grammar_abuse` | Malformed gufunc signature strings/tuples designed to trigger parser ambiguity or unstable failure classes. | Deterministic parse reject; stable error-class family. | Same class with bounded diagnostic addenda only. | contract rows `P2C005-R02`/`R03`; `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-E signature grammar property corpus + packet-F signature oracle fixtures. | `ufunc_signature_parse_failed`, `ufunc_fixed_signature_invalid` |
| `P2C005-RISK-03` | `override_precedence_abuse` | Mixed `__array_ufunc__` operand sets crafted to invert legacy override ordering or smuggle invalid override payloads. | Preserve legacy precedence; reject malformed override return classes. | Same precedence with fail-closed malformed payload handling and reason-code logging. | contract row `P2C005-R04`; `fnp_runtime::evaluate_policy_override`. | packet-E override precedence matrix + packet-F override differential lane. | `ufunc_override_precedence_violation` |
| `P2C005-RISK-04` | `dispatch_resolution_confusion` | Adversarial dtype/signature/method combinations intended to induce non-deterministic kernel/loop selection. | Deterministic dispatch selection or deterministic rejection class. | Same outward behavior; unknown/incompatible semantics fail closed. | contract row `P2C005-R05`; dispatch control checks. | packet-E dispatch determinism tests + packet-F dispatch oracle matrix. | `ufunc_dispatch_resolution_failed` |
| `P2C005-RISK-05` | `type_resolution_poisoning` | Payloads targeting promotion/type-resolver pathways to coerce unstable dtype outcomes. | Preserve deterministic type-resolution success/failure class. | Same class with bounded guardrails and audited reason codes. | contract row `P2C005-R06`; `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-E type-resolution properties + packet-F type oracle fixtures. | `ufunc_type_resolution_invalid` |
| `P2C005-RISK-06` | `gufunc_exception_suppression` | Inputs attempting to trigger gufunc loop exceptions that get swallowed or partially committed. | Exception propagation class remains visible and deterministic. | Same class with deterministic policy audit linkage. | contract row `P2C005-R07`; `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-E gufunc exception tests + packet-F gufunc error oracle. | `gufunc_loop_exception_propagated` |
| `P2C005-RISK-07` | `reduction_axis_payload_abuse` | Hostile `axis`/`keepdims`/`where` payloads intended to bypass reduction wrapper bounds and semantic checks. | Stable reject/success classes for reduction wrapper semantics. | Same classes with bounded axis/where validation and fail-closed unknown classes. | contract row `P2C005-R08`; `security_control_checks_v1.yaml` -> `malformed_shape`. | packet-E reduction wrapper laws + packet-F reduction differential lane + packet-G replay checks. | `ufunc_reduction_contract_violation` |
| `P2C005-RISK-08` | `loop_registry_tampering` | Malformed custom-loop registration metadata for user dtypes intended to bypass dispatch invariants. | Reject unsupported/malformed registration classes deterministically. | Same class; unknown registration semantics fail closed. | contract row `P2C005-R09`; `security_control_checks_v1.yaml` -> `unknown_metadata_version`. | packet-E loop-registry contract tests + packet-F adversarial registry fixtures. | `ufunc_loop_registry_invalid` |
| `P2C005-RISK-09` | `policy_override_abuse` | Attempts to admit incompatible semantics through runtime override channels. | No override for unknown/incompatible semantics; fail-closed. | Audited-only override for explicitly allowlisted compatible cases; fail-closed otherwise. | runtime override gate + contract row `P2C005-R10`. | runtime-policy adversarial suite + security gate. | `override_*` |
| `P2C005-RISK-10` | `unknown_metadata_version` | Unknown wire mode/class metadata at ufunc packet policy boundaries. | Fail-closed. | Fail-closed with deterministic reason-code emission. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`; contract row `P2C005-R10`. | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`. | `wire_unknown_*` |
| `P2C005-RISK-11` | `adversarial_fixture` | Poisoned fixture payloads designed to hide signature/dispatch/reduction parity drift. | Reject malformed fixture bundles. | Reject + quarantine/audit references with no silent recovery. | `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-F fixture schema validation + test-contract gate. | `fixture_contract_violation` |
| `P2C005-RISK-12` | `corrupt_durable_artifact` | Tampered packet durability artifacts (sidecar/scrub/decode-proof mismatch) for ufunc parity claims. | Fail validation gate on integrity mismatch. | Deterministic bounded recovery only with successful decode/hash proof. | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact`. | `validate_phase2c_packet --packet-id FNP-P2C-005`. | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible ufunc/gufunc semantics are fail-closed in strict and hardened modes.
- Hardened mode may add bounded validation and audit enrichment, but cannot alter API-visible success/failure class for covered contract rows.
- Recovery behavior remains explicit and deterministic: `allow`, `full_validate`, or `fail_closed`.
- Policy and replay records must include `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, and `reason_code`.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E ufunc signature/override/reduction invariant corpus (landed for current scope) | detect parser, precedence, reduction-shape, and loop-registry contract violations | `artifacts/phase2c/FNP-P2C-005/unit_property_evidence.json` |
| Differential/metamorphic | packet-F ufunc/gufunc oracle corpus (landed for current scope) | enforce strict parity for dispatch/type-resolution and failure-class behavior | `artifacts/phase2c/FNP-P2C-005/differential_metamorphic_adversarial_evidence.json` |
| E2E/replay | packet-G/H ufunc dispatch workflow scenarios (landed for current optimization/replay scope) | verify strict/hardened replay traceability and policy-forensics linkage | `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_reliability.json` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C005-RES-01` | `fnp-ufunc` does not yet cover the full legacy dispatch/signature boundary (notably custom-loop semantics). | keep unsupported semantics fail-closed and block parity promotion until packet-D boundary work lands | `bd-23m.16.4` + packet-E baseline tests |
| `P2C005-RES-02` | packet-scoped signature/override/reduction property corpus is landed for current scope but still needs broader custom-loop and gufunc edge coverage. | expand packet-E suite from the current deterministic shrink/replay baseline without weakening reason-code checks | `artifacts/phase2c/FNP-P2C-005/unit_property_evidence.json` |
| `P2C005-RES-03` | differential coverage for dispatch/type-resolution/loop registry passes current fixtures and still needs full legacy loop-registry breadth. | expand packet-F oracle/adversarial fixtures and keep differential gate coverage mandatory | `artifacts/phase2c/FNP-P2C-005/differential_metamorphic_adversarial_evidence.json` |
| `P2C005-RES-04` | packet-scoped replay forensics are encoded for the current ufunc journey but need more override/gufunc path breadth. | extend packet-G scenario logs while preserving required structured fields and reason-code linkage | `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_reliability.json` |
| `P2C005-RES-05` | hardened budget/calibration thresholds are now profiled for packet-H hotspot behavior but not tuned for full corpus scale. | use packet-H profile/isomorphism evidence as calibration baseline and keep conservative fallback trigger active for newly added corpus classes | `artifacts/phase2c/FNP-P2C-005/final_evidence_pack.json` |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| signature parse complexity | `<= 2_048` signature tokens/core-dim nodes per request | `fail_closed` with `ufunc_signature_parse_failed` |
| override evaluations | `<= 16` override checks per request | `fail_closed` with audited override budget reason code |
| dispatch candidate evaluations | `<= 512` candidate loop/resolver checks per dispatch | `fail_closed` with `ufunc_dispatch_resolution_failed` |
| reduction validation retries | `<= 64` axis/where validation retries per request | `fail_closed` with `ufunc_reduction_contract_violation` |
| loop registration entries | `<= 256` registration descriptors per request | reject with `ufunc_loop_registry_invalid` |
| packet-local audit payload | `<= 64 MiB` structured event payload | truncate optional diagnostics while preserving mandatory fields |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| ambiguous signature keyword state | `{reject, full_validate}` | incompatible semantics admitted into dispatch pipeline |
| override precedence conflict | `{use_override, use_builtin, reject}` | precedence inversion causing legacy behavior drift |
| uncertain dispatch/type-resolution path | `{resolve, reject}` | non-deterministic loop selection or dtype drift |
| hostile reduction payload | `{accept_reduction, reject}` | silent shape/value corruption in reduction output |
| unknown metadata class | `{fail_closed}` | undefined policy semantics admitted into runtime path |

### Calibration and fallback trigger

- Trigger fallback when either condition is true:
  - strict vs hardened packet failure-class drift exceeds `0.1%`, or
  - unknown/uncategorized reason-code rate exceeds `0.01%`.
- Fallback action: force conservative deterministic behavior (`full_validate` or `fail_closed`) until recalibration artifacts are produced and validated.

### Packet-H calibration artifact

- Profile baseline/rebaseline: `artifacts/phase2c/FNP-P2C-005/optimization_profile_report.json`.
- Isomorphism evidence: `artifacts/phase2c/FNP-P2C-005/optimization_profile_isomorphism_evidence.json`.
- Workflow replay evidence: `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_e2e.jsonl`.
- Workflow forensics index: `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_artifact_index.json`.
- Workflow reliability report: `artifacts/phase2c/FNP-P2C-005/workflow_scenario_packet005_opt_reliability.json`.
- Current packet-H calibration signal:
  - p95 latency delta `-72.205%`,
  - p95 throughput delta `+259.779%`,
  - failure-class drift `0` across packet-H isomorphism checks,
  - workflow gate replay coverage ratio `1.0` (298/298 scenarios passed).

## Alien Recommendation Contract Mapping

- Graveyard mappings: `alien_cs_graveyard.md` §0.4, §0.19, §6.12.
- FrankenSuite mappings: `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12, §0.13, §0.19.
- EV gate: policy/optimization levers ship only if `EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction) >= 2.0`; otherwise explicit deferred parity debt remains.
- Isomorphism proof artifacts required for behavior-affecting changes:
  - ordering/tie-break note,
  - before/after golden parity checks,
  - reproducible benchmark delta artifact.
- Hotspot evidence requirement for non-doc follow-on work: attach baseline/profile artifacts before changing dispatch policy/optimization behavior (or document profiler-unavailable fallback).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/_core/tests/test_ufunc.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_umath.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_overrides.py`
- `legacy_numpy_code/numpy/numpy/_core/tests/test_multiarray.py`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-005/parity_report.raptorq.json`
- `artifacts/phase2c/FNP-P2C-005/parity_report.scrub_report.json`
- `artifacts/phase2c/FNP-P2C-005/parity_report.decode_proof.json`
- `artifacts/phase2c/FNP-P2C-005/packet_readiness_report.json`
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## residual_risk_monitoring

owner: `packet-005-maintainers`  
follow_up_gate: `bd-23m.11 readiness drill + packet-I residual risk review`

follow_up_actions:
- expand packet-005 workflow scenario breadth for override and gufunc edge families before readiness sign-off.
- recalibrate hardened dispatch/type-resolution budgets against the full adversarial ufunc corpus and document drift trends.

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-005/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-005` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-005`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
