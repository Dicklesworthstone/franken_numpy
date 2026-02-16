# FNP-P2C-008 Risk Note

packet_id: `FNP-P2C-008`  
subsystem: `linalg bridge first wave`

## compatibility_risks

| risk_id | threat_class | hostile input / abuse vector | strict mode handling | hardened mode handling | control anchors | verification hooks | reason_code family |
|---|---|---|---|---|---|---|---|
| `P2C008-RISK-01` | `shape_contract_abuse` | Invalid ndim/square-ness/axis metadata crafted to force undefined solver/decomposition routing. | Deterministic reject with stable class. | Same class with bounded diagnostics and audit linkage. | contract row `P2C008-R01`; `security_control_checks_v1.yaml` -> `malformed_shape`. | packet-E shape/property tests + packet-F shape differential corpus. | `linalg_shape_contract_violation` |
| `P2C008-RISK-02` | `singularity_evasion` | Nearly singular/singular systems intended to trigger unstable success/failure behavior in `solve`/`inv`. | Preserve legacy-compatible failure classes for singular/incompatible systems. | Same class; no permissive fallback for incompatible inputs. | contract row `P2C008-R02`; solver boundary controls. | packet-E solve/inv laws + packet-F solver oracle fixtures. | `linalg_solver_singularity` |
| `P2C008-RISK-03` | `factorization_mode_confusion` | Hostile mode payloads for `qr`/`svd`/`cholesky` designed to desynchronize output-shape family and failure classes. | Deterministic mode validation and stable class behavior. | Same behavior with bounded mode checks and deterministic reason codes. | contract rows `P2C008-R03`/`R04`/`R05`. | packet-E decomposition mode corpus + packet-F mode oracle lanes. | `linalg_cholesky_contract_violation`, `linalg_qr_mode_invalid`, `linalg_svd_nonconvergence` |
| `P2C008-RISK-04` | `spectral_nonconvergence_masking` | Inputs crafted to hide non-convergence or invalid hermitian branch handling in spectral paths. | Deterministic non-convergence and branch-class behavior. | Same classes with fail-closed unknown branch metadata handling. | contract row `P2C008-R06`; spectral policy checks. | packet-E spectral branch laws + packet-F spectral oracle lane. | `linalg_spectral_convergence_failed` |
| `P2C008-RISK-05` | `lstsq_tuple_drift` | Least-squares inputs targeting tuple field drift (`x`, `residuals`, `rank`, `s`) under tolerance edges. | Deterministic tuple class outcomes and failure classes. | Same classes with bounded tolerance validation. | contract row `P2C008-R07`; tolerance controls. | packet-E lstsq tuple invariants + packet-F lstsq oracle checks. | `linalg_lstsq_tuple_contract_violation` |
| `P2C008-RISK-06` | `tolerance_policy_poisoning` | Adversarial `rcond`/tolerance and order/axis payloads targeting norm/det/rank/pinv decision drift. | Stable output/error class behavior for supported tolerance metadata. | Same classes with bounded tolerance-search caps. | contract row `P2C008-R08`; runtime tolerance guards. | packet-E norm/det/rank/pinv properties + packet-F tolerance differential fixtures. | `linalg_norm_det_rank_policy_violation` |
| `P2C008-RISK-07` | `backend_bridge_tampering` | Invalid backend adapter/lapack-lite parameter states intended to bypass deterministic error taxonomy. | Deterministic backend error-class mapping and reject behavior. | Same mapping with fail-closed unsupported backend state handling. | contract row `P2C008-R09`; backend seam controls. | packet-E backend hook tests + packet-F backend differential lane. | `linalg_backend_bridge_invalid` |
| `P2C008-RISK-08` | `batched_shape_amplification` | Large batched matrix inputs designed to induce shape-route confusion or unstable class outcomes. | Deterministic batch-shape contract checks before execution. | Same outcomes with bounded batch validation budgets. | contract rows `P2C008-R01`/`R04`/`R05`/`R06`. | packet-E batched-shape properties + packet-F batched oracle lane + packet-G workflow replay. | `linalg_shape_contract_violation` |
| `P2C008-RISK-09` | `policy_override_abuse` | Attempts to admit incompatible linalg semantics through runtime override channels. | Unknown/incompatible semantics fail closed. | Audited-only override for explicitly allowlisted compatible cases; fail-closed otherwise. | runtime override gate + contract row `P2C008-R10`. | runtime policy adversarial suite + security gate. | `override_*` |
| `P2C008-RISK-10` | `unknown_metadata_version` | Unknown wire mode/class metadata entering linalg policy boundaries. | Fail-closed. | Fail-closed with deterministic reason-code emission. | `security_control_checks_v1.yaml` -> `unknown_metadata_version`; contract row `P2C008-R10`. | `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`. | `wire_unknown_*` |
| `P2C008-RISK-11` | `adversarial_fixture` | Poisoned fixture payloads intended to conceal linalg parity drift or replay gaps. | Reject malformed fixture bundles. | Reject + quarantine/audit references with no silent recovery. | `security_control_checks_v1.yaml` -> `adversarial_fixture`. | packet-F fixture schema validation + test-contract gate. | `fixture_contract_violation` |
| `P2C008-RISK-12` | `corrupt_durable_artifact` | Tampered packet durability evidence (sidecar/scrub/decode-proof mismatch) for linalg parity claims. | Fail validation gate on integrity mismatch. | Deterministic bounded recovery only with successful decode/hash proof. | `security_control_checks_v1.yaml` -> `corrupt_durable_artifact`. | `validate_phase2c_packet --packet-id FNP-P2C-008`. | `artifact_integrity_*` |

## Threat Envelope and Hardened Recovery

- Unknown or incompatible linalg semantics are fail-closed in strict and hardened modes.
- Hardened mode may add bounded validation and audit enrichment, but cannot alter API-visible success/failure classes for covered contract rows.
- Recovery behavior remains explicit and deterministic: `allow`, `full_validate`, or `fail_closed`.
- Policy/replay records must include `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, and `reason_code`.

## Adversarial Fixture Set Definition

| lane | fixture family | objective | owner bead |
|---|---|---|---|
| Fuzz/property | packet-E linalg invariant corpus (planned) | detect solver/decomposition/spectral/tolerance contract violations with shrinkable counterexamples | `bd-23m.19.5` |
| Differential/metamorphic | packet-F linalg oracle/adversarial corpus (implemented) | enforce strict parity for linalg output/failure-class behavior across golden and hostile cases | `bd-23m.19.6` |
| E2E/replay | packet-G linalg workflow scenarios (planned) | verify strict/hardened replay traceability and policy-forensics linkage | `bd-23m.19.7` |

## Residual Risks and Compensating Controls

| residual_id | residual risk | compensating controls | closure gate |
|---|---|---|---|
| `P2C008-RES-01` | `fnp-linalg` is still skeletal for the full packet boundary surface. | keep unsupported semantics fail-closed and block parity promotion until packet-D boundary work lands | `bd-23m.19.4` + packet-E baseline tests |
| `P2C008-RES-02` | packet-scoped unit/property linalg corpus is incomplete. | require packet-E suite with deterministic shrink/replay logging before parity claims | `bd-23m.19.5` |
| `P2C008-RES-03` | closed: differential/adversarial coverage for singular/non-convergence/tolerance-edge classes is now wired into packet-F suites and gates. | maintain packet-F fixture lanes and differential gate coverage (`linalg_differential`, `linalg_metamorphic`, `linalg_adversarial`). | `bd-23m.19.6` |
| `P2C008-RES-04` | packet-scoped e2e replay for linalg journeys is incomplete. | add packet-G scenario logs with required structured fields and reason-code linkage | `bd-23m.19.7` |
| `P2C008-RES-05` | hardened budget/calibration thresholds are defined but not tuned for full linalg corpus scale. | trigger conservative fallback on drift and recalibrate from packet-H evidence before packet-I sign-off | `bd-23m.19.8` + packet-I closure |

## Budgeted Mode and Decision-Theoretic Controls

### Explicit bounded caps (hardened policy path)

| control | cap | deterministic exhaustion behavior |
|---|---|---|
| solver/decomposition retry budget | `<= 32` retries per request | `fail_closed` with operation reason code family |
| tolerance-search depth | `<= 128` tolerance evaluations per request | `fail_closed` with `linalg_norm_det_rank_policy_violation` |
| backend bridge revalidation attempts | `<= 64` checks per request | reject with `linalg_backend_bridge_invalid` |
| batched shape validations | `<= 2_000_000` matrix-lane checks per request | abort with `linalg_shape_contract_violation` |
| policy override evaluations | `<= 16` override checks per request | fallback to conservative default (`fail_closed`) with audited reason code |
| packet-local audit payload | `<= 64 MiB` structured event payload | truncate optional diagnostics while preserving mandatory fields |

### Expected-loss model

| state | action set | primary loss if wrong |
|---|---|---|
| ambiguous solver/factorization legality | `{reject, full_validate}` | invalid input admitted causing unstable failure classes |
| singularity uncertainty | `{attempt_solve, reject}` | false-success numerical outputs where failure class should surface |
| spectral branch uncertainty | `{execute_branch, reject}` | incorrect eigenvalue/vector class or masked non-convergence |
| backend seam anomaly | `{bridge_call, reject}` | nondeterministic backend error behavior exposed to API |
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
- Hotspot evidence requirement for non-doc follow-on work: attach baseline/profile artifacts before changing linalg policy/optimization behavior (or document profiler-unavailable fallback).

## oracle_tests

- `legacy_numpy_code/numpy/numpy/linalg/tests/test_linalg.py`
- `legacy_numpy_code/numpy/numpy/linalg/tests/test_regression.py`
- `legacy_numpy_code/numpy/numpy/linalg/_linalg.py`
- `legacy_numpy_code/numpy/numpy/linalg/umath_linalg.cpp`

## raptorq_artifacts

- `artifacts/phase2c/FNP-P2C-008/parity_report.raptorq.json` (planned at packet-I)
- `artifacts/phase2c/FNP-P2C-008/parity_report.decode_proof.json` (planned at packet-I)
- `artifacts/raptorq/conformance_bundle_v1.sidecar.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.scrub_report.json` (program-level baseline reference)
- `artifacts/raptorq/conformance_bundle_v1.decode_proof.json` (program-level baseline reference)

## Rollback Handle

- Rollback command path: `git restore --source <last-green-commit> -- artifacts/phase2c/FNP-P2C-008/risk_note.md`
- Baseline comparator to beat/restore:
  - last green `rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-008` packet report,
  - plus last green security/test/workflow gate artifacts tied to packet `FNP-P2C-008`.
- If comparator is not met, restore risk-note baseline and re-run packet gates before reattempting policy changes.
