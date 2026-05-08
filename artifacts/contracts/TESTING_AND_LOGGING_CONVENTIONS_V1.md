# Testing and Logging Conventions V1

Contract ID: `test-logging-contract-v1`  
Schema Version: `1`

This document is the human-readable companion to `artifacts/contracts/test_logging_contract_v1.json`.
It defines mandatory unit/property testing conventions, fixture naming policy, structured logging rules, and gate commands.

## 1. Mandatory Invariant Families

Every conformance expansion must preserve and extend these invariant families:

1. `shape_stride_legality`
2. `dtype_promotion_table`
3. `runtime_policy_fail_closed`
4. `ufunc_metamorphic_relations`
5. `ufunc_adversarial_error_surface`
6. `user_workflow_golden_journeys`

Current suite anchors:

- `shape_stride_legality` -> `run_shape_stride_suite`
- `dtype_promotion_table` -> `run_dtype_promotion_suite`
- `runtime_policy_fail_closed` -> `run_runtime_policy_suite`, `run_runtime_policy_adversarial_suite`
- `ufunc_metamorphic_relations` -> `run_ufunc_metamorphic_suite`
- `ufunc_adversarial_error_surface` -> `run_ufunc_adversarial_suite`
- `user_workflow_golden_journeys` -> `workflow_scenarios::run_user_workflow_scenario_suite`

## 2. Property Testing + Shrink Requirements

Property/adversarial fixtures must carry deterministic replay context and shrink metadata.

Mandatory rules:

1. Deterministic replay seed is required.
2. Shrinking order is `shape` minimization before `value` minimization.
3. `max_shrink_steps` is bounded (currently `32`).
4. Shrink-related failures must include a `reason_code`.

## 3. Structured Log Contract

Required structured log fields:

1. `fixture_id`
2. `seed`
3. `mode`
4. `env_fingerprint`
5. `artifact_refs`
6. `reason_code`

Replay semantics:

1. Property/adversarial runs are deterministic by `seed`.
2. Policy/adversarial fixtures require non-empty `reason_code`.
3. Policy/adversarial fixtures require non-empty `artifact_refs`.

## 4. Fixture Naming Convention

Fixture IDs must satisfy:

1. lowercase only
2. start with alphanumeric
3. characters limited to `a-z0-9_`

Examples:

- valid: `sum_axis_out_of_bounds`
- invalid: `_bad_prefix`
- invalid: `BadFixture`
- invalid: `bad-hyphen`

## 5. Canonical Test Helper APIs

The contract currently requires these helper API anchors:

1. `fnp_conformance::run_all_core_suites`
2. `fnp_conformance::test_contracts::run_test_contract_suite`
3. `fnp_conformance::set_runtime_policy_log_path`
4. `fnp_conformance::workflow_scenarios::run_user_workflow_scenario_suite`

## 6. Enforcement Gates

Contract enforcement commands:

1. `cargo test -p fnp-conformance test_contract_suite_is_green -- --nocapture`
2. `cargo run -p fnp-conformance --bin run_test_contract_gate`
3. `cargo run -p fnp-conformance --bin run_workflow_scenario_gate`
4. `scripts/e2e/run_test_contract_gate.sh`
5. `scripts/e2e/run_workflow_scenario_gate.sh`

Gate behavior:

1. Missing contract fields or invalid schema fails the suite.
2. Missing seed/reason_code/artifact_refs in required fixture classes fails the suite.
3. Missing required gate script fails the suite.

## 9. Failure Forensics Artifact Index

`run_workflow_scenario_gate` emits a machine-readable artifact index JSON (`schema_version = 1`) to support one-hop triage from gate failure -> evidence -> replay:

1. `triage.failure_class`: deterministic class (`deterministic_regression`, `flake_budget`, `coverage_floor`, `artifact_log_validation`, or `scenario_assertion`)
2. `triage.first_bad_evidence`: first failing evidence string (scenario/step/fixture or first gate failure)
3. `triage.suggested_next_action`: deterministic next diagnostic command
4. `failure_envelopes[*]`: standardized failure envelope with `reason_code`, `fixture_lineage`, `replay_command`, and `evidence_refs`
5. `artifacts[*]`: deterministic references for workflow log, fixture corpus, runner script, and gate report

Default output path:

- `artifacts/logs/workflow_scenario_artifact_index_<ts>.json`

## 8. Workflow Scenario Corpus Requirements

The workflow corpus fixture (`crates/fnp-conformance/fixtures/workflow_scenario_corpus.json`) must:

1. Include at least one scenario for each category: `high_frequency`, `high_risk`, `adversarial`.
2. Encode strict/hardened expected statuses (`pass` or `fail_closed`) per scenario.
3. Provide deterministic replay metadata (`id`, `seed`, `env_fingerprint`, `artifact_refs`, `reason_code`).
4. Link each scenario to differential fixture IDs and executable e2e script paths.
5. Include non-empty prioritized gap entries with explicit `bead_id` and `owner`.

## 7. Method-Stack Mapping (Alien/Optimization/Durability/Compatibility)

Mapped graveyard sections:

1. `alien_cs_graveyard.md` §6.12 (`Property-Based Testing with Shrinking`)
2. `alien_cs_graveyard.md` §0.19 (`Evidence Ledger Schema`)
3. `alien_cs_graveyard.md` §0.3 (`Isomorphism Proof Block`)
4. `alien_cs_graveyard.md` §0.4 (`Decision-Theoretic Runtime Contracts`)
5. `alien_cs_graveyard.md` §1.1 (`RaptorQ Fountain Codes`)

Mapped FrankenSuite summary sections:

1. `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.12 (`Evidence Ledger Schema + CI Gates`)
2. `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.13 (`Archetype Playbooks for FrankenSuite`)
3. `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md` §0.19 (`Policy-as-Data`)

RaptorQ applicability note:

- Long-lived bundles (conformance fixtures, benchmark baselines, reproducibility ledgers) require sidecar/scrub/decode-proof artifacts.
- Ephemeral local gate logs are out of RaptorQ scope unless explicitly promoted to durable artifacts.

## 10. Worked Failure Investigations

Unit/property gate investigation:

```bash
FNP_TEST_CONTRACT_LOG_PATH=artifacts/logs/test_contract_e2e_manual.jsonl # evidence-validator-ignore(owner=conformance, reason=manual-investigation-output)
FNP_TEST_CONTRACT_REPORT_PATH=artifacts/logs/test_contract_gate_report.json # evidence-validator-ignore(owner=conformance, reason=manual-investigation-output)
rch exec -- cargo run -p fnp-conformance --bin run_test_contract_gate -- \
  --log-path "$FNP_TEST_CONTRACT_LOG_PATH" \
  | tee "$FNP_TEST_CONTRACT_REPORT_PATH"
```

If this fails, inspect missing fields quickly:

```bash
jq -r '.suites[] | select(.suite=="runtime_policy_log_contract") | .failures[]' \
  "$FNP_TEST_CONTRACT_REPORT_PATH"
```

Differential gate investigation:

```bash
rch exec -- cargo run -p fnp-conformance --bin run_ufunc_differential
jq -r '.failures[:5][] | [.id, .reason] | @tsv' \
  crates/fnp-conformance/fixtures/oracle_outputs/ufunc_differential_report.json
```

E2E workflow gate investigation with artifact index:

```bash
FNP_WORKFLOW_E2E_LOG=artifacts/logs/workflow_scenario_e2e_manual.jsonl # evidence-validator-ignore(owner=conformance, reason=manual-investigation-output)
FNP_WORKFLOW_RELIABILITY=artifacts/logs/workflow_scenario_reliability_manual.json # evidence-validator-ignore(owner=conformance, reason=manual-investigation-output)
FNP_WORKFLOW_ARTIFACT_INDEX=artifacts/logs/workflow_scenario_artifact_index_manual.json # evidence-validator-ignore(owner=conformance, reason=manual-investigation-output)
scripts/e2e/run_workflow_scenario_gate.sh \
  "$FNP_WORKFLOW_E2E_LOG" \
  "$FNP_WORKFLOW_RELIABILITY" \
  "$FNP_WORKFLOW_ARTIFACT_INDEX"
jq '.triage' "$FNP_WORKFLOW_ARTIFACT_INDEX"
```
