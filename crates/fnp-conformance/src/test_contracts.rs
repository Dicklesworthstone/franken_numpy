#![forbid(unsafe_code)]

use crate::{HarnessConfig, SuiteReport};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const REQUIRED_LOG_FIELDS: &[&str] = &[
    "fixture_id",
    "seed",
    "mode",
    "env_fingerprint",
    "artifact_refs",
    "reason_code",
];

const REQUIRED_INVARIANT_FAMILIES: &[&str] = &[
    "shape_stride_legality",
    "dtype_promotion_table",
    "runtime_policy_fail_closed",
    "io_parser_adversarial_surface",
    "linalg_differential_parity",
    "linalg_metamorphic_relations",
    "linalg_adversarial_error_surface",
    "ufunc_metamorphic_relations",
    "ufunc_adversarial_error_surface",
    "user_workflow_golden_journeys",
];

const REQUIRED_TEST_HELPER_APIS: &[&str] = &[
    "fnp_conformance::run_all_core_suites",
    "fnp_conformance::run_linalg_differential_suite",
    "fnp_conformance::run_linalg_metamorphic_suite",
    "fnp_conformance::run_linalg_adversarial_suite",
    "fnp_conformance::run_io_adversarial_suite",
    "fnp_conformance::run_crash_signature_regression_suite",
    "fnp_conformance::test_contracts::run_test_contract_suite",
    "fnp_conformance::set_runtime_policy_log_path",
    "fnp_conformance::workflow_scenarios::run_user_workflow_scenario_suite",
];

const REQUIRED_GATE_SCRIPTS: &[&str] = &[
    "scripts/e2e/run_test_contract_gate.sh",
    "scripts/e2e/run_workflow_scenario_gate.sh",
];
const REQUIRED_HUMAN_DOCS: &[&str] = &[
    "artifacts/contracts/TESTING_AND_LOGGING_CONVENTIONS_V1.md",
    "artifacts/contracts/USER_WORKFLOW_SCENARIO_CORPUS_V1.md",
];

const REQUIRED_FIXTURE_FILES: &[&str] = &[
    "runtime_policy_cases.json",
    "runtime_policy_adversarial_cases.json",
    "io_adversarial_cases.json",
    "linalg_differential_cases.json",
    "linalg_metamorphic_cases.json",
    "linalg_adversarial_cases.json",
    "ufunc_input_cases.json",
    "ufunc_metamorphic_cases.json",
    "ufunc_adversarial_cases.json",
    "workflow_scenario_corpus.json",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestLoggingContract {
    schema_version: u8,
    contract_version: String,
    required_structured_log_fields: Vec<String>,
    mandatory_invariant_families: Vec<String>,
    shrink_requirements: ShrinkRequirements,
    required_test_helper_apis: Vec<String>,
    fixture_id_policy: FixtureIdPolicy,
    replay_semantics: ReplaySemantics,
    required_gate_scripts: Vec<String>,
    required_human_docs: Vec<String>,
    required_fixture_collections: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShrinkRequirements {
    deterministic_seed_replay: bool,
    minimize_shape_before_values: bool,
    max_shrink_steps: u32,
    must_emit_shrink_reason_code: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureIdPolicy {
    allowed_chars: String,
    must_start_with_alnum: bool,
    must_be_lowercase: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplaySemantics {
    deterministic_seed_required_for_property_and_adversarial: bool,
    require_reason_code_for_policy_and_adversarial: bool,
    require_artifact_refs_for_policy_and_adversarial: bool,
}

pub fn run_test_contract_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let contract_path = config.contract_root.join("test_logging_contract_v1.json");
    let raw = fs::read_to_string(&contract_path)
        .map_err(|err| format!("failed reading {}: {err}", contract_path.display()))?;
    let contract: TestLoggingContract = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid JSON {}: {err}", contract_path.display()))?;

    let mut report = SuiteReport {
        suite: "test_contracts",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    record_check(
        &mut report,
        contract.schema_version == 1,
        "test logging contract schema_version must be 1".to_string(),
    );
    record_check(
        &mut report,
        contract.contract_version == "test-logging-contract-v1",
        "test logging contract version mismatch".to_string(),
    );

    let required_field_set = contract
        .required_structured_log_fields
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for required in REQUIRED_LOG_FIELDS {
        record_check(
            &mut report,
            required_field_set.contains(required),
            format!("required_structured_log_fields missing {required}"),
        );
    }

    let invariant_set = contract
        .mandatory_invariant_families
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for required in REQUIRED_INVARIANT_FAMILIES {
        record_check(
            &mut report,
            invariant_set.contains(required),
            format!("mandatory_invariant_families missing {required}"),
        );
    }

    record_check(
        &mut report,
        contract.shrink_requirements.deterministic_seed_replay,
        "shrink_requirements.deterministic_seed_replay must be true".to_string(),
    );
    record_check(
        &mut report,
        contract.shrink_requirements.minimize_shape_before_values,
        "shrink_requirements.minimize_shape_before_values must be true".to_string(),
    );
    record_check(
        &mut report,
        contract.shrink_requirements.max_shrink_steps > 0,
        "shrink_requirements.max_shrink_steps must be > 0".to_string(),
    );
    record_check(
        &mut report,
        contract.shrink_requirements.must_emit_shrink_reason_code,
        "shrink_requirements.must_emit_shrink_reason_code must be true".to_string(),
    );
    record_check(
        &mut report,
        !contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
            || contract.shrink_requirements.must_emit_shrink_reason_code,
        "reason_code replay semantics require shrink reason codes".to_string(),
    );

    let helper_set = contract
        .required_test_helper_apis
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for required in REQUIRED_TEST_HELPER_APIS {
        record_check(
            &mut report,
            helper_set.contains(required),
            format!("required_test_helper_apis missing {required}"),
        );
    }

    let gate_set = contract
        .required_gate_scripts
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let human_docs_set = contract
        .required_human_docs
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let repo_root = config
        .contract_root
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "unable to derive repository root from contract_root".to_string())?;
    for required in REQUIRED_GATE_SCRIPTS {
        record_check(
            &mut report,
            gate_set.contains(required),
            format!("required_gate_scripts missing {required}"),
        );
        let script_path = repo_root.join(required);
        record_check(
            &mut report,
            script_path.is_file(),
            format!(
                "required gate script does not exist: {}",
                script_path.display()
            ),
        );
    }
    for required in REQUIRED_HUMAN_DOCS {
        record_check(
            &mut report,
            human_docs_set.contains(required),
            format!("required_human_docs missing {required}"),
        );
        let doc_path = repo_root.join(required);
        record_check(
            &mut report,
            doc_path.is_file(),
            format!("required human doc does not exist: {}", doc_path.display()),
        );
    }

    for fixture_file in REQUIRED_FIXTURE_FILES {
        let present = contract
            .required_fixture_collections
            .iter()
            .any(|entry| entry == &format!("crates/fnp-conformance/fixtures/{fixture_file}"));
        record_check(
            &mut report,
            present,
            format!("required_fixture_collections missing {fixture_file}"),
        );
    }

    validate_runtime_policy_fixtures(config, &contract, &mut report)?;
    validate_runtime_policy_adversarial_fixtures(config, &contract, &mut report)?;
    validate_io_adversarial_fixtures(config, &contract, &mut report)?;
    validate_linalg_differential_fixtures(config, &contract, &mut report)?;
    validate_linalg_metamorphic_fixtures(config, &contract, &mut report)?;
    validate_linalg_adversarial_fixtures(config, &contract, &mut report)?;
    validate_ufunc_input_fixtures(config, &contract, &mut report)?;
    validate_ufunc_metamorphic_fixtures(config, &contract, &mut report)?;
    validate_ufunc_adversarial_fixtures(config, &contract, &mut report)?;
    validate_workflow_scenario_fixtures(config, &contract, &mut report)?;

    Ok(report)
}

fn validate_fixture_id(id: &str, policy: &FixtureIdPolicy) -> bool {
    if id.is_empty() {
        return false;
    }
    if policy.must_be_lowercase && id != id.to_lowercase() {
        return false;
    }
    if policy.must_start_with_alnum {
        let first = id.as_bytes()[0];
        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return false;
        }
    }

    // Current policy is ASCII alnum + underscore.
    if policy.allowed_chars != "a-z0-9_" {
        return false;
    }

    id.bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

fn validate_runtime_policy_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("runtime_policy_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "runtime_policy_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("runtime_policy_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("runtime_policy_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(report, false, "runtime_policy_cases missing id".to_string());
        }

        record_check(
            report,
            has_u64(obj, "seed"),
            "runtime_policy_cases missing seed".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "env_fingerprint"),
            "runtime_policy_cases missing env_fingerprint".to_string(),
        );
        if contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string(obj, "reason_code"),
                "runtime_policy_cases missing reason_code".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_artifact_refs_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string_array(obj, "artifact_refs"),
                "runtime_policy_cases missing artifact_refs".to_string(),
            );
        }
    }

    Ok(())
}

fn validate_runtime_policy_adversarial_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config
        .fixture_root
        .join("runtime_policy_adversarial_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "runtime_policy_adversarial_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("runtime_policy_adversarial_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("runtime_policy_adversarial_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "runtime_policy_adversarial_cases missing id".to_string(),
            );
        }

        record_check(
            report,
            has_u64(obj, "seed"),
            "runtime_policy_adversarial_cases missing seed".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "env_fingerprint"),
            "runtime_policy_adversarial_cases missing env_fingerprint".to_string(),
        );
        if contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string(obj, "reason_code"),
                "runtime_policy_adversarial_cases missing reason_code".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_artifact_refs_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string_array(obj, "artifact_refs"),
                "runtime_policy_adversarial_cases missing artifact_refs".to_string(),
            );
        }
    }

    Ok(())
}

fn validate_ufunc_input_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("ufunc_input_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "ufunc_input_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("ufunc_input_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("ufunc_input_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(report, false, "ufunc_input_cases missing id".to_string());
        }
    }

    Ok(())
}

fn validate_io_adversarial_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("io_adversarial_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "io_adversarial_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("io_adversarial_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("io_adversarial_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(report, false, "io_adversarial_cases missing id".to_string());
        }

        if contract
            .replay_semantics
            .deterministic_seed_required_for_property_and_adversarial
        {
            record_check(
                report,
                has_u64(obj, "seed"),
                "io_adversarial_cases missing deterministic seed".to_string(),
            );
        }

        if contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string(obj, "reason_code"),
                "io_adversarial_cases missing reason_code".to_string(),
            );
        }

        if contract
            .replay_semantics
            .require_artifact_refs_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string_array(obj, "artifact_refs"),
                "io_adversarial_cases missing artifact_refs".to_string(),
            );
        }

        record_check(
            report,
            has_non_empty_string(obj, "operation"),
            "io_adversarial_cases missing operation".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "expected_error_contains"),
            "io_adversarial_cases missing expected_error_contains".to_string(),
        );
        let severity = required_string(obj, "severity");
        if let Some(severity) = severity {
            record_check(
                report,
                matches!(severity, "low" | "medium" | "high" | "critical"),
                format!("io_adversarial_cases invalid severity {severity}"),
            );
        } else {
            record_check(
                report,
                false,
                "io_adversarial_cases missing severity".to_string(),
            );
        }
    }

    Ok(())
}

fn validate_linalg_differential_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("linalg_differential_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "linalg_differential_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("linalg_differential_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("linalg_differential_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "linalg_differential_cases missing id".to_string(),
            );
        }

        if contract
            .replay_semantics
            .deterministic_seed_required_for_property_and_adversarial
        {
            record_check(
                report,
                has_u64(obj, "seed"),
                "linalg_differential_cases missing deterministic seed".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string(obj, "reason_code"),
                "linalg_differential_cases missing reason_code".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_artifact_refs_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string_array(obj, "artifact_refs"),
                "linalg_differential_cases missing artifact_refs".to_string(),
            );
        }
        record_check(
            report,
            has_non_empty_string(obj, "env_fingerprint"),
            "linalg_differential_cases missing env_fingerprint".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "operation"),
            "linalg_differential_cases missing operation".to_string(),
        );
    }

    Ok(())
}

fn validate_linalg_metamorphic_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("linalg_metamorphic_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "linalg_metamorphic_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("linalg_metamorphic_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("linalg_metamorphic_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "linalg_metamorphic_cases missing id".to_string(),
            );
        }

        if contract
            .replay_semantics
            .deterministic_seed_required_for_property_and_adversarial
        {
            record_check(
                report,
                has_u64(obj, "seed"),
                "linalg_metamorphic_cases missing deterministic seed".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string(obj, "reason_code"),
                "linalg_metamorphic_cases missing reason_code".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_artifact_refs_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string_array(obj, "artifact_refs"),
                "linalg_metamorphic_cases missing artifact_refs".to_string(),
            );
        }
        record_check(
            report,
            has_non_empty_string(obj, "env_fingerprint"),
            "linalg_metamorphic_cases missing env_fingerprint".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "relation"),
            "linalg_metamorphic_cases missing relation".to_string(),
        );
    }

    Ok(())
}

fn validate_linalg_adversarial_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("linalg_adversarial_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "linalg_adversarial_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("linalg_adversarial_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("linalg_adversarial_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "linalg_adversarial_cases missing id".to_string(),
            );
        }

        if contract
            .replay_semantics
            .deterministic_seed_required_for_property_and_adversarial
        {
            record_check(
                report,
                has_u64(obj, "seed"),
                "linalg_adversarial_cases missing deterministic seed".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_reason_code_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string(obj, "reason_code"),
                "linalg_adversarial_cases missing reason_code".to_string(),
            );
            record_check(
                report,
                has_non_empty_string(obj, "expected_reason_code"),
                "linalg_adversarial_cases missing expected_reason_code".to_string(),
            );
        }
        if contract
            .replay_semantics
            .require_artifact_refs_for_policy_and_adversarial
        {
            record_check(
                report,
                has_non_empty_string_array(obj, "artifact_refs"),
                "linalg_adversarial_cases missing artifact_refs".to_string(),
            );
        }
        record_check(
            report,
            has_non_empty_string(obj, "env_fingerprint"),
            "linalg_adversarial_cases missing env_fingerprint".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "operation"),
            "linalg_adversarial_cases missing operation".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "expected_error_contains"),
            "linalg_adversarial_cases missing expected_error_contains".to_string(),
        );

        let severity = required_string(obj, "severity");
        if let Some(severity) = severity {
            record_check(
                report,
                matches!(severity, "low" | "medium" | "high" | "critical"),
                format!("linalg_adversarial_cases invalid severity {severity}"),
            );
        } else {
            record_check(
                report,
                false,
                "linalg_adversarial_cases missing severity".to_string(),
            );
        }
    }

    Ok(())
}

fn validate_ufunc_metamorphic_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("ufunc_metamorphic_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "ufunc_metamorphic_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("ufunc_metamorphic_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("ufunc_metamorphic_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "ufunc_metamorphic_cases missing id".to_string(),
            );
        }

        if contract
            .replay_semantics
            .deterministic_seed_required_for_property_and_adversarial
        {
            record_check(
                report,
                has_u64(obj, "seed"),
                "ufunc_metamorphic_cases missing deterministic seed".to_string(),
            );
        }
    }

    Ok(())
}

fn validate_ufunc_adversarial_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("ufunc_adversarial_cases.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "ufunc_adversarial_cases must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("ufunc_adversarial_cases invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("ufunc_adversarial_cases duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "ufunc_adversarial_cases missing id".to_string(),
            );
        }

        if contract
            .replay_semantics
            .deterministic_seed_required_for_property_and_adversarial
        {
            record_check(
                report,
                has_u64(obj, "seed"),
                "ufunc_adversarial_cases missing deterministic seed".to_string(),
            );
        }

        record_check(
            report,
            has_non_empty_string(obj, "expected_error_contains"),
            "ufunc_adversarial_cases missing expected_error_contains".to_string(),
        );
    }

    Ok(())
}

fn validate_workflow_scenario_fixtures(
    config: &HarnessConfig,
    contract: &TestLoggingContract,
    report: &mut SuiteReport,
) -> Result<(), String> {
    let path = config.fixture_root.join("workflow_scenario_corpus.json");
    let values = load_fixture_array(&path)?;
    let mut ids = BTreeSet::new();
    let mut categories = BTreeSet::new();

    let ufunc_ids = collect_fixture_ids(
        &config.fixture_root.join("ufunc_input_cases.json"),
        "ufunc_input_cases",
    )?;
    let runtime_policy_ids = collect_fixture_ids(
        &config.fixture_root.join("runtime_policy_cases.json"),
        "runtime_policy_cases",
    )?;
    let runtime_policy_wire_ids = collect_fixture_ids(
        &config
            .fixture_root
            .join("runtime_policy_adversarial_cases.json"),
        "runtime_policy_adversarial_cases",
    )?;

    let repo_root = config
        .fixture_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| "unable to derive repository root from fixture_root".to_string())?;

    for value in values {
        let Some(obj) = value.as_object() else {
            record_check(
                report,
                false,
                "workflow_scenario_corpus must be an array of objects".to_string(),
            );
            continue;
        };

        let id = required_string(obj, "id");
        if let Some(id) = id {
            record_check(
                report,
                validate_fixture_id(id, &contract.fixture_id_policy),
                format!("workflow_scenario_corpus invalid fixture id {id}"),
            );
            record_check(
                report,
                ids.insert(id.to_string()),
                format!("workflow_scenario_corpus duplicate fixture id {id}"),
            );
        } else {
            record_check(
                report,
                false,
                "workflow_scenario_corpus missing id".to_string(),
            );
        }

        record_check(
            report,
            has_u64(obj, "seed"),
            "workflow_scenario_corpus missing seed".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "description"),
            "workflow_scenario_corpus missing description".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "env_fingerprint"),
            "workflow_scenario_corpus missing env_fingerprint".to_string(),
        );
        record_check(
            report,
            has_non_empty_string(obj, "reason_code"),
            "workflow_scenario_corpus missing reason_code".to_string(),
        );
        record_check(
            report,
            has_non_empty_string_array(obj, "artifact_refs"),
            "workflow_scenario_corpus missing artifact_refs".to_string(),
        );

        if let Some(category) = required_string(obj, "category") {
            record_check(
                report,
                matches!(category, "high_frequency" | "high_risk" | "adversarial"),
                format!("workflow_scenario_corpus invalid category {category}"),
            );
            categories.insert(category.to_string());
        } else {
            record_check(
                report,
                false,
                "workflow_scenario_corpus missing category".to_string(),
            );
        }

        for mode_key in ["strict", "hardened"] {
            let Some(mode_obj) = obj.get(mode_key).and_then(Value::as_object) else {
                record_check(
                    report,
                    false,
                    format!("workflow_scenario_corpus missing {mode_key} object"),
                );
                continue;
            };

            let status = required_string(mode_obj, "expected_status");
            if let Some(status) = status {
                record_check(
                    report,
                    matches!(status, "pass" | "fail_closed"),
                    format!("workflow_scenario_corpus invalid {mode_key}.expected_status {status}"),
                );
            } else {
                record_check(
                    report,
                    false,
                    format!("workflow_scenario_corpus missing {mode_key}.expected_status"),
                );
            }
        }

        let Some(steps) = obj.get("steps").and_then(Value::as_array) else {
            record_check(
                report,
                false,
                "workflow_scenario_corpus missing steps array".to_string(),
            );
            continue;
        };
        record_check(
            report,
            !steps.is_empty(),
            "workflow_scenario_corpus steps array must not be empty".to_string(),
        );

        for step in steps {
            let Some(step_obj) = step.as_object() else {
                record_check(
                    report,
                    false,
                    "workflow_scenario_corpus step must be an object".to_string(),
                );
                continue;
            };

            record_check(
                report,
                has_non_empty_string(step_obj, "id"),
                "workflow_scenario_corpus step missing id".to_string(),
            );

            let kind = required_string(step_obj, "kind");
            let case_id = required_string(step_obj, "case_id");

            match (kind, case_id) {
                (Some("ufunc_input"), Some(case_id)) => {
                    record_check(
                        report,
                        ufunc_ids.contains(case_id),
                        format!("workflow step references unknown ufunc_input case_id {case_id}"),
                    );

                    if let Some(value) = step_obj.get("expect_error_contains") {
                        let valid = value.as_str().is_some_and(|entry| !entry.trim().is_empty())
                            || value.is_null();
                        record_check(
                            report,
                            valid,
                            "workflow step expect_error_contains must be non-empty string or null"
                                .to_string(),
                        );
                    }
                }
                (Some("runtime_policy"), Some(case_id)) => {
                    record_check(
                        report,
                        runtime_policy_ids.contains(case_id),
                        format!(
                            "workflow step references unknown runtime_policy case_id {case_id}"
                        ),
                    );
                    for key in ["expected_action_strict", "expected_action_hardened"] {
                        let action = required_string(step_obj, key);
                        if let Some(action) = action {
                            record_check(
                                report,
                                matches!(action, "allow" | "full_validate" | "fail_closed"),
                                format!("workflow step has invalid {key} value {action}"),
                            );
                        } else {
                            record_check(
                                report,
                                false,
                                format!("workflow step missing required field {key}"),
                            );
                        }
                    }
                }
                (Some("runtime_policy_wire"), Some(case_id)) => {
                    record_check(
                        report,
                        runtime_policy_wire_ids.contains(case_id),
                        format!(
                            "workflow step references unknown runtime_policy_adversarial case_id {case_id}"
                        ),
                    );
                    for key in ["expected_action_strict", "expected_action_hardened"] {
                        let action = required_string(step_obj, key);
                        if let Some(action) = action {
                            record_check(
                                report,
                                matches!(action, "allow" | "full_validate" | "fail_closed"),
                                format!("workflow step has invalid {key} value {action}"),
                            );
                        } else {
                            record_check(
                                report,
                                false,
                                format!("workflow step missing required field {key}"),
                            );
                        }
                    }
                }
                (Some(other), _) => {
                    record_check(
                        report,
                        false,
                        format!("workflow step has unsupported kind {other}"),
                    );
                }
                _ => {
                    record_check(
                        report,
                        false,
                        "workflow step missing kind or case_id".to_string(),
                    );
                }
            }
        }

        let Some(links_obj) = obj.get("links").and_then(Value::as_object) else {
            record_check(
                report,
                false,
                "workflow_scenario_corpus missing links object".to_string(),
            );
            continue;
        };

        let differential_ids = links_obj
            .get("differential_fixture_ids")
            .and_then(Value::as_array);
        match differential_ids {
            Some(ids) if !ids.is_empty() => {
                for entry in ids {
                    let Some(entry) = entry.as_str().filter(|raw| !raw.trim().is_empty()) else {
                        record_check(
                            report,
                            false,
                            "workflow links.differential_fixture_ids entries must be non-empty strings"
                                .to_string(),
                        );
                        continue;
                    };
                    record_check(
                        report,
                        ufunc_ids.contains(entry),
                        format!(
                            "workflow links.differential_fixture_ids references unknown case id {entry}"
                        ),
                    );
                }
            }
            _ => record_check(
                report,
                false,
                "workflow links.differential_fixture_ids must be a non-empty array".to_string(),
            ),
        }

        let e2e_paths = links_obj.get("e2e_script_paths").and_then(Value::as_array);
        match e2e_paths {
            Some(paths) if !paths.is_empty() => {
                for script in paths {
                    let Some(script) = script.as_str().filter(|raw| !raw.trim().is_empty()) else {
                        record_check(
                            report,
                            false,
                            "workflow links.e2e_script_paths entries must be non-empty strings"
                                .to_string(),
                        );
                        continue;
                    };
                    let full = repo_root.join(script);
                    record_check(
                        report,
                        full.is_file(),
                        format!(
                            "workflow links.e2e_script_paths missing file {}",
                            full.display()
                        ),
                    );
                }
            }
            _ => record_check(
                report,
                false,
                "workflow links.e2e_script_paths must be a non-empty array".to_string(),
            ),
        }

        let Some(gaps) = obj.get("gaps").and_then(Value::as_array) else {
            record_check(
                report,
                false,
                "workflow_scenario_corpus missing gaps array".to_string(),
            );
            continue;
        };
        record_check(
            report,
            !gaps.is_empty(),
            "workflow_scenario_corpus gaps array must not be empty".to_string(),
        );

        for gap in gaps {
            let Some(gap_obj) = gap.as_object() else {
                record_check(
                    report,
                    false,
                    "workflow gap entry must be an object".to_string(),
                );
                continue;
            };
            for key in ["bead_id", "owner", "priority", "description"] {
                record_check(
                    report,
                    has_non_empty_string(gap_obj, key),
                    format!("workflow gap missing {}", key),
                );
            }
        }
    }

    for required in ["high_frequency", "high_risk", "adversarial"] {
        record_check(
            report,
            categories.contains(required),
            format!("workflow_scenario_corpus missing category {required}"),
        );
    }

    Ok(())
}

fn collect_fixture_ids(path: &Path, label: &str) -> Result<BTreeSet<String>, String> {
    let values = load_fixture_array(path)?;
    let mut ids = BTreeSet::new();
    for value in values {
        let Some(obj) = value.as_object() else {
            return Err(format!("{label} must be an array of objects"));
        };
        let Some(id) = required_string(obj, "id") else {
            return Err(format!("{label} missing id"));
        };
        if !ids.insert(id.to_string()) {
            return Err(format!("{label} duplicate id {id}"));
        }
    }
    Ok(ids)
}

fn load_fixture_array(path: &Path) -> Result<Vec<Value>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("invalid json {}: {err}", path.display()))
}

fn required_string<'a>(obj: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    obj.get(key)?
        .as_str()
        .filter(|value| !value.trim().is_empty())
}

fn has_u64(obj: &serde_json::Map<String, Value>, key: &str) -> bool {
    obj.get(key).is_some_and(serde_json::Value::is_u64)
}

fn has_non_empty_string(obj: &serde_json::Map<String, Value>, key: &str) -> bool {
    obj.get(key)
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

fn has_non_empty_string_array(obj: &serde_json::Map<String, Value>, key: &str) -> bool {
    let Some(values) = obj.get(key).and_then(serde_json::Value::as_array) else {
        return false;
    };
    !values.is_empty()
        && values
            .iter()
            .all(|item| item.as_str().is_some_and(|value| !value.trim().is_empty()))
}

fn record_check(report: &mut SuiteReport, passed: bool, failure: String) {
    report.case_count += 1;
    if passed {
        report.pass_count += 1;
    } else {
        report.failures.push(failure);
    }
}

#[cfg(test)]
mod tests {
    use super::{run_test_contract_suite, validate_fixture_id};
    use crate::HarnessConfig;

    #[test]
    fn fixture_id_policy_enforced() {
        let policy = super::FixtureIdPolicy {
            allowed_chars: "a-z0-9_".to_string(),
            must_start_with_alnum: true,
            must_be_lowercase: true,
        };
        assert!(validate_fixture_id("good_fixture_1", &policy));
        assert!(!validate_fixture_id("BadFixture", &policy));
        assert!(!validate_fixture_id("_bad_prefix", &policy));
        assert!(!validate_fixture_id("bad-hyphen", &policy));
    }

    #[test]
    fn test_contract_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let report = run_test_contract_suite(&cfg).expect("test contract suite should run");
        assert!(report.all_passed(), "failures={:?}", report.failures);
    }
}
