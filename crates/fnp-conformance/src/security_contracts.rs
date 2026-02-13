#![forbid(unsafe_code)]

use crate::{HarnessConfig, SuiteReport};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

const REQUIRED_THREAT_CLASSES: &[&str] = &[
    "malformed_shape",
    "unsafe_cast_path",
    "malicious_stride_alias",
    "malformed_npy_npz",
    "unknown_metadata_version",
    "adversarial_fixture",
    "corrupt_durable_artifact",
    "policy_override_abuse",
];

const REQUIRED_LOG_FIELDS: &[&str] = &[
    "fixture_id",
    "seed",
    "mode",
    "env_fingerprint",
    "artifact_refs",
    "reason_code",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HardenedAllowlist {
    schema_version: u8,
    allowlist_version: String,
    policy: AllowlistPolicy,
    allowed_deviation_classes: Vec<AllowedDeviationClass>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowlistPolicy {
    unknown_class_behavior: String,
    require_deterministic_audit_log: bool,
    require_api_contract_preservation: bool,
    require_bounded_recovery: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowedDeviationClass {
    class: String,
    description: String,
    applies_to_packets: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SecurityControlMap {
    schema_version: u8,
    control_map_version: String,
    threat_controls: Vec<ThreatControl>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThreatControl {
    threat_class: String,
    executable_checks: Vec<String>,
    fixture_hooks: Vec<String>,
    expected_log_fields: Vec<String>,
    compatibility_drift_gate: String,
    override_audit_requirement: String,
}

pub fn run_security_contract_suite(config: &HarnessConfig) -> Result<SuiteReport, String> {
    let matrix_path = config
        .contract_root
        .join("SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md");
    let allowlist_path = config.contract_root.join("hardened_mode_allowlist_v1.yaml");
    let control_map_path = config.contract_root.join("security_control_checks_v1.yaml");

    let matrix_raw = fs::read_to_string(&matrix_path)
        .map_err(|err| format!("failed reading {}: {err}", matrix_path.display()))?;
    let allowlist_raw = fs::read_to_string(&allowlist_path)
        .map_err(|err| format!("failed reading {}: {err}", allowlist_path.display()))?;
    let control_map_raw = fs::read_to_string(&control_map_path)
        .map_err(|err| format!("failed reading {}: {err}", control_map_path.display()))?;

    let allowlist: HardenedAllowlist = serde_yaml::from_str(&allowlist_raw)
        .map_err(|err| format!("invalid YAML {}: {err}", allowlist_path.display()))?;
    let control_map: SecurityControlMap = serde_yaml::from_str(&control_map_raw)
        .map_err(|err| format!("invalid YAML {}: {err}", control_map_path.display()))?;

    let mut report = SuiteReport {
        suite: "security_contracts",
        case_count: 0,
        pass_count: 0,
        failures: Vec::new(),
    };

    let matrix_classes = parse_threat_classes(&matrix_raw);
    for required in REQUIRED_THREAT_CLASSES {
        record_check(
            &mut report,
            matrix_classes.contains(*required),
            format!("threat matrix missing required class {required}"),
        );
    }

    record_check(
        &mut report,
        matrix_raw.contains("| Threat Class | Ingress Boundary | Strict Mode Policy | Hardened Mode Policy | Required Evidence Artifact |"),
        "threat matrix table header is missing or malformed".to_string(),
    );

    let mut controls_by_class = BTreeMap::new();
    for control in control_map.threat_controls {
        controls_by_class.insert(control.threat_class.clone(), control);
    }

    for required in REQUIRED_THREAT_CLASSES {
        if let Some(control) = controls_by_class.get(*required) {
            record_check(
                &mut report,
                !control.executable_checks.is_empty(),
                format!("{required}: executable_checks must not be empty"),
            );
            record_check(
                &mut report,
                !control.fixture_hooks.is_empty(),
                format!("{required}: fixture_hooks must not be empty"),
            );
            record_check(
                &mut report,
                !control.compatibility_drift_gate.trim().is_empty(),
                format!("{required}: compatibility_drift_gate must not be empty"),
            );
            record_check(
                &mut report,
                !control.override_audit_requirement.trim().is_empty(),
                format!("{required}: override_audit_requirement must not be empty"),
            );

            let expected_fields = control
                .expected_log_fields
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            for required_field in REQUIRED_LOG_FIELDS {
                record_check(
                    &mut report,
                    expected_fields.contains(required_field),
                    format!("{required}: expected_log_fields missing {required_field}"),
                );
            }
        } else {
            record_check(
                &mut report,
                false,
                format!("security control map missing threat_class {required}"),
            );
        }
    }

    record_check(
        &mut report,
        allowlist.schema_version == 1,
        "hardened allowlist schema_version must be 1".to_string(),
    );
    record_check(
        &mut report,
        allowlist.allowlist_version == "hardened-mode-allowlist-v1",
        "hardened allowlist version mismatch".to_string(),
    );
    record_check(
        &mut report,
        allowlist.policy.unknown_class_behavior == "fail_closed",
        "hardened allowlist must enforce fail_closed for unknown classes".to_string(),
    );
    record_check(
        &mut report,
        allowlist.policy.require_deterministic_audit_log,
        "hardened allowlist must require deterministic audit logs".to_string(),
    );
    record_check(
        &mut report,
        allowlist.policy.require_api_contract_preservation,
        "hardened allowlist must require API contract preservation".to_string(),
    );
    record_check(
        &mut report,
        allowlist.policy.require_bounded_recovery,
        "hardened allowlist must require bounded recovery".to_string(),
    );
    record_check(
        &mut report,
        !allowlist.allowed_deviation_classes.is_empty(),
        "hardened allowlist must declare at least one allowed deviation class".to_string(),
    );

    for allowed in &allowlist.allowed_deviation_classes {
        record_check(
            &mut report,
            !allowed.class.trim().is_empty(),
            "allowed_deviation_classes.class must not be empty".to_string(),
        );
        record_check(
            &mut report,
            !allowed.description.trim().is_empty(),
            format!(
                "allowed_deviation_classes[{}] description must not be empty",
                allowed.class
            ),
        );
        record_check(
            &mut report,
            !allowed.applies_to_packets.is_empty(),
            format!(
                "allowed_deviation_classes[{}] applies_to_packets must not be empty",
                allowed.class
            ),
        );
    }

    record_check(
        &mut report,
        control_map.schema_version == 1,
        "security control map schema_version must be 1".to_string(),
    );
    record_check(
        &mut report,
        control_map.control_map_version == "security-control-checks-v1",
        "security control map version mismatch".to_string(),
    );

    Ok(report)
}

fn parse_threat_classes(matrix_markdown: &str) -> BTreeSet<String> {
    let mut classes = BTreeSet::new();

    for line in matrix_markdown.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') || trimmed.contains("---") {
            continue;
        }

        let columns: Vec<_> = trimmed.split('|').map(str::trim).collect();
        if columns.len() < 6 {
            continue;
        }

        let class = columns[1];
        if class.is_empty() || class == "Threat Class" {
            continue;
        }
        classes.insert(class.to_string());
    }

    classes
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
    use super::{parse_threat_classes, run_security_contract_suite};
    use crate::HarnessConfig;

    #[test]
    fn matrix_parser_extracts_threat_classes() {
        let sample = "| Threat Class | Ingress Boundary | Strict Mode Policy | Hardened Mode Policy | Required Evidence Artifact |\n|---|---|---|---|---|\n| malformed_shape | API | reject | reject | shape audit |\n| policy_override_abuse | policy plane | audited override only | audited override only | override log |\n";
        let classes = parse_threat_classes(sample);
        assert!(classes.contains("malformed_shape"));
        assert!(classes.contains("policy_override_abuse"));
    }

    #[test]
    fn security_contract_suite_is_green() {
        let cfg = HarnessConfig::default_paths();
        let report = run_security_contract_suite(&cfg).expect("security contract suite should run");
        assert!(report.all_passed(), "failures={:?}", report.failures);
    }
}
