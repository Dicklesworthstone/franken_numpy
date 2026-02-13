#![forbid(unsafe_code)]

use crate::raptorq_artifacts::{DecodeProofArtifact, RaptorQSidecar};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const CONTRACT_SCHEMA_VERSION: &str = "phase2c-contract-v1";

const REQUIRED_FILES: &[&str] = &[
    "legacy_anchor_map.md",
    "contract_table.md",
    "fixture_manifest.json",
    "parity_gate.yaml",
    "risk_note.md",
    "parity_report.json",
    "parity_report.raptorq.json",
    "parity_report.decode_proof.json",
];

const LEGACY_ANCHOR_REQUIRED_TOKENS: &[&str] = &["packet_id", "legacy_paths", "legacy_symbols"];
const CONTRACT_TABLE_REQUIRED_TOKENS: &[&str] = &[
    "shape_stride_contract",
    "dtype_cast_contract",
    "error_contract",
    "memory_alias_contract",
    "strict_mode_policy",
    "hardened_mode_policy",
    "excluded_scope",
    "performance_sentinels",
];
const RISK_NOTE_REQUIRED_TOKENS: &[&str] =
    &["compatibility_risks", "oracle_tests", "raptorq_artifacts"];

const FIXTURE_MANIFEST_REQUIRED_PATHS: &[&[&str]] = &[
    &["schema_version"],
    &["packet_id"],
    &["oracle_tests"],
    &["fixtures"],
];
const PARITY_GATE_REQUIRED_PATHS: &[&[&str]] = &[
    &["schema_version"],
    &["packet_id"],
    &["strict_mode"],
    &["hardened_mode"],
    &["max_strict_drift"],
    &["max_hardened_divergence"],
];
const PARITY_REPORT_REQUIRED_PATHS: &[&[&str]] = &[
    &["schema_version"],
    &["packet_id"],
    &["strict_parity"],
    &["hardened_parity"],
    &["divergence_classes"],
    &["compatibility_drift_hash"],
];
const RAPTORQ_SIDECAR_REQUIRED_PATHS: &[&[&str]] = &[
    &["schema_version"],
    &["bundle_id"],
    &["source_hash"],
    &["source_size"],
    &["symbol_size"],
    &["symbols"],
];
const DECODE_PROOF_REQUIRED_PATHS: &[&[&str]] = &[
    &["schema_version"],
    &["bundle_id"],
    &["recovery_success"],
    &["expected_hash"],
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissingField {
    pub artifact: String,
    pub field_path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PacketReadinessReport {
    pub schema_version: u8,
    pub contract_schema_version: String,
    pub packet_id: String,
    pub packet_dir: String,
    pub status: String,
    pub missing_artifacts: Vec<String>,
    pub missing_fields: Vec<MissingField>,
    pub parse_errors: Vec<String>,
    pub checked_at_unix_ms: u128,
}

impl PacketReadinessReport {
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.status == "ready"
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureManifest {
    schema_version: u8,
    packet_id: String,
    oracle_tests: Vec<String>,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureEntry {
    id: String,
    input_path: String,
    oracle_case_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParityGate {
    schema_version: u8,
    packet_id: String,
    strict_mode: GateMode,
    hardened_mode: GateMode,
    max_strict_drift: f64,
    max_hardened_divergence: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateMode {
    pass_required: bool,
    min_pass_rate: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParityReport {
    schema_version: u8,
    packet_id: String,
    strict_parity: f64,
    hardened_parity: f64,
    divergence_classes: Vec<String>,
    compatibility_drift_hash: String,
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn add_missing_field(
    missing_fields: &mut Vec<MissingField>,
    artifact: &str,
    field_path: &str,
    reason: &str,
) {
    missing_fields.push(MissingField {
        artifact: artifact.to_string(),
        field_path: field_path.to_string(),
        reason: reason.to_string(),
    });
}

fn has_path(value: &Value, path: &[&str]) -> bool {
    let mut cursor = value;
    for segment in path {
        match cursor {
            Value::Object(map) => match map.get(*segment) {
                Some(next) => cursor = next,
                None => return false,
            },
            _ => return false,
        }
    }
    true
}

fn validate_markdown_tokens(
    path: &Path,
    artifact: &str,
    required_tokens: &[&str],
    missing_fields: &mut Vec<MissingField>,
    parse_errors: &mut Vec<String>,
) {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            parse_errors.push(format!("failed reading {}: {err}", path.display()));
            return;
        }
    };

    if raw.trim().is_empty() {
        add_missing_field(
            missing_fields,
            artifact,
            "<document_non_empty>",
            "document is empty",
        );
        return;
    }

    for token in required_tokens {
        if !raw.contains(token) {
            add_missing_field(
                missing_fields,
                artifact,
                token,
                "required token missing from document",
            );
        }
    }
}

fn validate_json_required_fields(
    path: &Path,
    artifact: &str,
    required_paths: &[&[&str]],
    missing_fields: &mut Vec<MissingField>,
    parse_errors: &mut Vec<String>,
) -> Option<Value> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            parse_errors.push(format!("failed reading {}: {err}", path.display()));
            return None;
        }
    };

    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(err) => {
            parse_errors.push(format!("invalid JSON {}: {err}", path.display()));
            return None;
        }
    };

    for required_path in required_paths {
        if !has_path(&value, required_path) {
            add_missing_field(
                missing_fields,
                artifact,
                &required_path.join("."),
                "mandatory field missing",
            );
        }
    }

    Some(value)
}

fn validate_yaml_required_fields(
    path: &Path,
    artifact: &str,
    required_paths: &[&[&str]],
    missing_fields: &mut Vec<MissingField>,
    parse_errors: &mut Vec<String>,
) -> Option<Value> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            parse_errors.push(format!("failed reading {}: {err}", path.display()));
            return None;
        }
    };

    let yaml_value: serde_yaml::Value = match serde_yaml::from_str(&raw) {
        Ok(value) => value,
        Err(err) => {
            parse_errors.push(format!("invalid YAML {}: {err}", path.display()));
            return None;
        }
    };

    let value: Value = match serde_json::to_value(yaml_value) {
        Ok(value) => value,
        Err(err) => {
            parse_errors.push(format!(
                "failed converting YAML to JSON value {}: {err}",
                path.display()
            ));
            return None;
        }
    };

    for required_path in required_paths {
        if !has_path(&value, required_path) {
            add_missing_field(
                missing_fields,
                artifact,
                &required_path.join("."),
                "mandatory field missing",
            );
        }
    }

    Some(value)
}

pub fn validate_phase2c_packet(packet_id: &str, packet_dir: &Path) -> PacketReadinessReport {
    let mut missing_artifacts = Vec::new();
    let mut missing_fields = Vec::new();
    let mut parse_errors = Vec::new();

    for required in REQUIRED_FILES {
        let path = packet_dir.join(required);
        if !path.is_file() {
            missing_artifacts.push((*required).to_string());
        }
    }

    let legacy_anchor_path = packet_dir.join("legacy_anchor_map.md");
    if legacy_anchor_path.is_file() {
        validate_markdown_tokens(
            &legacy_anchor_path,
            "legacy_anchor_map.md",
            LEGACY_ANCHOR_REQUIRED_TOKENS,
            &mut missing_fields,
            &mut parse_errors,
        );
    }

    let contract_table_path = packet_dir.join("contract_table.md");
    if contract_table_path.is_file() {
        validate_markdown_tokens(
            &contract_table_path,
            "contract_table.md",
            CONTRACT_TABLE_REQUIRED_TOKENS,
            &mut missing_fields,
            &mut parse_errors,
        );
    }

    let risk_note_path = packet_dir.join("risk_note.md");
    if risk_note_path.is_file() {
        validate_markdown_tokens(
            &risk_note_path,
            "risk_note.md",
            RISK_NOTE_REQUIRED_TOKENS,
            &mut missing_fields,
            &mut parse_errors,
        );
    }

    let fixture_manifest_path = packet_dir.join("fixture_manifest.json");
    if fixture_manifest_path.is_file()
        && let Some(value) = validate_json_required_fields(
            &fixture_manifest_path,
            "fixture_manifest.json",
            FIXTURE_MANIFEST_REQUIRED_PATHS,
            &mut missing_fields,
            &mut parse_errors,
        )
    {
        match serde_json::from_value::<FixtureManifest>(value) {
            Ok(manifest) => {
                if manifest.schema_version != 1 {
                    add_missing_field(
                        &mut missing_fields,
                        "fixture_manifest.json",
                        "schema_version",
                        "schema_version must be 1",
                    );
                }
                if manifest.packet_id != packet_id {
                    add_missing_field(
                        &mut missing_fields,
                        "fixture_manifest.json",
                        "packet_id",
                        "packet_id mismatch with selected packet",
                    );
                }
                if manifest.oracle_tests.is_empty() {
                    add_missing_field(
                        &mut missing_fields,
                        "fixture_manifest.json",
                        "oracle_tests",
                        "oracle_tests must not be empty",
                    );
                }
                if manifest.fixtures.is_empty() {
                    add_missing_field(
                        &mut missing_fields,
                        "fixture_manifest.json",
                        "fixtures",
                        "fixtures must not be empty",
                    );
                } else {
                    for (index, fixture) in manifest.fixtures.iter().enumerate() {
                        if fixture.id.trim().is_empty() {
                            add_missing_field(
                                &mut missing_fields,
                                "fixture_manifest.json",
                                &format!("fixtures[{index}].id"),
                                "fixture id must not be empty",
                            );
                        }
                        if fixture.input_path.trim().is_empty() {
                            add_missing_field(
                                &mut missing_fields,
                                "fixture_manifest.json",
                                &format!("fixtures[{index}].input_path"),
                                "input_path must not be empty",
                            );
                        }
                        if fixture.oracle_case_id.trim().is_empty() {
                            add_missing_field(
                                &mut missing_fields,
                                "fixture_manifest.json",
                                &format!("fixtures[{index}].oracle_case_id"),
                                "oracle_case_id must not be empty",
                            );
                        }
                    }
                }
            }
            Err(err) => parse_errors.push(format!(
                "fixture_manifest.json failed typed validation: {err}"
            )),
        }
    }

    let parity_gate_path = packet_dir.join("parity_gate.yaml");
    if parity_gate_path.is_file()
        && let Some(value) = validate_yaml_required_fields(
            &parity_gate_path,
            "parity_gate.yaml",
            PARITY_GATE_REQUIRED_PATHS,
            &mut missing_fields,
            &mut parse_errors,
        )
    {
        match serde_json::from_value::<ParityGate>(value) {
            Ok(gate) => {
                if gate.schema_version != 1 {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "schema_version",
                        "schema_version must be 1",
                    );
                }
                if gate.packet_id != packet_id {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "packet_id",
                        "packet_id mismatch with selected packet",
                    );
                }
                if gate.strict_mode.min_pass_rate <= 0.0 || gate.strict_mode.min_pass_rate > 1.0 {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "strict_mode.min_pass_rate",
                        "min_pass_rate must be in (0, 1]",
                    );
                }
                if gate.hardened_mode.min_pass_rate <= 0.0 || gate.hardened_mode.min_pass_rate > 1.0
                {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "hardened_mode.min_pass_rate",
                        "min_pass_rate must be in (0, 1]",
                    );
                }
                if !gate.strict_mode.pass_required {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "strict_mode.pass_required",
                        "strict mode must require pass",
                    );
                }
                if !gate.hardened_mode.pass_required {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "hardened_mode.pass_required",
                        "hardened mode must require pass",
                    );
                }
                if gate.max_strict_drift != 0.0 {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "max_strict_drift",
                        "strict drift budget must be 0.0",
                    );
                }
                if gate.max_hardened_divergence < 0.0 {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_gate.yaml",
                        "max_hardened_divergence",
                        "max_hardened_divergence must be >= 0.0",
                    );
                }
            }
            Err(err) => {
                parse_errors.push(format!("parity_gate.yaml failed typed validation: {err}"))
            }
        }
    }

    let parity_report_path = packet_dir.join("parity_report.json");
    if parity_report_path.is_file()
        && let Some(value) = validate_json_required_fields(
            &parity_report_path,
            "parity_report.json",
            PARITY_REPORT_REQUIRED_PATHS,
            &mut missing_fields,
            &mut parse_errors,
        )
    {
        match serde_json::from_value::<ParityReport>(value) {
            Ok(report) => {
                if report.schema_version != 1 {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_report.json",
                        "schema_version",
                        "schema_version must be 1",
                    );
                }
                if report.packet_id != packet_id {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_report.json",
                        "packet_id",
                        "packet_id mismatch with selected packet",
                    );
                }
                if !(0.0..=1.0).contains(&report.strict_parity) {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_report.json",
                        "strict_parity",
                        "strict_parity must be in [0, 1]",
                    );
                }
                if !(0.0..=1.0).contains(&report.hardened_parity) {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_report.json",
                        "hardened_parity",
                        "hardened_parity must be in [0, 1]",
                    );
                }
                if report.compatibility_drift_hash.trim().is_empty() {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_report.json",
                        "compatibility_drift_hash",
                        "compatibility_drift_hash must not be empty",
                    );
                }
                if report.divergence_classes.is_empty() {
                    add_missing_field(
                        &mut missing_fields,
                        "parity_report.json",
                        "divergence_classes",
                        "divergence_classes must be present (empty list permitted only if explicitly intentional)",
                    );
                }
            }
            Err(err) => {
                parse_errors.push(format!("parity_report.json failed typed validation: {err}"))
            }
        }
    }

    let sidecar_path = packet_dir.join("parity_report.raptorq.json");
    if sidecar_path.is_file()
        && let Some(value) = validate_json_required_fields(
            &sidecar_path,
            "parity_report.raptorq.json",
            RAPTORQ_SIDECAR_REQUIRED_PATHS,
            &mut missing_fields,
            &mut parse_errors,
        )
        && serde_json::from_value::<RaptorQSidecar>(value).is_err()
    {
        parse_errors.push("parity_report.raptorq.json failed typed sidecar validation".to_string());
    }

    let decode_proof_path = packet_dir.join("parity_report.decode_proof.json");
    if decode_proof_path.is_file()
        && let Some(value) = validate_json_required_fields(
            &decode_proof_path,
            "parity_report.decode_proof.json",
            DECODE_PROOF_REQUIRED_PATHS,
            &mut missing_fields,
            &mut parse_errors,
        )
        && serde_json::from_value::<DecodeProofArtifact>(value).is_err()
    {
        parse_errors.push(
            "parity_report.decode_proof.json failed typed decode-proof validation".to_string(),
        );
    }

    let status =
        if missing_artifacts.is_empty() && missing_fields.is_empty() && parse_errors.is_empty() {
            "ready"
        } else {
            "not_ready"
        };

    PacketReadinessReport {
        schema_version: 1,
        contract_schema_version: CONTRACT_SCHEMA_VERSION.to_string(),
        packet_id: packet_id.to_string(),
        packet_dir: packet_dir.display().to_string(),
        status: status.to_string(),
        missing_artifacts,
        missing_fields,
        parse_errors,
        checked_at_unix_ms: now_unix_ms(),
    }
}

pub fn write_packet_readiness_report(
    output_path: &Path,
    report: &PacketReadinessReport,
) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing readiness report: {err}"))?;
    fs::write(output_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))
}

#[cfg(test)]
mod tests {
    use super::{validate_phase2c_packet, write_packet_readiness_report};
    use std::fs;
    use std::path::Path;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("fnp_contract_schema_{name}_{ts}"));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, content).expect("write file");
    }

    fn write_valid_packet(packet_dir: &Path, packet_id: &str) {
        write(
            &packet_dir.join("legacy_anchor_map.md"),
            &format!(
                "# legacy_anchor_map\npacket_id: {packet_id}\nlegacy_paths: [\"numpy/_core/src/umath/ufunc_object.c\"]\nlegacy_symbols: [\"convert_ufunc_arguments\"]\n"
            ),
        );

        write(
            &packet_dir.join("contract_table.md"),
            "shape_stride_contract: deterministic\n\
             dtype_cast_contract: explicit\n\
             error_contract: fail_closed\n\
             memory_alias_contract: explicit\n\
             strict_mode_policy: exact\n\
             hardened_mode_policy: bounded\n\
             excluded_scope: out_of_v1\n\
             performance_sentinels: [broadcast_add]\n",
        );

        write(
            &packet_dir.join("risk_note.md"),
            "compatibility_risks: documented\noracle_tests: mapped\nraptorq_artifacts: required\n",
        );

        write(
            &packet_dir.join("fixture_manifest.json"),
            &format!(
                "{{\n  \"schema_version\": 1,\n  \"packet_id\": \"{packet_id}\",\n  \"oracle_tests\": [\"numpy/_core/tests/test_ufunc.py\"],\n  \"fixtures\": [{{\"id\": \"ufunc_add\", \"input_path\": \"crates/fnp-conformance/fixtures/ufunc_input_cases.json\", \"oracle_case_id\": \"add_broadcast_2x3_plus_3\"}}]\n}}\n"
            ),
        );

        write(
            &packet_dir.join("parity_gate.yaml"),
            &format!(
                "schema_version: 1\npacket_id: {packet_id}\nstrict_mode:\n  pass_required: true\n  min_pass_rate: 1.0\nhardened_mode:\n  pass_required: true\n  min_pass_rate: 0.99\nmax_strict_drift: 0.0\nmax_hardened_divergence: 0.01\n"
            ),
        );

        write(
            &packet_dir.join("parity_report.json"),
            &format!(
                "{{\n  \"schema_version\": 1,\n  \"packet_id\": \"{packet_id}\",\n  \"strict_parity\": 1.0,\n  \"hardened_parity\": 1.0,\n  \"divergence_classes\": [\"none\"],\n  \"compatibility_drift_hash\": \"sha256:deadbeef\"\n}}\n"
            ),
        );

        write(
            &packet_dir.join("parity_report.raptorq.json"),
            "{\n  \"schema_version\": 1,\n  \"bundle_id\": \"packet_sidecar\",\n  \"generated_at_unix_ms\": 0,\n  \"source_hash\": \"abc\",\n  \"source_size\": 1,\n  \"object_id_u128\": 1,\n  \"symbol_size\": 256,\n  \"max_block_size\": 256,\n  \"repair_overhead\": 1.25,\n  \"source_blocks\": 1,\n  \"source_symbols\": 1,\n  \"repair_symbols\": 1,\n  \"total_symbols\": 0,\n  \"symbols\": []\n}\n",
        );

        write(
            &packet_dir.join("parity_report.decode_proof.json"),
            "{\n  \"schema_version\": 1,\n  \"bundle_id\": \"packet_sidecar\",\n  \"generated_at_unix_ms\": 0,\n  \"dropped_symbol\": null,\n  \"recovery_symbols_used\": 0,\n  \"recovery_success\": true,\n  \"expected_hash\": \"abc\",\n  \"recovered_hash\": \"abc\",\n  \"error\": null\n}\n",
        );
    }

    #[test]
    fn ready_when_required_artifacts_and_fields_exist() {
        let packet_dir = temp_dir("ready");
        let packet_id = "FNP-P2C-001";
        write_valid_packet(&packet_dir, packet_id);

        let report = validate_phase2c_packet(packet_id, &packet_dir);
        assert_eq!(report.status, "ready", "report={report:?}");
        assert!(report.is_ready());
        assert!(report.missing_artifacts.is_empty());
        assert!(report.missing_fields.is_empty());
        assert!(report.parse_errors.is_empty());

        let out = packet_dir.join("packet_readiness_report.json");
        write_packet_readiness_report(&out, &report).expect("write readiness report");
        assert!(out.exists());
    }

    #[test]
    fn not_ready_when_required_file_is_missing() {
        let packet_dir = temp_dir("missing_file");
        let packet_id = "FNP-P2C-002";
        write_valid_packet(&packet_dir, packet_id);
        fs::remove_file(packet_dir.join("parity_gate.yaml")).expect("remove parity_gate");

        let report = validate_phase2c_packet(packet_id, &packet_dir);
        assert_eq!(report.status, "not_ready");
        assert!(
            report
                .missing_artifacts
                .iter()
                .any(|artifact| artifact == "parity_gate.yaml")
        );
    }

    #[test]
    fn not_ready_when_mandatory_field_is_missing() {
        let packet_dir = temp_dir("missing_field");
        let packet_id = "FNP-P2C-003";
        write_valid_packet(&packet_dir, packet_id);

        write(
            &packet_dir.join("fixture_manifest.json"),
            &format!(
                "{{\n  \"schema_version\": 1,\n  \"packet_id\": \"{packet_id}\",\n  \"fixtures\": [{{\"id\": \"ufunc_add\", \"input_path\": \"in.json\", \"oracle_case_id\": \"oracle_case\"}}]\n}}\n"
            ),
        );

        let report = validate_phase2c_packet(packet_id, &packet_dir);
        assert_eq!(report.status, "not_ready");
        assert!(
            report
                .missing_fields
                .iter()
                .any(|field| field.artifact == "fixture_manifest.json"
                    && field.field_path == "oracle_tests")
        );
    }
}
