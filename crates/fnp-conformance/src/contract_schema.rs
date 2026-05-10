#![forbid(unsafe_code)]

use crate::raptorq_artifacts::{DecodeProofArtifact, RaptorQSidecar};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortingLedgerFreshnessDiagnostic {
    pub packet_id: String,
    pub line_number: usize,
    pub column: String,
    pub stale_phrase: String,
    pub readiness_status: String,
    pub line: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortingLedgerFreshnessReport {
    pub schema_version: u8,
    pub ledger_path: String,
    pub phase2c_root: String,
    pub status: String,
    pub checked_packet_count: usize,
    pub ready_packet_count: usize,
    pub stale_row_count: usize,
    pub diagnostics: Vec<PortingLedgerFreshnessDiagnostic>,
    pub checked_at_unix_ms: u128,
}

impl PortingLedgerFreshnessReport {
    #[must_use]
    pub fn is_fresh(&self) -> bool {
        self.status == "fresh"
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

    let yaml_value: serde_yaml_ng::Value = match serde_yaml_ng::from_str(&raw) {
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

pub fn validate_phase2c_porting_ledger(
    ledger_path: &Path,
    phase2c_root: &Path,
) -> Result<PortingLedgerFreshnessReport, String> {
    let ledger = fs::read_to_string(ledger_path)
        .map_err(|err| format!("failed reading {}: {err}", ledger_path.display()))?;
    let readiness_reports = read_ready_packet_reports(phase2c_root)?;
    let rows = parse_packet_rows(&ledger);
    let stale_phrases = stale_packet_status_phrases();
    let mut diagnostics = Vec::new();

    for (packet_id, readiness) in &readiness_reports {
        if !readiness.is_ready()
            || !readiness.missing_artifacts.is_empty()
            || !readiness.missing_fields.is_empty()
            || !readiness.parse_errors.is_empty()
        {
            continue;
        }

        let Some(row) = rows.get(packet_id) else {
            diagnostics.push(PortingLedgerFreshnessDiagnostic {
                packet_id: packet_id.clone(),
                line_number: 0,
                column: "packet_row".to_string(),
                stale_phrase: "missing packet row".to_string(),
                readiness_status: readiness.status.clone(),
                line: String::new(),
            });
            continue;
        };

        for (column, value) in row.status_cells() {
            let normalized = normalize_stale_scan_text(value);
            for phrase in &stale_phrases {
                if normalized.contains(phrase) {
                    diagnostics.push(PortingLedgerFreshnessDiagnostic {
                        packet_id: packet_id.clone(),
                        line_number: row.line_number,
                        column: column.to_string(),
                        stale_phrase: phrase.to_string(),
                        readiness_status: readiness.status.clone(),
                        line: row.raw.clone(),
                    });
                }
            }
        }
    }

    diagnostics.sort_by(|lhs, rhs| {
        lhs.packet_id
            .cmp(&rhs.packet_id)
            .then(lhs.line_number.cmp(&rhs.line_number))
            .then(lhs.column.cmp(&rhs.column))
            .then(lhs.stale_phrase.cmp(&rhs.stale_phrase))
    });
    diagnostics.dedup();

    let stale_packets: BTreeSet<&str> = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.packet_id.as_str())
        .collect();
    let status = if diagnostics.is_empty() {
        "fresh"
    } else {
        "stale"
    };

    Ok(PortingLedgerFreshnessReport {
        schema_version: 1,
        ledger_path: ledger_path.display().to_string(),
        phase2c_root: phase2c_root.display().to_string(),
        status: status.to_string(),
        checked_packet_count: readiness_reports.len(),
        ready_packet_count: readiness_reports
            .values()
            .filter(|report| {
                report.is_ready()
                    && report.missing_artifacts.is_empty()
                    && report.missing_fields.is_empty()
                    && report.parse_errors.is_empty()
            })
            .count(),
        stale_row_count: stale_packets.len(),
        diagnostics,
        checked_at_unix_ms: now_unix_ms(),
    })
}

pub fn write_porting_ledger_freshness_report(
    output_path: &Path,
    report: &PortingLedgerFreshnessReport,
) -> Result<(), String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing porting ledger report: {err}"))?;
    fs::write(output_path, raw)
        .map_err(|err| format!("failed writing {}: {err}", output_path.display()))
}

#[derive(Debug, Clone)]
struct PacketLedgerRow {
    line_number: usize,
    raw: String,
    current_evidence_refs: String,
    parity_debt_status: String,
}

impl PacketLedgerRow {
    fn status_cells(&self) -> [(&'static str, &str); 2] {
        [
            ("current evidence refs", self.current_evidence_refs.as_str()),
            ("parity debt status", self.parity_debt_status.as_str()),
        ]
    }
}

fn read_ready_packet_reports(
    phase2c_root: &Path,
) -> Result<BTreeMap<String, PacketReadinessReport>, String> {
    let mut reports = BTreeMap::new();
    let entries = fs::read_dir(phase2c_root)
        .map_err(|err| format!("failed reading {}: {err}", phase2c_root.display()))?;

    for entry in entries {
        let entry = entry.map_err(|err| format!("failed reading phase2c entry: {err}"))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("FNP-P2C-") {
            continue;
        }
        let report_path = path.join("packet_readiness_report.json");
        if !report_path.is_file() {
            continue;
        }
        let raw = fs::read_to_string(&report_path)
            .map_err(|err| format!("failed reading {}: {err}", report_path.display()))?;
        let report: PacketReadinessReport = serde_json::from_str(&raw)
            .map_err(|err| format!("invalid readiness report {}: {err}", report_path.display()))?;
        reports.insert(report.packet_id.clone(), report);
    }

    Ok(reports)
}

fn parse_packet_rows(ledger: &str) -> BTreeMap<String, PacketLedgerRow> {
    let mut rows = BTreeMap::new();

    for (index, line) in ledger.lines().enumerate() {
        let Some(packet_id) = extract_packet_id_from_row(line) else {
            continue;
        };
        let cells = markdown_table_cells(line);
        if cells.len() < 12 {
            continue;
        }
        rows.insert(
            packet_id,
            PacketLedgerRow {
                line_number: index + 1,
                raw: line.to_string(),
                current_evidence_refs: cells.get(10).cloned().unwrap_or_default(),
                parity_debt_status: cells.get(11).cloned().unwrap_or_default(),
            },
        );
    }

    rows
}

fn extract_packet_id_from_row(line: &str) -> Option<String> {
    let (_, rest) = line.split_once("`FNP-P2C-")?;
    let (suffix, _) = rest.split_once('`')?;
    Some(format!("FNP-P2C-{suffix}"))
}

fn markdown_table_cells(line: &str) -> Vec<String> {
    line.trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn normalize_stale_scan_text(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn stale_packet_status_phrases() -> Vec<String> {
    [
        "anchor only",
        "open",
        "partial",
        "missing packet artifact",
        "missing packet artifacts",
        "missing artifacts",
        "e i pending",
        "packet e pending",
        "packet f pending",
        "packet g pending",
        "packet h pending",
        "packet i pending",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        PacketReadinessReport, validate_phase2c_packet, validate_phase2c_porting_ledger,
        write_packet_readiness_report, write_porting_ledger_freshness_report,
    };
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

    fn write_readiness_report(
        phase2c_root: &Path,
        packet_id: &str,
        status: &str,
        missing_artifacts: Vec<String>,
    ) {
        let report = PacketReadinessReport {
            schema_version: 1,
            contract_schema_version: super::CONTRACT_SCHEMA_VERSION.to_string(),
            packet_id: packet_id.to_string(),
            packet_dir: phase2c_root.join(packet_id).display().to_string(),
            status: status.to_string(),
            missing_artifacts,
            missing_fields: Vec::new(),
            parse_errors: Vec::new(),
            checked_at_unix_ms: 1,
        };
        write(
            &phase2c_root
                .join(packet_id)
                .join("packet_readiness_report.json"),
            &serde_json::to_string_pretty(&report).expect("serialize readiness report"),
        );
    }

    fn packet_ledger_row(packet_id: &str, current_evidence: &str, parity_status: &str) -> String {
        format!(
            "| `{packet_id}` | subsystem | anchors | contracts | strict | hardened | non-goals | unit | differential | e2e | {current_evidence} | {parity_status} | owner |\n"
        )
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
            "{\n  \"schema_version\": 1,\n  \"bundle_id\": \"packet_sidecar\",\n  \"generated_at_unix_ms\": 0,\n  \"source_hash\": \"abc\",\n  \"source_size\": 1,\n  \"object_id_u128\": 1,\n  \"symbol_size\": 256,\n  \"max_block_size\": 256,\n  \"repair_overhead\": 1.25,\n  \"source_blocks\": 1,\n  \"source_symbols\": 1,\n  \"repair_symbols\": 1,\n  \"total_symbols\": 0,\n  \"encoding_parallelism\": 1,\n  \"decoding_parallelism\": 1,\n  \"symbols\": []\n}\n",
        );

        write(
            &packet_dir.join("parity_report.decode_proof.json"),
            "{\n  \"schema_version\": 1,\n  \"bundle_id\": \"packet_sidecar\",\n  \"generated_at_unix_ms\": 0,\n  \"dropped_symbol\": null,\n  \"recovery_symbols_used\": 0,\n  \"recovery_success\": true,\n  \"expected_hash\": \"abc\",\n  \"recovered_hash\": \"abc\",\n  \"error\": null,\n  \"decoding_parallelism\": 1\n}\n",
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

    #[test]
    fn porting_ledger_rejects_stale_ready_packet_row() {
        let root = temp_dir("porting_ledger_stale");
        let phase2c_root = root.join("artifacts/phase2c");
        let ledger_path = root.join("PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md");
        write_readiness_report(&phase2c_root, "FNP-P2C-001", "ready", Vec::new());
        write(
            &ledger_path,
            &packet_ledger_row(
                "FNP-P2C-001",
                "anchor-only legacy map; missing packet artifacts",
                "open / partial",
            ),
        );

        let report =
            validate_phase2c_porting_ledger(&ledger_path, &phase2c_root).expect("validate ledger");
        assert_eq!(report.status, "stale");
        assert!(!report.is_fresh());
        assert_eq!(report.checked_packet_count, 1);
        assert_eq!(report.ready_packet_count, 1);
        assert_eq!(report.stale_row_count, 1);
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.packet_id == "FNP-P2C-001"
                    && diagnostic.stale_phrase == "anchor only")
        );
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.column == "parity debt status"
                    && diagnostic.stale_phrase == "open")
        );
    }

    #[test]
    fn porting_ledger_allows_stale_text_for_not_ready_packet() {
        let root = temp_dir("porting_ledger_not_ready");
        let phase2c_root = root.join("artifacts/phase2c");
        let ledger_path = root.join("PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md");
        write_readiness_report(
            &phase2c_root,
            "FNP-P2C-002",
            "not_ready",
            vec!["parity_report.json".to_string()],
        );
        write(
            &ledger_path,
            &packet_ledger_row("FNP-P2C-002", "anchor-only evidence", "open / partial"),
        );

        let report =
            validate_phase2c_porting_ledger(&ledger_path, &phase2c_root).expect("validate ledger");
        assert_eq!(report.status, "fresh");
        assert!(report.is_fresh());
        assert_eq!(report.checked_packet_count, 1);
        assert_eq!(report.ready_packet_count, 0);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn porting_ledger_accepts_clean_ready_packet_row_and_writes_report() {
        let root = temp_dir("porting_ledger_clean");
        let phase2c_root = root.join("artifacts/phase2c");
        let ledger_path = root.join("PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md");
        write_readiness_report(&phase2c_root, "FNP-P2C-003", "ready", Vec::new());
        write(
            &ledger_path,
            &packet_ledger_row(
                "FNP-P2C-003",
                "final_evidence_pack.json and packet_readiness_report.json",
                "ready: packet evidence validator-clean; residual breadth remains explicit",
            ),
        );

        let report =
            validate_phase2c_porting_ledger(&ledger_path, &phase2c_root).expect("validate ledger");
        assert_eq!(report.status, "fresh");
        assert_eq!(report.ready_packet_count, 1);
        assert!(report.diagnostics.is_empty());

        let out = root.join("report.json");
        write_porting_ledger_freshness_report(&out, &report).expect("write report");
        assert!(out.exists());
    }
}
