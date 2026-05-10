#![forbid(unsafe_code)]

use crate::diagnostic_oracle::DiagnosticCase;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const DIVERGENCE_LEDGER_SCHEMA_VERSION: &str = "diagnostic-divergence-ledger-v1";
pub const DEFAULT_DIVERGENCE_LEDGER_PATH: &str = "docs/DIVERGENCES.md";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceDisposition {
    Intentional,
    ParityDebt,
    UpstreamDrift,
}

impl DivergenceDisposition {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Intentional => "intentional",
            Self::ParityDebt => "parity_debt",
            Self::UpstreamDrift => "upstream_drift",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match normalize_cell(raw).as_str() {
            "intentional" | "accepted" => Some(Self::Intentional),
            "parity_debt" | "parity debt" | "debt" | "unimplemented_parity_debt" => {
                Some(Self::ParityDebt)
            }
            "upstream_drift" | "upstream drift" | "version_drift" => Some(Self::UpstreamDrift),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceLedgerEntry {
    pub id: String,
    pub disposition: DivergenceDisposition,
    pub surface: String,
    pub behavior: String,
    pub numpy_scope: String,
    pub strict_behavior: String,
    pub hardened_behavior: String,
    pub follow_up: String,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceExpectation {
    pub case_id: String,
    pub surface: String,
    pub disposition: DivergenceDisposition,
    pub ledger_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceLedgerReport {
    pub schema_version: String,
    pub status: String,
    pub ledger_path: String,
    pub entry_count: usize,
    pub intentional_count: usize,
    pub parity_debt_count: usize,
    pub upstream_drift_count: usize,
    pub expectation_count: usize,
    pub diagnostics: Vec<DivergenceLedgerDiagnostic>,
}

impl DivergenceLedgerReport {
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == "error")
    }

    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        output.push_str("# Divergence Ledger Check\n\n");
        output.push_str(&format!("status: `{}`\n\n", self.status));
        output.push_str("| metric | value |\n");
        output.push_str("|---|---:|\n");
        output.push_str(&format!("| entries | {} |\n", self.entry_count));
        output.push_str(&format!("| intentional | {} |\n", self.intentional_count));
        output.push_str(&format!("| parity debt | {} |\n", self.parity_debt_count));
        output.push_str(&format!(
            "| upstream drift | {} |\n",
            self.upstream_drift_count
        ));
        output.push_str(&format!(
            "| expectations | {} |\n\n",
            self.expectation_count
        ));
        output.push_str("| severity | reason | ledger id | case id | message |\n");
        output.push_str("|---|---|---|---|---|\n");
        for diagnostic in &self.diagnostics {
            output.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                diagnostic.severity,
                diagnostic.reason_code,
                diagnostic.ledger_id.as_deref().unwrap_or(""),
                diagnostic.case_id.as_deref().unwrap_or(""),
                diagnostic.message.replace('|', "\\|")
            ));
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceLedgerDiagnostic {
    pub severity: String,
    pub reason_code: String,
    pub ledger_id: Option<String>,
    pub case_id: Option<String>,
    pub message: String,
}

pub fn load_ledger(path: &Path) -> Result<Vec<DivergenceLedgerEntry>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("read divergence ledger {}: {err}", path.display()))?;
    parse_ledger_markdown(&raw)
}

pub fn parse_ledger_markdown(raw: &str) -> Result<Vec<DivergenceLedgerEntry>, String> {
    let mut entries = Vec::new();
    for (line_number, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
            continue;
        }
        if is_header_or_separator(trimmed) {
            continue;
        }
        let cells = parse_table_cells(trimmed);
        if cells.len() < 9 {
            continue;
        }
        let [
            id_cell,
            disposition_cell,
            surface_cell,
            behavior_cell,
            numpy_scope_cell,
            strict_cell,
            hardened_cell,
            follow_up_cell,
            evidence_cell,
            ..,
        ] = cells.as_slice()
        else {
            continue;
        };
        let id = normalize_id(id_cell);
        if id.is_empty() || !looks_like_divergence_id(&id) {
            continue;
        }
        let Some(disposition) = DivergenceDisposition::parse(disposition_cell) else {
            return Err(format!(
                "line {}: unknown divergence disposition {:?}",
                line_number + 1,
                disposition_cell
            ));
        };
        entries.push(DivergenceLedgerEntry {
            id,
            disposition,
            surface: normalize_cell(surface_cell),
            behavior: normalize_cell(behavior_cell),
            numpy_scope: normalize_cell(numpy_scope_cell),
            strict_behavior: normalize_cell(strict_cell),
            hardened_behavior: normalize_cell(hardened_cell),
            follow_up: normalize_cell(follow_up_cell),
            evidence: normalize_cell(evidence_cell),
        });
    }
    validate_entries(&entries)?;
    Ok(entries)
}

pub fn validate_entries(entries: &[DivergenceLedgerEntry]) -> Result<(), String> {
    let mut seen = BTreeMap::<&str, usize>::new();
    for entry in entries {
        if entry.id.trim().is_empty() {
            return Err("divergence ledger entry id must not be empty".to_string());
        }
        if entry.surface.trim().is_empty() {
            return Err(format!("{}: surface must not be empty", entry.id));
        }
        if entry.behavior.trim().is_empty() {
            return Err(format!("{}: behavior must not be empty", entry.id));
        }
        if entry.strict_behavior.trim().is_empty() {
            return Err(format!("{}: strict behavior must not be empty", entry.id));
        }
        if entry.hardened_behavior.trim().is_empty() {
            return Err(format!("{}: hardened behavior must not be empty", entry.id));
        }
        if let Some(previous_line) = seen.insert(entry.id.as_str(), seen.len() + 1) {
            return Err(format!(
                "{}: duplicate divergence ledger id (first seen at entry {})",
                entry.id, previous_line
            ));
        }
    }
    Ok(())
}

#[must_use]
pub fn expectations_from_diagnostic_cases(cases: &[DiagnosticCase]) -> Vec<DivergenceExpectation> {
    cases
        .iter()
        .filter_map(|case| {
            case.intentional_divergence
                .as_ref()
                .map(|ledger_id| DivergenceExpectation {
                    case_id: case.id.clone(),
                    surface: case.surface.clone(),
                    disposition: DivergenceDisposition::Intentional,
                    ledger_id: ledger_id.clone(),
                })
        })
        .collect()
}

#[must_use]
pub fn default_diagnostic_expectations() -> Vec<DivergenceExpectation> {
    vec![
        DivergenceExpectation {
            case_id: "fnp-python-runtime-warning-parity".to_string(),
            surface: "fnp-python arithmetic/statistics diagnostics".to_string(),
            disposition: DivergenceDisposition::ParityDebt,
            ledger_id: "PD-2F6L4".to_string(),
        },
        DivergenceExpectation {
            case_id: "fnp-python-indexing-text-io-diagnostic-classes".to_string(),
            surface: "fnp-python indexing/text IO diagnostics".to_string(),
            disposition: DivergenceDisposition::ParityDebt,
            ledger_id: "PD-09EPN".to_string(),
        },
    ]
}

#[must_use]
pub fn evaluate_divergence_ledger(
    ledger_path: &Path,
    entries: &[DivergenceLedgerEntry],
    expectations: &[DivergenceExpectation],
) -> DivergenceLedgerReport {
    let entry_by_id = entries
        .iter()
        .map(|entry| (entry.id.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let mut diagnostics = Vec::new();
    for expectation in expectations {
        match entry_by_id.get(expectation.ledger_id.as_str()) {
            Some(entry) if entry.disposition == expectation.disposition => {}
            Some(entry) => diagnostics.push(DivergenceLedgerDiagnostic {
                severity: "error".to_string(),
                reason_code: "divergence_disposition_mismatch".to_string(),
                ledger_id: Some(expectation.ledger_id.clone()),
                case_id: Some(expectation.case_id.clone()),
                message: format!(
                    "diagnostic expectation requires {} but ledger records {}",
                    expectation.disposition.as_str(),
                    entry.disposition.as_str()
                ),
            }),
            None => diagnostics.push(DivergenceLedgerDiagnostic {
                severity: "error".to_string(),
                reason_code: match expectation.disposition {
                    DivergenceDisposition::Intentional => "missing_intentional_divergence",
                    DivergenceDisposition::ParityDebt => "missing_parity_debt",
                    DivergenceDisposition::UpstreamDrift => "missing_upstream_drift",
                }
                .to_string(),
                ledger_id: Some(expectation.ledger_id.clone()),
                case_id: Some(expectation.case_id.clone()),
                message: format!(
                    "{} expectation has no ledger entry for surface {}",
                    expectation.disposition.as_str(),
                    expectation.surface
                ),
            }),
        }
    }
    let intentional_count = entries
        .iter()
        .filter(|entry| entry.disposition == DivergenceDisposition::Intentional)
        .count();
    let parity_debt_count = entries
        .iter()
        .filter(|entry| entry.disposition == DivergenceDisposition::ParityDebt)
        .count();
    let upstream_drift_count = entries
        .iter()
        .filter(|entry| entry.disposition == DivergenceDisposition::UpstreamDrift)
        .count();
    let status = if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == "error")
    {
        "fail"
    } else {
        "pass"
    };
    DivergenceLedgerReport {
        schema_version: DIVERGENCE_LEDGER_SCHEMA_VERSION.to_string(),
        status: status.to_string(),
        ledger_path: ledger_path.display().to_string(),
        entry_count: entries.len(),
        intentional_count,
        parity_debt_count,
        upstream_drift_count,
        expectation_count: expectations.len(),
        diagnostics,
    }
}

fn parse_table_cells(line: &str) -> Vec<String> {
    line.trim_matches('|')
        .split('|')
        .map(normalize_cell)
        .collect()
}

fn is_header_or_separator(line: &str) -> bool {
    let cells = parse_table_cells(line);
    cells
        .iter()
        .all(|cell| cell.chars().all(|ch| matches!(ch, '-' | ':' | ' ')))
        || cells
            .first()
            .is_some_and(|first| normalize_cell(first).eq_ignore_ascii_case("id"))
}

fn looks_like_divergence_id(id: &str) -> bool {
    id.starts_with("DIV-") || id.starts_with("PD-") || id.starts_with("UD-")
}

fn normalize_id(raw: &str) -> String {
    normalize_cell(raw).to_ascii_uppercase()
}

fn normalize_cell(raw: &str) -> String {
    raw.trim()
        .trim_matches('`')
        .replace("<br>", "; ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use crate::diagnostic_oracle::{
        DiagnosticCase, DiagnosticExpectation, DiagnosticMode, DiagnosticOutcome,
        DiagnosticRequirementLevel,
    };
    use crate::divergence_ledger::{
        DivergenceDisposition, DivergenceExpectation, evaluate_divergence_ledger,
        expectations_from_diagnostic_cases, parse_ledger_markdown,
    };
    use std::path::Path;

    const LEDGER: &str = r#"
| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |
|---|---|---|---|---|---|---|---|---|
| `DIV-001` | intentional | fnp-python | deliberately rejected legacy alias | NumPy 2.x | fail closed | fail closed with audit | none | policy note |
| `PD-2F6L4` | parity_debt | fnp-python | missing RuntimeWarning | NumPy 2.x | parity debt | parity debt | franken_numpy-2f6l4 | diagnostic shard |
"#;

    #[test]
    fn divergence_ledger_parses_markdown_entries() {
        let entries = parse_ledger_markdown(LEDGER).expect("parse ledger");

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "DIV-001");
        assert_eq!(entries[0].disposition, DivergenceDisposition::Intentional);
        assert_eq!(entries[1].id, "PD-2F6L4");
        assert_eq!(entries[1].disposition, DivergenceDisposition::ParityDebt);
    }

    #[test]
    fn divergence_ledger_fails_missing_intentional_case() {
        let entries = parse_ledger_markdown(LEDGER).expect("parse ledger");
        let expectations = [DivergenceExpectation {
            case_id: "synthetic_case".to_string(),
            surface: "fnp-python".to_string(),
            disposition: DivergenceDisposition::Intentional,
            ledger_id: "DIV-MISSING".to_string(),
        }];

        let report =
            evaluate_divergence_ledger(Path::new("docs/DIVERGENCES.md"), &entries, &expectations);

        assert!(report.has_errors());
        assert_eq!(report.status, "fail");
        let diagnostic = report
            .diagnostics
            .first()
            .expect("missing divergence diagnostic");
        assert_eq!(diagnostic.reason_code, "missing_intentional_divergence");
    }

    #[test]
    fn divergence_ledger_distinguishes_parity_debt() {
        let entries = parse_ledger_markdown(LEDGER).expect("parse ledger");
        let expectations = [DivergenceExpectation {
            case_id: "warning_parity".to_string(),
            surface: "fnp-python".to_string(),
            disposition: DivergenceDisposition::ParityDebt,
            ledger_id: "PD-2F6L4".to_string(),
        }];

        let report =
            evaluate_divergence_ledger(Path::new("docs/DIVERGENCES.md"), &entries, &expectations);

        assert_eq!(report.status, "pass");
        assert_eq!(report.intentional_count, 1);
        assert_eq!(report.parity_debt_count, 1);
    }

    #[test]
    fn divergence_ledger_extracts_intentional_diagnostic_cases() {
        let case = DiagnosticCase {
            id: "case_with_divergence".to_string(),
            surface: "fnp-python".to_string(),
            requirement_level: DiagnosticRequirementLevel::Must,
            mode: DiagnosticMode::Strict,
            python: "pass".to_string(),
            expected: DiagnosticExpectation {
                outcome: DiagnosticOutcome::Success,
                exception_class: None,
                warning_categories: Vec::new(),
                message_fragments: Vec::new(),
            },
            version_guards: Vec::new(),
            intentional_divergence: Some("DIV-001".to_string()),
            exploratory: false,
        };

        let expectations = expectations_from_diagnostic_cases(&[case]);

        assert_eq!(
            expectations,
            [DivergenceExpectation {
                case_id: "case_with_divergence".to_string(),
                surface: "fnp-python".to_string(),
                disposition: DivergenceDisposition::Intentional,
                ledger_id: "DIV-001".to_string(),
            }]
        );
    }
}
