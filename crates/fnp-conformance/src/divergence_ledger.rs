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

    /// Appends diagnostics from a later pass (the repository audit) and re-derives `status`.
    pub fn extend_diagnostics(&mut self, diagnostics: Vec<DivergenceLedgerDiagnostic>) {
        self.diagnostics.extend(diagnostics);
        self.status = if self.has_errors() { "fail" } else { "pass" }.to_string();
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
    Vec::new()
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

/// Where a test is switched off in the source tree: an `#[ignore]` attribute, or an
/// `ExpectedFail("...")` outcome from the fnp-python conformance harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestMarkerKind {
    Ignore,
    ExpectedFail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestMarker {
    /// `path:line`, relative to the repository root.
    pub location: String,
    pub kind: TestMarkerKind,
    /// The attribute's reason string; `None` for a bare `#[ignore]`.
    pub reason: Option<String>,
}

/// Words that mark a reason as a tolerated NumPy divergence rather than a manual perf or
/// measurement run. Such a marker must cite a ledger row, because five DISC-011 tests sat
/// `#[ignore]`d for four months while tracked nowhere.
const PARITY_REASON_WORDS: [&str; 6] = [
    "parity gap",
    "parity debt",
    "disc-",
    "divergen",
    "xfail",
    "upstream",
];

/// Finds every `#[ignore]` attribute and `ExpectedFail("...")` literal in one source file.
/// Only attributes that open a line count, so doc comments and string fixtures that merely
/// mention `#[ignore]` are not markers.
#[must_use]
pub fn test_markers_in_source(path_label: &str, source: &str) -> Vec<TestMarker> {
    let mut markers = Vec::new();
    let mut offset = 0;
    for (index, line) in source.split_inclusive('\n').enumerate() {
        let line_start = offset;
        offset += line.len();
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        let (kind, after) = if let Some(rest) = trimmed.strip_prefix("#[ignore") {
            (TestMarkerKind::Ignore, rest)
        } else if !trimmed.starts_with("//")
            && let Some(position) = trimmed.find("ExpectedFail(")
            // `"ExpectedFail("` inside a string literal (this scanner's own needle) is not a call.
            && !trimmed[..position].ends_with('"')
        {
            (
                TestMarkerKind::ExpectedFail,
                &trimmed[position + "ExpectedFail(".len()..],
            )
        } else {
            continue;
        };
        let after_trimmed = after.trim_start();
        let reason = match kind {
            TestMarkerKind::Ignore if after_trimmed.starts_with(']') => None,
            TestMarkerKind::Ignore => {
                let Some(rest) = after_trimmed.strip_prefix('=') else {
                    continue;
                };
                let quote = line_start + indent + (trimmed.len() - rest.len());
                string_literal_at(source, quote)
            }
            TestMarkerKind::ExpectedFail => {
                if !after_trimmed.starts_with('"') {
                    // `ExpectedFail(&'static str)` in the enum itself, or a non-literal reason.
                    continue;
                }
                let quote = line_start + indent + (trimmed.len() - after_trimmed.len());
                string_literal_at(source, quote)
            }
        };
        markers.push(TestMarker {
            location: format!("{path_label}:{}", index + 1),
            kind,
            reason,
        });
    }
    markers
}

/// Reads the Rust string literal whose opening quote is at or after `start`, following
/// escapes and line breaks. Returns `None` if no terminated literal starts there.
fn string_literal_at(source: &str, start: usize) -> Option<String> {
    let tail = source.get(start..)?;
    let open = tail.find('"')?;
    let mut value = String::new();
    let mut chars = tail[open + 1..].chars();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some(value),
            '\\' => {
                if let Some(escaped) = chars.next() {
                    value.push(escaped);
                }
            }
            other => value.push(other),
        }
    }
    None
}

/// Walks `crates/*/{src,tests,benches,examples}` under `repo_root` and returns every test
/// marker, in path order.
pub fn scan_test_markers(repo_root: &Path) -> Result<Vec<TestMarker>, String> {
    let crates_dir = repo_root.join("crates");
    let mut crate_dirs = read_dir_sorted(&crates_dir)?;
    crate_dirs.retain(|path| path.is_dir());
    let mut markers = Vec::new();
    for crate_dir in crate_dirs {
        for subdir in ["src", "tests", "benches", "examples"] {
            let mut pending = vec![crate_dir.join(subdir)];
            while let Some(dir) = pending.pop() {
                if !dir.is_dir() {
                    continue;
                }
                for path in read_dir_sorted(&dir)?.into_iter().rev() {
                    if path.is_dir() {
                        pending.push(path);
                    } else if path.extension().is_some_and(|ext| ext == "rs") {
                        let source = fs::read_to_string(&path)
                            .map_err(|err| format!("read {}: {err}", path.display()))?;
                        let label = path
                            .strip_prefix(repo_root)
                            .unwrap_or(&path)
                            .to_string_lossy()
                            .replace('\\', "/");
                        markers.extend(test_markers_in_source(&label, &source));
                    }
                }
            }
        }
    }
    Ok(markers)
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut paths = fs::read_dir(dir)
        .map_err(|err| format!("read dir {}: {err}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("read dir {}: {err}", dir.display()))?;
    paths.sort();
    Ok(paths)
}

/// Ledger ids (`DIV-`, `PD-`, `UD-`) quoted anywhere in `text`.
#[must_use]
pub fn ledger_ids_in(text: &str) -> Vec<String> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'))
        .map(|token| token.trim_matches(|ch| ch == '-' || ch == '_'))
        .filter(|token| {
            let upper = token.to_ascii_uppercase();
            looks_like_divergence_id(&upper) && upper.len() > 4
        })
        .map(str::to_ascii_uppercase)
        .collect()
}

/// Every parity marker must name a live ledger row; every `#[ignore]` must give a reason.
#[must_use]
pub fn test_marker_diagnostics(
    entries: &[DivergenceLedgerEntry],
    markers: &[TestMarker],
) -> Vec<DivergenceLedgerDiagnostic> {
    let mut diagnostics = Vec::new();
    for marker in markers {
        let Some(reason) = marker.reason.as_deref() else {
            diagnostics.push(repository_error(
                "ignore_without_reason",
                None,
                format!(
                    "{}: bare #[ignore]; give a reason, and cite a ledger id if it tolerates a NumPy divergence",
                    marker.location
                ),
            ));
            continue;
        };
        let lowered = reason.to_ascii_lowercase();
        let tolerates_divergence = marker.kind == TestMarkerKind::ExpectedFail
            || PARITY_REASON_WORDS
                .iter()
                .any(|word| lowered.contains(word));
        if !tolerates_divergence {
            continue;
        }
        let ids = ledger_ids_in(reason);
        if ids.is_empty() {
            diagnostics.push(repository_error(
                "unregistered_parity_marker",
                None,
                format!(
                    "{}: marker tolerates a NumPy divergence but cites no DIV-/PD-/UD- row of docs/DIVERGENCES.md",
                    marker.location
                ),
            ));
        }
        for id in ids {
            if !entries.iter().any(|entry| entry.id == id) {
                diagnostics.push(repository_error(
                    "unknown_ledger_id",
                    Some(id.clone()),
                    format!("{}: cites {id}, which is not a ledger row", marker.location),
                ));
            }
        }
    }
    diagnostics
}

/// `path.rs::test_fn` probes named in a ledger evidence cell.
#[must_use]
pub fn evidence_probes(evidence: &str) -> Vec<(String, String)> {
    evidence
        .split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';' | '(' | ')'))
        .map(|token| token.trim_matches(|ch| matches!(ch, '`' | '.' | ':')))
        .filter_map(|token| {
            let (path, item) = token.split_once(".rs::")?;
            let name = item.rsplit("::").next()?;
            (!name.is_empty()).then(|| (format!("{path}.rs"), name.to_string()))
        })
        .collect()
}

/// Each ledger row must name at least one probe, and every probe must still exist, so a
/// row cannot outlive the test that proves it.
#[must_use]
pub fn evidence_probe_diagnostics(
    repo_root: &Path,
    entries: &[DivergenceLedgerEntry],
) -> Vec<DivergenceLedgerDiagnostic> {
    let mut diagnostics = Vec::new();
    for entry in entries {
        let probes = evidence_probes(&entry.evidence);
        if probes.is_empty() {
            diagnostics.push(repository_error(
                "ledger_row_without_probe",
                Some(entry.id.clone()),
                "evidence names no `path.rs::test_fn` probe".to_string(),
            ));
        }
        for (path, name) in probes {
            let defined = fs::read_to_string(repo_root.join(&path))
                .is_ok_and(|source| source_defines_fn(&source, &name));
            if !defined {
                diagnostics.push(repository_error(
                    "ledger_probe_missing",
                    Some(entry.id.clone()),
                    format!("evidence probe {path}::{name} does not exist"),
                ));
            }
        }
    }
    diagnostics
}

fn source_defines_fn(source: &str, name: &str) -> bool {
    let needle = format!("fn {name}");
    source.match_indices(&needle).any(|(index, _)| {
        let before_ok = source[..index]
            .chars()
            .next_back()
            .is_none_or(char::is_whitespace);
        let after_ok = source[index + needle.len()..]
            .chars()
            .next()
            .is_some_and(|ch| ch == '(' || ch == '<');
        before_ok && after_ok
    })
}

/// The repository half of the gate: test markers against ledger rows, and ledger rows
/// against their probes. Adds one `info` line so a run shows how much it inspected.
pub fn audit_repository(
    repo_root: &Path,
    entries: &[DivergenceLedgerEntry],
) -> Result<Vec<DivergenceLedgerDiagnostic>, String> {
    let markers = scan_test_markers(repo_root)?;
    let citing = markers
        .iter()
        .filter(|marker| {
            marker
                .reason
                .as_deref()
                .is_some_and(|reason| !ledger_ids_in(reason).is_empty())
        })
        .count();
    let mut diagnostics = vec![DivergenceLedgerDiagnostic {
        severity: "info".to_string(),
        reason_code: "repository_audit".to_string(),
        ledger_id: None,
        case_id: None,
        message: format!(
            "scanned {} test markers under {}/crates; {citing} cite a ledger row",
            markers.len(),
            repo_root.display()
        ),
    }];
    diagnostics.extend(test_marker_diagnostics(entries, &markers));
    diagnostics.extend(evidence_probe_diagnostics(repo_root, entries));
    Ok(diagnostics)
}

fn repository_error(
    reason_code: &str,
    ledger_id: Option<String>,
    message: String,
) -> DivergenceLedgerDiagnostic {
    DivergenceLedgerDiagnostic {
        severity: "error".to_string(),
        reason_code: reason_code.to_string(),
        ledger_id,
        case_id: None,
        message,
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
        DEFAULT_DIVERGENCE_LEDGER_PATH, DivergenceDisposition, DivergenceExpectation,
        TestMarkerKind, audit_repository, evaluate_divergence_ledger, evidence_probe_diagnostics,
        evidence_probes, expectations_from_diagnostic_cases, load_ledger, parse_ledger_markdown,
        test_marker_diagnostics, test_markers_in_source,
    };
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

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

    #[test]
    fn test_markers_parse_attributes_but_not_mentions() {
        let source = [
            "/// A doc comment mentioning #[ignore] is not a marker.",
            "#[test]",
            "#[ignore]",
            "fn bare() {}",
            "    #[ignore = \"perf A/B; run with --ignored\"]",
            "#[ignore = \"PARITY GAP (PD-2F6L4): spans",
            "two lines with an \\\"escaped\\\" quote\"]",
            "    ExpectedFail(&'static str),",
            "        CaseOutcome::ExpectedFail(\"DIV-001 zero-fill\")",
            "// CaseOutcome::ExpectedFail(\"commented out\")",
            "    let needle = trimmed.find(\"ExpectedFail(\");",
        ]
        .join("\n");

        let markers = test_markers_in_source("crates/x/tests/t.rs", &source);

        let summary = markers
            .iter()
            .map(|marker| {
                (
                    marker.location.as_str(),
                    marker.kind,
                    marker.reason.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                ("crates/x/tests/t.rs:3", TestMarkerKind::Ignore, None),
                (
                    "crates/x/tests/t.rs:5",
                    TestMarkerKind::Ignore,
                    Some("perf A/B; run with --ignored")
                ),
                (
                    "crates/x/tests/t.rs:6",
                    TestMarkerKind::Ignore,
                    Some("PARITY GAP (PD-2F6L4): spans\ntwo lines with an \"escaped\" quote")
                ),
                (
                    "crates/x/tests/t.rs:9",
                    TestMarkerKind::ExpectedFail,
                    Some("DIV-001 zero-fill")
                ),
            ]
        );
    }

    #[test]
    fn unregistered_parity_ignore_fails_and_names_its_location() {
        let entries = parse_ledger_markdown(LEDGER).expect("parse ledger");
        let source = "#[test]\n#[ignore = \"PARITY GAP: test\"]\nfn scratch() {}\n";
        let markers = test_markers_in_source("crates/x/tests/scratch.rs", source);

        let diagnostics = test_marker_diagnostics(&entries, &markers);

        assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
        assert_eq!(diagnostics[0].severity, "error");
        assert_eq!(diagnostics[0].reason_code, "unregistered_parity_marker");
        assert!(
            diagnostics[0]
                .message
                .starts_with("crates/x/tests/scratch.rs:2:"),
            "{}",
            diagnostics[0].message
        );
    }

    #[test]
    fn parity_markers_must_cite_live_rows_and_ignores_need_reasons() {
        let entries = parse_ledger_markdown(LEDGER).expect("parse ledger");
        let source = [
            "#[ignore = \"parity debt tracked as PD-2F6L4\"]",
            "#[ignore = \"perf timing; run with --release -- --ignored\"]",
            "#[ignore = \"DISC-011 divergence, see PD-GONE\"]",
            "#[ignore]",
            "CaseOutcome::ExpectedFail(\"known\")",
        ]
        .join("\n");
        let markers = test_markers_in_source("t.rs", &source);

        let diagnostics = test_marker_diagnostics(&entries, &markers);

        let codes = diagnostics
            .iter()
            .map(|diagnostic| (diagnostic.reason_code.as_str(), diagnostic.message.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            codes.iter().map(|(code, _)| *code).collect::<Vec<_>>(),
            [
                "unknown_ledger_id",
                "ignore_without_reason",
                "unregistered_parity_marker"
            ],
            "{codes:#?}"
        );
        assert!(codes[0].1.starts_with("t.rs:3:"));
        assert!(codes[1].1.starts_with("t.rs:4:"));
        assert!(codes[2].1.starts_with("t.rs:5:"));
    }

    #[test]
    fn ledger_rows_need_an_existing_probe() {
        assert_eq!(
            evidence_probes("`crates/a/tests/b.rs::c_d`, crates/e.rs::m::f; prose"),
            [
                ("crates/a/tests/b.rs".to_string(), "c_d".to_string()),
                ("crates/e.rs".to_string(), "f".to_string()),
            ]
        );
        let live =
            "crates/fnp-conformance/src/divergence_ledger.rs::ledger_rows_need_an_existing_probe";
        let ledger = format!(
            "| ID | Disposition | Surface | Affected behavior | NumPy scope | Strict behavior | Hardened behavior | Follow-up | Evidence |\n\
             |---|---|---|---|---|---|---|---|---|\n\
             | PD-LIVE | parity_debt | s | b | n | x | y | f | {live} |\n\
             | PD-GONE | parity_debt | s | b | n | x | y | f | crates/fnp-conformance/src/divergence_ledger.rs::no_such_probe |\n\
             | PD-NONE | parity_debt | s | b | n | x | y | f | a sentence with no probe |\n"
        );
        let entries = parse_ledger_markdown(&ledger).expect("parse ledger");

        let diagnostics = evidence_probe_diagnostics(&repo_root(), &entries);

        let found = diagnostics
            .iter()
            .map(|diagnostic| {
                (
                    diagnostic.ledger_id.as_deref().unwrap_or(""),
                    diagnostic.reason_code.as_str(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            found,
            [
                ("PD-GONE", "ledger_probe_missing"),
                ("PD-NONE", "ledger_row_without_probe")
            ]
        );
    }

    /// The gate itself, run over this repository: every tolerated divergence in a test
    /// marker names a live ledger row, and every row's probe exists.
    #[test]
    fn repository_markers_and_ledger_rows_agree() {
        let root = repo_root();
        let entries =
            load_ledger(&root.join(DEFAULT_DIVERGENCE_LEDGER_PATH)).expect("load repo ledger");

        let diagnostics = audit_repository(&root, &entries).expect("audit repository");

        let errors = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == "error")
            .collect::<Vec<_>>();
        assert!(errors.is_empty(), "divergence ledger gate: {errors:#?}");
        // Non-vacuity: the scan must actually see this repository's markers (it carries
        // dozens of perf-run `#[ignore]`s) and at least one ledger citation.
        let summary = &diagnostics[0].message;
        let scanned = summary
            .split_whitespace()
            .nth(1)
            .and_then(|count| count.parse::<usize>().ok())
            .expect("audit summary names a marker count");
        assert!(scanned >= 50, "{summary}");
        assert!(!summary.contains("; 0 cite"), "{summary}");
    }
}
