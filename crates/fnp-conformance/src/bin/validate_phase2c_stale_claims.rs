#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_REPORT_PATH: &str = "artifacts/logs/phase2c_stale_claims_report.json";
const DEFAULT_MARKDOWN_PATH: &str = "artifacts/logs/phase2c_stale_claims_report.md";
const CENTRAL_LEDGER: &str = "artifacts/contracts/PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md";
const PACKET_DOCS: &[&str] = &[
    "behavior_extraction_ledger.md",
    "risk_note.md",
    "implementation_plan.md",
];

fn main() {
    if let Err(err) = run() {
        eprintln!("validate_phase2c_stale_claims failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse()?;
    let report = build_report(&options.repo_root, &options.phase2c_root)?;
    write_json_report(&options.report_out, &report)?;
    if let Some(markdown_out) = &options.markdown_out {
        write_markdown_report(markdown_out, &report)?;
    }

    println!(
        "status={} scanned_files={} ready_packets={} diagnostics={} report={}",
        report.status,
        report.scanned_file_count,
        report.ready_packet_count,
        report.diagnostics.len(),
        options.report_out.display()
    );
    if let Some(markdown_out) = &options.markdown_out {
        println!("markdown_report={}", markdown_out.display());
    }
    if options.print_diagnostics {
        for diagnostic in &report.diagnostics {
            println!(
                "diagnostic packet={} source={}:{} phrase={} line={}",
                diagnostic.packet_id.as_deref().unwrap_or("unknown"),
                diagnostic.source_path,
                diagnostic.line_number,
                diagnostic.stale_phrase,
                diagnostic.line
            );
        }
    }

    if !report.is_fresh() {
        std::process::exit(2);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Options {
    repo_root: PathBuf,
    phase2c_root: PathBuf,
    report_out: PathBuf,
    markdown_out: Option<PathBuf>,
    print_diagnostics: bool,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut repo_root = default_repo_root();
        let mut phase2c_root = repo_root.join("artifacts/phase2c");
        let mut report_out = repo_root.join(DEFAULT_REPORT_PATH);
        let mut markdown_out: Option<PathBuf> = None;
        let mut print_diagnostics = false;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--repo-root requires a value".to_string())?;
                    let value = PathBuf::from(value);
                    phase2c_root = value.join("artifacts/phase2c");
                    if report_out.ends_with(DEFAULT_REPORT_PATH) {
                        report_out = value.join(DEFAULT_REPORT_PATH);
                    }
                    repo_root = value;
                }
                "--phase2c-root" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--phase2c-root requires a value".to_string())?;
                    phase2c_root = PathBuf::from(value);
                }
                "--report-out" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--report-out requires a value".to_string())?;
                    report_out = PathBuf::from(value);
                }
                "--markdown-out" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--markdown-out requires a value".to_string())?;
                    markdown_out = Some(PathBuf::from(value));
                }
                "--default-markdown-out" => {
                    markdown_out = Some(repo_root.join(DEFAULT_MARKDOWN_PATH));
                }
                "--print-diagnostics" => {
                    print_diagnostics = true;
                }
                "--help" | "-h" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}\n{}", usage())),
            }
        }

        Ok(Self {
            repo_root,
            phase2c_root,
            report_out,
            markdown_out,
            print_diagnostics,
        })
    }
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin validate_phase2c_stale_claims -- [--repo-root <path>] [--phase2c-root <path>] [--report-out <path>] [--markdown-out <path>] [--default-markdown-out] [--print-diagnostics]".to_string()
}

fn default_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Debug, Serialize)]
struct StaleClaimReport {
    schema_version: u32,
    status: String,
    repo_root: String,
    phase2c_root: String,
    scanned_file_count: usize,
    scanned_sources: Vec<String>,
    ready_packet_count: usize,
    ready_packets: Vec<String>,
    diagnostics: Vec<StaleClaimDiagnostic>,
    checked_at_unix_ms: u128,
}

impl StaleClaimReport {
    fn is_fresh(&self) -> bool {
        self.diagnostics.is_empty()
    }

    fn to_markdown(&self) -> String {
        let mut out = Vec::new();
        out.push("# Phase2C Stale-Claim Report".to_string());
        out.push(String::new());
        out.push(format!("- status: {}", self.status));
        out.push(format!("- scanned_file_count: {}", self.scanned_file_count));
        out.push(format!("- ready_packet_count: {}", self.ready_packet_count));
        out.push(format!("- diagnostics: {}", self.diagnostics.len()));
        out.push(String::new());
        out.push("| packet | source | line | stale phrase | context |".to_string());
        out.push("|---|---|---:|---|---|".to_string());
        for diagnostic in &self.diagnostics {
            out.push(format!(
                "| {} | `{}` | {} | `{}` | {} |",
                diagnostic.packet_id.as_deref().unwrap_or("unknown"),
                diagnostic.source_path,
                diagnostic.line_number,
                diagnostic.stale_phrase,
                markdown_escape(&diagnostic.context)
            ));
        }
        out.push(String::new());
        out.join("\n")
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct StaleClaimDiagnostic {
    packet_id: Option<String>,
    source_path: String,
    line_number: usize,
    stale_phrase: String,
    context: String,
    line: String,
    remediation: String,
}

#[derive(Debug, Deserialize)]
struct PacketReadinessReport {
    packet_id: String,
    status: String,
    #[serde(default)]
    missing_artifacts: Vec<serde_json::Value>,
    #[serde(default)]
    missing_fields: Vec<serde_json::Value>,
    #[serde(default)]
    parse_errors: Vec<serde_json::Value>,
}

impl PacketReadinessReport {
    fn is_ready(&self) -> bool {
        self.status == "ready"
            && self.missing_artifacts.is_empty()
            && self.missing_fields.is_empty()
            && self.parse_errors.is_empty()
    }
}

#[derive(Debug, Clone)]
struct SourceDocument {
    packet_id: Option<String>,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct StalePattern {
    phrase: &'static str,
    context: &'static [&'static str],
}

fn build_report(repo_root: &Path, phase2c_root: &Path) -> Result<StaleClaimReport, String> {
    let readiness = ready_packet_reports(phase2c_root)?;
    let ready_packets = readiness.keys().cloned().collect::<Vec<_>>();
    let sources = source_documents(repo_root, phase2c_root, &ready_packets);
    let patterns = stale_patterns();
    let mut diagnostics = Vec::new();
    let mut scanned_sources = Vec::new();

    for source in &sources {
        if !source.path.is_file() {
            continue;
        }
        scanned_sources.push(display_relative(repo_root, &source.path));
        let raw = fs::read_to_string(&source.path)
            .map_err(|err| format!("failed reading {}: {err}", source.path.display()))?;
        diagnostics.extend(scan_source(
            repo_root,
            &source.path,
            source.packet_id.clone(),
            &raw,
            &patterns,
        ));
    }

    diagnostics.sort();
    diagnostics.dedup();
    scanned_sources.sort();

    let status = if diagnostics.is_empty() {
        "fresh"
    } else {
        "stale"
    }
    .to_string();

    Ok(StaleClaimReport {
        schema_version: SCHEMA_VERSION,
        status,
        repo_root: repo_root.display().to_string(),
        phase2c_root: phase2c_root.display().to_string(),
        scanned_file_count: scanned_sources.len(),
        scanned_sources,
        ready_packet_count: ready_packets.len(),
        ready_packets,
        diagnostics,
        checked_at_unix_ms: now_unix_ms(),
    })
}

fn ready_packet_reports(
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
        if report.is_ready() {
            reports.insert(report.packet_id.clone(), report);
        }
    }
    Ok(reports)
}

fn source_documents(
    repo_root: &Path,
    phase2c_root: &Path,
    ready_packets: &[String],
) -> Vec<SourceDocument> {
    let mut sources = vec![SourceDocument {
        packet_id: None,
        path: repo_root.join(CENTRAL_LEDGER),
    }];

    for packet_id in ready_packets {
        let packet_dir = phase2c_root.join(packet_id);
        for doc in PACKET_DOCS {
            sources.push(SourceDocument {
                packet_id: Some(packet_id.clone()),
                path: packet_dir.join(doc),
            });
        }
    }

    sources
}

fn scan_source(
    repo_root: &Path,
    source_path: &Path,
    default_packet_id: Option<String>,
    raw: &str,
    patterns: &[StalePattern],
) -> Vec<StaleClaimDiagnostic> {
    let source_path_display = display_relative(repo_root, source_path);
    let mut diagnostics = Vec::new();

    for (index, line) in raw.lines().enumerate() {
        let normalized = normalize(line);
        if normalized.is_empty() || allowed_context(&normalized) {
            continue;
        }
        let stale_phrases = matching_stale_phrases(&normalized, patterns);
        if stale_phrases.is_empty() {
            continue;
        }
        for stale_phrase in stale_phrases {
            diagnostics.push(StaleClaimDiagnostic {
                packet_id: packet_id_for_line(line).or_else(|| default_packet_id.clone()),
                source_path: source_path_display.clone(),
                line_number: index + 1,
                stale_phrase: stale_phrase.to_string(),
                context: context_for_line(&normalized),
                line: line.trim().to_string(),
                remediation: "Refresh stale evidence/status wording to current validator-ready packet artifacts, or rewrite it as explicit residual parity debt.".to_string(),
            });
        }
    }

    diagnostics
}

fn matching_stale_phrases<'a>(normalized: &str, patterns: &'a [StalePattern]) -> Vec<&'a str> {
    patterns
        .iter()
        .filter_map(|pattern| {
            if !normalized.contains(pattern.phrase) {
                return None;
            }
            if pattern.context.is_empty()
                || pattern
                    .context
                    .iter()
                    .any(|keyword| normalized.contains(keyword))
            {
                return Some(pattern.phrase);
            }
            None
        })
        .collect()
}

fn stale_patterns() -> Vec<StalePattern> {
    vec![
        StalePattern {
            phrase: "planned verification hook",
            context: &[],
        },
        StalePattern {
            phrase: "not yet represented",
            context: &["evidence", "artifact", "validator", "packet"],
        },
        StalePattern {
            phrase: "not yet materialized",
            context: &["evidence", "artifact", "validator", "packet"],
        },
        StalePattern {
            phrase: "not yet emitted",
            context: &["evidence", "artifact", "validator", "packet"],
        },
        StalePattern {
            phrase: "not yet wired",
            context: &["evidence", "artifact", "validator", "packet"],
        },
        StalePattern {
            phrase: "anchor only",
            context: &["evidence", "status", "proof", "packet", "ledger"],
        },
        StalePattern {
            phrase: "missing packet artifact",
            context: &[],
        },
        StalePattern {
            phrase: "missing artifacts",
            context: &["evidence", "status", "proof", "packet", "validator"],
        },
        StalePattern {
            phrase: "future durability",
            context: &["evidence", "artifact", "packet"],
        },
        StalePattern {
            phrase: "future sidecar",
            context: &["evidence", "artifact", "packet"],
        },
        StalePattern {
            phrase: "open",
            context: &["evidence status", "proof status", "ledger status"],
        },
        StalePattern {
            phrase: "partial",
            context: &["evidence status", "proof status", "ledger status"],
        },
        StalePattern {
            phrase: "pending",
            context: &["packet e", "packet f", "packet g", "packet h", "packet i"],
        },
    ]
}

fn allowed_context(normalized: &str) -> bool {
    if normalized.starts_with("---") {
        return true;
    }
    if normalized.contains("verification lane | planned hook | artifact target")
        || normalized.contains("planned module boundary")
        || normalized.contains("planned ids")
        || normalized.contains("planned verification assets")
        || normalized.contains("high level summary of frankensuite planned")
        || normalized.contains("open ambiguities")
        || normalized.contains("open memmap")
        || normalized.contains("partial silent recovery")
        || normalized.contains("partial reads")
        || normalized.contains("missing artifacts 0")
        || normalized.contains("missing artifacts=0")
        || normalized.contains("zero missing artifacts")
        || normalized.contains("no missing artifacts")
    {
        return true;
    }

    let residual_context = [
        "residual",
        "parity debt",
        "breadth",
        "unsupported semantics",
        "fail closed",
        "future algorithm",
        "future version",
        "future policy",
        "future reshape",
        "future rust native backend",
        "not yet tuned",
        "not yet modeled",
        "not yet codified",
        "not yet closure complete",
    ];
    residual_context
        .iter()
        .any(|allowed| normalized.contains(allowed))
}

fn context_for_line(normalized: &str) -> String {
    let contexts = [
        "evidence",
        "artifact",
        "validator",
        "packet",
        "ledger",
        "status",
        "proof",
        "verification",
    ];
    contexts
        .iter()
        .copied()
        .filter(|context| normalized.contains(context))
        .collect::<Vec<_>>()
        .join(",")
}

fn packet_id_for_line(line: &str) -> Option<String> {
    let (_, rest) = line.split_once("FNP-P2C-")?;
    let suffix = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if suffix.len() == 3 {
        Some(format!("FNP-P2C-{suffix}"))
    } else {
        None
    }
}

fn normalize(line: &str) -> String {
    line.to_ascii_lowercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn write_json_report(path: &Path, report: &StaleClaimReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed serializing stale-claim report: {err}"))?;
    fs::write(path, format!("{raw}\n"))
        .map_err(|err| format!("failed writing {}: {err}", path.display()))
}

fn write_markdown_report(path: &Path, report: &StaleClaimReport) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed creating {}: {err}", parent.display()))?;
    }
    fs::write(path, report.to_markdown())
        .map_err(|err| format!("failed writing {}: {err}", path.display()))
}

fn display_relative(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|")
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let suffix = now_unix_ms();
        let root = std::env::temp_dir().join(format!("fnp_phase2c_stale_claims_{name}_{suffix}"));
        fs::create_dir_all(&root).expect("create temp root");
        root
    }

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, content).expect("write test file");
    }

    fn write_ready_report(root: &Path, packet_id: &str) {
        write(
            &root
                .join("artifacts/phase2c")
                .join(packet_id)
                .join("packet_readiness_report.json"),
            &format!(
                r#"{{
  "packet_id": "{packet_id}",
  "status": "ready",
  "missing_artifacts": [],
  "missing_fields": [],
  "parse_errors": []
}}"#
            ),
        );
    }

    fn write_clean_ledger(root: &Path, packet_id: &str) {
        write(
            &root.join(CENTRAL_LEDGER),
            &format!(
                "| packet_id | current evidence refs | parity debt status |\n|---|---|---|\n| `{packet_id}` | final_evidence_pack.json and packet_readiness_report.json | ready: validator-clean packet evidence; residual breadth remains explicit parity debt |\n"
            ),
        );
    }

    #[test]
    fn stale_claim_scanner_rejects_planned_evidence_claim_for_ready_packet() {
        let root = temp_root("rejects_planned_evidence");
        let packet_id = "FNP-P2C-003";
        write_ready_report(&root, packet_id);
        write_clean_ledger(&root, packet_id);
        write(
            &root
                .join("artifacts/phase2c")
                .join(packet_id)
                .join("behavior_extraction_ledger.md"),
            "Unit/property evidence is not yet represented; Planned Verification Hooks remain pending for packet-I proof.\n",
        );

        let report = build_report(&root, &root.join("artifacts/phase2c")).expect("build report");

        assert_eq!(report.status, "stale");
        assert_eq!(report.diagnostics.len(), 3);
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.stale_phrase == "planned verification hook")
        );
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.stale_phrase == "not yet represented")
        );
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.stale_phrase == "pending")
        );
    }

    #[test]
    fn stale_claim_scanner_allows_residual_debt_and_table_headers() {
        let root = temp_root("allows_residual_debt");
        let packet_id = "FNP-P2C-001";
        write_ready_report(&root, packet_id);
        write_clean_ledger(&root, packet_id);
        write(
            &root
                .join("artifacts/phase2c")
                .join(packet_id)
                .join("behavior_extraction_ledger.md"),
            "\
| Verification lane | Planned hook | Artifact target |\n\
| `P2C001-U01` | Full no-copy reshape alias-preservation parity is not yet modeled end-to-end in Rust. | high |\n\
| `P2C001-U02` | residual breadth remains for a future reshape surface. | medium |\n",
        );

        let report = build_report(&root, &root.join("artifacts/phase2c")).expect("build report");

        assert_eq!(report.status, "fresh");
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn stale_claim_scanner_allows_green_missing_artifact_counts() {
        let root = temp_root("allows_green_counts");
        let packet_id = "FNP-P2C-006";
        write_ready_report(&root, packet_id);
        write_clean_ledger(&root, packet_id);
        write(
            &root
                .join("artifacts/phase2c")
                .join(packet_id)
                .join("implementation_plan.md"),
            "\
- Packet validator reports `status=ready`, `missing_artifacts=0`, `missing_fields=0`, and `parse_errors=0`.\n\
- packet validator reports ready with zero missing artifacts, missing fields, or parse errors.\n\
- packet validator reports `ready` with no missing artifacts, fields, or parse errors.\n",
        );

        let report = build_report(&root, &root.join("artifacts/phase2c")).expect("build report");

        assert_eq!(report.status, "fresh");
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn stale_claim_scanner_flags_central_ledger_stale_status_claim() {
        let root = temp_root("central_ledger_status");
        let packet_id = "FNP-P2C-009";
        write_ready_report(&root, packet_id);
        write(
            &root.join(CENTRAL_LEDGER),
            &format!(
                "| packet_id | current evidence refs | parity debt status |\n|---|---|---|\n| `{packet_id}` | evidence status anchor-only; missing packet artifacts | proof status partial |\n"
            ),
        );

        let report = build_report(&root, &root.join("artifacts/phase2c")).expect("build report");

        assert_eq!(report.status, "stale");
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.stale_phrase == "anchor only")
        );
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.stale_phrase == "missing packet artifact")
        );
    }

    #[test]
    fn stale_claim_scanner_writes_json_and_markdown_reports() {
        let root = temp_root("writes_reports");
        let packet_id = "FNP-P2C-004";
        write_ready_report(&root, packet_id);
        write_clean_ledger(&root, packet_id);

        let report = build_report(&root, &root.join("artifacts/phase2c")).expect("build report");
        let json_path = root.join("target/stale.json");
        let markdown_path = root.join("target/stale.md");
        write_json_report(&json_path, &report).expect("write json");
        write_markdown_report(&markdown_path, &report).expect("write markdown");

        assert!(json_path.is_file());
        assert!(markdown_path.is_file());
    }
}
