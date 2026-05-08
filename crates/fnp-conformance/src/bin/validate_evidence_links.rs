#![forbid(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: &str = "evidence-link-report-v1";
const DEFAULT_REPORT_PATH: &str = "target/evidence_link_report.json";
const EXCEPTION_MARKER: &str = "evidence-validator-ignore";
const SOURCE_DOCUMENTS: &[&str] = &["README.md", "FEATURE_PARITY.md"];
const CONTRACTS_DIR: &str = "artifacts/contracts";
const CONTRACT_EXTENSIONS: &[&str] = &["json", "md", "yaml", "yml"];
const PATH_PREFIXES: &[&str] = &[
    ".beads/",
    "artifacts/",
    "crates/",
    "docs/",
    "legacy_numpy_code/",
    "scripts/",
];
const ROOT_FILES: &[&str] = &[
    "AGENTS.md",
    "Cargo.toml",
    "FEATURE_PARITY.md",
    "LICENSE",
    "PROPOSED_ARCHITECTURE.md",
    "README.md",
    "rust-toolchain.toml",
];
const PATH_EXTENSIONS: &[&str] = &[
    ".decode_proof.json",
    ".raptorq.json",
    ".scrub_report.json",
    ".jsonl",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".rs",
    ".sh",
    ".txt",
    ".toml",
    ".npy",
];

fn main() {
    if let Err(err) = run() {
        eprintln!("validate_evidence_links failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = Options::parse()?;
    let report = build_report(&options.repo_root);

    if let Some(parent) = options.report_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("create report dir {}: {err}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("serialize evidence link report: {err}"))?;
    fs::write(&options.report_path, format!("{json}\n"))
        .map_err(|err| format!("write {}: {err}", options.report_path.display()))?;

    println!(
        "wrote {}: status={} sources={} checked_references={} diagnostics={} exceptions={}",
        options.report_path.display(),
        report.status,
        report.source_count,
        report.checked_reference_count,
        report.diagnostics.len(),
        report.exceptions.len()
    );
    if options.print_diagnostics {
        for diagnostic in &report.diagnostics {
            println!(
                "diagnostic reason_code={} source={}:{} reference={} message={}",
                diagnostic.reason_code,
                diagnostic.source_path,
                diagnostic.line,
                diagnostic.reference,
                diagnostic.message
            );
        }
    }

    if options.fail_on_stale && report.has_failures() {
        return Err(format!(
            "{} stale evidence reference(s) found",
            report.failure_count()
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Options {
    repo_root: PathBuf,
    report_path: PathBuf,
    fail_on_stale: bool,
    print_diagnostics: bool,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut repo_root = default_repo_root();
        let mut report_path = repo_root.join(DEFAULT_REPORT_PATH);
        let mut fail_on_stale = false;
        let mut print_diagnostics = false;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--repo-root" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--repo-root requires a value".to_string())?;
                    repo_root = PathBuf::from(value);
                    if report_path.ends_with(DEFAULT_REPORT_PATH) {
                        report_path = repo_root.join(DEFAULT_REPORT_PATH);
                    }
                }
                "--report-path" | "--output-path" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a value"))?;
                    report_path = PathBuf::from(value);
                }
                "--fail-on-stale" => {
                    fail_on_stale = true;
                }
                "--print-diagnostics" => {
                    print_diagnostics = true;
                }
                "--help" | "-h" => {
                    return Err(usage());
                }
                other => return Err(format!("unknown argument '{other}'\n{}", usage())),
            }
        }

        Ok(Self {
            repo_root,
            report_path,
            fail_on_stale,
            print_diagnostics,
        })
    }
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin validate_evidence_links -- [--repo-root <path>] [--report-path <path>] [--fail-on-stale] [--print-diagnostics]".to_string()
}

fn default_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Debug, Serialize)]
struct EvidenceLinkReport {
    schema_version: &'static str,
    status: &'static str,
    source_count: usize,
    scanned_sources: Vec<String>,
    checked_reference_count: usize,
    diagnostics: Vec<EvidenceDiagnostic>,
    exceptions: Vec<PolicyException>,
}

impl EvidenceLinkReport {
    fn has_failures(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == "error")
    }

    fn failure_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == "error")
            .count()
    }
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceDiagnostic {
    severity: &'static str,
    reason_code: &'static str,
    source_path: String,
    line: usize,
    reference: String,
    message: String,
    remediation: String,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyException {
    source_path: String,
    line: usize,
    owner: String,
    reason: String,
    applies_to: String,
}

#[derive(Debug, Clone)]
struct SourceLine {
    source_path: String,
    line: usize,
    text: String,
}

fn build_report(repo_root: &Path) -> EvidenceLinkReport {
    let source_paths = source_paths(repo_root);
    let scanned_sources = source_paths
        .iter()
        .map(|path| display_relative(repo_root, path))
        .collect::<Vec<_>>();
    let mut checked_reference_count = 0;
    let mut diagnostics = Vec::new();
    let mut exceptions = Vec::new();

    for source_path in &source_paths {
        let relative_source = display_relative(repo_root, source_path);
        let content = match fs::read_to_string(source_path) {
            Ok(content) => content,
            Err(err) => {
                diagnostics.push(EvidenceDiagnostic {
                    severity: "error",
                    reason_code: "source_read_error",
                    source_path: relative_source,
                    line: 0,
                    reference: source_path.display().to_string(),
                    message: format!("failed to read source file: {err}"),
                    remediation: "restore the source file or remove it from the evidence scan"
                        .to_string(),
                });
                continue;
            }
        };

        for (index, line) in content.lines().enumerate() {
            let source_line = SourceLine {
                source_path: relative_source.clone(),
                line: index + 1,
                text: line.to_string(),
            };
            let mut line_diagnostics = Vec::new();

            match parse_policy_exception(&source_line) {
                Some(Ok(exception)) => {
                    exceptions.push(exception);
                    continue;
                }
                Some(Err(message)) => {
                    line_diagnostics.push(EvidenceDiagnostic {
                        severity: "error",
                        reason_code: "malformed_policy_exception",
                        source_path: source_line.source_path.clone(),
                        line: source_line.line,
                        reference: EXCEPTION_MARKER.to_string(),
                        message,
                        remediation:
                            "use evidence-validator-ignore(owner=<owner>, reason=<reason>)"
                                .to_string(),
                    });
                }
                None => {}
            }

            checked_reference_count +=
                validate_path_references(repo_root, &source_line, &mut line_diagnostics);
            checked_reference_count +=
                validate_binary_references(repo_root, &source_line, &mut line_diagnostics);
            checked_reference_count +=
                validate_test_references(repo_root, &source_line, &mut line_diagnostics);
            checked_reference_count +=
                validate_planned_evidence(&source_line, &mut line_diagnostics);

            diagnostics.extend(line_diagnostics);
        }
    }

    let status = if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == "error")
    {
        "fail"
    } else {
        "pass"
    };

    EvidenceLinkReport {
        schema_version: SCHEMA_VERSION,
        status,
        source_count: scanned_sources.len(),
        scanned_sources,
        checked_reference_count,
        diagnostics,
        exceptions,
    }
}

fn source_paths(repo_root: &Path) -> Vec<PathBuf> {
    let mut paths = SOURCE_DOCUMENTS
        .iter()
        .map(|path| repo_root.join(path))
        .collect::<Vec<_>>();

    let contracts_dir = repo_root.join(CONTRACTS_DIR);
    if let Ok(entries) = fs::read_dir(&contracts_dir) {
        let mut contract_paths = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| CONTRACT_EXTENSIONS.contains(&extension))
            })
            .collect::<Vec<_>>();
        contract_paths.sort();
        paths.extend(contract_paths);
    }

    paths
}

fn validate_path_references(
    repo_root: &Path,
    source_line: &SourceLine,
    diagnostics: &mut Vec<EvidenceDiagnostic>,
) -> usize {
    let candidates = extract_path_candidates(&source_line.text);
    let mut checked = 0;

    for candidate in candidates {
        checked += 1;
        let path = repo_root.join(&candidate);
        if !path.exists() {
            diagnostics.push(EvidenceDiagnostic {
                severity: "error",
                reason_code: "missing_file",
                source_path: source_line.source_path.clone(),
                line: source_line.line,
                reference: candidate.clone(),
                message: format!("referenced evidence path does not exist: {candidate}"),
                remediation:
                    "update the claim to an existing artifact or add an explicit policy exception"
                        .to_string(),
            });
        }
    }

    checked
}

fn validate_binary_references(
    repo_root: &Path,
    source_line: &SourceLine,
    diagnostics: &mut Vec<EvidenceDiagnostic>,
) -> usize {
    let bins = extract_cargo_bin_references(&source_line.text);
    let mut checked = 0;

    for bin in bins {
        checked += 1;
        if !binary_exists(repo_root, &bin) {
            diagnostics.push(EvidenceDiagnostic {
                severity: "error",
                reason_code: "stale_binary_name",
                source_path: source_line.source_path.clone(),
                line: source_line.line,
                reference: bin.clone(),
                message: format!(
                    "cargo --bin target is not present in any src/bin directory: {bin}"
                ),
                remediation: "rename the command to a live binary or add the missing binary"
                    .to_string(),
            });
        }
    }

    checked
}

fn validate_test_references(
    repo_root: &Path,
    source_line: &SourceLine,
    diagnostics: &mut Vec<EvidenceDiagnostic>,
) -> usize {
    let references = extract_cargo_test_references(&source_line.text);
    let mut checked = 0;

    for test_ref in references {
        checked += 1;
        if !test_ref_exists(repo_root, &test_ref) {
            diagnostics.push(EvidenceDiagnostic {
                severity: "error",
                reason_code: "stale_test_target",
                source_path: source_line.source_path.clone(),
                line: source_line.line,
                reference: test_ref.display_name(),
                message: "cargo test target or filter is not backed by live Rust test code"
                    .to_string(),
                remediation: "update the command to a live test target/filter or add coverage"
                    .to_string(),
            });
        }
    }

    checked
}

fn validate_planned_evidence(
    source_line: &SourceLine,
    diagnostics: &mut Vec<EvidenceDiagnostic>,
) -> usize {
    let lower = source_line.text.to_lowercase();
    let planned_claim = ["planned evidence", "planned artifact", "planned gate"]
        .iter()
        .any(|needle| lower.contains(needle));
    if !planned_claim {
        return 0;
    }

    diagnostics.push(EvidenceDiagnostic {
        severity: "error",
        reason_code: "planned_evidence_without_live_artifact",
        source_path: source_line.source_path.clone(),
        line: source_line.line,
        reference: source_line.text.trim().to_string(),
        message: "planned evidence language must not stand in for a live artifact".to_string(),
        remediation:
            "replace the claim with a live artifact reference or add an explicit owner/reason exception"
                .to_string(),
    });
    1
}

fn extract_path_candidates(line: &str) -> BTreeSet<String> {
    let mut tokens = Vec::new();
    tokens.extend(extract_backtick_spans(line));
    tokens.extend(extract_markdown_link_targets(line));
    tokens.extend(extract_quoted_strings(line));
    tokens.extend(line.split_whitespace().map(ToOwned::to_owned));

    tokens
        .into_iter()
        .filter_map(|token| normalize_path_candidate(&token))
        .collect()
}

fn extract_backtick_spans(line: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after_start = &rest[start + 1..];
        let Some(end) = after_start.find('`') else {
            break;
        };
        let span = &after_start[..end];
        if !span.is_empty() {
            spans.push(span.to_string());
        }
        rest = &after_start[end + 1..];
    }
    spans
}

fn extract_markdown_link_targets(line: &str) -> Vec<String> {
    let mut targets = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find("](") {
        let after_start = &rest[start + 2..];
        let Some(end) = after_start.find(')') else {
            break;
        };
        targets.push(after_start[..end].to_string());
        rest = &after_start[end + 1..];
    }
    targets
}

fn extract_quoted_strings(line: &str) -> Vec<String> {
    let mut strings = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escaped = false;

    for ch in line.chars() {
        if escaped {
            if in_string {
                current.push(ch);
            }
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            if in_string && !current.is_empty() {
                strings.push(current.clone());
                current.clear();
            }
            in_string = !in_string;
            continue;
        }
        if in_string {
            current.push(ch);
        }
    }

    strings
}

fn normalize_path_candidate(token: &str) -> Option<String> {
    let mut candidate = token
        .trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '"' | '\''
                    | '`'
                    | ','
                    | ';'
                    | ':'
                    | '.'
                    | ')'
                    | '('
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '<'
                    | '>'
            )
        })
        .trim_start_matches("./")
        .to_string();

    if let Some(anchor) = candidate.find('#') {
        candidate.truncate(anchor);
    }
    if let Some(stripped) = strip_line_suffix(&candidate) {
        candidate = stripped;
    }

    if candidate.is_empty()
        || candidate.contains('*')
        || candidate.contains('<')
        || candidate.contains('>')
        || candidate.contains(' ')
        || candidate.starts_with("http://")
        || candidate.starts_with("https://")
        || candidate.starts_with('/')
        || candidate.starts_with('$')
    {
        return None;
    }

    let looks_like_root_file = ROOT_FILES.contains(&candidate.as_str());
    let looks_like_path = PATH_PREFIXES
        .iter()
        .any(|prefix| candidate.starts_with(prefix))
        || PATH_EXTENSIONS
            .iter()
            .any(|extension| candidate.ends_with(extension) && candidate.contains('/'))
        || looks_like_root_file;

    if looks_like_path {
        Some(candidate)
    } else {
        None
    }
}

fn strip_line_suffix(candidate: &str) -> Option<String> {
    let (path, line) = candidate.rsplit_once(':')?;
    if !line.is_empty() && line.chars().all(|ch| ch.is_ascii_digit()) {
        Some(path.to_string())
    } else {
        None
    }
}

fn extract_cargo_bin_references(line: &str) -> BTreeSet<String> {
    let tokens = command_tokens(line);
    let mut bins = BTreeSet::new();
    for (index, token) in tokens.iter().enumerate() {
        if matches!(token.as_str(), "--bin")
            && let Some(bin) = tokens.get(index + 1)
            && is_concrete_reference(bin)
        {
            bins.insert(bin.clone());
        }
    }
    bins
}

#[derive(Debug, Clone)]
struct CargoTestReference {
    package: Option<String>,
    integration_target: Option<String>,
    filter: Option<String>,
}

impl CargoTestReference {
    fn display_name(&self) -> String {
        let package = self.package.as_deref().unwrap_or("<workspace>");
        match (&self.integration_target, &self.filter) {
            (Some(target), Some(filter)) => format!("{package} --test {target} {filter}"),
            (Some(target), None) => format!("{package} --test {target}"),
            (None, Some(filter)) => format!("{package} {filter}"),
            (None, None) => package.to_string(),
        }
    }
}

fn extract_cargo_test_references(line: &str) -> Vec<CargoTestReference> {
    let tokens = command_tokens(line);
    let Some(test_index) = find_cargo_test_index(&tokens) else {
        return Vec::new();
    };

    let mut package = None;
    let mut integration_target = None;
    let mut filter = None;
    let mut index = test_index + 1;
    while index < tokens.len() {
        let token = &tokens[index];
        if matches!(token.as_str(), "--") {
            break;
        }
        match token.as_str() {
            "-p" | "--package" => {
                if let Some(value) = tokens.get(index + 1) {
                    package = Some(value.clone());
                }
                index += 2;
            }
            "--test" => {
                if let Some(value) = tokens.get(index + 1)
                    && is_concrete_reference(value)
                {
                    integration_target = Some(value.clone());
                }
                index += 2;
            }
            "--workspace" | "--all-targets" | "--all-features" | "--lib" | "--bins" | "--tests"
            | "--benches" | "--examples" => {
                index += 1;
            }
            other if other.starts_with('-') || other.contains('=') => {
                index += 1;
            }
            other => {
                if filter.is_none() && is_concrete_reference(other) {
                    filter = Some(other.to_string());
                }
                index += 1;
            }
        }
    }

    if package.is_some() || integration_target.is_some() || filter.is_some() {
        vec![CargoTestReference {
            package,
            integration_target,
            filter,
        }]
    } else {
        Vec::new()
    }
}

fn find_cargo_test_index(tokens: &[String]) -> Option<usize> {
    tokens
        .windows(2)
        .position(|window| window[0] == "cargo" && window[1] == "test")
        .map(|position| position + 1)
}

fn command_tokens(line: &str) -> Vec<String> {
    line.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| {
                    matches!(
                        ch,
                        '`' | '"' | '\'' | ',' | ';' | '.' | ')' | '(' | '[' | ']' | '{' | '}'
                    )
                })
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn is_concrete_reference(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('<')
        && !value.contains('>')
        && !value.contains('*')
        && value != "_"
}

fn binary_exists(repo_root: &Path, bin: &str) -> bool {
    let crates_dir = repo_root.join("crates");
    if let Ok(entries) = fs::read_dir(crates_dir) {
        for entry in entries.filter_map(Result::ok) {
            let bin_path = entry.path().join("src/bin").join(format!("{bin}.rs"));
            if bin_path.exists() {
                return true;
            }
        }
    }
    repo_root.join("src/bin").join(format!("{bin}.rs")).exists()
}

fn test_ref_exists(repo_root: &Path, test_ref: &CargoTestReference) -> bool {
    let Some(package_dir) = package_dir(repo_root, test_ref.package.as_deref()) else {
        return false;
    };

    if let Some(target) = &test_ref.integration_target {
        let path = package_dir.join("tests").join(format!("{target}.rs"));
        if !path.exists() {
            return false;
        }
    }

    if let Some(filter) = &test_ref.filter {
        return rust_tree_contains_literal(&package_dir, filter);
    }

    true
}

fn package_dir(repo_root: &Path, package: Option<&str>) -> Option<PathBuf> {
    let package = package?;
    let direct = repo_root.join("crates").join(package);
    if direct.exists() {
        return Some(direct);
    }
    let normalized = package.replace('-', "_");
    let underscored = repo_root.join("crates").join(normalized);
    if underscored.exists() {
        return Some(underscored);
    }
    None
}

fn rust_tree_contains_literal(root: &Path, needle: &str) -> bool {
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "target")
            {
                continue;
            }
            let Ok(entries) = fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.filter_map(Result::ok) {
                stack.push(entry.path());
            }
            continue;
        }

        let is_rust_file = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension == "rs");
        if !is_rust_file {
            continue;
        }

        let stem_matches = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| stem == needle);
        if stem_matches {
            return true;
        }

        if fs::read_to_string(&path).is_ok_and(|content| content.contains(needle)) {
            return true;
        }
    }
    false
}

fn parse_policy_exception(source_line: &SourceLine) -> Option<Result<PolicyException, String>> {
    let marker_start = source_line.text.find(EXCEPTION_MARKER)?;
    let marker_text = &source_line.text[marker_start..];
    let Some(open) = marker_text.find('(') else {
        return Some(Err(format!(
            "{}:{} policy exception is missing '(owner=..., reason=...)'",
            source_line.source_path, source_line.line
        )));
    };
    let Some(close) = marker_text[open + 1..].find(')') else {
        return Some(Err(format!(
            "{}:{} policy exception is missing closing ')'",
            source_line.source_path, source_line.line
        )));
    };
    let fields = &marker_text[open + 1..open + 1 + close];
    let mut owner = None;
    let mut reason = None;
    for field in fields.split(',') {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        match key {
            "owner" if !value.is_empty() => owner = Some(value.to_string()),
            "reason" if !value.is_empty() => reason = Some(value.to_string()),
            _ => {}
        }
    }

    let Some(owner) = owner else {
        return Some(Err(format!(
            "{}:{} policy exception is missing owner",
            source_line.source_path, source_line.line
        )));
    };
    let Some(reason) = reason else {
        return Some(Err(format!(
            "{}:{} policy exception is missing reason",
            source_line.source_path, source_line.line
        )));
    };

    Some(Ok(PolicyException {
        source_path: source_line.source_path.clone(),
        line: source_line.line,
        owner,
        reason,
        applies_to: source_line.text.trim().to_string(),
    }))
}

fn display_relative(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{EvidenceLinkReport, build_report};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn validate_evidence_links_accepts_live_references() {
        let repo = fixture_repo("live");
        write_file(&repo, "README.md", "`artifacts/proofs/live.json`\n");
        write_file(
            &repo,
            "FEATURE_PARITY.md",
            "Run `cargo run -p fnp-conformance --bin live_gate`.\n",
        );
        write_file(&repo, "artifacts/proofs/live.json", "{}\n");
        write_file(
            &repo,
            "artifacts/contracts/contract.md",
            "No evidence refs.\n",
        );
        write_file(
            &repo,
            "crates/fnp-conformance/src/bin/live_gate.rs",
            "fn main() {}\n",
        );

        let report = build_report(&repo);

        assert_eq!(report.status, "pass", "{:?}", report.diagnostics);
        assert!(report.diagnostics.is_empty());
        assert!(report.checked_reference_count >= 2);
    }

    #[test]
    fn validate_evidence_links_reports_missing_files() {
        let repo = fixture_repo("missing-file");
        write_file(&repo, "README.md", "`artifacts/proofs/missing.json`\n");
        write_file(&repo, "FEATURE_PARITY.md", "No refs.\n");
        write_file(&repo, "artifacts/contracts/contract.md", "No refs.\n");

        let report = build_report(&repo);

        assert_eq!(report.status, "fail");
        assert_has_reason(&report, "missing_file");
    }

    #[test]
    fn validate_evidence_links_reports_missing_test_targets() {
        let repo = fixture_repo("missing-test");
        write_file(
            &repo,
            "README.md",
            "`cargo test -p fnp-conformance absent_filter`\n",
        );
        write_file(&repo, "FEATURE_PARITY.md", "No refs.\n");
        write_file(&repo, "artifacts/contracts/contract.md", "No refs.\n");
        write_file(
            &repo,
            "crates/fnp-conformance/src/lib.rs",
            "pub fn live() {}\n",
        );

        let report = build_report(&repo);

        assert_eq!(report.status, "fail");
        assert_has_reason(&report, "stale_test_target");
    }

    #[test]
    fn validate_evidence_links_honors_explicit_policy_exception() {
        let repo = fixture_repo("exception");
        write_file(
            &repo,
            "README.md",
            "`artifacts/proofs/external.json` <!-- evidence-validator-ignore(owner=docs, reason=external-ledger) -->\n",
        );
        write_file(&repo, "FEATURE_PARITY.md", "No refs.\n");
        write_file(&repo, "artifacts/contracts/contract.md", "No refs.\n");

        let report = build_report(&repo);

        assert_eq!(report.status, "pass");
        assert!(report.diagnostics.is_empty());
        assert_eq!(report.exceptions.len(), 1);
        assert_eq!(report.exceptions[0].owner, "docs");
        assert_eq!(report.exceptions[0].reason, "external-ledger");
    }

    fn assert_has_reason(report: &EvidenceLinkReport, reason: &str) {
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.reason_code == reason),
            "report should contain {reason}: {:?}",
            report.diagnostics
        );
    }

    fn fixture_repo(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("fnp_evidence_links_{name}_{nanos}"))
    }

    fn write_file(repo: &Path, relative: &str, content: &str) {
        let path = repo.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create fixture directory");
        }
        fs::write(path, content).expect("write fixture file");
    }
}
