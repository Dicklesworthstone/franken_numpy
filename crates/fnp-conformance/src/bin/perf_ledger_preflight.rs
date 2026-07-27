use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const DEFAULT_LEDGER: &str = "docs/NEGATIVE_EVIDENCE.md";
const VERDICT_LEDGERS: [&str; 1] = [DEFAULT_LEDGER];
const ENFORCEMENT_DATE: &str = "2026-07-26";
const RESULT_CLASS_MARKER: &str = "**Campaign result class:**";
const NULL_CONTROL_MARKER: &str = "**A/A null control (same invocation):**";
const INCUMBENT_ARM_MARKER: &str = "**Legacy incumbent arm (same invocation):**";
const MAINTENANCE_SELF_SPEEDUP: &str = "maintenance-self-speedup";
const INCUMBENT_WIN: &str = "incumbent-win";

#[derive(Debug)]
enum Mode {
    Query { lever: String, surface: String },
    AuditStaged,
    AuditFile(PathBuf),
    SelfCheck,
}

#[derive(Debug)]
struct Config {
    ledger_path: PathBuf,
    mode: Mode,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let args = env::args().skip(1).collect::<Vec<_>>();
        let mut ledger_path = PathBuf::from(DEFAULT_LEDGER);
        let mut lever = None;
        let mut surface = None;
        let mut audit_staged = false;
        let mut audit_file = None;
        let mut self_check = false;
        let mut index = 0usize;

        while let Some(argument) = args.get(index) {
            match argument.as_str() {
                "--ledger" => {
                    index += 1;
                    ledger_path = PathBuf::from(required_value(&args, index, "--ledger")?);
                }
                "--lever" => {
                    index += 1;
                    lever = Some(required_value(&args, index, "--lever")?);
                }
                "--surface" => {
                    index += 1;
                    surface = Some(required_value(&args, index, "--surface")?);
                }
                "--audit-staged" => audit_staged = true,
                "--audit-file" => {
                    index += 1;
                    audit_file = Some(PathBuf::from(required_value(&args, index, "--audit-file")?));
                }
                "--self-check" => self_check = true,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument: {unknown}")),
            }
            index += 1;
        }

        let selected_modes = usize::from(audit_staged)
            + usize::from(audit_file.is_some())
            + usize::from(self_check)
            + usize::from(lever.is_some());
        if selected_modes != 1 {
            return Err(
                "select exactly one mode: --lever/--surface, --audit-staged, --audit-file, or --self-check"
                    .to_owned(),
            );
        }

        let mode = if self_check {
            if surface.is_some() {
                return Err("--surface is only valid with --lever".to_owned());
            }
            Mode::SelfCheck
        } else if audit_staged {
            if surface.is_some() {
                return Err("--surface is only valid with --lever".to_owned());
            }
            Mode::AuditStaged
        } else if let Some(path) = audit_file {
            if surface.is_some() {
                return Err("--surface is only valid with --lever".to_owned());
            }
            Mode::AuditFile(path)
        } else {
            Mode::Query {
                lever: lever.ok_or("--lever is required in query mode")?,
                surface: surface.ok_or("--surface is required with --lever")?,
            }
        };

        Ok(Self { ledger_path, mode })
    }
}

#[derive(Clone, Debug)]
struct LedgerEntry {
    heading: String,
    body: String,
}

#[derive(Debug)]
struct PositionedEntry {
    entry: LedgerEntry,
    first_line: usize,
    last_line: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChangedLineRange {
    first_line: usize,
    last_line: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Violation {
    heading: String,
    reason: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultClass {
    MaintenanceSelfSpeedup,
    IncumbentWin,
    Missing,
    Ambiguous,
    Invalid,
}

fn required_value(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn print_help() {
    println!(
        "\
perf_ledger_preflight - duplicate-candidate and ledger-integrity gate

USAGE:
  perf_ledger_preflight --lever <DESCRIPTION> --surface <TARGET>
  perf_ledger_preflight --audit-staged
  perf_ledger_preflight --audit-file <PATH>
  perf_ledger_preflight --self-check

OPTIONS:
  --ledger <PATH>       Negative-evidence ledger (default: {DEFAULT_LEDGER})
  --lever <TEXT>        Proposed optimization lever
  --surface <TEXT>      Target function, module, file, or benchmark surface
  --audit-staged        Validate added or modified staged ledger entries
  --audit-file <PATH>   Validate entries in a standalone fixture
  --self-check          Audit the enforced live ledger, then mutation-test its
                        own valid rows against all hardened predicates
  -h, --help            Show this help

Exit 0 means clear. Exit 2 means blocked. Query mode prints matching prior
entries and their retry predicates. Audit mode refuses a REJECT without either
a measured A/A null or a COUNTED_MECHANISM field, refuses a REJECT without a
retry predicate, and refuses a KEEP without an executing-ELF SHA-256 or an
explicit Campaign result class of maintenance-self-speedup or incumbent-win.
An incumbent-win additionally requires the actual NumPy name/version/artifact
SHA-256, a shared invocation ID, measured ratio, and a measured A/A null from
that invocation.
"
    );
}

fn split_entries(text: &str) -> Vec<LedgerEntry> {
    split_positioned_entries(text)
        .into_iter()
        .map(|positioned| positioned.entry)
        .collect()
}

fn split_positioned_entries(text: &str) -> Vec<PositionedEntry> {
    let mut entries = Vec::new();
    let mut heading = None::<String>;
    let mut body = String::new();
    let mut first_line = 0usize;
    let line_count = text.lines().count();

    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        if line.starts_with("## ") {
            if let Some(previous_heading) = heading.replace(line.to_owned()) {
                entries.push(PositionedEntry {
                    entry: LedgerEntry {
                        heading: previous_heading,
                        body: std::mem::take(&mut body),
                    },
                    first_line,
                    last_line: line_number - 1,
                });
            } else {
                heading = Some(line.to_owned());
            }
            first_line = line_number;
        } else if heading.is_some() {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(last_heading) = heading {
        entries.push(PositionedEntry {
            entry: LedgerEntry {
                heading: last_heading,
                body,
            },
            first_line,
            last_line: line_count,
        });
    }
    entries
}

fn entry_text(entry: &LedgerEntry) -> String {
    format!("{}\n{}", entry.heading, entry.body)
}

fn entry_date(entry: &LedgerEntry) -> Option<&str> {
    let date = entry
        .heading
        .strip_prefix("## ")?
        .split_whitespace()
        .next()?;
    let looks_like_date = date.len() == 10
        && date.bytes().filter(|byte| *byte == b'-').count() == 2
        && date
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'-');
    looks_like_date.then_some(date)
}

fn normalized_contains(haystack: &str, needle: &str) -> bool {
    let needle = needle.trim().to_ascii_lowercase();
    !needle.is_empty() && haystack.to_ascii_lowercase().contains(&needle)
}

fn retry_predicate(entry: &LedgerEntry) -> Option<String> {
    let lines = entry.body.lines().collect::<Vec<_>>();
    let start = lines.iter().position(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("retry condition")
            || lower.contains("retry predicate")
            || lower.contains("retry only")
            || lower.contains("reopen only")
            || lower.contains("revisit only")
    })?;

    let mut predicate = Vec::new();
    for line in lines.iter().skip(start).take(10) {
        if line.trim().is_empty() && !predicate.is_empty() {
            break;
        }
        predicate.push(line.trim());
    }
    Some(predicate.join(" "))
}

fn query_ledger(ledger: &str, lever: &str, surface: &str) -> bool {
    let entries = split_entries(ledger);
    let matches = entries
        .iter()
        .filter(|entry| {
            let text = entry_text(entry);
            normalized_contains(&text, surface) || normalized_contains(&text, lever)
        })
        .collect::<Vec<_>>();

    if matches.is_empty() {
        println!("CLEAR no prior ledger row matched lever={lever:?} surface={surface:?}");
        return true;
    }

    println!(
        "BLOCKED prior ledger evidence matched lever={lever:?} surface={surface:?} matches={}",
        matches.len()
    );
    for entry in matches {
        println!("match heading={}", entry.heading);
        println!(
            "retry_predicate={}",
            retry_predicate(entry).unwrap_or_else(|| "<missing>".to_owned())
        );
    }
    false
}

fn is_reject(entry: &LedgerEntry) -> bool {
    let heading = entry.heading.to_ascii_uppercase();
    if [
        "REJECT",
        "NO-SHIP",
        "NO SHIP",
        "HOLD",
        "BLOCKER",
        "DROPPED",
        "BENCH-BLOCKED",
    ]
    .iter()
    .any(|verdict| heading.contains(verdict))
    {
        return true;
    }
    entry.body.lines().any(|line| {
        let upper = line.trim().to_ascii_uppercase();
        upper.starts_with("VERDICT:")
            && ["REJECT", "NO-SHIP", "NO SHIP", "HOLD", "DROPPED"]
                .iter()
                .any(|verdict| upper.contains(verdict))
    })
}

fn is_keep(entry: &LedgerEntry) -> bool {
    let heading = entry.heading.to_ascii_uppercase();
    if heading.contains("KEEP") || heading.contains("WIN (SHIP") || heading.contains("WIN:") {
        return true;
    }
    entry.body.lines().any(|line| {
        let upper = line.trim().to_ascii_uppercase();
        upper.starts_with("VERDICT:") && (upper.contains("KEEP") || upper.contains("SHIP"))
    })
}

fn result_class(entry: &LedgerEntry) -> ResultClass {
    let mut values = entry
        .body
        .lines()
        .filter_map(|line| marker_value(line, RESULT_CLASS_MARKER));
    let Some(value) = values.next() else {
        return ResultClass::Missing;
    };
    if values.next().is_some() {
        return ResultClass::Ambiguous;
    }
    match value.trim_matches('`') {
        MAINTENANCE_SELF_SPEEDUP => ResultClass::MaintenanceSelfSpeedup,
        INCUMBENT_WIN => ResultClass::IncumbentWin,
        _ => ResultClass::Invalid,
    }
}

fn marker_value<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    line.trim().strip_prefix(marker).map(str::trim)
}

fn marker_entry_value<'a>(entry: &'a LedgerEntry, marker: &str) -> Option<&'a str> {
    entry
        .body
        .lines()
        .find_map(|line| marker_value(line, marker))
}

fn token_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let mut values = text
        .split_whitespace()
        .filter_map(|token| token.strip_prefix(key))
        .map(|value| value.trim_matches(|character| matches!(character, '`' | ',' | ';')));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn has_incumbent_win_contract(entry: &LedgerEntry) -> bool {
    let measured_null =
        marker_entry_value(entry, NULL_CONTROL_MARKER).is_some_and(line_has_decimal_measurement);
    let Some(incumbent) = marker_entry_value(entry, INCUMBENT_ARM_MARKER) else {
        return false;
    };
    let actual_numpy =
        token_value(incumbent, "name=").is_some_and(|name| name.eq_ignore_ascii_case("numpy"));
    let pinned_version = token_value(incumbent, "version=").is_some_and(|version| {
        !version.is_empty()
            && !version.eq_ignore_ascii_case("unavailable")
            && version.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'_' | b'-')
            })
    });
    let pinned_artifact = token_value(incumbent, "artifact_sha256=").is_some_and(|artifact| {
        is_lowercase_sha256(artifact)
            && executing_elf_sha256(entry)
                .is_none_or(|candidate| !candidate.eq_ignore_ascii_case(artifact))
    });
    let shared_invocation = token_value(incumbent, "invocation_id=").is_some_and(|identifier| {
        !identifier.is_empty()
            && identifier.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
            })
    });
    let measured_ratio = token_value(incumbent, "measured_ratio=")
        .and_then(|ratio| ratio.strip_suffix('x'))
        .and_then(|ratio| ratio.parse::<f64>().ok())
        .is_some_and(|ratio| ratio.is_finite() && ratio > 0.0);
    measured_null
        && actual_numpy
        && pinned_version
        && pinned_artifact
        && shared_invocation
        && measured_ratio
}

fn line_has_decimal_measurement(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let has_decimal = line.as_bytes().windows(3).any(|window| {
        window[0].is_ascii_digit() && window[1] == b'.' && window[2].is_ascii_digit()
    });
    let has_measurement_shape = lower.contains("null_base_aa")
        || lower.contains("ratio")
        || lower.contains("median=")
        || lower.contains("median:")
        || (lower.contains('[') && lower.contains(']'))
        || lower
            .split_once(':')
            .is_some_and(|(_, value)| value.bytes().any(|byte| byte.is_ascii_digit()));
    has_decimal && has_measurement_shape
}

fn has_measured_null(entry: &LedgerEntry) -> bool {
    entry.body.lines().any(|line| {
        let lower = line.to_ascii_lowercase();
        let positive_marker = lower.contains("null_base_aa")
            || lower.contains("a/a")
            || lower.contains("null control");
        let negative_marker = lower.contains("no a/a")
            || lower.contains("without a/a")
            || lower.contains("lacks a/a")
            || lower.contains("no null")
            || lower.contains("without null")
            || lower.contains("lacks null");
        positive_marker && !negative_marker && line_has_decimal_measurement(line)
    })
}

fn has_counted_mechanism(entry: &LedgerEntry) -> bool {
    entry.body.lines().any(|line| {
        let Some((label, evidence)) = line.split_once(':') else {
            return false;
        };
        label.trim().eq_ignore_ascii_case("COUNTED_MECHANISM")
            && evidence.bytes().any(|byte| byte.is_ascii_digit())
    })
}

fn first_sha256(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut run_start = 0usize;
    let mut run = 0usize;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte.is_ascii_hexdigit() {
            if run == 0 {
                run_start = index;
            }
            run += 1;
            if run == 64 {
                return Some(text[run_start..=index].to_owned());
            }
        } else {
            run = 0;
        }
    }
    None
}

fn executing_elf_sha256(entry: &LedgerEntry) -> Option<String> {
    entry.body.lines().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        let executing_marker = lower.contains("bench_elf_sha256")
            || lower.contains("executing elf sha")
            || lower.contains("executing-elf sha")
            || lower.contains("executing binary sha")
            || lower.contains("executable sha")
            || lower.contains("executable hash");
        (executing_marker && !lower.contains("unavailable"))
            .then(|| first_sha256(line))
            .flatten()
    })
}

fn has_elf_sha256(entry: &LedgerEntry) -> bool {
    executing_elf_sha256(entry).is_some()
}

fn audit_entries(entries: &[LedgerEntry]) -> Vec<Violation> {
    let mut violations = Vec::new();
    for entry in entries {
        if is_reject(entry) {
            if !has_measured_null(entry) && !has_counted_mechanism(entry) {
                violations.push(Violation {
                    heading: entry.heading.clone(),
                    reason: "REJECT lacks measured A/A null or COUNTED_MECHANISM",
                });
            }
            if retry_predicate(entry).is_none() {
                violations.push(Violation {
                    heading: entry.heading.clone(),
                    reason: "REJECT lacks concrete retry predicate",
                });
            }
        }
        if is_keep(entry) && !has_elf_sha256(entry) {
            violations.push(Violation {
                heading: entry.heading.clone(),
                reason: "KEEP lacks executing-ELF SHA-256",
            });
        }
        if is_keep(entry) {
            match result_class(entry) {
                ResultClass::MaintenanceSelfSpeedup => {}
                ResultClass::IncumbentWin => {
                    if !has_incumbent_win_contract(entry) {
                        violations.push(Violation {
                            heading: entry.heading.clone(),
                            reason: "incumbent-win lacks complete same-invocation NumPy evidence",
                        });
                    }
                }
                ResultClass::Missing => violations.push(Violation {
                    heading: entry.heading.clone(),
                    reason: "KEEP lacks exact campaign result class",
                }),
                ResultClass::Ambiguous => violations.push(Violation {
                    heading: entry.heading.clone(),
                    reason: "KEEP declares multiple campaign result classes",
                }),
                ResultClass::Invalid => violations.push(Violation {
                    heading: entry.heading.clone(),
                    reason: "KEEP declares an invalid campaign result class",
                }),
            }
        }
    }
    violations
}

fn staged_diff(ledger_path: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .args(["diff", "--cached", "--unified=0", "--"])
        .arg(ledger_path)
        .output()
        .map_err(|error| format!("failed to run staged ledger diff: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "staged ledger diff failed with status {}",
            output.status
        ));
    }

    String::from_utf8(output.stdout)
        .map_err(|error| format!("staged ledger diff was not UTF-8: {error}"))
}

fn staged_file_text(ledger_path: &Path) -> Result<String, String> {
    let index_path = format!(":{}", ledger_path.display());
    let output = Command::new("git")
        .args(["show", &index_path])
        .output()
        .map_err(|error| format!("failed to read staged ledger: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "reading staged ledger failed with status {}",
            output.status
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("staged ledger was not UTF-8: {error}"))
}

fn parse_new_line_range(hunk_header: &str) -> Result<ChangedLineRange, String> {
    let new_range = hunk_header
        .split_whitespace()
        .find(|token| token.starts_with('+'))
        .ok_or_else(|| format!("malformed staged ledger hunk: {hunk_header}"))?
        .trim_start_matches('+');
    let (first, count) = match new_range.split_once(',') {
        Some((first, count)) => (first, count),
        None => (new_range, "1"),
    };
    let first_line = first
        .parse::<usize>()
        .map_err(|_| format!("invalid staged ledger hunk start: {hunk_header}"))?;
    let line_count = count
        .parse::<usize>()
        .map_err(|_| format!("invalid staged ledger hunk count: {hunk_header}"))?;
    if line_count == 0 {
        return Err(
            "staged ledger removes an entire line range; the verdict ledger is append-only"
                .to_owned(),
        );
    }
    Ok(ChangedLineRange {
        first_line,
        last_line: first_line + line_count - 1,
    })
}

fn changed_line_ranges(diff: &str) -> Result<Vec<ChangedLineRange>, String> {
    diff.lines()
        .filter(|line| line.starts_with("@@ "))
        .map(parse_new_line_range)
        .collect()
}

fn touched_entries(staged_text: &str, diff: &str) -> Result<Vec<LedgerEntry>, String> {
    let ranges = changed_line_ranges(diff)?;
    let touched = split_positioned_entries(staged_text)
        .into_iter()
        .filter(|entry| {
            ranges.iter().any(|range| {
                range.first_line <= entry.last_line && range.last_line >= entry.first_line
            })
        })
        .map(|entry| entry.entry)
        .collect::<Vec<_>>();
    if !ranges.is_empty() && touched.is_empty() {
        return Err("staged ledger change did not reach a recognized ## verdict entry".to_owned());
    }
    Ok(touched)
}

fn staged_touched_entries(ledger_path: &Path) -> Result<Vec<LedgerEntry>, String> {
    let diff = staged_diff(ledger_path)?;
    if diff.trim().is_empty() {
        return Ok(Vec::new());
    }
    let staged_text = staged_file_text(ledger_path)?;
    touched_entries(&staged_text, &diff)
}

fn audit_entries_with_output(entries: &[LedgerEntry]) -> bool {
    if entries.is_empty() {
        println!("CLEAR no added or modified ledger entries");
        return true;
    }
    let violations = audit_entries(entries);
    if violations.is_empty() {
        println!(
            "CLEAR ledger additions and modifications satisfy null/mechanism, retry, ELF, and result-class gates"
        );
        return true;
    }
    println!("BLOCKED ledger_integrity violations={}", violations.len());
    for violation in violations {
        println!(
            "violation heading={} reason={}",
            violation.heading, violation.reason
        );
    }
    false
}

fn audit_text(text: &str) -> bool {
    if text.trim().is_empty() {
        println!("CLEAR empty audit fixture");
        return true;
    }
    let entries = split_entries(text);
    if entries.is_empty() {
        println!("BLOCKED audit fixture contains no recognized ## verdict entry");
        return false;
    }
    audit_entries_with_output(&entries)
}

fn strip_null_evidence(entry: &LedgerEntry) -> LedgerEntry {
    let body = entry
        .body
        .lines()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            !lower.contains("null_base_aa")
                && !lower.contains("a/a")
                && !lower.contains("null control")
                && !lower.contains("null arm")
                && !lower.contains("null floor")
                && !lower.contains("null ratio")
                && !lower.contains("null median")
        })
        .collect::<Vec<_>>()
        .join("\n");
    LedgerEntry {
        heading: entry.heading.clone(),
        body: format!(
            "{body}\nA/A null must follow the median-CI policy in campaign section 2.3 for 41 rounds.\nRetry only after a quiet pinned rerun.\n"
        ),
    }
}

fn with_campaign_class(entry: &LedgerEntry, class: Option<&str>) -> LedgerEntry {
    let mut body = entry
        .body
        .lines()
        .filter(|line| marker_value(line, RESULT_CLASS_MARKER).is_none())
        .collect::<Vec<_>>()
        .join("\n");
    if let Some(class) = class {
        body.push_str(&format!("\n{RESULT_CLASS_MARKER} {class}\n"));
    }
    LedgerEntry {
        heading: entry.heading.clone(),
        body,
    }
}

fn incumbent_claim_mutation(entry: &LedgerEntry, incumbent_fields: &str) -> LedgerEntry {
    let mut mutation = with_campaign_class(entry, Some(INCUMBENT_WIN));
    mutation.body.push_str(&format!(
        "\n{NULL_CONTROL_MARKER} baseline/null median ratio 1.001x, CI [0.99, 1.01].\n\
         {INCUMBENT_ARM_MARKER} {incumbent_fields}\n"
    ));
    mutation
}

fn mutation_has_violation(entry: &LedgerEntry, reason: &'static str) -> bool {
    audit_entries(std::slice::from_ref(entry))
        .iter()
        .any(|violation| violation.reason == reason)
}

fn self_check_ledger(ledger: &str) -> bool {
    let entries = split_entries(ledger);
    let enforced = entries
        .iter()
        .filter(|entry| entry_date(entry).is_some_and(|date| date >= ENFORCEMENT_DATE))
        .cloned()
        .collect::<Vec<_>>();
    let live_violations = audit_entries(&enforced);
    if !live_violations.is_empty() {
        println!(
            "BLOCKED live_ledger_integrity violations={}",
            live_violations.len()
        );
        for violation in live_violations {
            println!(
                "violation heading={} reason={}",
                violation.heading, violation.reason
            );
        }
        return false;
    }

    let Some(reject_seed) = enforced.iter().find(|entry| {
        is_reject(entry)
            && has_measured_null(entry)
            && !has_counted_mechanism(entry)
            && retry_predicate(entry).is_some()
    }) else {
        println!("BLOCKED self-check found no valid live REJECT seed");
        return false;
    };
    let policy_only_null = strip_null_evidence(reject_seed);
    let caught_policy_only_null = mutation_has_violation(
        &policy_only_null,
        "REJECT lacks measured A/A null or COUNTED_MECHANISM",
    );

    let mut uncounted_mechanism = strip_null_evidence(reject_seed);
    uncounted_mechanism
        .body
        .push_str("COUNTED_MECHANISM: allocation traffic unchanged.\n");
    let caught_uncounted_mechanism = mutation_has_violation(
        &uncounted_mechanism,
        "REJECT lacks measured A/A null or COUNTED_MECHANISM",
    );

    let Some(keep_seed) = enforced
        .iter()
        .find(|entry| is_keep(entry) && has_elf_sha256(entry))
    else {
        println!("BLOCKED self-check found no valid live KEEP seed");
        return false;
    };
    let Some(candidate_elf_sha256) = executing_elf_sha256(keep_seed) else {
        println!("BLOCKED self-check KEEP seed carried no executing-ELF SHA-256");
        return false;
    };
    let incumbent_fixture_sha256 = if candidate_elf_sha256
        == "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    {
        "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    } else {
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    };
    let mut unavailable_elf = keep_seed.clone();
    unavailable_elf.body = keep_seed
        .body
        .lines()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            !lower.contains("bench_elf_sha256")
                && !lower.contains("executing elf sha")
                && !lower.contains("executing-elf sha")
                && !lower.contains("executing binary sha")
                && !lower.contains("executable sha")
                && !lower.contains("executable hash")
        })
        .collect::<Vec<_>>()
        .join("\n");
    unavailable_elf.body.push_str(&format!(
        "\nExecuting ELF SHA-256: unavailable.\nsource_sha256={incumbent_fixture_sha256}\n"
    ));
    let caught_unavailable_elf =
        mutation_has_violation(&unavailable_elf, "KEEP lacks executing-ELF SHA-256");

    let unclassified_keep = with_campaign_class(keep_seed, None);
    let caught_unclassified_keep =
        mutation_has_violation(&unclassified_keep, "KEEP lacks exact campaign result class");

    let mut ambiguous_keep = keep_seed.clone();
    ambiguous_keep.body.push_str(&format!(
        "\n{RESULT_CLASS_MARKER} {MAINTENANCE_SELF_SPEEDUP}\n"
    ));
    let caught_ambiguous_keep = mutation_has_violation(
        &ambiguous_keep,
        "KEEP declares multiple campaign result classes",
    );

    let invalid_class_alias = with_campaign_class(keep_seed, Some("self-speedup"));
    let caught_invalid_class_alias = mutation_has_violation(
        &invalid_class_alias,
        "KEEP declares an invalid campaign result class",
    );

    let self_as_incumbent = incumbent_claim_mutation(
        keep_seed,
        &format!(
            "name=self-baseline version=2.4.6 artifact_sha256={incumbent_fixture_sha256} invocation_id=run-42 measured_ratio=1.2x"
        ),
    );
    let caught_self_as_incumbent = mutation_has_violation(
        &self_as_incumbent,
        "incumbent-win lacks complete same-invocation NumPy evidence",
    );

    let unpinned_incumbent = incumbent_claim_mutation(
        keep_seed,
        "name=NumPy version=2.4.6 artifact_sha256=unavailable invocation_id=run-42 measured_ratio=1.2x",
    );
    let caught_unpinned_incumbent = mutation_has_violation(
        &unpinned_incumbent,
        "incumbent-win lacks complete same-invocation NumPy evidence",
    );

    let substituted_candidate_elf = incumbent_claim_mutation(
        keep_seed,
        &format!(
            "name=NumPy version=2.4.6 artifact_sha256={candidate_elf_sha256} invocation_id=run-42 measured_ratio=1.2x"
        ),
    );
    let caught_substituted_candidate_elf = mutation_has_violation(
        &substituted_candidate_elf,
        "incumbent-win lacks complete same-invocation NumPy evidence",
    );

    let missing_invocation = incumbent_claim_mutation(
        keep_seed,
        &format!(
            "name=NumPy version=2.4.6 artifact_sha256={incumbent_fixture_sha256} measured_ratio=1.2x"
        ),
    );
    let caught_missing_invocation = mutation_has_violation(
        &missing_invocation,
        "incumbent-win lacks complete same-invocation NumPy evidence",
    );

    let missing_ratio = incumbent_claim_mutation(
        keep_seed,
        &format!(
            "name=NumPy version=2.4.6 artifact_sha256={incumbent_fixture_sha256} invocation_id=run-42"
        ),
    );
    let caught_missing_ratio = mutation_has_violation(
        &missing_ratio,
        "incumbent-win lacks complete same-invocation NumPy evidence",
    );

    let defects_caught = [
        caught_policy_only_null,
        caught_uncounted_mechanism,
        caught_unavailable_elf,
        caught_unclassified_keep,
        caught_ambiguous_keep,
        caught_invalid_class_alias,
        caught_self_as_incumbent,
        caught_unpinned_incumbent,
        caught_substituted_candidate_elf,
        caught_missing_invocation,
        caught_missing_ratio,
    ]
    .into_iter()
    .filter(|caught| *caught)
    .count();
    if defects_caught != 11 {
        println!(
            "BLOCKED self-check defects_caught={defects_caught}/11 policy_only_null={caught_policy_only_null} uncounted_mechanism={caught_uncounted_mechanism} unavailable_elf_with_source_hash={caught_unavailable_elf} unclassified_keep={caught_unclassified_keep} ambiguous_keep={caught_ambiguous_keep} invalid_class_alias={caught_invalid_class_alias} self_as_incumbent={caught_self_as_incumbent} unpinned_incumbent={caught_unpinned_incumbent} substituted_candidate_elf={caught_substituted_candidate_elf} missing_invocation={caught_missing_invocation} missing_ratio={caught_missing_ratio}"
        );
        return false;
    }

    println!(
        "SELF_CHECK PASS own_ledger_entries={} defects_caught=11/11 reject_seed={} keep_seed={}",
        enforced.len(),
        reject_seed.heading,
        keep_seed.heading
    );
    true
}

fn run() -> Result<ExitCode, String> {
    let config = Config::parse()?;
    let clear = match config.mode {
        Mode::Query { lever, surface } => {
            let ledger = fs::read_to_string(&config.ledger_path).map_err(|error| {
                format!(
                    "failed to read ledger {}: {error}",
                    config.ledger_path.display()
                )
            })?;
            query_ledger(&ledger, &lever, &surface)
        }
        Mode::AuditStaged => {
            if !VERDICT_LEDGERS
                .iter()
                .any(|ledger| config.ledger_path == Path::new(ledger))
            {
                return Err(format!(
                    "unsupported verdict ledger {}; known paths: {}",
                    config.ledger_path.display(),
                    VERDICT_LEDGERS.join(", ")
                ));
            }
            audit_entries_with_output(&staged_touched_entries(&config.ledger_path)?)
        }
        Mode::AuditFile(path) => {
            let text = fs::read_to_string(&path).map_err(|error| {
                format!("failed to read audit file {}: {error}", path.display())
            })?;
            audit_text(&text)
        }
        Mode::SelfCheck => {
            let ledger = fs::read_to_string(&config.ledger_path).map_err(|error| {
                format!(
                    "failed to read ledger {}: {error}",
                    config.ledger_path.display()
                )
            })?;
            self_check_ledger(&ledger)
        }
    };
    Ok(if clear {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("perf_ledger_preflight: {error}");
            ExitCode::from(64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_printing_match_predicate_is_blocking() {
        let ledger = "\
## 2026-07-01 - REJECT: Zipf cache

Surface: `Generator::zipf`.
A/A null: 1.001 [0.995, 1.006].
Retry only when the worker is pinned.
";
        assert!(!query_ledger(ledger, "cache powers", "Generator::zipf"));
    }

    #[test]
    fn reject_without_null_or_counted_mechanism_is_refused() {
        let entries = split_entries(
            "\
## 2026-07-01 - REJECT: undecidable

Effect ratio: 1.001.
Retry only after a focused harness exists.
",
        );
        assert_eq!(audit_entries(&entries).len(), 1);
    }

    #[test]
    fn null_policy_without_a_numeric_measurement_is_refused() {
        let entries = split_entries(
            "\
## 2026-07-01 - REJECT: policy text only

A/A null must follow the median-CI policy in campaign section 2.3 for 41 rounds.
Retry only after a focused harness exists.
",
        );
        assert_eq!(audit_entries(&entries).len(), 1);
    }

    #[test]
    fn counted_mechanism_rescues_reject_without_null() {
        let entries = split_entries(
            "\
## 2026-07-01 - REJECT: no work removed

COUNTED_MECHANISM: retired instructions unchanged at 10,000 per call.
Retry only if the algorithm removes an independently counted pass.
",
        );
        assert!(audit_entries(&entries).is_empty());
    }

    #[test]
    fn uncounted_mechanism_label_does_not_rescue_reject() {
        let entries = split_entries(
            "\
## 2026-07-01 - REJECT: prose mechanism

COUNTED_MECHANISM: allocation traffic unchanged.
Retry only if the algorithm removes an independently counted pass.
",
        );
        assert_eq!(audit_entries(&entries).len(), 1);
    }

    #[test]
    fn keep_requires_executing_elf_sha256() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

Effect ratio: 1.5.
**Campaign result class:** maintenance-self-speedup
",
        );
        assert_eq!(audit_entries(&entries).len(), 1);
    }

    #[test]
    fn keep_with_executing_elf_sha256_is_allowed() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** maintenance-self-speedup
",
        );
        assert!(audit_entries(&entries).is_empty());
    }

    #[test]
    fn unclassified_keep_is_refused() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
",
        );
        assert_eq!(
            audit_entries(&entries),
            [Violation {
                heading: "## 2026-07-01 - KEEP: candidate".to_owned(),
                reason: "KEEP lacks exact campaign result class",
            }]
        );
    }

    #[test]
    fn keep_cannot_declare_both_win_classes() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** maintenance-self-speedup
**Campaign result class:** incumbent-win
",
        );
        assert_eq!(
            audit_entries(&entries),
            [Violation {
                heading: "## 2026-07-01 - KEEP: candidate".to_owned(),
                reason: "KEEP declares multiple campaign result classes",
            }]
        );
    }

    #[test]
    fn invalid_campaign_class_is_refused() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** self-speedup
",
        );
        assert_eq!(
            audit_entries(&entries),
            [Violation {
                heading: "## 2026-07-01 - KEEP: candidate".to_owned(),
                reason: "KEEP declares an invalid campaign result class",
            }]
        );
    }

    #[test]
    fn campaign_class_value_is_case_sensitive() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** INCUMBENT-WIN
",
        );
        assert_eq!(
            audit_entries(&entries),
            [Violation {
                heading: "## 2026-07-01 - KEEP: candidate".to_owned(),
                reason: "KEEP declares an invalid campaign result class",
            }]
        );
    }

    #[test]
    fn incumbent_win_requires_explicit_same_invocation_fields() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

A/A null: 1.001 [0.995, 1.006].
NumPy was discussed and the harness should run it in the same invocation.
bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** incumbent-win
",
        );
        assert_eq!(
            audit_entries(&entries),
            [Violation {
                heading: "## 2026-07-01 - KEEP: candidate".to_owned(),
                reason: "incumbent-win lacks complete same-invocation NumPy evidence",
            }]
        );
    }

    #[test]
    fn incumbent_win_with_complete_contract_is_allowed() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** incumbent-win
**A/A null control (same invocation):** baseline/null median ratio 1.001x, CI [0.995, 1.006].
**Legacy incumbent arm (same invocation):** name=NumPy version=2.3.1 artifact_sha256=abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789 invocation_id=run-42 measured_ratio=3.250x
",
        );
        assert!(audit_entries(&entries).is_empty());
    }

    #[test]
    fn incumbent_artifact_cannot_reuse_candidate_elf_sha256() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

bench_elf_sha256=0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF.
**Campaign result class:** incumbent-win
**A/A null control (same invocation):** baseline/null median ratio 1.001x, CI [0.995, 1.006].
**Legacy incumbent arm (same invocation):** name=NumPy version=2.3.1 artifact_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef invocation_id=run-42 measured_ratio=3.250x
",
        );
        assert_eq!(
            audit_entries(&entries),
            [Violation {
                heading: "## 2026-07-01 - KEEP: candidate".to_owned(),
                reason: "incumbent-win lacks complete same-invocation NumPy evidence",
            }]
        );
    }

    #[test]
    fn negative_evidence_staged_heading_boundary_fails_then_passes() {
        assert_eq!(VERDICT_LEDGERS, ["docs/NEGATIVE_EVIDENCE.md"]);
        let diff = "\
diff --git a/docs/NEGATIVE_EVIDENCE.md b/docs/NEGATIVE_EVIDENCE.md
index 00000000..11111111 100644
--- a/docs/NEGATIVE_EVIDENCE.md
+++ b/docs/NEGATIVE_EVIDENCE.md
@@ -1,0 +2,3 @@
";
        let missing_class = "\
# Ledger
## 2026-07-27 - WIN (KEEP): candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
";
        let touched = touched_entries(missing_class, diff).expect("select staged entry");
        assert_eq!(touched.len(), 1);
        assert!(!audit_entries_with_output(&touched));

        let classified =
            format!("{missing_class}**Campaign result class:** maintenance-self-speedup\n");
        let touched = touched_entries(&classified, diff).expect("select staged entry");
        assert!(audit_entries_with_output(&touched));
    }

    #[test]
    fn staged_body_edit_audits_the_whole_entry() {
        let staged = "\
# Ledger
## 2026-07-27 - WIN (KEEP): candidate

bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** maintenance-self-speedup
";
        let diff = "\
@@ -4 +4 @@
";
        let touched = touched_entries(staged, diff).expect("select staged entry");
        assert_eq!(touched.len(), 1);
        assert!(audit_entries(&touched).is_empty());
    }

    #[test]
    fn elf_marker_cannot_borrow_an_unrelated_source_sha256() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

Executing ELF SHA-256: unavailable.
source_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
**Campaign result class:** maintenance-self-speedup
",
        );
        assert_eq!(audit_entries(&entries).len(), 1);
    }

    #[test]
    fn live_ledger_self_check_catches_all_hardened_defect_classes() {
        let ledger_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(DEFAULT_LEDGER);
        let ledger = fs::read_to_string(ledger_path).expect("live ledger should be readable");
        assert!(self_check_ledger(&ledger));
    }
}
