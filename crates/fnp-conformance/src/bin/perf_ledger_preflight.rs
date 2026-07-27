use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const DEFAULT_LEDGER: &str = "docs/NEGATIVE_EVIDENCE.md";

#[derive(Debug)]
enum Mode {
    Query { lever: String, surface: String },
    AuditStaged,
    AuditFile(PathBuf),
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
            + usize::from(lever.is_some());
        if selected_modes != 1 {
            return Err(
                "select exactly one mode: --lever/--surface, --audit-staged, or --audit-file"
                    .to_owned(),
            );
        }

        let mode = if audit_staged {
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

#[derive(Debug)]
struct LedgerEntry {
    heading: String,
    body: String,
}

#[derive(Debug, PartialEq, Eq)]
struct Violation {
    heading: String,
    reason: &'static str,
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

OPTIONS:
  --ledger <PATH>       Negative-evidence ledger (default: {DEFAULT_LEDGER})
  --lever <TEXT>        Proposed optimization lever
  --surface <TEXT>      Target function, module, file, or benchmark surface
  --audit-staged        Validate newly staged ledger entries
  --audit-file <PATH>   Validate entries in a standalone fixture
  -h, --help            Show this help

Exit 0 means clear. Exit 2 means blocked. Query mode prints matching prior
entries and their retry predicates. Audit mode refuses a REJECT without either
a measured A/A null or a COUNTED_MECHANISM field, refuses a REJECT without a
retry predicate, and refuses a KEEP without an executing-ELF SHA-256.
"
    );
}

fn split_entries(text: &str) -> Vec<LedgerEntry> {
    let mut entries = Vec::new();
    let mut heading = None::<String>;
    let mut body = String::new();

    for line in text.lines() {
        if line.starts_with("## ") {
            if let Some(previous_heading) = heading.replace(line.to_owned()) {
                entries.push(LedgerEntry {
                    heading: previous_heading,
                    body: std::mem::take(&mut body),
                });
            }
        } else if heading.is_some() {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(last_heading) = heading {
        entries.push(LedgerEntry {
            heading: last_heading,
            body,
        });
    }
    entries
}

fn entry_text(entry: &LedgerEntry) -> String {
    format!("{}\n{}", entry.heading, entry.body)
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
    if heading.contains("REJECT") || heading.contains("NO-SHIP") || heading.contains("NO SHIP") {
        return true;
    }
    entry.body.lines().any(|line| {
        let upper = line.trim().to_ascii_uppercase();
        upper.starts_with("VERDICT:")
            && (upper.contains("REJECT") || upper.contains("NO-SHIP") || upper.contains("NO SHIP"))
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

fn contains_sha256(text: &str) -> bool {
    let mut run = 0usize;
    for byte in text.bytes() {
        if byte.is_ascii_hexdigit() {
            run += 1;
            if run == 64 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

fn has_elf_sha256(entry: &LedgerEntry) -> bool {
    entry.body.lines().any(|line| {
        let lower = line.to_ascii_lowercase();
        let executing_marker = lower.contains("bench_elf_sha256")
            || lower.contains("executing elf sha")
            || lower.contains("executing-elf sha")
            || lower.contains("executing binary sha")
            || lower.contains("executable sha")
            || lower.contains("executable hash");
        executing_marker && !lower.contains("unavailable") && contains_sha256(line)
    })
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
    }
    violations
}

fn staged_added_text(ledger_path: &Path) -> Result<String, String> {
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

    let diff = String::from_utf8(output.stdout)
        .map_err(|error| format!("staged ledger diff was not UTF-8: {error}"))?;
    let mut additions = String::new();
    for line in diff.lines() {
        if line.starts_with("+++") {
            continue;
        }
        if let Some(added) = line.strip_prefix('+') {
            additions.push_str(added);
            additions.push('\n');
        }
    }
    Ok(additions)
}

fn audit_text(text: &str) -> bool {
    if text.trim().is_empty() {
        println!("CLEAR no new ledger entries");
        return true;
    }
    let entries = split_entries(text);
    if entries.is_empty() {
        println!("BLOCKED staged ledger additions contain no new ## entry");
        return false;
    }
    let violations = audit_entries(&entries);
    if violations.is_empty() {
        println!("CLEAR ledger additions satisfy null/mechanism, retry, and ELF gates");
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
        Mode::AuditStaged => audit_text(&staged_added_text(&config.ledger_path)?),
        Mode::AuditFile(path) => {
            let text = fs::read_to_string(&path).map_err(|error| {
                format!("failed to read audit file {}: {error}", path.display())
            })?;
            audit_text(&text)
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
",
        );
        assert!(audit_entries(&entries).is_empty());
    }

    #[test]
    fn elf_marker_cannot_borrow_an_unrelated_source_sha256() {
        let entries = split_entries(
            "\
## 2026-07-01 - KEEP: candidate

Executing ELF SHA-256: unavailable.
source_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.
",
        );
        assert_eq!(audit_entries(&entries).len(), 1);
    }
}
