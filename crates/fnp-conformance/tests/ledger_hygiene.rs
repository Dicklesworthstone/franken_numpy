//! Ledger-integrity gates for `docs/NEGATIVE_EVIDENCE.md`.
//!
//! Ledger integrity DECAYS. The 2026-07-25 fleet campaign audited eleven repos
//! and the spread was decided by exactly one thing: whether the repo had ever
//! institutionalized the check. Repos that audited once and then enforced it sit
//! at 1.7% void; repos that never did sit at 25-91%. The corrected hand audit
//! classified 109 actual rejected levers and found 71 (65.1%) VOID, including
//! 66 `VOID-NONULL`: an A/B ran, the row was rejected on a near-1.0 wall ratio,
//! and neither an A/A null control nor a counted mechanism was written down, so
//! the lever cannot be distinguished from the harness.
//!
//! These tests exist so that class cannot grow. A REJECT row dated on or after
//! [`ENFORCEMENT_DATE`] must record either an A/A null control or a counted
//! mechanism. Rows before that date are grandfathered and counted as debt by
//! `historical_void_nonull_debt_does_not_grow`.
//!
//! See `docs/LEDGER_RESURRECTION.md` for the taxonomy and the per-row audit.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// Rows dated on or after this must carry a null control or a counted mechanism.
/// Set to the date this gate landed; do not move it backwards to silence a row.
const ENFORCEMENT_DATE: &str = "2026-07-26";

/// Pre-[`ENFORCEMENT_DATE`] REJECT rows that record neither a null control nor a
/// counted mechanism, as measured by *these predicates* at the commit that
/// introduced this gate. The count may shrink (a row gains a null, or is re-run
/// and re-decided) but must never grow.
///
/// This is deliberately NOT the audit's canonical `VOID-NONULL` count of 66 in
/// `docs/LEDGER_RESURRECTION.md`. That figure additionally requires the claimed
/// ratio to be attributable to the candidate and excludes superseded/parse/survey
/// rows — judgements a CI gate should not be making. The gate measures only what
/// it can check mechanically, which is a strictly broader filter and yields 87
/// under the hardened predicates. The original loose predicates yielded 64
/// because a bare mention of A/A, cycles, faults, or allocations counted as
/// evidence; the 2026-07-27 model-integrity remediation closed that loophole and
/// exposed 23 additional historical rows. No row was added. Two numbers, two
/// definitions; do not reconcile them by loosening this one.
const HISTORICAL_VOID_NONULL_BUDGET: usize = 87;

fn ledger_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/NEGATIVE_EVIDENCE.md")
        .canonicalize()
        .expect("docs/NEGATIVE_EVIDENCE.md should exist")
}

struct Entry {
    heading: String,
    date: String,
    body: String,
    line: usize,
}

fn parse_entries() -> Vec<Entry> {
    let text = fs::read_to_string(ledger_path()).expect("ledger should be readable");
    let lines: Vec<&str> = text.lines().collect();
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("## "))
        .map(|(i, _)| i)
        .collect();

    let mut out = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(lines.len());
        let heading = lines[start][3..].trim().to_string();
        // Heading form: "YYYY-MM-DD - VERDICT: title". Rows that predate the
        // convention have no leading date and are treated as historical.
        let date = heading
            .split_whitespace()
            .next()
            .filter(|candidate_date| {
                candidate_date.len() == 10
                    && candidate_date.chars().filter(|c| *c == '-').count() == 2
            })
            .unwrap_or("")
            .to_string();
        out.push(Entry {
            heading,
            date,
            body: lines[start..end].join("\n"),
            line: start + 1,
        });
    }
    out
}

/// The verdict token is the text before the first colon in the heading.
fn is_reject(heading: &str) -> bool {
    let tail = match heading.split_once(" - ") {
        Some((_, rest)) => rest,
        None => heading,
    };
    let label = tail.split(':').next().unwrap_or(tail).to_uppercase();
    const NEGATIVE: [&str; 9] = [
        "REJECT",
        "NO-SHIP",
        "NO SHIP",
        "NOSHIP",
        "REVERT",
        "HOLD",
        "BENCH-BLOCKED",
        "DROPPED",
        "BLOCKER",
    ];
    const POSITIVE: [&str; 4] = ["WIN", "SHIP,", "KEEP", "LANDED"];
    // A "WIN (SHIP)" heading may legitimately contain the word REVERTED in prose.
    if POSITIVE.iter().any(|p| label.contains(p)) && !label.contains("REJECT") {
        return false;
    }
    NEGATIVE.iter().any(|n| label.contains(n))
}

/// A measured A/A null control. Policy prose or a statement that a null is
/// absent is not evidence.
fn records_null_control(body: &str) -> bool {
    body.lines().any(|line| {
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

/// A *counted* mechanism must use the explicit field and include a count.
/// Merely mentioning cycles, faults, or allocations is mechanism prose, not a
/// refutation.
fn records_counted_mechanism(body: &str) -> bool {
    body.lines().any(|line| {
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

fn records_executing_elf_sha256(body: &str) -> bool {
    body.lines().any(|line| {
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

fn is_keep(heading: &str) -> bool {
    let tail = match heading.split_once(" - ") {
        Some((_, rest)) => rest,
        None => heading,
    };
    let label = tail.split(':').next().unwrap_or(tail).to_uppercase();
    label.contains("KEEP") || label.contains("WIN") || label.contains("SHIP")
}

/// A rejection that no measurement could overturn - bit-exactness, observable
/// ordering, or another behavioral contract. These are not perf rejections.
fn is_behavioral_blocker(body: &str) -> bool {
    let lower = body.to_lowercase();
    [
        "bit-exact",
        "byte-exact",
        "not bit-reproducible",
        "not byte-reproducib",
        "non-deterministic",
        "last-ulp",
        "signed-zero",
        "order-unspecified",
        // "observable output" blockers: a value the caller can see that NumPy does
        // not canonicalize (e.g. QR's R row signs). No measurement overturns these.
        "observable output",
        "does not canonicalize",
        "parity-fatal",
        "parity minefield",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// A row that never obtained a timing at all: the bench, control, or worker died
/// first. Void, but not a *missing null* - the gate below asks for evidence the
/// row could not have produced.
fn is_unmeasured(body: &str) -> bool {
    let lower = body.to_lowercase();
    [
        "bench-blocked",
        "did not compile",
        "never ran",
        "never measured",
        "could not run",
        "could not measure",
        "could not obtain",
        "unmeasured",
        "workers unavailable",
        "queue_timeout",
        "bench-cost prohibitive",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// THE GATE. A REJECT row written from [`ENFORCEMENT_DATE`] onward must record an
/// A/A null control or a counted mechanism. Writing a bare near-1.0 rejection is
/// no longer merely discouraged - it fails CI.
#[test]
fn new_reject_rows_record_a_null_control_or_counted_mechanism() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| !e.date.is_empty() && e.date.as_str() >= ENFORCEMENT_DATE)
        .filter(|e| is_reject(&e.heading))
        .filter(|e| {
            !records_null_control(&e.body)
                && !records_counted_mechanism(&e.body)
                && !is_behavioral_blocker(&e.body)
                && !is_unmeasured(&e.body)
        })
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.is_empty(),
        "{} REJECT row(s) dated on/after {ENFORCEMENT_DATE} record neither an A/A null \
         control nor a counted mechanism, so the lever cannot be distinguished from the \
         harness:\n{}\n\nFix by adding one of:\n  \
         (a) an A/A null control measured in the SAME invocation as the A/B, or\n  \
         (b) a counted mechanism (instructions / cycles / syscalls / allocations / faults \
         unchanged), which refutes a lever regardless of any null, or\n  \
         (c) if the blocker is behavioral (bit-exactness, observable ordering) or the row \
         never obtained a timing, say so explicitly in the row.\n\
         See docs/LEDGER_RESURRECTION.md.",
        offenders.len(),
        offenders.join("\n")
    );
}

/// The historical debt is recorded, not silently tolerated. It may shrink; it may
/// not grow. A new bare rejection backdated before [`ENFORCEMENT_DATE`] to dodge
/// the gate above would trip this instead.
#[test]
fn historical_void_nonull_debt_does_not_grow() {
    let count = parse_entries()
        .into_iter()
        .filter(|e| e.date.is_empty() || e.date.as_str() < ENFORCEMENT_DATE)
        .filter(|e| is_reject(&e.heading))
        .filter(|e| {
            !records_null_control(&e.body)
                && !records_counted_mechanism(&e.body)
                && !is_behavioral_blocker(&e.body)
                && !is_unmeasured(&e.body)
        })
        .count();

    assert!(
        count <= HISTORICAL_VOID_NONULL_BUDGET,
        "historical VOID-NONULL rows grew from {HISTORICAL_VOID_NONULL_BUDGET} to {count}. \
         Either a pre-{ENFORCEMENT_DATE} row was added without a null control, or an \
         existing row lost one. New rejections must be dated and must carry evidence."
    );
}

/// A KEEP without the exact executing ELF is not reproducible evidence. The
/// marker and its full hash must share one line so an unavailable marker cannot
/// borrow an unrelated source hash elsewhere in the row.
#[test]
fn new_keep_rows_carry_an_executing_elf_sha256() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| !e.date.is_empty() && e.date.as_str() >= ENFORCEMENT_DATE)
        .filter(|e| is_keep(&e.heading))
        .filter(|e| !records_executing_elf_sha256(&e.body))
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.is_empty(),
        "{} KEEP row(s) dated on/after {ENFORCEMENT_DATE} lack a self-reported \
         executing-ELF SHA-256 on one line:\n{}\n\
         Add the benchmark output line `bench_elf_sha256=<64 hex>` from the \
         invocation that produced the verdict. A source or bench hash does not count.",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Every REJECT row must carry a retry predicate. A rejection without one is a
/// dead end nobody can reopen, which is how a void row becomes permanent.
#[test]
fn new_reject_rows_carry_a_retry_predicate() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| !e.date.is_empty() && e.date.as_str() >= ENFORCEMENT_DATE)
        .filter(|e| is_reject(&e.heading))
        .filter(|e| {
            let lower = e.body.to_lowercase();
            !lower.contains("retry") && !lower.contains("reopen") && !lower.contains("do not retry")
        })
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.is_empty(),
        "{} REJECT row(s) dated on/after {ENFORCEMENT_DATE} carry no retry predicate:\n{}\n\
         Every rejection needs a concrete condition under which it may be retried \
         (or an explicit 'do not retry' with the reason).",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Headings must be uniquely identifiable so the preflight can cite a row and so
/// two agents cannot silently write the same rejection twice.
#[test]
fn reject_headings_are_unique() {
    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();
    for entry in parse_entries()
        .into_iter()
        .filter(|e| is_reject(&e.heading))
    {
        if !seen.insert(entry.heading.clone()) {
            duplicates.push(format!(
                "  docs/NEGATIVE_EVIDENCE.md:{} — {}",
                entry.line, entry.heading
            ));
        }
    }
    assert!(
        duplicates.is_empty(),
        "duplicate REJECT headings — the same rejection was written twice:\n{}",
        duplicates.join("\n")
    );
}

#[test]
fn policy_prose_does_not_count_as_a_measured_null() {
    assert!(!records_null_control(
        "A/A null must follow campaign section 2.3 for 41 rounds."
    ));
}

#[test]
fn uncounted_mechanism_prose_does_not_count() {
    assert!(!records_counted_mechanism(
        "COUNTED_MECHANISM: allocation traffic unchanged."
    ));
}

#[test]
fn unavailable_elf_cannot_borrow_a_source_hash() {
    assert!(!records_executing_elf_sha256(
        "Executing ELF SHA-256: unavailable.\n\
         source_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ));
}
