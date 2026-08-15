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

/// Win rows dated on or after this must declare one exact campaign-result class
/// in their body.
const WIN_CLASS_ENFORCEMENT_DATE: &str = "2026-07-26";

/// Measured rows dated on or after this must name the worker their arms ran on.
///
/// The defect this exists for was measured elsewhere in the fleet on
/// 2026-08-15: the SAME cell run on two rch workers read 1.2693x and 0.0093x —
/// a 13.6x swing — with BOTH A/A nulls PASSING. A null controls
/// within-invocation noise; it cannot see between-worker differences in CPU
/// model, cache, memory bandwidth or contention, so it does not license
/// comparing one row against another. Worker identity is therefore part of a
/// row's meaning, not decoration: a row that does not name its worker cannot be
/// compared to any other row, and this repo already prints
/// `HOST_BASELINE host=<worker>` from inside the measuring process.
///
/// DELETION CONDITION: drop this gate if the ledger ever moves to a structured
/// row format that carries the worker as a required field of its own.
const WORKER_PROVENANCE_ENFORCEMENT_DATE: &str = "2026-08-15";

/// `incumbent-win` rows banked BEFORE [`WORKER_PROVENANCE_ENFORCEMENT_DATE`]
/// that name no worker. It was **44 of 44** when this gate landed; five were
/// then recovered from retained run logs under `.rch-bench-replay/`, matched to
/// their rows by `bench_elf_sha256` rather than by filename or date, so the
/// attribution is proven rather than inferred. **39 remain.**
///
/// The rest have no retained log and cannot be repaired by editing. Writing a
/// plausible worker in would be fabricated provenance — strictly worse than an
/// acknowledged gap — so the only ways this number falls are recovering a real
/// log or re-measuring the row. It may SHRINK; it is not a list to fill in.
///
/// It must never GROW. `new_measured_rows_name_their_worker` only sees rows
/// dated on/after the enforcement date, so without this a new unworkered
/// incumbent-win row backdated by one day would pass both gates. That is the
/// same backdating hole `historical_void_nonull_debt_does_not_grow` exists to
/// close, and it is closed the same way.
///
/// What it means for the remaining 39: their ratios stand as measured, but they
/// are worker-scoped and cannot be compared against each other or against any
/// newer row. The fleet measured a 13.6x swing for one cell across two workers
/// with both A/A nulls passing, so cross-row comparison without worker identity
/// is not conservative — it is unbounded.
const HISTORICAL_UNWORKERED_INCUMBENT_WIN_BUDGET: usize = 39;

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
    let tokens = || label.split(|character: char| !character.is_ascii_alphanumeric());
    let has_verdict_word = |word: &str| tokens().any(|token| token == word);
    // SHIP is the only one of the three verdict words that gets NEGATED in
    // practice, and the negation is not always adjacent: `NO-SHIP` / `NO SHIP`
    // were handled by substring, but `REJECT (measured, no production ship)`
    // slipped a word in between and was read as a positive SHIP verdict — so a
    // row whose verdict is literally REJECT was enforced as a KEEP, which is
    // what made this gate red on main (deadlock-audit-p126d). Suppress SHIP on
    // any NO token anywhere in the label instead.
    //
    // Deliberately NOT "is_keep is false whenever is_reject is true": seven
    // headings satisfy both, and six of them are genuinely dual-verdict
    // (`MAINTENANCE KEEP / INCUMBENT REJECT`, `SPLIT (SHIP c64 / REJECT c128)`,
    // `WIN (SHIP, ARCHITECTURAL) + embedded REJECT`, ...). Those MUST keep their
    // KEEP obligations, and they do — through their own KEEP/WIN tokens, which
    // this rule leaves alone. Over the whole ledger the change moves exactly one
    // row (1072 headings, is_keep 771 -> 770), the one beginning with REJECT;
    // see `is_keep_reads_negated_ship_verdicts_as_rejections` below.
    let ship_is_negated = has_verdict_word("NO");

    has_verdict_word("KEEP")
        || has_verdict_word("WIN")
        || (has_verdict_word("SHIP") && !ship_is_negated)
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

/// The worker a row's arms actually ran on. Accepts the harness's own
/// `HOST_BASELINE host=<worker>` line verbatim, or an explicit `worker=<name>`
/// field, so satisfying this is a paste rather than a rewrite. A placeholder
/// (`unavailable`, `unknown`, `n/a`) does not count — that is precisely the
/// value `HOSTNAME` yields over a non-interactive ssh, and accepting it would
/// certify the rows most likely to be missing their provenance.
fn records_measuring_worker(body: &str) -> bool {
    const PLACEHOLDERS: [&str; 5] = ["unavailable", "unknown", "n/a", "none", "tbd"];
    body.lines().any(|line| {
        let lower = line.to_ascii_lowercase();
        ["host=", "worker="]
            .iter()
            .filter_map(|marker| lower.split_once(*marker))
            .any(|(_, rest)| {
                let name = rest
                    .split(|c: char| c.is_whitespace() || c == ',' || c == ')')
                    .next()
                    .unwrap_or("")
                    .trim_matches(|c: char| c == '`' || c == '*' || c == '"');
                !name.is_empty() && !PLACEHOLDERS.contains(&name)
            })
    })
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

/// THE CLASSIFIER ITSELF. `is_keep` decides which rows the two KEEP gates below
/// apply to, so a misreading of a heading silently redirects the whole gate — it
/// can demand KEEP evidence from a rejection, or (worse, in the other direction)
/// let a real win skip it. Pin both directions on the exact wordings that have
/// occurred in this ledger.
///
/// The negated-SHIP case is not hypothetical: `REJECT (measured, no production
/// ship)` was read as a positive SHIP verdict and turned both KEEP gates red on
/// main (deadlock-audit-p126d), because the old guard tested for the literal
/// substrings `NO-SHIP` / `NO SHIP` and this negation had a word in between.
#[test]
fn is_keep_reads_negated_ship_verdicts_as_rejections() {
    // Negated SHIP, adjacent or not: never a KEEP.
    for heading in [
        "2026-08-04 - REJECT (measured, no production ship): RandomState Zipf cache retry",
        "2026-07-04 - NO-SHIP (REVERTED, ~PARITY): np.unique(2-D 'U'/'S', axis=0)",
        "2026-06-25 - NO SHIP: f64 `np.unique` duplicate-heavy bit HashSet",
    ] {
        assert!(
            !is_keep(heading),
            "a negated SHIP verdict must not be classified as a KEEP: {heading}"
        );
    }

    // Genuine keeps, including the dual-verdict forms this ledger really uses.
    // These carry a KEEP or WIN token of their own, which is exactly why the fix
    // above targets the SHIP token alone rather than exempting every heading
    // that also mentions a rejection.
    for heading in [
        "2026-07-10 - WIN (SHIP): f64 transcendental unary zero-copy fused-defer path",
        "2026-07-14 - SPLIT (SHIP c64 / REJECT c128): complex np.select arms",
        "2026-07-22 - WIN (SHIP, ARCHITECTURAL) + embedded REJECT: BENCH-BLOCKED is two costs",
        "2026-07-27 - MAINTENANCE KEEP / INCUMBENT REJECT: selected-bool loadtxt(usecols)",
        "2026-07-12 - STALE-REJECT REOPENED (WIN): integer ARRAY-q percentile/quantile",
    ] {
        assert!(
            is_keep(heading),
            "a heading carrying its own KEEP/WIN verdict must stay a KEEP: {heading}"
        );
    }
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

// The result schema is owned by `crates/fnp-conformance/src/bin/
// perf_ledger_preflight.rs`, which enforces it at pre-commit. CI intentionally
// duplicates the complete contract as a backstop; neither path may accept a row
// the other rejects.
const RESULT_CLASS_MARKER: &str = "**Campaign result class:**";
const NULL_CONTROL_MARKER: &str = "**A/A null control (same invocation):**";
const INCUMBENT_ARM_MARKER: &str = "**Legacy incumbent arm (same invocation):**";
const INCUMBENT_ISOLATION_MARKER: &str = "**Incumbent isolation proof:**";
const SHARED_COMPONENT_DISCLOSURE_MARKER: &str = "**Shared timed component disclosure:**";
const MAINTENANCE_SELF_SPEEDUP: &str = "maintenance-self-speedup";
const INCUMBENT_WIN: &str = "incumbent-win";

/// A disclosed shared component may not dominate the candidate's own job time.
/// Past that share the headline is mostly the incumbent's code and the row is
/// not measuring us.
const MAX_DISCLOSED_SHARED_SHARE_PCT: f64 = 50.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResultClass {
    MaintenanceSelfSpeedup,
    IncumbentWin,
    Missing,
    Ambiguous,
    Invalid,
}

fn marker_value<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    line.trim().strip_prefix(marker).map(str::trim)
}

fn marker_entry_value<'a>(body: &'a str, marker: &str) -> Option<&'a str> {
    body.lines().find_map(|line| marker_value(line, marker))
}

/// A token such as `name=numpy` from within a marker's value.
fn token_value<'a>(value: &'a str, token: &str) -> Option<&'a str> {
    let mut values = value
        .split_whitespace()
        .filter_map(|field| field.strip_prefix(token))
        .map(|field| field.trim_matches(|character| matches!(character, '`' | ',' | ';')));
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn result_class(body: &str) -> ResultClass {
    let mut values = body
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

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn executing_elf_sha256(body: &str) -> Option<String> {
    body.lines().find_map(|line| {
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

/// `shared_timed_component=none` is the default and needs nothing further.
///
/// A row may instead DISCLOSE a shared timed component, but only on terms that
/// are strictly harder to satisfy than claiming none, so disclosure can never
/// be the cheaper path:
///
/// * every named component must belong to the incumbent (`numpy.*`). Shared
///   incumbent code runs identically in both arms, so it can only dilute the
///   candidate's ratio. A component naming `fnp.*` is refused outright —
///   timing our own former path is `maintenance-self-speedup`, never an
///   incumbent win, and this must not become a hole in that wall.
/// * the row must declare `direction=conservative_for_candidate`, stating that
///   claim direction explicitly rather than leaving a reader to infer it.
/// * the row must quantify the shared share of candidate job time, with every
///   quoted share under [`MAX_DISCLOSED_SHARED_SHARE_PCT`].
/// * the disclosure's `components=` list must match the isolation marker's
///   `shared_timed_component=` list exactly, so the two lines cannot disagree.
fn disclosed_shared_component_is_admissible(body: &str, declared: &str) -> bool {
    let Some(disclosure) = marker_entry_value(body, SHARED_COMPONENT_DISCLOSURE_MARKER) else {
        return false;
    };
    if token_value(disclosure, "components=") != Some(declared) {
        return false;
    }
    let every_component_is_the_incumbents = declared
        .split(',')
        .map(str::trim)
        .all(|component| component.starts_with("numpy.") && component.len() > 6);
    let direction_declared =
        token_value(disclosure, "direction=") == Some("conservative_for_candidate");
    let quantified = token_value(disclosure, "share_of_candidate_pct=").is_some_and(|shares| {
        let mut shares = shares.split(',').map(str::trim).peekable();
        shares.peek().is_some()
            && shares.all(|share| {
                share.parse::<f64>().is_ok_and(|pct| {
                    pct.is_finite() && pct > 0.0 && pct < MAX_DISCLOSED_SHARED_SHARE_PCT
                })
            })
    });
    every_component_is_the_incumbents && direction_declared && quantified
}

/// An `incumbent-win` must carry the full same-invocation contract. In
/// particular, the incumbent artifact cannot be the candidate benchmark ELF:
/// that substitution is the provenance defect this gate was hardened to catch.
fn has_incumbent_win_contract(body: &str) -> bool {
    let measured_null =
        marker_entry_value(body, NULL_CONTROL_MARKER).is_some_and(line_has_decimal_measurement);
    let Some(incumbent) = marker_entry_value(body, INCUMBENT_ARM_MARKER) else {
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
            && executing_elf_sha256(body)
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
    let independent_end_to_end =
        marker_entry_value(body, INCUMBENT_ISOLATION_MARKER).is_some_and(|isolation| {
            let candidate = token_value(isolation, "candidate=");
            let incumbent = token_value(isolation, "incumbent=");
            let isolated = match token_value(isolation, "shared_timed_component=") {
                Some("none") => true,
                Some(declared) if !declared.is_empty() => {
                    disclosed_shared_component_is_admissible(body, declared)
                }
                _ => false,
            };
            candidate.is_some_and(|name| name.starts_with("fnp.") && name.len() > 4)
                && incumbent.is_some_and(|name| name.starts_with("numpy.") && name.len() > 6)
                && candidate != incumbent
                && isolated
        });
    measured_null
        && actual_numpy
        && pinned_version
        && pinned_artifact
        && shared_invocation
        && measured_ratio
        && independent_end_to_end
}

/// THE RESULT-CLASS GATE. Beating our own former path is maintenance, not
/// campaign output. Every KEEP must carry exactly one canonical class marker.
#[test]
fn new_keep_rows_declare_one_campaign_result_class() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| !e.date.is_empty() && e.date.as_str() >= WIN_CLASS_ENFORCEMENT_DATE)
        .filter(|e| is_keep(&e.heading))
        .filter(|e| {
            !matches!(
                result_class(&e.body),
                ResultClass::MaintenanceSelfSpeedup | ResultClass::IncumbentWin
            )
        })
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.is_empty(),
        "{} KEEP row(s) dated on/after {WIN_CLASS_ENFORCEMENT_DATE} lack exactly one \
         canonical `{RESULT_CLASS_MARKER}` marker:\n{}\n\n\
         Use `{MAINTENANCE_SELF_SPEEDUP}` for our own before/after measurement. It is \
         maintenance and must never be quoted as a competitive claim.\n\
         Use `{INCUMBENT_WIN}` only for an actual NumPy arm timed side-by-side in \
         the same invocation under the complete provenance contract.",
        offenders.len(),
        offenders.join("\n")
    );
}

/// An `incumbent-win` is campaign output only when the complete incumbent
/// contract is present and internally consistent.
#[test]
fn incumbent_win_rows_carry_the_complete_same_invocation_contract() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| !e.date.is_empty() && e.date.as_str() >= WIN_CLASS_ENFORCEMENT_DATE)
        .filter(|e| result_class(&e.body) == ResultClass::IncumbentWin)
        .filter(|e| !has_incumbent_win_contract(&e.body))
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.is_empty(),
        "{} `{INCUMBENT_WIN}` row(s) lack the complete contract:\n{}\n\n\
         Required in one invocation: exact A/A marker with a numeric measurement; \
         `name=NumPy`; pinned `version`; lowercase incumbent `artifact_sha256` distinct \
         from the candidate executing ELF; shared `invocation_id`; and numeric \
         `measured_ratio=<value>x`. The isolation marker must name distinct `fnp.*` \
         and `numpy.*` public arms and declare `shared_timed_component=none`, or else \
         disclose it: every component `numpy.*`, plus a \
         `{SHARED_COMPONENT_DISCLOSURE_MARKER}` line whose `components=` matches the \
         isolation marker, `direction=conservative_for_candidate`, and \
         `share_of_candidate_pct=` values all below \
         {MAX_DISCLOSED_SHARED_SHARE_PCT}.",
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

#[test]
fn keep_classifier_requires_a_positive_verdict_word() {
    for heading in [
        "2026-07-31 - REJECT (NO-SHIP): x",
        "2026-07-31 - REJECT (NO SHIP): x",
        "2026-07-31 - REJECT (NOSHIP): x",
        "2026-07-31 - CODE-ONLY: delegate SHIPPED but loses",
    ] {
        assert!(!is_keep(heading), "{heading}");
    }

    for heading in [
        "2026-07-31 - WIN (SHIP): x",
        "2026-07-31 - WIN (KEEP, INCUMBENT-WIN): x",
        "2026-07-31 - KEEP: x",
    ] {
        assert!(is_keep(heading), "{heading}");
    }
}

#[test]
fn win_class_marker_must_be_exact() {
    // Only the two canonical values count. A near-miss is not a declaration —
    // it is an unclassified win wearing a label, which is worse than none
    // because it reads as deliberate.
    assert_eq!(
        result_class("**Campaign result class:** incumbent-win"),
        ResultClass::IncumbentWin
    );
    assert_eq!(
        result_class("**Campaign result class:** maintenance-self-speedup"),
        ResultClass::MaintenanceSelfSpeedup
    );
    assert_eq!(
        result_class("**Campaign result class:** self-speedup"),
        ResultClass::Invalid
    );
    assert_eq!(
        result_class("**Campaign result class:** vs-incumbent"),
        ResultClass::Invalid
    );
    assert_eq!(
        result_class("**Campaign result class:** INCUMBENT-WIN"),
        ResultClass::Invalid
    );
    assert_eq!(
        result_class(
            "**Campaign result class:** incumbent-win\n\
             **Campaign result class:** maintenance-self-speedup"
        ),
        ResultClass::Ambiguous
    );
    assert_eq!(
        result_class("Campaign result class: incumbent-win"),
        ResultClass::Missing
    );
}

#[test]
fn incumbent_arm_tokens_must_share_the_marker_line() {
    // A continuation line is not part of the marker value. This exact mistake
    // blocked the first incumbent-win row at pre-commit.
    let body = "**A/A null control (same invocation):** ratio 1.001x.\n\
                bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n\
                **Legacy incumbent arm (same invocation):** name=NumPy version=2.4.6\n\
                artifact_sha256=43760732fe0ec60ac5e2d4d020b253ea720cf0d3996a362204c4d94934ebaabd invocation_id=run-42 measured_ratio=19.9x\n\
                **Incumbent isolation proof:** candidate=fnp.isin incumbent=numpy.isin shared_timed_component=none";
    assert!(!has_incumbent_win_contract(body));

    let body = "**A/A null control (same invocation):** ratio 1.001x.\n\
                bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n\
                **Legacy incumbent arm (same invocation):** name=NumPy version=2.4.6 artifact_sha256=43760732fe0ec60ac5e2d4d020b253ea720cf0d3996a362204c4d94934ebaabd invocation_id=run-42 measured_ratio=19.9x\n\
                **Incumbent isolation proof:** candidate=fnp.isin incumbent=numpy.isin shared_timed_component=none";
    assert!(has_incumbent_win_contract(body));
}

#[test]
fn candidate_elf_cannot_masquerade_as_the_incumbent_artifact() {
    let body = "**A/A null control (same invocation):** ratio 1.001x.\n\
                bench_elf_sha256=0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF\n\
                **Legacy incumbent arm (same invocation):** name=NumPy version=2.4.6 \
                artifact_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
                invocation_id=run-42 measured_ratio=19.9x\n\
                **Incumbent isolation proof:** candidate=fnp.isin incumbent=numpy.isin shared_timed_component=none";
    assert!(!has_incumbent_win_contract(body));
}

#[test]
fn incumbent_win_rejects_shared_timed_components() {
    let body = "**A/A null control (same invocation):** ratio 1.001x.\n\
                bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n\
                **Legacy incumbent arm (same invocation):** name=NumPy version=2.4.6 \
                artifact_sha256=43760732fe0ec60ac5e2d4d020b253ea720cf0d3996a362204c4d94934ebaabd \
                invocation_id=run-42 measured_ratio=19.9x\n\
                **Incumbent isolation proof:** candidate=fnp.isin incumbent=numpy.isin \
                shared_timed_component=numpy.empty";
    assert!(!has_incumbent_win_contract(body));
}

/// A shared component is admissible only when it is DISCLOSED on terms harder
/// than claiming none: incumbent-owned components, a declared claim direction,
/// and a quantified minority share that matches the isolation marker.
#[test]
fn incumbent_win_accepts_a_fully_disclosed_shared_component() {
    let prefix = "**A/A null control (same invocation):** ratio 1.001x.\n\
                  bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n\
                  **Legacy incumbent arm (same invocation):** name=NumPy version=2.4.6 \
                  artifact_sha256=43760732fe0ec60ac5e2d4d020b253ea720cf0d3996a362204c4d94934ebaabd \
                  invocation_id=run-42 measured_ratio=19.9x\n\
                  **Incumbent isolation proof:** candidate=fnp.workload.report \
                  incumbent=numpy.workload.report \
                  shared_timed_component=numpy.sum_int64_axis1,numpy.sum_int64_flat\n";

    // Complete disclosure: admissible.
    assert!(has_incumbent_win_contract(&format!(
        "{prefix}**Shared timed component disclosure:** \
         components=numpy.sum_int64_axis1,numpy.sum_int64_flat \
         direction=conservative_for_candidate share_of_candidate_pct=7.427,9.773,6.479"
    )));

    // Declared but never disclosed: the pre-hardening hole.
    assert!(!has_incumbent_win_contract(prefix));

    // Direction unstated: a reader would have to infer the claim direction.
    assert!(!has_incumbent_win_contract(&format!(
        "{prefix}**Shared timed component disclosure:** \
         components=numpy.sum_int64_axis1,numpy.sum_int64_flat \
         share_of_candidate_pct=7.427,9.773,6.479"
    )));

    // Unquantified share: disclosure without a number proves nothing.
    assert!(!has_incumbent_win_contract(&format!(
        "{prefix}**Shared timed component disclosure:** \
         components=numpy.sum_int64_axis1,numpy.sum_int64_flat \
         direction=conservative_for_candidate"
    )));

    // Disclosure list disagreeing with the isolation marker.
    assert!(!has_incumbent_win_contract(&format!(
        "{prefix}**Shared timed component disclosure:** components=numpy.sum_int64_axis1 \
         direction=conservative_for_candidate share_of_candidate_pct=7.427"
    )));

    // A majority share means the headline is mostly the incumbent's own code.
    assert!(!has_incumbent_win_contract(&format!(
        "{prefix}**Shared timed component disclosure:** \
         components=numpy.sum_int64_axis1,numpy.sum_int64_flat \
         direction=conservative_for_candidate share_of_candidate_pct=7.427,61.2"
    )));
}

/// Sharing OUR OWN code is `maintenance-self-speedup`. Disclosure must not
/// become a way to relabel that as an incumbent win.
#[test]
fn disclosure_cannot_launder_a_shared_fnp_component() {
    let body = "**A/A null control (same invocation):** ratio 1.001x.\n\
                bench_elf_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n\
                **Legacy incumbent arm (same invocation):** name=NumPy version=2.4.6 \
                artifact_sha256=43760732fe0ec60ac5e2d4d020b253ea720cf0d3996a362204c4d94934ebaabd \
                invocation_id=run-42 measured_ratio=19.9x\n\
                **Incumbent isolation proof:** candidate=fnp.workload.report \
                incumbent=numpy.workload.report shared_timed_component=fnp.sort\n\
                **Shared timed component disclosure:** components=fnp.sort \
                direction=conservative_for_candidate share_of_candidate_pct=3.0";
    assert!(!has_incumbent_win_contract(body));
}

#[test]
fn win_class_marker_is_read_from_the_body_not_the_heading() {
    assert_eq!(
        result_class("**Campaign result class:** incumbent-win"),
        ResultClass::IncumbentWin
    );
    assert_eq!(
        result_class("**Campaign result class:** maintenance-self-speedup"),
        ResultClass::MaintenanceSelfSpeedup
    );
    // A heading token without the body marker is not a declaration.
    assert_eq!(
        result_class("WIN (KEEP, VS-INCUMBENT): f64 isin"),
        ResultClass::Missing
    );
    assert_eq!(
        result_class("**Campaign result class:** fastest-ever"),
        ResultClass::Invalid
    );
}

/// THE WORKER GATE. A measured row written from
/// [`WORKER_PROVENANCE_ENFORCEMENT_DATE`] onward must name the worker its arms
/// ran on. Rows that never got a measurement (bench-blocked, queue refused) and
/// behavioral blockers are exempt: there is no worker to name.
#[test]
fn new_measured_rows_name_their_worker() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| !e.date.is_empty() && e.date.as_str() >= WORKER_PROVENANCE_ENFORCEMENT_DATE)
        .filter(|e| !is_unmeasured(&e.body) && !is_behavioral_blocker(&e.body))
        .filter(|e| !records_measuring_worker(&e.body))
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.is_empty(),
        "{} measured row(s) dated on/after {WORKER_PROVENANCE_ENFORCEMENT_DATE} do not name \
         the worker their arms ran on, so they cannot be compared to any other row:\n{}\n\n\
         Fix by pasting the harness's own provenance line into the row, e.g.\n  \
         HOST_BASELINE host=vmi1293453 cpu_model=... physical_cores=8 governor=...\n\
         A passing A/A null does NOT substitute: the null controls within-invocation noise \
         only, and the fleet has measured a 13.6x swing for one cell across two workers with \
         both nulls passing.",
        offenders.len(),
        offenders.join("\n"),
    );
}

#[test]
fn worker_detector_rejects_rows_without_a_named_worker() {
    // The harness line satisfies the gate verbatim.
    assert!(records_measuring_worker(
        "HOST_BASELINE host=vmi1293453 cpu_model=AMD_EPYC physical_cores=8 governor=unavailable"
    ));
    assert!(records_measuring_worker(
        "Measured on worker=hz1, 8 logical threads"
    ));

    // A row with a null control, a ratio and an ELF sha but no worker is exactly
    // the row this gate exists to catch — a naive "it has provenance" check that
    // keyed off the sha or the null would pass it.
    assert!(!records_measuring_worker(
        "null_base_aa ratio_median=0.997901 ci95=[0.960594,1.013642]\n\
         effect ratio_median=0.511247\n\
         bench_elf_sha256=61a1e5e3c0ffee61a1e5e3c0ffee61a1e5e3c0ffee61a1e5e3c0ffee61a1e5e3"
    ));

    // `governor=unavailable` sits on the same line as a real host in every
    // HOST_BASELINE, so the placeholder rule must reject only the host token
    // itself — not veto a line that merely contains the word somewhere.
    assert!(!records_measuring_worker(
        "HOST_BASELINE host=unavailable governor=performance"
    ));
    assert!(records_measuring_worker(
        "HOST_BASELINE host=hz2 governor=unavailable"
    ));
}

/// THE BACKDATING GUARD for the worker gate. `new_measured_rows_name_their_worker`
/// only inspects rows dated on/after [`WORKER_PROVENANCE_ENFORCEMENT_DATE`], so a
/// new unworkered `incumbent-win` row dated one day earlier would satisfy it. This
/// pins the historical population instead: it may shrink as rows are re-measured
/// and re-banked with their worker, and any growth is a new row dodging the gate.
#[test]
fn historical_unworkered_incumbent_win_debt_does_not_grow() {
    let offenders: Vec<String> = parse_entries()
        .into_iter()
        .filter(|e| e.date.is_empty() || e.date.as_str() < WORKER_PROVENANCE_ENFORCEMENT_DATE)
        .filter(|e| result_class(&e.body) == ResultClass::IncumbentWin)
        .filter(|e| !records_measuring_worker(&e.body))
        .map(|e| format!("  docs/NEGATIVE_EVIDENCE.md:{} — {}", e.line, e.heading))
        .collect();

    assert!(
        offenders.len() <= HISTORICAL_UNWORKERED_INCUMBENT_WIN_BUDGET,
        "unworkered pre-{WORKER_PROVENANCE_ENFORCEMENT_DATE} `{INCUMBENT_WIN}` rows grew to {} \
         (budget {HISTORICAL_UNWORKERED_INCUMBENT_WIN_BUDGET}):\n{}\n\n\
         Either a new row was backdated to dodge `new_measured_rows_name_their_worker`, or an \
         existing row lost its worker. A campaign-output row banked from now on must name the \
         worker it ran on. Do NOT clear this by writing a worker name into an old row — the \
         worker was never recorded there, and inventing one is fabricated provenance. Re-measure \
         and re-bank instead, which lowers the budget.",
        offenders.len(),
        offenders.join("\n"),
    );
}

/// The budget is a ceiling on a real population, not a magic number: if the
/// ledger ever cleans up and the true count drops, this catches the stale
/// constant so the ceiling gets tightened instead of quietly over-permitting.
#[test]
fn unworkered_incumbent_win_budget_is_not_stale() {
    let actual = parse_entries()
        .into_iter()
        .filter(|e| e.date.is_empty() || e.date.as_str() < WORKER_PROVENANCE_ENFORCEMENT_DATE)
        .filter(|e| result_class(&e.body) == ResultClass::IncumbentWin)
        .filter(|e| !records_measuring_worker(&e.body))
        .count();

    assert_eq!(
        actual, HISTORICAL_UNWORKERED_INCUMBENT_WIN_BUDGET,
        "the unworkered incumbent-win population is {actual} but the budget says \
         {HISTORICAL_UNWORKERED_INCUMBENT_WIN_BUDGET}. If rows were re-banked with their \
         worker, LOWER the budget to {actual} so it keeps ratcheting."
    );
}
