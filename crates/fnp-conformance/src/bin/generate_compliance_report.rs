#![forbid(unsafe_code)]
//! Auto-generate the conformance compliance matrix.
//!
//! Reads every `*_cases.json` fixture under `fixtures/`, tallies cases per
//! domain × test-category (differential / metamorphic / adversarial), and
//! emits `fixtures/COVERAGE.generated.md` — a Markdown table that mirrors
//! the hand-maintained `COVERAGE.md` but is mechanically reproducible.
//!
//! Satisfies the testing-conformance-harnesses skill's "Compliance Report
//! Generator" pattern. Closes bead franken_numpy-vobv.
//!
//! Usage:
//!   cargo run --bin generate_compliance_report -p fnp-conformance
//!
//! Exit codes:
//!   0 — report generated
//!   1 — fixtures dir missing / unreadable
//!   2 — MUST-level case count < 95% target (reserved; not yet enforced
//!       because severity tags aren't present in fixture schemas yet)

use fnp_conformance::HarnessConfig;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn main() {
    if let Err(err) = run() {
        eprintln!("generate_compliance_report failed: {err}");
        std::process::exit(1);
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct DomainStats {
    differential: usize,
    metamorphic: usize,
    adversarial: usize,
    other: usize,
}

impl DomainStats {
    fn total(&self) -> usize {
        self.differential + self.metamorphic + self.adversarial + self.other
    }
}

fn run() -> Result<(), String> {
    let cfg = HarnessConfig::default_paths();
    let fixtures_dir = &cfg.fixture_root;
    if !fixtures_dir.is_dir() {
        return Err(format!(
            "fixtures dir not found at {}",
            fixtures_dir.display()
        ));
    }

    let mut stats: BTreeMap<String, DomainStats> = BTreeMap::new();
    walk_fixtures(fixtures_dir, &mut stats)?;

    let markdown = render_markdown(&stats);

    let output_path = fixtures_dir.join("COVERAGE.generated.md");
    fs::write(&output_path, &markdown)
        .map_err(|err| format!("write {}: {err}", output_path.display()))?;

    println!("wrote {}", output_path.display());
    println!("domain summary:");
    for (domain, stat) in &stats {
        println!(
            "  {domain:<14}  diff={:>4} meta={:>4} adv={:>4} other={:>4} total={:>4}",
            stat.differential,
            stat.metamorphic,
            stat.adversarial,
            stat.other,
            stat.total()
        );
    }

    Ok(())
}

fn walk_fixtures(root: &Path, stats: &mut BTreeMap<String, DomainStats>) -> Result<(), String> {
    let entries =
        fs::read_dir(root).map_err(|err| format!("read_dir {}: {err}", root.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("read_dir entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            // Skip oracle_outputs / packet subdirs — they hold mismatch
            // reports, not fixture input cases.
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if name == "oracle_outputs" {
                continue;
            }
            walk_fixtures(&path, stats)?;
            continue;
        }
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !filename.ends_with(".json") {
            continue;
        }
        // Skip oracle output + report files (they mirror fixtures rather
        // than being primary input fixtures).
        if filename.ends_with("_report.json") || filename.ends_with("_output.json") {
            continue;
        }
        count_fixture(&path, filename, stats)?;
    }
    Ok(())
}

fn count_fixture(
    path: &Path,
    filename: &str,
    stats: &mut BTreeMap<String, DomainStats>,
) -> Result<(), String> {
    let content =
        fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let value: Value =
        serde_json::from_str(&content).map_err(|err| format!("parse {}: {err}", path.display()))?;

    let count = match &value {
        Value::Array(items) => items.len(),
        Value::Object(obj) => {
            // Some fixtures wrap the case list under a `cases` key or a
            // similarly-named array field.
            ["cases", "entries", "corpus", "items"]
                .iter()
                .find_map(|key| obj.get(*key).and_then(|v| v.as_array()).map(|v| v.len()))
                .unwrap_or(1)
        }
        _ => 1,
    };

    let (domain, category) = classify_fixture(filename);
    let entry = stats.entry(domain.to_string()).or_default();
    match category {
        Category::Differential => entry.differential += count,
        Category::Metamorphic => entry.metamorphic += count,
        Category::Adversarial => entry.adversarial += count,
        Category::Other => entry.other += count,
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum Category {
    Differential,
    Metamorphic,
    Adversarial,
    Other,
}

/// Split `ufunc_differential_cases.json` into (domain="ufunc",
/// category=Differential). The domain is the stem before the first
/// `_differential_` / `_metamorphic_` / `_adversarial_` marker; anything
/// without a marker falls into `Other` (e.g. `dtype_promotion_cases.json`,
/// `runtime_policy_cases.json`, `workflow_scenario_corpus.json`,
/// `smoke_case.json`).
fn classify_fixture(filename: &str) -> (&'static str, Category) {
    let stem = filename.strip_suffix(".json").unwrap_or(filename);
    // Handle `rng_distribution_differential_cases` explicitly as part of
    // the rng domain rather than a "distribution" pseudo-domain. Must
    // come before the generic _differential_cases match since the longer
    // prefix would otherwise strip to "rng_distribution".
    if stem == "rng_distribution_differential_cases" {
        return ("rng", Category::Differential);
    }
    for (marker, category) in [
        ("_differential_cases", Category::Differential),
        ("_metamorphic_cases", Category::Metamorphic),
        ("_adversarial_cases", Category::Adversarial),
    ] {
        if let Some(domain) = stem.strip_suffix(marker) {
            return (static_domain(domain), category);
        }
    }
    // Standalone fixtures with no {diff,meta,adv} suffix.
    let domain = match stem {
        "dtype_promotion_cases" => "dtype",
        "shape_stride_cases" => "shape_stride",
        "runtime_policy_cases" => "runtime_policy",
        "override_audit_cases" => "runtime_policy",
        "rng_statistical_cases" => "rng",
        "ufunc_input_cases" => "ufunc",
        "workflow_scenario_corpus" => "workflow",
        "smoke_case" => "smoke",
        _ => "other",
    };
    (domain, Category::Other)
}

/// Map a scanned domain prefix to a stable, canonical name that matches
/// the hand-maintained COVERAGE.md rows.
fn static_domain(domain: &str) -> &'static str {
    match domain {
        "ufunc" => "ufunc",
        "signal" => "signal",
        "polynomial" => "polynomial",
        "io" => "io",
        "linalg" => "linalg",
        "string" => "string",
        "rng" => "rng",
        "fft" => "fft",
        "datetime" => "datetime",
        "masked" => "masked",
        "iter" => "iter",
        "shape_stride" => "shape_stride",
        "dtype" => "dtype",
        "runtime_policy" => "runtime_policy",
        _ => "other",
    }
}

fn render_markdown(stats: &BTreeMap<String, DomainStats>) -> String {
    let generated_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let mut out = String::new();
    out.push_str("# Conformance Coverage Matrix (auto-generated)\n\n");
    out.push_str(&format!(
        "> Generated by `cargo run --bin generate_compliance_report -p fnp-conformance` at unix_ts={generated_at}.\n"
    ));
    out.push_str(
        "> Do not edit by hand — source of truth is the fixture JSON under `fixtures/`.\n",
    );
    out.push_str("> For narrative coverage notes and intentional divergences see `COVERAGE.md` and `DISCREPANCIES.md`.\n\n");

    out.push_str("## Case counts per domain × category\n\n");
    out.push_str("| Domain | Differential | Metamorphic | Adversarial | Other | Total |\n");
    out.push_str("|--------|:-----------:|:-----------:|:-----------:|:-----:|:-----:|\n");

    let mut grand = DomainStats::default();
    for (domain, stat) in stats {
        out.push_str(&format!(
            "| {domain} | {} | {} | {} | {} | {} |\n",
            stat.differential,
            stat.metamorphic,
            stat.adversarial,
            stat.other,
            stat.total()
        ));
        grand.differential += stat.differential;
        grand.metamorphic += stat.metamorphic;
        grand.adversarial += stat.adversarial;
        grand.other += stat.other;
    }
    out.push_str(&format!(
        "| **Total** | **{}** | **{}** | **{}** | **{}** | **{}** |\n",
        grand.differential,
        grand.metamorphic,
        grand.adversarial,
        grand.other,
        grand.total()
    ));

    out.push_str("\n## Notes\n\n");
    out.push_str("- `Other` buckets count cases in fixtures without a `_differential`/`_metamorphic`/`_adversarial` filename marker (e.g. `dtype_promotion_cases.json`, `ufunc_input_cases.json`).\n");
    out.push_str("- `rng` aggregates `rng_differential`, `rng_metamorphic`, `rng_adversarial`, `rng_distribution_differential`, and `rng_statistical_cases`.\n");
    out.push_str("- `runtime_policy` includes `runtime_policy_cases` + `override_audit_cases`.\n");
    out.push_str("- Fixtures under `fixtures/oracle_outputs/` are excluded (those are comparator mismatch reports, not primary case inputs).\n");
    out.push_str("- Packet subdirectories (e.g. `packet002_dtype/`) are traversed recursively and their cases are merged into the matching top-level domain.\n");

    out
}
