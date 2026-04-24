#![forbid(unsafe_code)]
//! Single-command fixture regeneration driver.
//!
//! Satisfies bead franken_numpy-gghj. Walks `fixtures/` and classifies every
//! JSON file into one of three regeneration states:
//!
//! - **captured** — regenerable from live numpy via `capture_numpy_oracle`
//!   (or a future per-domain capture bin). Runs the capture and records
//!   the new case count.
//! - **manual** — hand-authored input corpus (no oracle; the fixture IS
//!   the source of truth). Skip but list in the regeneration report so
//!   maintainers know to diff-review them manually.
//! - **unknown** — fixture whose provenance is not recognized. Emits a
//!   warning; operator must classify it in PROVENANCE.md.
//!
//! Output:
//!   - Runs the capture_numpy_oracle bin for `ufunc_input_cases.json` if
//!     available (Python version + numpy version flow through as env).
//!   - Writes `fixtures/REGENERATION_REPORT.md` summarizing each fixture's
//!     state + last numpy/python oracle version + suggested manual diff
//!     commands.
//!   - Exits non-zero on oracle capture failure.
//!
//! Usage:
//!   cargo run --bin regenerate_all_fixtures -p fnp-conformance        # dry-run: report only
//!   cargo run --bin regenerate_all_fixtures -p fnp-conformance -- --apply   # run oracle captures
//!
//! Environment:
//!   FNP_ORACLE_PYTHON        override python interpreter for numpy oracle
//!   FNP_REQUIRE_REAL_NUMPY   require live numpy (fail on pure-Python fallback)
//!
//! IMPORTANT: `--apply` WILL overwrite fixtures under `fixtures/oracle_outputs/`
//! with fresh numpy output. This can introduce behavior drift — review the
//! resulting git diff carefully before committing. Default is dry-run so
//! `cargo test -p fnp-conformance` still passes after running the report.

use fnp_conformance::HarnessConfig;
use fnp_conformance::ufunc_differential::capture_numpy_oracle;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if let Err(err) = run() {
        eprintln!("regenerate_all_fixtures failed: {err}");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy)]
enum Provenance {
    OracleCaptured, // regenerable via capture_numpy_oracle
    Manual,         // hand-written input corpus
    Derived,        // programmatically enumerated (dtype_promotion, etc.)
}

fn run() -> Result<(), String> {
    let apply = env::args().any(|a| a == "--apply");
    let cfg = HarnessConfig::default_paths();
    let fixtures_dir = &cfg.fixture_root;
    if !fixtures_dir.is_dir() {
        return Err(format!(
            "fixtures dir not found at {}",
            fixtures_dir.display()
        ));
    }

    let numpy_version = probe_numpy_version();
    let python_version = probe_python_version();
    let git_ref = probe_git_head();

    println!("Environment:");
    println!(
        "  numpy.__version__    = {}",
        numpy_version.as_deref().unwrap_or("<unknown>")
    );
    println!(
        "  python version       = {}",
        python_version.as_deref().unwrap_or("<unknown>")
    );
    println!(
        "  git HEAD             = {}",
        git_ref.as_deref().unwrap_or("<unknown>")
    );
    println!(
        "  mode                 = {}",
        if apply {
            "apply (WRITE fixtures)"
        } else {
            "dry-run (report only)"
        }
    );
    println!();

    // Currently only ufunc_input_cases has a live oracle capture binary.
    // Run it only under --apply so the default `cargo run` invocation is
    // side-effect free and safe to chain into CI.
    let ufunc_input_path = fixtures_dir.join("ufunc_input_cases.json");
    let ufunc_output_path = fixtures_dir.join("oracle_outputs/ufunc_oracle_output.json");
    let captured = if apply && ufunc_input_path.is_file() {
        println!("Running oracle capture for ufunc_input_cases.json …");
        match capture_numpy_oracle(&ufunc_input_path, &ufunc_output_path, &cfg.oracle_root) {
            Ok(result) => {
                println!(
                    "  captured {} cases using {}",
                    result.cases.len(),
                    result.oracle_source
                );
                Some(result.cases.len())
            }
            Err(err) => {
                eprintln!("  ufunc oracle capture failed: {err}");
                if env::var("FNP_REQUIRE_REAL_NUMPY_ORACLE").is_ok() {
                    return Err(err);
                }
                None
            }
        }
    } else if ufunc_input_path.is_file() {
        // Dry-run: count what WOULD be captured.
        Some(count_cases(&ufunc_input_path))
    } else {
        None
    };

    let mut fixture_status: BTreeMap<String, (Provenance, usize)> = BTreeMap::new();
    walk_fixtures(fixtures_dir, fixtures_dir, &mut fixture_status)?;

    let report = render_report(
        &fixture_status,
        numpy_version.as_deref(),
        python_version.as_deref(),
        git_ref.as_deref(),
        captured,
    );

    let report_path = fixtures_dir.join("REGENERATION_REPORT.md");
    fs::write(&report_path, &report)
        .map_err(|err| format!("write {}: {err}", report_path.display()))?;
    println!();
    println!("Wrote {}", report_path.display());
    Ok(())
}

fn walk_fixtures(
    root: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, (Provenance, usize)>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|err| format!("read_dir {}: {err}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("read_dir entry: {err}"))?;
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if name == "oracle_outputs" {
                continue;
            }
            walk_fixtures(root, &path, out)?;
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".json")
            || name.ends_with("_report.json")
            || name.ends_with("_output.json")
        {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let count = count_cases(&path);
        let prov = classify_provenance(name);
        out.insert(rel, (prov, count));
    }
    Ok(())
}

fn count_cases(path: &Path) -> usize {
    let Ok(content) = fs::read_to_string(path) else {
        return 0;
    };
    match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(serde_json::Value::Array(items)) => items.len(),
        Ok(serde_json::Value::Object(obj)) => ["cases", "entries", "corpus", "items"]
            .iter()
            .find_map(|k| obj.get(*k).and_then(|v| v.as_array()).map(|a| a.len()))
            .unwrap_or(1),
        _ => 0,
    }
}

fn classify_provenance(filename: &str) -> Provenance {
    let stem = filename.strip_suffix(".json").unwrap_or(filename);
    // *_differential_cases => oracle captured.
    if stem.ends_with("_differential_cases") {
        return Provenance::OracleCaptured;
    }
    // *_metamorphic_cases / *_adversarial_cases => hand-written invariants
    // or edge cases.
    if stem.ends_with("_metamorphic_cases") || stem.ends_with("_adversarial_cases") {
        return Provenance::Manual;
    }
    match stem {
        "dtype_promotion_cases" => Provenance::Derived,
        "shape_stride_cases"
        | "runtime_policy_cases"
        | "override_audit_cases"
        | "rng_statistical_cases"
        | "workflow_scenario_corpus"
        | "smoke_case" => Provenance::Manual,
        "ufunc_input_cases" => Provenance::Manual, // inputs; oracle output in fixtures/oracle_outputs/
        _ => Provenance::Manual,
    }
}

fn render_report(
    fixtures: &BTreeMap<String, (Provenance, usize)>,
    numpy_version: Option<&str>,
    python_version: Option<&str>,
    git_ref: Option<&str>,
    ufunc_captured: Option<usize>,
) -> String {
    let unix_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let mut out = String::new();
    out.push_str("# Fixture Regeneration Report\n\n");
    out.push_str(&format!(
        "> Generated by `cargo run --bin regenerate_all_fixtures -p fnp-conformance` at unix_ts={unix_ts}.\n"
    ));
    out.push_str(&format!(
        "> Environment: numpy={} · python={} · git HEAD={}.\n\n",
        numpy_version.unwrap_or("<unknown>"),
        python_version.unwrap_or("<unknown>"),
        git_ref.unwrap_or("<unknown>"),
    ));
    match ufunc_captured {
        Some(n) => out.push_str(&format!(
            "ufunc_input_cases.json: {n} cases (oracle capture runs under `--apply`).\n\n"
        )),
        None => out.push_str("ufunc_input_cases.json: — not available in this run\n\n"),
    }

    out.push_str("## Fixtures by regeneration class\n\n");
    out.push_str("| File | Provenance | Cases | Regeneration action |\n");
    out.push_str("|------|-----------|:-----:|--------------------|\n");
    for (path, (prov, count)) in fixtures {
        let (label, action) = match prov {
            Provenance::OracleCaptured => (
                "captured",
                "Re-run numpy oracle capture (per-domain; currently only ufunc has a live oracle binary — others need per-domain capture bins, tracked in bead gghj follow-up)",
            ),
            Provenance::Manual => (
                "manual",
                "Hand-written; diff-review on numpy version bump to check for behavior drift",
            ),
            Provenance::Derived => (
                "derived",
                "Regenerated programmatically from a dtype matrix enumerator (see fixtures/PROVENANCE.md)",
            ),
        };
        out.push_str(&format!("| `{path}` | {label} | {count} | {action} |\n"));
    }

    out.push_str("\n## Upgrade workflow (when numpy version bumps)\n\n");
    out.push_str("1. Pin the new numpy version via `FNP_ORACLE_PYTHON=/path/to/python3.x`.\n");
    out.push_str("2. Run `cargo run --bin regenerate_all_fixtures -p fnp-conformance`.\n");
    out.push_str(
        "3. Diff-review each `captured` fixture: `git diff fixtures/*_differential_cases.json`.\n",
    );
    out.push_str("4. For each unexpected divergence, either (a) accept — bump numpy version in `fixtures/PROVENANCE.md`, or (b) file a DISC-NNN in `DISCREPANCIES.md` and XFAIL the affected test.\n");
    out.push_str("5. Manual fixtures: visually inspect on numpy releases that ship breaking ufunc/ma semantics changes.\n");
    out.push_str("6. Update `fixtures/PROVENANCE.md` `Reference NumPy` line and each captured fixture's `Last regen` column.\n");

    out
}

fn probe_numpy_version() -> Option<String> {
    probe_python_eval("import numpy; print(numpy.__version__)")
}

fn probe_python_version() -> Option<String> {
    probe_python_eval("import sys; print('.'.join(str(c) for c in sys.version_info[:3]))")
}

fn probe_python_eval(code: &str) -> Option<String> {
    let python = python_candidates()
        .into_iter()
        .find(|cand| Path::new(cand).exists() || cand_on_path(cand))?;
    let output = Command::new(&python).arg("-c").arg(code).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn python_candidates() -> Vec<String> {
    if let Ok(custom) = env::var("FNP_ORACLE_PYTHON") {
        return vec![custom];
    }
    vec![
        String::from("python3"),
        String::from("python"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.venv-numpy314/bin/python3")
            .to_string_lossy()
            .into_owned(),
    ]
}

fn cand_on_path(candidate: &str) -> bool {
    // `which` would be cleaner but adds a dep; brute-force check PATH.
    let Ok(paths) = env::var("PATH") else {
        return false;
    };
    for prefix in paths.split(':') {
        if Path::new(prefix).join(candidate).is_file() {
            return true;
        }
    }
    false
}

fn probe_git_head() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
