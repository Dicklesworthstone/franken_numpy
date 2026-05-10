#![forbid(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const SCHEMA_VERSION: &str = "validation_recipe_selector.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Markdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Options {
    changed_paths: Vec<String>,
    crate_names: Vec<String>,
    labels: Vec<String>,
    format: OutputFormat,
    report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct SelectorInput {
    changed_paths: Vec<String>,
    crate_names: Vec<String>,
    labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ValidationRecipe {
    id: String,
    title: String,
    reason: String,
    applies_to: Vec<String>,
    commands: Vec<String>,
    expected_reports: Vec<String>,
    prerequisites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ValidationRecipeReport {
    schema_version: &'static str,
    input: SelectorInput,
    target_crates: Vec<String>,
    risk_level: String,
    workspace_required: bool,
    manual_review_required: bool,
    recipes: Vec<ValidationRecipe>,
    warnings: Vec<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("run_validation_recipe_selector failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args(std::env::args().skip(1))?;
    let report = build_report(&options);
    let rendered = match options.format {
        OutputFormat::Json => serde_json::to_string_pretty(&report)
            .map_err(|err| format!("serialize report: {err}"))?,
        OutputFormat::Markdown => render_markdown(&report),
    };

    if let Some(path) = options.report_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("create report directory {}: {err}", parent.display()))?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .map_err(|err| format!("write report {}: {err}", path.display()))?;
        println!("wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut changed_paths = Vec::new();
    let mut crate_names = Vec::new();
    let mut labels = Vec::new();
    let mut format = OutputFormat::Json;
    let mut report_path = None;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--changed" | "--path" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                changed_paths.extend(split_values(&value));
            }
            "--crate" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--crate requires a value".to_string())?;
                crate_names.extend(split_values(&value));
            }
            "--label" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--label requires a value".to_string())?;
                labels.extend(split_values(&value));
            }
            "--format" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--format requires json or markdown".to_string())?;
                format = match value.as_str() {
                    "json" => OutputFormat::Json,
                    "markdown" | "md" => OutputFormat::Markdown,
                    other => return Err(format!("unsupported format '{other}'\n{}", usage())),
                };
            }
            "--report-path" | "--output-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                report_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument '{other}'\n{}", usage())),
        }
    }

    changed_paths = normalize_values(changed_paths);
    crate_names = normalize_values(crate_names);
    labels = normalize_values(labels);
    Ok(Options {
        changed_paths,
        crate_names,
        labels,
        format,
        report_path,
    })
}

fn split_values(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_values(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| normalize_path(&value))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn normalize_path(value: &str) -> String {
    value
        .trim()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

fn build_report(options: &Options) -> ValidationRecipeReport {
    let mut crates = options
        .crate_names
        .iter()
        .map(|name| canonical_crate_name(name))
        .collect::<BTreeSet<_>>();
    let mut warnings = Vec::new();
    let mut unknown_paths = Vec::new();

    for path in &options.changed_paths {
        match crate_from_path(path) {
            Some(crate_name) => {
                crates.insert(crate_name.to_string());
            }
            None if is_docs_path(path) => {}
            None => unknown_paths.push(path.clone()),
        }
    }

    let labels = options
        .labels
        .iter()
        .map(|label| label.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    let needs_porting_ledger_freshness =
        needs_porting_ledger_freshness_recipe(&options.changed_paths, &labels);
    let docs_only = !options.changed_paths.is_empty()
        && options.crate_names.is_empty()
        && options.labels.is_empty()
        && !needs_porting_ledger_freshness
        && options
            .changed_paths
            .iter()
            .all(|path| is_docs_only_path(path));
    let mut recipes = BTreeMap::<String, ValidationRecipe>::new();

    if docs_only {
        insert_recipe(&mut recipes, docs_recipe(&options.changed_paths));
    } else {
        for crate_name in &crates {
            match crate_name.as_str() {
                "fnp-io" => insert_recipe(&mut recipes, fnp_io_recipe(&options.changed_paths)),
                "fnp-python" => {
                    insert_recipe(
                        &mut recipes,
                        fnp_python_recipe(&options.changed_paths, &labels),
                    );
                }
                "fnp-conformance" => insert_recipe(
                    &mut recipes,
                    fnp_conformance_recipe(&options.changed_paths, &labels),
                ),
                other if other.starts_with("fnp-") => {
                    insert_recipe(
                        &mut recipes,
                        generic_crate_recipe(other, &options.changed_paths),
                    );
                }
                other => warnings.push(format!("unknown crate selector '{other}'")),
            }
        }

        if labels
            .iter()
            .any(|label| matches!(label.as_str(), "perf" | "performance" | "benchmark"))
        {
            insert_recipe(&mut recipes, performance_recipe());
        }
        if labels
            .iter()
            .any(|label| label.contains("packet") || label.starts_with("fnp-p2c-"))
            || options
                .changed_paths
                .iter()
                .any(|path| path.starts_with("artifacts/phase2c/"))
        {
            insert_recipe(&mut recipes, packet_recipe(&labels, &options.changed_paths));
        }
    }
    if needs_porting_ledger_freshness {
        insert_recipe(&mut recipes, porting_ledger_freshness_recipe());
    }

    if !unknown_paths.is_empty() {
        warnings.push(format!(
            "no crate mapping for: {}",
            unknown_paths.join(", ")
        ));
        insert_recipe(&mut recipes, manual_recipe(&options.changed_paths));
    }
    if recipes.is_empty() {
        insert_recipe(&mut recipes, manual_recipe(&options.changed_paths));
    }

    let target_crates = crates.into_iter().collect::<Vec<_>>();
    let manual_review_required = !unknown_paths.is_empty();
    let risk_level = risk_level(options, &target_crates, docs_only, manual_review_required);
    ValidationRecipeReport {
        schema_version: SCHEMA_VERSION,
        input: SelectorInput {
            changed_paths: options.changed_paths.clone(),
            crate_names: options.crate_names.clone(),
            labels: options.labels.clone(),
        },
        target_crates,
        risk_level,
        workspace_required: false,
        manual_review_required,
        recipes: recipes.into_values().collect(),
        warnings,
    }
}

fn insert_recipe(recipes: &mut BTreeMap<String, ValidationRecipe>, recipe: ValidationRecipe) {
    recipes.entry(recipe.id.clone()).or_insert(recipe);
}

fn canonical_crate_name(name: &str) -> String {
    let name = name.trim();
    if name.starts_with("fnp-") {
        name.to_string()
    } else {
        format!("fnp-{name}")
    }
}

fn crate_from_path(path: &str) -> Option<String> {
    let mut parts = path.split('/');
    if parts.next()? != "crates" {
        return None;
    }
    let crate_name = parts.next()?;
    if crate_name.starts_with("fnp-") {
        Some(crate_name.to_string())
    } else {
        None
    }
}

fn is_docs_path(path: &str) -> bool {
    path.starts_with("docs/")
        || path.starts_with("artifacts/")
        || path.ends_with(".md")
        || matches!(path, "README.md" | "AGENTS.md" | "CHANGELOG.md")
}

fn is_docs_only_path(path: &str) -> bool {
    is_docs_path(path) && !path.starts_with("artifacts/phase2c/")
}

fn needs_porting_ledger_freshness_recipe(paths: &[String], labels: &BTreeSet<String>) -> bool {
    let phase2c_label = labels.iter().any(|label| {
        matches!(label.as_str(), "phase2c" | "stale-evidence" | "fail-closed")
            || label.starts_with("fnp-p2c-")
    });
    let relevant_path = paths.iter().any(|path| {
        path == "artifacts/contracts/PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md"
            || path.starts_with("artifacts/phase2c/")
            || path.ends_with("validate_phase2c_porting_ledger.rs")
    });
    let phase2c_context_label = labels
        .iter()
        .any(|label| matches!(label.as_str(), "validation" | "evidence"))
        && relevant_path;
    phase2c_label || relevant_path || phase2c_context_label
}

fn risk_level(
    options: &Options,
    target_crates: &[String],
    docs_only: bool,
    manual_review_required: bool,
) -> String {
    if docs_only {
        return "low".to_string();
    }
    if manual_review_required {
        return "manual".to_string();
    }
    if options.changed_paths.iter().any(|path| {
        path.contains("/src/")
            && (path.contains("fnp-io")
                || path.contains("fnp-python")
                || path.contains("fnp-ndarray")
                || path.contains("fnp-dtype")
                || path.contains("fnp-ufunc"))
    }) || options.labels.iter().any(|label| {
        let label = label.to_ascii_lowercase();
        label.contains("security")
            || label.contains("parser")
            || label.contains("diagnostic")
            || label.contains("performance")
            || label.contains("packet")
    }) {
        return "high".to_string();
    }
    if !target_crates.is_empty() {
        "medium".to_string()
    } else {
        "manual".to_string()
    }
}

fn changed_arg(paths: &[String]) -> String {
    if paths.is_empty() {
        "<changed-files>".to_string()
    } else {
        paths.join(" ")
    }
}

fn ubs_command(paths: &[String]) -> String {
    format!("ubs {}", changed_arg(paths))
}

fn diff_check_command(paths: &[String]) -> String {
    format!("git diff --check -- {}", changed_arg(paths))
}

fn common_rch_prerequisites() -> Vec<String> {
    vec![
        "Set a bead-specific CARGO_TARGET_DIR under /data/projects/.cargo-target-franken_numpy-<agent>-<bead>".to_string(),
        "Set RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR so rch forwards the target dir".to_string(),
        "Run cargo only through rch exec -- cargo ...".to_string(),
    ]
}

fn docs_recipe(paths: &[String]) -> ValidationRecipe {
    ValidationRecipe {
        id: "docs-only".to_string(),
        title: "Docs-only validation".to_string(),
        reason: "Changed paths are documentation or artifact text; no Rust build is required by default.".to_string(),
        applies_to: paths.to_vec(),
        commands: vec![diff_check_command(paths)],
        expected_reports: Vec::new(),
        prerequisites: Vec::new(),
    }
}

fn fnp_io_recipe(paths: &[String]) -> ValidationRecipe {
    ValidationRecipe {
        id: "fnp-io-parser-diagnostics".to_string(),
        title: "fnp-io parser and NPZ diagnostics".to_string(),
        reason: "IO parser, NPY, or NPZ changes need focused parser tests plus the diagnostic oracle gate.".to_string(),
        applies_to: paths.to_vec(),
        commands: vec![
            "rch exec -- cargo test -p fnp-io npy_npz_diagnostic -- --nocapture".to_string(),
            "rch exec -- cargo test -p fnp-conformance io_diagnostics -- --nocapture".to_string(),
            "rch exec -- cargo run -p fnp-conformance --bin run_io_diagnostics -- --report-path target/io_diagnostics.json --jsonl-path target/io_diagnostics.jsonl".to_string(),
            "rch exec -- cargo clippy -p fnp-io --all-targets -- -D warnings".to_string(),
            ubs_command(paths),
        ],
        expected_reports: vec![
            "target/io_diagnostics.json".to_string(),
            "target/io_diagnostics.jsonl".to_string(),
        ],
        prerequisites: {
            let mut prerequisites = common_rch_prerequisites();
            prerequisites.push("FNP_ORACLE_PYTHON may point at a NumPy-enabled interpreter for oracle diagnostics".to_string());
            prerequisites
        },
    }
}

fn fnp_python_recipe(paths: &[String], labels: &BTreeSet<String>) -> ValidationRecipe {
    let diagnostic_surface = paths
        .iter()
        .any(|path| path.contains("conformance_diagnostics"))
        || labels.iter().any(|label| {
            matches!(
                label.as_str(),
                "diagnostic" | "diagnostics" | "warning" | "warnings" | "exception" | "exceptions"
            )
        });
    let mut commands = Vec::new();
    if diagnostic_surface {
        commands.push(
            "rch exec -- cargo test -p fnp-python --test conformance_diagnostics -- --nocapture"
                .to_string(),
        );
    }
    commands.extend([
        "rch exec -- cargo run -p fnp-conformance --bin run_fnp_python_conformance_shards -- --shard fnp-python-smoke --dry-run --report-path artifacts/logs/fnp_python_conformance_shards.jsonl".to_string(),
        "rch exec -- cargo run -p fnp-conformance --bin run_fnp_python_api_coverage -- --report-path artifacts/logs/fnp_python_api_coverage.json".to_string(),
        "rch exec -- cargo clippy -p fnp-python --all-targets -- -D warnings".to_string(),
        ubs_command(paths),
    ]);

    ValidationRecipe {
        id: "fnp-python-diagnostic-shard".to_string(),
        title: "fnp-python conformance shard selection".to_string(),
        reason: "Python surface changes should validate the matching diagnostic/conformance shard and keep the API coverage ledger current.".to_string(),
        applies_to: paths.to_vec(),
        commands,
        expected_reports: vec![
            "artifacts/logs/fnp_python_conformance_shards.jsonl".to_string(),
            "artifacts/logs/fnp_python_api_coverage.json".to_string(),
        ],
        prerequisites: {
            let mut prerequisites = common_rch_prerequisites();
            prerequisites.push("Python environment must import both fnp_python and numpy for conformance shards".to_string());
            prerequisites
        },
    }
}

fn fnp_conformance_recipe(paths: &[String], labels: &BTreeSet<String>) -> ValidationRecipe {
    let mut commands = vec![
        "rch exec -- cargo test -p fnp-conformance validation_recipe_selector -- --nocapture"
            .to_string(),
    ];
    if labels.contains("diagnostic") || paths.iter().any(|path| path.contains("diagnostic")) {
        commands.push("rch exec -- cargo run -p fnp-conformance --bin run_diagnostic_oracle -- --smoke --report-path target/diagnostic_oracle.json --jsonl-path target/diagnostic_oracle.jsonl".to_string());
    }
    commands.extend([
        "rch exec -- cargo clippy -p fnp-conformance --all-targets -- -D warnings".to_string(),
        ubs_command(paths),
    ]);
    ValidationRecipe {
        id: "fnp-conformance-selector".to_string(),
        title: "fnp-conformance selector or harness validation".to_string(),
        reason:
            "Conformance harness changes need their focused unit tests and crate-level lint gate."
                .to_string(),
        applies_to: paths.to_vec(),
        commands,
        expected_reports: Vec::new(),
        prerequisites: common_rch_prerequisites(),
    }
}

fn generic_crate_recipe(crate_name: &str, paths: &[String]) -> ValidationRecipe {
    ValidationRecipe {
        id: format!("{crate_name}-focused"),
        title: format!("{crate_name} focused validation"),
        reason: format!(
            "{crate_name} changes can be proven with package-scoped tests and linting."
        ),
        applies_to: paths.to_vec(),
        commands: vec![
            format!("rch exec -- cargo test -p {crate_name} -- --nocapture"),
            format!("rch exec -- cargo clippy -p {crate_name} --all-targets -- -D warnings"),
            ubs_command(paths),
        ],
        expected_reports: Vec::new(),
        prerequisites: common_rch_prerequisites(),
    }
}

fn performance_recipe() -> ValidationRecipe {
    ValidationRecipe {
        id: "performance-budget-gate".to_string(),
        title: "Performance budget gate".to_string(),
        reason: "Performance-labeled work should preserve benchmark baseline and budget reports.".to_string(),
        applies_to: vec!["performance".to_string()],
        commands: vec![
            "rch exec -- cargo run -p fnp-conformance --bin generate_benchmark_baseline".to_string(),
            "rch exec -- cargo run -p fnp-conformance --bin run_performance_budget_gate -- --generate-candidate".to_string(),
        ],
        expected_reports: vec![
            "artifacts/benchmarks/baseline_*.json".to_string(),
            "target/performance_budget_gate.json".to_string(),
        ],
        prerequisites: common_rch_prerequisites(),
    }
}

fn packet_recipe(labels: &BTreeSet<String>, paths: &[String]) -> ValidationRecipe {
    let packet_id = labels
        .iter()
        .find(|label| label.starts_with("fnp-p2c-"))
        .cloned()
        .or_else(|| packet_id_from_paths(paths))
        .unwrap_or_else(|| "FNP-P2C-<id>".to_string());
    let packet_id = packet_id.to_ascii_uppercase();
    ValidationRecipe {
        id: "phase2c-packet-gate".to_string(),
        title: "Phase2C packet readiness gate".to_string(),
        reason: "Packet/artifact changes need the packet readiness validator and generated report."
            .to_string(),
        applies_to: vec![packet_id.clone()],
        commands: vec![format!(
            "rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id {packet_id}"
        )],
        expected_reports: vec![format!(
            "artifacts/phase2c/{packet_id}/packet_readiness_report.json"
        )],
        prerequisites: common_rch_prerequisites(),
    }
}

fn porting_ledger_freshness_recipe() -> ValidationRecipe {
    ValidationRecipe {
        id: "phase2c-porting-ledger-freshness".to_string(),
        title: "Phase2C porting ledger freshness verifier".to_string(),
        reason: "Phase2C control-plane or packet-evidence changes should prove the central porting ledger does not contain stale proof-status claims for validator-ready packets.".to_string(),
        applies_to: vec![
            "artifacts/contracts/PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md".to_string(),
            "artifacts/phase2c/FNP-P2C-*/packet_readiness_report.json".to_string(),
        ],
        commands: vec![
            "rch exec -- cargo run -p fnp-conformance --bin validate_phase2c_porting_ledger -- --report-out target/phase2c_porting_ledger_freshness_report.json".to_string(),
        ],
        expected_reports: vec![
            "target/phase2c_porting_ledger_freshness_report.json".to_string(),
        ],
        prerequisites: {
            let mut prerequisites = common_rch_prerequisites();
            prerequisites.push(
                "Requires the central porting ledger and packet_readiness_report.json files for FNP-P2C-* packets.".to_string(),
            );
            prerequisites.push(
                "Failure means at least one ready packet row still advertises stale anchor-only/open/partial/missing-artifact status; fix the ledger or packet report before proceeding.".to_string(),
            );
            prerequisites
        },
    }
}

fn packet_id_from_paths(paths: &[String]) -> Option<String> {
    paths.iter().find_map(|path| {
        path.split('/')
            .find(|part| part.starts_with("FNP-P2C-"))
            .map(ToOwned::to_owned)
    })
}

fn manual_recipe(paths: &[String]) -> ValidationRecipe {
    ValidationRecipe {
        id: "manual-triage".to_string(),
        title: "Manual validation triage".to_string(),
        reason: "Changed paths do not map to a known crate or validation family.".to_string(),
        applies_to: paths.to_vec(),
        commands: vec![diff_check_command(paths), ubs_command(paths)],
        expected_reports: Vec::new(),
        prerequisites: vec![
            "Inspect the path owner before choosing a cargo package; do not default to workspace-wide validation without a concrete reason.".to_string(),
        ],
    }
}

fn render_markdown(report: &ValidationRecipeReport) -> String {
    let mut out = String::new();
    out.push_str("# Validation Recipe\n\n");
    out.push_str(&format!("- schema: `{}`\n", report.schema_version));
    out.push_str(&format!("- risk: `{}`\n", report.risk_level));
    out.push_str(&format!(
        "- workspace_required: `{}`\n",
        report.workspace_required
    ));
    if !report.target_crates.is_empty() {
        out.push_str(&format!(
            "- target_crates: `{}`\n",
            report.target_crates.join(", ")
        ));
    }
    if !report.warnings.is_empty() {
        out.push_str("\n## Warnings\n\n");
        for warning in &report.warnings {
            out.push_str(&format!("- {warning}\n"));
        }
    }
    out.push_str("\n## Recipes\n");
    for recipe in &report.recipes {
        out.push_str(&format!("\n### {}\n\n", recipe.title));
        out.push_str(&format!("{}\n\n", recipe.reason));
        if !recipe.commands.is_empty() {
            out.push_str("Commands:\n");
            for command in &recipe.commands {
                out.push_str(&format!("- `{command}`\n"));
            }
        }
        if !recipe.expected_reports.is_empty() {
            out.push_str("\nExpected reports:\n");
            for path in &recipe.expected_reports {
                out.push_str(&format!("- `{path}`\n"));
            }
        }
    }
    out
}

fn usage() -> String {
    "Usage: cargo run -p fnp-conformance --bin run_validation_recipe_selector -- --changed <path>[,<path>...] [--crate <crate>] [--label <label>] [--format json|markdown] [--report-path <path>]".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report_for(args: &[&str]) -> ValidationRecipeReport {
        let options = parse_args(args.iter().map(|arg| (*arg).to_string())).expect("parse args");
        build_report(&options)
    }

    #[test]
    fn validation_recipe_selector_maps_changed_path_to_crate() {
        let report = report_for(&["--changed", "crates/fnp-io/src/lib.rs"]);

        assert_eq!(report.target_crates, vec!["fnp-io"]);
        assert!(
            report
                .recipes
                .iter()
                .any(|recipe| recipe.id == "fnp-io-parser-diagnostics")
        );
    }

    #[test]
    fn validation_recipe_selector_selects_fnp_python_diagnostic_shard() {
        let report = report_for(&[
            "--changed",
            "crates/fnp-python/tests/conformance_diagnostics.rs",
            "--label",
            "warning",
        ]);

        let recipe = report
            .recipes
            .iter()
            .find(|recipe| recipe.id == "fnp-python-diagnostic-shard")
            .expect("fnp-python recipe");
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("conformance_diagnostics"))
        );
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("run_fnp_python_conformance_shards"))
        );
    }

    #[test]
    fn validation_recipe_selector_selects_fnp_io_parser_gates() {
        let report = report_for(&["--changed", "crates/fnp-io/tests/npy_npz_diagnostic.rs"]);
        let recipe = report
            .recipes
            .iter()
            .find(|recipe| recipe.id == "fnp-io-parser-diagnostics")
            .expect("fnp-io recipe");

        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("npy_npz_diagnostic"))
        );
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("run_io_diagnostics"))
        );
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.starts_with("ubs "))
        );
    }

    #[test]
    fn validation_recipe_selector_keeps_docs_only_narrow() {
        let report = report_for(&["--changed", "docs/DIVERGENCES.md"]);

        assert_eq!(report.risk_level, "low");
        assert!(!report.workspace_required);
        assert_eq!(
            report.recipes,
            vec![docs_recipe(&["docs/DIVERGENCES.md".to_string()])]
        );
        assert!(
            report.recipes[0]
                .commands
                .iter()
                .all(|command| !command.contains("cargo"))
        );
    }

    #[test]
    fn validation_recipe_selector_marks_unknown_paths_for_manual_triage() {
        let report = report_for(&["--changed", "misc/generated/input.dat"]);

        assert!(report.manual_review_required);
        assert!(!report.workspace_required);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("misc/generated/input.dat"))
        );
        assert!(
            report
                .recipes
                .iter()
                .any(|recipe| recipe.id == "manual-triage")
        );
    }

    #[test]
    fn validation_recipe_selector_does_not_default_narrow_crates_to_workspace() {
        let report = report_for(&["--changed", "crates/fnp-ndarray/src/lib.rs"]);

        assert_eq!(report.target_crates, vec!["fnp-ndarray"]);
        assert!(!report.workspace_required);
        assert!(
            report
                .recipes
                .iter()
                .flat_map(|recipe| recipe.commands.iter())
                .all(|command| !command.contains("--workspace"))
        );
    }

    #[test]
    fn validation_recipe_selector_maps_phase2c_artifacts_to_packet_gate() {
        let report = report_for(&["--changed", "artifacts/phase2c/FNP-P2C-009/risk_note.md"]);

        let recipe = report
            .recipes
            .iter()
            .find(|recipe| recipe.id == "phase2c-packet-gate")
            .expect("packet recipe");
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("--packet-id FNP-P2C-009"))
        );
        assert!(
            report
                .recipes
                .iter()
                .any(|recipe| recipe.id == "phase2c-porting-ledger-freshness")
        );
        assert!(!report.workspace_required);
    }

    #[test]
    fn validation_recipe_selector_exposes_porting_ledger_freshness_for_control_ledger() {
        let report = report_for(&[
            "--changed",
            "artifacts/contracts/PORTING_ESSENCE_EXTRACTION_LEDGER_V1.md",
        ]);

        assert_eq!(report.risk_level, "manual");
        let recipe = report
            .recipes
            .iter()
            .find(|recipe| recipe.id == "phase2c-porting-ledger-freshness")
            .expect("porting ledger freshness recipe");
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("validate_phase2c_porting_ledger"))
        );
        assert!(recipe.commands.iter().any(|command| {
            command.contains("target/phase2c_porting_ledger_freshness_report.json")
        }));
        assert!(
            recipe
                .expected_reports
                .iter()
                .any(|path| path == "target/phase2c_porting_ledger_freshness_report.json")
        );
        assert!(
            recipe
                .prerequisites
                .iter()
                .any(|prereq| prereq.contains("stale anchor-only/open/partial"))
        );
    }

    #[test]
    fn validation_recipe_selector_keeps_generic_validation_labels_out_of_phase2c_freshness() {
        let report = report_for(&[
            "--changed",
            "crates/fnp-runtime/src/lib.rs",
            "--label",
            "validation",
        ]);

        assert!(
            report
                .recipes
                .iter()
                .all(|recipe| recipe.id != "phase2c-porting-ledger-freshness")
        );
    }

    #[test]
    fn validation_recipe_selector_keeps_high_risk_crates_from_suppressing_required_gates() {
        let report = report_for(&["--changed", "crates/fnp-io/src/lib.rs"]);
        let recipe = report
            .recipes
            .iter()
            .find(|recipe| recipe.id == "fnp-io-parser-diagnostics")
            .expect("fnp-io recipe");

        assert_eq!(report.risk_level, "high");
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("clippy -p fnp-io"))
        );
        assert!(
            recipe
                .commands
                .iter()
                .any(|command| command.contains("run_io_diagnostics"))
        );
    }
}
