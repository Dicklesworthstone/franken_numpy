#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const SHARD_MANIFEST_SCHEMA_VERSION: &str = "fnp-python-conformance-shards-v1";
pub const SHARD_REPORT_SCHEMA_VERSION: &str = "fnp-python-conformance-shard-report-v1";

const SMOKE_SUITES: &[&str] = &[
    "conformance_all",
    "conformance_array_equal",
    "conformance_asarray",
    "conformance_dtype_utils",
    "conformance_zeros_ones",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonConformanceManifest {
    pub schema_version: String,
    pub generated_by: String,
    pub suite_count: usize,
    pub shard_count: usize,
    pub shards: Vec<FnpPythonConformanceShard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonConformanceShard {
    pub id: String,
    pub domain: String,
    pub expected_cost: String,
    pub suite_count: usize,
    pub suites: Vec<FnpPythonConformanceSuite>,
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonConformanceSuite {
    pub suite_name: String,
    pub test_file: String,
    pub domain: String,
    pub expected_cost: String,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestDiagnostic {
    pub severity: String,
    pub reason_code: String,
    pub message: String,
    pub suite_name: Option<String>,
}

#[derive(Debug, Clone)]
struct ShardBuilder {
    id: String,
    domain: String,
    expected_cost: String,
    suites: Vec<FnpPythonConformanceSuite>,
}

impl ShardBuilder {
    fn new(id: String, domain: String, expected_cost: String) -> Self {
        Self {
            id,
            domain,
            expected_cost,
            suites: Vec::new(),
        }
    }

    fn push(&mut self, suite: FnpPythonConformanceSuite) {
        if cost_rank(&suite.expected_cost) > cost_rank(&self.expected_cost) {
            self.expected_cost.clone_from(&suite.expected_cost);
        }
        self.suites.push(suite);
    }

    fn build(mut self) -> FnpPythonConformanceShard {
        self.suites.sort_by(|lhs, rhs| {
            lhs.suite_name
                .cmp(&rhs.suite_name)
                .then_with(|| lhs.test_file.cmp(&rhs.test_file))
        });
        let commands = self
            .suites
            .iter()
            .map(|suite| suite.command.clone())
            .collect::<Vec<_>>();
        FnpPythonConformanceShard {
            id: self.id,
            domain: self.domain,
            expected_cost: self.expected_cost,
            suite_count: self.suites.len(),
            suites: self.suites,
            commands,
        }
    }
}

#[must_use]
pub fn default_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn build_manifest(repo_root: &Path) -> Result<FnpPythonConformanceManifest, String> {
    let suites = discover_suites(repo_root)?;
    Ok(build_manifest_from_suites(suites))
}

pub fn build_manifest_from_suites(
    suites: Vec<FnpPythonConformanceSuite>,
) -> FnpPythonConformanceManifest {
    let mut builders: BTreeMap<String, ShardBuilder> = BTreeMap::new();
    for suite in suites {
        let shard_id = shard_id_for_suite(&suite.suite_name, &suite.domain);
        builders
            .entry(shard_id.clone())
            .or_insert_with(|| {
                ShardBuilder::new(shard_id, suite.domain.clone(), suite.expected_cost.clone())
            })
            .push(suite);
    }

    let shards = builders
        .into_values()
        .map(ShardBuilder::build)
        .collect::<Vec<_>>();
    let suite_count = shards.iter().map(|shard| shard.suite_count).sum();
    FnpPythonConformanceManifest {
        schema_version: SHARD_MANIFEST_SCHEMA_VERSION.to_string(),
        generated_by: "fnp-conformance".to_string(),
        suite_count,
        shard_count: shards.len(),
        shards,
    }
}

pub fn discover_suites(repo_root: &Path) -> Result<Vec<FnpPythonConformanceSuite>, String> {
    let tests_dir = repo_root.join("crates/fnp-python/tests");
    let entries = fs::read_dir(&tests_dir)
        .map_err(|err| format!("failed reading {}: {err}", tests_dir.display()))?;

    let mut suites = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed reading tests dir entry: {err}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.starts_with("conformance_") || !file_name.ends_with(".rs") {
            continue;
        }
        let Some(suite_name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let (domain, expected_cost) = classify_suite(suite_name);
        suites.push(FnpPythonConformanceSuite {
            suite_name: suite_name.to_string(),
            test_file: relative_path(repo_root, &path),
            domain: domain.to_string(),
            expected_cost: expected_cost.to_string(),
            command: cargo_test_command(suite_name, true).join(" "),
        });
    }

    suites.sort_by(|lhs, rhs| lhs.suite_name.cmp(&rhs.suite_name));
    Ok(suites)
}

#[must_use]
pub fn cargo_test_command(suite_name: &str, use_rch: bool) -> Vec<String> {
    let cargo_args = [
        "cargo",
        "test",
        "-p",
        "fnp-python",
        "--test",
        suite_name,
        "--",
        "--nocapture",
    ];
    if use_rch {
        ["rch", "exec", "--"]
            .into_iter()
            .chain(cargo_args)
            .map(str::to_string)
            .collect()
    } else {
        cargo_args.into_iter().map(str::to_string).collect()
    }
}

#[must_use]
pub fn classify_suite(suite_name: &str) -> (&'static str, &'static str) {
    if SMOKE_SUITES.contains(&suite_name) {
        return ("smoke", "smoke");
    }

    let family = suite_name.trim_start_matches("conformance_");
    if family.contains("linalg") {
        ("linalg", "heavy")
    } else if family.contains("random") {
        ("random", "heavy")
    } else if family.contains("fft") {
        ("fft", "heavy")
    } else if family.contains("einsum") || family.contains("tensor") {
        ("tensor", "heavy")
    } else if family.contains("polynomial") || family.contains("poly_") || family == "poly_ops" {
        ("polynomial", "heavy")
    } else if family.contains("histogram") {
        ("histogram", "medium")
    } else if family == "io" {
        ("io", "medium")
    } else if family.contains("dtype") {
        ("dtype", "medium")
    } else if family.contains("ma") || family.contains("masked") {
        ("masked-array", "medium")
    } else if is_reduction_family(family) {
        ("reductions", "medium")
    } else if is_shape_family(family) {
        ("shape-array", "medium")
    } else if is_indexing_family(family) {
        ("indexing-searching", "medium")
    } else if is_math_family(family) {
        ("ufunc-math", "medium")
    } else {
        ("core", "smoke")
    }
}

pub fn validate_manifest(
    manifest: &FnpPythonConformanceManifest,
    discovered_suite_names: &BTreeSet<String>,
) -> Vec<ManifestDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut seen = BTreeSet::new();
    for shard in &manifest.shards {
        for suite in &shard.suites {
            if !seen.insert(suite.suite_name.clone()) {
                diagnostics.push(ManifestDiagnostic {
                    severity: "error".to_string(),
                    reason_code: "duplicate_suite".to_string(),
                    message: format!(
                        "suite '{}' appears in more than one shard",
                        suite.suite_name
                    ),
                    suite_name: Some(suite.suite_name.clone()),
                });
            }
            if !discovered_suite_names.contains(&suite.suite_name) {
                diagnostics.push(ManifestDiagnostic {
                    severity: "error".to_string(),
                    reason_code: "stale_suite".to_string(),
                    message: format!(
                        "suite '{}' is present in the manifest but not on disk",
                        suite.suite_name
                    ),
                    suite_name: Some(suite.suite_name.clone()),
                });
            }
        }
    }

    for suite_name in discovered_suite_names {
        if !seen.contains(suite_name) {
            diagnostics.push(ManifestDiagnostic {
                severity: "error".to_string(),
                reason_code: "missing_suite".to_string(),
                message: format!("suite '{suite_name}' is missing from the manifest"),
                suite_name: Some(suite_name.clone()),
            });
        }
    }

    diagnostics
}

#[must_use]
pub fn discovered_suite_names(manifest: &FnpPythonConformanceManifest) -> BTreeSet<String> {
    manifest
        .shards
        .iter()
        .flat_map(|shard| shard.suites.iter())
        .map(|suite| suite.suite_name.clone())
        .collect()
}

pub fn select_shards<'a>(
    manifest: &'a FnpPythonConformanceManifest,
    selector: &str,
) -> Result<Vec<&'a FnpPythonConformanceShard>, String> {
    if selector == "all" {
        return Ok(manifest.shards.iter().collect());
    }
    let selected = manifest
        .shards
        .iter()
        .find(|shard| shard.id == selector)
        .ok_or_else(|| format!("unknown shard '{selector}'"))?;
    Ok(vec![selected])
}

#[must_use]
pub fn shard_ids(manifest: &FnpPythonConformanceManifest) -> Vec<String> {
    manifest
        .shards
        .iter()
        .map(|shard| shard.id.clone())
        .collect()
}

fn shard_id_for_suite(suite_name: &str, domain: &str) -> String {
    if SMOKE_SUITES.contains(&suite_name) {
        "fnp-python-smoke".to_string()
    } else {
        format!("fnp-python-{domain}")
    }
}

fn relative_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn is_reduction_family(family: &str) -> bool {
    family.contains("sum")
        || family.contains("prod")
        || family.contains("mean")
        || family.contains("std")
        || family.contains("var")
        || family.contains("max")
        || family.contains("min")
        || family.contains("ptp")
        || family.contains("any")
        || family.contains("all")
        || family.contains("cumsum")
        || family.contains("cumulative")
        || family.contains("nan_funcs")
}

fn is_shape_family(family: &str) -> bool {
    family.contains("array_creation")
        || family.contains("array_manip")
        || family.contains("array_transform")
        || family.contains("atleast")
        || family.contains("block")
        || family.contains("concat")
        || family.contains("meshgrid")
        || family.contains("moveaxis")
        || family.contains("reshape")
        || family.contains("shape")
        || family.contains("split")
        || family.contains("stack")
        || family.contains("tile")
        || family.contains("trace")
        || family.contains("stride")
}

fn is_indexing_family(family: &str) -> bool {
    family.contains("arg")
        || family.contains("compress")
        || family.contains("extract")
        || family.contains("flatnonzero")
        || family.contains("index")
        || family.contains("lexsort")
        || family.contains("partition")
        || family.contains("search")
        || family.contains("setops")
        || family.contains("sort")
        || family.contains("take")
        || family.contains("unravel")
        || family.contains("where")
}

fn is_math_family(family: &str) -> bool {
    family.contains("angle")
        || family.contains("arithmetic")
        || family.contains("arctan")
        || family.contains("binary_math")
        || family.contains("bitwise")
        || family.contains("cbrt")
        || family.contains("close")
        || family.contains("comparison")
        || family.contains("complex")
        || family.contains("convolution")
        || family.contains("diff_gradient")
        || family.contains("divmod")
        || family.contains("dot")
        || family.contains("exp")
        || family.contains("fabs")
        || family.contains("float_power")
        || family.contains("fmax")
        || family.contains("fp_")
        || family.contains("frexp")
        || family.contains("gcd")
        || family.contains("heaviside")
        || family.contains("interp")
        || family.contains("isclose")
        || family.contains("isinf")
        || family.contains("log")
        || family.contains("nan_to_num")
        || family.contains("percentile")
        || family.contains("piecewise")
        || family.contains("range")
        || family.contains("reciprocal")
        || family.contains("round")
        || family.contains("sinc")
        || family.contains("special")
        || family.contains("statistics")
        || family.contains("trig")
        || family.contains("vector")
        || family.contains("window")
}

fn cost_rank(cost: &str) -> u8 {
    match cost {
        "heavy" => 2,
        "medium" => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FnpPythonConformanceManifest, FnpPythonConformanceShard, FnpPythonConformanceSuite,
        build_manifest_from_suites, classify_suite, validate_manifest,
    };
    use std::collections::BTreeSet;

    #[test]
    fn manifest_json_round_trips() {
        let manifest = build_manifest_from_suites(vec![suite("conformance_all")]);
        let raw = serde_json::to_string(&manifest).expect("serialize manifest");
        let parsed: FnpPythonConformanceManifest =
            serde_json::from_str(&raw).expect("parse manifest");
        assert_eq!(parsed.schema_version, "fnp-python-conformance-shards-v1");
        assert_eq!(parsed.suite_count, 1);
        assert_eq!(parsed.shards[0].id, "fnp-python-smoke");
    }

    #[test]
    fn validation_flags_duplicate_suite() {
        let manifest = FnpPythonConformanceManifest {
            schema_version: "fnp-python-conformance-shards-v1".to_string(),
            generated_by: "test".to_string(),
            suite_count: 2,
            shard_count: 2,
            shards: vec![
                shard("fnp-python-smoke", vec![suite("conformance_all")]),
                shard("fnp-python-core", vec![suite("conformance_all")]),
            ],
        };
        let diagnostics = validate_manifest(&manifest, &set(["conformance_all"].into_iter()));
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.reason_code == "duplicate_suite")
        );
    }

    #[test]
    fn validation_flags_missing_suite() {
        let manifest = build_manifest_from_suites(vec![suite("conformance_all")]);
        let diagnostics = validate_manifest(
            &manifest,
            &set(["conformance_all", "conformance_linalg"].into_iter()),
        );
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.reason_code == "missing_suite")
        );
    }

    #[test]
    fn validation_flags_stale_suite() {
        let manifest = build_manifest_from_suites(vec![suite("conformance_missing")]);
        let diagnostics = validate_manifest(&manifest, &set(["conformance_all"].into_iter()));
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.reason_code == "stale_suite")
        );
    }

    #[test]
    fn classification_keeps_high_cost_domains_separate() {
        assert_eq!(
            classify_suite("conformance_linalg_advanced"),
            ("linalg", "heavy")
        );
        assert_eq!(classify_suite("conformance_fft"), ("fft", "heavy"));
        assert_eq!(
            classify_suite("conformance_dtype_promotion"),
            ("dtype", "medium")
        );
    }

    fn suite(name: &str) -> FnpPythonConformanceSuite {
        let (domain, expected_cost) = classify_suite(name);
        FnpPythonConformanceSuite {
            suite_name: name.to_string(),
            test_file: format!("crates/fnp-python/tests/{name}.rs"),
            domain: domain.to_string(),
            expected_cost: expected_cost.to_string(),
            command: format!("rch exec -- cargo test -p fnp-python --test {name} -- --nocapture"),
        }
    }

    fn shard(id: &str, suites: Vec<FnpPythonConformanceSuite>) -> FnpPythonConformanceShard {
        FnpPythonConformanceShard {
            id: id.to_string(),
            domain: "test".to_string(),
            expected_cost: "smoke".to_string(),
            suite_count: suites.len(),
            commands: suites.iter().map(|suite| suite.command.clone()).collect(),
            suites,
        }
    }

    fn set<'a>(items: impl Iterator<Item = &'a str>) -> BTreeSet<String> {
        items.map(str::to_string).collect()
    }
}
