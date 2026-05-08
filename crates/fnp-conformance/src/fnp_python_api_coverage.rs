#![forbid(unsafe_code)]

use crate::fnp_python_conformance_shards::{
    FnpPythonConformanceManifest, FnpPythonConformanceSuite, build_manifest, default_repo_root,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const API_COVERAGE_SCHEMA_VERSION: &str = "fnp-python-api-coverage-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiCoverageReport {
    pub schema_version: String,
    pub generated_by: String,
    pub export_count: usize,
    pub covered_count: usize,
    pub missing_count: usize,
    pub excluded_count: usize,
    pub orphan_suite_count: usize,
    pub fail_on_missing: bool,
    pub exports: Vec<FnpPythonApiExportCoverage>,
    pub missing_exports: Vec<FnpPythonApiMissingExport>,
    pub excluded_exports: Vec<FnpPythonApiExcludedExport>,
    pub orphan_suites: Vec<FnpPythonApiOrphanSuite>,
}

impl FnpPythonApiCoverageReport {
    #[must_use]
    pub fn has_missing_exports(&self) -> bool {
        self.missing_count > 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiExportCoverage {
    pub name: String,
    pub source_kind: FnpPythonApiExportKind,
    pub source_file: String,
    pub status: FnpPythonApiCoverageStatus,
    pub evidence: Vec<FnpPythonApiCoverageEvidence>,
    pub exclusion: Option<FnpPythonApiExclusion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FnpPythonApiExportKind {
    Function,
    Class,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FnpPythonApiCoverageStatus {
    Covered,
    Missing,
    Excluded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiCoverageEvidence {
    pub evidence_kind: FnpPythonApiCoverageEvidenceKind,
    pub suite_name: Option<String>,
    pub test_file: String,
    pub shard_id: Option<String>,
    pub match_reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FnpPythonApiCoverageEvidenceKind {
    ConformanceSuite,
    GoldenArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiExclusion {
    pub reason_code: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiMissingExport {
    pub name: String,
    pub source_kind: FnpPythonApiExportKind,
    pub source_file: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiExcludedExport {
    pub name: String,
    pub source_kind: FnpPythonApiExportKind,
    pub reason_code: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FnpPythonApiOrphanSuite {
    pub suite_name: String,
    pub test_file: String,
    pub shard_id: Option<String>,
    pub reason_code: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnpPythonApiExport {
    pub name: String,
    pub source_kind: FnpPythonApiExportKind,
    pub source_file: String,
}

pub fn build_api_coverage_report(
    repo_root: &Path,
    fail_on_missing: bool,
) -> Result<FnpPythonApiCoverageReport, String> {
    let manifest = build_manifest(repo_root)?;
    let source_path = repo_root.join("crates/fnp-python/src/lib.rs");
    let source = fs::read_to_string(&source_path)
        .map_err(|err| format!("read {}: {err}", source_path.display()))?;
    let exports = extract_public_exports(&source);
    let suite_sources = read_suite_sources(repo_root, &manifest)?;
    let golden_sources = read_golden_sources(repo_root);
    Ok(build_api_coverage_report_from_inputs(
        exports,
        &manifest,
        &suite_sources,
        &golden_sources,
        fail_on_missing,
    ))
}

#[must_use]
pub fn default_report_path() -> PathBuf {
    default_repo_root().join("target/fnp_python_api_coverage.json")
}

#[must_use]
pub fn build_api_coverage_report_from_inputs(
    exports: Vec<FnpPythonApiExport>,
    manifest: &FnpPythonConformanceManifest,
    suite_sources: &BTreeMap<String, String>,
    golden_sources: &BTreeMap<String, String>,
    fail_on_missing: bool,
) -> FnpPythonApiCoverageReport {
    let shard_by_suite = shard_by_suite(manifest);
    let mut suite_hit_counts: BTreeMap<String, usize> = manifest
        .shards
        .iter()
        .flat_map(|shard| shard.suites.iter())
        .map(|suite| (suite.suite_name.clone(), 0usize))
        .collect();

    let mut rows = Vec::new();
    let mut missing_exports = Vec::new();
    let mut excluded_exports = Vec::new();

    for export in dedupe_exports(exports) {
        let evidence = coverage_evidence_for_export(
            &export.name,
            manifest,
            suite_sources,
            golden_sources,
            &shard_by_suite,
        );

        for item in &evidence {
            if let Some(suite_name) = &item.suite_name
                && let Some(count) = suite_hit_counts.get_mut(suite_name)
            {
                *count += 1;
            }
        }

        let exclusion = if evidence.is_empty() {
            default_exclusion_for(&export)
        } else {
            None
        };
        let status = if !evidence.is_empty() {
            FnpPythonApiCoverageStatus::Covered
        } else if exclusion.is_some() {
            FnpPythonApiCoverageStatus::Excluded
        } else {
            FnpPythonApiCoverageStatus::Missing
        };

        match status {
            FnpPythonApiCoverageStatus::Covered => {}
            FnpPythonApiCoverageStatus::Missing => {
                missing_exports.push(FnpPythonApiMissingExport {
                    name: export.name.clone(),
                    source_kind: export.source_kind,
                    source_file: export.source_file.clone(),
                    reason_code: "public_export_without_evidence".to_string(),
                });
            }
            FnpPythonApiCoverageStatus::Excluded => {
                let exclusion = exclusion.as_ref().expect("excluded export has exclusion");
                excluded_exports.push(FnpPythonApiExcludedExport {
                    name: export.name.clone(),
                    source_kind: export.source_kind,
                    reason_code: exclusion.reason_code.clone(),
                    reason: exclusion.reason.clone(),
                });
            }
        }

        rows.push(FnpPythonApiExportCoverage {
            name: export.name,
            source_kind: export.source_kind,
            source_file: export.source_file,
            status,
            evidence,
            exclusion,
        });
    }

    rows.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    missing_exports.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
    excluded_exports.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));

    let orphan_suites = manifest
        .shards
        .iter()
        .flat_map(|shard| {
            shard
                .suites
                .iter()
                .map(move |suite| (shard.id.clone(), suite))
        })
        .filter_map(|(shard_id, suite)| {
            if suite_hit_counts
                .get(&suite.suite_name)
                .copied()
                .unwrap_or_default()
                == 0
            {
                Some(FnpPythonApiOrphanSuite {
                    suite_name: suite.suite_name.clone(),
                    test_file: suite.test_file.clone(),
                    shard_id: Some(shard_id),
                    reason_code: "suite_has_no_export_reference".to_string(),
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let export_count = rows.len();
    let missing_count = missing_exports.len();
    let excluded_count = excluded_exports.len();
    let covered_count = rows
        .iter()
        .filter(|row| row.status == FnpPythonApiCoverageStatus::Covered)
        .count();

    FnpPythonApiCoverageReport {
        schema_version: API_COVERAGE_SCHEMA_VERSION.to_string(),
        generated_by: "fnp-conformance".to_string(),
        export_count,
        covered_count,
        missing_count,
        excluded_count,
        orphan_suite_count: orphan_suites.len(),
        fail_on_missing,
        exports: rows,
        missing_exports,
        excluded_exports,
        orphan_suites,
    }
}

#[must_use]
pub fn extract_public_exports(source: &str) -> Vec<FnpPythonApiExport> {
    let pyfunction_names = pyfunction_public_names(source);
    let pyclass_names = pyclass_public_names(source);
    let body = extract_fnp_python_body(source).unwrap_or(source);
    let mut exports = Vec::new();

    for rust_name in extract_registered_functions(body) {
        let name = pyfunction_names
            .get(&rust_name)
            .cloned()
            .unwrap_or_else(|| rust_name.clone());
        exports.push(FnpPythonApiExport {
            name,
            source_kind: FnpPythonApiExportKind::Function,
            source_file: "crates/fnp-python/src/lib.rs".to_string(),
        });
    }

    for rust_name in extract_registered_classes(body) {
        let name = pyclass_names
            .get(&rust_name)
            .cloned()
            .unwrap_or_else(|| rust_name.trim_start_matches("Py").to_string());
        exports.push(FnpPythonApiExport {
            name,
            source_kind: FnpPythonApiExportKind::Class,
            source_file: "crates/fnp-python/src/lib.rs".to_string(),
        });
    }

    for name in extract_registered_objects(body) {
        exports.push(FnpPythonApiExport {
            name,
            source_kind: FnpPythonApiExportKind::Object,
            source_file: "crates/fnp-python/src/lib.rs".to_string(),
        });
    }

    dedupe_exports(exports)
}

fn dedupe_exports(exports: Vec<FnpPythonApiExport>) -> Vec<FnpPythonApiExport> {
    let mut by_key = BTreeMap::new();
    for export in exports {
        by_key
            .entry((export.name.clone(), export.source_kind))
            .or_insert(export);
    }
    by_key.into_values().collect()
}

fn extract_fnp_python_body(source: &str) -> Option<&str> {
    let start = source.find("pub fn fnp_python")?;
    let open = source.get(start..)?.find('{')? + start;
    let test_module = source
        .get(open..)?
        .find("\n#[cfg(test)]")
        .map(|offset| open + offset)
        .unwrap_or(source.len());
    source.get(open + 1..test_module)
}

fn extract_registered_functions(body: &str) -> Vec<String> {
    let mut names = BTreeSet::new();
    let needle = "m.add_function(wrap_pyfunction!(";
    let mut cursor = 0usize;
    while let Some(tail) = body.get(cursor..) {
        let Some(found) = tail.find(needle) else {
            break;
        };
        let start = cursor + found + needle.len();
        let Some(rest) = body.get(start..) else {
            break;
        };
        let trimmed = rest.trim_start();
        let skipped = rest.len() - trimmed.len();
        let ident = take_identifier(trimmed);
        if !ident.is_empty() {
            names.insert(ident.to_string());
        }
        cursor = start + skipped + ident.len();
    }
    names.into_iter().collect()
}

fn extract_registered_classes(body: &str) -> Vec<String> {
    let mut names = BTreeSet::new();
    let needle = "m.add_class::<";
    let mut cursor = 0usize;
    while let Some(tail) = body.get(cursor..) {
        let Some(found) = tail.find(needle) else {
            break;
        };
        let start = cursor + found + needle.len();
        let Some(rest) = body.get(start..) else {
            break;
        };
        let ident = take_identifier(rest);
        if !ident.is_empty() {
            names.insert(ident.to_string());
        }
        cursor = start + ident.len();
    }
    names.into_iter().collect()
}

fn extract_registered_objects(body: &str) -> Vec<String> {
    let mut names = BTreeSet::new();
    let needle = "m.add(\"";
    let mut cursor = 0usize;
    while let Some(tail) = body.get(cursor..) {
        let Some(found) = tail.find(needle) else {
            break;
        };
        let start = cursor + found + needle.len();
        let Some(rest) = body.get(start..) else {
            break;
        };
        if let Some(end) = rest.find('"') {
            if let Some(name) = rest.get(..end) {
                names.insert(name.to_string());
            }
            cursor = start + end;
        } else {
            break;
        }
    }
    names.into_iter().collect()
}

fn take_identifier(source: &str) -> &str {
    let end = source
        .char_indices()
        .find_map(|(idx, ch)| {
            if ch == '_' || ch.is_ascii_alphanumeric() {
                None
            } else {
                Some(idx)
            }
        })
        .unwrap_or(source.len());
    source.get(..end).unwrap_or(source)
}

fn pyfunction_public_names(source: &str) -> BTreeMap<String, String> {
    let lines = source.lines().collect::<Vec<_>>();
    let mut names = BTreeMap::new();
    for (idx, line) in lines.iter().enumerate() {
        let Some(fn_name) = function_name_from_line(line) else {
            continue;
        };
        let attrs = attribute_block_before(&lines, idx);
        if !attrs.contains("pyfunction") {
            continue;
        }
        let public_name = extract_name_attribute(&attrs).unwrap_or_else(|| fn_name.clone());
        names.insert(fn_name, public_name);
    }
    names
}

fn pyclass_public_names(source: &str) -> BTreeMap<String, String> {
    let lines = source.lines().collect::<Vec<_>>();
    let mut names = BTreeMap::new();
    for (idx, line) in lines.iter().enumerate() {
        let Some(struct_name) = struct_name_from_line(line) else {
            continue;
        };
        let attrs = attribute_block_before(&lines, idx);
        if !attrs.contains("pyclass") {
            continue;
        }
        let public_name = extract_name_attribute(&attrs).unwrap_or_else(|| {
            struct_name
                .strip_prefix("Py")
                .unwrap_or(&struct_name)
                .to_string()
        });
        names.insert(struct_name, public_name);
    }
    names
}

fn attribute_block_before(lines: &[&str], idx: usize) -> String {
    let mut start = idx;
    while start > 0 {
        let Some(previous) = lines.get(start - 1).map(|line| line.trim()) else {
            break;
        };
        if previous.is_empty()
            || previous == "}"
            || previous.starts_with("fn ")
            || previous.starts_with("pub fn ")
        {
            break;
        }
        start -= 1;
    }
    lines.get(start..idx).unwrap_or_default().join("\n")
}

fn function_name_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let after_fn = trimmed.strip_prefix("fn ")?;
    Some(take_identifier(after_fn).to_string())
}

fn struct_name_from_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let after_struct = trimmed
        .strip_prefix("pub struct ")
        .or_else(|| trimmed.strip_prefix("struct "))?;
    Some(take_identifier(after_struct).to_string())
}

fn extract_name_attribute(attrs: &str) -> Option<String> {
    let needle = "name = \"";
    let start = attrs.rfind(needle)? + needle.len();
    let rest = attrs.get(start..)?;
    let end = rest.find('"')?;
    rest.get(..end).map(str::to_string)
}

fn read_suite_sources(
    repo_root: &Path,
    manifest: &FnpPythonConformanceManifest,
) -> Result<BTreeMap<String, String>, String> {
    let mut sources = BTreeMap::new();
    for suite in manifest.shards.iter().flat_map(|shard| shard.suites.iter()) {
        let path = repo_root.join(&suite.test_file);
        let source =
            fs::read_to_string(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
        sources.insert(suite.suite_name.clone(), source);
    }
    Ok(sources)
}

fn read_golden_sources(repo_root: &Path) -> BTreeMap<String, String> {
    [
        "crates/fnp-python/COMPLIANCE.generated.md",
        "crates/fnp-python/POLYNOMIAL_COMPLIANCE.generated.md",
        "crates/fnp-python/RANDOM_COMPLIANCE.generated.md",
        "crates/fnp-python/tests/golden_native_functions.rs",
    ]
    .into_iter()
    .filter_map(|relative| {
        let path = repo_root.join(relative);
        let source = fs::read_to_string(path).ok()?;
        Some((relative.to_string(), source))
    })
    .collect()
}

fn coverage_evidence_for_export(
    export_name: &str,
    manifest: &FnpPythonConformanceManifest,
    suite_sources: &BTreeMap<String, String>,
    golden_sources: &BTreeMap<String, String>,
    shard_by_suite: &BTreeMap<String, String>,
) -> Vec<FnpPythonApiCoverageEvidence> {
    let mut evidence = Vec::new();

    for suite in manifest.shards.iter().flat_map(|shard| shard.suites.iter()) {
        let Some(source) = suite_sources.get(&suite.suite_name) else {
            continue;
        };
        if source_references_export(source, export_name) {
            evidence.push(suite_evidence(
                suite,
                shard_by_suite,
                "direct_api_reference",
            ));
        }
    }

    for (path, source) in golden_sources {
        if golden_references_export(path, source, export_name) {
            evidence.push(FnpPythonApiCoverageEvidence {
                evidence_kind: FnpPythonApiCoverageEvidenceKind::GoldenArtifact,
                suite_name: None,
                test_file: path.clone(),
                shard_id: None,
                match_reason: "golden_artifact_reference".to_string(),
            });
        }
    }

    evidence.sort_by(|lhs, rhs| {
        lhs.test_file
            .cmp(&rhs.test_file)
            .then_with(|| lhs.suite_name.cmp(&rhs.suite_name))
    });
    evidence.dedup();
    evidence
}

fn suite_evidence(
    suite: &FnpPythonConformanceSuite,
    shard_by_suite: &BTreeMap<String, String>,
    match_reason: &str,
) -> FnpPythonApiCoverageEvidence {
    FnpPythonApiCoverageEvidence {
        evidence_kind: FnpPythonApiCoverageEvidenceKind::ConformanceSuite,
        suite_name: Some(suite.suite_name.clone()),
        test_file: suite.test_file.clone(),
        shard_id: shard_by_suite.get(&suite.suite_name).cloned(),
        match_reason: match_reason.to_string(),
    }
}

fn shard_by_suite(manifest: &FnpPythonConformanceManifest) -> BTreeMap<String, String> {
    manifest
        .shards
        .iter()
        .flat_map(|shard| {
            shard
                .suites
                .iter()
                .map(move |suite| (suite.suite_name.clone(), shard.id.clone()))
        })
        .collect()
}

fn source_references_export(source: &str, export_name: &str) -> bool {
    evidence_name_variants(export_name).iter().any(|name| {
        let direct_patterns = [
            format!("fnp.{name}"),
            format!(".{name}("),
            format!("module.getattr(\"{name}\")"),
            format!("module.getattr('{name}')"),
            format!("numpy.getattr(\"{name}\")"),
            format!("numpy.getattr('{name}')"),
            format!("getattr(\"{name}\")"),
            format!("getattr('{name}')"),
        ];
        if direct_patterns
            .iter()
            .any(|pattern| source.contains(pattern))
        {
            return true;
        }

        source.lines().any(|line| {
            let trimmed = line.trim();
            trimmed == format!("\"{name}\",")
                || trimmed == format!("'{name}',")
                || trimmed == format!("\"{name}\"")
                || trimmed == format!("'{name}'")
        })
    })
}

fn golden_references_export(path: &str, source: &str, export_name: &str) -> bool {
    evidence_name_variants(export_name).iter().any(|name| {
        if path.ends_with(".md") {
            return source.lines().any(|line| {
                line.contains(&format!("`{name}`"))
                    || line.contains(&format!("| `{name}` |"))
                    || (line.contains(&format!("`{name}_")) && line.contains("_matches_numpy"))
            });
        }
        source_references_export(source, name)
            || source.contains(&format!("golden_{name}_"))
            || source.contains(&format!("{name}_matches_numpy"))
    })
}

fn evidence_name_variants(name: &str) -> BTreeSet<String> {
    let mut variants = BTreeSet::from([name.to_string()]);
    if let Some(stripped) = name.strip_suffix('_')
        && !stripped.is_empty()
    {
        variants.insert(stripped.to_string());
    }
    variants
}

fn default_exclusion_for(export: &FnpPythonApiExport) -> Option<FnpPythonApiExclusion> {
    match export.source_kind {
        FnpPythonApiExportKind::Object => object_exclusion(&export.name),
        FnpPythonApiExportKind::Class => Some(FnpPythonApiExclusion {
            reason_code: "pyclass_surface_unit_tested".to_string(),
            reason: "PyO3 class construction and methods are covered by fnp-python unit tests rather than shard-level conformance suites.".to_string(),
        }),
        FnpPythonApiExportKind::Function => function_exclusion(&export.name),
    }
}

fn object_exclusion(name: &str) -> Option<FnpPythonApiExclusion> {
    let reason = match name {
        "__version__" => (
            "module_metadata",
            "module version metadata is checked by fnp-python unit tests, not a NumPy behavior conformance suite",
        ),
        "pi" | "e" | "euler_gamma" | "inf" | "nan" | "little_endian" => (
            "scalar_constant",
            "scalar constants are static value exports; value parity is checked in fnp-python unit tests and README evidence",
        ),
        "mgrid" | "ogrid" | "r_" | "c_" => (
            "index_singleton",
            "indexing singleton behavior is exercised through Python wrapper tests rather than shard-level API coverage",
        ),
        "random" | "polynomial" | "fft" | "linalg" | "ma" | "testing" | "exceptions" | "dtypes"
        | "lib" => (
            "namespace_aggregator",
            "namespace module itself is an aggregator; member functions carry the conformance evidence",
        ),
        _ => return None,
    };
    Some(FnpPythonApiExclusion {
        reason_code: reason.0.to_string(),
        reason: reason.1.to_string(),
    })
}

fn function_exclusion(name: &str) -> Option<FnpPythonApiExclusion> {
    let reason = if name.starts_with("testing_") {
        Some((
            "flat_namespace_alias",
            "flat numpy.testing alias; canonical nested testing surface carries conformance evidence",
        ))
    } else if name.starts_with("recfunctions_") {
        Some((
            "flat_namespace_alias",
            "flat numpy.lib.recfunctions alias; canonical nested recfunctions surface carries conformance evidence",
        ))
    } else if name.starts_with("scimath_") || name.starts_with("array_utils_") {
        Some((
            "flat_namespace_alias",
            "flat numpy.lib helper alias; canonical nested lib surface carries conformance evidence",
        ))
    } else if name.starts_with("linalg_") {
        Some((
            "flat_namespace_alias",
            "flat numpy.linalg alias; canonical nested linalg surface carries conformance evidence",
        ))
    } else {
        None
    }?;

    Some(FnpPythonApiExclusion {
        reason_code: reason.0.to_string(),
        reason: reason.1.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        FnpPythonApiCoverageStatus, FnpPythonApiExport, FnpPythonApiExportKind,
        build_api_coverage_report_from_inputs, extract_public_exports,
    };
    use crate::fnp_python_conformance_shards::{
        FnpPythonConformanceManifest, FnpPythonConformanceShard, FnpPythonConformanceSuite,
    };
    use std::collections::BTreeMap;

    #[test]
    fn extracts_top_level_exports_and_pyo3_name_overrides() {
        let source = r#"
            #[pyclass(name = "Nditer")]
            pub struct PyNditer;

            #[pyfunction]
            #[pyo3(name = "where")]
            fn where_py() {}

            #[pyfunction]
            fn add() {}

            #[pymodule]
            pub fn fnp_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
                m.add_class::<PyNditer>()?;
                m.add_function(wrap_pyfunction!(where_py, m)?)?;
                m.add_function(wrap_pyfunction!(
                    add,
                    m
                )?)?;
                m.add("__version__", "0.1.0")?;
                Ok(())
            }
        "#;
        let names = extract_public_exports(source)
            .into_iter()
            .map(|export| (export.name, export.source_kind))
            .collect::<Vec<_>>();
        assert!(names.contains(&("Nditer".to_string(), FnpPythonApiExportKind::Class)));
        assert!(names.contains(&("where".to_string(), FnpPythonApiExportKind::Function)));
        assert!(names.contains(&("add".to_string(), FnpPythonApiExportKind::Function)));
        assert!(names.contains(&("__version__".to_string(), FnpPythonApiExportKind::Object)));
    }

    #[test]
    fn report_marks_covered_missing_orphan_and_excluded_surfaces() {
        let exports = vec![
            export("covered", FnpPythonApiExportKind::Function),
            export("missing", FnpPythonApiExportKind::Function),
            export("__version__", FnpPythonApiExportKind::Object),
        ];
        let manifest = manifest(vec![
            suite(
                "conformance_covered",
                "crates/fnp-python/tests/conformance_covered.rs",
            ),
            suite(
                "conformance_orphan",
                "crates/fnp-python/tests/conformance_orphan.rs",
            ),
        ]);
        let suite_sources = BTreeMap::from([
            (
                "conformance_covered".to_string(),
                "result = fnp.covered([1, 2])".to_string(),
            ),
            ("conformance_orphan".to_string(), "assert True".to_string()),
        ]);
        let report = build_api_coverage_report_from_inputs(
            exports,
            &manifest,
            &suite_sources,
            &BTreeMap::new(),
            true,
        );

        assert_eq!(report.covered_count, 1);
        assert_eq!(report.missing_exports[0].name, "missing");
        assert_eq!(report.excluded_exports[0].name, "__version__");
        assert_eq!(report.orphan_suites[0].suite_name, "conformance_orphan");
        assert_eq!(
            report
                .exports
                .iter()
                .find(|row| row.name == "covered")
                .expect("covered row")
                .status,
            FnpPythonApiCoverageStatus::Covered
        );
    }

    #[test]
    fn golden_artifacts_cover_exports_without_suite_references() {
        let exports = vec![export("array_equal", FnpPythonApiExportKind::Function)];
        let manifest = manifest(vec![suite(
            "conformance_orphan",
            "crates/fnp-python/tests/conformance_orphan.rs",
        )]);
        let suite_sources =
            BTreeMap::from([("conformance_orphan".to_string(), "assert True".to_string())]);
        let golden_sources = BTreeMap::from([(
            "crates/fnp-python/COMPLIANCE.generated.md".to_string(),
            "| `array_equal_matches_numpy_across_shapes` | `constructor` | MUST |".to_string(),
        )]);
        let report = build_api_coverage_report_from_inputs(
            exports,
            &manifest,
            &suite_sources,
            &golden_sources,
            false,
        );

        assert_eq!(report.covered_count, 1);
        assert!(report.missing_exports.is_empty());
    }

    fn export(name: &str, source_kind: FnpPythonApiExportKind) -> FnpPythonApiExport {
        FnpPythonApiExport {
            name: name.to_string(),
            source_kind,
            source_file: "crates/fnp-python/src/lib.rs".to_string(),
        }
    }

    fn manifest(suites: Vec<FnpPythonConformanceSuite>) -> FnpPythonConformanceManifest {
        FnpPythonConformanceManifest {
            schema_version: "fnp-python-conformance-shards-v1".to_string(),
            generated_by: "test".to_string(),
            suite_count: suites.len(),
            shard_count: 1,
            shards: vec![FnpPythonConformanceShard {
                id: "fnp-python-test".to_string(),
                domain: "test".to_string(),
                expected_cost: "smoke".to_string(),
                suite_count: suites.len(),
                commands: suites.iter().map(|suite| suite.command.clone()).collect(),
                suites,
            }],
        }
    }

    fn suite(name: &str, test_file: &str) -> FnpPythonConformanceSuite {
        FnpPythonConformanceSuite {
            suite_name: name.to_string(),
            test_file: test_file.to_string(),
            domain: "test".to_string(),
            expected_cost: "smoke".to_string(),
            command: format!("rch exec -- cargo test -p fnp-python --test {name}"),
        }
    }
}
