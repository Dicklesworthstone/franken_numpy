#![forbid(unsafe_code)]

use fnp_conformance::diagnostic_oracle::{
    DiagnosticCase, DiagnosticCaseVerdict, DiagnosticExpectation, DiagnosticMode,
    DiagnosticOracleOptions, DiagnosticOracleReport, DiagnosticOutcome, DiagnosticRequirementLevel,
    evaluate_diagnostic_report, run_diagnostic_oracle,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const IO_DIAGNOSTIC_SUITE_VERSION: &str = "io-diagnostics-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoDiagnosticRun {
    pub suite_version: String,
    pub case_count: usize,
    pub surface_summary: Vec<IoDiagnosticSurfaceSummary>,
    pub report: DiagnosticOracleReport,
    pub verdicts: Vec<DiagnosticCaseVerdict>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoDiagnosticSurfaceSummary {
    pub surface: String,
    pub case_count: usize,
    pub requirement_counts: BTreeMap<String, usize>,
}

pub fn run_io_diagnostics(options: &DiagnosticOracleOptions) -> Result<IoDiagnosticRun, String> {
    let cases = io_diagnostic_cases();
    let report = run_diagnostic_oracle(&cases, options)
        .map_err(|err| format!("oracle execution failed before IO target comparison: {err}"))?;
    let verdicts = evaluate_diagnostic_report(&cases, &report);
    Ok(IoDiagnosticRun {
        suite_version: IO_DIAGNOSTIC_SUITE_VERSION.to_string(),
        case_count: cases.len(),
        surface_summary: io_diagnostic_surface_summary(&cases),
        report,
        verdicts,
    })
}

#[must_use]
pub fn io_diagnostic_cases() -> Vec<DiagnosticCase> {
    vec![
        exception_case(
            "io_npy_load_empty_bytes",
            "fnp-io.npy.load",
            "import io\nnp.load(io.BytesIO(b''), allow_pickle=False)",
            "EOFError",
            &["No data"],
        ),
        exception_case(
            "io_npy_load_bad_magic_pickle_disallowed",
            "fnp-io.npy.load",
            "import io\nnp.load(io.BytesIO(b'not-npy'), allow_pickle=False)",
            "ValueError",
            &["pickled", "allow_pickle"],
        ),
        exception_case(
            "io_npy_load_truncated_magic_pickle_disallowed",
            "fnp-io.npy.load",
            "import io\nnp.load(io.BytesIO(b'\\x93NUMP'), allow_pickle=False)",
            "ValueError",
            &["pickled", "allow_pickle"],
        ),
        exception_case(
            "io_npy_load_object_pickle_disallowed",
            "fnp-io.npy.load",
            "import io\nbuf = io.BytesIO()\nnp.save(buf, np.array([{'a': 1}], dtype=object))\nbuf.seek(0)\nnp.load(buf, allow_pickle=False)",
            "ValueError",
            &["Object arrays", "allow_pickle=False"],
        ),
        exception_case(
            "io_npy_save_object_pickle_disallowed",
            "fnp-io.npy.save",
            "import io\nbuf = io.BytesIO()\nnp.save(buf, np.array([{'a': 1}], dtype=object), allow_pickle=False)",
            "ValueError",
            &["Object arrays", "allow_pickle=False"],
        ),
        exception_case(
            "io_npz_load_bad_zip",
            "fnp-io.npz.load",
            "import io\nnp.load(io.BytesIO(b'PK\\x03\\x04bad'), allow_pickle=False)",
            "BadZipFile",
            &["not a zip file"],
        ),
        exception_case(
            "io_npz_load_truncated_eocd",
            "fnp-io.npz.load",
            "import io\nnp.load(io.BytesIO(b'PK\\x05\\x06' + b'\\x00' * 10), allow_pickle=False)",
            "BadZipFile",
            &["not a zip file"],
        ),
        exception_case(
            "io_npz_missing_member",
            "fnp-io.npz.member",
            "import io\nbuf = io.BytesIO()\nnp.savez(buf, x=np.array([1]))\nbuf.seek(0)\narchive = np.load(buf, allow_pickle=False)\narchive['missing']",
            "KeyError",
            &["missing"],
        ),
        exception_case(
            "io_npz_object_member_pickle_disallowed",
            "fnp-io.npz.member",
            "import io\nbuf = io.BytesIO()\nnp.savez(buf, x=np.array([{'a': 1}], dtype=object))\nbuf.seek(0)\narchive = np.load(buf, allow_pickle=False)\narchive['x']",
            "ValueError",
            &["Object arrays", "allow_pickle=False"],
        ),
        exception_case(
            "io_npz_savez_duplicate_name",
            "fnp-io.npz.savez",
            "import io\nbuf = io.BytesIO()\nnp.savez(buf, x=np.array([1]), **{'x': np.array([2])})",
            "TypeError",
            &["multiple values", "x"],
        ),
        exception_case(
            "io_text_loadtxt_bad_token",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.StringIO('1 2\\n3 x'))",
            "ValueError",
            &["could not convert", "row 1"],
        ),
        exception_case(
            "io_text_loadtxt_ragged_rows",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.StringIO('1 2\\n3 4 5'))",
            "ValueError",
            &["columns changed", "row 2"],
        ),
        exception_case(
            "io_text_loadtxt_bad_usecols",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.StringIO('1 2\\n3 4'), usecols=[2])",
            "ValueError",
            &["invalid column index", "2"],
        ),
        exception_case(
            "io_text_loadtxt_negative_skiprows",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.StringIO('1 2'), skiprows=-1)",
            "ValueError",
            &["nonnegative"],
        ),
        exception_case(
            "io_text_loadtxt_negative_max_rows",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.StringIO('1 2'), max_rows=-1)",
            "ValueError",
            &["nonnegative"],
        ),
        exception_case(
            "io_text_loadtxt_bad_encoding",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.BytesIO(b'1 2'), encoding='definitely-not-an-encoding')",
            "LookupError",
            &["unknown encoding"],
        ),
        exception_case(
            "io_text_loadtxt_invalid_ndmin",
            "fnp-io.text.loadtxt",
            "import io\nnp.loadtxt(io.StringIO('1 2'), ndmin=4)",
            "ValueError",
            &["ndmin", "4"],
        ),
        exception_case(
            "io_text_genfromtxt_ragged_invalid",
            "fnp-io.text.genfromtxt",
            "import io\nnp.genfromtxt(io.StringIO('1 2\\n3 4 5'), invalid_raise=True)",
            "ValueError",
            &["Some errors"],
        ),
        exception_case(
            "io_text_genfromtxt_invalid_usecols",
            "fnp-io.text.genfromtxt",
            "import io\nnp.genfromtxt(io.StringIO('1 2'), usecols=[5])",
            "ValueError",
            &["Some errors"],
        ),
        exception_case(
            "io_text_genfromtxt_invalid_names",
            "fnp-io.text.genfromtxt",
            "import io\nnp.genfromtxt(io.StringIO('1 2'), names=['x'])",
            "ValueError",
            &["could not assign", "structure"],
        ),
        exception_case(
            "io_fromfile_missing_path",
            "fnp-io.fromfile",
            "np.fromfile('/definitely/not/here/franken-numpy', dtype=np.float64)",
            "FileNotFoundError",
            &["No such file"],
        ),
        exception_case(
            "io_frombuffer_bad_offset",
            "fnp-io.frombuffer",
            "np.frombuffer(b'abc', dtype=np.int16, offset=2)",
            "ValueError",
            &["buffer size", "element size"],
        ),
    ]
}

#[must_use]
pub fn io_diagnostic_surface_summary(cases: &[DiagnosticCase]) -> Vec<IoDiagnosticSurfaceSummary> {
    let mut counts = BTreeMap::<&str, BTreeMap<&str, usize>>::new();
    for case in cases {
        *counts
            .entry(case.surface.as_str())
            .or_default()
            .entry(requirement_key(case.requirement_level))
            .or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(surface, requirement_counts)| {
            let requirement_counts = requirement_counts
                .into_iter()
                .map(|(requirement, count)| (requirement.to_string(), count))
                .collect::<BTreeMap<_, _>>();
            IoDiagnosticSurfaceSummary {
                case_count: requirement_counts.values().sum(),
                surface: surface.to_string(),
                requirement_counts,
            }
        })
        .collect()
}

fn requirement_key(requirement_level: DiagnosticRequirementLevel) -> &'static str {
    match requirement_level {
        DiagnosticRequirementLevel::Must => "must",
        DiagnosticRequirementLevel::Should => "should",
        DiagnosticRequirementLevel::May => "may",
    }
}

fn exception_case(
    id: &str,
    surface: &str,
    python: &str,
    exception_class: &str,
    message_fragments: &[&str],
) -> DiagnosticCase {
    DiagnosticCase {
        id: id.to_string(),
        surface: surface.to_string(),
        requirement_level: DiagnosticRequirementLevel::Must,
        mode: DiagnosticMode::Strict,
        python: python.to_string(),
        expected: DiagnosticExpectation {
            outcome: DiagnosticOutcome::Exception,
            exception_class: Some(exception_class.to_string()),
            warning_categories: Vec::new(),
            message_fragments: to_strings(message_fragments),
        },
        version_guards: Vec::new(),
        intentional_divergence: None,
        exploratory: false,
    }
}

fn to_strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::{io_diagnostic_cases, io_diagnostic_surface_summary, run_io_diagnostics};
    use fnp_conformance::diagnostic_oracle::{
        DiagnosticOracleOptions, DiagnosticVerdictStatus, resolve_oracle_python,
    };
    use std::collections::BTreeSet;

    #[test]
    fn io_diagnostics_case_inventory_covers_required_surfaces() {
        let cases = io_diagnostic_cases();
        assert!(cases.len() >= 20);

        let ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), cases.len());

        for required_id in [
            "io_npy_load_object_pickle_disallowed",
            "io_npz_object_member_pickle_disallowed",
            "io_text_loadtxt_ragged_rows",
            "io_text_genfromtxt_invalid_usecols",
            "io_fromfile_missing_path",
        ] {
            assert!(ids.contains(required_id), "missing {required_id}");
        }

        let surfaces = cases
            .iter()
            .map(|case| case.surface.as_str())
            .collect::<BTreeSet<_>>();
        for prefix in [
            "fnp-io.npy",
            "fnp-io.npz",
            "fnp-io.text",
            "fnp-io.fromfile",
            "fnp-io.frombuffer",
        ] {
            assert!(
                surfaces.iter().any(|surface| surface.starts_with(prefix)),
                "missing surface prefix {prefix}"
            );
        }
    }

    #[test]
    fn io_diagnostics_surface_summary_maps_contract_sections() {
        let cases = io_diagnostic_cases();
        let summary = io_diagnostic_surface_summary(&cases);
        let total = summary
            .iter()
            .map(|surface| surface.case_count)
            .sum::<usize>();
        assert_eq!(total, cases.len());
        assert!(
            summary
                .iter()
                .any(|surface| surface.surface == "fnp-io.npy.load")
        );
        assert!(
            summary
                .iter()
                .any(|surface| surface.surface == "fnp-io.npz.member")
        );
    }

    #[test]
    fn io_diagnostics_oracle_report_matches_expectations() {
        let options = DiagnosticOracleOptions {
            python: resolve_oracle_python(),
            require_numpy: true,
        };
        let run = run_io_diagnostics(&options).expect("run IO diagnostics");
        assert_eq!(run.case_count, io_diagnostic_cases().len());
        assert!(
            run.report.diagnostics.is_empty(),
            "oracle execution diagnostics should be separate from target mismatches: {:?}",
            run.report.diagnostics
        );
        assert!(
            run.verdicts
                .iter()
                .all(|verdict| verdict.status != DiagnosticVerdictStatus::Fail),
            "{:?}",
            run.verdicts
        );
    }
}
