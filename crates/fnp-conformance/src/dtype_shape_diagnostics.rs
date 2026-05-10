#![forbid(unsafe_code)]

use crate::diagnostic_oracle::{
    DiagnosticCase, DiagnosticCaseVerdict, DiagnosticExpectation, DiagnosticMode,
    DiagnosticOracleOptions, DiagnosticOracleReport, DiagnosticOutcome, DiagnosticRequirementLevel,
    DiagnosticVersionGuard, evaluate_diagnostic_report, run_diagnostic_oracle,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DTYPE_SHAPE_DIAGNOSTIC_SUITE_VERSION: &str = "dtype-shape-diagnostics-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DTypeShapeDiagnosticRun {
    pub suite_version: String,
    pub case_count: usize,
    pub surface_summary: Vec<DiagnosticSurfaceSummary>,
    pub report: DiagnosticOracleReport,
    pub verdicts: Vec<DiagnosticCaseVerdict>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSurfaceSummary {
    pub surface: String,
    pub case_count: usize,
    pub requirement_counts: BTreeMap<String, usize>,
}

pub fn run_dtype_shape_diagnostics(
    options: &DiagnosticOracleOptions,
) -> Result<DTypeShapeDiagnosticRun, String> {
    let cases = dtype_shape_diagnostic_cases();
    let report = run_diagnostic_oracle(&cases, options)
        .map_err(|err| format!("oracle execution failed before target comparison: {err}"))?;
    let verdicts = evaluate_diagnostic_report(&cases, &report);
    Ok(DTypeShapeDiagnosticRun {
        suite_version: DTYPE_SHAPE_DIAGNOSTIC_SUITE_VERSION.to_string(),
        case_count: cases.len(),
        surface_summary: dtype_shape_surface_summary(&cases),
        report,
        verdicts,
    })
}

#[must_use]
pub fn dtype_shape_diagnostic_cases() -> Vec<DiagnosticCase> {
    vec![
        exception_case(
            "dtype_astype_float64_to_int64_safe",
            "fnp-dtype.cast_policy",
            "np.array([1.5], dtype=np.float64).astype(np.int64, casting='safe')",
            "TypeError",
            &["Cannot cast", "float64", "int64", "safe"],
        ),
        exception_case(
            "dtype_astype_int32_to_float64_no",
            "fnp-dtype.cast_policy",
            "np.array([1], dtype=np.int32).astype(np.float64, casting='no')",
            "TypeError",
            &["Cannot cast", "int32", "float64", "no"],
        ),
        exception_case(
            "dtype_astype_uint64_to_int64_safe",
            "fnp-dtype.cast_policy",
            "np.array([1], dtype=np.uint64).astype(np.int64, casting='safe')",
            "TypeError",
            &["Cannot cast", "uint64", "int64", "safe"],
        ),
        exception_case(
            "dtype_invalid_name",
            "fnp-dtype.dtype_parser",
            "np.dtype('not-a-real-dtype')",
            "TypeError",
            &["not-a-real-dtype", "not understood"],
        ),
        exception_case(
            "dtype_duplicate_field_name",
            "fnp-dtype.structured_fields",
            "np.dtype([('x', 'i4'), ('x', 'i4')])",
            "ValueError",
            &["field", "occurs more than once"],
        ),
        warning_case(
            "dtype_complex_to_float_cast_warning",
            "fnp-dtype.cast_policy",
            "np.array([1+2j]).astype(np.float64)",
            &["ComplexWarning"],
            &["Casting complex values to real"],
        ),
        versioned_exception_case(
            "dtype_promote_datetime_complex",
            "fnp-dtype.promotion",
            "np.promote_types(np.dtype('datetime64[D]'), np.dtype('complex128'))",
            "DTypePromotionError",
            &["do not have a common DType"],
            "2.0.0",
            "NumPy 2.x exposes this promotion failure as DTypePromotionError.",
        ),
        success_case(
            "dtype_result_type_unicode_int",
            "fnp-dtype.promotion",
            "np.result_type(np.array(['a'], dtype='U1'), np.array([1], dtype='int64'))",
        ),
        exception_case(
            "dtype_invalid_casting_mode",
            "fnp-dtype.cast_policy",
            "np.array([1]).astype(np.float64, casting='not_a_mode')",
            "ValueError",
            &["casting must be one of", "not_a_mode"],
        ),
        exception_case(
            "dtype_invalid_byte_order_token",
            "fnp-dtype.dtype_parser",
            "np.dtype('>')",
            "TypeError",
            &[">", "not understood"],
        ),
        exception_case(
            "shape_empty_negative_dimension",
            "fnp-ndarray.shape",
            "np.empty((-1,))",
            "ValueError",
            &["negative dimensions"],
        ),
        exception_case(
            "shape_overflowed_empty_dimension",
            "fnp-ndarray.shape",
            "np.empty((2**63,), dtype=np.uint8)",
            "ValueError",
            &["Maximum allowed dimension"],
        ),
        exception_case(
            "shape_reshape_multiple_unknown_dimensions",
            "fnp-ndarray.reshape",
            "np.array([1, 2, 3]).reshape(-1, -1)",
            "ValueError",
            &["one unknown dimension"],
        ),
        exception_case(
            "shape_reshape_incompatible_unknown_dimension",
            "fnp-ndarray.reshape",
            "np.array([1, 2, 3]).reshape(2, -1)",
            "ValueError",
            &["cannot reshape", "size 3"],
        ),
        exception_case(
            "shape_broadcast_shapes_incompatible",
            "fnp-ndarray.broadcast",
            "np.broadcast_shapes((2, 3), (4, 3))",
            "ValueError",
            &["shape mismatch", "(2, 3)", "(4, 3)"],
        ),
        exception_case(
            "shape_broadcast_to_incompatible",
            "fnp-ndarray.broadcast",
            "np.broadcast_to(np.array([1, 2, 3]), (2, 2))",
            "ValueError",
            &["could not be broadcast", "requested shape"],
        ),
        exception_case(
            "shape_ndarray_buffer_too_small_for_strides",
            "fnp-ndarray.strides",
            "np.ndarray(shape=(2, 2), dtype=np.int64, buffer=bytearray(8), strides=(8, 8))",
            "ValueError",
            &["strides", "size of buffer"],
        ),
        exception_case(
            "shape_as_strided_stride_rank_mismatch",
            "fnp-ndarray.strides",
            "np.lib.stride_tricks.as_strided(np.array([1, 2, 3]), shape=(2,), strides=(8, 8))",
            "ValueError",
            &["mismatch", "strides", "shape"],
        ),
        exception_case(
            "shape_index_out_of_bounds",
            "fnp-ndarray.indexing",
            "np.array([1, 2, 3])[5]",
            "IndexError",
            &["out of bounds", "axis 0", "size 3"],
        ),
        exception_case(
            "shape_take_out_of_bounds_raise",
            "fnp-ndarray.indexing",
            "np.take(np.array([1, 2, 3]), [3], mode='raise')",
            "IndexError",
            &["out of bounds", "axis 0", "size 3"],
        ),
        success_case(
            "shape_scalar_reshape_empty_shape",
            "fnp-ndarray.reshape",
            "np.reshape(np.array(7), ())",
        ),
        success_case(
            "shape_empty_shape_broadcast_success",
            "fnp-ndarray.broadcast",
            "np.broadcast_shapes((), (3,))",
        ),
    ]
}

#[must_use]
pub fn dtype_shape_surface_summary(cases: &[DiagnosticCase]) -> Vec<DiagnosticSurfaceSummary> {
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
            DiagnosticSurfaceSummary {
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

fn success_case(id: &str, surface: &str, python: &str) -> DiagnosticCase {
    DiagnosticCase {
        id: id.to_string(),
        surface: surface.to_string(),
        requirement_level: DiagnosticRequirementLevel::Must,
        mode: DiagnosticMode::Strict,
        python: python.to_string(),
        expected: DiagnosticExpectation {
            outcome: DiagnosticOutcome::Success,
            exception_class: None,
            warning_categories: Vec::new(),
            message_fragments: Vec::new(),
        },
        version_guards: Vec::new(),
        intentional_divergence: None,
        exploratory: false,
    }
}

fn exception_case(
    id: &str,
    surface: &str,
    python: &str,
    exception_class: &str,
    message_fragments: &[&str],
) -> DiagnosticCase {
    let mut case = success_case(id, surface, python);
    case.expected = DiagnosticExpectation {
        outcome: DiagnosticOutcome::Exception,
        exception_class: Some(exception_class.to_string()),
        warning_categories: Vec::new(),
        message_fragments: to_strings(message_fragments),
    };
    case
}

fn versioned_exception_case(
    id: &str,
    surface: &str,
    python: &str,
    exception_class: &str,
    message_fragments: &[&str],
    min_numpy: &str,
    reason: &str,
) -> DiagnosticCase {
    let mut case = exception_case(id, surface, python, exception_class, message_fragments);
    case.version_guards.push(DiagnosticVersionGuard {
        package: "numpy".to_string(),
        min_inclusive: Some(min_numpy.to_string()),
        max_exclusive: None,
        reason: reason.to_string(),
    });
    case
}

fn warning_case(
    id: &str,
    surface: &str,
    python: &str,
    warning_categories: &[&str],
    message_fragments: &[&str],
) -> DiagnosticCase {
    let mut case = success_case(id, surface, python);
    case.expected.warning_categories = to_strings(warning_categories);
    case.expected.message_fragments = to_strings(message_fragments);
    case
}

fn to_strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        dtype_shape_diagnostic_cases, dtype_shape_surface_summary, run_dtype_shape_diagnostics,
    };
    use crate::diagnostic_oracle::{
        DiagnosticOracleOptions, DiagnosticVerdictStatus, resolve_oracle_python,
    };
    use std::collections::BTreeSet;

    #[test]
    fn dtype_shape_diagnostics_case_inventory_covers_required_surfaces() {
        let cases = dtype_shape_diagnostic_cases();
        assert!(cases.len() >= 20);

        let ids = cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), cases.len());

        for required_id in [
            "shape_empty_negative_dimension",
            "shape_overflowed_empty_dimension",
            "shape_reshape_multiple_unknown_dimensions",
            "shape_broadcast_shapes_incompatible",
            "dtype_invalid_casting_mode",
        ] {
            assert!(ids.contains(required_id), "missing {required_id}");
        }

        let surfaces = cases
            .iter()
            .map(|case| case.surface.as_str())
            .collect::<BTreeSet<_>>();
        assert!(
            surfaces
                .iter()
                .any(|surface| surface.starts_with("fnp-dtype"))
        );
        assert!(
            surfaces
                .iter()
                .any(|surface| surface.starts_with("fnp-ndarray"))
        );
    }

    #[test]
    fn dtype_shape_diagnostics_surface_summary_maps_contract_sections() {
        let cases = dtype_shape_diagnostic_cases();
        let summary = dtype_shape_surface_summary(&cases);
        let total = summary
            .iter()
            .map(|surface| surface.case_count)
            .sum::<usize>();
        assert_eq!(total, cases.len());
        assert!(
            summary
                .iter()
                .any(|surface| surface.surface == "fnp-dtype.cast_policy")
        );
        assert!(
            summary
                .iter()
                .any(|surface| surface.surface == "fnp-ndarray.broadcast")
        );
    }

    #[test]
    fn dtype_shape_diagnostics_oracle_report_matches_expectations() {
        let options = DiagnosticOracleOptions {
            python: resolve_oracle_python(),
            require_numpy: true,
        };
        let run = run_dtype_shape_diagnostics(&options).expect("run dtype shape diagnostics");
        assert_eq!(run.case_count, dtype_shape_diagnostic_cases().len());
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
