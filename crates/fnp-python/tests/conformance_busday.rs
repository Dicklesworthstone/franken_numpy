//! Conformance matrix: business day functions.
//!
//! Differential parity for fnp_python's business day surface:
//!
//!   busday_count, busday_offset, is_busday
//!
//! Finding: These 3 business day functions are exposed but had ZERO
//! conformance tests, despite being commonly used for financial/date
//! calculations.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_datetime64<'py>(
    py: Python<'py>,
    date_str: &str,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let numpy = py.import("numpy")?;
    numpy.getattr("datetime64")?.call1((date_str,))
}

fn np_datetime64_array<'py>(
    py: Python<'py>,
    dates: Vec<&str>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let numpy = py.import("numpy")?;
    let arr_fn = numpy.getattr("array")?;
    let dt64 = numpy.getattr("datetime64")?;
    let date_objs: Vec<_> = dates.iter().map(|d| dt64.call1((*d,)).unwrap()).collect();
    arr_fn.call1((date_objs,))
}

#[test]
fn conformance_busday_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── busday_count (MUST) ─────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "busday-count-single-week",
            "busday_count",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let start = np_datetime64(py, "2024-01-01")?;
                let end = np_datetime64(py, "2024-01-08")?;
                PyTuple::new(py, [start, end])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-count-same-day",
            "busday_count",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let start = np_datetime64(py, "2024-01-15")?;
                let end = np_datetime64(py, "2024-01-15")?;
                PyTuple::new(py, [start, end])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-count-full-month",
            "busday_count",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let start = np_datetime64(py, "2024-03-01")?;
                let end = np_datetime64(py, "2024-04-01")?;
                PyTuple::new(py, [start, end])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-count-negative-range",
            "busday_count",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let start = np_datetime64(py, "2024-01-15")?;
                let end = np_datetime64(py, "2024-01-10")?;
                PyTuple::new(py, [start, end])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-count-array-input",
            "busday_count",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let starts = np_datetime64_array(py, vec!["2024-01-01", "2024-02-01"])?;
                let ends = np_datetime64_array(py, vec!["2024-01-31", "2024-02-29"])?;
                PyTuple::new(py, [starts, ends])
            },
            no_kwargs,
        );

        // ─── busday_offset (MUST) ────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "busday-offset-next-day",
            "busday_offset",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-15")?;
                PyTuple::new(py, [date, 1_i64.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-offset-five-days",
            "busday_offset",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-15")?;
                PyTuple::new(py, [date, 5_i64.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-offset-negative",
            "busday_offset",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-15")?;
                PyTuple::new(py, [date, (-3_i64).into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-offset-from-weekend",
            "busday_offset",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-13")?; // Saturday
                PyTuple::new(py, [date, 1_i64.into_pyobject(py)?.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("roll", "forward")?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "busday-offset-array-input",
            "busday_offset",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let dates =
                    np_datetime64_array(py, vec!["2024-01-15", "2024-01-16", "2024-01-17"])?;
                PyTuple::new(py, [dates, 1_i64.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );

        // ─── is_busday (MUST) ────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "is-busday-weekday",
            "is_busday",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-15")?; // Monday
                PyTuple::new(py, [date])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "is-busday-saturday",
            "is_busday",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-13")?; // Saturday
                PyTuple::new(py, [date])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "is-busday-sunday",
            "is_busday",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let date = np_datetime64(py, "2024-01-14")?; // Sunday
                PyTuple::new(py, [date])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "is-busday-array",
            "is_busday",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let dates = np_datetime64_array(
                    py,
                    vec!["2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16"],
                )?;
                PyTuple::new(py, [dates])
            },
            no_kwargs,
        );

        let summary = TOTALS.summarize("busday");
        eprintln!("\n=== fnp-python conformance matrix: busday ===");
        eprintln!("{summary}");

        let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            failures, 0,
            "{failures} conformance case(s) failed in busday family \
             (see JSON verdict lines above)",
        );

        Ok(())
    });
}
