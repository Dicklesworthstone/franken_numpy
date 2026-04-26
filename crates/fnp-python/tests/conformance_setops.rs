//! Conformance matrix: setops family.
//!
//! Differential parity for fnp_python's set-operation surface:
//!
//!   intersect1d, union1d, setdiff1d, setxor1d, isin,
//!   unique, ediff1d
//!
//! All are native bodies that delegate to numpy on edge cases (string
//! arrays, structured dtypes). The harness exercises pass-paths under
//! Strict comparison and SHOULD-tier kwargs (`assume_unique`,
//! `return_indices`, `return_inverse`) that are most prone to silently
//! drop if a wrapper drifts from numpy's signature.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_1d_i<'py>(
    py: Python<'py>,
    values: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_f<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

#[test]
fn conformance_setops_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── intersect1d (MUST + SHOULD assume_unique / return_indices) ─
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-overlap",
            "intersect1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6, 7])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-disjoint",
            "intersect1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3])?,
                        np_array_1d_i(py, vec![10, 20, 30])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-assume_unique",
            "intersect1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6, 7])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("assume_unique", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-return_indices",
            "intersect1d",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6, 7])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("return_indices", true)?;
                Ok(Some(kw))
            },
        );

        // ─── union1d (MUST) ────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-union1d-overlap",
            "union1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-union1d-empty-rhs",
            "union1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3])?,
                        np_array_1d_i(py, vec![])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── setdiff1d (MUST + SHOULD assume_unique) ───────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-setdiff1d-overlap",
            "setdiff1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-setdiff1d-assume_unique",
            "setdiff1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("assume_unique", true)?;
                Ok(Some(kw))
            },
        );

        // ─── setxor1d (MUST) ───────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-setxor1d-overlap",
            "setxor1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── isin (MUST + SHOULD invert / assume_unique) ───────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-1d",
            "isin",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![2, 4, 6])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-invert",
            "isin",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![2, 4, 6])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("invert", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-assume_unique",
            "isin",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![2, 4, 6])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("assume_unique", true)?;
                Ok(Some(kw))
            },
        );

        // ─── unique (MUST + SHOULD return_inverse / return_counts) ─────
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-1d-with-dupes",
            "unique",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![3, 1, 2, 1, 3, 2, 4])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-return_inverse",
            "unique",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![3, 1, 2, 1, 3, 2, 4])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("return_inverse", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-return_counts",
            "unique",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![3, 1, 2, 1, 3, 2, 4])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("return_counts", true)?;
                Ok(Some(kw))
            },
        );

        // ─── ediff1d (MUST + SHOULD to_begin / to_end) ─────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-ediff1d-1d",
            "ediff1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_f(py, vec![1.0, 3.0, 6.0, 10.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-ediff1d-to_begin-to_end",
            "ediff1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_f(py, vec![1.0, 3.0, 6.0, 10.0])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("to_begin", np_array_1d_f(py, vec![-99.0])?)?;
                kw.set_item("to_end", np_array_1d_f(py, vec![99.0])?)?;
                Ok(Some(kw))
            },
        );

        // ─── empty-input edge cases (MAY) ──────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-both-empty",
            "intersect1d",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![])?,
                        np_array_1d_i(py, vec![])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-empty",
            "unique",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![])?]),
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("setops");
    eprintln!("\n=== fnp-python conformance matrix: setops ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in setops family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
