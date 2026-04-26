//! Conformance matrix: indexing family.
//!
//! Differential parity for fnp_python's read-only indexing surface:
//!
//!   take, take_along_axis, compress, choose, extract, select
//!
//! All are native bodies that delegate to numpy on edge cases (out-of-
//! bounds → IndexError vs ValueError, dtype/shape mismatches). The
//! harness pins MUST cases at smallest valid shapes, SHOULD cases on
//! the kwargs most likely to drift (`axis=`, `mode=`), and MAY cases on
//! empty / all-false edges.
//!
//! Mutation functions (`put`, `put_along_axis`) are out of scope here —
//! they mutate `arr` in place and return None, which the existing
//! run_case harness can't compare cleanly.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

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

fn np_array_2d_f<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn np_array_2d_i<'py>(
    py: Python<'py>,
    rows: Vec<Vec<i64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn np_array_1d_b<'py>(
    py: Python<'py>,
    values: Vec<bool>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

#[test]
fn conformance_indexing_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── take (MUST + SHOULD axis= negative-axis) ──────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-take-1d",
            "take",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![10.0, 20.0, 30.0, 40.0, 50.0])?,
                        np_array_1d_i(py, vec![0, 2, 4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-take-2d-axis0",
            "take",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_2d_f(
                            py,
                            vec![
                                vec![1.0, 2.0, 3.0],
                                vec![4.0, 5.0, 6.0],
                                vec![7.0, 8.0, 9.0],
                            ],
                        )?,
                        np_array_1d_i(py, vec![0, 2])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 0_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-take-2d-axis-neg1",
            "take",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_2d_f(
                            py,
                            vec![
                                vec![1.0, 2.0, 3.0],
                                vec![4.0, 5.0, 6.0],
                            ],
                        )?,
                        np_array_1d_i(py, vec![0, 2])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", -1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── take_along_axis (MUST + SHOULD axis=-1) ───────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-take_along_axis-2d-axis1",
            "take_along_axis",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_2d_f(
                            py,
                            vec![
                                vec![10.0, 20.0, 30.0],
                                vec![40.0, 50.0, 60.0],
                            ],
                        )?,
                        np_array_2d_i(py, vec![vec![2, 0], vec![1, 2]])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-take_along_axis-axis-neg1",
            "take_along_axis",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_2d_f(
                            py,
                            vec![
                                vec![10.0, 20.0, 30.0],
                                vec![40.0, 50.0, 60.0],
                            ],
                        )?,
                        np_array_2d_i(py, vec![vec![2, 0], vec![1, 2]])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", -1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── compress (MUST + SHOULD axis=) ────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-compress-1d",
            "compress",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_b(py, vec![true, false, true, false, true])?,
                        np_array_1d_f(py, vec![10.0, 20.0, 30.0, 40.0, 50.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-compress-2d-axis1",
            "compress",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_b(py, vec![true, false, true])?,
                        np_array_2d_f(
                            py,
                            vec![
                                vec![1.0, 2.0, 3.0],
                                vec![4.0, 5.0, 6.0],
                            ],
                        )?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── choose (MUST + SHOULD mode=) ──────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-choose-2-options",
            "choose",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let selectors = np_array_1d_i(py, vec![0, 1, 0, 1, 0])?;
                let opt0 = np_array_1d_f(py, vec![10.0, 11.0, 12.0, 13.0, 14.0])?;
                let opt1 = np_array_1d_f(py, vec![20.0, 21.0, 22.0, 23.0, 24.0])?;
                let choices = PyList::new(py, [opt0, opt1])?;
                PyTuple::new(py, [selectors, choices.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-choose-mode-wrap",
            "choose",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                // Out-of-range selector index — wrap mode rotates into bounds
                // instead of raising. If our wrapper dropped `mode=`, this
                // would raise IndexError instead of returning array values.
                let selectors = np_array_1d_i(py, vec![0, 3, 1, 4])?; // 3,4 OOB
                let opt0 = np_array_1d_f(py, vec![10.0, 11.0, 12.0, 13.0])?;
                let opt1 = np_array_1d_f(py, vec![20.0, 21.0, 22.0, 23.0])?;
                let choices = PyList::new(py, [opt0, opt1])?;
                PyTuple::new(py, [selectors, choices.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("mode", "wrap")?;
                Ok(Some(kw))
            },
        );

        // ─── extract (MUST) ────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-extract-1d",
            "extract",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_b(py, vec![true, false, true, false, true])?,
                        np_array_1d_f(py, vec![10.0, 20.0, 30.0, 40.0, 50.0])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── select (MUST + SHOULD default=) ───────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-select-two-conditions",
            "select",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let x = np_array_1d_f(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
                let cond0 = np_array_1d_b(py, vec![true, false, false, false, false])?;
                let cond1 = np_array_1d_b(py, vec![false, false, true, false, false])?;
                let condlist = PyList::new(py, [cond0, cond1])?;
                let choice0 = np_array_1d_f(py, vec![100.0, 100.0, 100.0, 100.0, 100.0])?;
                let choice1 = np_array_1d_f(py, vec![200.0, 200.0, 200.0, 200.0, 200.0])?;
                let choicelist = PyList::new(py, [choice0, choice1])?;
                let _ = x;
                PyTuple::new(py, [condlist.into_any(), choicelist.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-select-with-default",
            "select",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let cond0 = np_array_1d_b(py, vec![true, false, false, false])?;
                let cond1 = np_array_1d_b(py, vec![false, false, true, false])?;
                let condlist = PyList::new(py, [cond0, cond1])?;
                let choice0 = np_array_1d_f(py, vec![100.0, 100.0, 100.0, 100.0])?;
                let choice1 = np_array_1d_f(py, vec![200.0, 200.0, 200.0, 200.0])?;
                let choicelist = PyList::new(py, [choice0, choice1])?;
                let default = np_array_1d_f(py, vec![-1.0, -1.0, -1.0, -1.0])?;
                PyTuple::new(
                    py,
                    [condlist.into_any(), choicelist.into_any(), default],
                )
            },
            no_kwargs,
        );

        // ─── edge cases (MAY) ──────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "indexing-take-empty-indices",
            "take",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                // Empty `[]` defaults to float64 in numpy — force int64 so
                // both sides take the success path (empty take → empty
                // result) rather than diverging on the same TypeError with
                // different message bodies.
                let array = py.import("numpy")?.getattr("array")?;
                let kw = PyDict::new(py);
                kw.set_item("dtype", "int64")?;
                let empty_int: Vec<i64> = vec![];
                let empty_indices = array.call((empty_int,), Some(&kw))?;
                PyTuple::new(
                    py,
                    [np_array_1d_f(py, vec![1.0, 2.0, 3.0])?, empty_indices],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "indexing-compress-all-false-mask",
            "compress",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_b(py, vec![false, false, false])?,
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0])?,
                    ],
                )
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("indexing");
    eprintln!("\n=== fnp-python conformance matrix: indexing ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in indexing family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
