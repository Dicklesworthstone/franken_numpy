//! Conformance matrix: searching family.
//!
//! Differential parity for fnp_python's searching/sorting surface:
//!
//!   nonzero, flatnonzero, argwhere, count_nonzero,
//!   argmax, argmin, searchsorted, where
//!
//! Several of these are native (UFuncArray-backed: nonzero, argwhere,
//! count_nonzero, searchsorted, where) and several pass through to
//! numpy via `*args, **kwargs` (argmax, argmin). Per the yffc lesson,
//! native bodies are the ones most prone to silently dropping kwargs
//! that callers depend on, so SHOULD cases here exercise `axis=`,
//! `side=`, `keepdims=` to pin those surface bits.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_1d_f<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_i<'py>(
    py: Python<'py>,
    values: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_s<'py>(
    py: Python<'py>,
    values: Vec<&str>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_object_s<'py>(
    py: Python<'py>,
    values: Vec<&str>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr("object_")?)?;
    numpy.getattr("array")?.call((values,), Some(&kwargs))
}

fn np_array_2d_f<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn np_array_2d_s<'py>(
    py: Python<'py>,
    rows: Vec<Vec<&str>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

#[test]
fn conformance_searching_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── nonzero / flatnonzero / argwhere (MUST) ───────────────────
        run_case(
            py,
            &module,
            &numpy,
            "searching-nonzero-1d",
            "nonzero",
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, 0, 2, 0, 3])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-nonzero-2d",
            "nonzero",
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-flatnonzero-1d",
            "flatnonzero",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, 0, 2, 0, 3])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-flatnonzero-string-truthiness",
            "flatnonzero",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-flatnonzero-object-truthiness",
            "flatnonzero",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_object_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-argwhere-2d",
            "argwhere",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-argwhere-string-truthiness",
            "argwhere",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-argwhere-object-truthiness",
            "argwhere",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_object_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );

        // ─── count_nonzero (MUST + SHOULD axis=) ───────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-1d",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, 0, 2, 0, 3])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-string-truthiness",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-object-truthiness",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_object_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-2d-axis0",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(
                        py,
                        vec![
                            vec![0.0, 1.0, 2.0],
                            vec![3.0, 0.0, 0.0],
                            vec![4.0, 5.0, 0.0],
                        ],
                    )?],
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
            "searching-count_nonzero-numpy-int64-axis",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let numpy = py.import("numpy")?;
                let kw = PyDict::new(py);
                kw.set_item("axis", numpy.getattr("int64")?.call1((0_i64,))?)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-2d-axis1-keepdims",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(
                        py,
                        vec![
                            vec![0.0, 1.0, 2.0],
                            vec![3.0, 0.0, 0.0],
                            vec![4.0, 5.0, 0.0],
                        ],
                    )?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                kw.set_item("keepdims", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-string-axis1",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_s(
                        py,
                        vec![vec!["", "x", ""], vec!["a", "", "b"]],
                    )?],
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
            "searching-count_nonzero-empty-tuple-axis",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", PyTuple::empty(py))?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-tuple-axis-keepdims",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", (0_i64, 1_i64))?;
                kw.set_item("keepdims", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-list-axis-rejected",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", vec![0_i64])?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-numpy-bool-axis-rejected",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let numpy = py.import("numpy")?;
                let kw = PyDict::new(py);
                kw.set_item("axis", numpy.getattr("bool_")?.call1((false,))?)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-tuple-bool-axis-rejected",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", PyTuple::new(py, [true])?)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-tuple-numpy-bool-axis-rejected",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let numpy = py.import("numpy")?;
                let kw = PyDict::new(py);
                let axis = PyTuple::new(py, [numpy.getattr("bool_")?.call1((false,))?])?;
                kw.set_item("axis", axis)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-tuple-noninteger-axis-rejected",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", (0_i64, "x"))?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-ndarray-vector-axis-rejected",
            "count_nonzero",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let numpy = py.import("numpy")?;
                let kw = PyDict::new(py);
                kw.set_item("axis", numpy.getattr("array")?.call1((vec![0_i64],))?)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-count_nonzero-ndarray-scalar-axis",
            "count_nonzero",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(py, vec![vec![0.0, 1.0], vec![2.0, 0.0]])?],
                )
            },
            |py| {
                let numpy = py.import("numpy")?;
                let kw = PyDict::new(py);
                kw.set_item("axis", numpy.getattr("array")?.call1((0_i64,))?)?;
                Ok(Some(kw))
            },
        );

        // ─── argmax / argmin (MUST + SHOULD axis) ──────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "searching-argmax-1d",
            "argmax",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_1d_f(py, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-argmin-1d",
            "argmin",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_1d_f(py, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-argmax-2d-axis0",
            "argmax",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(
                        py,
                        vec![vec![1.0, 5.0, 3.0], vec![4.0, 2.0, 6.0]],
                    )?],
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
            "searching-argmin-2d-axis1",
            "argmin",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(
                        py,
                        vec![vec![1.0, 5.0, 3.0], vec![4.0, 2.0, 6.0]],
                    )?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── searchsorted (MUST + SHOULD side=) ────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "searching-searchsorted-scalar",
            "searchsorted",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 3.0, 5.0, 7.0, 9.0])?,
                        4.0_f64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-searchsorted-array",
            "searchsorted",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 3.0, 5.0, 7.0, 9.0])?,
                        np_array_1d_f(py, vec![0.0, 4.0, 5.0, 6.0, 10.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        // Regression: prior scalar-detection fallback misclassified Python
        // lists as scalars (no `ndim` → unwrap_or(true)) and returned a
        // scalar where numpy returns array([2]). MUST tier so the fix
        // can't silently regress.
        run_case(
            py,
            &module,
            &numpy,
            "searching-searchsorted-python-list-input",
            "searchsorted",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let v_list = pyo3::types::PyList::new(py, vec![4_i64])?.into_any();
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 3.0, 5.0, 7.0, 9.0])?.into_any(),
                        v_list,
                    ],
                )
            },
            no_kwargs,
        );
        // Regression sibling: 0-D ndarray input must still return scalar.
        run_case(
            py,
            &module,
            &numpy,
            "searching-searchsorted-0d-array-input",
            "searchsorted",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let int64_4 = py.import("numpy")?.getattr("int64")?.call1((4_i64,))?;
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 3.0, 5.0, 7.0, 9.0])?.into_any(),
                        int64_4,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-searchsorted-side-right",
            "searchsorted",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 3.0, 5.0, 5.0, 7.0])?,
                        np_array_1d_f(py, vec![5.0, 5.0])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("side", "right")?;
                Ok(Some(kw))
            },
        );

        // ─── where (MUST 1-arg + 3-arg) ────────────────────────────────
        // 1-arg form returns a tuple of nonzero indices.
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-1arg-as-nonzero",
            "where",
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, 0, 2, 0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-string-condition-as-nonzero",
            "where",
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-object-condition-as-nonzero",
            "where",
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_object_s(py, vec!["", "x", "0", "false"])?]),
            no_kwargs,
        );
        // 3-arg form selects from x or y based on condition.
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-3arg-select",
            "where",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let cond = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![true, false, true, false],))?;
                PyTuple::new(
                    py,
                    [
                        cond,
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0, 4.0])?,
                        np_array_1d_f(py, vec![10.0, 20.0, 30.0, 40.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-string-condition-select",
            "where",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_s(py, vec!["", "x", "0", "false"])?,
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                        np_array_1d_i(py, vec![10, 20, 30, 40])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-object-condition-select",
            "where",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_object_s(py, vec!["", "x", "0", "false"])?,
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                        np_array_1d_i(py, vec![10, 20, 30, 40])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-string-choices",
            "where",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let cond = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![true, false, true, false],))?;
                PyTuple::new(
                    py,
                    [
                        cond,
                        np_array_1d_s(py, vec!["a", "b", "c", "d"])?,
                        np_array_1d_s(py, vec!["w", "x", "y", "z"])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-where-object-choices",
            "where",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let cond = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![true, false, true, false],))?;
                PyTuple::new(
                    py,
                    [
                        cond,
                        np_array_1d_object_s(py, vec!["a", "b", "c", "d"])?,
                        np_array_1d_object_s(py, vec!["w", "x", "y", "z"])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── edge cases (MAY) ──────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "searching-nonzero-all-zero",
            "nonzero",
            RequirementLevel::May,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 0, 0, 0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "searching-argwhere-all-zero",
            "argwhere",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 0, 0, 0])?]),
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("searching");
    eprintln!("\n=== fnp-python conformance matrix: searching ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in searching family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
