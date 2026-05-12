//! Conformance matrix: reductions family.
//!
//! Drives the same harness through reduction ufuncs against numpy.
//! Functions covered: sum, prod, min, max, mean, median, std, var,
//! ptp, all, any, argmin, argmax, cumsum, cumprod. Edge axes:
//! axis=None / 0 / 1, keepdims=True/False, empty input, NaN
//! propagation, dtype promotion (int → float64 for mean/var/std).

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_int_1d<'py>(
    py: Python<'py>,
    items: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((items,))
}

fn np_int_1d_dtype<'py>(
    py: Python<'py>,
    items: Vec<i64>,
    dtype: &str,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr(dtype)?)?;
    numpy.getattr("array")?.call((items,), Some(&kwargs))
}

fn np_float_1d<'py>(
    py: Python<'py>,
    items: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((items,))
}

fn np_int_2d<'py>(
    py: Python<'py>,
    rows: Vec<Vec<i64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn axis_kwargs<'py>(py: Python<'py>, axis: i64) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let kw = PyDict::new(py);
    kw.set_item("axis", axis)?;
    Ok(Some(kw))
}

fn axis_keepdims<'py>(
    py: Python<'py>,
    axis: i64,
    keepdims: bool,
) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let kw = PyDict::new(py);
    kw.set_item("axis", axis)?;
    kw.set_item("keepdims", keepdims)?;
    Ok(Some(kw))
}

fn dtype_kwargs<'py>(py: Python<'py>, dtype: &str) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let numpy = py.import("numpy")?;
    let kw = PyDict::new(py);
    kw.set_item("dtype", numpy.getattr(dtype)?)?;
    Ok(Some(kw))
}

fn axis_dtype_keepdims<'py>(
    py: Python<'py>,
    axis: i64,
    dtype: &str,
    keepdims: bool,
) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let numpy = py.import("numpy")?;
    let kw = PyDict::new(py);
    kw.set_item("axis", axis)?;
    kw.set_item("dtype", numpy.getattr(dtype)?)?;
    kw.set_item("keepdims", keepdims)?;
    Ok(Some(kw))
}

#[test]
fn conformance_reductions_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── sum ────────────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-1d-int",
            "sum",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4, 5])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-2d-axis-none",
            "sum",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_int_2d(py, vec![vec![1, 2], vec![3, 4], vec![5, 6]])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-2d-axis-0",
            "sum",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_int_2d(py, vec![vec![1, 2], vec![3, 4], vec![5, 6]])?],
                )
            },
            |py| axis_kwargs(py, 0),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-2d-axis-1-keepdims",
            "sum",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_2d(py, vec![vec![1, 2], vec![3, 4]])?]),
            |py| axis_keepdims(py, 1, true),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-empty",
            "sum",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-explicit-dtype-float32",
            "sum",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d_dtype(py, vec![1, 2, 3, 4, 5], "int16")?]),
            |py| dtype_kwargs(py, "float32"),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-axis-explicit-dtype-keepdims",
            "sum",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_int_2d(py, vec![vec![1, 2], vec![3, 4], vec![5, 6]])?],
                )
            },
            |py| axis_dtype_keepdims(py, 0, "int64", true),
        );

        // ─── prod ───────────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-prod-1d",
            "prod",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-prod-empty",
            "prod",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-prod-explicit-dtype-float64",
            "prod",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d_dtype(py, vec![2, 3, 4], "int16")?]),
            |py| dtype_kwargs(py, "float64"),
        );

        // ─── min / max ──────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-min-1d",
            "min",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![3, 1, 4, 1, 5, 9])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-max-2d-axis-0",
            "max",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_int_2d(py, vec![vec![5, 1], vec![2, 8], vec![3, 4]])?],
                )
            },
            |py| axis_kwargs(py, 0),
        );

        // ─── mean / median / std / var ──────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-mean-int-promotes-float",
            "mean",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4, 5])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-mean-2d-axis-1",
            "mean",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_2d(py, vec![vec![1, 2, 3], vec![4, 5, 6]])?]),
            |py| axis_kwargs(py, 1),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-mean-explicit-dtype-float32",
            "mean",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d_dtype(py, vec![1, 2, 3, 4, 5], "int16")?]),
            |py| dtype_kwargs(py, "float32"),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-median-odd-len",
            "median",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 3, 5, 7, 9])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-median-even-len",
            "median",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4, 5, 6])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-std-default",
            "std",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_float_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-var-default",
            "var",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_float_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-var-ddof-1",
            "var",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_float_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("ddof", 1_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-var-explicit-dtype-float32",
            "var",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d_dtype(py, vec![1, 2, 3, 4, 5], "int16")?]),
            |py| dtype_kwargs(py, "float32"),
        );

        // ─── ptp ────────────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-ptp-1d",
            "ptp",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![3, 1, 4, 1, 5, 9, 2, 6])?]),
            no_kwargs,
        );

        // ─── all / any ──────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-all-true",
            "all",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-all-with-zero",
            "all",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 0, 3])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-any-all-zero",
            "any",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![0, 0, 0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-any-empty",
            "any",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![])?]),
            no_kwargs,
        );

        // ─── argmin / argmax ─────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-argmin-1d",
            "argmin",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![3, 1, 4, 1, 5, 9])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-argmax-2d-axis-1",
            "argmax",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_2d(py, vec![vec![5, 1, 8], vec![2, 9, 3]])?]),
            |py| axis_kwargs(py, 1),
        );

        // ─── cumsum / cumprod ───────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-cumsum-1d",
            "cumsum",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4, 5])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-cumsum-2d-axis-0",
            "cumsum",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_int_2d(py, vec![vec![1, 2], vec![3, 4], vec![5, 6]])?],
                )
            },
            |py| axis_kwargs(py, 0),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-cumsum-explicit-dtype-int64",
            "cumsum",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d_dtype(py, vec![1, 2, 3, 4, 5], "int16")?]),
            |py| dtype_kwargs(py, "int64"),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-cumprod-1d",
            "cumprod",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_int_1d(py, vec![1, 2, 3, 4])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-cumprod-explicit-dtype-float64",
            "cumprod",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_int_1d_dtype(py, vec![2, 3, 4], "int16")?]),
            |py| dtype_kwargs(py, "float64"),
        );

        // ─── NaN propagation in float reducers ──────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-mean-with-nan-propagates",
            "mean",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_float_1d(py, vec![1.0, f64::NAN, 3.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-sum-with-inf",
            "sum",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_float_1d(py, vec![1.0, f64::INFINITY, 3.0])?]),
            no_kwargs,
        );

        // ─── NaN-aware reducers (May tier — niche) ─────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "reductions-nansum-with-nan",
            "nansum",
            RequirementLevel::May,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_float_1d(py, vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-nansum-explicit-dtype-float32",
            "nansum",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_float_1d(py, vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0])?],
                )
            },
            |py| dtype_kwargs(py, "float32"),
        );
        run_case(
            py,
            &module,
            &numpy,
            "reductions-nanmean-with-nan",
            "nanmean",
            RequirementLevel::May,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_float_1d(py, vec![1.0, f64::NAN, 3.0])?]),
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("reductions");
    eprintln!("\n=== fnp-python conformance matrix: reductions ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in reductions family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
