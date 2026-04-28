//! Conformance matrix: masked-array family.
//!
//! Differential parity for the `fnp_python.ma` submodule.  The masked
//! array surface is a README-promised API family, but unlike array
//! creation, FFT, linalg, setops, and testing it did not have a
//! standalone family-level integration harness.  This matrix pins mask
//! propagation, masked reductions, weighted-average tuple returns,
//! all-mask construction, and representative error paths against
//! `numpy.ma`.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case_resolved, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
    Ok(None)
}

fn eval_expr<'py>(py: Python<'py>, expr: &str) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let globals = PyDict::new(py);
    globals.set_item("np", numpy)?;
    py.import("builtins")?
        .getattr("eval")?
        .call((expr, &globals), None::<&Bound<'py, PyDict>>)
}

fn args_from_exprs<'py>(py: Python<'py>, exprs: &[&str]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items = Vec::with_capacity(exprs.len());
    for expr in exprs {
        items.push(eval_expr(py, expr)?.into_any());
    }
    PyTuple::new(py, items)
}

fn kwargs_axis_keepdims<'py>(py: Python<'py>, axis: i64) -> PyResult<Option<Bound<'py, PyDict>>> {
    let kw = PyDict::new(py);
    kw.set_item("axis", axis)?;
    kw.set_item("keepdims", true)?;
    Ok(Some(kw))
}

#[test]
fn conformance_ma_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;
        let ma = module.getattr("ma")?;
        let numpy_ma = numpy.getattr("ma")?;

        // Construction and all-mask shape/dtype contracts.
        run_case_resolved(
            py,
            "ma-array-mask-fill-hardmask",
            "ma.array",
            &ma.getattr("array")?,
            &numpy_ma.getattr("array")?,
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| args_from_exprs(py, &["[[1, 2, 3], [4, 5, 6]]"]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item(
                    "mask",
                    eval_expr(py, "[[False, True, False], [True, False, False]]")?,
                )?;
                kw.set_item("fill_value", -99_i64)?;
                kw.set_item("hard_mask", true)?;
                Ok(Some(kw))
            },
        );
        run_case_resolved(
            py,
            "ma-masked_all-2d-int16",
            "ma.masked_all",
            &ma.getattr("masked_all")?,
            &numpy_ma.getattr("masked_all")?,
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| args_from_exprs(py, &["(2, 3)"]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", numpy.getattr("int16")?)?;
                Ok(Some(kw))
            },
        );

        // Mask-preserving selectors and fillers.
        run_case_resolved(
            py,
            "ma-masked_where-broadcast-condition",
            "ma.masked_where",
            &ma.getattr("masked_where")?,
            &numpy_ma.getattr("masked_where")?,
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| {
                args_from_exprs(
                    py,
                    &[
                        "np.array([[False, True, False], [True, False, False]])",
                        "np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[False, False, True], [False, True, False]])",
                    ],
                )
            },
            no_kwargs,
        );
        run_case_resolved(
            py,
            "ma-filled-explicit-fill-value",
            "ma.filled",
            &ma.getattr("filled")?,
            &numpy_ma.getattr("filled")?,
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                args_from_exprs(
                    py,
                    &[
                        "np.ma.array([1.5, 2.5, 3.5], mask=[False, True, False], fill_value=-7.0)",
                        "-11.0",
                    ],
                )
            },
            no_kwargs,
        );
        run_case_resolved(
            py,
            "ma-masked_invalid-nan-inf",
            "ma.masked_invalid",
            &ma.getattr("masked_invalid")?,
            &numpy_ma.getattr("masked_invalid")?,
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| args_from_exprs(py, &["np.array([1.0, np.nan, np.inf, -np.inf, 5.0])"]),
            no_kwargs,
        );

        // Native wrappers registered under fnp_python.ma.
        run_case_resolved(
            py,
            "ma-count-axis1-keepdims",
            "ma.count",
            &ma.getattr("count")?,
            &numpy_ma.getattr("count")?,
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                args_from_exprs(
                    py,
                    &[
                        "np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [True, True, False]])",
                    ],
                )
            },
            |py| kwargs_axis_keepdims(py, 1),
        );
        run_case_resolved(
            py,
            "ma-average-axis1-weighted-returned",
            "ma.average",
            &ma.getattr("average")?,
            &numpy_ma.getattr("average")?,
            RequirementLevel::Must,
            CompareMode::Surface,
            t,
            |py| {
                args_from_exprs(
                    py,
                    &[
                        "np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[False, True, False], [False, False, True]])",
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                kw.set_item(
                    "weights",
                    eval_expr(
                        py,
                        "np.ma.array([[1.0, 9.0, 2.0], [2.0, 3.0, 10.0]], mask=[[False, True, False], [False, False, True]])",
                    )?,
                )?;
                kw.set_item("returned", true)?;
                Ok(Some(kw))
            },
        );
        run_case_resolved(
            py,
            "ma-ediff1d-masked-boundaries",
            "ma.ediff1d",
            &ma.getattr("ediff1d")?,
            &numpy_ma.getattr("ediff1d")?,
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| {
                args_from_exprs(
                    py,
                    &["np.ma.array([1, 4, 9, 16], mask=[False, True, False, False])"],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("to_begin", eval_expr(py, "np.ma.array([-1], mask=[True])")?)?;
                kw.set_item("to_end", eval_expr(py, "np.ma.array([99], mask=[False])")?)?;
                Ok(Some(kw))
            },
        );

        // Error alignment.
        run_case_resolved(
            py,
            "ma-count-invalid-axis",
            "ma.count",
            &ma.getattr("count")?,
            &numpy_ma.getattr("count")?,
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| args_from_exprs(py, &["np.ma.array([1, 2, 3], mask=[False, True, False])"]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 2_i64)?;
                Ok(Some(kw))
            },
        );

        let summary = TOTALS.summarize("ma");
        eprintln!("\n=== fnp-python conformance matrix: ma ===");
        eprintln!("{summary}");

        let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            failures, 0,
            "{failures} conformance case(s) failed in ma family \
             (see JSON verdict lines above)",
        );

        Ok(())
    });
}
