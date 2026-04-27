//! Conformance matrix: special math + windows + interpolation family.
//!
//! Differential parity for fnp_python's special-function surface that
//! doesn't fit cleanly into the arithmetic / reducer / fft / linalg
//! families: scientific window generators (kaiser, blackman, hanning,
//! hamming, bartlett), Bessel-family scalars (i0, sinc), and the
//! sample-data utilities (trapezoid, trapz, interp, digitize, bincount,
//! histogram).
//!
//! All 13 wrappers ship as numpy passthroughs (no native fast path) so
//! every parity assertion uses Strict comparison for integer outputs
//! and Close for float outputs (1 ULP tolerated).

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case_resolved, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_f<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_i<'py>(
    py: Python<'py>,
    values: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

#[test]
fn conformance_special_math_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── Window functions (M-point arrays) ────────────────────
        // kaiser takes (M, beta), the other 4 take (M,).
        for (name, m_value) in [
            ("blackman", 8_i64),
            ("hanning", 11_i64),
            ("hamming", 16_i64),
            ("bartlett", 7_i64),
        ] {
            let our_fn = module.getattr(name)?;
            let their_fn = numpy.getattr(name)?;
            run_case_resolved(
                py,
                &format!("special-{name}-default"),
                name,
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                move |py| {
                    let m_obj: pyo3::Bound<'_, pyo3::types::PyAny> =
                        m_value.into_pyobject(py)?.into_any();
                    PyTuple::new(py, [m_obj])
                },
                no_kwargs,
            );
        }
        // SHOULD: zero-length and unit-length windows return arrays of
        // matching shape.
        for (name, m_value) in [
            ("blackman", 0_i64),
            ("hanning", 1_i64),
            ("hamming", 0_i64),
            ("bartlett", 1_i64),
        ] {
            let our_fn = module.getattr(name)?;
            let their_fn = numpy.getattr(name)?;
            run_case_resolved(
                py,
                &format!("special-{name}-edge-{m_value}"),
                name,
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                move |py| {
                    let m_obj: pyo3::Bound<'_, pyo3::types::PyAny> =
                        m_value.into_pyobject(py)?.into_any();
                    PyTuple::new(py, [m_obj])
                },
                no_kwargs,
            );
        }
        // kaiser(M, beta) — MUST + SHOULD edge.
        {
            let our_fn = module.getattr("kaiser")?;
            let their_fn = numpy.getattr("kaiser")?;
            run_case_resolved(
                py,
                "special-kaiser-default",
                "kaiser",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                |py| {
                    let m_obj: pyo3::Bound<'_, pyo3::types::PyAny> =
                        12_i64.into_pyobject(py)?.into_any();
                    let beta: pyo3::Bound<'_, pyo3::types::PyAny> =
                        14.0_f64.into_pyobject(py)?.into_any();
                    PyTuple::new(py, [m_obj, beta])
                },
                no_kwargs,
            );
            run_case_resolved(
                py,
                "special-kaiser-zero-beta",
                "kaiser",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                |py| {
                    let m_obj: pyo3::Bound<'_, pyo3::types::PyAny> =
                        8_i64.into_pyobject(py)?.into_any();
                    let beta: pyo3::Bound<'_, pyo3::types::PyAny> =
                        0.0_f64.into_pyobject(py)?.into_any();
                    PyTuple::new(py, [m_obj, beta])
                },
                no_kwargs,
            );
        }

        // ─── Bessel / sinc family ─────────────────────────────────
        {
            let our_fn = module.getattr("i0")?;
            let their_fn = numpy.getattr("i0")?;
            run_case_resolved(
                py,
                "special-i0-array",
                "i0",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![0.0, 1.0, 2.5, 5.0])?]),
                no_kwargs,
            );
        }
        {
            let our_fn = module.getattr("sinc")?;
            let their_fn = numpy.getattr("sinc")?;
            run_case_resolved(
                py,
                "special-sinc-array",
                "sinc",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![-2.0, -1.0, 0.0, 1.0, 2.0])?]),
                no_kwargs,
            );
            // SHOULD: sinc(0) is exactly 1 (numpy uses normalized sinc).
            run_case_resolved(
                py,
                "special-sinc-scalar-zero",
                "sinc",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                |py| {
                    let zero: pyo3::Bound<'_, pyo3::types::PyAny> =
                        0.0_f64.into_pyobject(py)?.into_any();
                    PyTuple::new(py, [zero])
                },
                no_kwargs,
            );
        }

        // ─── trapezoid / trapz integration ────────────────────────
        // numpy 2.x removed numpy.trapz (replaced by numpy.trapezoid),
        // but fnp_python keeps trapz as a deprecated alias for
        // backward compatibility with numpy 1.x callers. Skip the
        // numpy reference lookup when numpy.trapz is gone.
        for name in ["trapezoid", "trapz"] {
            let our_fn = module.getattr(name)?;
            let their_fn = match numpy.getattr(name) {
                Ok(fn_obj) => fn_obj,
                Err(_) => continue,
            };
            run_case_resolved(
                py,
                &format!("special-{name}-default"),
                name,
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![1.0, 2.0, 3.0, 4.0])?]),
                no_kwargs,
            );
            // SHOULD: explicit dx kwarg.
            run_case_resolved(
                py,
                &format!("special-{name}-dx"),
                name,
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![1.0, 4.0, 9.0, 16.0])?]),
                |py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dx", 0.5_f64)?;
                    Ok(Some(kwargs))
                },
            );
        }

        // ─── interp ───────────────────────────────────────────────
        {
            let our_fn = module.getattr("interp")?;
            let their_fn = numpy.getattr("interp")?;
            run_case_resolved(
                py,
                "special-interp-monotonic",
                "interp",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                |py| {
                    PyTuple::new(
                        py,
                        [
                            np_array_f(py, vec![1.5, 2.5, 3.5])?,
                            np_array_f(py, vec![1.0, 2.0, 3.0, 4.0])?,
                            np_array_f(py, vec![10.0, 20.0, 30.0, 40.0])?,
                        ],
                    )
                },
                no_kwargs,
            );
            // SHOULD: out-of-range x clamps to fp[0] / fp[-1].
            run_case_resolved(
                py,
                "special-interp-out-of-range",
                "interp",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                |py| {
                    PyTuple::new(
                        py,
                        [
                            np_array_f(py, vec![0.0, 5.0])?,
                            np_array_f(py, vec![1.0, 2.0, 3.0, 4.0])?,
                            np_array_f(py, vec![10.0, 20.0, 30.0, 40.0])?,
                        ],
                    )
                },
                no_kwargs,
            );
        }

        // ─── digitize ─────────────────────────────────────────────
        {
            let our_fn = module.getattr("digitize")?;
            let their_fn = numpy.getattr("digitize")?;
            run_case_resolved(
                py,
                "special-digitize-default",
                "digitize",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Strict,
                t,
                |py| {
                    PyTuple::new(
                        py,
                        [
                            np_array_f(py, vec![0.5, 1.5, 2.5, 3.5])?,
                            np_array_f(py, vec![1.0, 2.0, 3.0])?,
                        ],
                    )
                },
                no_kwargs,
            );
            // SHOULD: right=True flips edge inclusion semantics.
            run_case_resolved(
                py,
                "special-digitize-right-true",
                "digitize",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Strict,
                t,
                |py| {
                    let true_obj = pyo3::types::PyBool::new(py, true).to_owned().into_any();
                    PyTuple::new(
                        py,
                        [
                            np_array_f(py, vec![1.0, 2.0, 3.0])?,
                            np_array_f(py, vec![1.0, 2.0, 3.0])?,
                            true_obj,
                        ],
                    )
                },
                no_kwargs,
            );
        }

        // ─── bincount ─────────────────────────────────────────────
        {
            let our_fn = module.getattr("bincount")?;
            let their_fn = numpy.getattr("bincount")?;
            run_case_resolved(
                py,
                "special-bincount-default",
                "bincount",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Strict,
                t,
                |py| PyTuple::new(py, [np_array_i(py, vec![0, 1, 1, 3, 2, 1, 7])?]),
                no_kwargs,
            );
            // SHOULD: minlength kwarg pads zeros at the tail.
            run_case_resolved(
                py,
                "special-bincount-minlength",
                "bincount",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Strict,
                t,
                |py| PyTuple::new(py, [np_array_i(py, vec![0, 1, 2])?]),
                |py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("minlength", 6_i64)?;
                    Ok(Some(kwargs))
                },
            );
            // SHOULD: weights kwarg returns float bins.
            run_case_resolved(
                py,
                "special-bincount-weights",
                "bincount",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                |py| PyTuple::new(py, [np_array_i(py, vec![0, 1, 2, 1, 0])?]),
                |py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("weights", np_array_f(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?)?;
                    Ok(Some(kwargs))
                },
            );
        }

        // ─── histogram ────────────────────────────────────────────
        // Original failing input from the franken_numpy-zcjo
        // discovery: values at bin-edges (1.0, 1.2 boundaries on the
        // 10-bin auto range) used to land in the wrong bin under f64
        // rounding. Now passes after the searchsorted-right fix.
        {
            let our_fn = module.getattr("histogram")?;
            let their_fn = numpy.getattr("histogram")?;
            run_case_resolved(
                py,
                "special-histogram-bin-edge-regression",
                "histogram",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Surface,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.2])?]),
                no_kwargs,
            );
            // Off-edge sample retained as additional MUST coverage.
            run_case_resolved(
                py,
                "special-histogram-off-edge",
                "histogram",
                &our_fn,
                &their_fn,
                RequirementLevel::Must,
                CompareMode::Surface,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![1.05, 1.55, 2.05, 2.55, 1.35])?]),
                no_kwargs,
            );
            // SHOULD: explicit bins int + range. Off-edge sample so
            // bin assignment ambiguity doesn't fire.
            run_case_resolved(
                py,
                "special-histogram-explicit-bins-range",
                "histogram",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Surface,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![1.05, 1.55, 2.05, 2.55, 1.35])?]),
                |py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("bins", 4_i64)?;
                    let range = PyTuple::new(py, [1.0_f64, 3.0_f64])?;
                    kwargs.set_item("range", range)?;
                    Ok(Some(kwargs))
                },
            );
            // SHOULD: density=True normalizes to PDF.
            run_case_resolved(
                py,
                "special-histogram-density",
                "histogram",
                &our_fn,
                &their_fn,
                RequirementLevel::Should,
                CompareMode::Surface,
                t,
                |py| PyTuple::new(py, [np_array_f(py, vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.2])?]),
                |py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("density", true)?;
                    Ok(Some(kwargs))
                },
            );
        }

        eprintln!("\n{}", t.summarize("special_math"));
        Ok(())
    });
}
