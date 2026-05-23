//! Conformance tests for numpy.convolve and numpy.correlate against NumPy oracle.
//!
//! Tests 1D convolution and correlation across all modes (full, same, valid),
//! various array sizes, and edge cases.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

type ConvCase<'a> = (&'a str, &'a str, &'a [f64], &'a [f64], &'a str);

fn mode_kwargs<'py>(py: Python<'py>, mode: &str) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("mode", mode)?;
    Ok(Some(kwargs))
}

fn np_array_1d_f<'py>(
    py: Python<'py>,
    values: &[f64],
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?
        .getattr("array")?
        .call1((values.to_vec(),))
}

fn np_array_1d_complex<'py>(
    py: Python<'py>,
    values: &[(f64, f64)],
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let np = py.import("numpy")?;
    let complex_list: Vec<_> = values
        .iter()
        .map(|(r, i)| pyo3::types::PyComplex::from_doubles(py, *r, *i))
        .collect();
    let arr = np.getattr("array")?.call1((complex_list,))?;
    arr.call_method1("astype", (np.getattr("complex128")?,))
}

#[test]
fn convolution_fnp_python_module_paths_match_numpy() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        let cases: &[ConvCase<'_>] = &[
            (
                "convolve",
                "basic-full",
                &[1.0, 2.0, 3.0],
                &[0.0, 1.0, 0.5],
                "full",
            ),
            (
                "convolve",
                "negative-kernel-full",
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                &[1.0, -1.0],
                "full",
            ),
            (
                "convolve",
                "single-left-full",
                &[1.0],
                &[1.0, 2.0, 3.0, 4.0],
                "full",
            ),
            (
                "convolve",
                "kernel-longer-valid",
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                "valid",
            ),
            (
                "convolve",
                "same-odd-kernel",
                &[1.0, 2.0, 3.0, 4.0],
                &[1.0, 2.0, 3.0],
                "same",
            ),
            (
                "convolve",
                "same-kernel-longer",
                &[1.0, 2.0],
                &[1.0, 2.0, 3.0],
                "same",
            ),
            (
                "convolve",
                "fractional-valid",
                &[0.5, 1.0, 1.5, 2.0, 2.5],
                &[2.0, -1.0, 2.0],
                "valid",
            ),
            (
                "convolve",
                "symmetric-full",
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                &[1.0, 2.0, 3.0, 2.0, 1.0],
                "full",
            ),
            (
                "correlate",
                "basic-full",
                &[1.0, 2.0, 3.0],
                &[0.0, 1.0, 0.5],
                "full",
            ),
            (
                "correlate",
                "negative-kernel-full",
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                &[1.0, -1.0],
                "full",
            ),
            (
                "correlate",
                "reverse-full",
                &[1.0, 2.0, 3.0, 4.0],
                &[4.0, 3.0, 2.0, 1.0],
                "full",
            ),
            (
                "correlate",
                "same-even",
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                &[1.0, 2.0, 1.0],
                "same",
            ),
            (
                "correlate",
                "same-kernel-longer",
                &[1.0, 2.0],
                &[1.0, 2.0, 3.0],
                "same",
            ),
            (
                "correlate",
                "valid-kernel-longer",
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                "valid",
            ),
            (
                "correlate",
                "fractional-valid",
                &[0.5, 1.0, 1.5, 2.0, 2.5],
                &[2.0, -1.0, 2.0],
                "valid",
            ),
            (
                "correlate",
                "second-difference-full",
                &[0.0, 1.0, 2.0, 3.0, 4.0],
                &[1.0, -2.0, 1.0],
                "full",
            ),
        ];

        for (function, label, a, v, mode) in cases {
            let a = (*a).to_vec();
            let v = (*v).to_vec();
            let mode = (*mode).to_string();
            run_case(
                py,
                &module,
                &numpy,
                &format!("{function}-fnp-python-{label}"),
                function,
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                move |py| PyTuple::new(py, [np_array_1d_f(py, &a)?, np_array_1d_f(py, &v)?]),
                move |py| mode_kwargs(py, &mode),
            );
        }

        // Complex dtype tests
        let complex_cases: &[(&str, &str, &[(f64, f64)], &[(f64, f64)], &str)] = &[
            (
                "convolve",
                "complex-full",
                &[(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)],
                &[(0.5, 0.5), (1.0, -0.5)],
                "full",
            ),
            (
                "correlate",
                "complex-full",
                &[(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)],
                &[(0.5, 0.5), (1.0, -0.5)],
                "full",
            ),
            (
                "convolve",
                "complex-same",
                &[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0)],
                &[(1.0, 0.5), (0.5, -0.5), (0.0, 1.0)],
                "same",
            ),
            (
                "correlate",
                "complex-valid",
                &[(1.0, 1.0), (2.0, -1.0), (3.0, 2.0), (4.0, -2.0)],
                &[(0.5, 0.0), (1.0, 0.0)],
                "valid",
            ),
        ];

        for (function, label, a, v, mode) in complex_cases {
            let a = (*a).to_vec();
            let v = (*v).to_vec();
            let mode = (*mode).to_string();
            run_case(
                py,
                &module,
                &numpy,
                &format!("{function}-{label}"),
                function,
                RequirementLevel::Should,
                CompareMode::Close,
                t,
                move |py| {
                    PyTuple::new(py, [np_array_1d_complex(py, &a)?, np_array_1d_complex(py, &v)?])
                },
                move |py| mode_kwargs(py, &mode),
            );
        }

        eprintln!("{}", TOTALS.summarize("convolution-fnp-python"));
        Ok(())
    });
}
