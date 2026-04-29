//! Conformance tests for numpy.ptp (peak-to-peak) against NumPy oracle.
//!
//! Tests the native Rust ptp implementation against NumPy across various
//! input shapes, axis parameters, and data types.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn axis_kwargs<'py>(py: Python<'py>, axis: i64) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("axis", axis)?;
    Ok(Some(kwargs))
}

fn np_array_1d_f<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_2d_f<'py>(
    py: Python<'py>,
    values: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_u8<'py>(
    py: Python<'py>,
    values: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", py.import("numpy")?.getattr("uint8")?)?;
    py.import("numpy")?
        .getattr("array")?
        .call((values,), Some(&kwargs))
}

#[test]
fn ptp_native_fnp_python_path_matches_numpy() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        let flat_cases: &[&[f64]] = &[
            &[1.0, -2.0, 5.0, 3.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[5.0, 4.0, 3.0, 2.0, 1.0],
            &[1.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[-1.0, -2.0, -3.0],
            &[-100.0, 0.0, 100.0],
            &[0.5, 1.5, 2.5],
            &[1e-10, 2e-10, 3e-10],
            &[1e10, 2e10, 3e10],
            &[0.123456789, 0.987654321],
        ];

        for (idx, values) in flat_cases.iter().enumerate() {
            let values = (*values).to_vec();
            run_case(
                py,
                &module,
                &numpy,
                &format!("ptp-native-flat-{idx}"),
                "ptp",
                RequirementLevel::Must,
                CompareMode::Surface,
                t,
                move |py| PyTuple::new(py, [np_array_1d_f(py, values.clone())?]),
                no_kwargs,
            );
        }

        let axis_cases: &[(Vec<Vec<f64>>, i64)] = &[
            (vec![vec![1.0, 5.0, -1.0], vec![4.0, 2.0, 3.0]], 0),
            (vec![vec![1.0, 5.0, -1.0], vec![4.0, 2.0, 3.0]], -1),
            (vec![vec![0.5, 1.5], vec![2.5, 3.5], vec![4.5, 5.5]], 0),
            (vec![vec![0.5, 1.5], vec![2.5, 3.5], vec![4.5, 5.5]], 1),
            (vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]], 0),
            (vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]], 1),
        ];

        for (idx, (values, axis)) in axis_cases.iter().enumerate() {
            let values = values.clone();
            let axis = *axis;
            run_case(
                py,
                &module,
                &numpy,
                &format!("ptp-native-axis-{idx}"),
                "ptp",
                RequirementLevel::Must,
                CompareMode::Close,
                t,
                move |py| PyTuple::new(py, [np_array_2d_f(py, values.clone())?]),
                move |py| axis_kwargs(py, axis),
            );
        }

        run_case(
            py,
            &module,
            &numpy,
            "ptp-native-nan-propagates",
            "ptp",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_f(py, vec![1.0, f64::NAN, 3.0])?]),
            no_kwargs,
        );

        run_case(
            py,
            &module,
            &numpy,
            "ptp-native-uint8-dtype",
            "ptp",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_u8(py, vec![255, 0, 128])?]),
            no_kwargs,
        );

        eprintln!("{}", TOTALS.summarize("ptp-native"));
        Ok(())
    });
}
