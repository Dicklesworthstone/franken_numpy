//! Conformance tests for numpy.cumsum and numpy.cumprod against NumPy oracle.
//!
//! Tests the native Rust cumsum/cumprod implementations against NumPy across
//! various input shapes, axis parameters, and data types.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn axis_kwargs<'py>(
    py: Python<'py>,
    axis: Option<i64>,
) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    let Some(axis) = axis else {
        return Ok(None);
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("axis", axis)?;
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

fn np_array_2d_f<'py>(
    py: Python<'py>,
    values: &[Vec<f64>],
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

fn np_array_2d_complex<'py>(
    py: Python<'py>,
    values: &[Vec<(f64, f64)>],
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let np = py.import("numpy")?;
    let nested: Vec<Vec<_>> = values
        .iter()
        .map(|row| {
            row.iter()
                .map(|(r, i)| pyo3::types::PyComplex::from_doubles(py, *r, *i))
                .collect()
        })
        .collect();
    let arr = np.getattr("array")?.call1((nested,))?;
    arr.call_method1("astype", (np.getattr("complex128")?,))
}

#[test]
fn cumsum_cumprod_native_fnp_python_paths_match_numpy() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        let flat_cases: &[&[f64]] = &[
            &[1.0, 2.0, 3.0],
            &[1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
            &[0.5, 1.5, 2.5],
            &[-100.0, 0.0, 100.0],
            &[1e-10, 2e-10, 3e-10],
            &[1e10, 2e10, 3e10],
            &[0.0, 0.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
            &[2.0, 0.5, 2.0, 0.5],
            &[-2.0, -3.0, -4.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[5.0, 4.0, 3.0, 2.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0],
            &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            &[0.99, 1.0, 1.01],
            &[-0.01, 0.0, 0.01],
            &[1000.0, 1001.0, 1002.0, 1003.0],
            &[1.5, 1.5, 1.5, 1.5],
            &[0.25, 4.0, 0.25, 4.0],
        ];

        for (idx, values) in flat_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = (*values).to_vec();
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-native-flat-{idx}"),
                    function,
                    RequirementLevel::Must,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_1d_f(py, &values)?]),
                    no_kwargs,
                );
            }
        }

        let axis_cases: &[(Vec<Vec<f64>>, i64)] = &[
            (vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 0),
            (vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 1),
            (vec![vec![0.5, 1.5], vec![2.5, 3.5], vec![4.5, 5.5]], -1),
            (vec![vec![-1.0, 2.0], vec![-3.0, 4.0], vec![-5.0, 6.0]], -2),
            (vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]], 0),
            (vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]], 1),
        ];

        for (idx, (values, axis)) in axis_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = values.clone();
                let axis = *axis;
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-native-axis-{idx}"),
                    function,
                    RequirementLevel::Must,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_2d_f(py, &values)?]),
                    move |py| axis_kwargs(py, Some(axis)),
                );
            }
        }

        // ─── complex dtype tests (SHOULD) ──────────────────────────────────
        let complex_flat_cases: &[&[(f64, f64)]] = &[
            &[(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)],
            &[(0.5, 0.5), (1.5, -1.5), (2.5, 2.5)],
            &[(-1.0, 1.0), (0.0, 0.0), (1.0, -1.0)],
        ];

        for (idx, values) in complex_flat_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = (*values).to_vec();
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-complex-flat-{idx}"),
                    function,
                    RequirementLevel::Should,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_1d_complex(py, &values)?]),
                    no_kwargs,
                );
            }
        }

        let complex_axis_cases: &[(Vec<Vec<(f64, f64)>>, i64)] = &[
            (
                vec![vec![(1.0, 1.0), (2.0, -1.0)], vec![(3.0, 2.0), (4.0, -2.0)]],
                0,
            ),
            (
                vec![vec![(1.0, 1.0), (2.0, -1.0)], vec![(3.0, 2.0), (4.0, -2.0)]],
                1,
            ),
        ];

        for (idx, (values, axis)) in complex_axis_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = values.clone();
                let axis = *axis;
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-complex-axis-{idx}"),
                    function,
                    RequirementLevel::Should,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_2d_complex(py, &values)?]),
                    move |py| axis_kwargs(py, Some(axis)),
                );
            }
        }

        // ─── NaN propagation tests (SHOULD) ─────────────────────────────────
        let nan_cases: &[&[f64]] = &[
            &[1.0, f64::NAN, 3.0, 4.0],
            &[f64::NAN, 1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0, f64::NAN],
            &[f64::NAN, f64::NAN, f64::NAN],
        ];

        for (idx, values) in nan_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = (*values).to_vec();
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-nan-propagation-{idx}"),
                    function,
                    RequirementLevel::Should,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_1d_f(py, &values)?]),
                    no_kwargs,
                );
            }
        }

        // ─── Inf handling tests (SHOULD) ───────────────────────────────────
        let inf_cases: &[&[f64]] = &[
            &[1.0, f64::INFINITY, 3.0],
            &[f64::NEG_INFINITY, 1.0, 2.0],
            &[1.0, 2.0, f64::INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::INFINITY],
        ];

        for (idx, values) in inf_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = (*values).to_vec();
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-inf-handling-{idx}"),
                    function,
                    RequirementLevel::Should,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_1d_f(py, &values)?]),
                    no_kwargs,
                );
            }
        }

        // ─── Signed-zero tests (SHOULD) ────────────────────────────────────
        let signed_zero_cases: &[&[f64]] = &[
            &[0.0, 0.0, 0.0],    // cumsum: [0, 0, 0], cumprod: [0, 0, 0]
            &[-0.0, -0.0, -0.0], // cumsum: [-0, -0, -0], cumprod: [-0, 0, -0]
            &[0.0, -0.0, 0.0],   // cumsum: [0, 0, 0], cumprod: [0, -0, -0]
            &[1.0, -0.0, 1.0],   // cumprod signed-zero in middle
        ];

        for (idx, values) in signed_zero_cases.iter().enumerate() {
            for function in ["cumsum", "cumprod"] {
                let values = (*values).to_vec();
                run_case(
                    py,
                    &module,
                    &numpy,
                    &format!("{function}-signed-zero-{idx}"),
                    function,
                    RequirementLevel::Should,
                    CompareMode::Close,
                    t,
                    move |py| PyTuple::new(py, [np_array_1d_f(py, &values)?]),
                    no_kwargs,
                );
            }
        }

        eprintln!("{}", TOTALS.summarize("cumsum-cumprod-native"));
        Ok(())
    });
}
