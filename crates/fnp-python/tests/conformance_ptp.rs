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

/// Locks the zero-copy f64 per-axis path (try_zerocopy_f64_ptp_axis): a large
/// contiguous float64 array reduced along axis 0 and axis 1, with NaN/inf/-0.0
/// rows, must be BYTE-IDENTICAL to numpy.ptp (numpy is the oracle), plus an
/// FNV-1a golden over the fnp bytes for drift. This exercises the borrowed-buffer
/// reduction that replaced the extract_precise_numeric_array full-input copy.
#[test]
fn ptp_zerocopy_f64_axis_matches_numpy_bytes_and_golden() {
    with_fnp_and_numpy(|py, module, numpy| {
        // Deterministic LCG f64 matrix (240x160) with embedded NaN/inf/-0.0.
        let locals = PyDict::new(py);
        locals.set_item("np", &numpy)?;
        py.run(
            std::ffi::CString::new(
                "import numpy as _np\n\
                 s=0x9E3779B9\n\
                 r,c=240,160\n\
                 M=_np.empty((r,c),dtype=_np.float64)\n\
                 for i in range(r):\n\
                 \x20 for j in range(c):\n\
                 \x20  s=(s*6364136223846793005+1)&0xFFFFFFFFFFFFFFFF\n\
                 \x20  M[i,j]=((s>>33)/4294967295.0)-0.5\n\
                 M[3,7]=_np.nan; M[50,80]=_np.inf; M[110,20]=-_np.inf\n\
                 M[200,100]=-0.0; M[200,101]=0.0\n",
            )?
            .as_c_str(),
            None,
            Some(&locals),
        )?;
        let m = locals.get_item("M")?.unwrap();

        let mut h: u64 = 0xcbf29ce484222325;
        for axis in [0i64, 1i64] {
            let kwargs = PyDict::new(py);
            kwargs.set_item("axis", axis)?;
            let actual = module
                .getattr("ptp")?
                .call((&m,), Some(&kwargs))?
                .call_method0("tobytes")?
                .extract::<Vec<u8>>()?;
            let expected = numpy
                .getattr("ptp")?
                .call((&m,), Some(&kwargs))?
                .call_method0("tobytes")?
                .extract::<Vec<u8>>()?;
            assert_eq!(
                actual, expected,
                "zero-copy f64 ptp(axis={axis}) must be byte-identical to numpy.ptp"
            );
            for b in &actual {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(
            h, 0x1df562ae07a2a4c0,
            "ptp zero-copy f64 axis golden FNV drifted"
        );
        Ok(())
    });
}
