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
        type ComplexConvCase = (
            &'static str,
            &'static str,
            &'static [(f64, f64)],
            &'static [(f64, f64)],
            &'static str,
        );
        let complex_cases: &[ComplexConvCase] = &[
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
                    PyTuple::new(
                        py,
                        [np_array_1d_complex(py, &a)?, np_array_1d_complex(py, &v)?],
                    )
                },
                move |py| mode_kwargs(py, &mode),
            );
        }

        eprintln!("{}", TOTALS.summarize("convolution-fnp-python"));
        Ok(())
    });
}

#[test]
fn int_convolve_correlate_native_parallel_bit_exact_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        ns.set_item("fnp", &module)?;
        ns.set_item("np", &numpy)?;
        let script = r#"
rng = np.random.default_rng(31)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint8]:
    info = np.iinfo(dt)
    a = rng.integers(info.min // 4, info.max // 4, 5000).astype(dt)
    v = rng.integers(info.min // 4, info.max // 4, 300).astype(dt)
    for mode in ['full', 'same', 'valid']:
        rc = fnp.convolve(a, v, mode); ec = np.convolve(a, v, mode)
        ok = ok and rc.dtype == ec.dtype and rc.shape == ec.shape and rc.tobytes() == ec.tobytes()
        rk = fnp.correlate(a, v, mode); ek = np.correlate(a, v, mode)
        ok = ok and rk.dtype == ek.dtype and rk.shape == ek.shape and rk.tobytes() == ek.tobytes()
    a2 = rng.integers(info.min // 8, info.max // 8, 200).astype(dt)
    v2 = rng.integers(info.min // 8, info.max // 8, 4000).astype(dt)
    for mode in ['full', 'same', 'valid']:
        ok = ok and fnp.convolve(a2, v2, mode).tobytes() == np.convolve(a2, v2, mode).tobytes()
a = np.full(4000, 5_000_000_000, dtype=np.int64)
v = np.full(300, 5_000_000_000, dtype=np.int64)
ok = ok and fnp.convolve(a, v, 'full').tobytes() == np.convolve(a, v, 'full').tobytes()
result_ok = bool(ok)
"#;
        py.run(
            std::ffi::CString::new(script).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )?;
        let ok: bool = ns.get_item("result_ok")?.unwrap().extract()?;
        assert!(
            ok,
            "native integer convolve/correlate must be bit-identical to numpy"
        );
        Ok(())
    });
}
