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
        TOTALS.assert_no_failures("convolution-fnp-python");
        Ok(())
    });
}

#[test]
fn f64_convolution_and_correlation_preserve_numpy_raw_bytes() {
    // This is deliberately a raw-byte, finite-value negative case.  The previous
    // native f64 gather followed a different reduction order and was close but not
    // identical to NumPy for this 256x128 input (up to 1.1e-12).  `allclose` would
    // accept that incorrect implementation, so it cannot protect this contract.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        ns.set_item("fnp", &module)?;
        ns.set_item("np", &numpy)?;
        let script = r#"
a = np.arange(256, dtype=np.float64) / 17.0 - 3.0
v = np.arange(128, dtype=np.float64) / 19.0 - 2.0
ok = True
for op in ("convolve", "correlate"):
    native = getattr(fnp, op)
    oracle = getattr(np, op)
    for mode in ("full", "same", "valid"):
        got = native(a, v, mode)
        expected = oracle(a, v, mode)
        ok = ok and got.dtype == expected.dtype and got.shape == expected.shape
        ok = ok and got.tobytes() == expected.tobytes()
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
            "f64 convolve/correlate must preserve NumPy raw output bytes"
        );
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
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8]:
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
    # Equal lengths: 'valid' has ONE output and goes to numpy, 'full'/'same' stay native; and a
    # long signal with a 3-tap kernel, which the per-task minimum splits into large chunks.
    a3 = rng.integers(info.min // 8, info.max // 8, 4000).astype(dt)
    v3 = rng.integers(info.min // 8, info.max // 8, 4000).astype(dt)
    a4 = rng.integers(info.min // 8, info.max // 8, 1 << 17).astype(dt)
    v4 = np.array([1, 5, 9]).astype(dt)
    for x, y in ((a3, v3), (a4, v4), (v4, a4)):
        for mode in ['full', 'same', 'valid']:
            for name in ('convolve', 'correlate'):
                ours, theirs = getattr(fnp, name)(x, y, mode), getattr(np, name)(x, y, mode)
                ok = ok and ours.dtype == theirs.dtype and ours.tobytes() == theirs.tobytes()
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

/// The integer convolve / correlate route is parallel over outputs, so it answers a call only
/// when there are at least two tasks' worth of outputs (each task 2^16 multiply-adds) and at
/// least 64 - decided from the shapes. A spy on numpy's own function must see no call where the
/// route answers (a 64-tap 'valid' correlate of 4096 elements: 4033 outputs, which a fixed floor
/// of 4096 outputs sent to numpy) and one call where it declines (a single-output 'valid'
/// correlate of equal lengths, 41 outputs), with numpy's bytes either way.
#[test]
fn int_convolve_correlate_route_engages_by_output_tasks() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        ns.set_item("fnp", &module)?;
        ns.set_item("np", &numpy)?;
        let script = r#"
def numpy_calls(name, a, v, mode):
    real, calls = getattr(np, name), []
    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    setattr(np, name, spy)
    try:
        ours = getattr(fnp, name)(a, v, mode)
    finally:
        setattr(np, name, real)
    return len(calls), ours

rng = np.random.default_rng(41)
bad = []
# (len(a), len(v), mode, native): outputs vs max(2 * outputs per task, 64)
cells = [(4096, 64, "valid", True), (4096 + 100, 4096, "valid", True),
         (4096, 4096, "full", True), (4096, 4096, "valid", False),
         (4096 + 40, 4096, "valid", False), (100, 1000, "full", False)]
# A host whose pool cannot run the route (one thread) answers everything through numpy.
probe = rng.integers(-9, 9, 4096).astype(np.int64)
pool = numpy_calls("convolve", probe, probe, "full")[0] == 0
for dt in (np.int64, np.int8):
    for n, m, mode, native in cells:
        a = rng.integers(-9, 9, n).astype(dt)
        v = rng.integers(-9, 9, m).astype(dt)
        for name in ("convolve", "correlate"):
            calls, ours = numpy_calls(name, a, v, mode)
            theirs = getattr(np, name)(a, v, mode)
            if ours.dtype != theirs.dtype or ours.tobytes() != theirs.tobytes():
                bad.append(f"{np.dtype(dt).name} {name} {n}x{m} {mode} bytes")
            if pool and calls != (0 if native else 1):
                bad.append(f"{np.dtype(dt).name} {name} {n}x{m} {mode} numpy calls {calls}")
result = (pool, bad)
"#;
        py.run(
            std::ffi::CString::new(script).unwrap().as_c_str(),
            Some(&ns),
            Some(&ns),
        )?;
        let result = ns.get_item("result")?.unwrap();
        let bad: Vec<String> = result.get_item(1)?.extract()?;
        assert!(
            bad.is_empty(),
            "integer convolve/correlate must engage by output tasks with numpy's bytes \
             (pool={}): {bad:?}",
            result.get_item(0)?
        );
        Ok(())
    });
}
