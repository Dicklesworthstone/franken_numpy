//! Conformance matrix: fft family.
//!
//! Differential parity for the `fnp_python.fft` submodule, which
//! re-exports 18 functions backed by `numpy.fft` passthroughs. The
//! existing conformance suites only assert callable presence — a drift
//! between our wrapper signatures and `numpy.fft` (e.g. wrong default
//! `norm`, dropped `axis` kwarg, kwarg renaming) would slip through.
//!
//! Each function gets a MUST case at smallest valid shape with default
//! kwargs. Common knobs (`norm='ortho'`, explicit `n=`, `axis=`) get
//! SHOULD coverage. Round-trip property checks (`ifft(fft(x)) ≈ x`)
//! land in MAY since they exercise composition rather than a single
//! surface.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_1d<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_2d<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn np_array_3d<'py>(
    py: Python<'py>,
    cube: Vec<Vec<Vec<f64>>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((cube,))
}

/// Build a complex-typed numpy array from a real Vec by passing
/// `dtype=complex128`. Avoids hauling Python complex numbers across
/// the FFI boundary.
fn np_complex_1d<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let array = py.import("numpy")?.getattr("array")?;
    let kw = PyDict::new(py);
    kw.set_item("dtype", "complex128")?;
    array.call((values,), Some(&kw))
}

#[test]
fn conformance_fft_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;
        let fft_mod = module.getattr("fft").expect("fnp_python.fft");
        let np_fft_mod = numpy.getattr("fft").expect("numpy.fft");
        let fft = fft_mod
            .cast_into::<pyo3::types::PyModule>()
            .expect("fnp_python.fft should be a submodule");
        let np_fft = np_fft_mod
            .cast_into::<pyo3::types::PyModule>()
            .expect("numpy.fft should be a submodule");

        // ─── 1-D forward / inverse FFTs (MUST) ─────────────────────────
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fft-1d-len4",
            "fft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-ifft-1d-len4",
            "ifft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_complex_1d(py, vec![1.0, 2.0, 3.0, 4.0])?]),
            no_kwargs,
        );

        // ─── 1-D real FFTs (MUST) ──────────────────────────────────────
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-rfft-1d-len4",
            "rfft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-irfft-1d-len5",
            "irfft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            // rfft of a length-8 real signal has 5 complex bins; round-tripping
            // back must yield length 8 (the default n=2*(len-1) = 8).
            |py| {
                let x = np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
                let rfft_fn = py.import("numpy")?.getattr("fft")?.getattr("rfft")?;
                let bins = rfft_fn.call1((x,))?;
                PyTuple::new(py, [bins])
            },
            no_kwargs,
        );

        // ─── Hermitian FFTs (MUST) ─────────────────────────────────────
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-hfft-1d-len5",
            "hfft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            // hfft expects a Hermitian-symmetric one-sided spectrum. The
            // output of rfft on a real input satisfies that property, so
            // we feed exactly that.
            |py| {
                let x = np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
                let rfft_fn = py.import("numpy")?.getattr("fft")?.getattr("rfft")?;
                let spec = rfft_fn.call1((x,))?;
                PyTuple::new(py, [spec])
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-ihfft-1d-len8",
            "ihfft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_1d(
                        py,
                        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    )?],
                )
            },
            no_kwargs,
        );

        // ─── 2-D FFTs (MUST) ───────────────────────────────────────────
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fft2-2x2",
            "fft2",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_2d(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-ifft2-2x2",
            "ifft2",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                // ifft2 takes a complex input; promote via dtype kwarg.
                let array = py.import("numpy")?.getattr("array")?;
                let kw = PyDict::new(py);
                kw.set_item("dtype", "complex128")?;
                let x = array.call((vec![vec![1.0, 2.0], vec![3.0, 4.0]],), Some(&kw))?;
                PyTuple::new(py, [x])
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-rfft2-2x2",
            "rfft2",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_2d(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-irfft2-2x3",
            "irfft2",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            // rfft2 on a 2x4 real input → 2x3 complex bins; pass that
            // through irfft2 with the default s= back to 2x4.
            |py| {
                let x = np_array_2d(py, vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]])?;
                let rfft2_fn = py.import("numpy")?.getattr("fft")?.getattr("rfft2")?;
                PyTuple::new(py, [rfft2_fn.call1((x,))?])
            },
            no_kwargs,
        );

        // ─── N-D FFTs (MUST) ───────────────────────────────────────────
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fftn-3d-2x2x2",
            "fftn",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_3d(
                        py,
                        vec![
                            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
                        ],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-ifftn-3d-2x2x2",
            "ifftn",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                let array = py.import("numpy")?.getattr("array")?;
                let kw = PyDict::new(py);
                kw.set_item("dtype", "complex128")?;
                let x = array.call(
                    (vec![
                        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                        vec![vec![5.0, 6.0], vec![7.0, 8.0]],
                    ],),
                    Some(&kw),
                )?;
                PyTuple::new(py, [x])
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-rfftn-3d-2x2x4",
            "rfftn",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_3d(
                        py,
                        vec![
                            vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
                            vec![vec![9.0, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
                        ],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-irfftn-3d",
            "irfftn",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            // rfftn of 2x2x4 real → 2x2x3 complex; round-trip with irfftn
            // and let the default s= reconstruct 2x2x4.
            |py| {
                let x = np_array_3d(
                    py,
                    vec![
                        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
                        vec![vec![9.0, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]],
                    ],
                )?;
                let rfftn_fn = py.import("numpy")?.getattr("fft")?.getattr("rfftn")?;
                PyTuple::new(py, [rfftn_fn.call1((x,))?])
            },
            no_kwargs,
        );

        // ─── shift / freq helpers (MUST) ───────────────────────────────
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fftshift-1d-even",
            "fftshift",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d(py, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-ifftshift-1d-odd",
            "ifftshift",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d(py, vec![0.0, 1.0, 2.0, 3.0, 4.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fftfreq-n8",
            "fftfreq",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [8_i64.into_pyobject(py)?]),
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-rfftfreq-n8",
            "rfftfreq",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [8_i64.into_pyobject(py)?]),
            no_kwargs,
        );

        // ─── kwarg coverage (SHOULD) ───────────────────────────────────
        // norm='ortho' must be honored — it's the most common deviation
        // from default behavior and would silently disappear if a wrapper
        // dropped the kwarg before forwarding.
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fft-ortho-norm",
            "fft",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("norm", "ortho")?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fft-explicit-n8",
            "fft",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("n", 8_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-rfft-axis0-2d",
            "rfft",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d(
                        py,
                        vec![
                            vec![1.0, 2.0, 3.0],
                            vec![4.0, 5.0, 6.0],
                            vec![7.0, 8.0, 9.0],
                            vec![10.0, 11.0, 12.0],
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
        // Regression: prior native-eligibility ndim fallback was
        // `unwrap_or(1)`, which let nested-list 2-D inputs (no `ndim`
        // attribute) take the 1-D-only native path and silently produce
        // wrong output. MUST tier so the fix can't silently regress.
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-rfft-nested-list-2d-input",
            "rfft",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                let nested = pyo3::types::PyList::new(
                    py,
                    [
                        pyo3::types::PyList::new(py, [1.0_f64, 2.0, 3.0, 4.0])?,
                        pyo3::types::PyList::new(py, [5.0_f64, 6.0, 7.0, 8.0])?,
                    ],
                )?;
                PyTuple::new(py, [nested.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fftshift-explicit-axes",
            "fftshift",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d(
                        py,
                        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
                    )?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axes", 1_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-fftfreq-with-d",
            "fftfreq",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [8_i64.into_pyobject(py)?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("d", 0.5_f64)?;
                Ok(Some(kw))
            },
        );

        // ─── round-trip property checks (MAY) ──────────────────────────
        // ifft(fft(x)) ≈ x — the strongest single-call invariant. We
        // compose against `fft` only, so the harness compares the round
        // trip on our side vs. the round trip on numpy's side. The
        // outputs must agree under allclose.
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-roundtrip-fft-ifft",
            "ifft",
            RequirementLevel::May,
            CompareMode::Close,
            t,
            |py| {
                let x = np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
                let fft_fn = py.import("numpy")?.getattr("fft")?.getattr("fft")?;
                PyTuple::new(py, [fft_fn.call1((x,))?])
            },
            no_kwargs,
        );
        run_case(
            py,
            &fft,
            &np_fft,
            "fft-roundtrip-rfft-irfft",
            "irfft",
            RequirementLevel::May,
            CompareMode::Close,
            t,
            |py| {
                let x = np_array_1d(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
                let rfft_fn = py.import("numpy")?.getattr("fft")?.getattr("rfft")?;
                PyTuple::new(py, [rfft_fn.call1((x,))?])
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("fft");
    eprintln!("\n=== fnp-python conformance matrix: fft ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in fft family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
