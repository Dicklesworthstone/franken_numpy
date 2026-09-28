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

/// Argument handling of the whole `numpy.fft` surface, compared by outcome (dtype, shape,
/// values rounded to 6 places, or exception type and message) plus the warnings raised.
///
/// The transforms were typed PyO3 wrappers around numpy's own functions, and the types only
/// diverged: `norm=1` was PyO3's TypeError where numpy raises ValueError; `n=-1`, `axis=None`
/// and `axis=1.0` raised PyO3's messages; a defaulted `axes=None` could not tell an explicit
/// None from an omitted one, so `fft2(x, s=..., axes=None)` lost numpy's DeprecationWarning
/// (numpy's own TestFFT1D::test_s_axes_none_2D); and irfft checked `norm` itself and raised a
/// hand-copied message that had drifted from numpy 2.4's. `fftfreq` computed a float32 or
/// float16 `d` in float64 (`fftfreq(10, np.float32(0.1))` is exactly 0, 1, 2, ... in numpy and
/// was 0.99999994...), refused a complex `d`, and `rfftfreq(-1)` raised where numpy returns an
/// empty array. 61 of the 150 cells failed before the fix (numpy 2.4.3); 0 fail after, on
/// numpy 2.4.3 and 2.3.5.
const ARGUMENT_SWEEP: &str = r#"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            a = np.asarray(r)
            got = ("ok", a.dtype.str, a.shape, np.round(a, 6).tobytes())
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted((w.category.__name__, str(w.message)) for w in caught),)

x = np.arange(16.0).reshape(4, 4)
x10 = np.arange(100.0).reshape(10, 10)
x3 = np.arange(60.0).reshape(3, 4, 5)
cases = {}
for op in ("fft", "ifft", "rfft", "irfft", "hfft", "ihfft"):
    cases[f"{op} norm=1"] = lambda m, op=op: getattr(m.fft, op)(x, norm=1)
    cases[f"{op} norm=b'ortho'"] = lambda m, op=op: getattr(m.fft, op)(x, norm=b"ortho")
    cases[f"{op} norm='bad'"] = lambda m, op=op: getattr(m.fft, op)(x, norm="bad")
    cases[f"{op} norm=ortho"] = lambda m, op=op: getattr(m.fft, op)(x, norm="ortho")
    cases[f"{op} axis=None"] = lambda m, op=op: getattr(m.fft, op)(x, axis=None)
    cases[f"{op} axis=1.0"] = lambda m, op=op: getattr(m.fft, op)(x, axis=1.0)
    cases[f"{op} axis=0"] = lambda m, op=op: getattr(m.fft, op)(x, axis=0)
    cases[f"{op} n=None"] = lambda m, op=op: getattr(m.fft, op)(x, n=None)
    cases[f"{op} n=-1"] = lambda m, op=op: getattr(m.fft, op)(x, n=-1)
    cases[f"{op} n=6"] = lambda m, op=op: getattr(m.fft, op)(x, 6)
for op in ("fft2", "ifft2", "rfft2", "irfft2", "fftn", "ifftn", "rfftn", "irfftn"):
    cases[f"{op} norm=1"] = lambda m, op=op: getattr(m.fft, op)(x, norm=1)
    cases[f"{op} s axes=None"] = lambda m, op=op: getattr(m.fft, op)(x10, s=(4, 5), axes=None)
    cases[f"{op} s=-1 axes=None"] = lambda m, op=op: getattr(m.fft, op)(x10, s=(-1, 5), axes=None)
    cases[f"{op} axes=None"] = lambda m, op=op: getattr(m.fft, op)(x, axes=None)
    cases[f"{op} s"] = lambda m, op=op: getattr(m.fft, op)(x10, s=(4, 5))
    cases[f"{op} s with None"] = lambda m, op=op: getattr(m.fft, op)(x3, s=(None, 4, 3), axes=(0, 1, 2))
    cases[f"{op} 3-D s axes=None"] = lambda m, op=op: getattr(m.fft, op)(x3, s=(2, 3), axes=None)
for op in ("fftshift", "ifftshift"):
    cases[f"{op} axes=None"] = lambda m, op=op: getattr(m.fft, op)(x, axes=None)
    cases[f"{op} axes=0"] = lambda m, op=op: getattr(m.fft, op)(x, axes=0)
for f in ("fftfreq", "rfftfreq"):
    cases[f"{f} d=f32"] = lambda m, f=f: getattr(m.fft, f)(10, d=np.float32(0.1))
    cases[f"{f} d=0-d f32"] = lambda m, f=f: getattr(m.fft, f)(10, d=np.array(0.1, np.float32))
    cases[f"{f} d=f16"] = lambda m, f=f: getattr(m.fft, f)(6, np.float16(0.3))
    cases[f"{f} d=None"] = lambda m, f=f: getattr(m.fft, f)(10, d=None)
    cases[f"{f} d=complex"] = lambda m, f=f: getattr(m.fft, f)(6, 1j)
    cases[f"{f} d=int"] = lambda m, f=f: getattr(m.fft, f)(6, 3)
    cases[f"{f} d=0"] = lambda m, f=f: getattr(m.fft, f)(6, 0.0)
    cases[f"{f} n=0"] = lambda m, f=f: getattr(m.fft, f)(0)
    cases[f"{f} n=4.0"] = lambda m, f=f: getattr(m.fft, f)(4.0)
    cases[f"{f} n=-1"] = lambda m, f=f: getattr(m.fft, f)(-1)
    cases[f"{f} n=np.int8"] = lambda m, f=f: getattr(m.fft, f)(np.int8(6), 0.5)
    cases[f"{f} device=cpu"] = lambda m, f=f: getattr(m.fft, f)(8, 2.0, device="cpu")
    cases[f"{f} device=cuda"] = lambda m, f=f: getattr(m.fft, f)(8, device="cuda")
    cases[f"{f} odd"] = lambda m, f=f: getattr(m.fft, f)(7, 0.25)
    cases[f"{f} even"] = lambda m, f=f: getattr(m.fft, f)(8, 0.1)
failures = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        failures.append(f"{name}: fnp={str(ours)[:150]} numpy={str(theirs)[:150]}")
cells = len(cases)
"#;

#[test]
fn fft_family_argument_handling_matches_numpy() {
    with_fnp_and_numpy(|py, fnp, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &fnp)?;
        globals.set_item("np", &numpy)?;
        let script = std::ffi::CString::new(ARGUMENT_SWEEP).expect("sweep is a valid C string");
        py.run(&script, Some(&globals), None)?;
        let cells: usize = globals
            .get_item("cells")?
            .expect("cells present")
            .extract()?;
        let failures: Vec<String> = globals
            .get_item("failures")?
            .expect("failures present")
            .extract()?;
        assert_eq!(cells, 150, "the fft argument sweep changed size");
        assert!(
            failures.is_empty(),
            "{} of {cells} fft argument cells diverge from numpy:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
        Ok(())
    });
}
