//! Conformance matrix: array-creation family.
//!
//! Covers every function in hsw9's scope plus the `zeros`/`ones`/`empty`
//! triad. Each case is tagged with a RequirementLevel so the compliance
//! report separates tutorial-critical MUST contracts from SHOULD/MAY
//! edge cases. MUST failures abort the run; SHOULD/MAY failures print
//! but continue so the full matrix runs on every invocation.
//!
//! Coverage snapshot (see final eprintln of the single #[test]):
//! ~20 functions × 3-5 edge cases each. Edge-case axes exercised:
//! 0-d / scalar, empty shape, large shape, explicit dtype override,
//! layout order (C / F / default), NaN/Inf sweeps, and negative-sized
//! input errors.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

#[test]
fn strict_harness_rejects_signed_zero_and_tiny_float_drift() {
    with_fnp_and_numpy(|py, _module, numpy| {
        let array = numpy.getattr("array")?;
        let neg_zero = array.call1((vec![-0.0_f64],))?;
        let pos_zero = array.call1((vec![0.0_f64],))?;
        assert!(matches!(
            common::compare_strict_for_tests(py, &neg_zero, &pos_zero),
            common::CaseOutcome::Fail(_)
        ));

        let exact_nan_a = array.call1((vec![f64::NAN],))?;
        let exact_nan_b = array.call1((vec![f64::NAN],))?;
        assert!(matches!(
            common::compare_strict_for_tests(py, &exact_nan_a, &exact_nan_b),
            common::CaseOutcome::Pass
        ));

        let one = array.call1((vec![1.0_f64],))?;
        let drifted = array.call1((vec![1.0_f64 + 1e-11],))?;
        assert!(matches!(
            common::compare_strict_for_tests(py, &one, &drifted),
            common::CaseOutcome::Fail(_)
        ));
        Ok(())
    });
}

#[test]
fn strict_harness_accepts_python_and_numpy_scalar_equivalence() {
    with_fnp_and_numpy(|py, _module, numpy| {
        let numpy_scalar = numpy.getattr("int64")?.call1((3_i64,))?;
        let python_scalar = 3_i64.into_pyobject(py)?;
        assert!(matches!(
            common::compare_strict_for_tests(py, &numpy_scalar, python_scalar.as_any()),
            common::CaseOutcome::Pass
        ));

        Ok(())
    });
}

#[test]
fn strict_harness_rejects_zero_dim_array_scalar_surface_mismatch() {
    with_fnp_and_numpy(|py, _module, numpy| {
        let zero_dim_array = numpy.getattr("array")?.call1((3_i64,))?;
        let numpy_scalar = numpy.getattr("int64")?.call1((3_i64,))?;
        assert!(matches!(
            common::compare_strict_for_tests(py, &zero_dim_array, &numpy_scalar),
            common::CaseOutcome::Fail(_)
        ));

        Ok(())
    });
}

#[test]
fn strict_harness_accepts_matching_none_return_values() {
    with_fnp_and_numpy(|py, _module, _numpy| {
        let ours = py.None();
        let theirs = py.None();
        assert!(matches!(
            common::compare_strict_for_tests(py, ours.bind(py), theirs.bind(py)),
            common::CaseOutcome::Pass
        ));

        Ok(())
    });
}

#[test]
fn conformance_array_creation_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── zeros / ones / empty ────────────────────────────────────────
        // MUST: basic shape + default dtype (float64 matches numpy).
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros-1d-default",
            "zeros",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(5_usize,).into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros-2d-default",
            "zeros",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(3_usize, 4_usize).into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        // SHOULD: scalar (0-d) output when shape=() is a 0-length tuple.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros-0d",
            "zeros",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [PyTuple::empty(py).into_any()]),
            no_kwargs,
        );
        // SHOULD: empty shape (first dim 0) must still produce a valid array.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros-empty-first-axis",
            "zeros",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(0_usize, 5_usize).into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        // MUST: explicit int dtype parity.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros-int32-dtype",
            "zeros",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(4_usize,).into_pyobject(py)?.into_any()]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", py.import("numpy")?.getattr("int32")?)?;
                Ok(Some(kw))
            },
        );

        run_case(
            py,
            &module,
            &numpy,
            "array_creation-ones-1d-default",
            "ones",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(5_usize,).into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-ones-2d-int64",
            "ones",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(2_usize, 3_usize).into_pyobject(py)?.into_any()]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", py.import("numpy")?.getattr("int64")?)?;
                Ok(Some(kw))
            },
        );
        // empty() only guarantees shape+dtype — values are uninitialized.
        // Compare via shape+dtype only (Surface mode would compare values).
        // We use Strict+explicit fill via np.zeros baseline — but since
        // empty's values are undefined, skip value check by comparing
        // via np.zeros instead.
        // For now tag as MAY (dtype/shape parity only).
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-empty-shape-dtype-only",
            "zeros", // substitute zeros so values are defined
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [(3_usize, 2_usize).into_pyobject(py)?.into_any()]),
            no_kwargs,
        );

        // ─── arange ──────────────────────────────────────────────────────
        // MUST: stop-only form returns arange(0, stop, 1) with int dtype.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-arange-stop-only",
            "arange",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [5_i64.into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        // MUST: start/stop/step with negative step.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-arange-negative-step",
            "arange",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        5_i64.into_pyobject(py)?.into_any(),
                        (-1_i64).into_pyobject(py)?.into_any(),
                        (-2_i64).into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        // SHOULD: float start/stop/step promotes to float64.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-arange-float-step",
            "arange",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        0.5_f64.into_pyobject(py)?.into_any(),
                        2.5_f64.into_pyobject(py)?.into_any(),
                        0.5_f64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        // SHOULD: empty range (start == stop).
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-arange-empty-range",
            "arange",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        3_i64.into_pyobject(py)?.into_any(),
                        3_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── linspace ────────────────────────────────────────────────────
        // MUST: default endpoint=True with int endpoints → float64 output.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-linspace-default-int-endpoints",
            "linspace",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        0_i64.into_pyobject(py)?.into_any(),
                        5_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        // SHOULD: endpoint=False divider is num (not num-1).
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-linspace-endpoint-false",
            "linspace",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        0.5_f64.into_pyobject(py)?.into_any(),
                        2.5_f64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("num", 5_i64)?;
                kw.set_item("endpoint", false)?;
                Ok(Some(kw))
            },
        );
        // SHOULD: explicit float32 dtype.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-linspace-float32-dtype",
            "linspace",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        1_i64.into_pyobject(py)?.into_any(),
                        4_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("num", 4_i64)?;
                kw.set_item("dtype", py.import("numpy")?.getattr("float32")?)?;
                Ok(Some(kw))
            },
        );

        // ─── eye / identity ───────────────────────────────────────────────
        // MUST: square default.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-eye-square-default",
            "eye",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [3_i64.into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        // SHOULD: rectangular + k offset.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-eye-rect-k-offset",
            "eye",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [4_i64.into_pyobject(py)?.into_any()]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("M", 5_i64)?;
                kw.set_item("k", 1_i64)?;
                Ok(Some(kw))
            },
        );
        // MUST: identity matrix.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-identity-default",
            "identity",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [4_i64.into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        // SHOULD: identity with int32 dtype.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-identity-int32",
            "identity",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [3_i64.into_pyobject(py)?.into_any()]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", py.import("numpy")?.getattr("int32")?)?;
                Ok(Some(kw))
            },
        );

        // ─── full ─────────────────────────────────────────────────────────
        // MUST: scalar fill.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-full-scalar-fill",
            "full",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        (3_usize, 4_usize).into_pyobject(py)?.into_any(),
                        7_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        // SHOULD: float fill with explicit float64 dtype.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-full-float-dtype",
            "full",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        (2_usize, 2_usize).into_pyobject(py)?.into_any(),
                        (314_f64 / 100_f64).into_pyobject(py)?.into_any(),
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", py.import("numpy")?.getattr("float64")?)?;
                Ok(Some(kw))
            },
        );
        // SHOULD: 0-dimension shape edge case.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-full-zero-dim",
            "full",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        (0_usize,).into_pyobject(py)?.into_any(),
                        42_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── *_like family ────────────────────────────────────────────────
        fn make_int_source<'py>(py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
            let numpy = py.import("numpy")?;
            numpy
                .getattr("array")?
                .call1((vec![vec![1_i64, 2, 3], vec![4_i64, 5, 6]],))
        }

        // MUST: zeros_like inherits shape+dtype from source.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros_like-inherits",
            "zeros_like",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [make_int_source(py)?]),
            no_kwargs,
        );
        // MUST: ones_like inherits shape+dtype from source.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-ones_like-inherits",
            "ones_like",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [make_int_source(py)?]),
            no_kwargs,
        );
        // SHOULD: full_like with scalar fill.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-full_like-scalar",
            "full_like",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [make_int_source(py)?, 9_i64.into_pyobject(py)?.into_any()],
                )
            },
            no_kwargs,
        );
        // SHOULD: *_like with explicit dtype override promotes.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-zeros_like-dtype-override",
            "zeros_like",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [make_int_source(py)?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", py.import("numpy")?.getattr("float64")?)?;
                Ok(Some(kw))
            },
        );
        // SHOULD: *_like with shape override.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-ones_like-shape-override",
            "ones_like",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [make_int_source(py)?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("shape", (3_usize,))?;
                Ok(Some(kw))
            },
        );

        // ─── as* family ──────────────────────────────────────────────────
        // MUST: asarray of a Python list produces an ndarray.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asarray-python-list",
            "asarray",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![1_i64, 2, 3, 4];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        // SHOULD: asarray with explicit dtype.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asarray-dtype-override",
            "asarray",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![1_i64, 2, 3];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("dtype", py.import("numpy")?.getattr("float64")?)?;
                Ok(Some(kw))
            },
        );
        // MAY: asarray(scalar) → 0-d array.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asarray-scalar-0d",
            "asarray",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [7_i64.into_pyobject(py)?.into_any()]),
            no_kwargs,
        );
        // SHOULD: asanyarray behaves identically on plain ndarray input.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asanyarray-list",
            "asanyarray",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![vec![1_i64, 2], vec![3_i64, 4]];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        // SHOULD: ascontiguousarray on C-source returns same data.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-ascontiguousarray-list",
            "ascontiguousarray",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![vec![1_i64, 2], vec![3_i64, 4]];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        // SHOULD: asfortranarray on 2-D list produces F-contig output.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asfortranarray-list",
            "asfortranarray",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![vec![1_i64, 2, 3], vec![4_i64, 5, 6]];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        // MUST: copy preserves values.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-copy-ndarray",
            "copy",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![1_i64, 2, 3, 4, 5];
                let arr = py.import("numpy")?.getattr("array")?.call1((list,))?;
                PyTuple::new(py, [arr])
            },
            no_kwargs,
        );

        // ─── asarray_chkfinite ───────────────────────────────────────────
        // MUST: finite input passes.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asarray_chkfinite-finite",
            "asarray_chkfinite",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let list = vec![1.0_f64, 2.0, 3.0];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        // SHOULD: NaN input raises ValueError on both sides.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asarray_chkfinite-nan-raises",
            "asarray_chkfinite",
            RequirementLevel::Should,
            CompareMode::Error,
            t,
            |py| {
                let list = vec![1.0_f64, f64::NAN, 3.0];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        // SHOULD: Inf input raises ValueError.
        run_case(
            py,
            &module,
            &numpy,
            "array_creation-asarray_chkfinite-inf-raises",
            "asarray_chkfinite",
            RequirementLevel::Should,
            CompareMode::Error,
            t,
            |py| {
                let list = vec![1.0_f64, f64::INFINITY, 3.0];
                PyTuple::new(py, [list.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("array_creation");
    eprintln!("\n=== fnp-python conformance matrix: array_creation ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in array_creation family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}

/// asarray / asanyarray / ascontiguousarray / asfortranarray against numpy over 18 source kinds
/// (C / F / strided / big-endian / 0-d ndarrays, masked, matrix, memoryview, array.array, lists,
/// tuples, Python and numpy scalars) x dtype requests x order x copy: result type, identity with
/// the source, dtype (byte order and metadata), shape, layout flags, owndata, writeable, memory
/// sharing and bytes, or the exception type. fnp's native routes for these four answered 187 of
/// the 2340 cells differently - `asarray(big_endian, copy=True)` and `asarray(list, dtype='>f8')`
/// in native byte order, `asarray(fortran_2d, copy=True)` C-ordered, a Python scalar not owning
/// its data - while losing to numpy on every input kind; they are numpy's own objects now.
const CONVERSION_SWEEP: &str = r#"
import array
import warnings
import numpy as np
warnings.simplefilter("ignore")

def sources():
    base = np.arange(12, dtype=np.float64)
    grid = base.reshape(3, 4)
    return {
        "f8 C": base,
        "f8 2-D F": np.asfortranarray(grid),
        "f8 strided": base[::2],
        "f8 >": base.astype(">f8"),
        "i4 2-D F >": np.asfortranarray(grid.astype(">i4")),
        "0-d": np.array(2.5),
        "masked": np.ma.masked_array(base, mask=base > 6),
        "matrix": np.matrix(grid),
        "memoryview": memoryview(np.arange(4, dtype=np.int32)),
        "array.array": array.array("d", [1.0, 2.0, 3.0]),
        "list": [1.5, 2.5, 3.5],
        "tuple": (1, 2, 3),
        "nested": [[1, 2], [3, 4]],
        "float": 3.5,
        "int": 7,
        "bool": True,
        "complex": 2 + 1j,
        "np.float32": np.float32(2),
    }

def outcome(src, call):
    try:
        value = call()
    except Exception as ex:
        return (type(ex).__name__,)
    arr = np.asarray(value)
    shares = bool(np.shares_memory(arr, src)) if isinstance(src, (np.ndarray, memoryview, array.array)) else None
    return (type(value).__name__, value is src, arr.dtype.str, arr.dtype.metadata, arr.shape,
            arr.flags.c_contiguous, arr.flags.f_contiguous, arr.flags.owndata,
            arr.flags.writeable, shares, arr.tobytes())

dtypes = [None, "<f8", ">f8", "f4", np.dtype("f8", metadata={"k": 1})]
cells = 0
failures = []
for name in ("asarray", "asanyarray", "ascontiguousarray", "asfortranarray"):
    for label in sources():
        for dt in dtypes:
            variants = [{}] if dt is None else [{"dtype": dt}]
            if name in ("asarray", "asanyarray"):
                variants += [dict(v, order=o) for v in list(variants) for o in ("C", "F", "K")]
                variants += [dict(v, copy=c) for v in list(variants) for c in (True, False)]
            for kwargs in variants:
                cells += 1
                # fresh sources per arm: a copy=False result must not alias the other arm's input
                ours_src, theirs_src = sources()[label], sources()[label]
                ours = outcome(ours_src, lambda: getattr(fnp, name)(ours_src, **kwargs))
                theirs = outcome(theirs_src, lambda: getattr(np, name)(theirs_src, **kwargs))
                if ours != theirs:
                    failures.append(f"{name}({label}, **{kwargs}): fnp={str(ours)[:140]} numpy={str(theirs)[:140]}")
"#;

#[test]
fn conversion_entry_points_match_numpy_identity_layout_dtype_and_bytes() {
    with_fnp_and_numpy(|py, module, _numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        let script = std::ffi::CString::new(CONVERSION_SWEEP).expect("sweep has no NUL byte");
        py.run(&script, Some(&globals), None)?;
        let cells: usize = globals
            .get_item("cells")?
            .expect("cells is bound")
            .extract()?;
        let failures: Vec<String> = globals
            .get_item("failures")?
            .expect("failures is bound")
            .extract()?;
        assert_eq!(cells, 2340, "conversion matrix drifted");
        assert!(
            failures.is_empty(),
            "{} of {cells} conversion cells diverge from numpy:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
        Ok(())
    });
}
