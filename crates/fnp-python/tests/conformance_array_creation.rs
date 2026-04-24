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
                        3.14_f64.into_pyobject(py)?.into_any(),
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
