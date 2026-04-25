//! Conformance matrix: arithmetic-ops family.
//!
//! Covers element-wise arithmetic ufuncs: add, subtract, multiply,
//! divide, true_divide, floor_divide, mod, remainder, power, square,
//! negative, positive, fabs, conjugate, reciprocal, abs, absolute. The
//! goal is to lock in dtype-promotion rules (int+float → float64,
//! bool → int64), broadcasting (scalar↔array, 1-D↔2-D), and divide-by-
//! zero / NaN / Inf surfaces against the numpy reference.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_int<'py>(
    py: Python<'py>,
    items: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((items,))
}

fn np_array_float<'py>(
    py: Python<'py>,
    items: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((items,))
}

#[test]
fn conformance_arithmetic_ops_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // Build args closures take ownership of i64 by copy; we reuse
        // them for both ours and theirs since the harness rebuilds args
        // for each side.

        // ─── add (binary scalar / array / broadcasting) ─────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-add-scalar-int",
            "add",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        3_i64.into_pyobject(py)?.into_any(),
                        4_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-add-array-int",
            "add",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![1, 2, 3])?,
                        np_array_int(py, vec![10, 20, 30])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-add-broadcast-scalar-array",
            "add",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        10_i64.into_pyobject(py)?.into_any(),
                        np_array_int(py, vec![1, 2, 3])?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-add-int-float-promote",
            "add",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![1, 2, 3])?,
                        np_array_float(py, vec![0.5, 1.5, 2.5])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── subtract ────────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-subtract-array",
            "subtract",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![10, 20, 30])?,
                        np_array_int(py, vec![1, 2, 3])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-subtract-negative-result",
            "subtract",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![1, 2, 3])?,
                        np_array_int(py, vec![10, 20, 30])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── multiply ────────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-multiply-array",
            "multiply",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![1, 2, 3])?,
                        np_array_int(py, vec![4, 5, 6])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-multiply-zero-array",
            "multiply",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![1, 2, 3])?,
                        np_array_int(py, vec![0, 0, 0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-multiply-empty-broadcast",
            "multiply",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![])?, np_array_int(py, vec![])?]),
            no_kwargs,
        );

        // ─── divide / true_divide ────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-divide-int-promotes-float",
            "divide",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![10, 20, 30])?,
                        np_array_int(py, vec![2, 4, 6])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-divide-by-zero-float",
            "divide",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_float(py, vec![1.0, -1.0, 0.0])?,
                        np_array_float(py, vec![0.0, 0.0, 0.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-true_divide-int",
            "true_divide",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![7, 8, 9])?,
                        np_array_int(py, vec![2, 2, 2])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── floor_divide / mod / remainder ─────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-floor_divide-int",
            "floor_divide",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![7, 8, 9])?,
                        np_array_int(py, vec![2, 3, 4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-floor_divide-negative",
            "floor_divide",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![-7, -8, 9])?,
                        np_array_int(py, vec![2, 3, -4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-mod-int",
            "mod",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![7, 8, 9])?,
                        np_array_int(py, vec![2, 3, 4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-mod-negative-divisor",
            "mod",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![7, -8, 9])?,
                        np_array_int(py, vec![-3, 3, -4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-remainder-float",
            "remainder",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_float(py, vec![5.5, 7.25, -3.5])?,
                        np_array_float(py, vec![2.0, 1.5, 1.0])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── power ───────────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-power-int",
            "power",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![2, 3, 4])?,
                        np_array_int(py, vec![3, 2, 1])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-power-zero-exp",
            "power",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_int(py, vec![5, 7, 11])?,
                        np_array_int(py, vec![0, 0, 0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-power-float-fractional-exp",
            "power",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_float(py, vec![4.0, 9.0, 16.0])?,
                        np_array_float(py, vec![0.5, 0.5, 0.5])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── unary: square / negative / positive ─────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-square-int",
            "square",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![1, 2, 3, 4, 5])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-square-float-with-nan",
            "square",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_float(py, vec![1.0, f64::NAN, 3.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-negative-int",
            "negative",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![1, -2, 3, -4, 5])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-positive-int",
            "positive",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![1, -2, 3, -4, 5])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-positive-with-inf",
            "positive",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_float(
                        py,
                        vec![1.0, f64::INFINITY, f64::NEG_INFINITY],
                    )?],
                )
            },
            no_kwargs,
        );

        // ─── unary: fabs / conjugate / reciprocal ────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-fabs-float",
            "fabs",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_float(py, vec![-1.5, 2.5, -3.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-fabs-int-promotes-float",
            "fabs",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![-1, 2, -3])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-conjugate-real",
            "conjugate",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![1, -2, 3])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-reciprocal-float",
            "reciprocal",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_float(py, vec![1.0, 2.0, 4.0, 8.0])?]),
            no_kwargs,
        );

        // ─── abs / absolute (alias) ──────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-abs-int",
            "abs",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_int(py, vec![-1, 2, -3, 4])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "arith-absolute-float",
            "absolute",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_array_float(py, vec![-1.5, 2.0, -3.5])?]),
            no_kwargs,
        );

        // ─── 0-d / scalar edge case ──────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "arith-add-0d-array",
            "add",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                let a = py.import("numpy")?.getattr("int64")?.call1((7_i64,))?;
                let b = py.import("numpy")?.getattr("int64")?.call1((3_i64,))?;
                PyTuple::new(py, [a, b])
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("arithmetic_ops");
    eprintln!("\n=== fnp-python conformance matrix: arithmetic_ops ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in arithmetic_ops family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
