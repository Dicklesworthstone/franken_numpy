//! Conformance matrix: array_manip family.
//!
//! Differential parity for fnp_python's array-manipulation surface:
//!
//!   reshape, transpose, swapaxes, moveaxis, expand_dims, squeeze,
//!   concatenate, stack, vstack, hstack, tile, repeat, flip, roll
//!
//! All are native bodies that delegate to numpy on edge cases. The
//! harness pins MUST cases at smallest valid shapes, SHOULD cases on
//! the kwargs/positional args most prone to silent drift (`axis=`,
//! `order=`, `axes=`), and MAY cases on degenerate shapes (1x1, empty).

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_1d_f<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_2d_f<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn np_array_3d_f<'py>(
    py: Python<'py>,
    cube: Vec<Vec<Vec<f64>>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((cube,))
}

#[test]
fn conformance_array_manip_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── reshape (MUST + SHOULD order=) ────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-reshape-1d-to-2x3",
            "reshape",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.into_any(),
                        PyTuple::new(py, [2_i64, 3])?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-reshape-fortran-order",
            "reshape",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.into_any(),
                        PyTuple::new(py, [2_i64, 3])?.into_any(),
                        "F".into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── transpose (MUST + SHOULD axes=) ───────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-transpose-2d-default",
            "transpose",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_2d_f(
                        py,
                        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-transpose-3d-permute",
            "transpose",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_3d_f(
                            py,
                            vec![
                                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
                            ],
                        )?
                        .into_any(),
                        PyTuple::new(py, [2_i64, 0, 1])?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── swapaxes (MUST) ───────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-swapaxes-3d-0-2",
            "swapaxes",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_3d_f(
                            py,
                            vec![
                                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
                            ],
                        )?
                        .into_any(),
                        0_i64.into_pyobject(py)?.into_any(),
                        2_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── moveaxis (MUST) ───────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-moveaxis-3d",
            "moveaxis",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_3d_f(
                            py,
                            vec![
                                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
                            ],
                        )?
                        .into_any(),
                        0_i64.into_pyobject(py)?.into_any(),
                        (-1_i64).into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── expand_dims (MUST) ────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-expand_dims-axis0",
            "expand_dims",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0])?.into_any(),
                        0_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── squeeze (MUST + SHOULD axis=) ─────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-squeeze-default",
            "squeeze",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_3d_f(
                        py,
                        // Shape (1, 3, 1) — should squeeze to (3,).
                        vec![vec![vec![1.0], vec![2.0], vec![3.0]]],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-squeeze-axis0-only",
            "squeeze",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_3d_f(
                            py,
                            vec![vec![vec![1.0], vec![2.0], vec![3.0]]],
                        )?
                        .into_any(),
                        0_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── concatenate (MUST + SHOULD axis=) ─────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-concatenate-1d",
            "concatenate",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let a = np_array_1d_f(py, vec![1.0, 2.0])?;
                let b = np_array_1d_f(py, vec![3.0, 4.0, 5.0])?;
                let lst = PyList::new(py, [a, b])?;
                PyTuple::new(py, [lst.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-concatenate-2d-axis1",
            "concatenate",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let a = np_array_2d_f(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
                let b = np_array_2d_f(py, vec![vec![10.0], vec![20.0]])?;
                let lst = PyList::new(py, [a, b])?;
                PyTuple::new(py, [lst.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── stack / vstack / hstack (MUST) ────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-stack-1d-default-axis0",
            "stack",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let a = np_array_1d_f(py, vec![1.0, 2.0, 3.0])?;
                let b = np_array_1d_f(py, vec![4.0, 5.0, 6.0])?;
                let lst = PyList::new(py, [a, b])?;
                PyTuple::new(py, [lst.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-vstack-1d",
            "vstack",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let a = np_array_1d_f(py, vec![1.0, 2.0, 3.0])?;
                let b = np_array_1d_f(py, vec![4.0, 5.0, 6.0])?;
                let lst = PyList::new(py, [a, b])?;
                PyTuple::new(py, [lst.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-hstack-1d",
            "hstack",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let a = np_array_1d_f(py, vec![1.0, 2.0])?;
                let b = np_array_1d_f(py, vec![3.0, 4.0, 5.0])?;
                let lst = PyList::new(py, [a, b])?;
                PyTuple::new(py, [lst.into_any()])
            },
            no_kwargs,
        );

        // ─── tile / repeat (MUST + SHOULD repeat axis=) ────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-tile-1d",
            "tile",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0])?.into_any(),
                        3_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-repeat-1d-scalar-reps",
            "repeat",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0])?.into_any(),
                        2_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-repeat-2d-axis1",
            "repeat",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_2d_f(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?
                            .into_any(),
                        3_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── flip / roll (MUST + SHOULD axis=) ─────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-flip-1d-default",
            "flip",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_f(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-flip-2d-axis0",
            "flip",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_2d_f(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?
                            .into_any(),
                        0_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-roll-1d-shift2",
            "roll",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![1.0, 2.0, 3.0, 4.0, 5.0])?.into_any(),
                        2_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── degenerate shapes (MAY) ───────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "manip-reshape-1x1",
            "reshape",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_f(py, vec![42.0])?.into_any(),
                        PyTuple::new(py, [1_i64, 1])?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "manip-concatenate-empty-list-second",
            "concatenate",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                // Two non-empty arrays with one having length-zero — numpy
                // accepts this and returns the concatenation of the
                // non-empty parts.
                let a = np_array_1d_f(py, vec![1.0, 2.0, 3.0])?;
                let b = np_array_1d_f(py, vec![])?;
                let lst = PyList::new(py, [a, b])?;
                PyTuple::new(py, [lst.into_any()])
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("array_manip");
    eprintln!("\n=== fnp-python conformance matrix: array_manip ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in array_manip family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
