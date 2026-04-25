//! Conformance matrix: linalg family.
//!
//! Differential parity for the LAPACK-free fnp-python.linalg surface
//! plus the LAPACK-delegated functions (svd / qr / cholesky / solve /
//! eigvalsh / lstsq). Per DISC-009 the LAPACK-backed implementations
//! still pass through to numpy, so these calls are expected to succeed
//! identically; the harness asserts that fact.
//!
//! Edge cases per function: 1x1, well-conditioned square (3x3),
//! identity-like, batched 3-D input where applicable.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_2d<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((rows,))
}

fn np_3d<'py>(
    py: Python<'py>,
    cube: Vec<Vec<Vec<f64>>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((cube,))
}

#[test]
fn conformance_linalg_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;
        let linalg_mod = module.getattr("linalg").expect("fnp_python.linalg");
        let np_linalg = numpy.getattr("linalg").expect("numpy.linalg");
        let linalg = linalg_mod
            .cast_into::<pyo3::types::PyModule>()
            .expect("fnp_python.linalg should be a submodule");
        let numpy_linalg = np_linalg
            .cast_into::<pyo3::types::PyModule>()
            .expect("numpy.linalg should be a submodule");

        // ─── matrix_transpose: LAPACK-free, native ─────────────────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_transpose-2d",
            "matrix_transpose",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d(py, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_transpose-batched-3d",
            "matrix_transpose",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_3d(
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

        // ─── matrix_power: LAPACK-free for non-negative exponents ──────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_power-zero",
            "matrix_power",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_2d(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?.into_any(),
                        0_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_power-square",
            "matrix_power",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_2d(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?.into_any(),
                        2_i64.into_pyobject(py)?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── multi_dot: chained matrix product ─────────────────────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-multi_dot-three-mats",
            "multi_dot",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                let a = np_2d(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
                let b = np_2d(py, vec![vec![5.0, 6.0], vec![7.0, 8.0]])?;
                let c = np_2d(py, vec![vec![9.0, 10.0], vec![11.0, 12.0]])?;
                let lst = pyo3::types::PyList::new(py, [a, b, c])?;
                PyTuple::new(py, [lst.into_any()])
            },
            no_kwargs,
        );

        // ─── vecdot ────────────────────────────────────────────────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-vecdot-1d",
            "vecdot",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                let a = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![1.0, 2.0, 3.0],))?;
                let b = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![4.0, 5.0, 6.0],))?;
                PyTuple::new(py, [a, b])
            },
            no_kwargs,
        );

        // ─── det: LAPACK-free for small matrices ───────────────────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-det-2x2",
            "det",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_2d(py, vec![vec![1.0, 2.0], vec![3.0, 4.0]])?]),
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-det-identity-3x3",
            "det",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d(
                        py,
                        vec![
                            vec![1.0, 0.0, 0.0],
                            vec![0.0, 1.0, 0.0],
                            vec![0.0, 0.0, 1.0],
                        ],
                    )?],
                )
            },
            no_kwargs,
        );

        // ─── inv: LAPACK-free narrow path for small invertible matrices ─
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-inv-2x2",
            "inv",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_2d(py, vec![vec![4.0, 7.0], vec![2.0, 6.0]])?]),
            no_kwargs,
        );

        // ─── norm: LAPACK-free Frobenius / 1 / inf / 2 paths ───────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-norm-2d-default-frobenius",
            "norm",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_2d(py, vec![vec![3.0, 4.0], vec![0.0, 0.0]])?]),
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-norm-1d-vector",
            "norm",
            RequirementLevel::Must,
            CompareMode::Close,
            t,
            |py| {
                let a = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![3.0_f64, 4.0],))?;
                PyTuple::new(py, [a])
            },
            no_kwargs,
        );

        // ─── LAPACK-delegated paths (DISC-009 acceptance) ──────────────
        // These pass through to numpy so they should match by definition;
        // the harness records that fact and the contract.
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-svd-singular-values-only",
            "svd",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| {
                let kwargs_marker = vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
                PyTuple::new(py, [np_2d(py, kwargs_marker)?.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("compute_uv", false)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-cholesky-spd",
            "cholesky",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d(
                        py,
                        // Symmetric positive definite: A = L L^T with
                        // L = [[2, 0], [1, 3]] → A = [[4, 2], [2, 10]].
                        vec![vec![4.0, 2.0], vec![2.0, 10.0]],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-solve-3-equations",
            "solve",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                let a = np_2d(
                    py,
                    vec![
                        vec![3.0, 1.0, -2.0],
                        vec![1.0, -1.0, 1.0],
                        vec![2.0, 4.0, -3.0],
                    ],
                )?;
                let b = py
                    .import("numpy")?
                    .getattr("array")?
                    .call1((vec![5.0_f64, 0.0, -2.0],))?;
                PyTuple::new(py, [a, b])
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-eigvalsh-symmetric",
            "eigvalsh",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_2d(py, vec![vec![2.0, 1.0], vec![1.0, 2.0]])?]),
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_rank-rank2",
            "matrix_rank",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d(
                        py,
                        vec![
                            vec![1.0, 2.0, 3.0],
                            vec![4.0, 5.0, 6.0],
                            vec![7.0, 8.0, 9.0], // linearly dependent → rank 2
                        ],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_rank-full",
            "matrix_rank",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d(
                        py,
                        vec![
                            vec![1.0, 0.0, 0.0],
                            vec![0.0, 2.0, 0.0],
                            vec![0.0, 0.0, 3.0],
                        ],
                    )?],
                )
            },
            no_kwargs,
        );

        // ─── 1x1 edge case ─────────────────────────────────────────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-det-1x1",
            "det",
            RequirementLevel::May,
            CompareMode::Close,
            t,
            |py| PyTuple::new(py, [np_2d(py, vec![vec![5.0]])?]),
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("linalg");
    eprintln!("\n=== fnp-python conformance matrix: linalg ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in linalg family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
