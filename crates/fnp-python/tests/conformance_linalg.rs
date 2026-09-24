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

fn np_2d_complex<'py>(
    py: Python<'py>,
    rows: Vec<Vec<(f64, f64)>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let np = py.import("numpy")?;
    let nested: Vec<Vec<_>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|(r, i)| pyo3::types::PyComplex::from_doubles(py, *r, *i))
                .collect()
        })
        .collect();
    let arr = np.getattr("array")?.call1((nested,))?;
    arr.call_method1("astype", (np.getattr("complex128")?,))
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

        // ─── complex dtype tests (SHOULD) ──────────────────────────────
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-matrix_transpose-complex-2d",
            "matrix_transpose",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d_complex(
                        py,
                        vec![
                            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)],
                            vec![(4.0, -2.0), (5.0, 1.0), (6.0, -1.0)],
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
            "linalg-det-complex-2x2",
            "det",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d_complex(
                        py,
                        vec![vec![(1.0, 1.0), (2.0, -1.0)], vec![(3.0, 2.0), (4.0, -2.0)]],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-norm-complex-2d",
            "norm",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d_complex(
                        py,
                        vec![vec![(3.0, 0.0), (0.0, 4.0)], vec![(0.0, 0.0), (0.0, 0.0)]],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &linalg,
            &numpy_linalg,
            "linalg-inv-complex-2x2",
            "inv",
            RequirementLevel::Should,
            CompareMode::Close,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_2d_complex(
                        py,
                        vec![vec![(4.0, 0.0), (7.0, 1.0)], vec![(2.0, -1.0), (6.0, 0.0)]],
                    )?],
                )
            },
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

/// Stacks of 0x0 matrices (and other empty operands) must behave like numpy, never PANIC.
/// Two native batched paths divided or chunked by `n * n` without excluding n == 0:
/// `solve` on a (k, 0, 0) stack panicked with "chunk size must be non-zero" and
/// `eigh`/`eigvalsh` with "attempt to divide by zero", surfacing as `PanicException` where
/// numpy returns empty results (numpy's own TestSolve / TestEigh / TestEigvalsh
/// `test_generalized_empty_*` cases). A non-positive `tensorinv` `ind` is numpy's
/// ValueError, not an OverflowError. Outcome = exception type, or dtype and shape per output.
#[test]
fn empty_matrix_stacks_never_panic_and_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        // A Rust panic reaches Python as `PanicException` (a BaseException), so the sweep
        // catches BaseException and records the type like any other outcome.
        let code = std::ffi::CString::new(
            r#"
def outcome(fn, m):
    try:
        r = fn(m)
        rs = r if isinstance(r, tuple) else (r,)
        return ("ok", [(np.asarray(x).dtype.str, np.shape(x)) for x in rs])
    except BaseException as exc:
        return ("err", type(exc).__name__)
z = np.zeros
cases = {
    "solve (2,0,0)x(2,0)": lambda m: m.solve(z((2, 0, 0)), z((2, 0))),
    "solve (3,0,0)x(3,0,5)": lambda m: m.solve(z((3, 0, 0)), z((3, 0, 5))),
    "solve (0,0)x(0,)": lambda m: m.solve(z((0, 0)), z((0,))),
    "solve (0,2,2)x(0,2)": lambda m: m.solve(z((0, 2, 2)), z((0, 2))),
    "inv (2,0,0)": lambda m: m.inv(z((2, 0, 0))),
    "eigh (2,0,0)": lambda m: m.eigh(z((2, 0, 0))),
    "eigh (0,2,2)": lambda m: m.eigh(z((0, 2, 2))),
    "eigvalsh (2,0,0)": lambda m: m.eigvalsh(z((2, 0, 0))),
    "eigvalsh (0,3,3)": lambda m: m.eigvalsh(z((0, 3, 3))),
    "eig (2,0,0)": lambda m: m.eig(z((2, 0, 0))),
    "eigvals (2,0,0)": lambda m: m.eigvals(z((2, 0, 0))),
    "det (2,0,0)": lambda m: m.det(z((2, 0, 0))),
    "slogdet (2,0,0)": lambda m: m.slogdet(z((2, 0, 0))),
    "cholesky (2,0,0)": lambda m: m.cholesky(z((2, 0, 0))),
    "qr (2,0,0)": lambda m: m.qr(z((2, 0, 0))),
    "svd (2,0,0)": lambda m: m.svd(z((2, 0, 0))),
    "pinv (2,0,0)": lambda m: m.pinv(z((2, 0, 0))),
    "matrix_power (2,0,0)": lambda m: m.matrix_power(z((2, 0, 0)), 3),
    "matrix_rank (2,0,0)": lambda m: m.matrix_rank(z((2, 0, 0))),
    "tensorinv ind=-2": lambda m: m.tensorinv(np.eye(4).reshape(4, 2, 2), ind=-2),
    "tensorinv ind=0": lambda m: m.tensorinv(np.eye(4).reshape(4, 2, 2), ind=0),
    "tensorinv ind=1": lambda m: m.tensorinv(np.eye(4).reshape(4, 2, 2), ind=1),
}
result = [k for k, fn in cases.items() if outcome(fn, fnp.linalg) != outcome(fn, np.linalg)]
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let bad: Vec<String> = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert!(
            bad.is_empty(),
            "empty matrix stacks must match numpy and never panic: {bad:?}"
        );
        Ok(())
    });
}

/// fnp.linalg against numpy.linalg over 15 operand classes (float64/float32/complex/int/bool,
/// batched, singular, 1x1, 0x0, rectangular, SPD, symmetric batch, Fortran-ordered, NaN) and every
/// decomposition / solver / norm / cond / rank / power option (595 cases). README's linalg
/// contract is tolerance-based for values, so values must agree within 1e-9 of the largest
/// magnitude (signs ignored where a factor is only unique up to sign); the result type, dtype,
/// shape and exception type must match exactly. Before the fix (bead .8): pinv of a float32
/// matrix returned float64, because the native SVD works in float64 where numpy works in float32.
#[test]
fn linalg_results_match_numpy_types_exactly_and_values_within_tolerance() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("fnp", &module)?;
        globals.set_item("np", &numpy)?;
        let code = std::ffi::CString::new(
            r#"
rng = np.random.default_rng(31)
M = {
    "f8": rng.standard_normal((4, 4)), "f4": rng.standard_normal((5, 5)).astype(np.float32),
    "c16": rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)),
    "i8": rng.integers(-5, 5, (3, 3)), "batch": rng.standard_normal((6, 3, 3)),
    "singular": np.array([[1.0, 2.0], [2.0, 4.0]]), "1x1": np.array([[3.0]]), "0x0": np.zeros((0, 0)),
    "5x3": rng.standard_normal((5, 3)), "3x5": rng.standard_normal((3, 5)),
    "spd": (lambda a: a @ a.T + 4 * np.eye(4))(rng.standard_normal((4, 4))),
    "sym batch": (lambda a: a + np.swapaxes(a, -1, -2))(rng.standard_normal((5, 4, 4))),
    "F-order": np.asfortranarray(rng.standard_normal((4, 4))),
    "nan": np.array([[1.0, np.nan, 0], [0, 1, 0], [0, 0, 1]]), "bool": np.array([[True, False], [False, True]]),
}
cases = []
def add(name, fn, signs=False):
    cases.append((name, fn, signs))
for tag, a in M.items():
    for f in ("det", "slogdet", "inv", "pinv", "matrix_rank", "cond", "eigvals", "eigvalsh", "svdvals", "cholesky", "norm"):
        add(f"{f} {tag}", lambda m, f=f, a=a: getattr(m.linalg, f)(a))
    for f in ("qr", "svd", "eig", "eigh"):
        add(f"{f} {tag}", lambda m, f=f, a=a: getattr(m.linalg, f)(a), True)
    for p in (3, -1, 0):
        add(f"matrix_power {p} {tag}", lambda m, a=a, p=p: m.linalg.matrix_power(a, p))
    for o in (None, "fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf):
        add(f"norm ord={o} {tag}", lambda m, a=a, o=o: m.linalg.norm(a, ord=o))
    add(f"norm axis=-1 {tag}", lambda m, a=a: m.linalg.norm(a, axis=-1))
    add(f"norm keepdims {tag}", lambda m, a=a: m.linalg.norm(a, axis=(-2, -1), keepdims=True))
    add(f"cond 1 {tag}", lambda m, a=a: m.linalg.cond(a, 1))
    add(f"cond fro {tag}", lambda m, a=a: m.linalg.cond(a, "fro"))
    add(f"pinv hermitian {tag}", lambda m, a=a: m.linalg.pinv(a, hermitian=True))
    add(f"matrix_rank tol {tag}", lambda m, a=a: m.linalg.matrix_rank(a, tol=1e-3))
    add(f"qr r {tag}", lambda m, a=a: m.linalg.qr(a, mode="r"), True)
    add(f"qr complete {tag}", lambda m, a=a: m.linalg.qr(a, mode="complete"), True)
    add(f"svd no uv {tag}", lambda m, a=a: m.linalg.svd(a, compute_uv=False))
    add(f"svd reduced {tag}", lambda m, a=a: m.linalg.svd(a, full_matrices=False), True)
    add(f"eigh U {tag}", lambda m, a=a: m.linalg.eigh(a, UPLO="U"), True)
    add(f"cholesky upper {tag}", lambda m, a=a: m.linalg.cholesky(a, upper=True))
add("solve", lambda m: m.linalg.solve(M["f8"], np.arange(4.0)))
add("solve batch", lambda m: m.linalg.solve(M["batch"], np.ones((6, 3, 1))))
add("solve singular", lambda m: m.linalg.solve(M["singular"], [1.0, 2.0]))
add("lstsq", lambda m: m.linalg.lstsq(M["5x3"], np.arange(5.0), rcond=None))
add("lstsq rank-deficient", lambda m: m.linalg.lstsq(np.ones((4, 2)), np.arange(4.0), rcond=None))
add("tensorsolve", lambda m: m.linalg.tensorsolve(np.eye(6).reshape(2, 3, 6), np.arange(6.0).reshape(2, 3)))
add("multi_dot", lambda m: m.linalg.multi_dot([M["5x3"], M["3x5"], M["5x3"]]))
add("det 1-D raises", lambda m: m.linalg.det(np.arange(3.0)))
add("inv non-square raises", lambda m: m.linalg.inv(M["5x3"]))
add("cholesky not SPD raises", lambda m: m.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]])))
def cmp(r, s, signs):
    if hasattr(s, "_fields") or isinstance(s, (tuple, list)):
        return (type(r).__name__ == type(s).__name__ and len(r) == len(s)
                and all(cmp(x, y, signs) for x, y in zip(r, s)))
    if type(r) is not type(s):
        return False
    r2, s2 = np.asarray(r), np.asarray(s)
    if r2.dtype != s2.dtype or r2.shape != s2.shape:
        return False
    if r2.dtype.kind in "iub" or r2.size == 0:
        return np.array_equal(r2, s2)
    a, b = (np.abs(r2), np.abs(s2)) if signs else (r2, s2)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    with np.errstate(all="ignore"):
        finite = np.isfinite(b)
        if not np.array_equal(a[~finite & ~np.isnan(b)], b[~finite & ~np.isnan(b)]):
            return False
        if not finite.any():
            return True
        scale = float(np.max(np.abs(b[finite])))
        return float(np.max(np.abs(a[finite] - b[finite]))) <= 1e-9 * max(scale, 1e-300)
def outcome(fn, m):
    try:
        return fn(m), None
    except Exception as ex:
        return None, type(ex).__name__
bad = []
for name, fn, signs in cases:
    r, re = outcome(fn, fnp)
    s, se = outcome(fn, np)
    if re != se or (se is None and not cmp(r, s, signs)):
        bad.append(name)
result = (len(cases), bad)
"#,
        )
        .expect("script has no NUL");
        py.run(&code, Some(&globals), None)?;
        let (count, bad): (usize, Vec<String>) = globals
            .get_item("result")?
            .expect("script sets result")
            .extract()?;
        assert_eq!(count, 595, "case table drifted");
        assert!(bad.is_empty(), "linalg must match numpy: {bad:#?}");
        Ok(())
    });
}
