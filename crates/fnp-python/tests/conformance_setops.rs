//! Conformance matrix: setops family.
//!
//! Differential parity for fnp_python's set-operation surface:
//!
//!   intersect1d, union1d, setdiff1d, setxor1d, isin,
//!   unique, ediff1d
//!
//! All are native bodies that delegate to numpy on edge cases (string
//! arrays, structured dtypes). The harness exercises pass-paths under
//! Strict comparison and SHOULD-tier kwargs (`assume_unique`,
//! `return_indices`, `return_inverse`) that are most prone to silently
//! drop if a wrapper drifts from numpy's signature.

mod common;

use common::{CompareMode, RequirementLevel, Totals, run_case, with_fnp_and_numpy};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn no_kwargs<'py>(_py: Python<'py>) -> PyResult<Option<pyo3::Bound<'py, PyDict>>> {
    Ok(None)
}

fn np_array_1d_i<'py>(
    py: Python<'py>,
    values: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_f<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_repeated_f64_grid<'py>(
    py: Python<'py>,
    n: i64,
    mul: i64,
    modulus: i64,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let raw = py.import("numpy")?.call_method1("arange", (n,))?;
    raw.call_method1("__mul__", (mul,))?
        .call_method1("__mod__", (modulus,))?
        .call_method1("__truediv__", (16.0_f64,))?
        .call_method1("astype", ("float64",))
}

fn np_array_1d_complex<'py>(
    py: Python<'py>,
    values: Vec<(f64, f64)>,
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
fn union1d_complex128_dense_integral_grid_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 x = np.arange(300_000, dtype=np.int64)\n\
                 y = np.arange(300_000, dtype=np.int64)\n\
                 a = (((x * 17) % 600) + 1j * ((x * 31) % 600)).astype(np.complex128)\n\
                 b = ((((y * 29) + 7) % 600) + 1j * (((y * 43) + 11) % 600)).astype(np.complex128)\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let a = ns
            .get_item("a")?
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing a"))?;
        let b = ns
            .get_item("b")?
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing b"))?;
        let ours = module.getattr("union1d")?.call1((&a, &b))?;
        let theirs = numpy.getattr("union1d")?.call1((&a, &b))?;
        let equal: bool = numpy
            .getattr("array_equal")?
            .call1((&ours, &theirs))?
            .extract()?;
        assert!(
            equal,
            "dense integral complex128 union1d diverged from numpy"
        );
        let dtype = ours.getattr("dtype")?.str()?.to_string();
        assert_eq!(dtype, "complex128");
        Ok(())
    });
}

#[test]
fn intersect_setdiff_complex128_sorted_merge_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 x = np.arange(330_000, dtype=np.int64)\n\
                 y = np.arange(330_000, dtype=np.int64)\n\
                 a = (((x * 1_000_003) % 2_000_003) + 1j * ((x * 1_000_033) % 2_000_033)).astype(np.complex128)\n\
                 shared = a[::3]\n\
                 fresh = ((((y * 1_000_087) + 17) % 2_000_089) + 1j * (((y * 1_000_093) + 31) % 2_000_099)).astype(np.complex128)\n\
                 b = np.concatenate([shared, fresh[:220_000]])\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let a = ns
            .get_item("a")?
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing a"))?;
        let b = ns
            .get_item("b")?
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing b"))?;
        let array_equal = numpy.getattr("array_equal")?;
        for op in ["intersect1d", "setdiff1d"] {
            let ours = module.getattr(op)?.call1((&a, &b))?;
            let theirs = numpy.getattr(op)?.call1((&a, &b))?;
            let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
            assert!(
                equal,
                "complex128 sorted-merge {op} diverged from numpy"
            );
            let dtype = ours.getattr("dtype")?.str()?.to_string();
            assert_eq!(dtype, "complex128");
        }
        Ok(())
    });
}

#[test]
fn union_and_setxor_u8_packed_latin1_strings_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(123)\n\
                 a = rng.integers(97, 123, (90_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 fresh = rng.integers(97, 123, (70_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 b = np.concatenate([a[:30_000], fresh])\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let a = ns
            .get_item("a")?
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing a"))?;
        let b = ns
            .get_item("b")?
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing b"))?;
        let array_equal = numpy.getattr("array_equal")?;
        for op in ["union1d", "setxor1d"] {
            let ours = module.getattr(op)?.call1((&a, &b))?;
            let theirs = numpy.getattr(op)?.call1((&a, &b))?;
            let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
            assert!(equal, "packed Latin-1 U8 {op} diverged from numpy");
        }
        Ok(())
    });
}

#[test]
fn setxor1d_mixed_struct_dense_integral_float_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(321)\n\
                 dt = [('id','<i8'),('val','<f8')]\n\
                 a = np.zeros(100_000, dtype=dt)\n\
                 a['id'] = rng.integers(-200, 700, 100_000)\n\
                 a['val'] = rng.integers(-150, 850, 100_000).astype(np.float64)\n\
                 b = np.zeros(100_000, dtype=dt)\n\
                 b['id'][:30_000] = a['id'][:30_000]\n\
                 b['val'][:30_000] = a['val'][:30_000]\n\
                 b['id'][30_000:] = rng.integers(-200, 700, 70_000)\n\
                 b['val'][30_000:] = rng.integers(-150, 850, 70_000).astype(np.float64)\n\
                 edge_a = np.array([(1, -0.0), (2, 1.5), (3, 3.0)], dtype=dt)\n\
                 edge_b = np.array([(1, 0.0), (4, 4.0)], dtype=dt)\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let setxor = module.getattr("setxor1d")?;
        let numpy_setxor = numpy.getattr("setxor1d")?;
        let array_equal = numpy.getattr("array_equal")?;
        for (left_name, right_name) in [("a", "b"), ("edge_a", "edge_b")] {
            let left = ns.get_item(left_name)?.expect("left");
            let right = ns.get_item(right_name)?.expect("right");
            let ours = setxor.call1((&left, &right))?;
            let theirs = numpy_setxor.call1((&left, &right))?;
            let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
            assert!(
                equal,
                "mixed structured setxor1d diverged from numpy for {left_name}/{right_name}"
            );
        }
        Ok(())
    });
}

#[test]
fn conformance_setops_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── intersect1d (MUST + SHOULD assume_unique / return_indices) ─
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-overlap",
            "intersect1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6, 7])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-disjoint",
            "intersect1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3])?,
                        np_array_1d_i(py, vec![10, 20, 30])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-assume_unique",
            "intersect1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6, 7])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("assume_unique", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-f64-large-repeated",
            "intersect1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_repeated_f64_grid(py, 1_100_000, 1, 65_536)?,
                        np_repeated_f64_grid(py, 1_100_000, 7, 65_536)?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-return_indices",
            "intersect1d",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6, 7])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("return_indices", true)?;
                Ok(Some(kw))
            },
        );

        // ─── union1d (MUST) ────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-union1d-overlap",
            "union1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-union1d-empty-rhs",
            "union1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3])?,
                        np_array_1d_i(py, vec![])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── setdiff1d (MUST + SHOULD assume_unique) ───────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-setdiff1d-overlap",
            "setdiff1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-setdiff1d-assume_unique",
            "setdiff1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![3, 4])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("assume_unique", true)?;
                Ok(Some(kw))
            },
        );

        // ─── setxor1d (MUST) ───────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-setxor1d-overlap",
            "setxor1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                        np_array_1d_i(py, vec![3, 4, 5, 6])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── isin (MUST + SHOULD invert / assume_unique) ───────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-1d",
            "isin",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![2, 4, 6])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-invert",
            "isin",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![2, 4, 6])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("invert", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-assume_unique",
            "isin",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 3, 4, 5])?,
                        np_array_1d_i(py, vec![2, 4, 6])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("assume_unique", true)?;
                Ok(Some(kw))
            },
        );

        // ─── unique (MUST + SHOULD return_inverse / return_counts) ─────
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-1d-with-dupes",
            "unique",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![3, 1, 2, 1, 3, 2, 4])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-return_inverse",
            "unique",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![3, 1, 2, 1, 3, 2, 4])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("return_inverse", true)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-return_counts",
            "unique",
            RequirementLevel::Should,
            CompareMode::Surface,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![3, 1, 2, 1, 3, 2, 4])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("return_counts", true)?;
                Ok(Some(kw))
            },
        );

        // ─── ediff1d (MUST + SHOULD to_begin / to_end) ─────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-ediff1d-1d",
            "ediff1d",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_f(py, vec![1.0, 3.0, 6.0, 10.0])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-ediff1d-to_begin-to_end",
            "ediff1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_f(py, vec![1.0, 3.0, 6.0, 10.0])?]),
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("to_begin", np_array_1d_f(py, vec![-99.0])?)?;
                kw.set_item("to_end", np_array_1d_f(py, vec![99.0])?)?;
                Ok(Some(kw))
            },
        );

        // ─── empty-input edge cases (MAY) ──────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-both-empty",
            "intersect1d",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![])?, np_array_1d_i(py, vec![])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-empty",
            "unique",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![])?]),
            no_kwargs,
        );

        // ─── complex dtype tests (SHOULD) ──────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "setops-intersect1d-complex",
            "intersect1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_complex(py, vec![(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)])?,
                        np_array_1d_complex(py, vec![(2.0, -1.0), (4.0, 4.0), (3.0, 2.0)])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-union1d-complex",
            "union1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_complex(py, vec![(1.0, 1.0), (2.0, -1.0)])?,
                        np_array_1d_complex(py, vec![(2.0, -1.0), (3.0, 2.0)])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-setdiff1d-complex",
            "setdiff1d",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_complex(py, vec![(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)])?,
                        np_array_1d_complex(py, vec![(2.0, -1.0)])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-unique-complex",
            "unique",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [np_array_1d_complex(
                        py,
                        vec![(1.0, 1.0), (2.0, -1.0), (1.0, 1.0), (3.0, 2.0)],
                    )?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "setops-isin-complex",
            "isin",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_complex(py, vec![(1.0, 1.0), (2.0, -1.0), (3.0, 2.0)])?,
                        np_array_1d_complex(py, vec![(2.0, -1.0), (4.0, 4.0)])?,
                    ],
                )
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("setops");
    eprintln!("\n=== fnp-python conformance matrix: setops ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in setops family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
