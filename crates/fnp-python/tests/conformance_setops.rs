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
fn intersect_setdiff_packed_latin1_u8_s8_strings_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(456)\n\
                 u_a = rng.integers(97, 123, (90_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 u_fresh = rng.integers(97, 123, (70_000, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 u_b = np.concatenate([u_a[:30_000], u_fresh])\n\
                 s_a = rng.integers(97, 123, (90_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
                 s_fresh = rng.integers(97, 123, (70_000, 8), dtype=np.uint8).view('S8').reshape(-1)\n\
                 s_b = np.concatenate([s_a[:30_000], s_fresh])\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        for (left_name, right_name) in [("u_a", "u_b"), ("s_a", "s_b")] {
            let left = ns
                .get_item(left_name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing left"))?;
            let right = ns
                .get_item(right_name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing right"))?;
            for op in ["intersect1d", "setdiff1d"] {
                let ours = module.getattr(op)?.call1((&left, &right))?;
                let theirs = numpy.getattr(op)?.call1((&left, &right))?;
                let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
                assert!(equal, "packed Latin-1 string {op} diverged for {left_name}");
                let ours_dtype = ours.getattr("dtype")?.str()?;
                let theirs_dtype = theirs.getattr("dtype")?.str()?;
                let same_dtype = ours_dtype == theirs_dtype.to_str()?;
                assert!(same_dtype, "packed Latin-1 string {op} changed dtype");
            }
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
                 edge_b = np.array([(1, 0.0), (4, 4.0)], dtype=dt)\n\
                 dt32 = [('id','<i4'),('val','<f4')]\n\
                 a32 = np.zeros(100_000, dtype=dt32)\n\
                 a32['id'] = rng.integers(-200, 700, 100_000, dtype=np.int32)\n\
                 a32['val'] = rng.integers(-150, 850, 100_000).astype(np.float32)\n\
                 b32 = np.zeros(100_000, dtype=dt32)\n\
                 b32['id'][:30_000] = a32['id'][:30_000]\n\
                 b32['val'][:30_000] = a32['val'][:30_000]\n\
                 b32['id'][30_000:] = rng.integers(-200, 700, 70_000, dtype=np.int32)\n\
                 b32['val'][30_000:] = rng.integers(-150, 850, 70_000).astype(np.float32)\n\
                 edge32_a = np.array([(1, -0.0), (2, 1.5), (3, 3.0)], dtype=dt32)\n\
                 edge32_b = np.array([(1, 0.0), (4, 4.0)], dtype=dt32)\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        for op in ["setxor1d", "intersect1d", "setdiff1d"] {
            let ours_fn = module.getattr(op)?;
            let numpy_fn = numpy.getattr(op)?;
            for (left_name, right_name) in [
                ("a", "b"),
                ("a32", "b32"),
                ("edge_a", "edge_b"),
                ("edge32_a", "edge32_b"),
            ] {
                let left = ns.get_item(left_name)?.expect("left");
                let right = ns.get_item(right_name)?.expect("right");
                let ours = ours_fn.call1((&left, &right))?;
                let theirs = numpy_fn.call1((&left, &right))?;
                let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
                assert!(
                    equal,
                    "mixed structured {op} diverged from numpy for {left_name}/{right_name}"
                );
            }
        }
        Ok(())
    });
}

#[test]
fn mixed_struct_dense_bitplanes_all_supported_field_widths_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 cases = {}\n\
                 for case_id, int_code in enumerate(('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8')):\n\
                 \x20\x20\x20\x20low, high = ((0, 41) if int_code[0] == 'u' else (-20, 21))\n\
                 \x20\x20\x20\x20for float_code in ('f4', 'f8'):\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20rng = np.random.default_rng(10_000 + 10 * case_id + int(float_code[1]))\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20dt = np.dtype([('id', int_code), ('val', float_code)], align=False)\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20a = np.zeros(40_000, dtype=dt)\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20a['id'] = rng.integers(low, high, 40_000, dtype=np.dtype(int_code))\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20a['val'] = rng.integers(-15, 26, 40_000).astype(np.dtype(float_code))\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20b = np.zeros(40_000, dtype=dt)\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20b[:10_000] = a[:10_000]\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20b['id'][10_000:] = rng.integers(low, high, 30_000, dtype=np.dtype(int_code))\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20b['val'][10_000:] = rng.integers(-15, 26, 30_000).astype(np.dtype(float_code))\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20cases[f'{int_code}_{float_code}'] = (a, b)\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let cases = ns.get_item("cases")?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err("structured cases were not generated")
        })?;
        let array_equal = numpy.getattr("array_equal")?;
        for int_code in ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"] {
            for float_code in ["f4", "f8"] {
                let label = format!("{int_code}_{float_code}");
                let pair = cases.get_item(&label)?;
                let left = pair.get_item(0)?;
                let right = pair.get_item(1)?;
                for op in ["intersect1d", "setdiff1d", "setxor1d"] {
                    let ours = module.getattr(op)?.call1((&left, &right))?;
                    let theirs = numpy.getattr(op)?.call1((&left, &right))?;
                    assert!(
                        array_equal.call1((&ours, &theirs))?.extract::<bool>()?,
                        "mixed structured {op} diverged for {label}"
                    );
                    assert!(
                        ours.getattr("dtype")?.eq(theirs.getattr("dtype")?)?,
                        "mixed structured {op} changed dtype for {label}"
                    );
                    let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
                    let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
                    assert_eq!(
                        ours_bytes, theirs_bytes,
                        "mixed structured {op} changed output bytes for {label}"
                    );
                }
            }
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

#[test]
fn intersect_setdiff_complex128_dense_integral_grid_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        // `a` lives on a [0,600)x[0,600) integral grid (the direct-domain presence-grid
        // route). `b` mixes values inside that grid (genuine intersect/setdiff overlap)
        // with a block shifted into [1000,1600) that lies OUTSIDE a's grid, so both ops
        // exercise the b-out-of-range cells (present in b, never emitted).
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 x = np.arange(300_000, dtype=np.int64)\n\
                 y = np.arange(300_000, dtype=np.int64)\n\
                 a = (((x * 17) % 600) + 1j * ((x * 31) % 600)).astype(np.complex128)\n\
                 b_lo = ((((y * 29) + 7) % 600) + 1j * (((y * 43) + 11) % 600)).astype(np.complex128)\n\
                 b_hi = ((1000 + (y % 600)) + 1j * (1000 + ((y * 7) % 600))).astype(np.complex128)\n\
                 b = np.concatenate([b_lo[:200_000], b_hi[:100_000]])\n"
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
            assert!(equal, "dense integral complex128 {op} diverged from numpy");
            let dtype = ours.getattr("dtype")?.str()?.to_string();
            assert_eq!(dtype, "complex128");
        }
        Ok(())
    });
}

#[test]
fn unique_and_sort_string_packed_latin1_large_matches_numpy() {
    // Large-n fixed-width Latin-1 U8/S6 arrays (n >= 1<<17) take the packed-u64 (key, index)
    // sort path in both unique and sort. Packed key order == codepoint order; unique first-of-run
    // and sorted record sequence must be byte-exact vs numpy.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(19)\n\
                 n = 300_000\n\
                 u8 = rng.integers(97, 123, (n, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 s6 = np.array([bytes(r) for r in rng.integers(97, 105, (60_000, 6), dtype=np.uint8)], dtype='S6')\n\
                 s6 = np.tile(s6, 5)[:n]\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        for name in ["u8", "s6"] {
            let arr = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing arr"))?;
            for op in ["unique", "sort"] {
                let ours = module.getattr(op)?.call1((&arr,))?;
                let theirs = numpy.getattr(op)?.call1((&arr,))?;
                let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
                assert!(equal, "packed Latin-1 {name} {op} diverged from numpy");
            }
        }
        Ok(())
    });
}

#[test]
fn unique_packed_wide_latin1_u9_u16_matches_numpy() {
    // U16 records no longer fit the packed-u64 path. The two-word key captures all 16
    // Latin-1 codepoints in NumPy lexicographic order while retaining the original index
    // as the deterministic tie-break used by the unchanged gather/dedup pipeline.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(281)\n\
                 n = 300_000\n\
                 u9 = rng.integers(97, 123, (n, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 u16 = rng.integers(97, 123, (n, 16), dtype=np.uint32).reshape(-1).view('U16')\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        for name in ["u9", "u16"] {
            let arr = ns.get_item(name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {name}"))
            })?;
            let ours = module.getattr("unique")?.call1((&arr,))?;
            let theirs = numpy.getattr("unique")?.call1((&arr,))?;
            let equal: bool = numpy
                .getattr("array_equal")?
                .call1((&ours, &theirs))?
                .extract()?;
            assert!(
                equal,
                "packed-wide Latin-1 {name} unique diverged from numpy"
            );
            assert_eq!(
                ours.getattr("dtype")?.str()?.to_string(),
                theirs.getattr("dtype")?.str()?.to_string()
            );
            let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
            let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
            assert_eq!(
                ours_bytes, theirs_bytes,
                "{name} unique output bytes diverged"
            );
        }
        Ok(())
    });
}

#[test]
fn intersect_setdiff_packed_wide_latin1_u9_u16_matches_numpy() {
    // U9..U16 no longer fit the packed-u64 set-algebra path; the two-word key branch must
    // reproduce numpy's sorted set semantics (values in both / values only in a) byte-exactly.
    // b shares half its values with a so intersect and setdiff are both non-trivial. The u12w
    // pair plants one wide (>0xFF) codepoint, which must defer to the numpy-identical fallback.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(282)\n\
                 n = 200_000\n\
                 a9 = rng.integers(97, 123, (n, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 f9 = rng.integers(97, 123, (n // 2, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 b9 = np.concatenate([a9[: n // 2], f9])\n\
                 a16 = rng.integers(97, 123, (n, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 f16 = rng.integers(97, 123, (n // 2, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 b16 = np.concatenate([a16[: n // 2], f16])\n\
                 a12 = rng.integers(97, 123, (n, 12), dtype=np.uint32).reshape(-1).view('U12')\n\
                 f12 = rng.integers(97, 123, (n // 2, 12), dtype=np.uint32).reshape(-1).view('U12')\n\
                 b12w = np.concatenate([a12[: n // 2], f12])\n\
                 raw = a12.view(np.uint32).copy()\n\
                 raw[5] = 0x0101\n\
                 a12w = raw.view('U12')\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        for (a_name, b_name) in [("a9", "b9"), ("a16", "b16"), ("a12w", "b12w")] {
            let a = ns.get_item(a_name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {a_name}"))
            })?;
            let b = ns.get_item(b_name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {b_name}"))
            })?;
            for op in ["intersect1d", "setdiff1d"] {
                let ours = module.getattr(op)?.call1((&a, &b))?;
                let theirs = numpy.getattr(op)?.call1((&a, &b))?;
                let equal: bool = numpy
                    .getattr("array_equal")?
                    .call1((&ours, &theirs))?
                    .extract()?;
                assert!(
                    equal,
                    "packed-wide Latin-1 {a_name} {op} diverged from numpy"
                );
                assert_eq!(
                    ours.getattr("dtype")?.str()?.to_string(),
                    theirs.getattr("dtype")?.str()?.to_string()
                );
                let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
                let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
                assert_eq!(
                    ours_bytes, theirs_bytes,
                    "{a_name} {op} output bytes diverged"
                );
            }
        }
        Ok(())
    });
}

#[test]
fn setxor_union_packed_wide_latin1_u9_u16_matches_numpy() {
    // Completes the U9..U16 two-word-key set algebra: setxor1d (source-tagged run-composition
    // over wide keys) and union1d (dedicated pack of each operand instead of concat+unique).
    // b shares half its values with a so all runs (pure-a, pure-b, both) occur. The a12w pair
    // plants one wide (>0xFF) codepoint, which must defer to the numpy-identical fallback.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(283)\n\
                 n = 200_000\n\
                 a9 = rng.integers(97, 123, (n, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 f9 = rng.integers(97, 123, (n // 2, 9), dtype=np.uint32).reshape(-1).view('U9')\n\
                 b9 = np.concatenate([a9[: n // 2], f9])\n\
                 a16 = rng.integers(97, 123, (n, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 f16 = rng.integers(97, 123, (n // 2, 16), dtype=np.uint32).reshape(-1).view('U16')\n\
                 b16 = np.concatenate([a16[: n // 2], f16])\n\
                 a12 = rng.integers(97, 123, (n, 12), dtype=np.uint32).reshape(-1).view('U12')\n\
                 f12 = rng.integers(97, 123, (n // 2, 12), dtype=np.uint32).reshape(-1).view('U12')\n\
                 b12w = np.concatenate([a12[: n // 2], f12])\n\
                 raw = a12.view(np.uint32).copy()\n\
                 raw[7] = 0x0142\n\
                 a12w = raw.view('U12')\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        for (a_name, b_name) in [("a9", "b9"), ("a16", "b16"), ("a12w", "b12w")] {
            let a = ns.get_item(a_name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {a_name}"))
            })?;
            let b = ns.get_item(b_name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {b_name}"))
            })?;
            for op in ["setxor1d", "union1d"] {
                let ours = module.getattr(op)?.call1((&a, &b))?;
                let theirs = numpy.getattr(op)?.call1((&a, &b))?;
                let equal: bool = numpy
                    .getattr("array_equal")?
                    .call1((&ours, &theirs))?
                    .extract()?;
                assert!(
                    equal,
                    "packed-wide Latin-1 {a_name} {op} diverged from numpy"
                );
                assert_eq!(
                    ours.getattr("dtype")?.str()?.to_string(),
                    theirs.getattr("dtype")?.str()?.to_string()
                );
                let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
                let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
                assert_eq!(
                    ours_bytes, theirs_bytes,
                    "{a_name} {op} output bytes diverged"
                );
            }
        }
        Ok(())
    });
}

#[test]
fn wide_bytes_s9_s16_setops_match_numpy() {
    // S9..S16 records take the same two-word key route as U9..U16 but pack raw bytes (stride 1,
    // no Latin-1 gate — 'S' byte order is numpy order for ALL byte values). Data deliberately
    // includes high bytes (0x80..=0xFF) and embedded nulls, which the 'U' route must defer on
    // but the 'S' route must handle natively. All five wide-key ops in one sweep.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(284)\n\
                 n = 200_000\n\
                 a9 = rng.integers(0, 256, (n, 9), dtype=np.uint8).view('S9').reshape(-1)\n\
                 f9 = rng.integers(0, 256, (n // 2, 9), dtype=np.uint8).view('S9').reshape(-1)\n\
                 b9 = np.concatenate([a9[: n // 2], f9])\n\
                 a16 = rng.integers(0, 256, (n, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 f16 = rng.integers(0, 256, (n // 2, 16), dtype=np.uint8).view('S16').reshape(-1)\n\
                 b16 = np.concatenate([a16[: n // 2], f16])\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        for (a_name, b_name) in [("a9", "b9"), ("a16", "b16")] {
            let a = ns.get_item(a_name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {a_name}"))
            })?;
            let b = ns.get_item(b_name)?.ok_or_else(|| {
                pyo3::exceptions::PyAssertionError::new_err(format!("missing {b_name}"))
            })?;
            let ours_u = module.getattr("unique")?.call1((&a,))?;
            let theirs_u = numpy.getattr("unique")?.call1((&a,))?;
            let equal: bool = numpy
                .getattr("array_equal")?
                .call1((&ours_u, &theirs_u))?
                .extract()?;
            assert!(equal, "wide-bytes {a_name} unique diverged from numpy");
            for op in ["intersect1d", "setdiff1d", "setxor1d", "union1d"] {
                let ours = module.getattr(op)?.call1((&a, &b))?;
                let theirs = numpy.getattr(op)?.call1((&a, &b))?;
                let equal: bool = numpy
                    .getattr("array_equal")?
                    .call1((&ours, &theirs))?
                    .extract()?;
                assert!(equal, "wide-bytes {a_name} {op} diverged from numpy");
                assert_eq!(
                    ours.getattr("dtype")?.str()?.to_string(),
                    theirs.getattr("dtype")?.str()?.to_string()
                );
                let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
                let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
                assert_eq!(
                    ours_bytes, theirs_bytes,
                    "{a_name} {op} output bytes diverged"
                );
            }
        }
        Ok(())
    });
}

#[test]
fn unique_full_string_packed_latin1_large_matches_numpy() {
    // unique(..., return_index/inverse/counts) on large Latin-1 U8/S6 takes the packed-u64
    // (key, index) path; first-occurrence index, inverse map, and counts must all be byte-exact.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(23)\n\
                 n = 300_000\n\
                 u8 = rng.integers(97, 101, (n, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 s6 = np.array([bytes(r) for r in rng.integers(97, 101, (40_000, 6), dtype=np.uint8)], dtype='S6')\n\
                 s6 = np.tile(s6, 8)[:n]\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        let kw = PyDict::new(py);
        kw.set_item("return_index", true)?;
        kw.set_item("return_inverse", true)?;
        kw.set_item("return_counts", true)?;
        for name in ["u8", "s6"] {
            let arr = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing arr"))?;
            let ours = module.getattr("unique")?.call((&arr,), Some(&kw))?;
            let theirs = numpy.getattr("unique")?.call((&arr,), Some(&kw))?;
            for k in 0..4 {
                let o = ours.get_item(k)?;
                let t = theirs.get_item(k)?;
                let equal: bool = array_equal.call1((&o, &t))?.extract()?;
                assert!(equal, "packed Latin-1 {name} unique_full field {k} diverged from numpy");
            }
        }
        Ok(())
    });
}

#[test]
fn setxor1d_complex128_dense_integral_grid_matches_numpy() {
    // c128 setxor1d over a dense integer grid. `b` spans values both inside and OUTSIDE a's range,
    // since setxor keeps b-only cells too (union-range grid). Byte-exact vs numpy for finite integral.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 x = np.arange(300_000, dtype=np.int64)\n\
                 y = np.arange(300_000, dtype=np.int64)\n\
                 a = (((x * 17) % 600) + 1j * ((x * 31) % 600)).astype(np.complex128)\n\
                 b_lo = ((((y * 29) + 7) % 600) + 1j * (((y * 43) + 11) % 600)).astype(np.complex128)\n\
                 b_hi = ((700 + (y % 400)) + 1j * (700 + ((y * 7) % 400))).astype(np.complex128)\n\
                 b = np.concatenate([b_lo[:200_000], b_hi[:100_000]])\n"
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
        let ours = module.getattr("setxor1d")?.call1((&a, &b))?;
        let theirs = numpy.getattr("setxor1d")?.call1((&a, &b))?;
        let equal: bool = numpy.getattr("array_equal")?.call1((&ours, &theirs))?.extract()?;
        assert!(equal, "dense integral complex128 setxor1d diverged from numpy");
        let dtype = ours.getattr("dtype")?.str()?.to_string();
        assert_eq!(dtype, "complex128");
        Ok(())
    });
}
