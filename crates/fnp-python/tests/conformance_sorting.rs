//! Conformance matrix: sorting family.
//!
//! Differential parity for fnp_python's sorting surface:
//!
//!   sort, argsort, lexsort, partition, argpartition, sort_complex
//!
//! Finding: These 6 sorting functions are exposed but had ZERO conformance
//! tests, despite being core numpy operations.

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
    values: Vec<Vec<f64>>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_1d_int<'py>(
    py: Python<'py>,
    values: Vec<i64>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

fn np_array_complex<'py>(py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    let numpy = py.import("numpy")?;
    let globals = PyDict::new(py);
    globals.set_item("numpy", numpy)?;
    py.eval(
        pyo3::ffi::c_str!("numpy.array([3+4j, 1+2j, 2+1j, 1+1j])"),
        Some(&globals),
        None,
    )
}

#[test]
fn sort_argsort_explicit_kind_large_unique_last_axis_matches_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("np", numpy)?;
        globals.set_item("fnp", module)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
rows = 1025
cols = 1025
n = rows * cols
base = ((np.arange(n, dtype=np.int64) * 48271) % 2147483647).astype(np.float64)
a = base.reshape(rows, cols)

for kind in ("stable", "mergesort", "heapsort"):
    fs = fnp.sort(a, kind=kind)
    ns = np.sort(a, kind=kind)
    assert fs.dtype == ns.dtype
    assert fs.shape == ns.shape
    assert np.array_equal(fs, ns), kind

    fa = fnp.argsort(a, kind=kind)
    na = np.argsort(a, kind=kind)
    assert fa.dtype == na.dtype
    assert fa.shape == na.shape
    assert np.array_equal(fa, na), kind
"#
            ),
            Some(&globals),
            None,
        )?;
        Ok(())
    });
}

#[test]
fn conformance_sorting_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── sort (MUST) ─────────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "sorting-sort-1d-float",
            "sort",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_1d(py, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])?;
                PyTuple::new(py, [arr])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-sort-2d-default-axis",
            "sort",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_2d(py, vec![vec![3.0, 1.0, 2.0], vec![6.0, 4.0, 5.0]])?;
                PyTuple::new(py, [arr])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-sort-axis0",
            "sort",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_2d(py, vec![vec![3.0, 1.0, 2.0], vec![1.0, 4.0, 5.0]])?;
                PyTuple::new(py, [arr])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 0_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-sort-kind-mergesort",
            "sort",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_1d(py, vec![5.0, 2.0, 8.0, 1.0, 9.0])?;
                PyTuple::new(py, [arr])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("kind", "mergesort")?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-sort-kind-stable-conflict",
            "sort",
            RequirementLevel::Should,
            CompareMode::Error,
            t,
            |py| {
                let arr = np_array_1d(py, vec![2.0, 1.0, 1.0])?;
                PyTuple::new(py, [arr])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("kind", "quicksort")?;
                kw.set_item("stable", false)?;
                Ok(Some(kw))
            },
        );

        // ─── argsort (MUST) ──────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "sorting-argsort-1d",
            "argsort",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_1d(py, vec![3.0, 1.0, 4.0, 1.0, 5.0])?;
                PyTuple::new(py, [arr])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-argsort-2d-axis1",
            "argsort",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_2d(py, vec![vec![3.0, 1.0, 2.0], vec![6.0, 4.0, 5.0]])?;
                PyTuple::new(py, [arr])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-argsort-stable",
            "argsort",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_1d(py, vec![2.0, 1.0, 2.0, 1.0, 2.0])?;
                PyTuple::new(py, [arr])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("kind", "stable")?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-argsort-kind-stable-conflict",
            "argsort",
            RequirementLevel::Should,
            CompareMode::Error,
            t,
            |py| {
                let arr = np_array_1d(py, vec![2.0, 1.0, 1.0])?;
                PyTuple::new(py, [arr])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("kind", "quicksort")?;
                kw.set_item("stable", true)?;
                Ok(Some(kw))
            },
        );

        // ─── lexsort (MUST) ──────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "sorting-lexsort-2-keys",
            "lexsort",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let numpy = py.import("numpy")?;
                let first_names = numpy.getattr("array")?.call1((vec!["a", "b", "a", "b"],))?;
                let last_names = numpy.getattr("array")?.call1((vec!["x", "x", "y", "y"],))?;
                let keys = PyTuple::new(py, [first_names, last_names])?;
                PyTuple::new(py, [keys])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-lexsort-numeric-keys",
            "lexsort",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let primary = np_array_1d_int(py, vec![1, 2, 1, 2])?;
                let secondary = np_array_1d_int(py, vec![10, 20, 20, 10])?;
                let keys = PyTuple::new(py, [secondary, primary])?;
                PyTuple::new(py, [keys])
            },
            no_kwargs,
        );

        // ─── partition (MUST) ────────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "sorting-partition-kth-scalar",
            "partition",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_1d(py, vec![3.0, 4.0, 2.0, 1.0, 5.0])?;
                PyTuple::new(py, [arr, 2_i64.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-partition-axis",
            "partition",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_2d(py, vec![vec![3.0, 1.0, 2.0], vec![6.0, 4.0, 5.0]])?;
                PyTuple::new(py, [arr, 1_i64.into_pyobject(py)?.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── argpartition (MUST) ─────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "sorting-argpartition-kth-scalar",
            "argpartition",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_1d(py, vec![3.0, 4.0, 2.0, 1.0, 5.0])?;
                PyTuple::new(py, [arr, 2_i64.into_pyobject(py)?.into_any()])
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "sorting-argpartition-axis",
            "argpartition",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_2d(py, vec![vec![3.0, 1.0, 2.0], vec![6.0, 4.0, 5.0]])?;
                PyTuple::new(py, [arr, 1_i64.into_pyobject(py)?.into_any()])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("axis", 1_i64)?;
                Ok(Some(kw))
            },
        );

        // ─── sort_complex (MUST) ─────────────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "sorting-sort_complex-basic",
            "sort_complex",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                let arr = np_array_complex(py)?;
                PyTuple::new(py, [arr])
            },
            no_kwargs,
        );

        let summary = TOTALS.summarize("sorting");
        eprintln!("\n=== fnp-python conformance matrix: sorting ===");
        eprintln!("{summary}");

        let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            failures, 0,
            "{failures} conformance case(s) failed in sorting family \
             (see JSON verdict lines above)",
        );

        Ok(())
    });
}

#[test]
fn lexsort_bounded_integer_valued_counting_sort_matches_numpy() {
    // Large-n, bounded-range integer-valued keys (both int and integral-float, incl. negatives)
    // exercise the stable counting-sort fast path. numpy's lexsort is stable; the counting scatter
    // in ascending original-index order must reproduce that tie-break byte-exactly.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(7)\n\
                 n = 300_000\n\
                 k0 = rng.integers(0, 100, n).astype(np.int64)\n\
                 k1 = rng.integers(-50, 50, n).astype(np.int32)\n\
                 k2 = rng.integers(0, 100, n).astype(np.int16)\n\
                 keys_int = (k0, k1, k2)\n\
                 keys_f64 = (k0.astype(np.float64), k1.astype(np.float64), k2.astype(np.float64))\n\
                 # heavy-tie case: tiny ranges so many equal composites test stability\n\
                 t0 = rng.integers(0, 3, n).astype(np.int64)\n\
                 t1 = rng.integers(-2, 2, n).astype(np.int64)\n\
                 keys_tie = (t0, t1)\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        for name in ["keys_int", "keys_f64", "keys_tie"] {
            let keys = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing keys"))?;
            let ours = module.getattr("lexsort")?.call1((&keys,))?;
            let theirs = numpy.getattr("lexsort")?.call1((&keys,))?;
            let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
            assert!(equal, "lexsort counting-sort {name} diverged from numpy");
            let dtype = ours.getattr("dtype")?.str()?.to_string();
            assert_eq!(dtype, "int64");
        }
        Ok(())
    });
}

#[test]
fn argsort_string_stable_packed_latin1_matches_numpy() {
    // Fixed-width Latin-1 U6/U8/S6 records (n >= 1<<18) take the gather-free packed-u64
    // (key, index) pair-sort path. Stable argsort ties break by original index; the packed
    // key must reproduce numpy's codepoint order and stable tie-break byte-exactly.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(11)\n\
                 n = 300_000\n\
                 u6 = rng.integers(97, 100, (n, 6), dtype=np.uint32).reshape(-1).view('U6')\n\
                 u8 = rng.integers(97, 123, (n, 8), dtype=np.uint32).reshape(-1).view('U8')\n\
                 s6 = np.array([bytes(r) for r in rng.integers(97, 100, (50_000, 6), dtype=np.uint8)], dtype='S6')\n\
                 s6 = np.tile(s6, 6)[:n]\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        let array_equal = numpy.getattr("array_equal")?;
        let kw = PyDict::new(py);
        kw.set_item("kind", "stable")?;
        for name in ["u6", "u8", "s6"] {
            let arr = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing arr"))?;
            let ours = module.getattr("argsort")?.call((&arr,), Some(&kw))?;
            let theirs = numpy.getattr("argsort")?.call((&arr,), Some(&kw))?;
            let equal: bool = array_equal.call1((&ours, &theirs))?.extract()?;
            assert!(equal, "packed Latin-1 {name} stable argsort diverged from numpy");
        }
        Ok(())
    });
}

#[test]
fn f16_sort_flat_widening_matches_numpy() {
    // np.sort(1-D float16) routes through the exact f32 widen/sort/narrow fast path for
    // finite non-(-0.0) data (bead deadlock-audit-98chw); NaN and -0.0 inputs must defer to
    // the numpy-identical fallback. Byte-equality (tobytes) is the bar in every case, over
    // random data with inf, subnormals, and dense ties, at n above the 1<<17 gate.
    with_fnp_and_numpy(|py, module, numpy| {
        let ns = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import numpy as np\n\
                 rng = np.random.default_rng(285)\n\
                 n = 200_000\n\
                 clean = np.round(rng.standard_normal(n), 2).astype(np.float16)\n\
                 clean[clean == 0] = np.float16(0.25)\n\
                 clean[:64] = np.float16(np.inf)\n\
                 clean[64:128] = np.float16(-np.inf)\n\
                 clean[128:192] = np.float16(6e-8)\n\
                 clean[192] = np.float16(0.0)\n\
                 with_nan = clean.copy()\n\
                 with_nan[777] = np.float16(np.nan)\n\
                 with_negzero = clean.copy()\n\
                 with_negzero[999] = np.float16(-0.0)\n\
                 small = clean[:1000].copy()\n"
            ),
            Some(&ns),
            Some(&ns),
        )?;
        for name in ["clean", "with_nan", "with_negzero", "small"] {
            let arr = ns
                .get_item(name)?
                .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err("missing arr"))?;
            let ours = module.getattr("sort")?.call1((&arr,))?;
            let theirs = numpy.getattr("sort")?.call1((&arr,))?;
            assert_eq!(
                ours.getattr("dtype")?.str()?.to_string(),
                theirs.getattr("dtype")?.str()?.to_string(),
                "{name}: dtype diverged"
            );
            let ours_bytes: Vec<u8> = ours.call_method0("tobytes")?.extract()?;
            let theirs_bytes: Vec<u8> = theirs.call_method0("tobytes")?.extract()?;
            if ours_bytes != theirs_bytes {
                // Divergence found. numpy 2.3.x's direct float16 np.sort emits globally
                // MIS-SORTED output on AVX-512 workers (x86-simd-sort fp16 defect: observed
                // on hz2/numpy 2.3.5 - equal bit-multisets, fnp == its own f32-widened sort,
                // yet direct != widened, which is impossible for two correct sorts). When the
                // oracle's own output is not ascending, byte-equality is unattainable and
                // WRONG to demand; require instead that fnp's output is a correct sort:
                // identical value multiset + ascending order + equal to numpy's own
                // f32-widened composition. On healthy numpy builds the strict byte-equality
                // above is the bar.
                let mut ours_ms: Vec<u16> = ours_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let mut theirs_ms: Vec<u16> = theirs_bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                ours_ms.sort_unstable();
                theirs_ms.sort_unstable();
                assert_eq!(ours_ms, theirs_ms, "{name}: f16 sort value multisets diverged");
                let np_sorted: bool = {
                    // is numpy's own direct output ascending? (NaN-free cases only reach here)
                    let d = numpy
                        .getattr("diff")?
                        .call1((theirs.call_method1("astype", ("float32",))?,))?;
                    numpy
                        .getattr("all")?
                        .call1((d.call_method1("__ge__", (0.0_f64,))?,))?
                        .extract()?
                };
                let widened = arr.call_method1("astype", ("float32",))?;
                let wsorted = numpy.getattr("sort")?.call1((&widened,))?;
                let wnarrow = wsorted.call_method1("astype", ("float16",))?;
                let wbytes: Vec<u8> = wnarrow.call_method0("tobytes")?.extract()?;
                let np_version: String = numpy.getattr("__version__")?.extract()?;
                assert!(
                    !np_sorted,
                    "{name}: fnp f16 sort diverged from a correctly-sorted numpy oracle (numpy {np_version})"
                );
                assert_eq!(
                    ours_bytes, wbytes,
                    "{name}: fnp f16 sort does not match the f32-widened reference either (numpy {np_version})"
                );
                eprintln!(
                    "NOTE {name}: numpy {np_version} direct f16 sort emitted NON-ASCENDING output \
                     (upstream x86-simd-sort fp16 defect); fnp output verified correct instead."
                );
            }
        }
        Ok(())
    });
}
