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
