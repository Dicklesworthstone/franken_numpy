//! Conformance matrix: testing family.
//!
//! Differential parity for the `fnp_python.testing` submodule, which
//! re-exports 8 native assert_* implementations:
//!
//!   assert_equal
//!   assert_almost_equal
//!   assert_array_almost_equal
//!   assert_array_less
//!   assert_approx_equal
//!   assert_string_equal      (native body — pure-python difflib path)
//!   assert_allclose          (native body — UFuncArray::allclose path)
//!   assert_array_equal       (native body — UFuncArray::eq path)
//!
//! Each native body has two branches of interest:
//!   - the success path returns None silently
//!   - the failure path raises AssertionError with a message format
//!     that downstream log scrapers depend on
//!
//! The harness covers both branches per assert: pass-cases under
//! CompareMode::Strict (both return None → None == None) and fail-cases
//! under CompareMode::Error (both raise AssertionError, types matched).
//! SHOULD cases exercise the per-assert tolerance kwargs (`decimal`,
//! `significant`, `rtol`, `atol`, `equal_nan`, `strict`) since those are
//! the surface bits most likely to silently drift if a wrapper drops a
//! kwarg en route to the numpy fallback.

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

#[test]
fn conformance_testing_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;
        let testing_mod = module.getattr("testing").expect("fnp_python.testing");
        let np_testing_mod = numpy.getattr("testing").expect("numpy.testing");
        let testing = testing_mod
            .cast_into::<pyo3::types::PyModule>()
            .expect("fnp_python.testing should be a submodule");
        let np_testing = np_testing_mod
            .cast_into::<pyo3::types::PyModule>()
            .expect("numpy.testing should be a submodule");

        // ─── pass-path: both return None (MUST) ────────────────────────
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_equal-int-pass",
            "assert_equal",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [42_i64.into_pyobject(py)?, 42_i64.into_pyobject(py)?]),
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_almost_equal-pass",
            "assert_almost_equal",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        1.0_f64.into_pyobject(py)?,
                        1.00000001_f64.into_pyobject(py)?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_almost_equal-pass",
            "assert_array_almost_equal",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![1.000001, 2.000001, 3.000001])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_less-pass",
            "assert_array_less",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![10.0, 20.0, 30.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_approx_equal-pass",
            "assert_approx_equal",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        1234567.0_f64.into_pyobject(py)?,
                        1234567.5_f64.into_pyobject(py)?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_string_equal-pass",
            "assert_string_equal",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, ["hello world", "hello world"]),
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_allclose-pass",
            "assert_allclose",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_equal-pass",
            "assert_array_equal",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── fail-path: both raise AssertionError (MUST) ───────────────
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_equal-int-fail",
            "assert_equal",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| PyTuple::new(py, [1_i64.into_pyobject(py)?, 2_i64.into_pyobject(py)?]),
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_almost_equal-fail",
            "assert_almost_equal",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [1.0_f64.into_pyobject(py)?, 1.5_f64.into_pyobject(py)?],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_almost_equal-fail",
            "assert_array_almost_equal",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![1.0, 2.0, 5.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_less-fail",
            "assert_array_less",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![10.0, 20.0, 30.0])?,
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_approx_equal-fail",
            "assert_approx_equal",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        1.0_f64.into_pyobject(py)?,
                        2.0_f64.into_pyobject(py)?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_string_equal-fail",
            "assert_string_equal",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| PyTuple::new(py, ["foo", "bar"]),
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_allclose-fail",
            "assert_allclose",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![10.0, 20.0, 30.0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_equal-fail",
            "assert_array_equal",
            RequirementLevel::Must,
            CompareMode::Error,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![1.0, 2.0, 99.0])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── kwarg coverage (SHOULD) ───────────────────────────────────
        // decimal=2 loosens the tolerance — values that fail at default
        // precision should pass when explicitly relaxed. If the wrapper
        // dropped the `decimal` kwarg we'd see a fail-vs-pass mismatch.
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_almost_equal-decimal2-pass",
            "assert_almost_equal",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [1.0_f64.into_pyobject(py)?, 1.001_f64.into_pyobject(py)?],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("decimal", 2_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_almost_equal-decimal1-pass",
            "assert_array_almost_equal",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0])?,
                        np_array_1d(py, vec![1.04, 2.04])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("decimal", 1_i64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_allclose-rtol-loose-pass",
            "assert_allclose",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d(py, vec![1.0, 2.0, 3.0])?,
                        np_array_1d(py, vec![1.4, 2.4, 3.4])?,
                    ],
                )
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("rtol", 0.5_f64)?;
                Ok(Some(kw))
            },
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_allclose-equal_nan-true-pass",
            "assert_allclose",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                let nan = py.import("numpy")?.getattr("nan")?;
                let array = py.import("numpy")?.getattr("array")?;
                let a = array.call1((vec![1.0_f64, f64::NAN, 3.0],))?;
                let b = array.call1((vec![1.0_f64, f64::NAN, 3.0],))?;
                let _ = nan; // keep import side effect explicit
                PyTuple::new(py, [a, b])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("equal_nan", true)?;
                Ok(Some(kw))
            },
        );
        // strict=True on assert_array_equal forces dtype/shape match —
        // the native body delegates to numpy, so we expect identical
        // pass/fail behavior.
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_array_equal-strict-dtype-mismatch",
            "assert_array_equal",
            RequirementLevel::Should,
            CompareMode::Error,
            t,
            |py| {
                let array = py.import("numpy")?.getattr("array")?;
                let kw_int = PyDict::new(py);
                kw_int.set_item("dtype", "int64")?;
                let kw_float = PyDict::new(py);
                kw_float.set_item("dtype", "float64")?;
                let a = array.call((vec![1_i64, 2, 3],), Some(&kw_int))?;
                let b = array.call((vec![1.0_f64, 2.0, 3.0],), Some(&kw_float))?;
                PyTuple::new(py, [a, b])
            },
            |py| {
                let kw = PyDict::new(py);
                kw.set_item("strict", true)?;
                Ok(Some(kw))
            },
        );

        // ─── multiline string diff (MAY) ───────────────────────────────
        // assert_string_equal renders a difflib diff in its
        // AssertionError message. Numpy's exact format is the contract;
        // we reproduce it via stdlib difflib in the native body. Both
        // raise AssertionError on this mismatch — the Error compare mode
        // verifies the type, not the body. The 2axo/u98f commits already
        // pin the message format via dedicated unit tests.
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_string_equal-multiline-fail",
            "assert_string_equal",
            RequirementLevel::May,
            CompareMode::Error,
            t,
            |py| PyTuple::new(py, ["alpha\nbeta\ngamma\n", "alpha\nDELTA\ngamma\n"]),
            no_kwargs,
        );
        run_case(
            py,
            &testing,
            &np_testing,
            "testing-assert_equal-nested-dict-pass",
            "assert_equal",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| {
                let dict_a = PyDict::new(py);
                dict_a.set_item("k1", vec![1_i64, 2, 3])?;
                dict_a.set_item("k2", "hello")?;
                let dict_b = PyDict::new(py);
                dict_b.set_item("k1", vec![1_i64, 2, 3])?;
                dict_b.set_item("k2", "hello")?;
                PyTuple::new(py, [dict_a, dict_b])
            },
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("testing");
    eprintln!("\n=== fnp-python conformance matrix: testing ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in testing family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
