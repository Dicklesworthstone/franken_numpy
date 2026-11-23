//! Full-thread f64 `dot` dispatch.
//!
//! The native f64 GEMM is intentionally disabled unless a BLAS implementation
//! is pinned to one thread. In the normal full-thread regime, exact f64 ndarrays
//! therefore always reach NumPy after several integer/f16 probes that cannot
//! apply. This module recognizes that already-delegating regime and performs
//! the live NumPy call directly, preserving NumPy's SIMD/pairwise order.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyAny;

fn blas_is_single_threaded() -> bool {
    [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    .iter()
    .any(|var| matches!(std::env::var(var), Ok(ref value) if value.trim() == "1"))
}

fn dtype_is_f64(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(value
        .getattr(intern!(py, "dtype"))?
        .getattr(intern!(py, "char"))?
        .extract::<char>()?
        == 'd')
}

/// Return NumPy's live `dot` result for exact float64 ndarrays when native GEMM
/// is profile-disabled. Subclasses and all non-f64 inputs retain the existing
/// dispatcher, and a supplied non-None `out` retains NumPy's normal path.
pub(super) fn try_full_thread_f64_dot(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Py<PyAny>>,
) -> PyResult<Option<Py<PyAny>>> {
    try_full_thread_f64_dot_with_single_threaded(py, a, b, out, blas_is_single_threaded())
}

fn try_full_thread_f64_dot_with_single_threaded(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Py<PyAny>>,
    single_threaded: bool,
) -> PyResult<Option<Py<PyAny>>> {
    if out.is_some_and(|value| !value.bind(py).is_none()) || single_threaded {
        return Ok(None);
    }

    let ndarray = crate::cached_ndarray_type(py)?;
    if !a.is_exact_instance(ndarray)
        || !b.is_exact_instance(ndarray)
        || !dtype_is_f64(py, a)?
        || !dtype_is_f64(py, b)?
    {
        return Ok(None);
    }

    let numpy = crate::cached_numpy(py)?;
    Ok(Some(
        numpy
            .getattr(intern!(py, "dot"))?
            .call((a, b), None)?
            .unbind(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        blas_is_single_threaded, try_full_thread_f64_dot,
        try_full_thread_f64_dot_with_single_threaded,
    };
    use pyo3::prelude::*;

    #[test]
    fn full_thread_f64_dot_passthrough_matches_numpy_and_declines_non_f64() {
        Python::initialize();
        Python::attach(|py| {
            let np = py.import("numpy").expect("numpy available");
            let a_f64 = np
                .call_method1("arange", (4096_i64,))
                .unwrap()
                .call_method1("reshape", ((64_usize, 64_usize),))
                .unwrap()
                .call_method1("astype", ("float64",))
                .unwrap();
            let b_f64 = np
                .call_method1("arange", (4160_i64,))
                .unwrap()
                .call_method1("reshape", ((64_usize, 65_usize),))
                .unwrap()
                .call_method1("astype", ("float64",))
                .unwrap();

            let passthrough =
                try_full_thread_f64_dot_with_single_threaded(py, &a_f64, &b_f64, None, false)
                    .expect("passthrough call succeeds")
                    .expect("exact f64 ndarrays engage passthrough");
            assert!(
                try_full_thread_f64_dot_with_single_threaded(py, &a_f64, &b_f64, None, true)
                    .expect("single-threaded probe succeeds")
                    .is_none()
            );
            let ambient = try_full_thread_f64_dot(py, &a_f64, &b_f64, None)
                .expect("ambient passthrough probe succeeds");
            assert_eq!(ambient.is_none(), blas_is_single_threaded());

            let via_dot = crate::dot(py, a_f64.clone().unbind(), b_f64.clone().unbind(), None)
                .expect("crate::dot succeeds");
            let expected = np.call_method1("dot", (&a_f64, &b_f64)).unwrap();

            assert!(
                np.call_method1("array_equal", (passthrough.bind(py), &expected))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            assert!(
                np.call_method1("array_equal", (via_dot.bind(py), &expected))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );

            let a_i32 = a_f64.call_method1("astype", ("int32",)).unwrap();
            assert!(
                try_full_thread_f64_dot_with_single_threaded(py, &a_i32, &b_f64, None, false)
                    .expect("non-f64 probe succeeds")
                    .is_none()
            );
        });
    }
}
