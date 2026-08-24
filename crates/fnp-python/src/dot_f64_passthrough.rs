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
    if out.is_some_and(|value| !value.bind(py).is_none()) || blas_is_single_threaded() {
        return Ok(None);
    }

    let numpy = py.import("numpy")?;
    let ndarray = numpy.getattr(intern!(py, "ndarray"))?;
    if !a.is_exact_instance(&ndarray)
        || !b.is_exact_instance(&ndarray)
        || !dtype_is_f64(py, a)?
        || !dtype_is_f64(py, b)?
    {
        return Ok(None);
    }

    Ok(Some(
        numpy
            .getattr(intern!(py, "dot"))?
            .call((a, b), None)?
            .unbind(),
    ))
}
