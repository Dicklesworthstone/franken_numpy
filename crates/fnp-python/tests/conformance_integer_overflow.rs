//! Conformance regression tests for the integer-overflow / wraparound contract of
//! reductions and products that promote or preserve integer dtypes.
//!
//! NumPy accumulates these in a fixed native integer accumulator with
//! two's-complement wraparound (sum/prod widen narrow ints to int64/uint64;
//! trace/cross/vdot follow the same accumulator rules). fnp historically computed
//! several of these in f64 and cast back, which silently lost precision past 2^53,
//! raised on out-of-range f64, or widened the result dtype. Those were fixed in
//! the trace (int64/uint64), vdot (dtype + int32 precision), sum/prod (narrow-int
//! wrapping), and cross (int32/uint32) work. These cases pin the contract at the
//! PYTHON API boundary (the Rust unit tests cover the kernels, but the pyo3
//! wrappers can regress independently) by diffing fnp against the NumPy oracle
//! byte-for-byte and dtype-for-dtype.

mod common;

use common::with_fnp_and_numpy;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

/// Build a numpy ndarray of the given integer dtype from i128 values (cast into
/// range by numpy itself via the dtype kwarg), shaped per `shape`.
fn np_int_array<'py>(
    py: Python<'py>,
    numpy: &Bound<'py, PyModule>,
    values: &[i128],
    shape: &[usize],
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // numpy.array(values, dtype=dtype).reshape(shape) — values are full-magnitude
    // Python ints (i128 → PyInt, so 2^63-range uint64 values construct correctly);
    // every value must already fit the requested dtype (numpy rejects, not wraps,
    // an out-of-range Python int at construction time).
    let py_vals = PyList::new(py, values.iter().copied())?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    let arr = numpy.getattr("array")?.call((py_vals,), Some(&kwargs))?;
    if shape.len() > 1 {
        let shape_tuple = PyTuple::new(py, shape.iter().copied())?;
        arr.call_method1("reshape", (shape_tuple,))
    } else {
        Ok(arr)
    }
}

/// Assert fnp and numpy return byte-identical, dtype-identical results.
fn assert_same(label: &str, fnp_res: &Bound<'_, PyAny>, np_res: &Bound<'_, PyAny>) -> PyResult<()> {
    let np = fnp_res.py().import("numpy")?;
    let g = np.getattr("asarray")?.call1((fnp_res,))?;
    let x = np.getattr("asarray")?.call1((np_res,))?;
    let g_dtype = g.getattr("dtype")?.str()?.to_string();
    let x_dtype = x.getattr("dtype")?.str()?.to_string();
    assert_eq!(g_dtype, x_dtype, "{label}: dtype mismatch");
    let g_bytes = g.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    let x_bytes = x.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    assert_eq!(
        g_bytes, x_bytes,
        "{label}: bytes mismatch (fnp={:?} numpy={:?})",
        g.call_method0("tolist")?.str()?.to_string(),
        x.call_method0("tolist")?.str()?.to_string()
    );
    Ok(())
}

const INT_DTYPES: &[&str] = &[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

#[test]
fn integer_overflow_reductions_and_products_match_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        // ── sum / prod: narrow ints promote to int64/uint64 and wrap ──
        // A row of moderately-large values whose product overflows the accumulator.
        for dt in INT_DTYPES {
            let row: Vec<i128> = vec![9, 8, 7, 6, 5, 11, 13, 3];
            let a = np_int_array(py, &numpy, &row, &[1, row.len()], dt)?;
            for op in ["sum", "prod"] {
                for axis in [None, Some(0_i64), Some(1_i64)] {
                    let kwargs = PyDict::new(py);
                    if let Some(ax) = axis {
                        kwargs.set_item("axis", ax)?;
                    }
                    let fnp_res = module.getattr(op)?.call((a.clone(),), Some(&kwargs))?;
                    let np_res = numpy.getattr(op)?.call((a.clone(),), Some(&kwargs))?;
                    assert_same(&format!("{op} {dt} axis={axis:?}"), &fnp_res, &np_res)?;
                }
            }
        }

        // ── trace: int64/uint64 diagonals whose sum overflows the (preserved)
        // accumulator wrap natively; narrow ints promote to int64/uint64 ──
        // Overflow case (only int64/uint64 can hold values large enough to wrap
        // their own accumulator): two near-max diagonal entries.
        let i64_mat: Vec<i128> = vec![1_i128 << 62, 0, 0, 0, 1_i128 << 62, 0, 0, 0, 8];
        let a = np_int_array(py, &numpy, &i64_mat, &[3, 3], "int64")?;
        assert_same(
            "trace int64 overflow",
            &module.getattr("trace")?.call1((a.clone(),))?,
            &numpy.getattr("trace")?.call1((a.clone(),))?,
        )?;
        let u64_mat: Vec<i128> = vec![1_i128 << 63, 0, 0, 0, 1_i128 << 63, 0, 0, 0, 8];
        let a = np_int_array(py, &numpy, &u64_mat, &[3, 3], "uint64")?;
        assert_same(
            "trace uint64 overflow",
            &module.getattr("trace")?.call1((a.clone(),))?,
            &numpy.getattr("trace")?.call1((a.clone(),))?,
        )?;
        // Dtype-promotion case across all int dtypes (small in-range diagonal).
        for dt in INT_DTYPES {
            let mat: Vec<i128> = vec![5, 0, 0, 0, 7, 0, 0, 0, 11];
            let a = np_int_array(py, &numpy, &mat, &[3, 3], dt)?;
            assert_same(
                &format!("trace {dt}"),
                &module.getattr("trace")?.call1((a.clone(),))?,
                &numpy.getattr("trace")?.call1((a.clone(),))?,
            )?;
        }

        // ── vdot: preserves input dtype with wraparound (not the sum accumulator) ──
        for dt in INT_DTYPES {
            let a = np_int_array(py, &numpy, &[100, 100, 3], &[3], dt)?;
            let b = np_int_array(py, &numpy, &[100, 100, 7], &[3], dt)?;
            let fnp_res = module.getattr("vdot")?.call1((a.clone(), b.clone()))?;
            let np_res = numpy.getattr("vdot")?.call1((a.clone(), b.clone()))?;
            assert_same(&format!("vdot {dt}"), &fnp_res, &np_res)?;
        }

        // ── cross: preserves the promoted integer dtype with wraparound ──
        for dt in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"] {
            // Values fit int8 (so they construct for every dtype) but their cross
            // component products (e.g. 120*110, 100*90) overflow int8/int16 → wrap.
            let a = np_int_array(py, &numpy, &[120, 100, 3], &[3], dt)?;
            let b = np_int_array(py, &numpy, &[5, 110, 90], &[3], dt)?;
            let fnp_res = module.getattr("cross")?.call1((a.clone(), b.clone()))?;
            let np_res = numpy.getattr("cross")?.call1((a.clone(), b.clone()))?;
            assert_same(&format!("cross {dt}"), &fnp_res, &np_res)?;
        }

        Ok(())
    });
}
