//! Conformance matrix: bitwise family.
//!
//! Differential parity for fnp_python's bitwise surface:
//!
//!   bitwise_and, bitwise_or, bitwise_xor,
//!   bitwise_not, bitwise_invert, invert,
//!   left_shift, right_shift,
//!   bitwise_left_shift, bitwise_right_shift,
//!   bitwise_count
//!
//! All 11 are currently `core_numpy_passthrough` wrappers, so this
//! harness is primarily a regression gate against future native ports
//! that might silently drift on dtype, broadcasting, or operand order.

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

fn np_array_1d_u8<'py>(
    py: Python<'py>,
    values: Vec<u32>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    // Vec<u8> would marshal to Python `bytes` (numpy then tries to parse
    // it as a literal int) — pass values as i64-promoted ints via Vec<u32>
    // and let the dtype kwarg force uint8 storage.
    let array = py.import("numpy")?.getattr("array")?;
    let kw = PyDict::new(py);
    kw.set_item("dtype", "uint8")?;
    array.call((values,), Some(&kw))
}

fn np_array_1d_b<'py>(
    py: Python<'py>,
    values: Vec<bool>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyAny>> {
    py.import("numpy")?.getattr("array")?.call1((values,))
}

#[test]
fn conformance_bitwise_matrix() {
    static TOTALS: Totals = Totals::new();

    with_fnp_and_numpy(|py, module, numpy| {
        let t = &TOTALS;

        // ─── bitwise_and / or / xor (MUST + SHOULD bool/uint8) ─────────
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-and-int64",
            "bitwise_and",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![0b1100, 0b1010, 0b1111])?,
                        np_array_1d_i(py, vec![0b1010, 0b0101, 0b1001])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-or-int64",
            "bitwise_or",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![0b1100, 0b1010, 0b1111])?,
                        np_array_1d_i(py, vec![0b1010, 0b0101, 0b1001])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-xor-int64",
            "bitwise_xor",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![0b1100, 0b1010, 0b1111])?,
                        np_array_1d_i(py, vec![0b1010, 0b0101, 0b1001])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-and-bool",
            "bitwise_and",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_b(py, vec![true, true, false, false])?,
                        np_array_1d_b(py, vec![true, false, true, false])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-xor-uint8",
            "bitwise_xor",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_u8(py, vec![0xff, 0x0f, 0xa5])?,
                        np_array_1d_u8(py, vec![0xaa, 0xf0, 0x5a])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── bitwise_not / bitwise_invert / invert (MUST) ──────────────
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-not-int64",
            "bitwise_not",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, -1, 42])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-invert-int64-array-api-alias",
            "bitwise_invert",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, -1, 42])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-invert-toplevel",
            "invert",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_u8(py, vec![0x00, 0xff, 0xa5])?]),
            no_kwargs,
        );

        // ─── left_shift / right_shift (MUST + SHOULD aliases) ──────────
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-left_shift-int64",
            "left_shift",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 4, 8])?,
                        np_array_1d_i(py, vec![1, 2, 3, 0])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-right_shift-int64",
            "right_shift",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![16, 32, 64, 128])?,
                        np_array_1d_i(py, vec![1, 2, 3, 4])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-left_shift-array-api-alias",
            "bitwise_left_shift",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![1, 2, 4])?,
                        np_array_1d_i(py, vec![3, 2, 1])?,
                    ],
                )
            },
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-right_shift-array-api-alias",
            "bitwise_right_shift",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        np_array_1d_i(py, vec![64, 32, 16])?,
                        np_array_1d_i(py, vec![3, 2, 1])?,
                    ],
                )
            },
            no_kwargs,
        );

        // ─── bitwise_count (MUST + SHOULD uint8) ───────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-count-int64",
            "bitwise_count",
            RequirementLevel::Must,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![0, 1, 7, 255])?]),
            no_kwargs,
        );
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-count-uint8",
            "bitwise_count",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_u8(py, vec![0x00, 0xff, 0xa5, 0x80])?]),
            no_kwargs,
        );

        // ─── broadcasting (SHOULD) ─────────────────────────────────────
        // Scalar ⊕ array — left operand broadcasts to match right shape.
        // If a wrapper accidentally swapped operand order, bitwise_and
        // wouldn't surface it (commutative) but left_shift would.
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-left_shift-scalar-broadcast",
            "left_shift",
            RequirementLevel::Should,
            CompareMode::Strict,
            t,
            |py| {
                PyTuple::new(
                    py,
                    [
                        1_i64.into_pyobject(py)?.into_any(),
                        np_array_1d_i(py, vec![0, 1, 2, 3, 4])?.into_any(),
                    ],
                )
            },
            no_kwargs,
        );

        // ─── degenerate shapes (MAY) ───────────────────────────────────
        run_case(
            py,
            &module,
            &numpy,
            "bitwise-and-empty",
            "bitwise_and",
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
            "bitwise-not-empty",
            "bitwise_not",
            RequirementLevel::May,
            CompareMode::Strict,
            t,
            |py| PyTuple::new(py, [np_array_1d_i(py, vec![])?]),
            no_kwargs,
        );

        Ok(())
    });

    let summary = TOTALS.summarize("bitwise");
    eprintln!("\n=== fnp-python conformance matrix: bitwise ===");
    eprintln!("{summary}");
    let failures = TOTALS.fail_count.load(std::sync::atomic::Ordering::Relaxed);
    if failures > 0 {
        panic!(
            "{failures} conformance case(s) failed in bitwise family \
             (MUST failures already panicked; SHOULD/MAY failures aggregated above)"
        );
    }
}
