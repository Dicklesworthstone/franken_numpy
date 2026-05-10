//! Diagnostic conformance tests for fnp-python wrappers.
//!
//! These cases compare observable warning and exception surfaces against NumPy:
//! outcome class, exception type, and warning category order/count.

use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| format!("python3 stdin unavailable\nScript: {script}"))?
        .write_all(script.as_bytes())
        .map_err(|error| format!("failed to write python script: {error}\nScript: {script}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("python3 wait failed: {error}\nScript: {script}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn fnp_script(body: String) -> String {
    let library_name = format!(
        "{}fnp_python{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let module_path = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join(&library_name)))
        .unwrap_or_else(|| library_name.into());
    let module_literal = format!("{module_path:?}");
    format!(
        "import importlib.util\n\
         import numpy as np\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

#[test]
fn warnings_and_exceptions_match_numpy_by_surface() -> Result<(), String> {
    let script = fnp_script(
        r#"
import io
import json
import warnings
from numpy.lib import recfunctions as rfn

np.seterr(all="warn")

def observe(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn()
        except Exception as exc:
            return {
                "outcome": "exception",
                "exception": type(exc).__name__,
                "warnings": [type(item.message).__name__ for item in caught],
            }
        return {
            "outcome": "success",
            "exception": None,
            "warnings": [type(item.message).__name__ for item in caught],
        }

def compare(fnp_fn, numpy_fn):
    return observe(fnp_fn), observe(numpy_fn)

singular = np.array([[1.0, 2.0], [2.0, 4.0]])
base_struct = np.array([(1, 1.5), (2, 2.5)], dtype=[("a", "i4"), ("b", "f8")])

cases = [
    (
        "random_integers_deprecated_warning",
        "random",
        lambda: compare(
            lambda: fnp.random.RandomState(123).random_integers(1, 3),
            lambda: np.random.RandomState(123).random_integers(1, 3),
        ),
    ),
    (
        "true_divide_float_zero_warning",
        "arithmetic",
        lambda: compare(
            lambda: fnp.true_divide(np.array([1.0]), np.array([0.0])),
            lambda: np.true_divide(np.array([1.0]), np.array([0.0])),
        ),
    ),
    (
        "floor_divide_float_zero_warning",
        "arithmetic",
        lambda: compare(
            lambda: fnp.floor_divide(np.array([1.0]), np.array([0.0])),
            lambda: np.floor_divide(np.array([1.0]), np.array([0.0])),
        ),
    ),
    (
        "array_bad_dtype_typeerror",
        "array-creation",
        lambda: compare(lambda: fnp.array([1], dtype="not-a-dtype"), lambda: np.array([1], dtype="not-a-dtype")),
    ),
    (
        "asarray_chkfinite_nan_valueerror",
        "array-creation",
        lambda: compare(lambda: fnp.asarray_chkfinite([np.nan]), lambda: np.asarray_chkfinite([np.nan])),
    ),
    (
        "broadcast_shapes_mismatch_valueerror",
        "shape",
        lambda: compare(lambda: fnp.broadcast_shapes((2,), (3,)), lambda: np.broadcast_shapes((2,), (3,))),
    ),
    (
        "reshape_bad_size_valueerror",
        "shape",
        lambda: compare(lambda: fnp.reshape(np.arange(4), (3, 2)), lambda: np.reshape(np.arange(4), (3, 2))),
    ),
    (
        "reshape_multiple_unknowns_valueerror",
        "shape",
        lambda: compare(lambda: fnp.reshape(np.arange(4), (-1, -1)), lambda: np.reshape(np.arange(4), (-1, -1))),
    ),
    (
        "can_cast_bad_casting_valueerror",
        "dtype",
        lambda: compare(
            lambda: fnp.can_cast("int64", "int32", casting="not-a-casting"),
            lambda: np.can_cast("int64", "int32", casting="not-a-casting"),
        ),
    ),
    (
        "promote_types_incompatible_dtype_error",
        "dtype",
        lambda: compare(
            lambda: fnp.promote_types("datetime64[D]", "complex64"),
            lambda: np.promote_types("datetime64[D]", "complex64"),
        ),
    ),
    (
        "common_type_string_typeerror",
        "dtype",
        lambda: compare(
            lambda: fnp.common_type(np.array(["a"], dtype="U1")),
            lambda: np.common_type(np.array(["a"], dtype="U1")),
        ),
    ),
    (
        "take_out_of_bounds_indexerror",
        "indexing",
        lambda: compare(lambda: fnp.take(np.array([1, 2]), [5]), lambda: np.take(np.array([1, 2]), [5])),
    ),
    (
        "take_along_axis_oob_indexerror",
        "indexing",
        lambda: compare(
            lambda: fnp.take_along_axis(np.array([[1, 2]]), np.array([[2]]), axis=1),
            lambda: np.take_along_axis(np.array([[1, 2]]), np.array([[2]]), axis=1),
        ),
    ),
    (
        "put_out_of_bounds_indexerror",
        "indexing",
        lambda: compare(
            lambda: fnp.put(np.array([1, 2]), [5], [9]),
            lambda: np.put(np.array([1, 2]), [5], [9]),
        ),
    ),
    (
        "put_along_axis_oob_indexerror",
        "indexing",
        lambda: compare(
            lambda: fnp.put_along_axis(np.zeros((1, 2)), np.array([[2]]), np.ones((1, 1)), axis=1),
            lambda: np.put_along_axis(np.zeros((1, 2)), np.array([[2]]), np.ones((1, 1)), axis=1),
        ),
    ),
    (
        "concatenate_axis_oob_axiserror",
        "shape",
        lambda: compare(
            lambda: fnp.concatenate([np.array([1]), np.array([2])], axis=2),
            lambda: np.concatenate([np.array([1]), np.array([2])], axis=2),
        ),
    ),
    (
        "stack_shape_mismatch_valueerror",
        "shape",
        lambda: compare(
            lambda: fnp.stack([np.array([1]), np.array([1, 2])]),
            lambda: np.stack([np.array([1]), np.array([1, 2])]),
        ),
    ),
    (
        "split_uneven_valueerror",
        "shape",
        lambda: compare(lambda: fnp.split(np.arange(5), 2), lambda: np.split(np.arange(5), 2)),
    ),
    (
        "argmax_axis_oob_axiserror",
        "indexing",
        lambda: compare(lambda: fnp.argmax(np.array([1, 2]), axis=2), lambda: np.argmax(np.array([1, 2]), axis=2)),
    ),
    (
        "argmin_axis_oob_axiserror",
        "indexing",
        lambda: compare(lambda: fnp.argmin(np.array([1, 2]), axis=2), lambda: np.argmin(np.array([1, 2]), axis=2)),
    ),
    (
        "ptp_axis_oob_axiserror",
        "statistics",
        lambda: compare(lambda: fnp.ptp(np.array([1, 2]), axis=2), lambda: np.ptp(np.array([1, 2]), axis=2)),
    ),
    (
        "percentile_empty_indexerror",
        "statistics",
        lambda: compare(lambda: fnp.percentile(np.array([]), 50), lambda: np.percentile(np.array([]), 50)),
    ),
    (
        "quantile_empty_indexerror",
        "statistics",
        lambda: compare(lambda: fnp.quantile(np.array([]), 0.5), lambda: np.quantile(np.array([]), 0.5)),
    ),
    (
        "diagonal_repeated_axis_valueerror",
        "shape",
        lambda: compare(
            lambda: fnp.diagonal(np.arange(4).reshape(2, 2), axis1=0, axis2=0),
            lambda: np.diagonal(np.arange(4).reshape(2, 2), axis1=0, axis2=0),
        ),
    ),
    (
        "choose_mode_bad_valueerror",
        "indexing",
        lambda: compare(
            lambda: fnp.choose([0, 1], [np.array([1, 2]), np.array([3, 4])], mode="not-a-mode"),
            lambda: np.choose([0, 1], [np.array([1, 2]), np.array([3, 4])], mode="not-a-mode"),
        ),
    ),
    (
        "linalg_inv_singular_linalgerror",
        "linalg",
        lambda: compare(lambda: fnp.linalg.inv(singular), lambda: np.linalg.inv(singular)),
    ),
    (
        "linalg_solve_singular_linalgerror",
        "linalg",
        lambda: compare(
            lambda: fnp.linalg.solve(singular, np.array([1.0, 2.0])),
            lambda: np.linalg.solve(singular, np.array([1.0, 2.0])),
        ),
    ),
    (
        "testing_assert_equal_assertionerror",
        "testing",
        lambda: compare(lambda: fnp.testing.assert_equal(1, 2), lambda: np.testing.assert_equal(1, 2)),
    ),
    (
        "testing_assert_allclose_assertionerror",
        "testing",
        lambda: compare(
            lambda: fnp.testing.assert_allclose([1.0], [2.0]),
            lambda: np.testing.assert_allclose([1.0], [2.0]),
        ),
    ),
    (
        "recfunctions_merge_duplicate_valueerror",
        "recfunctions",
        lambda: compare(
            lambda: fnp.recfunctions_merge_arrays([base_struct[["a"]], base_struct[["a"]]], usemask=False),
            lambda: rfn.merge_arrays([base_struct[["a"]], base_struct[["a"]]], usemask=False),
        ),
    ),
    (
        "recfunctions_append_duplicate_valueerror",
        "recfunctions",
        lambda: compare(
            lambda: fnp.recfunctions_append_fields(base_struct, "a", np.array([7, 8]), usemask=False),
            lambda: rfn.append_fields(base_struct, "a", np.array([7, 8]), usemask=False),
        ),
    ),
    (
        "recfunctions_unstructured_field_count_valueerror",
        "recfunctions",
        lambda: compare(
            lambda: fnp.recfunctions_unstructured_to_structured(
                np.arange(6).reshape(2, 3),
                dtype=np.dtype([("x", "i4"), ("y", "i4")]),
            ),
            lambda: rfn.unstructured_to_structured(
                np.arange(6).reshape(2, 3),
                dtype=np.dtype([("x", "i4"), ("y", "i4")]),
            ),
        ),
    ),
    (
        "histogram_zero_bins_valueerror",
        "histogram",
        lambda: compare(lambda: fnp.histogram([1, 2], bins=0), lambda: np.histogram([1, 2], bins=0)),
    ),
    (
        "bincount_negative_valueerror",
        "histogram",
        lambda: compare(lambda: fnp.bincount([-1, 2]), lambda: np.bincount([-1, 2])),
    ),
    (
        "fromiter_negative_count_valueerror",
        "array-creation",
        lambda: compare(
            lambda: fnp.fromiter(iter([1, 2]), dtype=np.int64, count=-2),
            lambda: np.fromiter(iter([1, 2]), dtype=np.int64, count=-2),
        ),
    ),
]

failures = []
for case_id, family, check in cases:
    try:
        got, expected = check()
    except Exception as exc:
        failures.append({
            "id": case_id,
            "family": family,
            "reason": "case-error",
            "exception": type(exc).__name__,
            "message": str(exc),
        })
        continue
    if got != expected:
        failures.append({
            "id": case_id,
            "family": family,
            "got": got,
            "expected": expected,
        })

print(json.dumps(failures, sort_keys=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "[]",
        "fnp-python warning/exception diagnostics diverged from NumPy"
    );
    Ok(())
}
