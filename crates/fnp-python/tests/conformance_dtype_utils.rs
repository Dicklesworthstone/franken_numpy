//! Conformance tests for numpy dtype utility functions against NumPy oracle.
//!
//! Tests broadcast_shapes, can_cast, common_type, promote_types.

use std::process::Command;

fn numpy_oracle(script: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .args(["-c", script])
        .output()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
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

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shapes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn broadcast_shapes_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_shapes((3, 4), (4,))
expected = np.broadcast_shapes((3, 4), (4,))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_shapes basic should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_shapes_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_shapes((3, 4), ())
expected = np.broadcast_shapes((3, 4), ())
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_shapes scalar should match numpy"
    );
    Ok(())
}

#[test]
fn broadcast_shapes_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_shapes((1, 2), (3, 1), (3, 2))
expected = np.broadcast_shapes((1, 2), (3, 1), (3, 2))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "broadcast_shapes multiple should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// can_cast
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn can_cast_int_to_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('int32', 'float64')
expected = np.can_cast('int32', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "can_cast int to float should match numpy"
    );
    Ok(())
}

#[test]
fn can_cast_float_to_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('float64', 'int32')
expected = np.can_cast('float64', 'int32')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "can_cast float to int should match numpy"
    );
    Ok(())
}

#[test]
fn can_cast_same_type() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('float64', 'float64')
expected = np.can_cast('float64', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "can_cast same type should match numpy"
    );
    Ok(())
}

#[test]
fn can_cast_with_casting() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('int64', 'int32', casting='same_kind')
expected = np.can_cast('int64', 'int32', casting='same_kind')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "can_cast with casting should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// common_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn common_type_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2], dtype='int32')
b = np.array([1.0, 2.0], dtype='float64')
result = fnp.common_type(a, b)
expected = np.common_type(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "common_type int+float should match numpy"
    );
    Ok(())
}

#[test]
fn common_type_floats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0], dtype='float32')
b = np.array([1.0], dtype='float64')
result = fnp.common_type(a, b)
expected = np.common_type(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "common_type floats should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// promote_types
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn promote_types_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.promote_types('int32', 'float32')
expected = np.promote_types('int32', 'float32')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "promote_types int+float should match numpy"
    );
    Ok(())
}

#[test]
fn promote_types_ints() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.promote_types('int8', 'int16')
expected = np.promote_types('int8', 'int16')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "promote_types ints should match numpy"
    );
    Ok(())
}

#[test]
fn promote_types_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.promote_types('float64', 'float64')
expected = np.promote_types('float64', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "promote_types same should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dtype predicates and datetime dtype helpers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isdtype_issubdtype_isfortran_and_isnat_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
isdtype_cases = [
    (np.dtype("int32"), "integral"),
    (np.dtype("uint64"), "unsigned integer"),
    (np.dtype("float64"), "real floating"),
    (np.dtype("complex128"), "complex floating"),
    (np.dtype("bool"), "bool"),
]
issubdtype_cases = [
    (np.int32, np.integer),
    (np.float64, np.floating),
    (np.complex128, np.number),
    (np.bool_, np.integer),
]
c_array = np.arange(6).reshape(2, 3)
f_array = np.asfortranarray(c_array)
dates = np.array(["NaT", "2024-01-02", "2024-01-03"], dtype="datetime64[D]")
durations = np.array(["NaT", 1, 2], dtype="timedelta64[D]")
match = (
    all(fnp.isdtype(dtype, kind) == np.isdtype(dtype, kind) for dtype, kind in isdtype_cases)
    and all(fnp.issubdtype(lhs, rhs) == np.issubdtype(lhs, rhs) for lhs, rhs in issubdtype_cases)
    and fnp.isfortran(c_array) == np.isfortran(c_array)
    and fnp.isfortran(f_array) == np.isfortran(f_array)
    and np.array_equal(fnp.isnat(dates), np.isnat(dates))
    and np.array_equal(fnp.isnat(durations), np.isnat(durations))
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isdtype/issubdtype/isfortran/isnat should match numpy"
    );
    Ok(())
}

#[test]
fn datetime_data_and_datetime_as_string_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
dates = np.array(["2024-01-02T03:04:05.123", "NaT"], dtype="datetime64[ms]")
dtypes = [
    np.dtype("datetime64[ms]"),
    np.dtype("datetime64[D]"),
    np.dtype("timedelta64[us]"),
]
match = (
    all(fnp.datetime_data(dtype) == np.datetime_data(dtype) for dtype in dtypes)
    and np.array_equal(
        fnp.datetime_as_string(dates, unit="ms"),
        np.datetime_as_string(dates, unit="ms"),
    )
    and np.array_equal(
        fnp.datetime_as_string(dates, unit="s", timezone="UTC"),
        np.datetime_as_string(dates, unit="s", timezone="UTC"),
    )
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "datetime_data/datetime_as_string should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// structured-array recfunctions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn recfunctions_drop_rename_append_flat_structured_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib import recfunctions as rfn

base = np.array(
    [(1, 1.5, b"aa"), (2, 2.5, b"bb")],
    dtype=[("a", "i4"), ("b", "f8"), ("c", "S2")],
)

drop_result = fnp.recfunctions_drop_fields(base, ["b"], usemask=False)
drop_expected = rfn.drop_fields(base, ["b"], usemask=False)

rename_result = fnp.recfunctions_rename_fields(base, {"a": "alpha", "c": "label"})
rename_expected = rfn.rename_fields(base, {"a": "alpha", "c": "label"})

append_data = np.array([7, 8], dtype=np.int16)
append_result = fnp.recfunctions_append_fields(base, "flag", append_data, usemask=False)
append_expected = rfn.append_fields(base, "flag", append_data, usemask=False)

match = (
    drop_result.dtype == drop_expected.dtype
    and np.array_equal(drop_result, drop_expected)
    and rename_result.dtype == rename_expected.dtype
    and np.array_equal(rename_result, rename_expected)
    and append_result.dtype == append_expected.dtype
    and np.array_equal(append_result, append_expected)
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "flat structured recfunctions should match numpy"
    );
    Ok(())
}

#[test]
fn recfunctions_merge_and_unstructured_to_structured() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib import recfunctions as rfn

left = np.array([(1,), (2,)], dtype=[("x", "i2")])
right = np.array([(1.25,), (2.5,)], dtype=[("y", "f4")])
merge_result = fnp.recfunctions_merge_arrays([left, right], usemask=False, flatten=False)
merge_expected = rfn.merge_arrays([left, right], usemask=False, flatten=False)

plain = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
target_dtype = np.dtype([("count", "i4"), ("ratio", "f8")])
struct_result = fnp.recfunctions_unstructured_to_structured(plain, dtype=target_dtype)
struct_expected = rfn.unstructured_to_structured(plain, dtype=target_dtype)

match = (
    merge_result.dtype == merge_expected.dtype
    and np.array_equal(merge_result, merge_expected)
    and struct_result.dtype == struct_expected.dtype
    and struct_result.shape == struct_expected.shape
    and np.array_equal(struct_result, struct_expected)
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "merge/unstructured structured recfunctions should match numpy"
    );
    Ok(())
}

#[test]
fn recfunctions_merge_arrays_multi_field_input_wraps_under_fn() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib import recfunctions as rfn

# Multi-field input must be wrapped under a synthesized fN; single-field
# input keeps its name flat at the top level.
multi = np.array([(1, 2), (3, 4)], dtype=[("a", "i2"), ("b", "i4")])
single = np.array([(0.5,), (1.5,)], dtype=[("c", "f4")])
got = fnp.recfunctions_merge_arrays([multi, single], usemask=False, flatten=False)
expected = rfn.merge_arrays([multi, single], usemask=False, flatten=False)
print(got.dtype == expected.dtype and np.array_equal(got, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "merge_arrays must wrap multi-field inputs under fN while leaving single-field names flat"
    );
    Ok(())
}

#[test]
fn recfunctions_merge_arrays_two_multi_field_inputs_get_distinct_fn_keys() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib import recfunctions as rfn

m1 = np.array([(1, 2)], dtype=[("a", "i2"), ("b", "i4")])
m2 = np.array([(3, 4)], dtype=[("p", "i2"), ("q", "i4")])
got = fnp.recfunctions_merge_arrays([m1, m2], usemask=False, flatten=False)
expected = rfn.merge_arrays([m1, m2], usemask=False, flatten=False)
print(got.dtype == expected.dtype and np.array_equal(got, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "merge_arrays must wrap each multi-field input under its own fN key"
    );
    Ok(())
}

#[test]
fn recfunctions_merge_arrays_three_unique_single_field_inputs_stay_flat() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib import recfunctions as rfn

a = np.array([(1,)], dtype=[("x", "i2")])
b = np.array([(2.5,)], dtype=[("y", "f4")])
c = np.array([(7,)], dtype=[("z", "i8")])
got = fnp.recfunctions_merge_arrays([a, b, c], usemask=False, flatten=False)
expected = rfn.merge_arrays([a, b, c], usemask=False, flatten=False)
print(got.dtype == expected.dtype and np.array_equal(got, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "merge_arrays of three unique single-field inputs must stay flat"
    );
    Ok(())
}

#[test]
fn recfunctions_merge_arrays_duplicate_field_name_raises() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib import recfunctions as rfn

a = np.array([(1,)], dtype=[("x", "i2")])
b = np.array([(2.5,)], dtype=[("x", "f4")])

def errored(fn):
    try:
        fn()
        return False
    except Exception:
        return True

ours_err = errored(lambda: fnp.recfunctions_merge_arrays([a, b], usemask=False, flatten=False))
numpy_err = errored(lambda: rfn.merge_arrays([a, b], usemask=False, flatten=False))
print(ours_err == numpy_err == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "merge_arrays must raise on duplicate top-level field names like numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn can_cast_implies_promotion() -> Result<(), String> {
    let script = fnp_script(
        r#"
# If can_cast is true for safe casting, promote_types should give target
can = fnp.can_cast('int32', 'float64', casting='safe')
promoted = fnp.promote_types('int32', 'float64')
print(can == True and promoted == np.dtype('float64'))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "can_cast implies compatible promotion"
    );
    Ok(())
}
