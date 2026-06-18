//! Conformance tests for numpy dtype utility functions against NumPy oracle.
//!
//! Tests broadcast_shapes, can_cast, common_type, promote_types.

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
fn dtype_utils_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def normalize(value):
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(value))
    if isinstance(value, np.dtype):
        return ("dtype", str(value))
    return ("other", type(value).__name__, str(value))

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    (
        "broadcast_shapes list shape",
        "broadcast_shapes",
        lambda: (([2, 1], (3,)), {}),
    ),
    (
        "broadcast_shapes scalar and empty shape",
        "broadcast_shapes",
        lambda: (((1,), (), (3, 1)), {}),
    ),
    (
        "broadcast_shapes incompatible error",
        "broadcast_shapes",
        lambda: (((2,), (3,)), {}),
    ),
    (
        "can_cast safe narrowing",
        "can_cast",
        lambda: ((np.dtype("int16"), "int8"), {"casting": "safe"}),
    ),
    (
        "can_cast same kind int to uint",
        "can_cast",
        lambda: (("int16", "uint16"), {"casting": "same_kind"}),
    ),
    (
        "can_cast invalid casting error",
        "can_cast",
        lambda: (("int16", "uint16"), {"casting": "not-a-casting"}),
    ),
    (
        "common_type mixed precision arrays",
        "common_type",
        lambda: ((np.array([1], dtype=np.int16), np.array([1.0], dtype=np.float32)), {}),
    ),
    (
        "common_type complex scalar array",
        "common_type",
        lambda: ((np.array([1 + 2j], dtype=np.complex64), np.array([1.0], dtype=np.float16)), {}),
    ),
    (
        "promote_types byte order",
        "promote_types",
        lambda: ((">i2", "<i4"), {}),
    ),
    ("promote_types object", "promote_types", lambda: (("object", "float64"), {})),
    ("promote_types invalid dtype error", "promote_types", lambda: (("not-a-dtype", "float64"), {})),
    ("issubdtype tuple input error", "issubdtype", lambda: (((np.int32, np.integer),), {})),
    ("isdtype invalid kind error", "isdtype", lambda: ((np.dtype("int32"), "not-a-kind"), {})),
]

ok = True
for label, name, factory in cases:
    args, kwargs = factory()
    actual = outcome(getattr(fnp, name), *args, **kwargs)
    args, kwargs = factory()
    expected = outcome(getattr(np, name), *args, **kwargs)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dtype utility Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
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

#[test]
fn recfunctions_structured_dtype_oracle_corpus() -> Result<(), String> {
    let script = fnp_script(
        r#"
import json
from numpy.lib import recfunctions as rfn
import numpy.ma as ma

base = np.array(
    [(1, 1.5, b"aa"), (2, 2.5, b"bb")],
    dtype=[("a", "i4"), ("b", "f8"), ("c", "S2")],
)
other = np.array([(10,), (20,)], dtype=[("d", "i2")])
nested = np.array(
    [((1, 2), 3.5), ((4, 5), 6.5)],
    dtype=[("pair", [("left", "i2"), ("right", "i2")]), ("score", "f4")],
)
byte_order = np.array(
    [(1, 2), (3, 4)],
    dtype=[("big", ">i2"), ("little", "<i4")],
)
object_records = np.array(
    [({"key": 1}, 4), ({"key": 2}, 5)],
    dtype=[("payload", "O"), ("rank", "i2")],
)
empty_records = np.array([], dtype=[("a", "i4"), ("b", "f8")])

def array_match(actual, expected):
    if actual.shape != expected.shape:
        return False
    if repr(actual.dtype) != repr(expected.dtype):
        return False
    if ma.isMaskedArray(actual) or ma.isMaskedArray(expected):
        return bool(ma.allequal(actual, expected)) and np.array_equal(
            ma.getmaskarray(actual),
            ma.getmaskarray(expected),
        )
    return bool(np.array_equal(actual, expected))

def value_match(actual, expected):
    if isinstance(expected, np.ndarray) or ma.isMaskedArray(expected):
        return array_match(actual, expected)
    return actual == expected

def exc_surface(fn):
    try:
        fn()
    except Exception as exc:
        return ("err", type(exc).__name__)
    return ("ok", None)

def compare_arrays(actual_fn, expected_fn):
    return array_match(actual_fn(), expected_fn())

def compare_values(actual_fn, expected_fn):
    return value_match(actual_fn(), expected_fn())

def compare_repr(actual_fn, expected_fn):
    return repr(actual_fn()) == repr(expected_fn())

def compare_errors(actual_fn, expected_fn):
    return exc_surface(actual_fn) == exc_surface(expected_fn)

corpus = [
    (
        "drop_single_flat_field",
        "recfunctions.drop_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_drop_fields(base, "b", usemask=False),
            lambda: rfn.drop_fields(base, "b", usemask=False),
        ),
    ),
    (
        "drop_multiple_flat_fields",
        "recfunctions.drop_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_drop_fields(base, ["a", "c"], usemask=False),
            lambda: rfn.drop_fields(base, ["a", "c"], usemask=False),
        ),
    ),
    (
        "drop_nested_field",
        "recfunctions.drop_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_drop_fields(nested, "pair", usemask=False),
            lambda: rfn.drop_fields(nested, "pair", usemask=False),
        ),
    ),
    (
        "drop_object_field",
        "recfunctions.drop_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_drop_fields(object_records, "payload", usemask=False),
            lambda: rfn.drop_fields(object_records, "payload", usemask=False),
        ),
    ),
    (
        "drop_empty_records",
        "recfunctions.drop_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_drop_fields(empty_records, "b", usemask=False),
            lambda: rfn.drop_fields(empty_records, "b", usemask=False),
        ),
    ),
    (
        "rename_flat_fields",
        "recfunctions.rename_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_rename_fields(base, {"a": "alpha", "c": "label"}),
            lambda: rfn.rename_fields(base, {"a": "alpha", "c": "label"}),
        ),
    ),
    (
        "rename_nested_top_level_field",
        "recfunctions.rename_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_rename_fields(nested, {"pair": "coords"}),
            lambda: rfn.rename_fields(nested, {"pair": "coords"}),
        ),
    ),
    (
        "append_single_field",
        "recfunctions.append_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_append_fields(base, "flag", np.array([7, 8]), usemask=False),
            lambda: rfn.append_fields(base, "flag", np.array([7, 8]), usemask=False),
        ),
    ),
    (
        "append_multiple_fields",
        "recfunctions.append_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_append_fields(
                base,
                ["flag", "code"],
                [np.array([7, 8]), np.array([b"x", b"y"], dtype="S1")],
                usemask=False,
            ),
            lambda: rfn.append_fields(
                base,
                ["flag", "code"],
                [np.array([7, 8]), np.array([b"x", b"y"], dtype="S1")],
                usemask=False,
            ),
        ),
    ),
    (
        "append_typed_float_field",
        "recfunctions.append_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_append_fields(
                base,
                "weight",
                np.array([7, 8], dtype="i2"),
                dtypes=np.float64,
                usemask=False,
            ),
            lambda: rfn.append_fields(
                base,
                "weight",
                np.array([7, 8], dtype="i2"),
                dtypes=np.float64,
                usemask=False,
            ),
        ),
    ),
    (
        "append_duplicate_field_error",
        "recfunctions.append_fields",
        lambda: compare_errors(
            lambda: fnp.recfunctions_append_fields(base, "a", np.array([7, 8]), usemask=False),
            lambda: rfn.append_fields(base, "a", np.array([7, 8]), usemask=False),
        ),
    ),
    (
        "merge_single_field_flatten_false_regression",
        "recfunctions.merge_arrays",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_merge_arrays([other], usemask=False, flatten=False),
            lambda: rfn.merge_arrays([other], usemask=False, flatten=False),
        ),
    ),
    (
        "merge_flat_single_fields_flatten_false",
        "recfunctions.merge_arrays",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_merge_arrays([base[["a"]], other], usemask=False, flatten=False),
            lambda: rfn.merge_arrays([base[["a"]], other], usemask=False, flatten=False),
        ),
    ),
    (
        "merge_multi_and_single_flatten_false",
        "recfunctions.merge_arrays",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_merge_arrays([base[["a", "b"]], other], usemask=False, flatten=False),
            lambda: rfn.merge_arrays([base[["a", "b"]], other], usemask=False, flatten=False),
        ),
    ),
    (
        "merge_multi_fields_flatten_true",
        "recfunctions.merge_arrays",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_merge_arrays([base[["a", "b"]], other], usemask=False, flatten=True),
            lambda: rfn.merge_arrays([base[["a", "b"]], other], usemask=False, flatten=True),
        ),
    ),
    (
        "merge_duplicate_field_error",
        "recfunctions.merge_arrays",
        lambda: compare_errors(
            lambda: fnp.recfunctions_merge_arrays([base[["a"]], base[["a"]]], usemask=False, flatten=False),
            lambda: rfn.merge_arrays([base[["a"]], base[["a"]]], usemask=False, flatten=False),
        ),
    ),
    (
        "unstructured_to_structured_explicit_dtype",
        "recfunctions.unstructured_to_structured",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_unstructured_to_structured(
                np.arange(6, dtype="f8").reshape(2, 3),
                dtype=np.dtype([("x", "i4"), ("y", "f8"), ("z", "f8")]),
            ),
            lambda: rfn.unstructured_to_structured(
                np.arange(6, dtype="f8").reshape(2, 3),
                dtype=np.dtype([("x", "i4"), ("y", "f8"), ("z", "f8")]),
            ),
        ),
    ),
    (
        "unstructured_to_structured_names_auto_dtype",
        "recfunctions.unstructured_to_structured",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_unstructured_to_structured(
                np.arange(4, dtype="i2").reshape(2, 2),
                names=["left", "right"],
            ),
            lambda: rfn.unstructured_to_structured(
                np.arange(4, dtype="i2").reshape(2, 2),
                names=["left", "right"],
            ),
        ),
    ),
    (
        "unstructured_to_structured_invalid_field_count_error",
        "recfunctions.unstructured_to_structured",
        lambda: compare_errors(
            lambda: fnp.recfunctions_unstructured_to_structured(
                np.arange(6, dtype="i4").reshape(2, 3),
                dtype=np.dtype([("x", "i4"), ("y", "i4")]),
            ),
            lambda: rfn.unstructured_to_structured(
                np.arange(6, dtype="i4").reshape(2, 3),
                dtype=np.dtype([("x", "i4"), ("y", "i4")]),
            ),
        ),
    ),
    (
        "structured_to_unstructured_flat",
        "recfunctions.structured_to_unstructured",
        lambda: compare_arrays(
            lambda: fnp.structured_to_unstructured(base[["a", "b"]]),
            lambda: rfn.structured_to_unstructured(base[["a", "b"]]),
        ),
    ),
    (
        "structured_to_unstructured_nested",
        "recfunctions.structured_to_unstructured",
        lambda: compare_arrays(
            lambda: fnp.structured_to_unstructured(nested),
            lambda: rfn.structured_to_unstructured(nested),
        ),
    ),
    (
        "repack_fields_byte_order",
        "recfunctions.repack_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_repack_fields(byte_order, align=False),
            lambda: rfn.repack_fields(byte_order, align=False),
        ),
    ),
    (
        "require_fields_reorders_and_fills",
        "recfunctions.require_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_require_fields(
                base,
                np.dtype([("c", "S2"), ("missing", "i4"), ("a", "i4")]),
            ),
            lambda: rfn.require_fields(
                base,
                np.dtype([("c", "S2"), ("missing", "i4"), ("a", "i4")]),
            ),
        ),
    ),
    (
        "assign_fields_by_name_nested",
        "recfunctions.assign_fields_by_name",
        lambda: compare_arrays(
            lambda: (lambda dst: (fnp.recfunctions_assign_fields_by_name(dst, nested), dst)[1])(
                np.zeros(nested.shape, dtype=nested.dtype)
            ),
            lambda: (lambda dst: (rfn.assign_fields_by_name(dst, nested), dst)[1])(
                np.zeros(nested.shape, dtype=nested.dtype)
            ),
        ),
    ),
    (
        "recursive_fill_fields_nested",
        "recfunctions.recursive_fill_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_recursive_fill_fields(
                nested,
                np.zeros(nested.shape, dtype=nested.dtype),
            ),
            lambda: rfn.recursive_fill_fields(
                nested,
                np.zeros(nested.shape, dtype=nested.dtype),
            ),
        ),
    ),
    (
        "join_by_key",
        "recfunctions.join_by",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_join_by(
                "key",
                np.array([(1, 10), (2, 20)], dtype=[("key", "i4"), ("left", "i4")]),
                np.array([(1, 30), (2, 40)], dtype=[("key", "i4"), ("right", "i4")]),
                usemask=False,
            ),
            lambda: rfn.join_by(
                "key",
                np.array([(1, 10), (2, 20)], dtype=[("key", "i4"), ("left", "i4")]),
                np.array([(1, 30), (2, 40)], dtype=[("key", "i4"), ("right", "i4")]),
                usemask=False,
            ),
        ),
    ),
    (
        "find_duplicates_masked_key",
        "recfunctions.find_duplicates",
        lambda: compare_repr(
            lambda: fnp.recfunctions_find_duplicates(
                ma.array([(1, "a"), (1, "b"), (2, "c")], dtype=[("key", "i4"), ("v", "U1")]),
                key="key",
                return_index=True,
            ),
            lambda: rfn.find_duplicates(
                ma.array([(1, "a"), (1, "b"), (2, "c")], dtype=[("key", "i4"), ("v", "U1")]),
                key="key",
                return_index=True,
            ),
        ),
    ),
    (
        "get_names_nested",
        "recfunctions.get_names",
        lambda: compare_values(
            lambda: fnp.recfunctions_get_names(nested.dtype),
            lambda: rfn.get_names(nested.dtype),
        ),
    ),
    (
        "get_names_flat_nested",
        "recfunctions.get_names_flat",
        lambda: compare_values(
            lambda: fnp.recfunctions_get_names_flat(nested.dtype),
            lambda: rfn.get_names_flat(nested.dtype),
        ),
    ),
    (
        "flatten_descr_nested",
        "recfunctions.flatten_descr",
        lambda: compare_values(
            lambda: fnp.recfunctions_flatten_descr(nested.dtype),
            lambda: rfn.flatten_descr(nested.dtype),
        ),
    ),
    (
        "get_fieldstructure_nested",
        "recfunctions.get_fieldstructure",
        lambda: compare_values(
            lambda: fnp.recfunctions_get_fieldstructure(nested.dtype),
            lambda: rfn.get_fieldstructure(nested.dtype),
        ),
    ),
    (
        "apply_along_fields_sum",
        "recfunctions.apply_along_fields",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_apply_along_fields(
                np.sum,
                np.array([(1, 2, 3), (4, 5, 6)], dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")]),
            ),
            lambda: rfn.apply_along_fields(
                np.sum,
                np.array([(1, 2, 3), (4, 5, 6)], dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")]),
            ),
        ),
    ),
    (
        "stack_arrays_autoconvert",
        "recfunctions.stack_arrays",
        lambda: compare_arrays(
            lambda: fnp.recfunctions_stack_arrays(
                [base[["a"]], other],
                usemask=False,
                autoconvert=True,
            ),
            lambda: rfn.stack_arrays(
                [base[["a"]], other],
                usemask=False,
                autoconvert=True,
            ),
        ),
    ),
]

failures = []
for case_id, surface, check in corpus:
    try:
        if not check():
            failures.append({"id": case_id, "surface": surface, "reason": "mismatch"})
    except Exception as exc:
        failures.append({
            "id": case_id,
            "surface": surface,
            "reason": type(exc).__name__,
            "message": str(exc),
        })

print(json.dumps(failures, sort_keys=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "[]",
        "structured dtype/recfunctions oracle corpus mismatch"
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
