//! Conformance tests for numpy unary operations against NumPy oracle.
//!
//! Tests positive, negative, reciprocal functions.

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

#[test]
fn positive_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
result = fnp.positive(x)
expected = np.positive(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "positive basic should match numpy");
    Ok(())
}

#[test]
fn negative_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
result = fnp.negative(x)
expected = np.negative(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative basic should match numpy");
    Ok(())
}

#[test]
fn sign_unary_out_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def out_outcome(module, name, positional=False, bad_shape=False):
    fn = getattr(module, name)
    try:
        x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float64)
        out = np.empty((2,), dtype=np.float64) if bad_shape else np.empty(4, dtype=np.float64)
        if positional:
            result = fn(x, out)
        else:
            result = fn(x, out=out)
        return ("ok", result is out, out.dtype.str, tuple(out.shape), out.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc).splitlines()[0])

cases = [
    ("positive keyword out", "positive", False, False),
    ("negative keyword out", "negative", False, False),
    ("positive positional out", "positive", True, False),
    ("negative positional out", "negative", True, False),
    ("positive bad out shape", "positive", False, True),
    ("negative bad out shape", "negative", False, True),
]

ok = True
for label, name, positional, bad_shape in cases:
    actual = out_outcome(fnp, name, positional, bad_shape)
    expected = out_outcome(np, name, positional, bad_shape)
    if actual[0] != expected[0] or actual[1] != expected[1]:
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
        "positive/negative out keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int32_unary_wraps_and_matches_numpy_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib

cases = [
    np.array([-2147483648, -46341, -129, -2, -1, 0, 1, 2, 127, 46341, 2147483647], dtype=np.int32),
    np.arange(-32, 32, dtype=np.int32).reshape(8, 8),
    np.array([], dtype=np.int32),
]
ops = [("positive", fnp.positive, np.positive), ("negative", fnp.negative, np.negative), ("abs", fnp.abs, np.abs), ("square", fnp.square, np.square)]

def digest(which):
    chunks = []
    for name, fnp_op, np_op in ops:
        op = fnp_op if which == "fnp" else np_op
        for x in cases:
            out = op(x)
            chunks.append(name.encode())
            chunks.append(b"|")
            chunks.append(str(out.dtype).encode())
            chunks.append(b"|")
            chunks.append(np.asarray(out.shape, dtype=np.int64).tobytes())
            chunks.append(b"|")
            chunks.append(np.ascontiguousarray(out).view(np.uint8).tobytes())
            chunks.append(b";")
    return hashlib.sha256(b"".join(chunks)).hexdigest()

wrap_input = np.array([-2147483648, -46341, 46341], dtype=np.int32)
wraps = (
    np.array_equal(fnp.square(wrap_input), np.square(wrap_input))
    and fnp.square(wrap_input).dtype == np.dtype("int32")
    and fnp.abs(wrap_input[:1])[0] == np.abs(wrap_input[:1])[0]
    and fnp.negative(wrap_input[:1])[0] == np.negative(wrap_input[:1])[0]
)

ours = digest("fnp")
theirs = digest("numpy")
print(ours)
print(theirs)
print(ours == theirs)
print(wraps)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let lines: Vec<&str> = result.lines().collect();
    let expected_sha = "f8315dacdd43a78882f63bd17ad733375ba5fc3c5fa20459b25161c092ba182e";
    assert_eq!(
        lines.get(2).copied(),
        Some("True"),
        "int32 unary raw output hash must match numpy: {result}"
    );
    assert_eq!(
        lines.get(3).copied(),
        Some("True"),
        "int32 unary edge cases must preserve numpy wrapping: {result}"
    );
    assert_eq!(
        lines.first().copied(),
        Some(expected_sha),
        "int32 unary fnp hash changed: {result}"
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(expected_sha),
        "int32 unary numpy golden hash changed: {result}"
    );
    Ok(())
}

#[test]
fn narrow_integer_unary_wraps_and_matches_numpy_golden_sha256() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib

cases = [
    np.array([-128, -17, -2, -1, 0, 1, 2, 11, 127], dtype=np.int8),
    np.array([-32768, -257, -129, -2, -1, 0, 1, 2, 129, 257, 32767], dtype=np.int16),
    np.arange(-18, 18, dtype=np.int16).reshape(6, 6),
    np.array([0, 1, 2, 15, 16, 127, 128, 255], dtype=np.uint8),
    np.arange(64, dtype=np.uint8).reshape(8, 8),
    np.array([0, 1, 2, 255, 256, 32767, 32768, 65535], dtype=np.uint16),
    np.array([0, 1, 2, 65535, 65536, 2147483648, 4294967295], dtype=np.uint32),
    np.array([0, 1, 2, 4294967295, 4294967296, 9223372036854775808, 18446744073709551615], dtype=np.uint64),
    np.array([], dtype=np.int16),
    np.array([], dtype=np.uint32),
]
ops = [("positive", fnp.positive, np.positive), ("negative", fnp.negative, np.negative), ("abs", fnp.abs, np.abs), ("square", fnp.square, np.square)]

def digest(which):
    chunks = []
    for name, fnp_op, np_op in ops:
        op = fnp_op if which == "fnp" else np_op
        for x in cases:
            out = op(x)
            chunks.append(name.encode())
            chunks.append(b"|")
            chunks.append(str(x.dtype).encode())
            chunks.append(b"|")
            chunks.append(str(out.dtype).encode())
            chunks.append(b"|")
            chunks.append(np.asarray(out.shape, dtype=np.int64).tobytes())
            chunks.append(b"|")
            chunks.append(np.ascontiguousarray(out).view(np.uint8).tobytes())
            chunks.append(b";")
    return hashlib.sha256(b"".join(chunks)).hexdigest()

edge_checks = (
    fnp.square(np.array([-128], dtype=np.int8))[0] == np.square(np.array([-128], dtype=np.int8))[0]
    and fnp.abs(np.array([-32768], dtype=np.int16))[0] == np.abs(np.array([-32768], dtype=np.int16))[0]
    and fnp.negative(np.array([1], dtype=np.uint8))[0] == np.negative(np.array([1], dtype=np.uint8))[0]
    and fnp.square(np.array([18446744073709551615], dtype=np.uint64))[0] == np.square(np.array([18446744073709551615], dtype=np.uint64))[0]
)

ours = digest("fnp")
theirs = digest("numpy")
print(ours)
print(theirs)
print(ours == theirs)
print(edge_checks)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let lines: Vec<&str> = result.lines().collect();
    let expected_sha = "5fac5a6179d813fec16137caa57737b5761da194a9ddd9b014b1c12785895f09";
    assert_eq!(
        lines.get(2).copied(),
        Some("True"),
        "narrow integer unary raw output hash must match numpy: {result}"
    );
    assert_eq!(
        lines.get(3).copied(),
        Some("True"),
        "narrow integer unary edge cases must preserve numpy wrapping: {result}"
    );
    assert_eq!(
        lines.first().copied(),
        Some(expected_sha),
        "narrow integer unary fnp hash changed: {result}"
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(expected_sha),
        "narrow integer unary numpy golden hash changed: {result}"
    );
    Ok(())
}

#[test]
fn reciprocal_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 4.0, 5.0])
result = fnp.reciprocal(x)
expected = np.reciprocal(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "reciprocal basic should match numpy");
    Ok(())
}

#[test]
fn negative_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.negative(x)
np_result = np.negative(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "negative scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn reciprocal_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.reciprocal(x)
np_result = np.reciprocal(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "reciprocal scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn negative_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex128)
fnp_result = fnp.negative(z)
np_result = np.negative(z)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative complex should match numpy");
    Ok(())
}

#[test]
fn positive_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, -3+4j, 5-6j], dtype=np.complex128)
fnp_result = fnp.positive(z)
np_result = np.positive(z)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "positive complex should match numpy");
    Ok(())
}

#[test]
fn reciprocal_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2-1j, 3+0j], dtype=np.complex128)
fnp_result = fnp.reciprocal(z)
np_result = np.reciprocal(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "reciprocal complex should match numpy"
    );
    Ok(())
}

#[test]
fn positive_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.positive(x)
np_result = np.positive(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "positive special values should match numpy"
    );
    Ok(())
}

#[test]
fn negative_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.negative(x)
np_result = np.negative(x)
# Check values and sign bits for zeros
value_match = np.allclose(fnp_result, np_result, equal_nan=True)
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "negative special values should match numpy"
    );
    Ok(())
}

#[test]
fn negative_zero_sign() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
fnp_result = fnp.negative(x)
np_result = np.negative(x)
# negative(0.0) should be -0.0, negative(-0.0) should be 0.0
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
print(sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "negative zero sign should match numpy"
    );
    Ok(())
}

#[test]
fn positive_preserves_sign() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0, 1.0, -1.0])
fnp_result = fnp.positive(x)
np_result = np.positive(x)
# positive should preserve sign
sign_match = np.array_equal(np.signbit(fnp_result), np.signbit(np_result))
value_match = np.array_equal(fnp_result, np_result)
print(sign_match and value_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "positive should preserve sign");
    Ok(())
}
