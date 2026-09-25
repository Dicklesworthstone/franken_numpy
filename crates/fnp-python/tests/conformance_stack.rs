//! Conformance tests for numpy.stack, vstack, hstack, dstack against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy.

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

mod support;
use support::fnp_script;

// ─────────────────────────────────────────────────────────────────────────────
// stack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stack_helpers_python_container_and_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def clean(value):
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def normalize(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), clean(array.tolist()))

def outcome(call_fn, *args, **kwargs):
    try:
        return ("ok", normalize(call_fn(*args, **kwargs)))
    except Exception as exc:
        return ("err", type(exc).__name__)

cases = [
    ("stack tuple of Python lists", "stack", lambda: ((((1, 2), (3, 4)),), {})),
    (
        "stack axis minus one Python lists",
        "stack",
        lambda: (([[1, 2, 3], [4, 5, 6]],), {"axis": -1}),
    ),
    (
        "stack dtype casting unsafe",
        "stack",
        lambda: (([np.array([1.25, 2.75]), np.array([3.5, 4.5])],), {"dtype": np.int64, "casting": "unsafe"}),
    ),
    ("vstack Python list rows", "vstack", lambda: ((((1, 2, 3), (4, 5, 6)),), {})),
    (
        "hstack mixed list ndarray one dimensional",
        "hstack",
        lambda: (([[1, 2], np.array([3, 4], dtype=np.int16)],), {}),
    ),
    ("dstack one dimensional lists", "dstack", lambda: ((((1, 2, 3), (4, 5, 6)),), {})),
    (
        "column_stack one dimensional lists",
        "column_stack",
        lambda: ((((1, 2, 3), (4, 5, 6)),), {}),
    ),
    # dstack, column_stack and stack(axis=1) all enter the SAME native
    # column-interleave probe, which used to read `.dtype` off the first item
    # before checking it was an ndarray. The two cases above covered two of those
    # three doors; this one covers the third (deadlock-audit-3fvvr).
    (
        "stack axis one Python lists",
        "stack",
        lambda: (([[1, 2, 3], [4, 5, 6]],), {"axis": 1}),
    ),
    # Mixed sequences, both orders: a fix that only type-checks items[0] passes
    # the all-lists cases above and still raises on these.
    (
        "dstack mixed ndarray then list",
        "dstack",
        lambda: (([np.array([1, 2, 3]), [4, 5, 6]],), {}),
    ),
    (
        "column_stack mixed list then ndarray",
        "column_stack",
        lambda: (([[1, 2, 3], np.array([4, 5, 6])],), {}),
    ),
    ("stack empty sequence error", "stack", lambda: (([],), {})),
    (
        "stack shape mismatch error",
        "stack",
        lambda: (([np.array([1, 2]), np.array([[3, 4]])],), {}),
    ),
    (
        "hstack shape mismatch error",
        "hstack",
        lambda: (([np.ones((2, 2)), np.ones((3, 2))],), {}),
    ),
    (
        "vstack casting error",
        "vstack",
        lambda: (([np.array([1.5]), np.array([2.5])],), {"dtype": np.int64, "casting": "safe"}),
    ),
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

def stack_out_contract(stack_fn):
    out = np.empty((2, 2), dtype=np.float64)
    result = stack_fn([np.array([1.5, 2.5]), np.array([3.5, 4.5])], out=out)
    return (normalize(result), normalize(out), result is out)

actual_out = stack_out_contract(fnp.stack)
expected_out = stack_out_contract(np.stack)
if actual_out != expected_out:
    print("stack out contract")
    print(actual_out)
    print(expected_out)
    ok = False

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack helper Python-container and keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn stack_1d_arrays_default_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b])
expected = np.stack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack 1d default axis should match numpy"
    );
    Ok(())
}

#[test]
fn stack_1d_arrays_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b], axis=0)
expected = np.stack([a, b], axis=0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack 1d axis=0 should match numpy");
    Ok(())
}

#[test]
fn stack_1d_arrays_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b], axis=1)
expected = np.stack([a, b], axis=1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack 1d axis=1 should match numpy");
    Ok(())
}

#[test]
fn stack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.stack([a, b])
expected = np.stack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack 2d should match numpy");
    Ok(())
}

#[test]
fn stack_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b], axis=-1)
expected = np.stack([a, b], axis=-1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack negative axis should match numpy"
    );
    Ok(())
}

#[test]
fn stack_float_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
b = np.array([4.5, 5.5, 6.5])
result = fnp.stack([a, b])
expected = np.stack([a, b])
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack float arrays should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vstack_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.vstack([a, b])
expected = np.vstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vstack 1d arrays should match numpy");
    Ok(())
}

#[test]
fn vstack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.vstack([a, b])
expected = np.vstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vstack 2d arrays should match numpy");
    Ok(())
}

#[test]
fn vstack_mixed_dimensions() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([[4, 5, 6]])
result = fnp.vstack([a, b])
expected = np.vstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vstack mixed dimensions should match numpy"
    );
    Ok(())
}

#[test]
fn vstack_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.vstack([a])
expected = np.vstack([a])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vstack single array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hstack_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.hstack([a, b])
expected = np.hstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hstack 1d arrays should match numpy");
    Ok(())
}

#[test]
fn hstack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.hstack([a, b])
expected = np.hstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hstack 2d arrays should match numpy");
    Ok(())
}

#[test]
fn hstack_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.hstack([a])
expected = np.hstack([a])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hstack single array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dstack_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.dstack([a, b])
expected = np.dstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dstack 1d arrays should match numpy");
    Ok(())
}

#[test]
fn dstack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.dstack([a, b])
expected = np.dstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dstack 2d arrays should match numpy");
    Ok(())
}

#[test]
fn dstack_3d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.ones((2, 3, 4))
b = np.ones((2, 3, 4)) * 2
result = fnp.dstack([a, b])
expected = np.dstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dstack 3d arrays should match numpy");
    Ok(())
}

#[test]
fn dstack_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.dstack([a])
expected = np.dstack([a])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dstack single array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vstack_row_stack_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vstack_result = fnp.vstack([a, b])
row_stack_result = fnp.row_stack([a, b])
print(np.array_equal(vstack_result, row_stack_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vstack and row_stack should be equivalent"
    );
    Ok(())
}

#[test]
fn hstack_column_stack_1d_difference() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
hstack_result = fnp.hstack([a, b])
column_stack_result = fnp.column_stack([a, b])
hstack_expected = np.hstack([a, b])
column_stack_expected = np.column_stack([a, b])
print(np.array_equal(hstack_result, hstack_expected) and np.array_equal(column_stack_result, column_stack_expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hstack and column_stack 1d should match numpy"
    );
    Ok(())
}

#[test]
fn stack_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.stack([a, b])
np_result = np.stack([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack complex should match numpy");
    Ok(())
}

#[test]
fn hstack_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.hstack([a, b])
np_result = np.hstack([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hstack complex should match numpy");
    Ok(())
}

#[test]
fn vstack_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.vstack([a, b])
np_result = np.vstack([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vstack complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else if stderr.contains("AxisError") {
            "AxisError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn stack_shape_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(8).reshape(2, 4)
fnp.stack([a, b])
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(8).reshape(2, 4)
np.stack([a, b])
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "stack with shape mismatch should raise same error as numpy"
    );
}

#[test]
fn hstack_shape_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(9).reshape(3, 3)
fnp.hstack([a, b])
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(9).reshape(3, 3)
np.hstack([a, b])
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "hstack with incompatible first dimensions should raise same error as numpy"
    );
}

/// Locks the zero-copy 2-D vstack fast path (which routes through
/// `try_zerocopy_f64_concatenate_axis0`) to bit-exact parity with numpy. Stacked
/// rows are copied verbatim, so parity must hold at the IEEE-754 bit level.
/// Compares the sha256 of raw output bytes for 2-D float64 inputs and extremes.
#[test]
fn vstack_2d_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
chunks.append(np.asarray(mod.vstack([rng.standard_normal((100, 50)), rng.standard_normal((200, 50))])).tobytes())
chunks.append(np.asarray(mod.vstack([rng.standard_normal((30, 40)), rng.standard_normal((5, 40)), rng.standard_normal((17, 40))])).tobytes())
xe = np.array([[0.0, -0.0, np.inf], [-np.inf, np.nan, 1e308]], dtype=np.float64)
chunks.append(np.asarray(mod.vstack([xe, xe * 2])).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy 2-D vstack must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// Locks the zero-copy hstack fast path (which routes through
/// `try_zerocopy_f64_concatenate` — axis 0 for 1-D inputs, axis 1 for ndim>=2) to
/// bit-exact parity with numpy. Covers 1-D, 2-D, and 3-D f64 inputs and extremes.
#[test]
fn hstack_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
chunks.append(np.asarray(mod.hstack([rng.standard_normal(1000), rng.standard_normal(500)])).tobytes())
chunks.append(np.asarray(mod.hstack([rng.standard_normal((100, 50)), rng.standard_normal((100, 30))])).tobytes())
chunks.append(np.asarray(mod.hstack([rng.standard_normal((2, 30, 4)), rng.standard_normal((2, 5, 4))])).tobytes())
xe = np.array([[0.0, -0.0, np.inf], [-np.inf, np.nan, 1e308]], dtype=np.float64)
chunks.append(np.asarray(mod.hstack([xe, xe * 2])).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy hstack must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    assert_eq!(
        fnp_hash, "f04ba8386bb78b51ad88f8587002623eb251cca7645d6847a86aadcf35eac581",
        "golden sha256 of hstack raw output bytes"
    );
    Ok(())
}

/// numpy's AxisConcatenator (`r_` / `c_`) resolves `result_type(*arrays, *python_scalars)`
/// with the Python scalars WEAK (NEP 50): `r_[int8_array, 127]` is int8,
/// `r_[float32_array, 0.1]` is float32, and `r_[int8_array, 255]` raises OverflowError
/// (numpy's own test_nep50_with_axisconcatenator). The native concatenate widened all
/// three to int64/float64. 8 of the 24 cells failed before the fix (numpy 2.4.3); 0 fail
/// after, on numpy 2.4.3 and 2.3.5.
///
/// Controls: scalars alone, slices, complex steps, string directives, matrix mode, `c_`, and
/// `concatenate`/`hstack`/`append` (which do not take this route) keep matching.
#[test]
fn axis_concatenator_keeps_python_scalars_weak() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings

def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = call()
            if isinstance(r, np.matrix):
                got = ("matrix", r.dtype.str, r.shape, r.tobytes())
            else:
                a = np.asarray(r)
                data = repr(a.tolist()) if a.dtype == object else a.tobytes()
                got = ("ok", type(r).__name__, a.dtype.str, a.shape, data)
        except Exception as ex:
            got = (type(ex).__name__, str(ex))
    return got + (sorted({w.category.__name__ for w in caught}),)

i8 = np.arange(5, dtype=np.int8)
u8 = np.arange(3, dtype=np.uint8)
f32 = np.arange(3, dtype=np.float32)
cases = {
    "r_ int8 + 255": lambda m: m.r_[i8, 255],
    "r_ int8 + 127": lambda m: m.r_[i8, 127],
    "r_ int8 + -129": lambda m: m.r_[i8, -129],
    "r_ uint8 + -1": lambda m: m.r_[u8, -1],
    "r_ uint8 + 3.5": lambda m: m.r_[u8, 3.5],
    "r_ f32 + 1e40": lambda m: m.r_[f32, 1e40],
    "r_ f32 + 0.1": lambda m: m.r_[f32, 0.1],
    "r_ int8 + 2**70": lambda m: m.r_[i8, 2**70],
    "r_ int8 + True": lambda m: m.r_[i8, True],
    "r_ int8 + 1j": lambda m: m.r_[i8, 1j],
    "r_ scalars": lambda m: m.r_[1, 2, 3.5],
    "r_ int8 scalar + 300": lambda m: m.r_[np.int8(1), 300],
    "r_ slice": lambda m: m.r_[0:5:2, i8],
    "r_ complex step": lambda m: m.r_[0:1:5j],
    "r_ string axis": lambda m: m.r_["0,2", [1, 2], [3, 4]],
    "r_ matrix": lambda m: m.r_["r", [1, 2], [3, 4]],
    "c_ int8 + 255": lambda m: m.c_[i8, np.full(5, 255)],
    "c_ int8 + scalar": lambda m: m.c_[i8, i8],
    "concatenate int8 + [255]": lambda m: m.concatenate([i8, [255]]),
    "concatenate int8 + np.array(255)": lambda m: m.concatenate([i8, np.array([255])]),
    "hstack int8 + [255]": lambda m: m.hstack([i8, [255]]),
    "append int8 255": lambda m: m.append(i8, 255),
    "r_ bool": lambda m: m.r_[np.array([True]), 2],
    "r_ empty": lambda m: m.r_[()],
}
bad = []
for name, case in cases.items():
    ours, theirs = outcome(lambda: case(fnp)), outcome(lambda: case(np))
    if ours != theirs:
        bad.append(f"{name}: fnp={str(ours)[:170]} numpy={str(theirs)[:170]}")
print(len(cases), bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "24 []",
        "r_/c_ must keep Python scalars weak as numpy's AxisConcatenator does: {result}"
    );
    Ok(())
}
