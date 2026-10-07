//! Conformance tests for numpy basic binary operations against NumPy oracle.
//!
//! Tests add, subtract, multiply, divide functions.

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

#[test]
fn add_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([5, 6, 7, 8])
result = fnp.add(x1, x2)
expected = np.add(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add basic should match numpy");
    Ok(())
}

#[test]
fn subtract_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([5, 6, 7, 8])
x2 = np.array([1, 2, 3, 4])
result = fnp.subtract(x1, x2)
expected = np.subtract(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "subtract basic should match numpy");
    Ok(())
}

#[test]
fn multiply_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 3, 4, 5])
result = fnp.multiply(x1, x2)
expected = np.multiply(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multiply basic should match numpy");
    Ok(())
}

#[test]
fn divide_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([8.0, 9.0, 10.0, 12.0])
x2 = np.array([2.0, 3.0, 5.0, 4.0])
result = fnp.divide(x1, x2)
expected = np.divide(x1, x2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide basic should match numpy");
    Ok(())
}

#[test]
fn add_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.add(x1, x2)
np_result = np.add(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "add scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn subtract_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(5.0)
x2 = np.float64(3.0)
fnp_result = fnp.subtract(x1, x2)
np_result = np.subtract(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "subtract scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn multiply_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(3.0)
x2 = np.float64(5.0)
fnp_result = fnp.multiply(x1, x2)
np_result = np.multiply(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "multiply scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn divide_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(10.0)
x2 = np.float64(2.0)
fnp_result = fnp.divide(x1, x2)
np_result = np.divide(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "divide scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+2j, 3+4j], dtype=np.complex128)
z2 = np.array([5+6j, 7+8j], dtype=np.complex128)
fnp_result = fnp.add(z1, z2)
np_result = np.add(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add complex should match numpy");
    Ok(())
}

#[test]
fn subtract_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([5+6j, 7+8j], dtype=np.complex128)
z2 = np.array([1+2j, 3+4j], dtype=np.complex128)
fnp_result = fnp.subtract(z1, z2)
np_result = np.subtract(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "subtract complex should match numpy");
    Ok(())
}

#[test]
fn multiply_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([1+2j, 3+4j], dtype=np.complex128)
z2 = np.array([5+6j, 7+8j], dtype=np.complex128)
fnp_result = fnp.multiply(z1, z2)
np_result = np.multiply(z1, z2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "multiply complex should match numpy");
    Ok(())
}

#[test]
fn divide_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([5+10j, 15+20j], dtype=np.complex128)
z2 = np.array([1+2j, 3+4j], dtype=np.complex128)
fnp_result = fnp.divide(z1, z2)
np_result = np.divide(z1, z2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide complex should match numpy");
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
        if stderr.contains("ValueError") || stderr.contains("broadcast") || stderr.contains("shape")
        {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn add_broadcast_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(4).reshape(2, 2)
fnp.add(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(4).reshape(2, 2)
np.add(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "add with incompatible broadcast shapes should raise same error as numpy"
    );
}

#[test]
fn multiply_broadcast_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(8).reshape(4, 2)
fnp.multiply(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(8).reshape(4, 2)
np.multiply(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "multiply with incompatible broadcast shapes should raise same error as numpy"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests: NaN, Inf, signed zero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn add_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0, np.nan])
x2 = np.array([4.0, 5.0, np.nan, np.nan])
fnp_result = fnp.add(x1, x2)
np_result = np.add(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "add nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn subtract_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0, np.nan])
x2 = np.array([4.0, 5.0, np.nan, np.nan])
fnp_result = fnp.subtract(x1, x2)
np_result = np.subtract(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "subtract nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn multiply_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0, np.nan])
x2 = np.array([4.0, 5.0, np.nan, np.nan])
fnp_result = fnp.multiply(x1, x2)
np_result = np.multiply(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn divide_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([1.0, np.nan, 3.0, np.nan])
x2 = np.array([4.0, 5.0, np.nan, np.nan])
fnp_result = fnp.divide(x1, x2)
np_result = np.divide(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divide nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn add_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, np.inf, -np.inf, np.inf])
x2 = np.array([np.inf, np.inf, np.inf, -np.inf])
fnp_result = fnp.add(x1, x2)
np_result = np.add(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "add inf handling should match numpy");
    Ok(())
}

#[test]
fn subtract_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, np.inf, -np.inf, 1.0])
x2 = np.array([1.0, np.inf, -np.inf, np.inf])
fnp_result = fnp.subtract(x1, x2)
np_result = np.subtract(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "subtract inf handling should match numpy"
    );
    Ok(())
}

#[test]
fn multiply_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([np.inf, np.inf, -np.inf, 0.0])
x2 = np.array([2.0, -2.0, -np.inf, np.inf])
fnp_result = fnp.multiply(x1, x2)
np_result = np.multiply(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply inf handling should match numpy"
    );
    Ok(())
}

#[test]
fn divide_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([1.0, np.inf, -np.inf, np.inf])
x2 = np.array([0.0, 2.0, -np.inf, np.inf])
fnp_result = fnp.divide(x1, x2)
np_result = np.divide(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divide inf handling should match numpy"
    );
    Ok(())
}

#[test]
fn divide_by_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([1.0, -1.0, 0.0])
x2 = np.array([0.0, 0.0, 0.0])
fnp_result = fnp.divide(x1, x2)
np_result = np.divide(x1, x2)
print(np.allclose(fnp_result, np_result, equal_nan=True) or
      all((np.isinf(f) == np.isinf(n) and np.isnan(f) == np.isnan(n))
          for f, n in zip(fnp_result.flat, np_result.flat)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divide by zero should match numpy");
    Ok(())
}

#[test]
fn add_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754: 0.0 + (-0.0) = 0.0, (-0.0) + 0.0 = 0.0, (-0.0) + (-0.0) = -0.0
tests = [
    (0.0, 0.0),
    (0.0, -0.0),
    (-0.0, 0.0),
    (-0.0, -0.0),
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.add(np.float64(x1), np.float64(x2))
    np_result = np.add(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: add({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "add signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn subtract_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754: 0.0 - 0.0 = 0.0, 0.0 - (-0.0) = 0.0, (-0.0) - 0.0 = -0.0, (-0.0) - (-0.0) = 0.0
tests = [
    (0.0, 0.0),
    (0.0, -0.0),
    (-0.0, 0.0),
    (-0.0, -0.0),
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.subtract(np.float64(x1), np.float64(x2))
    np_result = np.subtract(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: subtract({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "subtract signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn multiply_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754: sign(product) = sign(x1) XOR sign(x2)
tests = [
    (0.0, 0.0),    # +0 * +0 = +0
    (0.0, -0.0),   # +0 * -0 = -0
    (-0.0, 0.0),   # -0 * +0 = -0
    (-0.0, -0.0),  # -0 * -0 = +0
    (1.0, -0.0),   # +1 * -0 = -0
    (-1.0, 0.0),   # -1 * +0 = -0
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.multiply(np.float64(x1), np.float64(x2))
    np_result = np.multiply(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: multiply({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "multiply signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn divide_signed_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
# IEEE 754: 0 / x preserves sign rules
tests = [
    (0.0, 1.0),    # +0 / +1 = +0
    (0.0, -1.0),   # +0 / -1 = -0
    (-0.0, 1.0),   # -0 / +1 = -0
    (-0.0, -1.0),  # -0 / -1 = +0
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.divide(np.float64(x1), np.float64(x2))
    np_result = np.divide(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: divide({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divide signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

/// A small plain float64 add / subtract / multiply is computed natively below the op's
/// `NumpyFasterBelow` crossover (`small_native_binary`): two same-shape C-contiguous float64
/// arrays, or one and a Python float / int or an `np.float64`. Every observable must stay
/// numpy's: type, dtype, shape, strides, bytes, `out` identity, warnings and errors, including
/// the results that must go back to numpy (overflow, `inf - inf`, NaN payloads, a product that
/// underflows under `under='warn'` or `all='raise'`), signed zeros, and every operand the route
/// declines (other dtypes, broadcasting, F order, strided, misaligned, 0-d, empty, matrix,
/// lists, bools, numpy float32 / int64 scalars, out-of-range ints). A spy on numpy.add /
/// numpy.multiply proves the small calls no longer run them, while an overflowing one still does.
#[test]
fn small_float64_arithmetic_matches_numpy_and_computes_natively() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*a, **k); x = np.asarray(r)
            res = ("ok", type(r).__name__, x.dtype.str, x.shape,
                   x.strides if isinstance(r, np.ndarray) else None, x.tobytes(), r is k.get("out"))
        except Exception as e:
            res = ("raise", type(e).__name__, str(e))
    return res + (tuple(sorted((x.category.__name__, str(x.message)) for x in w)),)
def misaligned(x):
    buf = np.zeros(x.nbytes + 1, np.uint8)
    buf[1:] = np.ascontiguousarray(x).view(np.uint8).ravel()
    return np.frombuffer(buf.data, dtype=x.dtype, count=x.size, offset=1).reshape(x.shape)
rng = np.random.default_rng(20261007)
base = rng.standard_normal(64) * 10
arrays = {"f8[64]": base, "f8[7]": base[:7], "f8 2-D": base.reshape(8, 8), "f8 3-D": base.reshape(2, 4, 8),
          "f8[1]": base[:1], "-0.0": np.full(9, -0.0), "+0.0": np.zeros(9), "inf": np.array([np.inf, 1.0, -np.inf]),
          "nan": np.array([np.nan, 1.0, 2.0]), "big": np.full(5, 1e308), "tiny": np.full(5, 1e-200),
          "sub": np.full(5, 5e-324), "F": np.asfortranarray(base.reshape(8, 8)), "strided": base[::2],
          "misaligned": misaligned(base[:9]), "f4": base[:9].astype("f4"), "i8": np.arange(9), "0-d": np.array(2.5),
          "f8[0]": np.zeros(0), "big8000": rng.standard_normal(8000), "matrix": np.matrix(base[:4].reshape(2, 2))}
scalars = {"2.5": 2.5, "-0.0s": -0.0, "3": 3, "2**60": 2**60, "2**70": 2**70, "True": True,
           "np.f64": np.float64(1.5), "np.f32": np.float32(1.5), "np.i64": np.int64(4), "inf s": float("inf"),
           "nan s": float("nan"), "1e308s": 1e308, "1e-200s": 1e-200, "list": [1.0] * 9}
pairs = []
for an, a in arrays.items():
    for bn, b in arrays.items():
        if an == bn or np.shape(a) == np.shape(b) or bn in ("f8[1]", "0-d") or an == "f8[1]":
            pairs.append((an, a, bn, b))
    for sn, sv in scalars.items():
        pairs.append((an, a, sn, sv))
        pairs.append((sn, sv, an, a))
pairs.append(("column", base[:3].reshape(3, 1), "row", base[:4].reshape(1, 4)))
cells, bad = 0, []
for name in ("add", "subtract", "multiply"):
    for an, a, bn, b in pairs:
        for kw in ({},) if np.ndim(a) == 0 and np.ndim(b) == 0 else ({}, {"dtype": "f8"}):
            for es in ({}, {"under": "warn"}, {"all": "raise"}, {"over": "raise"}):
                cells += 1
                with np.errstate(**es):
                    if outcome(getattr(fnp, name), a, b, **kw) != outcome(getattr(np, name), a, b, **kw):
                        bad.append((name, an, bn, kw, es))
x, y = base, base[::-1].copy()
expected = (np.add(x, y), np.multiply(x, 3), np.subtract(2.0, x))
real_add, real_multiply = np.add, np.multiply
calls = []
class Spy:
    def __init__(self, real):
        self.real = real
    def __call__(self, *args, **kwargs):
        calls.append(1)
        return self.real(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.real, name)
np.add, np.multiply = Spy(real_add), Spy(real_multiply)
got = (fnp.add(x, y), fnp.multiply(x, 3), fnp.subtract(2.0, x))
native = not calls and all(g.tobytes() == e.tobytes() for g, e in zip(got, expected))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fnp.add(np.full(4, 1e308), np.full(4, 1e308))
delegated_overflow = len(calls) == 1
np.add, np.multiply = real_add, real_multiply
print(cells, native, delegated_overflow, bad[:6])
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "16452 True True []");
    Ok(())
}

/// The small native add / subtract / multiply for float32, int64 and int32 (`small_native_binary`):
/// float32 in its own IEEE arithmetic with numpy's events left to numpy, the integers wrapping as
/// numpy's array loops do silently. Scalars ride only where NEP 50 keeps the array's dtype exactly
/// (a Python float32-exact value for float32, a Python int in range for int32, a Python int or an
/// np.int64 for int64); every other scalar - np.float64 / np.int64 against float32 / int32 (they
/// promote), Python floats against integers, ints past int32 (numpy's OverflowError), bools - is
/// numpy's. Every observable must stay numpy's across 12,072 cells under four errstates, and a spy
/// on numpy.add / numpy.multiply proves the small calls no longer run them.
#[test]
fn small_f32_and_int_arithmetic_matches_numpy_and_computes_natively() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*a, **k); x = np.asarray(r)
            res = ("ok", type(r).__name__, x.dtype.str, x.shape,
                   x.strides if isinstance(r, np.ndarray) else None, x.tobytes())
        except Exception as e:
            res = ("raise", type(e).__name__, str(e))
    return res + (tuple(sorted((x.category.__name__, str(x.message)) for x in w)),)
rng = np.random.default_rng(9)
i64 = rng.integers(-1000, 1000, 64)
i32 = i64.astype("i4")
f32 = (rng.standard_normal(64) * 10).astype("f4")
arrays = {"i8": i64, "i8 2-D": i64.reshape(8, 8), "i8 max": np.full(5, np.iinfo(np.int64).max),
          "i8 min": np.full(5, np.iinfo(np.int64).min), "i4": i32, "i4 max": np.full(5, np.iinfo(np.int32).max, "i4"),
          "i4 min": np.full(5, np.iinfo(np.int32).min, "i4"), "i4 strided": i32[::2], "f4": f32,
          "f4 2-D": f32.reshape(8, 8), "f4 -0": np.full(5, -0.0, "f4"), "f4 big": np.full(5, 3e38, "f4"),
          "f4 tiny": np.full(5, 1e-30, "f4"), "f4 inf": np.array([np.inf, 1, -np.inf], "f4"),
          "f4 nan": np.array([np.nan, 1], "f4"), "f4 sub": np.full(4, 1e-45, "f4"), "f8": i64.astype("f8"),
          "u1": i64.astype("u1"), "i2": i64.astype("i2"), "q": i64.astype("q"), "bool": i64 > 0,
          "f4 F": np.asfortranarray(f32.reshape(8, 8))}
scalars = {"3": 3, "-7": -7, "2**31": 2**31, "-2**31-1": -2**31 - 1, "2**31-1": 2**31 - 1, "2**40": 2**40,
           "2**63": 2**63, "2**24+1": 2**24 + 1, "2.5": 2.5, "0.1": 0.1, "1e300": 1e300, "-0.0": -0.0,
           "inf": float("inf"), "nan": float("nan"), "True": True, "np.i8": np.int64(5), "np.i4": np.int32(5),
           "np.f4": np.float32(1.5), "np.f8": np.float64(1.5), "np.u8": np.uint64(3)}
pairs = []
for an, a in arrays.items():
    for bn, b in arrays.items():
        if np.shape(a) == np.shape(b):
            pairs.append((an, a, bn, b))
    for sn, sv in scalars.items():
        pairs.append((an, a, sn, sv))
        pairs.append((sn, sv, an, a))
cells, bad = 0, []
for name in ("add", "subtract", "multiply"):
    for an, a, bn, b in pairs:
        for es in ({}, {"under": "warn"}, {"all": "raise"}, {"over": "raise"}):
            cells += 1
            with np.errstate(**es):
                if outcome(getattr(fnp, name), a, b) != outcome(getattr(np, name), a, b):
                    bad.append((name, an, bn, es))
real_add, real_multiply = np.add, np.multiply
calls = []
class Spy:
    def __init__(self, real):
        self.real = real
    def __call__(self, *args, **kwargs):
        calls.append(1)
        return self.real(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.real, name)
expected = (real_add(i64, i64), real_multiply(i32, 3), real_add(f32, f32), real_multiply(f32, 2.5))
np.add, np.multiply = Spy(real_add), Spy(real_multiply)
got = (fnp.add(i64, i64), fnp.multiply(i32, 3), fnp.add(f32, f32), fnp.multiply(f32, 2.5))
native = not calls and all(g.tobytes() == e.tobytes() for g, e in zip(got, expected))
np.add, np.multiply = real_add, real_multiply
print(cells, native, bad[:6])
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "12072 True []");
    Ok(())
}

/// A small plain float64 / float32 divide is computed natively too (`small_native_binary`): a
/// correctly rounded quotient in the array's own type, as numpy's loop computes it, with every
/// event left to numpy - a zero divisor ("divide by zero" / "invalid value"), overflow, and a
/// quotient that may have underflowed under `under='warn'` or `all='raise'`. Integer and float16
/// operands, and scalars NEP 50 would promote, stay numpy's. A spy on numpy.divide proves the
/// small calls no longer run it, while a zero divisor still does.
#[test]
fn small_float_divide_matches_numpy_and_computes_natively() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*a, **k); x = np.asarray(r)
            res = ("ok", type(r).__name__, x.dtype.str, x.shape,
                   x.strides if isinstance(r, np.ndarray) else None, x.tobytes())
        except Exception as e:
            res = ("raise", type(e).__name__, str(e))
    return res + (tuple(sorted((x.category.__name__, str(x.message)) for x in w)),)
rng = np.random.default_rng(20261007)
arrays = {}
for dt in ("f8", "f4"):
    base = (rng.standard_normal(64) * 10).astype(dt)
    arrays.update({
        f"{dt}[64]": base, f"{dt}[7]": base[:7], f"{dt} 2-D": base.reshape(8, 8),
        f"{dt} -0.0": np.full(7, -0.0, dt), f"{dt} 0.0": np.zeros(7, dt),
        f"{dt} inf": np.array([np.inf, 1.0, -np.inf, 2.0, 3.0, 4.0, 5.0], dt),
        f"{dt} nan": np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dt),
        f"{dt} big": np.full(7, np.finfo(dt).max / 4, dt), f"{dt} tiny": np.full(7, np.finfo(dt).tiny * 4, dt),
        f"{dt} sub": np.full(7, np.finfo(dt).smallest_subnormal, dt),
        f"{dt} strided": base[::2][:7], f"{dt} 8000": (rng.standard_normal(8000) + 3).astype(dt),
    })
arrays.update({"i8[7]": np.arange(1, 8), "i4[7]": np.arange(1, 8, dtype="i4"), "f2[7]": np.arange(1, 8, dtype="f2")})
scalars = {"2.5": 2.5, "-0.0s": -0.0, "0.0s": 0.0, "3": 3, "0": 0, "1e-300s": 1e-300, "1e300s": 1e300,
           "inf s": float("inf"), "np.f64": np.float64(0.1), "np.f32": np.float32(0.1)}
pairs = []
for an, a in arrays.items():
    for bn, b in arrays.items():
        if np.shape(a) == np.shape(b):
            pairs.append((an, a, bn, b))
    for sn, sv in scalars.items():
        pairs.append((an, a, sn, sv))
        pairs.append((sn, sv, an, a))
cells, bad = 0, []
for an, a, bn, b in pairs:
    for es in ({}, {"under": "warn"}, {"divide": "raise"}, {"all": "raise"}):
        cells += 1
        with np.errstate(**es):
            if outcome(fnp.divide, a, b) != outcome(np.divide, a, b):
                bad.append((an, bn, es))
x, y = arrays["f8[64]"], arrays["f8[64]"][::-1] + 30
x4, y4 = arrays["f4[64]"], arrays["f4[64]"][::-1] + 30
expected = (np.divide(x, y), np.divide(x, 3.0), np.divide(x4, y4), np.divide(2.0, y4))
real_divide = np.divide
calls = []
class Spy:
    def __init__(self, real):
        self.real = real
    def __call__(self, *args, **kwargs):
        calls.append(1)
        return self.real(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.real, name)
np.divide = Spy(real_divide)
got = (fnp.divide(x, y), fnp.divide(x, 3.0), fnp.divide(x4, y4), fnp.divide(2.0, y4))
native = not calls and all(g.tobytes() == e.tobytes() for g, e in zip(got, expected))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fnp.divide(x, np.zeros_like(x))
delegated_zero_divisor = len(calls) == 1
np.divide = real_divide
print(cells, native, delegated_zero_divisor, bad[:6])
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "3972 True True []");
    Ok(())
}

/// The small native add / subtract / multiply / divide write into the caller's `out=` (an array
/// or a one-array tuple) and return it, as numpy does, when it is an exact, C-contiguous,
/// writeable ndarray of exactly the result's dtype and shape. An `out` that shares memory with an
/// operand - `out=a`, `out=b`, both operands the same array, a view shifted one item either way -
/// gets numpy's as-if-no-overlap result, including when a zero divisor or an overflow hands the
/// call back to numpy (the operands must still be unwritten then). Every other `out` - another
/// dtype, a shape numpy broadcasts to, read-only, strided, F order, a matrix, a list, a 2-tuple,
/// None - stays numpy's. The returned object, `out`'s bytes, the operands' bytes afterwards and
/// the warnings must all be numpy's, and a spy proves the small calls no longer reach numpy.
#[test]
fn small_arithmetic_writes_into_out_and_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(20261007)

def run(lib, name, make, errstate):
    ops, kw, keep = make()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with np.errstate(**errstate):
            try:
                r = getattr(lib, name)(*ops, **kw)
                x = np.asarray(r)
                res = ("ok", type(r).__name__, x.dtype.str, x.shape, any(r is k for k in keep),
                       x.tobytes())
            except Exception as e:
                res = ("raise", type(e).__name__, str(e))
    after = tuple(np.asarray(k).tobytes() for k in keep)
    return res + (after, tuple(sorted((x.category.__name__, str(x.message)) for x in w)))

def into(a, b, o, wrap=None):
    # Fresh copies per arm: `out` is written, and an aliased operand with it.
    def make():
        a2, b2, o2 = a.copy(), b.copy(), o.copy()
        if wrap == "readonly":
            o2.flags.writeable = False
        return (a2, b2), {"out": (o2,) if wrap == "tuple" else o2}, (o2, a2, b2)
    return make

def inplace(a, b, which):
    def make():
        a2, b2 = a.copy(), b.copy()
        return (a2, b2), {"out": a2 if which == "a" else b2}, (a2, b2)
    return make

def cases_for(dt, op):
    n = 64
    if dt[0] == "f":
        a, b = ((rng.standard_normal(n) * 10).astype(dt) for _ in range(2))
    else:
        a, b = (rng.integers(-50, 50, n).astype(dt) for _ in range(2))
    if op == "divide":
        b = np.where(b == 0, 3, b).astype(dt)
    zero_b = b.copy()
    zero_b[5] = 0
    c = {
        "fresh out": into(a, b, np.empty(n, dt)),
        "tuple out": into(a, b, np.empty(n, dt), "tuple"),
        "out read-only": into(a, b, np.zeros(n, dt), "readonly"),
        "out=a": inplace(a, b, "a"),
        "out=b": inplace(a, b, "b"),
        "zero divisor out=a": inplace(a, zero_b, "a"),
        "zero divisor fresh": into(a, zero_b, np.zeros(n, dt)),
        "out other dtype": into(a, b, np.zeros(n, "f4" if dt != "f4" else "f8")),
        "out broadcast": into(a, b, np.zeros((2, n), dt)),
        "out 2-D": into(a.reshape(8, 8), b.reshape(8, 8), np.zeros((8, 8), dt)),
        "out F": into(a.reshape(8, 8), b.reshape(8, 8), np.zeros((8, 8), dt, order="F")),
        "out big": into(a.repeat(200), b.repeat(200), np.zeros(n * 200, dt)),
    }
    def same(x):
        x2 = x.copy()
        return (x2, x2), {"out": x2}, (x2,)
    c["out=a=b"] = lambda: same(b)
    def shifted(step):
        def make():
            buf = np.concatenate([a, a[:1]]).astype(dt)
            ins, out = (buf[:-1], buf[1:]) if step > 0 else (buf[1:], buf[:-1])
            return (ins, b.copy()), {"out": out}, (buf,)
        return make
    c["out shifted +1"], c["out shifted -1"] = shifted(1), shifted(-1)
    def strided():
        o = np.zeros(2 * n, dt)
        return (a.copy(), b.copy()), {"out": o[::2]}, (o,)
    c["out strided"] = strided
    def scalar_out(first):
        def make():
            a2, o = a.copy(), np.empty(n, dt)
            return ((2, a2) if first else (a2, 3)), {"out": o}, (o, a2)
        return make
    c["scalar first"], c["scalar second"] = scalar_out(True), scalar_out(False)
    for label, value in (("out=None", None), ("out=(None,)", (None,))):
        c[label] = (lambda value: lambda: ((a.copy(), b.copy()), {"out": value}, ()))(value)
    def other(kind):
        def make():
            o = np.zeros(n, dt)
            out = {"2-tuple": (o, o), "list": [o]}[kind]
            return (a.copy(), b.copy()), {"out": out}, (o,)
        return make
    c["out 2-tuple"], c["out list"] = other("2-tuple"), other("list")
    def matrix():
        o = np.matrix(np.zeros((8, 8), dt))
        return (a.copy().reshape(8, 8), b.copy().reshape(8, 8)), {"out": o}, (o,)
    c["out matrix"] = matrix
    if dt[0] == "f":
        info = np.finfo(dt)
        c["overflow out=a"] = lambda: same(np.full(n, info.max, dt))
        def underflow():
            a2 = np.full(n, info.tiny, dt)
            return (a2, np.full(n, 1e10 if dt == "f4" else 1e300, dt)), {"out": a2}, (a2,)
        c["underflow out=a"] = underflow
    return c

cells, bad = 0, []
for dt in ("f8", "f4", "i8", "i4"):
    for op in ("add", "subtract", "multiply", "divide"):
        if op == "divide" and dt[0] == "i":
            continue
        for label, make in cases_for(dt, op).items():
            for es in ({}, {"under": "warn"}, {"all": "raise"}):
                cells += 1
                if run(fnp, op, make, es) != run(np, op, make, es):
                    bad.append((dt, op, label, es))
real = {name: getattr(np, name) for name in ("add", "multiply", "divide")}
calls = []
class Spy:
    def __init__(self, real):
        self.real = real
    def __call__(self, *args, **kwargs):
        calls.append(1)
        return self.real(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.real, name)
a, b = rng.standard_normal(64), rng.standard_normal(64) + 10
o = np.empty(64)
for name in real:
    setattr(np, name, Spy(real[name]))
r1 = fnp.add(a, b, out=o)
r2 = fnp.multiply(a, 3.0, out=(a.copy(),))
r3 = fnp.divide(a, b, out=a)
native = not calls and r1 is o and r3 is a and isinstance(r2, np.ndarray)
for name in real:
    setattr(np, name, real[name])
print(cells, native, bad[:8])
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "1014 True []");
    Ok(())
}
