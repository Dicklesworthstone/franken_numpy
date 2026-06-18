//! Conformance tests for numpy rounding functions against NumPy oracle.
//!
//! Tests floor, ceil, trunc, rint functions.

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
fn floor_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.7, -1.5, -0.5, 0.0, 0.5, 1.5, 1.7])
result = fnp.floor(x)
expected = np.floor(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "floor basic should match numpy");
    Ok(())
}

#[test]
fn ceil_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.7, -1.5, -0.5, 0.0, 0.5, 1.5, 1.7])
result = fnp.ceil(x)
expected = np.ceil(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ceil basic should match numpy");
    Ok(())
}

#[test]
fn trunc_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.7, -1.5, -0.5, 0.0, 0.5, 1.5, 1.7])
result = fnp.trunc(x)
expected = np.trunc(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trunc basic should match numpy");
    Ok(())
}

#[test]
fn rint_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.7, -1.5, -0.5, 0.0, 0.5, 1.5, 1.7])
result = fnp.rint(x)
expected = np.rint(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rint basic should match numpy");
    Ok(())
}

#[test]
fn floor_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.7)
fnp_result = fnp.floor(x)
np_result = np.floor(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "floor scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn ceil_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.7)
fnp_result = fnp.ceil(x)
np_result = np.ceil(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "ceil scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn trunc_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.7)
fnp_result = fnp.trunc(x)
np_result = np.trunc(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "trunc scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn rint_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.7)
fnp_result = fnp.rint(x)
np_result = np.rint(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "rint scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn floor_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.floor(x)
np_result = np.floor(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor special values should match numpy"
    );
    Ok(())
}

#[test]
fn ceil_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.ceil(x)
np_result = np.ceil(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ceil special values should match numpy"
    );
    Ok(())
}

#[test]
fn trunc_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.trunc(x)
np_result = np.trunc(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "trunc special values should match numpy"
    );
    Ok(())
}

#[test]
fn rint_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.rint(x)
np_result = np.rint(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rint special values should match numpy"
    );
    Ok(())
}

#[test]
fn rint_bankers_rounding() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5])
fnp_result = fnp.rint(x)
np_result = np.rint(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rint bankers rounding should match numpy"
    );
    Ok(())
}

#[test]
fn rounding_out_keyword_surfaces_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def out_outcome(module, name, positional=False, bad_shape=False):
    fn = getattr(module, name)
    try:
        x = np.array([-1.7, -0.2, -0.0, 0.0, 1.2], dtype=np.float64)
        out = np.empty((2,), dtype=np.float64) if bad_shape else np.empty(5, dtype=np.float64)
        if positional:
            result = fn(x, out)
        else:
            result = fn(x, out=out)
        return ("ok", result is out, out.dtype.str, tuple(out.shape), out.tolist())
    except Exception as exc:
        return ("err", type(exc).__name__, str(exc).splitlines()[0])

cases = [
    ("floor keyword out", "floor", False, False),
    ("ceil keyword out", "ceil", False, False),
    ("trunc keyword out", "trunc", False, False),
    ("rint keyword out", "rint", False, False),
    ("floor positional out", "floor", True, False),
    ("ceil positional out", "ceil", True, False),
    ("trunc positional out", "trunc", True, False),
    ("rint positional out", "rint", True, False),
    ("floor bad out shape", "floor", False, True),
    ("ceil bad out shape", "ceil", False, True),
    ("trunc bad out shape", "trunc", False, True),
    ("rint bad out shape", "rint", False, True),
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
        "rounding out keyword surfaces should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn floor_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e15 + 0.5, -1e15 - 0.5, 1e16 + 0.1, -1e16 - 0.1])
fnp_result = fnp.floor(x)
np_result = np.floor(x)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor large values should match numpy"
    );
    Ok(())
}

#[test]
fn rounding_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Rounding functions preserve sign of zero: f(-0.0) = -0.0
# IEEE 754: floor(-0.0) = -0.0, ceil(-0.0) = -0.0, trunc(-0.0) = -0.0, rint(-0.0) = -0.0
funcs = [
    ('floor', fnp.floor, np.floor),
    ('ceil', fnp.ceil, np.ceil),
    ('trunc', fnp.trunc, np.trunc),
    ('rint', fnp.rint, np.rint),
]
all_pass = True
for name, fnp_f, np_f in funcs:
    for x in [0.0, -0.0]:
        fnp_result = fnp_f(np.float64(x))
        np_result = np_f(np.float64(x))
        fnp_sign = np.signbit(fnp_result)
        np_sign = np.signbit(np_result)
        if fnp_sign != np_sign:
            print(f"FAIL: {name}({x}) signbit fnp={fnp_sign} np={np_sign}")
            all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rounding signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rounding_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
tests_pass = True
for func_name in ['floor', 'ceil', 'trunc', 'rint']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x)
    np_result = np_func(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    tests_pass = tests_pass and (fnp_result.shape == np_result.shape)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rounding empty arrays should match numpy"
    );
    Ok(())
}

#[test]
fn rounding_near_integer_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
eps = np.finfo(np.float64).eps
x = np.array([1 - eps, 1 + eps, 2 - eps, 2 + eps, -1 - eps, -1 + eps])
tests_pass = True
for func_name in ['floor', 'ceil', 'trunc', 'rint']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x)
    np_result = np_func(x)
    tests_pass = tests_pass and np.allclose(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rounding near integer values should match numpy"
    );
    Ok(())
}

#[test]
fn rounding_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([2.7])
tests_pass = True
for func_name in ['floor', 'ceil', 'trunc', 'rint']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x)
    np_result = np_func(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rounding single element should match numpy"
    );
    Ok(())
}

#[test]
fn rounding_exact_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
tests_pass = True
for func_name in ['floor', 'ceil', 'trunc', 'rint']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x)
    np_result = np_func(x)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    tests_pass = tests_pass and np.array_equal(fnp_result, x)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rounding exact integers should match numpy"
    );
    Ok(())
}

#[test]
fn rounding_subnormal_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import sys
tiny = sys.float_info.min
subnormal = tiny / 2.0
x = np.array([subnormal, -subnormal, tiny, -tiny])
tests_pass = True
for func_name in ['floor', 'ceil', 'trunc', 'rint']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x)
    np_result = np_func(x)
    tests_pass = tests_pass and np.allclose(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "rounding subnormal numbers should match numpy"
    );
    Ok(())
}

#[test]
fn fix_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.7, -0.5, 0.5, 1.7])
result = fnp.fix(x)
expected = np.fix(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fix should match numpy");
    Ok(())
}

/// Locks the zero-copy f64 fast path for `numpy.fix` (round toward zero, reusing
/// the `UnaryOp::Trunc` zero-copy unary) to bit-exact parity. fix is exactly
/// trunc on reals, so parity must hold at the IEEE-754 bit level (signed zero,
/// half-integers, nan/inf). Compares sha256 of raw output bytes against the
/// NumPy oracle across multi-D float64 inputs and extreme values.
#[test]
fn fix_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for shp in [(1000,), (3, 4), (100003,)]:
    chunks.append(np.asarray(mod.fix(rng.standard_normal(shp) * 1e6)).tobytes())
xe = np.array([0.0, -0.0, 0.4, -0.4, 2.5, -2.5, np.inf, -np.inf, np.nan], dtype=np.float64)
chunks.append(np.asarray(mod.fix(xe)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy fix must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
