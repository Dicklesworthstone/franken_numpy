//! Conformance tests for numpy basic array creation functions against NumPy oracle.
//!
//! Tests empty, zeros, ones.

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
// empty
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn empty_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.empty(5)
expected = np.empty(5)
print(result.shape == expected.shape and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "empty 1d shape/dtype should match numpy"
    );
    Ok(())
}

#[test]
fn empty_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.empty((3, 4))
expected = np.empty((3, 4))
print(result.shape == expected.shape and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "empty 2d shape/dtype should match numpy"
    );
    Ok(())
}

#[test]
fn empty_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.empty((2, 3), dtype='int32')
expected = np.empty((2, 3), dtype='int32')
print(result.shape == expected.shape and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "empty with dtype should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// zeros
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zeros_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.zeros(5)
expected = np.zeros(5)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros 1d should match numpy");
    Ok(())
}

#[test]
fn zeros_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.zeros((3, 4))
expected = np.zeros((3, 4))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros 2d should match numpy");
    Ok(())
}

#[test]
fn zeros_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.zeros((2, 3), dtype='int32')
expected = np.zeros((2, 3), dtype='int32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros with dtype should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ones
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ones_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ones(5)
expected = np.ones(5)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones 1d should match numpy");
    Ok(())
}

#[test]
fn ones_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ones((3, 4))
expected = np.ones((3, 4))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones 2d should match numpy");
    Ok(())
}

#[test]
fn ones_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ones((2, 3), dtype='int32')
expected = np.ones((2, 3), dtype='int32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones with dtype should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zeros_all_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.zeros((3, 4))
print(np.all(result == 0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "zeros should be all zero");
    Ok(())
}

#[test]
fn ones_all_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.ones((3, 4))
print(np.all(result == 1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ones should be all one");
    Ok(())
}

#[test]
fn vander_native_cumprod_bitexact_matches_numpy() -> Result<(), String> {
    // Exercises the native fused-cumprod vander fast path against numpy bit-exactly
    // (atol=0, equal_nan=True) incl dtype/shape: default N, explicit N (wider and
    // narrower than len(x)), increasing True/False, negative/large x, a NaN/Inf x,
    // and an int-x fallthrough (numpy keeps int64 dtype).
    let script = fnp_script(
        r#"
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(31)
x = rng.standard_normal(5000) * 2.0
xspec = np.array([0.0, 1.0, -2.0, np.inf, -np.inf, np.nan, 1e3], dtype=np.float64)
xint = np.array([1, 2, 3, 4], dtype=np.int64)
ok = True
cases = [
    (x, None, False),
    (x, None, True),
    (x, 8, False),
    (x, 8, True),
    (x, 3, False),      # N < len(x)
    (xspec, 6, False),
    (xspec, 6, True),
    (xint, 4, False),   # int x -> numpy keeps int64 (fallthrough)
]
for xv, N, inc in cases:
    if N is None:
        f = fnp.vander(xv, increasing=inc)
        n = np.vander(xv, increasing=inc)
    else:
        f = fnp.vander(xv, N=N, increasing=inc)
        n = np.vander(xv, N=N, increasing=inc)
    if not same(f, n):
        print("FAIL", N, inc, np.asarray(f).ravel()[:6], np.asarray(n).ravel()[:6]); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native vander parity should match numpy: {result}"
    );
    Ok(())
}
