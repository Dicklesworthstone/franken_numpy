//! Conformance tests for numpy.outer against NumPy oracle.
//!
//! Tests outer (outer product).

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
fn outer_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer 1d arrays should match numpy");
    Ok(())
}

#[test]
fn outer_float_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 1.5])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "outer float arrays should match numpy"
    );
    Ok(())
}

#[test]
fn outer_with_out_parameter() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
out = np.zeros((3, 2), dtype=np.int64)
fnp.outer(a, b, out=out)
expected = np.outer(a, b)
print(np.array_equal(out, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "outer with out parameter should match numpy"
    );
    Ok(())
}

#[test]
fn outer_flattens_multidim() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "outer should flatten multidim inputs like numpy"
    );
    Ok(())
}

#[test]
fn outer_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
b = np.array([1, 2, 3])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "outer with empty array should match numpy"
    );
    Ok(())
}

#[test]
fn outer_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, np.nan, 0.0])
b = np.array([1.0, 0.0, np.nan])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer special values should match numpy");
    Ok(())
}

#[test]
fn outer_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+2j, 3+4j])
b = np.array([5+6j, 7+8j])
result = fnp.outer(a, b)
expected = np.outer(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer complex should match numpy");
    Ok(())
}

#[test]
fn outer_mixed_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.5, 2.5], dtype=np.float64)
result = fnp.outer(a, b)
expected = np.outer(a, b)
# Result should be promoted to float64
print(np.allclose(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "outer mixed dtypes should match numpy");
    Ok(())
}

#[test]
fn outer_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Outer product signed-zero parity (multiplication-based)
# outer(a, b)[i, j] = a[i] * b[j]
tests = [
    ([1.0, -0.0], [1.0, 2.0]),      # -0 * positive = -0
    ([0.0, 1.0], [-0.0, 2.0]),      # positive * -0 = -0
    ([-0.0, 1.0], [-1.0, 1.0]),     # -0 * negative = 0
    ([0.0, -0.0], [0.0, -0.0]),     # zero combinations
]
all_pass = True
for a_vals, b_vals in tests:
    a = np.array(a_vals)
    b = np.array(b_vals)
    fnp_result = fnp.outer(a, b)
    np_result = np.outer(a, b)
    fnp_signs = np.signbit(fnp_result).tolist()
    np_signs = np.signbit(np_result).tolist()
    if fnp_signs != np_signs:
        print(f"FAIL: outer({a_vals}, {b_vals})")
        print(f"  fnp signbit={fnp_signs} np signbit={np_signs}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: outer({a_vals}, {b_vals}) values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "outer signed-zero parity should match numpy: {result}"
    );
    Ok(())
}
