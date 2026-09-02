//! Conformance tests for native np.zeros and np.ones implementations.

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
// zeros
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zeros_1d_default_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros(5)
np_result = np.zeros(5)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np_result.dtype)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros 1d default dtype mismatch");
    Ok(())
}

#[test]
fn zeros_2d_tuple_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros((3, 4))
np_result = np.zeros((3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros 2d tuple shape mismatch");
    Ok(())
}

#[test]
fn zeros_with_int_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros((2, 3), dtype=np.int32)
np_result = np.zeros((2, 3), dtype=np.int32)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.int32)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros with int dtype mismatch");
    Ok(())
}

#[test]
fn zeros_with_bool_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros(4, dtype=bool)
np_result = np.zeros(4, dtype=bool)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.bool_)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros with bool dtype mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ones
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ones_1d_default_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones(5)
np_result = np.ones(5)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np_result.dtype)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones 1d default dtype mismatch");
    Ok(())
}

#[test]
fn ones_2d_tuple_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones((3, 4))
np_result = np.ones((3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones 2d tuple shape mismatch");
    Ok(())
}

#[test]
fn ones_with_int_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones((2, 3), dtype=np.int64)
np_result = np.ones((2, 3), dtype=np.int64)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.int64)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones with int dtype mismatch");
    Ok(())
}

#[test]
fn ones_with_float32_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones(10, dtype=np.float32)
np_result = np.ones(10, dtype=np.float32)
print(np.array_equal(fnp_result, np_result) and fnp_result.dtype == np.float32)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones with float32 dtype mismatch");
    Ok(())
}

#[test]
fn ones_3d_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ones((2, 3, 4))
np_result = np.ones((2, 3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (2, 3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ones 3d shape mismatch");
    Ok(())
}

#[test]
fn zeros_empty_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.zeros((0,))
np_result = np.zeros((0,))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (0,))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "zeros empty shape mismatch");
    Ok(())
}
