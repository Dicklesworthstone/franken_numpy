//! Conformance tests for numpy.diff against NumPy oracle.
//!
//! Tests diff (calculate n-th discrete difference).

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
fn diff_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 11])
result = fnp.diff(a)
expected = np.diff(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff basic should match numpy");
    Ok(())
}

#[test]
fn diff_n2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7, 11])
result = fnp.diff(a, n=2)
expected = np.diff(a, n=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff with n=2 should match numpy"
    );
    Ok(())
}

#[test]
fn diff_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diff(a, axis=0)
expected = np.diff(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff 2D axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn diff_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diff(a, axis=1)
expected = np.diff(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff 2D axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn diff_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.5, 4.5, 7.0])
result = fnp.diff(a)
expected = np.diff(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff float should match numpy"
    );
    Ok(())
}

#[test]
fn diff_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 3+4j, 6+9j], dtype=np.complex128)
fnp_result = fnp.diff(z)
np_result = np.diff(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff complex should match numpy");
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
        if stderr.contains("AxisError") || stderr.contains("axis") {
            "AxisError".to_string()
        } else if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn diff_axis_out_of_bounds_raises_axiserror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(12).reshape(3, 4)
fnp.diff(a, axis=5)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
np.diff(a, axis=5)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "diff with out-of-bounds axis should raise same error as numpy"
    );
}

#[test]
fn diff_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff special values should match numpy");
    Ok(())
}

#[test]
fn diff_constant_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0, 5.0, 5.0, 5.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
# Diff of constant should be zero
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff constant array should match numpy");
    Ok(())
}

#[test]
fn diff_two_elements() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 5.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff two elements should match numpy");
    Ok(())
}

#[test]
fn diff_with_prepend() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7])
fnp_result = fnp.diff(a, prepend=0)
np_result = np.diff(a, prepend=0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff with prepend should match numpy");
    Ok(())
}

#[test]
fn diff_with_append() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 4, 7])
fnp_result = fnp.diff(a, append=20)
np_result = np.diff(a, append=20)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff with append should match numpy");
    Ok(())
}
