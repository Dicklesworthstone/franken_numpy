//! Conformance tests for numpy.array_equal and numpy.array_equiv against NumPy oracle.

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
// array_equal basic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_identical_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 3]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal identical 1d mismatch");
    Ok(())
}

#[test]
fn array_equal_different_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 4]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal different values mismatch");
    Ok(())
}

#[test]
fn array_equal_different_shapes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 3, 4]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal different shapes mismatch");
    Ok(())
}

#[test]
fn array_equal_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal 2d arrays mismatch");
    Ok(())
}

#[test]
fn array_equal_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
b = np.array([])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal empty arrays mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equal with equal_nan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal nan default mismatch");
    Ok(())
}

#[test]
fn array_equal_nan_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.array_equal(a, b, equal_nan=True)
np_result = np.array_equal(a, b, equal_nan=True)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal nan=True mismatch");
    Ok(())
}

#[test]
fn array_equal_inf_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, -np.inf])
b = np.array([1.0, np.inf, -np.inf])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal inf values mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equal dtypes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_int_float_same_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal int/float mismatch");
    Ok(())
}

#[test]
fn array_equal_bool_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([True, False, True])
b = np.array([True, False, True])
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal bool arrays mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equiv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equiv_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [1, 2, 3]
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv identical mismatch");
    Ok(())
}

#[test]
fn array_equiv_broadcastable() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [1, 2]])
b = np.array([1, 2])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv broadcastable mismatch");
    Ok(())
}

#[test]
fn array_equiv_not_broadcastable() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv not broadcastable mismatch");
    Ok(())
}

#[test]
fn array_equiv_scalar_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 5, 5])
b = 5
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv scalar broadcast mismatch");
    Ok(())
}

#[test]
fn array_equiv_different_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv different values mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_scalar_inputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.array_equal(5, 5)
np_result = np.array_equal(5, 5)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal scalar inputs mismatch");
    Ok(())
}

#[test]
fn array_equal_nested_lists() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [[1, 2], [3, 4]]
b = [[1, 2], [3, 4]]
fnp_result = fnp.array_equal(a, b)
np_result = np.array_equal(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equal nested lists mismatch");
    Ok(())
}

#[test]
fn array_equiv_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([])
b = np.array([])
fnp_result = fnp.array_equiv(a, b)
np_result = np.array_equiv(a, b)
print(fnp_result == np_result == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "array_equiv empty arrays mismatch");
    Ok(())
}
