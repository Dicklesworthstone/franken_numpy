//! Conformance tests for numpy.compress, choose, diagonal against NumPy oracle.
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
// compress
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, True, False, True, True])
a = np.array([1, 2, 3, 4, 5])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 1d basic should match numpy"
    );
    Ok(())
}

#[test]
fn compress_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 2d no axis should match numpy"
    );
    Ok(())
}

#[test]
fn compress_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.compress(condition, a, axis=0)
expected = np.compress(condition, a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn compress_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.compress(condition, a, axis=1)
expected = np.compress(condition, a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn compress_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
a = np.array([1, 2, 3])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected) and len(result) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress all false should return empty"
    );
    Ok(())
}

#[test]
fn compress_string_payload_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
a = np.array(["alpha", "beta", "gamma"])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress should preserve NumPy string payload behavior"
    );
    Ok(())
}

#[test]
fn compress_string_condition_truthiness_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array(["", "x", "0"])
a = np.array([10, 20, 30])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress should match NumPy string condition truthiness"
    );
    Ok(())
}

#[test]
fn compress_object_condition_truthiness_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([object(), None, 1], dtype=object)
a = np.array([10, 20, 30])
result = fnp.compress(condition, a)
expected = np.compress(condition, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress should match NumPy object condition truthiness"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// choose
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn choose_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 2, 1, 0])
choices = [np.array([10, 10, 10, 10, 10]), np.array([20, 20, 20, 20, 20]), np.array([30, 30, 30, 30, 30])]
result = fnp.choose(a, choices)
expected = np.choose(a, choices)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "choose basic should match numpy");
    Ok(())
}

#[test]
fn choose_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[0, 1], [1, 0]])
choices = [np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])]
result = fnp.choose(a, choices)
expected = np.choose(a, choices)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "choose 2d should match numpy");
    Ok(())
}

#[test]
fn choose_float_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 0, 1])
choices = [np.array([1.5, 2.5, 3.5, 4.5]), np.array([10.5, 20.5, 30.5, 40.5])]
result = fnp.choose(a, choices)
expected = np.choose(a, choices)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "choose float arrays should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diagonal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diagonal_2d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diagonal(a)
expected = np.diagonal(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal 2d basic should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_offset_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diagonal(a, offset=1)
expected = np.diagonal(a, offset=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal offset=1 should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_offset_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.diagonal(a, offset=-1)
expected = np.diagonal(a, offset=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal offset=-1 should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_non_square() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = fnp.diagonal(a)
expected = np.diagonal(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal non-square should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_3d_default_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.diagonal(a)
expected = np.diagonal(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal 3d default axes should match numpy"
    );
    Ok(())
}

#[test]
fn diagonal_3d_custom_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.diagonal(a, axis1=0, axis2=2)
expected = np.diagonal(a, axis1=0, axis2=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diagonal 3d custom axes should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_extract_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False, True])
a = np.array([1, 2, 3, 4, 5])
compress_result = fnp.compress(condition, a)
extract_result = fnp.extract(condition, a)
print(np.array_equal(compress_result, extract_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "compress and extract should be equivalent for 1d"
    );
    Ok(())
}

#[test]
fn compress_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
condition = [True, False, True, False]
fnp_result = fnp.compress(condition, a)
np_result = np.compress(condition, a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "compress complex should match numpy");
    Ok(())
}

#[test]
fn choose_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
choices = [np.array([1+1j, 2+2j], dtype=np.complex128),
           np.array([3+3j, 4+4j], dtype=np.complex128)]
a = [0, 1]
fnp_result = fnp.choose(a, choices)
np_result = np.choose(a, choices)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "choose complex should match numpy");
    Ok(())
}
