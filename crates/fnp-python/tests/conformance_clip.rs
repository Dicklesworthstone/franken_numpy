//! Conformance tests for numpy.clip against NumPy oracle.
//!
//! Tests clip (clip array values to range).

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
fn clip_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
result = fnp.clip(a, 3, 7)
expected = np.clip(a, 3, 7)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip basic should match numpy");
    Ok(())
}

#[test]
fn clip_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
result = fnp.clip(a, 1.0, 4.0)
expected = np.clip(a, 1.0, 4.0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip float should match numpy");
    Ok(())
}

#[test]
fn clip_min_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.clip(a, 3, None)
expected = np.clip(a, 3, None)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip with min only should match numpy"
    );
    Ok(())
}

#[test]
fn clip_max_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.clip(a, None, 3)
expected = np.clip(a, None, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip with max only should match numpy"
    );
    Ok(())
}

#[test]
fn clip_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.clip(x, 2.0, 8.0)
np_result = np.clip(x, 2.0, 8.0)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "clip scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn clip_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 5+5j, 2+2j], dtype=np.complex128)
fnp_result = fnp.clip(z, 0+0j, 3+3j)
np_result = np.clip(z, 0+0j, 3+3j)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip complex should match numpy");
    Ok(())
}

#[test]
fn clip_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 5.0])
fnp_result = fnp.clip(a, 2.0, 4.0)
np_result = np.clip(a, 2.0, 4.0)
# NaN should propagate through clip
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn clip_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, 0.0])
fnp_result = fnp.clip(a, -1.0, 1.0)
np_result = np.clip(a, -1.0, 1.0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip inf handling should match numpy"
    );
    Ok(())
}

#[test]
fn clip_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-0.0, 0.0, -1.0, 1.0])
fnp_result = fnp.clip(a, -0.5, 0.5)
np_result = np.clip(a, -0.5, 0.5)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn clip_inf_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.clip(a, -np.inf, np.inf)
np_result = np.clip(a, -np.inf, np.inf)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip inf bounds should match numpy");
    Ok(())
}

#[test]
fn clip_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
a_min = np.array([2, 2, 2])
a_max = np.array([5, 5, 5])
fnp_result = fnp.clip(a, a_min, a_max)
np_result = np.clip(a, a_min, a_max)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn clip_out_parameter() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
out = np.empty_like(a)
fnp_result = fnp.clip(a, 2.0, 4.0, out=out)
np_out = np.empty_like(a)
np_result = np.clip(a, 2.0, 4.0, out=np_out)
print(np.array_equal(out, np_out) and fnp_result is out)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip out parameter should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clip_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
fnp_result = fnp.clip(a, 2.0, 4.0)
np_result = np.clip(a, 2.0, 4.0)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip empty array should match numpy");
    Ok(())
}

#[test]
fn clip_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0])
fnp_result = fnp.clip(a, 2.0, 4.0)
np_result = np.clip(a, 2.0, 4.0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip single element should match numpy"
    );
    Ok(())
}

#[test]
fn clip_equal_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.clip(a, 3.0, 3.0)
np_result = np.clip(a, 3.0, 3.0)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 3.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip equal bounds should match numpy"
    );
    Ok(())
}

#[test]
fn clip_integer_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a_int32 = np.array([1, 5, 10, 15, 20], dtype=np.int32)
a_int64 = np.array([1, 5, 10, 15, 20], dtype=np.int64)
tests_pass = True
for a in [a_int32, a_int64]:
    fnp_result = fnp.clip(a, 5, 15)
    np_result = np.clip(a, 5, 15)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip integer dtypes should match numpy"
    );
    Ok(())
}

#[test]
fn clip_all_within_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
fnp_result = fnp.clip(a, 1.0, 10.0)
np_result = np.clip(a, 1.0, 10.0)
print(np.array_equal(fnp_result, np_result) and np.array_equal(fnp_result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip all within bounds should match numpy"
    );
    Ok(())
}

#[test]
fn clip_all_below_min() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.clip(a, 10.0, 20.0)
np_result = np.clip(a, 10.0, 20.0)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 10.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip all below min should match numpy"
    );
    Ok(())
}

#[test]
fn clip_all_above_max() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
fnp_result = fnp.clip(a, 1.0, 5.0)
np_result = np.clip(a, 1.0, 5.0)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 5.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip all above max should match numpy"
    );
    Ok(())
}
