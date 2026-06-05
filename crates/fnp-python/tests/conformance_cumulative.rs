//! Conformance tests for numpy cumulative operations against NumPy oracle.
//!
//! Tests cumsum, cumprod, diff (diff is in here for completeness with incremental ops).

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
// cumsum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 1d should match numpy");
    Ok(())
}

#[test]
fn cumsum_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumsum 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn cumsum_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a, axis=0)
expected = np.cumsum(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn cumsum_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a, axis=1)
expected = np.cumsum(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn cumsum_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.1, 0.2, 0.3, 0.4])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumprod_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod 1d should match numpy");
    Ok(())
}

#[test]
fn cumprod_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a, axis=0)
expected = np.cumprod(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a, axis=1)
expected = np.cumprod(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.1, 1.2, 1.3, 1.4])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod float should match numpy");
    Ok(())
}

#[test]
fn cumprod_with_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 0, 4, 5])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod with zero should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_last_equals_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
cumsum_result = fnp.cumsum(a)
sum_result = fnp.sum(a)
print(cumsum_result[-1] == sum_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "last cumsum element should equal sum"
    );
    Ok(())
}

#[test]
fn cumprod_last_equals_prod() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
cumprod_result = fnp.cumprod(a)
prod_result = fnp.prod(a)
print(cumprod_result[-1] == prod_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "last cumprod element should equal prod"
    );
    Ok(())
}

#[test]
fn cumsum_diff_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
# For cumulative sum: diff(cumsum(a)) gives a[1:] (the original array without first element)
a = np.array([1, 2, 3, 4, 5])
cumsum_a = fnp.cumsum(a)
diff_cumsum = fnp.diff(cumsum_a)
# diff of cumsum gives the original elements (except first)
print(np.array_equal(diff_cumsum, a[1:]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff(cumsum(a)) should equal a[1:]");
    Ok(())
}

#[test]
fn cumsum_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
fnp_result = fnp.cumsum(z)
np_result = np.cumsum(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum complex should match numpy");
    Ok(())
}

#[test]
fn cumprod_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+0j, 0+1j], dtype=np.complex128)
fnp_result = fnp.cumprod(z)
np_result = np.cumprod(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod complex should match numpy");
    Ok(())
}
