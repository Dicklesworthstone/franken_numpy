//! Conformance tests for numpy apply/vectorize functions against NumPy oracle.
//!
//! Tests apply_along_axis, apply_over_axes, vectorize, select.

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
// apply_along_axis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn apply_along_axis_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.apply_along_axis(np.sum, 0, a)
expected = np.apply_along_axis(np.sum, 0, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_along_axis sum should match numpy");
    Ok(())
}

#[test]
fn apply_along_axis_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.apply_along_axis(np.mean, 1, a)
expected = np.apply_along_axis(np.mean, 1, a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_along_axis mean should match numpy");
    Ok(())
}

#[test]
fn apply_along_axis_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.apply_along_axis(np.sum, 2, a)
expected = np.apply_along_axis(np.sum, 2, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_along_axis 3d should match numpy");
    Ok(())
}

#[test]
fn apply_along_axis_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.apply_along_axis(np.sum, -1, a)
expected = np.apply_along_axis(np.sum, -1, a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_along_axis negative axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_over_axes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn apply_over_axes_sum_single() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.apply_over_axes(np.sum, a, 0)
expected = np.apply_over_axes(np.sum, a, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_over_axes sum single axis should match numpy");
    Ok(())
}

#[test]
fn apply_over_axes_sum_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.apply_over_axes(np.sum, a, [0, 2])
expected = np.apply_over_axes(np.sum, a, [0, 2])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_over_axes sum multiple axes should match numpy");
    Ok(())
}

#[test]
fn apply_over_axes_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4).astype(float)
result = fnp.apply_over_axes(np.mean, a, [1, 2])
expected = np.apply_over_axes(np.mean, a, [1, 2])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_over_axes mean should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vectorize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vectorize_exists() -> Result<(), String> {
    let script = fnp_script(
        r#"
# fnp.vectorize has a different API than np.vectorize
# Just verify it exists and is callable
print(callable(fnp.vectorize))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vectorize should be callable");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// select
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn select_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 3, x > 5]
choicelist = [x, x**2]
result = fnp.select(condlist, choicelist)
expected = np.select(condlist, choicelist)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select basic should match numpy");
    Ok(())
}

#[test]
fn select_with_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 3, x > 5]
choicelist = [x, x**2]
result = fnp.select(condlist, choicelist, default=-1)
expected = np.select(condlist, choicelist, default=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select with default should match numpy");
    Ok(())
}

#[test]
fn select_three_conditions() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 2, (x >= 2) & (x < 6), x >= 6]
choicelist = [0, 1, 2]
result = fnp.select(condlist, choicelist)
expected = np.select(condlist, choicelist)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select three conditions should match numpy");
    Ok(())
}

#[test]
fn select_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(12).reshape(3, 4)
condlist = [x < 3, x > 8]
choicelist = [x * 10, x * 100]
result = fnp.select(condlist, choicelist)
expected = np.select(condlist, choicelist)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn apply_along_axis_vs_direct_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
apply_result = fnp.apply_along_axis(np.sum, 0, a)
direct_result = fnp.sum(a, axis=0)
print(np.array_equal(apply_result, direct_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "apply_along_axis sum should equal direct sum");
    Ok(())
}

#[test]
fn select_vs_where() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.arange(10)
condlist = [x < 5]
choicelist = [x * 2]
select_result = fnp.select(condlist, choicelist, default=0)
where_result = fnp.where(x < 5, x * 2, 0)
print(np.array_equal(select_result, where_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "select should equal where for single condition");
    Ok(())
}
