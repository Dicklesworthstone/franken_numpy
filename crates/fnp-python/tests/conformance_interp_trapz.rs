//! Conformance tests for numpy interp and trapz functions against NumPy oracle.
//!
//! Tests interp, trapz.

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
// interp
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn interp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3]
fp = [3, 2, 0]
result = fnp.interp(2.5, xp, fp)
expected = np.interp(2.5, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp basic should match numpy");
    Ok(())
}

#[test]
fn interp_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3, 4, 5]
fp = [10, 20, 30, 40, 50]
x = [1.5, 2.5, 3.5]
result = fnp.interp(x, xp, fp)
expected = np.interp(x, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp array should match numpy");
    Ok(())
}

#[test]
fn interp_outside_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3]
fp = [10, 20, 30]
x = [0, 4]  # outside bounds
result = fnp.interp(x, xp, fp)
expected = np.interp(x, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp outside bounds should match numpy");
    Ok(())
}

#[test]
fn interp_with_left_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3]
fp = [10, 20, 30]
x = [0, 4]
result = fnp.interp(x, xp, fp, left=-1, right=-1)
expected = np.interp(x, xp, fp, left=-1, right=-1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp with left/right should match numpy");
    Ok(())
}

#[test]
fn interp_single_point() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [0, 1, 2, 3, 4]
fp = [0, 1, 4, 9, 16]
result = fnp.interp(1.5, xp, fp)
expected = np.interp(1.5, xp, fp)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp single point should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// trapz
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn trapz_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = [1, 2, 3, 4]
result = fnp.trapz(y)
expected = np.trapz(y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz basic should match numpy");
    Ok(())
}

#[test]
fn trapz_with_x() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = [1, 2, 3, 4]
x = [0, 1, 2, 3]
result = fnp.trapz(y, x)
expected = np.trapz(y, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz with x should match numpy");
    Ok(())
}

#[test]
fn trapz_with_dx() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = [1, 2, 3, 4]
result = fnp.trapz(y, dx=0.5)
expected = np.trapz(y, dx=0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz with dx should match numpy");
    Ok(())
}

#[test]
fn trapz_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.trapz(y, axis=0)
expected = np.trapz(y, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn trapz_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.trapz(y, axis=1)
expected = np.trapz(y, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn trapz_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.array([0.0, 1.0, 1.0, 0.0])
result = fnp.trapz(y)
expected = np.trapz(y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn interp_exact_points() -> Result<(), String> {
    let script = fnp_script(
        r#"
xp = [1, 2, 3, 4, 5]
fp = [10, 20, 30, 40, 50]
# Interpolating at exact points should return exact values
result = fnp.interp(xp, xp, fp)
print(np.allclose(result, fp))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "interp at exact points should return exact values");
    Ok(())
}

#[test]
fn trapz_rectangle_integration() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Trapezoidal rule on constant function should give width * height
y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # constant 5
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # width 4
result = fnp.trapz(y, x)
# Should be 5 * 4 = 20
print(np.allclose(result, 20.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trapz of constant should equal width * height");
    Ok(())
}
