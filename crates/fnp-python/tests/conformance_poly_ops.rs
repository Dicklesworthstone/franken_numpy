//! Conformance tests for numpy polynomial operations against NumPy oracle.
//!
//! Tests polyval, polyder, polyint, polyroots.

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
// polyval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyval_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # 1*x^2 + 2*x + 3
x = 2
result = fnp.polyval(p, x)
expected = np.polyval(p, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval basic should match numpy");
    Ok(())
}

#[test]
fn polyval_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, -2, 1]  # (x-1)^2
x = np.array([0, 1, 2, 3])
result = fnp.polyval(p, x)
expected = np.polyval(p, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval array should match numpy");
    Ok(())
}

#[test]
fn polyval_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1.5, -2.5, 0.5]
x = np.array([0.5, 1.5, 2.5])
result = fnp.polyval(p, x)
expected = np.polyval(p, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polyder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyder_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # x^2 + 2x + 3 -> 2x + 2
result = fnp.polyder(p)
expected = np.polyder(p)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyder basic should match numpy");
    Ok(())
}

#[test]
fn polyder_with_m() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3, 4]  # x^3 + 2x^2 + 3x + 4
result = fnp.polyder(p, m=2)  # second derivative
expected = np.polyder(p, m=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyder with m should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polyint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyint_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [2, 2]  # 2x + 2 -> x^2 + 2x + C
result = fnp.polyint(p)
expected = np.polyint(p)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyint basic should match numpy");
    Ok(())
}

#[test]
fn polyint_with_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [2, 2]
result = fnp.polyint(p, k=5)  # constant of integration
expected = np.polyint(p, k=5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyint with k should match numpy");
    Ok(())
}

#[test]
fn polyint_with_m() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [6]  # 6 -> 3x^2 -> x^3 (double integration)
result = fnp.polyint(p, m=2)
expected = np.polyint(p, m=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyint with m should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyder_polyint_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # x^2 + 2x + 3
integrated = fnp.polyint(p)
derived = fnp.polyder(integrated)
# Should get back original (up to precision)
print(np.allclose(derived, p))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyder(polyint(p)) should equal p");
    Ok(())
}

#[test]
fn polyval_at_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # x^2 + 2x + 3
# At x=0, should equal constant term
result = fnp.polyval(p, 0)
print(np.allclose(result, 3))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval at 0 should equal constant term");
    Ok(())
}
