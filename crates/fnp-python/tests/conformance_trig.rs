//! Conformance tests for numpy trigonometric functions against NumPy oracle.
//!
//! Tests sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh.

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
// sin
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi])
result = fnp.sin(x)
expected = np.sin(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sin basic should match numpy");
    Ok(())
}

#[test]
fn sin_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
result = fnp.sin(x)
expected = np.sin(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sin negative should match numpy");
    Ok(())
}

#[test]
fn sin_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([10*np.pi, 100*np.pi, 1000])
result = fnp.sin(x)
expected = np.sin(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sin large values should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cos
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cos_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi])
result = fnp.cos(x)
expected = np.cos(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cos basic should match numpy");
    Ok(())
}

#[test]
fn cos_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
result = fnp.cos(x)
expected = np.cos(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cos negative should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tan_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, np.pi/6, np.pi/4, np.pi/3])
result = fnp.tan(x)
expected = np.tan(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tan basic should match numpy");
    Ok(())
}

#[test]
fn tan_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.pi/4, -np.pi/6, 0, np.pi/6, np.pi/4])
result = fnp.tan(x)
expected = np.tan(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tan negative should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arcsin / asin
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arcsin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1, -0.5, 0, 0.5, 1])
result = fnp.arcsin(x)
expected = np.arcsin(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arcsin basic should match numpy");
    Ok(())
}

#[test]
fn asin_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1, 0, 1])
result = fnp.asin(x)
expected = np.arcsin(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asin should match arcsin");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arccos / acos
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arccos_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1, -0.5, 0, 0.5, 1])
result = fnp.arccos(x)
expected = np.arccos(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arccos basic should match numpy");
    Ok(())
}

#[test]
fn acos_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1, 0, 1])
result = fnp.acos(x)
expected = np.arccos(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "acos should match arccos");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arctan / atan
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arctan_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-10, -1, 0, 1, 10])
result = fnp.arctan(x)
expected = np.arctan(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arctan basic should match numpy");
    Ok(())
}

#[test]
fn atan_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1, 0, 1])
result = fnp.atan(x)
expected = np.arctan(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atan should match arctan");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// sinh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sinh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-2, -1, 0, 1, 2])
result = fnp.sinh(x)
expected = np.sinh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sinh basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cosh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cosh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-2, -1, 0, 1, 2])
result = fnp.cosh(x)
expected = np.cosh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cosh basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tanh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tanh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-2, -1, 0, 1, 2])
result = fnp.tanh(x)
expected = np.tanh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tanh basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arcsinh / asinh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arcsinh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-10, -1, 0, 1, 10])
result = fnp.arcsinh(x)
expected = np.arcsinh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arcsinh basic should match numpy");
    Ok(())
}

#[test]
fn asinh_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1, 0, 1])
result = fnp.asinh(x)
expected = np.arcsinh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "asinh should match arcsinh");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arccosh / acosh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arccosh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 5, 10])  # domain: x >= 1
result = fnp.arccosh(x)
expected = np.arccosh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arccosh basic should match numpy");
    Ok(())
}

#[test]
fn acosh_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
result = fnp.acosh(x)
expected = np.arccosh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "acosh should match arccosh");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// arctanh / atanh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn arctanh_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-0.9, -0.5, 0, 0.5, 0.9])  # domain: -1 < x < 1
result = fnp.arctanh(x)
expected = np.arctanh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arctanh basic should match numpy");
    Ok(())
}

#[test]
fn atanh_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-0.5, 0, 0.5])
result = fnp.atanh(x)
expected = np.arctanh(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "atanh should match arctanh");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sin_cos_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.linspace(0, 2*np.pi, 100)
sin_sq = fnp.sin(x) ** 2
cos_sq = fnp.cos(x) ** 2
# sin^2 + cos^2 = 1
print(np.allclose(sin_sq + cos_sq, 1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sin^2 + cos^2 should equal 1");
    Ok(())
}

#[test]
fn arcsin_sin_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.linspace(-1, 1, 50)
# arcsin(sin(arcsin(x))) = arcsin(x) for x in [-1, 1]
result = fnp.arcsin(fnp.sin(fnp.arcsin(x)))
expected = fnp.arcsin(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arcsin(sin(arcsin(x))) should equal arcsin(x)"
    );
    Ok(())
}

#[test]
fn sinh_cosh_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.linspace(-3, 3, 50)
cosh_sq = fnp.cosh(x) ** 2
sinh_sq = fnp.sinh(x) ** 2
# cosh^2 - sinh^2 = 1
print(np.allclose(cosh_sq - sinh_sq, 1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cosh^2 - sinh^2 should equal 1");
    Ok(())
}
