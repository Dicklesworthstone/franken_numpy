//! Metamorphic tests for array operations.
//!
//! Tests mathematical properties that should hold regardless of input:
//! - transpose(transpose(x)) == x
//! - sort(reverse(x)) == sort(x)
//! - abs(abs(x)) == abs(x) (idempotent)
//! - negative(negative(x)) == x (involution)
//! - exp(log(x)) == x for x > 0
//! - log(exp(x)) == x
//! - sin(x)**2 + cos(x)**2 == 1
//! - arcsin(sin(x)) == x for x in [-pi/2, pi/2]

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
fn transpose_transpose_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(5, 7)
result = fnp.transpose(fnp.transpose(x))
print(np.allclose(x, result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "transpose(transpose(x)) should equal x");
    Ok(())
}

#[test]
fn transpose_transpose_3d_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(3, 4, 5)
result = fnp.transpose(fnp.transpose(x))
print(np.allclose(x, result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "transpose(transpose(x)) should equal x for 3D");
    Ok(())
}

#[test]
fn sort_reverse_equals_sort() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
result1 = fnp.sort(x[::-1].copy())
result2 = fnp.sort(x)
print(np.allclose(result1, result2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sort(reverse(x)) should equal sort(x)");
    Ok(())
}

#[test]
fn abs_is_idempotent() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100) * 100
result = fnp.abs(fnp.abs(x))
expected = fnp.abs(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "abs(abs(x)) should equal abs(x)");
    Ok(())
}

#[test]
fn negative_negative_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100) * 100
result = fnp.negative(fnp.negative(x))
print(np.allclose(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative(negative(x)) should equal x");
    Ok(())
}

#[test]
fn exp_log_roundtrip_for_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.1, 100, 100)
result = fnp.exp(fnp.log(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp(log(x)) should equal x for positive x");
    Ok(())
}

#[test]
fn log_exp_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
result = fnp.log(fnp.exp(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log(exp(x)) should equal x");
    Ok(())
}

#[test]
fn pythagorean_identity_sin_cos() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
result = fnp.sin(x)**2 + fnp.cos(x)**2
print(np.allclose(result, 1.0, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sin(x)**2 + cos(x)**2 should equal 1");
    Ok(())
}

#[test]
fn arcsin_sin_roundtrip_in_domain() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-np.pi/2 + 0.01, np.pi/2 - 0.01, 100)
result = fnp.arcsin(fnp.sin(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arcsin(sin(x)) should equal x for x in [-pi/2, pi/2]");
    Ok(())
}

#[test]
fn arccos_cos_roundtrip_in_domain() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, np.pi - 0.01, 100)
result = fnp.arccos(fnp.cos(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arccos(cos(x)) should equal x for x in [0, pi]");
    Ok(())
}

#[test]
fn arctan_tan_roundtrip_in_domain() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 100)
result = fnp.arctan(fnp.tan(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arctan(tan(x)) should equal x for x in (-pi/2, pi/2)");
    Ok(())
}

#[test]
fn sinh_cosh_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-5, 5, 100)
# cosh(x)**2 - sinh(x)**2 = 1
result = fnp.cosh(x)**2 - fnp.sinh(x)**2
print(np.allclose(result, 1.0, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cosh(x)**2 - sinh(x)**2 should equal 1");
    Ok(())
}

#[test]
fn square_sqrt_roundtrip_for_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, 100, 100)
result = fnp.sqrt(fnp.square(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt(square(x)) should equal x for positive x");
    Ok(())
}

#[test]
fn cbrt_power3_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-100, 100, 100)
result = fnp.cbrt(x**3)
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cbrt(x**3) should equal x");
    Ok(())
}

#[test]
fn maximum_minimum_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# maximum(x, y) + minimum(x, y) = x + y
max_vals = fnp.maximum(x, y)
min_vals = fnp.minimum(x, y)
result = max_vals + min_vals
expected = x + y
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "maximum(x,y) + minimum(x,y) should equal x + y");
    Ok(())
}

#[test]
fn fmax_fmin_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# fmax(x, y) + fmin(x, y) = x + y (for non-NaN)
max_vals = fnp.fmax(x, y)
min_vals = fnp.fmin(x, y)
result = max_vals + min_vals
expected = x + y
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fmax(x,y) + fmin(x,y) should equal x + y for non-NaN");
    Ok(())
}

#[test]
fn log2_exp2_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
result = fnp.log2(fnp.exp2(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log2(exp2(x)) should equal x");
    Ok(())
}

#[test]
fn log10_power10_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-5, 5, 100)
result = fnp.log10(10.0**x)
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log10(10**x) should equal x");
    Ok(())
}

#[test]
fn hypot_pythagorean() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# hypot(x, y)**2 = x**2 + y**2
result = fnp.hypot(x, y)**2
expected = x**2 + y**2
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hypot(x,y)**2 should equal x**2 + y**2");
    Ok(())
}

#[test]
fn arctan2_atan_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
# For positive x, arctan2(y, x) = arctan(y/x)
y = np.random.randn(100)
x = np.random.uniform(0.1, 10, 100)  # positive x only
result = fnp.arctan2(y, x)
expected = fnp.arctan(y / x)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "arctan2(y,x) should equal arctan(y/x) for positive x");
    Ok(())
}
