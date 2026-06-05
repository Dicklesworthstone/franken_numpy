//! Conformance tests for numpy.divmod against NumPy oracle.
//!
//! Tests the native Rust divmod implementation against NumPy.

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
// divmod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn divmod_float64_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([7.0, 8.0, 9.0, 10.0])
y = np.array([3.0, 3.0, 3.0, 3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod float64 basic should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_negative_dividend() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-7.0, -8.0, -9.0, -10.0])
y = np.array([3.0, 3.0, 3.0, 3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod negative dividend should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_negative_divisor() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([7.0, 8.0, 9.0, 10.0])
y = np.array([-3.0, -3.0, -3.0, -3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod negative divisor should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_both_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-7.0, -8.0, -9.0, -10.0])
y = np.array([-3.0, -3.0, -3.0, -3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod both negative should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_int64() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([7, 8, 9, 10], dtype=np.int64)
y = np.array([3, 3, 3, 3], dtype=np.int64)
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.array_equal(q, qe) and np.array_equal(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divmod int64 should match numpy");
    Ok(())
}

#[test]
fn divmod_broadcast() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([[7.0, 8.0], [9.0, 10.0]])
y = np.array([3.0, 4.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
print(np.allclose(q, qe) and np.allclose(r, re))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divmod broadcast should match numpy");
    Ok(())
}

#[test]
fn divmod_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([1.0, 0.0, np.inf, np.nan])
y = np.array([0.0, 0.0, 1.0, 1.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
# Check inf/nan handling element by element
def check_special(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return np.allclose([a], [b])
q_match = all(check_special(q[i], qe[i]) for i in range(len(q)))
r_match = all(check_special(r[i], re[i]) for i in range(len(r)))
print(q_match and r_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod special values should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_exact_division() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([6.0, 9.0, 12.0, 15.0])
y = np.array([3.0, 3.0, 3.0, 3.0])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
# Remainder should be 0 for exact division
print(np.allclose(q, qe) and np.allclose(r, re) and np.allclose(r, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod exact division should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_signed_zero_signbits() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([6.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0])
y = np.array([-3.0, -3.0, 3.0, -3.0, -np.inf, np.inf, -np.inf])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
value_match = (
    np.array_equal(q, qe, equal_nan=True) and
    np.array_equal(r, re, equal_nan=True)
)
sign_match = (
    np.array_equal(np.signbit(q), np.signbit(qe)) and
    np.array_equal(np.signbit(r), np.signbit(re))
)
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod signed zero signbits should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_negative_inf_divisor() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x = np.array([1.0, -1.0, 0.0])
y = np.array([-np.inf, -np.inf, -np.inf])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
def check_special(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b):
        return np.sign(a) == np.sign(b)
    return np.allclose([a], [b])
q_match = all(check_special(q[i], qe[i]) for i in range(len(q)))
r_match = all(check_special(r[i], re[i]) for i in range(len(r)))
print(q_match and r_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod negative inf divisor should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_large_numbers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e300, 1e200, 1e100])
y = np.array([1e100, 1e50, 1e10])
q, r = fnp.divmod(x, y)
qe, re = np.divmod(x, y)
# Use relative tolerance for large numbers
print(np.allclose(q, qe, rtol=1e-10) and np.allclose(r, re, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "divmod large numbers should match numpy"
    );
    Ok(())
}

#[test]
fn divmod_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; x = np.float64(7.0); y = np.float64(3.0); q, r = np.divmod(x, y); print(type(q).__name__, type(r).__name__, q, r)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("x = np.float64(7.0); y = np.float64(3.0); q, r = fnp.divmod(x, y); print(type(q).__name__, type(r).__name__, q, r)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "divmod scalar return type mismatch\nnumpy: {numpy_result}\nfnp: {rust_result}"
    );

    Ok(())
}
