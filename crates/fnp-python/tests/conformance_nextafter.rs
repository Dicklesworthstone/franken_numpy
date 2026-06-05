//! Conformance tests for numpy.nextafter against NumPy oracle.
//!
//! Tests nextafter (next representable floating-point value).

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
fn nextafter_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([2.0, 3.0, 4.0])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nextafter basic should match numpy");
    Ok(())
}

#[test]
fn nextafter_toward_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, -1.0, 0.5])
x2 = np.array([0.0, 0.0, 0.0])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nextafter toward zero should match numpy"
    );
    Ok(())
}

#[test]
fn nextafter_same_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([1.0, 2.0])
x2 = np.array([1.0, 2.0])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nextafter with same value should match numpy"
    );
    Ok(())
}

#[test]
fn nextafter_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(1.0)
x2 = np.float64(2.0)
fnp_result = fnp.nextafter(x1, x2)
np_result = np.nextafter(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "nextafter scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nextafter_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, np.nan, 0.0])
x2 = np.array([0.0, 0.0, 0.0, np.inf])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
# Check nan positions and other values
match = True
for r, e in zip(result.flat, expected.flat):
    if np.isnan(e):
        if not np.isnan(r):
            match = False
    elif r != e:
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nextafter special values should match numpy"
    );
    Ok(())
}

#[test]
fn nextafter_subnormal() -> Result<(), String> {
    let script = fnp_script(
        r#"
tiny = np.finfo(np.float64).tiny
x1 = np.array([tiny / 2, tiny / 4, 0.0])
x2 = np.array([0.0, 0.0, tiny / 2])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nextafter subnormal should match numpy"
    );
    Ok(())
}

#[test]
fn nextafter_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 1.0])
x2 = np.array([-0.0, 0.0, -np.inf])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
# Check both values and sign bits
value_match = np.array_equal(result, expected)
sign_match = np.array_equal(np.signbit(result), np.signbit(expected))
print(value_match and sign_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nextafter negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn nextafter_max_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
fmax = np.finfo(np.float64).max
x1 = np.array([fmax, -fmax])
x2 = np.array([np.inf, -np.inf])
result = fnp.nextafter(x1, x2)
expected = np.nextafter(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nextafter max values should match numpy"
    );
    Ok(())
}
