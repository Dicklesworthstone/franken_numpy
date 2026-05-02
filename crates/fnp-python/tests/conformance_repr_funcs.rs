//! Conformance tests for numpy representation functions against NumPy oracle.
//!
//! Tests base_repr, binary_repr, format_float_positional, format_float_scientific.

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
// base_repr
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn base_repr_binary() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.base_repr(10, 2)
expected = np.base_repr(10, 2)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "base_repr binary should match numpy");
    Ok(())
}

#[test]
fn base_repr_hex() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.base_repr(255, 16)
expected = np.base_repr(255, 16)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "base_repr hex should match numpy");
    Ok(())
}

#[test]
fn base_repr_octal() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.base_repr(64, 8)
expected = np.base_repr(64, 8)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "base_repr octal should match numpy");
    Ok(())
}

#[test]
fn base_repr_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.base_repr(-10, 2)
expected = np.base_repr(-10, 2)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "base_repr negative should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// binary_repr
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn binary_repr_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.binary_repr(10)
expected = np.binary_repr(10)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "binary_repr basic should match numpy");
    Ok(())
}

#[test]
fn binary_repr_width() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.binary_repr(10, width=8)
expected = np.binary_repr(10, width=8)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "binary_repr width should match numpy");
    Ok(())
}

#[test]
fn binary_repr_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.binary_repr(-10, width=8)
expected = np.binary_repr(-10, width=8)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "binary_repr negative should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// format_float_positional
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn format_float_positional_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.format_float_positional(np.float64(1.23))
expected = np.format_float_positional(np.float64(1.23))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "format_float_positional basic should match numpy");
    Ok(())
}

#[test]
fn format_float_positional_precision() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.format_float_positional(np.float64(1.23456789), precision=4)
expected = np.format_float_positional(np.float64(1.23456789), precision=4)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "format_float_positional precision should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// format_float_scientific
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn format_float_scientific_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.format_float_scientific(np.float64(12345.6789))
expected = np.format_float_scientific(np.float64(12345.6789))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "format_float_scientific basic should match numpy");
    Ok(())
}

#[test]
fn format_float_scientific_precision() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.format_float_scientific(np.float64(12345.6789), precision=4)
expected = np.format_float_scientific(np.float64(12345.6789), precision=4)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "format_float_scientific precision should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn base_repr_2_equals_binary_repr() -> Result<(), String> {
    let script = fnp_script(
        r#"
base2 = fnp.base_repr(10, 2)
binary = fnp.binary_repr(10)
print(base2 == binary)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "base_repr base 2 should equal binary_repr");
    Ok(())
}

#[test]
fn format_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
val = np.float64(123.456)
formatted = fnp.format_float_positional(val)
# The formatted string should be parseable back
print(float(formatted) == val or abs(float(formatted) - val) < 1e-10)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "formatted float should round-trip");
    Ok(())
}
