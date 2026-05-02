//! Conformance tests for numpy moveaxis and pad functions against NumPy oracle.
//!
//! Tests moveaxis, pad.

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
// moveaxis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn moveaxis_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, 0, -1)
expected = np.moveaxis(a, 0, -1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis basic should match numpy");
    Ok(())
}

#[test]
fn moveaxis_single_to_single() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, 2, 0)
expected = np.moveaxis(a, 2, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis single to single should match numpy");
    Ok(())
}

#[test]
fn moveaxis_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, [0, 1], [-1, -2])
expected = np.moveaxis(a, [0, 1], [-1, -2])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis multiple should match numpy");
    Ok(())
}

#[test]
fn moveaxis_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
result = fnp.moveaxis(a, -1, 0)
expected = np.moveaxis(a, -1, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis negative should match numpy");
    Ok(())
}

#[test]
fn moveaxis_4d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(120).reshape(2, 3, 4, 5)
result = fnp.moveaxis(a, 1, 3)
expected = np.moveaxis(a, 1, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis 4d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// pad
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pad_constant_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='constant')
expected = np.pad(a, 2, mode='constant')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad constant 1d should match numpy");
    Ok(())
}

#[test]
fn pad_constant_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.pad(a, 2, mode='constant', constant_values=99)
expected = np.pad(a, 2, mode='constant', constant_values=99)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad constant value should match numpy");
    Ok(())
}

#[test]
fn pad_edge() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='edge')
expected = np.pad(a, 2, mode='edge')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad edge should match numpy");
    Ok(())
}

#[test]
fn pad_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.pad(a, 1, mode='constant')
expected = np.pad(a, 1, mode='constant')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad 2d should match numpy");
    Ok(())
}

#[test]
fn pad_asymmetric() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.pad(a, (1, 3), mode='constant')
expected = np.pad(a, (1, 3), mode='constant')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad asymmetric should match numpy");
    Ok(())
}

#[test]
fn pad_reflect() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='reflect')
expected = np.pad(a, 2, mode='reflect')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad reflect should match numpy");
    Ok(())
}

#[test]
fn pad_wrap() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.pad(a, 2, mode='wrap')
expected = np.pad(a, 2, mode='wrap')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad wrap should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn moveaxis_vs_transpose() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
moveaxis_result = fnp.moveaxis(a, 0, 1)
transpose_result = fnp.transpose(a)
print(np.array_equal(moveaxis_result, transpose_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "moveaxis should equal transpose for 2d swap");
    Ok(())
}

#[test]
fn pad_preserves_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3], dtype='float32')
result = fnp.pad(a, 1, mode='constant')
print(result.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "pad should preserve dtype");
    Ok(())
}
