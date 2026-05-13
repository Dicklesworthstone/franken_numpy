//! Conformance tests for numpy complex number functions against NumPy oracle.
//!
//! Tests real, imag, conj, conjugate.

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
// real
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn real_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.real(z)
expected = np.real(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "real complex should match numpy");
    Ok(())
}

#[test]
fn real_real_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
result = fnp.real(a)
expected = np.real(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real of real array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// imag
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn imag_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.imag(z)
expected = np.imag(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "imag complex should match numpy");
    Ok(())
}

#[test]
fn imag_real_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
result = fnp.imag(a)
expected = np.imag(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "imag of real array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// conj / conjugate
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn conj_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.conj(z)
expected = np.conj(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "conj complex should match numpy");
    Ok(())
}

#[test]
fn conjugate_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.conjugate(z)
expected = np.conjugate(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "conjugate complex should match numpy"
    );
    Ok(())
}

#[test]
fn conj_real_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
result = fnp.conj(a)
expected = np.conj(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "conj of real array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn real_imag_reconstruct() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
re = fnp.real(z)
im = fnp.imag(z)
reconstructed = re + 1j * im
print(np.allclose(z, reconstructed))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real + imag*1j should reconstruct original"
    );
    Ok(())
}

#[test]
fn conj_conj_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.conj(fnp.conj(z))
print(np.allclose(z, result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "conj(conj(z)) should equal z");
    Ok(())
}

#[test]
fn conj_equals_conjugate() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
conj_result = fnp.conj(z)
conjugate_result = fnp.conjugate(z)
print(np.allclose(conj_result, conjugate_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "conj should equal conjugate");
    Ok(())
}
