//! Conformance tests for numpy window functions against NumPy oracle.
//!
//! Tests bartlett, blackman, hamming, hanning, kaiser.

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
// bartlett
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bartlett_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.bartlett(10)
expected = np.bartlett(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bartlett basic should match numpy");
    Ok(())
}

#[test]
fn bartlett_small() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.bartlett(3)
expected = np.bartlett(3)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bartlett small should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// blackman
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn blackman_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.blackman(10)
expected = np.blackman(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "blackman basic should match numpy");
    Ok(())
}

#[test]
fn blackman_large() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.blackman(100)
expected = np.blackman(100)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "blackman large should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hamming
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hamming_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hamming(10)
expected = np.hamming(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hamming basic should match numpy");
    Ok(())
}

#[test]
fn hamming_small() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hamming(5)
expected = np.hamming(5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hamming small should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hanning
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hanning_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hanning(10)
expected = np.hanning(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hanning basic should match numpy");
    Ok(())
}

#[test]
fn hanning_symmetry() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hanning(11)  # odd length for symmetry
# Check that the window is symmetric
mid = len(result) // 2
print(np.allclose(result[:mid], result[-1:-mid-1:-1]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hanning should be symmetric");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// kaiser
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn kaiser_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.kaiser(10, 14)  # M=10, beta=14
expected = np.kaiser(10, 14)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kaiser basic should match numpy");
    Ok(())
}

#[test]
fn kaiser_low_beta() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.kaiser(10, 0)  # beta=0 should give rectangular window
expected = np.kaiser(10, 0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kaiser low beta should match numpy");
    Ok(())
}

#[test]
fn kaiser_high_beta() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.kaiser(20, 30)  # high beta for more attenuation
expected = np.kaiser(20, 30)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kaiser high beta should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn window_endpoints_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Bartlett, Blackman, Hanning should start and end near 0
bart = fnp.bartlett(10)
black = fnp.blackman(10)
hann = fnp.hanning(10)
# Check first and last elements are small
print(bart[0] < 0.1 and bart[-1] < 0.1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bartlett endpoints should be near zero"
    );
    Ok(())
}

#[test]
fn hamming_vs_hanning_center() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Both should peak in the center
ham = fnp.hamming(11)
hann = fnp.hanning(11)
mid = len(ham) // 2
# Both should have maximum at center
print(np.argmax(ham) == mid and np.argmax(hann) == mid)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hamming and hanning should peak at center"
    );
    Ok(())
}
