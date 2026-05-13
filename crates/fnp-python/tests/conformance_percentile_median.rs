//! Conformance tests for numpy percentile, quantile, median, ptp against NumPy oracle.
//!
//! Tests percentile, quantile, median, ptp.

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
// percentile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.percentile(a, 50)
expected = np.percentile(a, 50)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile basic should match numpy");
    Ok(())
}

#[test]
fn percentile_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.percentile(a, [25, 50, 75])
expected = np.percentile(a, [25, 50, 75])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile multiple should match numpy"
    );
    Ok(())
}

#[test]
fn percentile_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.percentile(a, 50, axis=0)
expected = np.percentile(a, 50, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "percentile 2d axis should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// quantile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn quantile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.quantile(a, 0.5)
expected = np.quantile(a, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile basic should match numpy");
    Ok(())
}

#[test]
fn quantile_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = fnp.quantile(a, [0.25, 0.5, 0.75])
expected = np.quantile(a, [0.25, 0.5, 0.75])
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "quantile multiple should match numpy"
    );
    Ok(())
}

#[test]
fn quantile_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.quantile(a, 0.5, axis=1)
expected = np.quantile(a, 0.5, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "quantile 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// median
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn median_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 2, 5, 4])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "median basic should match numpy");
    Ok(())
}

#[test]
fn median_even_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.median(a)
expected = np.median(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "median even count should match numpy"
    );
    Ok(())
}

#[test]
fn median_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.median(a, axis=1)
expected = np.median(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "median 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ptp (peak-to-peak)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ptp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.ptp(a)
expected = np.ptp(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp basic should match numpy");
    Ok(())
}

#[test]
fn ptp_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 5, 3], [2, 8, 1]])
result = fnp.ptp(a, axis=1)
expected = np.ptp(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp 2d axis should match numpy");
    Ok(())
}

#[test]
fn ptp_2d_all() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 5, 3], [2, 8, 1]])
result = fnp.ptp(a)
expected = np.ptp(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp 2d all should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn percentile_50_equals_median() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 3, 2, 5, 4])
p50 = fnp.percentile(a, 50)
med = fnp.median(a)
print(np.allclose(p50, med))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "percentile 50 should equal median");
    Ok(())
}

#[test]
fn quantile_05_equals_percentile_50() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = fnp.quantile(a, 0.5)
p = fnp.percentile(a, 50)
print(np.allclose(q, p))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "quantile 0.5 should equal percentile 50"
    );
    Ok(())
}

#[test]
fn ptp_equals_max_minus_min() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
ptp_val = fnp.ptp(a)
manual = np.max(a) - np.min(a)
print(np.allclose(ptp_val, manual))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "ptp should equal max - min");
    Ok(())
}
