//! Conformance tests for numpy histogram and bincount functions against NumPy oracle.
//!
//! Tests histogram, histogram_bin_edges, bincount, digitize.

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
// histogram
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a)
np_hist, np_edges = np.histogram(a)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram basic should match numpy");
    Ok(())
}

#[test]
fn histogram_with_bins() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a, bins=5)
np_hist, np_edges = np.histogram(a, bins=5)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram with bins should match numpy");
    Ok(())
}

#[test]
fn histogram_with_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
hist, edges = fnp.histogram(a, bins=5, range=(2, 8))
np_hist, np_edges = np.histogram(a, bins=5, range=(2, 8))
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram with range should match numpy");
    Ok(())
}

#[test]
fn histogram_with_explicit_edges() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
bin_edges = np.array([1, 2, 3, 4, 5])
hist, edges = fnp.histogram(a, bins=bin_edges)
np_hist, np_edges = np.histogram(a, bins=bin_edges)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram with explicit edges should match numpy");
    Ok(())
}

#[test]
fn histogram_density() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a, density=True)
np_hist, np_edges = np.histogram(a, density=True)
print(np.allclose(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram density should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bincount
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bincount_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount basic should match numpy");
    Ok(())
}

#[test]
fn bincount_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
w = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
result = fnp.bincount(a, weights=w)
expected = np.bincount(a, weights=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount with weights should match numpy");
    Ok(())
}

#[test]
fn bincount_with_minlength() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2])
result = fnp.bincount(a, minlength=5)
expected = np.bincount(a, minlength=5)
print(np.array_equal(result, expected) and len(result) == 5)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount with minlength should match numpy");
    Ok(())
}

#[test]
fn bincount_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=int)
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount empty should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// digitize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn digitize_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 0.8, 1.5, 2.3, 3.8, 5.0])
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize basic should match numpy");
    Ok(())
}

#[test]
fn digitize_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4])
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins, right=True)
expected = np.digitize(x, bins, right=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize right should match numpy");
    Ok(())
}

#[test]
fn digitize_decreasing() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 0.8, 1.5, 2.3, 3.8, 5.0])
bins = np.array([4, 3, 2, 1])  # decreasing
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize decreasing bins should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, _ = fnp.histogram(a)
print(np.sum(hist) == len(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram sum should equal array length");
    Ok(())
}

#[test]
fn bincount_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = fnp.bincount(a)
print(np.sum(result) == len(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount sum should equal array length");
    Ok(())
}

#[test]
fn digitize_searchsorted_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 1.5, 2.8, 4.2])
bins = np.array([1, 2, 3, 4])
digitize_result = fnp.digitize(x, bins)
searchsorted_result = fnp.searchsorted(bins, x, side='right')
print(np.array_equal(digitize_result, searchsorted_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize should equal searchsorted for increasing bins");
    Ok(())
}
