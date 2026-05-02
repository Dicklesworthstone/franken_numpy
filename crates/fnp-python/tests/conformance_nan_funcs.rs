//! Conformance tests for numpy nan-ignoring functions against NumPy oracle.
//!
//! Tests nansum, nanmean, nanstd, nanvar, nanmin, nanmax, nanargmin, nanargmax,
//! nanprod, nancumsum, nancumprod, nanmedian, nanpercentile, nanquantile.

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
// nansum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nansum_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4])
result = fnp.nansum(a)
expected = np.nansum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nansum basic should match numpy");
    Ok(())
}

#[test]
fn nansum_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan], [3, 4]])
result = fnp.nansum(a, axis=0)
expected = np.nansum(a, axis=0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nansum 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanmean
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmean_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4])
result = fnp.nanmean(a)
expected = np.nanmean(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean basic should match numpy");
    Ok(())
}

#[test]
fn nanmean_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = fnp.nanmean(a, axis=1)
expected = np.nanmean(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanstd / nanvar
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanstd_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
result = fnp.nanstd(a)
expected = np.nanstd(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanstd basic should match numpy");
    Ok(())
}

#[test]
fn nanvar_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
result = fnp.nanvar(a)
expected = np.nanvar(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanvar basic should match numpy");
    Ok(())
}

#[test]
fn nanstd_ddof() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
result = fnp.nanstd(a, ddof=1)
expected = np.nanstd(a, ddof=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanstd ddof should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanmin / nanmax
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, np.nan, 5])
result = fnp.nanmin(a)
expected = np.nanmin(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmin basic should match numpy");
    Ok(())
}

#[test]
fn nanmax_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, np.nan, 5])
result = fnp.nanmax(a)
expected = np.nanmax(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmax basic should match numpy");
    Ok(())
}

#[test]
fn nanmin_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan, 3], [np.nan, 5, 6]])
result = fnp.nanmin(a, axis=1)
expected = np.nanmin(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmin 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanargmin / nanargmax
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanargmin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.nan, 2, 1, np.nan, 5])
result = fnp.nanargmin(a)
expected = np.nanargmin(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanargmin basic should match numpy");
    Ok(())
}

#[test]
fn nanargmax_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.nan, 2, 5, np.nan, 1])
result = fnp.nanargmax(a)
expected = np.nanargmax(a)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanargmax basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanprod_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4])
result = fnp.nanprod(a)
expected = np.nanprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanprod basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nancumsum / nancumprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nancumsum_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4])
result = fnp.nancumsum(a)
expected = np.nancumsum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nancumsum basic should match numpy");
    Ok(())
}

#[test]
fn nancumprod_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4])
result = fnp.nancumprod(a)
expected = np.nancumprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nancumprod basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanmedian
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmedian_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, np.nan])
result = fnp.nanmedian(a)
expected = np.nanmedian(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmedian basic should match numpy");
    Ok(())
}

#[test]
fn nanmedian_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, np.nan, 3], [4, 5, np.nan]])
result = fnp.nanmedian(a, axis=1)
expected = np.nanmedian(a, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmedian 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// nanpercentile / nanquantile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanpercentile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, 5])
result = fnp.nanpercentile(a, 50)
expected = np.nanpercentile(a, 50)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanpercentile basic should match numpy");
    Ok(())
}

#[test]
fn nanquantile_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, 5])
result = fnp.nanquantile(a, 0.5)
expected = np.nanquantile(a, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanquantile basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nanmean_no_nan_equals_mean() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])  # no NaN
nanmean_result = fnp.nanmean(a)
mean_result = np.mean(a)
print(np.allclose(nanmean_result, mean_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanmean without NaN should equal mean");
    Ok(())
}

#[test]
fn nanstd_squared_equals_nanvar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, np.nan, 4, 5])
std = fnp.nanstd(a)
var = fnp.nanvar(a)
print(np.allclose(std**2, var))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanstd squared should equal nanvar");
    Ok(())
}

#[test]
fn nanpercentile_50_equals_nanmedian() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, np.nan, 3, 4, 5])
percentile = fnp.nanpercentile(a, 50)
median = fnp.nanmedian(a)
print(np.allclose(percentile, median))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "nanpercentile 50 should equal nanmedian");
    Ok(())
}
