//! Conformance tests for numpy.lib.stride_tricks functions against NumPy oracle.
//!
//! Tests sliding_window_view, as_strided.

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
// sliding_window_view
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sliding_window_view_1d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.sliding_window_view(a, 3)
expected = sliding_window_view(a, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view 1d basic should match numpy");
    Ok(())
}

#[test]
fn sliding_window_view_1d_window_size_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.array([1, 2, 3, 4, 5])
result = fnp.sliding_window_view(a, 1)
expected = sliding_window_view(a, 1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view window_size=1 should match numpy");
    Ok(())
}

#[test]
fn sliding_window_view_1d_full_window() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.array([1, 2, 3, 4])
result = fnp.sliding_window_view(a, 4)
expected = sliding_window_view(a, 4)
print(np.array_equal(result, expected) and result.shape == (1, 4))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view full window should match numpy");
    Ok(())
}

#[test]
fn sliding_window_view_2d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.arange(12).reshape(3, 4)
result = fnp.sliding_window_view(a, (2, 2))
expected = sliding_window_view(a, (2, 2))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view 2d basic should match numpy");
    Ok(())
}

#[test]
fn sliding_window_view_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.arange(12).reshape(3, 4)
result = fnp.sliding_window_view(a, 2, axis=0)
expected = sliding_window_view(a, 2, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view 2d with axis should match numpy");
    Ok(())
}

#[test]
fn sliding_window_view_shape_preservation() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.arange(20).reshape(4, 5)
result = fnp.sliding_window_view(a, (2, 3))
expected = sliding_window_view(a, (2, 3))
print(result.shape == expected.shape == (3, 3, 2, 3))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view shape should match numpy");
    Ok(())
}

#[test]
fn sliding_window_view_dtype_preserved() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import sliding_window_view
a = np.array([1.5, 2.5, 3.5, 4.5], dtype='float32')
result = fnp.sliding_window_view(a, 2)
expected = sliding_window_view(a, 2)
print(result.dtype == expected.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sliding_window_view should preserve dtype");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// as_strided
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn as_strided_simple_view() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import as_strided
a = np.arange(10)
result = fnp.as_strided(a, shape=(5,), strides=(a.strides[0] * 2,))
expected = as_strided(a, shape=(5,), strides=(a.strides[0] * 2,))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "as_strided simple view should match numpy");
    Ok(())
}

#[test]
fn as_strided_overlapping_windows() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import as_strided
a = np.arange(6)
result = fnp.as_strided(a, shape=(4, 3), strides=(a.strides[0], a.strides[0]))
expected = as_strided(a, shape=(4, 3), strides=(a.strides[0], a.strides[0]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "as_strided overlapping windows should match numpy");
    Ok(())
}

#[test]
fn as_strided_reshape_equivalent() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import as_strided
a = np.arange(12).reshape(3, 4)
result = fnp.as_strided(a, shape=(4, 3), strides=(a.strides[1], a.strides[0]))
expected = as_strided(a, shape=(4, 3), strides=(a.strides[1], a.strides[0]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "as_strided reshape equivalent should match numpy");
    Ok(())
}

#[test]
fn as_strided_broadcast_like() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import as_strided
a = np.array([1, 2, 3])
result = fnp.as_strided(a, shape=(3, 3), strides=(0, a.strides[0]))
expected = as_strided(a, shape=(3, 3), strides=(0, a.strides[0]))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "as_strided broadcast-like should match numpy");
    Ok(())
}

#[test]
fn as_strided_no_args_returns_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import as_strided
a = np.arange(6).reshape(2, 3)
result = fnp.as_strided(a)
expected = as_strided(a)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "as_strided with no args should return equivalent view");
    Ok(())
}

#[test]
fn as_strided_dtype_preserved() -> Result<(), String> {
    let script = fnp_script(
        r#"
from numpy.lib.stride_tricks import as_strided
a = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype='float64')
result = fnp.as_strided(a, shape=(3,), strides=(a.strides[0] * 2,))
expected = as_strided(a, shape=(3,), strides=(a.strides[0] * 2,))
print(result.dtype == expected.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "as_strided should preserve dtype");
    Ok(())
}
