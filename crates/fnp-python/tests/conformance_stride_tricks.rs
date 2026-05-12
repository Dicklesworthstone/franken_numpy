//! Conformance tests for numpy.lib.stride_tricks functions against NumPy oracle.
//!
//! Tests sliding_window_view, as_strided.

mod common;

use common::with_fnp_and_numpy;
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("Python oracle stdin should be available\nScript: {script}"))?;
    stdin
        .write_all(script.as_bytes())
        .map_err(|error| format!("Python oracle stdin write should succeed: {error}"))?;
    drop(stdin);
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Python oracle should finish: {error}\nScript: {script}"))?;
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view 1d basic should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view window_size=1 should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view full window should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view 2d basic should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view 2d with axis should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view shape should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "sliding_window_view should preserve dtype"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "as_strided simple view should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "as_strided overlapping windows should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "as_strided reshape equivalent should match numpy"
    );
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
    assert_eq!(
        result.trim(),
        "True",
        "as_strided broadcast-like should match numpy"
    );
    Ok(())
}

#[test]
fn nditer_walks_broadcast_stride_trick_views_like_numpy() {
    with_fnp_and_numpy(|py, module, numpy| {
        let globals = PyDict::new(py);
        globals.set_item("np", numpy.clone())?;
        globals.set_item("fnp", module.clone())?;
        py.import("builtins")?.getattr("exec")?.call(
            (
                r#"
def trace(view):
    iterator = np.nditer(view, flags=["multi_index"], order="C")
    return [(iterator.multi_index, int(value)) for value in iterator]

base = np.arange(3, dtype=np.int64)
ours_broadcast = fnp.broadcast_to(base, (2, 3))
expected_broadcast = np.broadcast_to(base, (2, 3))
ours_strided = fnp.as_strided(base, shape=(2, 3), strides=(0, base.strides[0]))
expected_strided = np.lib.stride_tricks.as_strided(
    base,
    shape=(2, 3),
    strides=(0, base.strides[0]),
)

checks = [
    ("broadcast shape", ours_broadcast.shape == expected_broadcast.shape),
    ("broadcast strides", ours_broadcast.strides == expected_broadcast.strides),
    (
        "broadcast readonly",
        ours_broadcast.flags.writeable == expected_broadcast.flags.writeable == False,
    ),
    (
        "broadcast memory sharing",
        np.shares_memory(ours_broadcast, base)
        == np.shares_memory(expected_broadcast, base),
    ),
    ("broadcast nditer trace", trace(ours_broadcast) == trace(expected_broadcast)),
    ("as_strided shape", ours_strided.shape == expected_strided.shape),
    ("as_strided strides", ours_strided.strides == expected_strided.strides),
    (
        "as_strided memory sharing",
        np.shares_memory(ours_strided, base) == np.shares_memory(expected_strided, base),
    ),
    ("as_strided nditer trace", trace(ours_strided) == trace(expected_strided)),
]
detail = [name for name, passed in checks if not passed]
ok = not detail
"#,
                &globals,
            ),
            None::<&Bound<'_, PyDict>>,
        )?;
        let Some(ok_obj) = globals.get_item("ok")? else {
            return Err(PyAssertionError::new_err(
                "nditer parity script did not set ok",
            ));
        };
        let ok = ok_obj.extract::<bool>()?;
        let detail = globals
            .get_item("detail")?
            .map(|value| value.str().map(|s| s.to_string()))
            .transpose()?
            .unwrap_or_else(|| "unknown nditer parity mismatch".to_string());
        assert!(ok, "{detail}");
        Ok(())
    });
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
    assert_eq!(
        result.trim(),
        "True",
        "as_strided with no args should return equivalent view"
    );
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
