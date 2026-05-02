//! Conformance tests for numpy take and put functions against NumPy oracle.
//!
//! Tests take, take_along_axis, put_along_axis.

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
// take
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([4, 3, 5, 7, 6, 8])
indices = np.array([0, 1, 4])
result = fnp.take(a, indices)
expected = np.take(a, indices)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take basic should match numpy");
    Ok(())
}

#[test]
fn take_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.take(a, [0, 2], axis=0)
expected = np.take(a, [0, 2], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take 2d should match numpy");
    Ok(())
}

#[test]
fn take_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.take(a, [0, 2], axis=1)
expected = np.take(a, [0, 2], axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take axis=1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// take_along_axis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_along_axis_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[10, 20, 30], [40, 50, 60]])
ai = np.array([[0], [2]])
result = fnp.take_along_axis(a, ai, axis=1)
expected = np.take_along_axis(a, ai, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take_along_axis basic should match numpy");
    Ok(())
}

#[test]
fn take_along_axis_argsort() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 2], [6, 4, 5]])
ai = np.argsort(a, axis=1)
result = fnp.take_along_axis(a, ai, axis=1)
expected = np.take_along_axis(a, ai, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take_along_axis argsort should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// put_along_axis
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn put_along_axis_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[10, 20, 30], [40, 50, 60]])
ai = np.array([[0], [2]])
fnp.put_along_axis(a, ai, 99, axis=1)
b = np.array([[10, 20, 30], [40, 50, 60]])
np.put_along_axis(b, ai, 99, axis=1)
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "put_along_axis basic should match numpy");
    Ok(())
}

#[test]
fn put_along_axis_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[10, 20, 30], [40, 50, 60]])
ai = np.array([[0, 1], [1, 2]])
values = np.array([[1, 2], [3, 4]])
fnp.put_along_axis(a, ai, values, axis=1)
b = np.array([[10, 20, 30], [40, 50, 60]])
np.put_along_axis(b, ai, values, axis=1)
print(np.array_equal(a, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "put_along_axis values should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_equals_indexing() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([4, 3, 5, 7, 6, 8])
indices = np.array([0, 1, 4])
take_result = fnp.take(a, indices)
index_result = a[indices]
print(np.array_equal(take_result, index_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take should equal direct indexing");
    Ok(())
}

#[test]
fn take_along_axis_sorts() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 1, 2], [6, 4, 5]])
ai = np.argsort(a, axis=1)
sorted_a = fnp.take_along_axis(a, ai, axis=1)
# Result should be sorted along axis 1
print(np.all(sorted_a[:, :-1] <= sorted_a[:, 1:]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "take_along_axis with argsort should sort");
    Ok(())
}
