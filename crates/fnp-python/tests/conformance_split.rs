//! Conformance tests for numpy split functions against NumPy oracle.
//!
//! Tests split, array_split, hsplit, vsplit, dsplit.

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
// split
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn split_equal_parts() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.split(a, 3)
expected = np.split(a, 3)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split equal parts should match numpy");
    Ok(())
}

#[test]
fn split_with_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.split(a, [2, 4])
expected = np.split(a, [2, 4])
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split with indices should match numpy");
    Ok(())
}

#[test]
fn split_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = fnp.split(a, 2, axis=0)
expected = np.split(a, 2, axis=0)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn split_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = fnp.split(a, 2, axis=1)
expected = np.split(a, 2, axis=1)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split 2d axis=1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_split
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_split_unequal_parts() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.array_split(a, 3)
expected = np.array_split(a, 3)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_split unequal parts should match numpy");
    Ok(())
}

#[test]
fn array_split_with_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7])
result = fnp.array_split(a, [2, 5])
expected = np.array_split(a, [2, 5])
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_split with indices should match numpy");
    Ok(())
}

#[test]
fn array_split_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.array_split(a, 2, axis=0)
expected = np.array_split(a, 2, axis=0)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_split 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hsplit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hsplit_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
result = fnp.hsplit(a, 3)
expected = np.hsplit(a, 3)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hsplit 1d should match numpy");
    Ok(())
}

#[test]
fn hsplit_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
result = fnp.hsplit(a, 2)
expected = np.hsplit(a, 2)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hsplit 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vsplit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vsplit_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
result = fnp.vsplit(a, 2)
expected = np.vsplit(a, 2)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vsplit basic should match numpy");
    Ok(())
}

#[test]
fn vsplit_unequal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.vsplit(a, [1, 2])
expected = np.vsplit(a, [1, 2])
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vsplit unequal should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dsplit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dsplit_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(16).reshape(2, 2, 4)
result = fnp.dsplit(a, 2)
expected = np.dsplit(a, 2)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dsplit basic should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn split_concatenate_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6])
splits = fnp.split(a, 3)
result = fnp.concatenate(splits)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "split then concatenate should be identity");
    Ok(())
}

#[test]
fn hsplit_hstack_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
splits = fnp.hsplit(a, 2)
result = fnp.hstack(splits)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hsplit then hstack should be identity");
    Ok(())
}

#[test]
fn vsplit_vstack_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
splits = fnp.vsplit(a, 2)
result = fnp.vstack(splits)
print(np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vsplit then vstack should be identity");
    Ok(())
}
