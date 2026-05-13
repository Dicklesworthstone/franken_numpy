//! Conformance tests for numpy array utility functions against NumPy oracle.
//!
//! Tests array_equal, array_equiv, around, angle, copy, ascontiguousarray, asfortranarray.

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
// array_equal
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
result = fnp.array_equal(a, b)
expected = np.array_equal(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_equal same should match numpy");
    Ok(())
}

#[test]
fn array_equal_different() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
result = fnp.array_equal(a, b)
expected = np.array_equal(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_equal different should match numpy"
    );
    Ok(())
}

#[test]
fn array_equal_different_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3]])
result = fnp.array_equal(a, b)
expected = np.array_equal(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_equal different shape should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// array_equiv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equiv_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
result = fnp.array_equiv(a, b)
expected = np.array_equiv(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "array_equiv same should match numpy");
    Ok(())
}

#[test]
fn array_equiv_broadcastable() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 1, 1])
b = np.array([[1, 1, 1], [1, 1, 1]])
result = fnp.array_equiv(a, b)
expected = np.array_equiv(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_equiv broadcastable should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// around (round)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn around_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.567, 2.345, 3.789])
result = fnp.around(a)
expected = np.around(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around basic should match numpy");
    Ok(())
}

#[test]
fn around_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.567, 2.345, 3.789])
result = fnp.around(a, decimals=2)
expected = np.around(a, decimals=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around decimals should match numpy");
    Ok(())
}

#[test]
fn around_negative_decimals() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1567, 2345, 3789])
result = fnp.around(a, decimals=-2)
expected = np.around(a, decimals=-2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around negative decimals should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// angle
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn angle_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 1+0j, 1-1j, 0+1j])
result = fnp.angle(z)
expected = np.angle(z)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "angle basic should match numpy");
    Ok(())
}

#[test]
fn angle_deg() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 1+0j, 1-1j, 0+1j])
result = fnp.angle(z, deg=True)
expected = np.angle(z, deg=True)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "angle deg should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// copy
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn copy_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.copy(a)
# Check values match and it's actually a copy
print(np.array_equal(result, a) and not np.shares_memory(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "copy basic should match numpy");
    Ok(())
}

#[test]
fn copy_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.copy(a)
print(np.array_equal(result, a) and not np.shares_memory(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "copy 2d should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ascontiguousarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ascontiguousarray_already_contiguous() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4])
result = fnp.ascontiguousarray(a)
expected = np.ascontiguousarray(a)
print(np.array_equal(result, expected) and result.flags['C_CONTIGUOUS'])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ascontiguousarray already contiguous should match numpy"
    );
    Ok(())
}

#[test]
fn ascontiguousarray_from_fortran() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.asfortranarray([[1, 2], [3, 4]])
result = fnp.ascontiguousarray(a)
expected = np.ascontiguousarray(a)
print(np.array_equal(result, expected) and result.flags['C_CONTIGUOUS'])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "ascontiguousarray from fortran should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// asfortranarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn asfortranarray_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.asfortranarray(a)
expected = np.asfortranarray(a)
print(np.array_equal(result, expected) and result.flags['F_CONTIGUOUS'])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asfortranarray basic should match numpy"
    );
    Ok(())
}

#[test]
fn asfortranarray_already_fortran() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.asfortranarray([[1, 2], [3, 4]])
result = fnp.asfortranarray(a)
expected = np.asfortranarray(a)
print(np.array_equal(result, expected) and result.flags['F_CONTIGUOUS'])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "asfortranarray already fortran should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn array_equal_vs_equiv_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 1, 1])
b = np.array([[1, 1, 1], [1, 1, 1]])
# array_equal should fail (different shapes), array_equiv should pass (broadcastable)
eq_result = fnp.array_equal(a, b)
equiv_result = fnp.array_equiv(a, b)
print(eq_result == False and equiv_result == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "array_equal vs array_equiv broadcasting behavior"
    );
    Ok(())
}

#[test]
fn copy_modifying_original_doesnt_affect_copy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = fnp.copy(a)
a[0] = 999
print(b[0] == 1)  # copy should be unaffected
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "modifying original should not affect copy"
    );
    Ok(())
}
