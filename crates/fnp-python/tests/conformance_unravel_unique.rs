//! Conformance tests for numpy unravel_index and unique_* functions against NumPy oracle.
//!
//! Tests unravel_index, unique_all, unique_counts, unique_inverse, unique_values.

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
// unravel_index
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unravel_index_scalar_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(22, (7, 6))
expected = np.unravel_index(22, (7, 6))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index scalar 2d should match numpy");
    Ok(())
}

#[test]
fn unravel_index_scalar_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(41, (3, 4, 5))
expected = np.unravel_index(41, (3, 4, 5))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index scalar 3d should match numpy");
    Ok(())
}

#[test]
fn unravel_index_array_indices() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index([22, 33, 41], (7, 6))
expected = np.unravel_index([22, 33, 41], (7, 6))
match = all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index array indices should match numpy");
    Ok(())
}

#[test]
fn unravel_index_fortran_order() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(22, (7, 6), order='F')
expected = np.unravel_index(22, (7, 6), order='F')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index fortran order should match numpy");
    Ok(())
}

#[test]
fn unravel_index_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(0, (3, 4, 5))
expected = np.unravel_index(0, (3, 4, 5))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index zero should match numpy");
    Ok(())
}

#[test]
fn unravel_index_last_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
shape = (3, 4, 5)
last_idx = 3 * 4 * 5 - 1
result = fnp.unravel_index(last_idx, shape)
expected = np.unravel_index(last_idx, shape)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index last element should match numpy");
    Ok(())
}

#[test]
fn unravel_index_1d_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.unravel_index(5, (10,))
expected = np.unravel_index(5, (10,))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unravel_index 1d shape should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_all
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_all_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_all(a)
expected = np.unique_all(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.indices, expected.indices) and
         np.array_equal(result.inverse_indices, expected.inverse_indices) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_all basic should match numpy");
    Ok(())
}

#[test]
fn unique_all_floats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 1.5, 3.5, 2.5])
result = fnp.unique_all(a)
expected = np.unique_all(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.indices, expected.indices) and
         np.array_equal(result.inverse_indices, expected.inverse_indices) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_all floats should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_counts
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_counts_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_counts(a)
expected = np.unique_counts(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_counts basic should match numpy");
    Ok(())
}

#[test]
fn unique_counts_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 5, 5, 5])
result = fnp.unique_counts(a)
expected = np.unique_counts(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.counts, expected.counts))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_counts single element should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_inverse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_inverse_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_inverse(a)
expected = np.unique_inverse(a)
match = (np.array_equal(result.values, expected.values) and
         np.array_equal(result.inverse_indices, expected.inverse_indices))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_inverse basic should match numpy");
    Ok(())
}

#[test]
fn unique_inverse_reconstruct() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
result = fnp.unique_inverse(a)
expected = np.unique_inverse(a)
reconstructed_fnp = result.values[result.inverse_indices]
reconstructed_np = expected.values[expected.inverse_indices]
print(np.array_equal(reconstructed_fnp, a) and np.array_equal(reconstructed_np, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_inverse should allow reconstruction");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique_values
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_values_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_values basic should match numpy");
    Ok(())
}

#[test]
fn unique_values_sorted() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 3, 1, 4, 2])
result = fnp.unique_values(a)
expected = np.unique_values(a)
is_sorted = np.all(result[:-1] <= result[1:])
print(np.array_equal(result, expected) and is_sorted)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_values should be sorted");
    Ok(())
}

#[test]
fn unique_values_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=int)
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected) and len(result) == 0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_values empty should match numpy");
    Ok(())
}

#[test]
fn unique_values_strings() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(['b', 'a', 'c', 'a', 'b'])
result = fnp.unique_values(a)
expected = np.unique_values(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique_values strings should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-function consistency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_functions_consistent() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 1, 4, 3, 2])
all_result = fnp.unique_all(a)
counts_result = fnp.unique_counts(a)
inverse_result = fnp.unique_inverse(a)
values_result = fnp.unique_values(a)
match = (np.array_equal(all_result.values, values_result) and
         np.array_equal(all_result.counts, counts_result.counts) and
         np.array_equal(all_result.inverse_indices, inverse_result.inverse_indices))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique functions should be consistent");
    Ok(())
}
