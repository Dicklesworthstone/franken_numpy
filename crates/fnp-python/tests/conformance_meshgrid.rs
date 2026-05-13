//! Conformance tests for numpy.meshgrid against NumPy oracle.
//!
//! Tests the native Rust meshgrid implementation against NumPy.

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

#[test]
fn meshgrid_2_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
y = np.array([4, 5])
result = fnp.meshgrid(x, y)
expected = np.meshgrid(x, y)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid 2 arrays should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_3_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2])
y = np.array([3, 4])
z = np.array([5, 6])
result = fnp.meshgrid(x, y, z)
expected = np.meshgrid(x, y, z)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid 3 arrays should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_indexing_xy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
y = np.array([4, 5])
result = fnp.meshgrid(x, y, indexing='xy')
expected = np.meshgrid(x, y, indexing='xy')
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid indexing=xy should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_indexing_ij() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
y = np.array([4, 5])
result = fnp.meshgrid(x, y, indexing='ij')
expected = np.meshgrid(x, y, indexing='ij')
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid indexing=ij should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_sparse_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
y = np.array([4, 5])
result = fnp.meshgrid(x, y, sparse=True)
expected = np.meshgrid(x, y, sparse=True)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid sparse=True should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_float_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.5, 2.5, 3.5])
y = np.array([4.5, 5.5])
result = fnp.meshgrid(x, y)
expected = np.meshgrid(x, y)
match = len(result) == len(expected) and all(np.allclose(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid float arrays should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_linspace_input() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.linspace(0, 1, 5)
y = np.linspace(0, 2, 3)
result = fnp.meshgrid(x, y)
expected = np.meshgrid(x, y)
match = len(result) == len(expected) and all(np.allclose(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid with linspace should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
result = fnp.meshgrid(x)
expected = np.meshgrid(x)
match = len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid single array should match numpy"
    );
    Ok(())
}

#[test]
fn meshgrid_shape_check() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])  # len 3
y = np.array([4, 5])     # len 2
X, Y = fnp.meshgrid(x, y)
Xe, Ye = np.meshgrid(x, y)
# xy indexing: shapes should be (len(y), len(x)) = (2, 3)
shape_match = X.shape == Xe.shape == (2, 3) and Y.shape == Ye.shape == (2, 3)
print(shape_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "meshgrid shapes should match numpy");
    Ok(())
}

#[test]
fn meshgrid_ij_shape_check() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])  # len 3
y = np.array([4, 5])     # len 2
X, Y = fnp.meshgrid(x, y, indexing='ij')
Xe, Ye = np.meshgrid(x, y, indexing='ij')
# ij indexing: shapes should be (len(x), len(y)) = (3, 2)
shape_match = X.shape == Xe.shape == (3, 2) and Y.shape == Ye.shape == (3, 2)
print(shape_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "meshgrid ij shapes should match numpy"
    );
    Ok(())
}
