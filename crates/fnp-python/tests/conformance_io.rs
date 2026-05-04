//! Conformance tests for numpy I/O functions against NumPy oracle.
//!
//! Tests loadtxt, genfromtxt.

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
         from io import StringIO\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// loadtxt
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn loadtxt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6\n7 8 9"
result = fnp.loadtxt(StringIO(data))
expected = np.loadtxt(StringIO(data))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt basic should match numpy");
    Ok(())
}

#[test]
fn loadtxt_with_delimiter() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3\n4,5,6\n7,8,9"
result = fnp.loadtxt(StringIO(data), delimiter=',')
expected = np.loadtxt(StringIO(data), delimiter=',')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt with delimiter should match numpy");
    Ok(())
}

#[test]
fn loadtxt_with_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), dtype='int32')
expected = np.loadtxt(StringIO(data), dtype='int32')
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt with dtype should match numpy");
    Ok(())
}

#[test]
fn loadtxt_with_skiprows() -> Result<(), String> {
    let script = fnp_script(
        r##"
data = "# header\n1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), skiprows=1)
expected = np.loadtxt(StringIO(data), skiprows=1)
print(np.array_equal(result, expected))
"##
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt with skiprows should match numpy");
    Ok(())
}

#[test]
fn loadtxt_with_usecols() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3 4\n5 6 7 8"
result = fnp.loadtxt(StringIO(data), usecols=(0, 2))
expected = np.loadtxt(StringIO(data), usecols=(0, 2))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt with usecols should match numpy");
    Ok(())
}

#[test]
fn loadtxt_with_comments() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "% comment\n1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), comments='%')
expected = np.loadtxt(StringIO(data), comments='%')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt with comments should match numpy");
    Ok(())
}

#[test]
fn loadtxt_unpack() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6"
result = fnp.loadtxt(StringIO(data), unpack=True)
expected = np.loadtxt(StringIO(data), unpack=True)
match = all(np.array_equal(r, e) for r, e in zip(result, expected))
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt unpack should match numpy");
    Ok(())
}

#[test]
fn loadtxt_max_rows() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6\n7 8 9"
result = fnp.loadtxt(StringIO(data), max_rows=2)
expected = np.loadtxt(StringIO(data), max_rows=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "loadtxt max_rows should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// genfromtxt
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn genfromtxt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1 2 3\n4 5 6\n7 8 9"
result = fnp.genfromtxt(StringIO(data))
expected = np.genfromtxt(StringIO(data))
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "genfromtxt basic should match numpy");
    Ok(())
}

#[test]
fn genfromtxt_with_delimiter() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3\n4,5,6"
result = fnp.genfromtxt(StringIO(data), delimiter=',')
expected = np.genfromtxt(StringIO(data), delimiter=',')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "genfromtxt with delimiter should match numpy");
    Ok(())
}

#[test]
fn genfromtxt_with_missing() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "1,2,3\n4,,6"
result = fnp.genfromtxt(StringIO(data), delimiter=',', filling_values=0)
expected = np.genfromtxt(StringIO(data), delimiter=',', filling_values=0)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "genfromtxt with missing should match numpy");
    Ok(())
}

#[test]
fn genfromtxt_with_skip_header() -> Result<(), String> {
    let script = fnp_script(
        r#"
data = "col1,col2,col3\n1,2,3\n4,5,6"
result = fnp.genfromtxt(StringIO(data), delimiter=',', skip_header=1)
expected = np.genfromtxt(StringIO(data), delimiter=',', skip_header=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "genfromtxt with skip_header should match numpy");
    Ok(())
}
