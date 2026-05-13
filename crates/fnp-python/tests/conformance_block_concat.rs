//! Conformance tests for numpy block and concat functions against NumPy oracle.
//!
//! Tests block, concat, kron.

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
// block
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn block_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = fnp.block([[A, B], [B, A]])
expected = np.block([[A, B], [B, A]])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "block basic should match numpy");
    Ok(())
}

#[test]
fn block_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.block([a, b])
expected = np.block([a, b])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "block 1d should match numpy");
    Ok(())
}

#[test]
fn block_nested() -> Result<(), String> {
    let script = fnp_script(
        r#"
A = np.ones((2, 2))
B = np.zeros((2, 2))
result = fnp.block([[A], [B]])
expected = np.block([[A], [B]])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "block nested should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// concat
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concat_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.concat([a, b])
expected = np.concat([a, b])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "concat 1d should match numpy");
    Ok(())
}

#[test]
fn concat_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
result = fnp.concat([a, b], axis=0)
expected = np.concat([a, b], axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "concat 2d axis0 should match numpy");
    Ok(())
}

#[test]
fn concat_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])
result = fnp.concat([a, b], axis=1)
expected = np.concat([a, b], axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "concat 2d axis1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// kron
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn kron_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[0, 5], [6, 7]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron basic should match numpy");
    Ok(())
}

#[test]
fn kron_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 10, 100])
b = np.array([5, 6, 7])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron 1d should match numpy");
    Ok(())
}

#[test]
fn kron_different_shapes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([5, 6, 7])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron different shapes should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn concat_vs_concatenate() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# concat should be same as concatenate
concat_result = fnp.concat([a, b])
concatenate_result = fnp.concatenate([a, b])
print(np.array_equal(concat_result, concatenate_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "concat should equal concatenate");
    Ok(())
}

#[test]
fn kron_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
eye = np.eye(2)
# Kronecker product with identity should scale the matrix
result = fnp.kron(a, eye)
expected = np.kron(a, eye)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron with identity should match numpy"
    );
    Ok(())
}
