//! Conformance tests for numpy isin, frombuffer, fromiter against NumPy oracle.
//!
//! Tests isin, frombuffer, fromiter.

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
// isin
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isin_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test_elements = np.array([2, 4])
result = fnp.isin(a, test_elements)
expected = np.isin(a, test_elements)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin basic should match numpy");
    Ok(())
}

#[test]
fn isin_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
test_elements = [1, 3]
result = fnp.isin(a, test_elements)
expected = np.isin(a, test_elements)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin 2d should match numpy");
    Ok(())
}

#[test]
fn isin_invert() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test_elements = np.array([2, 4])
result = fnp.isin(a, test_elements, invert=True)
expected = np.isin(a, test_elements, invert=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin invert should match numpy");
    Ok(())
}

#[test]
fn isin_empty_test() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test_elements = np.array([])
result = fnp.isin(a, test_elements)
expected = np.isin(a, test_elements)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin empty test should match numpy");
    Ok(())
}

// ────────────────────────────────────────���────────────────────────────────────
// frombuffer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn frombuffer_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf = b'\x01\x02\x03\x04'
result = fnp.frombuffer(buf, dtype='uint8')
expected = np.frombuffer(buf, dtype='uint8')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer basic should match numpy");
    Ok(())
}

#[test]
fn frombuffer_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
import struct
buf = struct.pack('4f', 1.0, 2.0, 3.0, 4.0)
result = fnp.frombuffer(buf, dtype='float32')
expected = np.frombuffer(buf, dtype='float32')
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer float should match numpy");
    Ok(())
}

#[test]
fn frombuffer_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf = b'\x01\x02\x03\x04\x05\x06'
result = fnp.frombuffer(buf, dtype='uint8', count=3)
expected = np.frombuffer(buf, dtype='uint8', count=3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer count should match numpy");
    Ok(())
}

#[test]
fn frombuffer_offset() -> Result<(), String> {
    let script = fnp_script(
        r#"
buf = b'\x01\x02\x03\x04\x05\x06'
result = fnp.frombuffer(buf, dtype='uint8', offset=2)
expected = np.frombuffer(buf, dtype='uint8', offset=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer offset should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fromiter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fromiter_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
iterable = (x*x for x in range(5))
result = fnp.fromiter(iterable, dtype='int64', count=5)
expected = np.fromiter((x*x for x in range(5)), dtype='int64', count=5)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromiter basic should match numpy");
    Ok(())
}

#[test]
fn fromiter_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
iterable = (x/2 for x in range(5))
result = fnp.fromiter(iterable, dtype='float64', count=5)
expected = np.fromiter((x/2 for x in range(5)), dtype='float64', count=5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromiter float should match numpy");
    Ok(())
}

#[test]
fn fromiter_list() -> Result<(), String> {
    let script = fnp_script(
        r#"
iterable = [1, 2, 3, 4, 5]
result = fnp.fromiter(iterable, dtype='int64')
expected = np.fromiter(iterable, dtype='int64')
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fromiter list should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isin_matches_membership() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
test = [2, 4]
result = fnp.isin(a, test)
# Manual check: positions 1 and 3 should be True
print(result[1] == True and result[3] == True and result[0] == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isin should correctly identify membership");
    Ok(())
}

#[test]
fn frombuffer_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
original = np.array([1, 2, 3, 4], dtype='int32')
buf = original.tobytes()
result = fnp.frombuffer(buf, dtype='int32')
print(np.array_equal(result, original))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "frombuffer roundtrip should preserve values");
    Ok(())
}
