//! Conformance tests for numpy partition functions against NumPy oracle.
//!
//! Tests partition, argpartition.

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
// partition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn partition_1d_kth_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 4, 2, 1, 5, 0])
result = fnp.partition(a, 2)
expected = np.partition(a, 2)
# Check that element at index 2 is correct (3rd smallest)
print(result[2] == expected[2] and np.sort(result[:2]).tolist() == np.sort(expected[:2]).tolist())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition 1d kth=2 should match numpy"
    );
    Ok(())
}

#[test]
fn partition_1d_kth_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 2, 8, 1, 9])
result = fnp.partition(a, 0)
expected = np.partition(a, 0)
# Element at index 0 should be the minimum
print(result[0] == expected[0])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition kth=0 should give minimum at index 0"
    );
    Ok(())
}

#[test]
fn partition_1d_kth_last() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 2, 8, 1, 9])
result = fnp.partition(a, 4)
expected = np.partition(a, 4)
# Element at index 4 should be the maximum
print(result[4] == expected[4])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition kth=last should give maximum at last index"
    );
    Ok(())
}

#[test]
fn partition_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 4, 2], [1, 5, 0], [6, 2, 8]])
result = fnp.partition(a, 1, axis=0)
expected = np.partition(a, 1, axis=0)
# Check the middle row has correct partitioned values
print(np.array_equal(result[1], expected[1]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn partition_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 4, 2], [1, 5, 0]])
result = fnp.partition(a, 1, axis=1)
expected = np.partition(a, 1, axis=1)
# Check column 1 has correct partitioned values per row
print(np.array_equal(result[:, 1], expected[:, 1]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn partition_kind_introselect_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([9, 4, 1, 7, 3, 6])
kth = 2
result = fnp.partition(a, kth, kind="introselect")
expected = np.partition(a, kth, kind="introselect")
print(result[kth] == expected[kth] and np.sort(result[:kth]).tolist() == np.sort(expected[:kth]).tolist())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition kind=introselect should match numpy"
    );
    Ok(())
}

#[test]
fn partition_invalid_kind_matches_numpy_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 2])
def classify(call):
    try:
        call()
        return ("ok", "")
    except Exception as error:
        return (type(error).__name__, str(error).splitlines()[0])
result = classify(lambda: fnp.partition(a, 1, kind="bogus"))
expected = classify(lambda: np.partition(a, 1, kind="bogus"))
print(result == expected and result[0] != "ok")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition invalid kind should preserve numpy error behavior"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// argpartition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn argpartition_1d_kth_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 4, 2, 1, 5, 0])
result = fnp.argpartition(a, 2)
expected = np.argpartition(a, 2)
# The index at position 2 should point to the same value
print(a[result[2]] == a[expected[2]])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition 1d kth=2 should match numpy"
    );
    Ok(())
}

#[test]
fn argpartition_1d_kth_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5, 2, 8, 1, 9])
result = fnp.argpartition(a, 0)
expected = np.argpartition(a, 0)
# Index at position 0 should point to minimum
print(a[result[0]] == a[expected[0]] == np.min(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition kth=0 should give index of minimum"
    );
    Ok(())
}

#[test]
fn argpartition_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 4, 2], [1, 5, 0], [6, 2, 8]])
result = fnp.argpartition(a, 1, axis=0)
expected = np.argpartition(a, 1, axis=0)
# Check that indices at position 1 point to same values
matches = True
for col in range(3):
    if a[result[1, col], col] != a[expected[1, col], col]:
        matches = False
print(matches)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn argpartition_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3, 4, 2], [1, 5, 0]])
result = fnp.argpartition(a, 1, axis=1)
expected = np.argpartition(a, 1, axis=1)
# Check that indices at position 1 point to same values per row
matches = True
for row in range(2):
    if a[row, result[row, 1]] != a[row, expected[row, 1]]:
        matches = False
print(matches)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn argpartition_kind_introselect_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([9, 4, 1, 7, 3, 6])
kth = 2
result = fnp.argpartition(a, kth, kind="introselect")
expected = np.argpartition(a, kth, kind="introselect")
print(a[result[kth]] == a[expected[kth]])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition kind=introselect should match numpy"
    );
    Ok(())
}

#[test]
fn argpartition_invalid_kind_matches_numpy_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3, 1, 2])
def classify(call):
    try:
        call()
        return ("ok", "")
    except Exception as error:
        return (type(error).__name__, str(error).splitlines()[0])
result = classify(lambda: fnp.argpartition(a, 1, kind="bogus"))
expected = classify(lambda: np.argpartition(a, 1, kind="bogus"))
print(result == expected and result[0] != "ok")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition invalid kind should preserve numpy error behavior"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn partition_argpartition_consistency() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([7, 2, 5, 1, 8, 3])
kth = 3
p = fnp.partition(a, kth)
ap = fnp.argpartition(a, kth)
# partition result should equal a indexed by argpartition
print(p[kth] == a[ap[kth]])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition and argpartition should be consistent"
    );
    Ok(())
}

#[test]
fn partition_preserves_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.5, 1.2, 4.8, 2.1], dtype='float32')
result = fnp.partition(a, 2)
print(result.dtype == a.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "partition should preserve dtype");
    Ok(())
}

#[test]
fn partition_0d_raises_axis_error() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(5)
try:
    fnp.partition(a, 0)
    print("no_error")
except np.exceptions.AxisError:
    print("axis_error")
except Exception as e:
    print(f"other: {type(e).__name__}")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "axis_error",
        "partition on 0-D should raise AxisError"
    );
    Ok(())
}

#[test]
fn argpartition_0d_returns_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array(5)
result = fnp.argpartition(a, 0)
expected = np.argpartition(a, 0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argpartition on 0-D should return [0] like numpy"
    );
    Ok(())
}
