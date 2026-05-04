//! Conformance tests for numpy.where against NumPy oracle.
//!
//! Tests the native Rust where implementation against NumPy.
//!
//! np.where has two modes:
//! - where(condition): returns tuple of indices where condition is True (like nonzero)
//! - where(condition, x, y): returns x where condition is True, y otherwise

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
// where(condition, x, y) - selection mode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn where_basic_selection() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where basic selection should match numpy");
    Ok(())
}

#[test]
fn where_float_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
x = np.array([1.5, 2.5, 3.5, 4.5])
y = np.array([10.5, 20.5, 30.5, 40.5])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where float values should match numpy");
    Ok(())
}

#[test]
fn where_2d_selection() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([[True, False], [False, True]])
x = np.array([[1, 2], [3, 4]])
y = np.array([[10, 20], [30, 40]])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where 2d selection should match numpy");
    Ok(())
}

#[test]
fn where_broadcast_condition() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
x = np.array([[1, 2], [3, 4]])
y = np.array([[10, 20], [30, 40]])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where broadcast condition should match numpy");
    Ok(())
}

#[test]
fn where_scalar_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
x = np.array([1, 1, 1])
y = np.array([0, 0, 0])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where scalar values should match numpy");
    Ok(())
}

#[test]
fn where_explicit_none_choice_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True, False])
y = np.array([1, 2, 3, 4])
left_none = fnp.where(condition, None, y)
left_expected = np.where(condition, None, y)
right_none = fnp.where(condition, y, None)
right_expected = np.where(condition, y, None)
both_none = fnp.where(condition, None, None)
both_expected = np.where(condition, None, None)
print(
    np.array_equal(left_none, left_expected)
    and left_none.dtype == left_expected.dtype
    and np.array_equal(right_none, right_expected)
    and right_none.dtype == right_expected.dtype
    and np.array_equal(both_none, both_expected)
    and both_none.dtype == both_expected.dtype
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where explicit None choices should match numpy object selection"
    );
    Ok(())
}

#[test]
fn where_rejects_positional_only_keywords() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False])
x = np.array([1, 2])
y = np.array([10, 20])

def error_type(call):
    try:
        call()
        return "OK"
    except Exception as exc:
        return type(exc).__name__

cases = [
    (
        error_type(lambda: fnp.where(condition=condition)),
        error_type(lambda: np.where(condition=condition)),
    ),
    (
        error_type(lambda: fnp.where(condition, x=x, y=y)),
        error_type(lambda: np.where(condition, x=x, y=y)),
    ),
]
print(all(ours == theirs == "TypeError" for ours, theirs in cases))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where positional-only keyword rejection should match numpy"
    );
    Ok(())
}

#[test]
fn where_all_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, True, True])
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and np.array_equal(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where all true should return x");
    Ok(())
}

#[test]
fn where_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and np.array_equal(result, y))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where all false should return y");
    Ok(())
}

#[test]
fn where_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([], dtype=bool)
x = np.array([], dtype=np.float64)
y = np.array([], dtype=np.float64)
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where empty array should match numpy");
    Ok(())
}

#[test]
fn where_numeric_condition() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([0, 1, 2, 0, -1])  # 0 is False, non-zero is True
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where numeric condition should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// where(condition) - index mode (like nonzero)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn where_index_mode_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, True, False, True, True])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where index mode 1d should match numpy");
    Ok(())
}

#[test]
fn where_index_mode_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([[True, False], [False, True]])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where index mode 2d should match numpy");
    Ok(())
}

#[test]
fn where_index_mode_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([False, False, False])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(len(r) == 0 for r in result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where index mode all false should return empty indices");
    Ok(())
}

#[test]
fn where_index_mode_all_true() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, True, True])
result = fnp.where(condition)
expected = np.where(condition)
print(len(result) == len(expected) and all(np.array_equal(r, e) for r, e in zip(result, expected)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where index mode all true should return all indices");
    Ok(())
}

#[test]
fn where_with_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
condition = np.array([True, False, True])
x = np.array([np.nan, 2.0, np.nan])
y = np.array([10.0, 20.0, 30.0])
result = fnp.where(condition, x, y)
expected = np.where(condition, x, y)
# NaN comparison needs special handling
match = all(
    (np.isnan(r) and np.isnan(e)) or (r == e)
    for r, e in zip(result.flat, expected.flat)
)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where with NaN should match numpy");
    Ok(())
}
