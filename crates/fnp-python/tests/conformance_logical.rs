//! Conformance tests for numpy logical operations against NumPy oracle.
//!
//! Tests logical_and, logical_or, logical_not, logical_xor.

use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| format!("python3 stdin pipe should be available\nScript: {script}"))?
        .write_all(script.as_bytes())
        .map_err(|error| {
            format!("failed to write Python oracle script: {error}\nScript: {script}")
        })?;

    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for Python oracle: {error}\nScript: {script}"))?;
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
fn logical_and_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
result = fnp.logical_and(x1, x2)
expected = np.logical_and(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logical_and basic should match numpy"
    );
    Ok(())
}

#[test]
fn logical_or_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
result = fnp.logical_or(x1, x2)
expected = np.logical_or(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_or basic should match numpy");
    Ok(())
}

#[test]
fn logical_not_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([True, False, True, False])
result = fnp.logical_not(x)
expected = np.logical_not(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logical_not basic should match numpy"
    );
    Ok(())
}

#[test]
fn logical_xor_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True, True, False, False])
x2 = np.array([True, False, True, False])
result = fnp.logical_xor(x1, x2)
expected = np.logical_xor(x1, x2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logical_xor basic should match numpy"
    );
    Ok(())
}

#[test]
fn logical_and_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.bool_(True)
x2 = np.bool_(False)
fnp_result = fnp.logical_and(x1, x2)
np_result = np.logical_and(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logical_and scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn logical_not_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.bool_(True)
fnp_result = fnp.logical_not(x)
np_result = np.logical_not(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "logical_not scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn logical_operations_on_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0, 1, 2, 0])
x2 = np.array([0, 0, 1, 3])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical operations on integers should match numpy");
    Ok(())
}

#[test]
fn logical_operations_on_floats() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, 1.0, 0.5, 0.0])
x2 = np.array([0.0, 0.0, 0.5, 1.5])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical operations on floats should match numpy");
    Ok(())
}

#[test]
fn logical_not_on_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, -1, 0])
fnp_result = fnp.logical_not(x)
np_result = np.logical_not(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical_not on integers should match numpy");
    Ok(())
}

#[test]
fn logical_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([[True, False], [True, False]])
x2 = np.array([True, False])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical broadcasting should match numpy");
    Ok(())
}

#[test]
fn logical_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.nan, 1.0, 0.0, np.nan])
x2 = np.array([1.0, np.nan, np.nan, np.nan])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
fnp_not = fnp.logical_not(x1)
np_not = np.logical_not(x1)
tests_pass = tests_pass and np.array_equal(fnp_not, np_not)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical nan handling should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn logical_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, np.inf, 0.0])
x2 = np.array([1.0, 0.0, -np.inf, np.inf])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
fnp_not = fnp.logical_not(x1)
np_not = np.logical_not(x1)
tests_pass = tests_pass and np.array_equal(fnp_not, np_not)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical inf handling should match numpy");
    Ok(())
}

#[test]
fn logical_signed_zero_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([0.0, -0.0, 0.0, -0.0])
x2 = np.array([0.0, 0.0, -0.0, -0.0])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
fnp_not_pos = fnp.logical_not(np.array([0.0]))
np_not_pos = np.logical_not(np.array([0.0]))
fnp_not_neg = fnp.logical_not(np.array([-0.0]))
np_not_neg = np.logical_not(np.array([-0.0]))
tests_pass = tests_pass and np.array_equal(fnp_not_pos, np_not_pos)
tests_pass = tests_pass and np.array_equal(fnp_not_neg, np_not_neg)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical signed zero handling should match numpy");
    Ok(())
}

#[test]
fn logical_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([], dtype=bool)
x2 = np.array([], dtype=bool)
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    tests_pass = tests_pass and (fnp_result.shape == np_result.shape)
fnp_not = fnp.logical_not(x1)
np_not = np.logical_not(x1)
tests_pass = tests_pass and np.array_equal(fnp_not, np_not)
tests_pass = tests_pass and (fnp_not.shape == np_not.shape)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical empty array should match numpy");
    Ok(())
}

#[test]
fn logical_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.array([True])
x2 = np.array([False])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x1, x2)
    np_result = np_func(x1, x2)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
fnp_not = fnp.logical_not(x1)
np_not = np.logical_not(x1)
tests_pass = tests_pass and np.array_equal(fnp_not, np_not)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical single element should match numpy");
    Ok(())
}

#[test]
fn logical_mixed_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
x_bool = np.array([True, False])
x_int = np.array([1, 0])
x_float = np.array([1.0, 0.0])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(x_bool, x_int)
    np_result = np_func(x_bool, x_int)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    fnp_result = fnp_func(x_int, x_float)
    np_result = np_func(x_int, x_float)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical mixed dtypes should match numpy");
    Ok(())
}

#[test]
fn logical_all_true_all_false() -> Result<(), String> {
    let script = fnp_script(
        r#"
all_true = np.array([True, True, True, True])
all_false = np.array([False, False, False, False])
tests_pass = True
for func_name in ['logical_and', 'logical_or', 'logical_xor']:
    fnp_func = getattr(fnp, func_name)
    np_func = getattr(np, func_name)
    fnp_result = fnp_func(all_true, all_true)
    np_result = np_func(all_true, all_true)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    fnp_result = fnp_func(all_false, all_false)
    np_result = np_func(all_false, all_false)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
    fnp_result = fnp_func(all_true, all_false)
    np_result = np_func(all_true, all_false)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logical all true/false should match numpy");
    Ok(())
}
