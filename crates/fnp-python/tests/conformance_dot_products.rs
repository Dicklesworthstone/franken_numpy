//! Conformance tests for numpy dot product functions against NumPy oracle.
//!
//! Tests vdot, tensordot, unwrap.

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
// vdot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vdot_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.vdot(a, b)
expected = np.vdot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vdot basic should match numpy");
    Ok(())
}

#[test]
fn vdot_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.vdot(a, b)  # flattens arrays first
expected = np.vdot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vdot 2d should match numpy");
    Ok(())
}

#[test]
fn vdot_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+2j, 3+4j])
b = np.array([5+6j, 7+8j])
result = fnp.vdot(a, b)
expected = np.vdot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vdot complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// tensordot
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensordot_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
result = fnp.tensordot(a, b)  # default axes=2 works for square matrices
expected = np.tensordot(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot basic should match numpy");
    Ok(())
}

#[test]
fn tensordot_axes1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.tensordot(a, b, axes=1)
expected = np.tensordot(a, b, axes=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tensordot axes=1 should match numpy");
    Ok(())
}

#[test]
fn tensordot_axes_tuple() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 4)
b = np.arange(12).reshape(4, 3)
result = fnp.tensordot(a, b, axes=([1], [0]))
expected = np.tensordot(a, b, axes=([1], [0]))
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot axes tuple should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unwrap
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unwrap_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
phase = np.array([0, np.pi/4, np.pi/2, -np.pi/2, -np.pi/4, 0])
result = fnp.unwrap(phase)
expected = np.unwrap(phase)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unwrap basic should match numpy");
    Ok(())
}

#[test]
fn unwrap_large_jump() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Phase with a discontinuity
phase = np.array([0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0])
result = fnp.unwrap(phase)
expected = np.unwrap(phase)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "unwrap large jump should match numpy"
    );
    Ok(())
}

#[test]
fn unwrap_2d_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
phase = np.array([[0, np.pi/4, np.pi/2], [0, -np.pi/4, -np.pi/2]])
result = fnp.unwrap(phase, axis=1)
expected = np.unwrap(phase, axis=1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unwrap 2d axis should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// fix
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fix_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([2.5, -2.5, 3.9, -3.9])
result = fnp.fix(a)
expected = np.fix(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fix basic should match numpy");
    Ok(())
}

#[test]
fn fix_integers() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, -1.0, 2.0, -2.0])
result = fnp.fix(a)
expected = np.fix(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fix integers should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vdot_equals_dot_for_real() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vdot_result = fnp.vdot(a, b)
dot_result = np.dot(a, b)
print(np.allclose(vdot_result, dot_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vdot should equal dot for real arrays"
    );
    Ok(())
}

#[test]
fn fix_truncates_toward_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([2.7, -2.7])
result = fnp.fix(a)
# fix should truncate toward zero: 2.7 -> 2, -2.7 -> -2
print(np.allclose(result, [2, -2]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fix should truncate toward zero");
    Ok(())
}

#[test]
fn fix_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.7)
fnp_result = fnp.fix(x)
np_result = np.fix(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "fix scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn dot_vdot_inner_scalar_return_type_matches_numpy() -> Result<(), String> {
    for func in &["dot", "vdot", "inner"] {
        let script = fnp_script(format!(
            r#"
x = np.float64(2.0)
y = np.float64(3.0)
fnp_result = fnp.{func}(x, y)
np_result = np.{func}(x, y)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        ));
        let result = numpy_oracle(&script)?;
        assert!(
            result.trim().starts_with("True"),
            "{func} scalar return type should match numpy: {result}"
        );
    }
    Ok(())
}
