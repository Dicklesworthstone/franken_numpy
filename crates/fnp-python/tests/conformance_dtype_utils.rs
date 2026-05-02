//! Conformance tests for numpy dtype utility functions against NumPy oracle.
//!
//! Tests broadcast_shapes, can_cast, common_type, promote_types.

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
// broadcast_shapes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn broadcast_shapes_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_shapes((3, 4), (4,))
expected = np.broadcast_shapes((3, 4), (4,))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "broadcast_shapes basic should match numpy");
    Ok(())
}

#[test]
fn broadcast_shapes_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_shapes((3, 4), ())
expected = np.broadcast_shapes((3, 4), ())
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "broadcast_shapes scalar should match numpy");
    Ok(())
}

#[test]
fn broadcast_shapes_multiple() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.broadcast_shapes((1, 2), (3, 1), (3, 2))
expected = np.broadcast_shapes((1, 2), (3, 1), (3, 2))
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "broadcast_shapes multiple should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// can_cast
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn can_cast_int_to_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('int32', 'float64')
expected = np.can_cast('int32', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "can_cast int to float should match numpy");
    Ok(())
}

#[test]
fn can_cast_float_to_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('float64', 'int32')
expected = np.can_cast('float64', 'int32')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "can_cast float to int should match numpy");
    Ok(())
}

#[test]
fn can_cast_same_type() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('float64', 'float64')
expected = np.can_cast('float64', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "can_cast same type should match numpy");
    Ok(())
}

#[test]
fn can_cast_with_casting() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.can_cast('int64', 'int32', casting='same_kind')
expected = np.can_cast('int64', 'int32', casting='same_kind')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "can_cast with casting should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// common_type
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn common_type_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2], dtype='int32')
b = np.array([1.0, 2.0], dtype='float64')
result = fnp.common_type(a, b)
expected = np.common_type(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "common_type int+float should match numpy");
    Ok(())
}

#[test]
fn common_type_floats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0], dtype='float32')
b = np.array([1.0], dtype='float64')
result = fnp.common_type(a, b)
expected = np.common_type(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "common_type floats should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// promote_types
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn promote_types_int_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.promote_types('int32', 'float32')
expected = np.promote_types('int32', 'float32')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "promote_types int+float should match numpy");
    Ok(())
}

#[test]
fn promote_types_ints() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.promote_types('int8', 'int16')
expected = np.promote_types('int8', 'int16')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "promote_types ints should match numpy");
    Ok(())
}

#[test]
fn promote_types_same() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.promote_types('float64', 'float64')
expected = np.promote_types('float64', 'float64')
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "promote_types same should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn can_cast_implies_promotion() -> Result<(), String> {
    let script = fnp_script(
        r#"
# If can_cast is true for safe casting, promote_types should give target
can = fnp.can_cast('int32', 'float64', casting='safe')
promoted = fnp.promote_types('int32', 'float64')
print(can == True and promoted == np.dtype('float64'))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "can_cast implies compatible promotion");
    Ok(())
}
