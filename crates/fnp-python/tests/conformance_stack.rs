//! Conformance tests for numpy.stack, vstack, hstack, dstack against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy.

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
// stack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stack_1d_arrays_default_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b])
expected = np.stack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack 1d default axis should match numpy"
    );
    Ok(())
}

#[test]
fn stack_1d_arrays_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b], axis=0)
expected = np.stack([a, b], axis=0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack 1d axis=0 should match numpy");
    Ok(())
}

#[test]
fn stack_1d_arrays_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b], axis=1)
expected = np.stack([a, b], axis=1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack 1d axis=1 should match numpy");
    Ok(())
}

#[test]
fn stack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.stack([a, b])
expected = np.stack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack 2d should match numpy");
    Ok(())
}

#[test]
fn stack_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.stack([a, b], axis=-1)
expected = np.stack([a, b], axis=-1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack negative axis should match numpy"
    );
    Ok(())
}

#[test]
fn stack_float_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
b = np.array([4.5, 5.5, 6.5])
result = fnp.stack([a, b])
expected = np.stack([a, b])
print(np.allclose(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "stack float arrays should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// vstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vstack_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.vstack([a, b])
expected = np.vstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vstack 1d arrays should match numpy");
    Ok(())
}

#[test]
fn vstack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.vstack([a, b])
expected = np.vstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vstack 2d arrays should match numpy");
    Ok(())
}

#[test]
fn vstack_mixed_dimensions() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([[4, 5, 6]])
result = fnp.vstack([a, b])
expected = np.vstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vstack mixed dimensions should match numpy"
    );
    Ok(())
}

#[test]
fn vstack_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.vstack([a])
expected = np.vstack([a])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vstack single array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hstack_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.hstack([a, b])
expected = np.hstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hstack 1d arrays should match numpy");
    Ok(())
}

#[test]
fn hstack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.hstack([a, b])
expected = np.hstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hstack 2d arrays should match numpy");
    Ok(())
}

#[test]
fn hstack_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.hstack([a])
expected = np.hstack([a])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hstack single array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dstack
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dstack_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.dstack([a, b])
expected = np.dstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dstack 1d arrays should match numpy");
    Ok(())
}

#[test]
fn dstack_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.dstack([a, b])
expected = np.dstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dstack 2d arrays should match numpy");
    Ok(())
}

#[test]
fn dstack_3d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.ones((2, 3, 4))
b = np.ones((2, 3, 4)) * 2
result = fnp.dstack([a, b])
expected = np.dstack([a, b])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dstack 3d arrays should match numpy");
    Ok(())
}

#[test]
fn dstack_single_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.dstack([a])
expected = np.dstack([a])
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dstack single array should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn vstack_row_stack_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vstack_result = fnp.vstack([a, b])
row_stack_result = fnp.row_stack([a, b])
print(np.array_equal(vstack_result, row_stack_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "vstack and row_stack should be equivalent"
    );
    Ok(())
}

#[test]
fn hstack_column_stack_1d_difference() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
hstack_result = fnp.hstack([a, b])
column_stack_result = fnp.column_stack([a, b])
hstack_expected = np.hstack([a, b])
column_stack_expected = np.column_stack([a, b])
print(np.array_equal(hstack_result, hstack_expected) and np.array_equal(column_stack_result, column_stack_expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hstack and column_stack 1d should match numpy"
    );
    Ok(())
}

#[test]
fn stack_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.stack([a, b])
np_result = np.stack([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "stack complex should match numpy");
    Ok(())
}

#[test]
fn hstack_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.hstack([a, b])
np_result = np.hstack([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hstack complex should match numpy");
    Ok(())
}

#[test]
fn vstack_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
b = np.array([3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.vstack([a, b])
np_result = np.vstack([a, b])
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "vstack complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else if stderr.contains("AxisError") {
            "AxisError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn stack_shape_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(8).reshape(2, 4)
fnp.stack([a, b])
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(8).reshape(2, 4)
np.stack([a, b])
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "stack with shape mismatch should raise same error as numpy"
    );
}

#[test]
fn hstack_shape_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(9).reshape(3, 3)
fnp.hstack([a, b])
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(9).reshape(3, 3)
np.hstack([a, b])
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "hstack with incompatible first dimensions should raise same error as numpy"
    );
}

/// Locks the zero-copy 2-D vstack fast path (which routes through
/// `try_zerocopy_f64_concatenate_axis0`) to bit-exact parity with numpy. Stacked
/// rows are copied verbatim, so parity must hold at the IEEE-754 bit level.
/// Compares the sha256 of raw output bytes for 2-D float64 inputs and extremes.
#[test]
fn vstack_2d_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
chunks.append(np.asarray(mod.vstack([rng.standard_normal((100, 50)), rng.standard_normal((200, 50))])).tobytes())
chunks.append(np.asarray(mod.vstack([rng.standard_normal((30, 40)), rng.standard_normal((5, 40)), rng.standard_normal((17, 40))])).tobytes())
xe = np.array([[0.0, -0.0, np.inf], [-np.inf, np.nan, 1e308]], dtype=np.float64)
chunks.append(np.asarray(mod.vstack([xe, xe * 2])).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!("import numpy as np\n{}", body.replace("MODULE", "np")))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy 2-D vstack must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
