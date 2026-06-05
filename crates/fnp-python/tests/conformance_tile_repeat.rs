//! Conformance tests for numpy.tile and numpy.repeat against NumPy oracle.
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
// tile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tile_1d_scalar_reps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.tile(a, 3)
expected = np.tile(a, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tile 1d with scalar reps should match numpy"
    );
    Ok(())
}

#[test]
fn tile_1d_tuple_reps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.tile(a, (2, 3))
expected = np.tile(a, (2, 3))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tile 1d with tuple reps should match numpy"
    );
    Ok(())
}

#[test]
fn tile_2d_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.tile(a, 2)
expected = np.tile(a, 2)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tile 2d with scalar reps should match numpy"
    );
    Ok(())
}

#[test]
fn tile_2d_with_2d_reps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.tile(a, (2, 3))
expected = np.tile(a, (2, 3))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tile 2d with 2d reps should match numpy"
    );
    Ok(())
}

#[test]
fn tile_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
result = fnp.tile(a, 2)
expected = np.tile(a, 2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tile float array should match numpy");
    Ok(())
}

#[test]
fn tile_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
result = fnp.tile(a, 3)
expected = np.tile(a, 3)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tile empty array should match numpy");
    Ok(())
}

#[test]
fn tile_zero_reps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.tile(a, 0)
expected = np.tile(a, 0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tile with zero reps should match numpy"
    );
    Ok(())
}

#[test]
fn tile_higher_dim_reps() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
result = fnp.tile(a, (2, 2, 2))
expected = np.tile(a, (2, 2, 2))
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tile with higher dim reps should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// repeat
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn repeat_1d_scalar_repeats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.repeat(a, 2)
expected = np.repeat(a, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat 1d with scalar repeats should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_1d_array_repeats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.repeat(a, [1, 2, 3])
expected = np.repeat(a, [1, 2, 3])
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat 1d with array repeats should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.repeat(a, 2)
expected = np.repeat(a, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat 2d without axis should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.repeat(a, 2, axis=0)
expected = np.repeat(a, 2, axis=0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "repeat 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn repeat_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.repeat(a, 2, axis=1)
expected = np.repeat(a, 2, axis=1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "repeat 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn repeat_float_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.5, 2.5, 3.5])
result = fnp.repeat(a, 3)
expected = np.repeat(a, 3)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat float array should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
result = fnp.repeat(a, 2)
expected = np.repeat(a, 2)
print(np.array_equal(result, expected) and result.dtype == expected.dtype)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat empty array should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_zero_repeats() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
result = fnp.repeat(a, 0)
expected = np.repeat(a, 0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat with zero repeats should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_negative_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.repeat(a, 2, axis=-1)
expected = np.repeat(a, 2, axis=-1)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat with negative axis should match numpy"
    );
    Ok(())
}

#[test]
fn repeat_variable_repeats_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.repeat(a, [1, 2], axis=0)
expected = np.repeat(a, [1, 2], axis=0)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "repeat with variable repeats along axis should match numpy"
    );
    Ok(())
}

#[test]
fn tile_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
fnp_result = fnp.tile(a, 2)
np_result = np.tile(a, 2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "tile complex should match numpy");
    Ok(())
}

#[test]
fn repeat_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j], dtype=np.complex128)
fnp_result = fnp.repeat(a, 2)
np_result = np.repeat(a, 2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "repeat complex should match numpy");
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
fn repeat_axis_out_of_bounds_raises_axiserror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
fnp.repeat(a, 2, axis=5)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
np.repeat(a, 2, axis=5)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "repeat with out-of-bounds axis should raise same error as numpy"
    );
}

/// Locks the zero-copy block-copy fast path for `numpy.tile` of a 1-D f64 array
/// with scalar reps (`try_zerocopy_f64_tile`) to bit-exact parity. tile lays the
/// input down end to end, copying every value verbatim, so parity must hold at
/// the IEEE-754 bit level (signed zero, nan, inf). Compares sha256 of raw output
/// bytes against numpy across several rep counts and extreme values.
#[test]
fn tile_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    for reps in [1, 2, 7]:
        chunks.append(np.asarray(mod.tile(rng.standard_normal(n), reps)).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
chunks.append(np.asarray(mod.tile(xe, 5)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy tile must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
