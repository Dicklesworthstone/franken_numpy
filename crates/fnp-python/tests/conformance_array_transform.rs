//! Conformance tests for numpy array transformation functions against NumPy oracle.
//!
//! Tests roll, rot90, flip, flipud, fliplr.

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
// roll
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn roll_1d_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.roll(a, 2)
expected = np.roll(a, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll 1d positive should match numpy");
    Ok(())
}

#[test]
fn roll_1d_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.roll(a, -2)
expected = np.roll(a, -2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll 1d negative should match numpy");
    Ok(())
}

#[test]
fn roll_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.roll(a, 2)
expected = np.roll(a, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll 2d no axis should match numpy");
    Ok(())
}

#[test]
fn roll_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.roll(a, 1, axis=0)
expected = np.roll(a, 1, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn roll_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = fnp.roll(a, 1, axis=1)
expected = np.roll(a, 1, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn roll_zero_shift() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.roll(a, 0)
expected = np.roll(a, 0)
print(np.array_equal(result, expected) and np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll zero shift should be identity");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// rot90
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rot90_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.rot90(a)
expected = np.rot90(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rot90 basic should match numpy");
    Ok(())
}

#[test]
fn rot90_k2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.rot90(a, k=2)
expected = np.rot90(a, k=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rot90 k=2 should match numpy");
    Ok(())
}

#[test]
fn rot90_k3() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.rot90(a, k=3)
expected = np.rot90(a, k=3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rot90 k=3 should match numpy");
    Ok(())
}

#[test]
fn rot90_k4_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.rot90(a, k=4)
expected = np.rot90(a, k=4)
print(np.array_equal(result, expected) and np.array_equal(result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rot90 k=4 should be identity");
    Ok(())
}

#[test]
fn rot90_negative_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
result = fnp.rot90(a, k=-1)
expected = np.rot90(a, k=-1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rot90 negative k should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flip_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.flip(a)
expected = np.flip(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flip 1d should match numpy");
    Ok(())
}

#[test]
fn flip_2d_all_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.flip(a)
expected = np.flip(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flip 2d all axes should match numpy");
    Ok(())
}

#[test]
fn flip_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.flip(a, axis=0)
expected = np.flip(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flip 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn flip_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.flip(a, axis=1)
expected = np.flip(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flip 2d axis=1 should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flipud / fliplr
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flipud_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
result = fnp.flipud(a)
expected = np.flipud(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flipud should match numpy");
    Ok(())
}

#[test]
fn fliplr_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.fliplr(a)
expected = np.fliplr(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fliplr should match numpy");
    Ok(())
}

#[test]
fn flipud_equals_flip_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4], [5, 6]])
flipud_result = fnp.flipud(a)
flip_axis0_result = fnp.flip(a, axis=0)
print(np.array_equal(flipud_result, flip_axis0_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flipud should equal flip(axis=0)");
    Ok(())
}

#[test]
fn fliplr_equals_flip_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
fliplr_result = fnp.fliplr(a)
flip_axis1_result = fnp.flip(a, axis=1)
print(np.array_equal(fliplr_result, flip_axis1_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "fliplr should equal flip(axis=1)");
    Ok(())
}

#[test]
fn flip_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)
fnp_result = fnp.flip(a)
np_result = np.flip(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "flip complex should match numpy");
    Ok(())
}

#[test]
fn roll_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2-1j, 3+2j, 4-2j], dtype=np.complex128)
fnp_result = fnp.roll(a, 2)
np_result = np.roll(a, 2)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "roll complex should match numpy");
    Ok(())
}

#[test]
fn rot90_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2-1j], [3+2j, 4-2j]], dtype=np.complex128)
fnp_result = fnp.rot90(a)
np_result = np.rot90(a)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rot90 complex should match numpy");
    Ok(())
}

/// Locks the zero-copy flatten-roll fast path (`try_zerocopy_f64_roll`) to
/// bit-exact parity with numpy. roll moves values verbatim, so parity must hold
/// at the IEEE-754 bit level (signed zero, nan payloads, inf). Compares the
/// sha256 of raw output bytes across shifts that exceed/equal/negate the length,
/// 1-D and multi-D (axis=None flatten) inputs, and extreme values — all of which
/// take the zero-copy path.
#[test]
fn roll_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    x = rng.standard_normal(n)
    for s in [1, -1, 7, n + 5, -999]:
        chunks.append(np.asarray(mod.roll(x, s)).tobytes())
for shp in [(3, 4), (5, 5, 5)]:
    x = rng.standard_normal(shp)
    for s in [1, -7, 123]:
        chunks.append(np.asarray(mod.roll(x, s)).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
chunks.append(np.asarray(mod.roll(xe, 2)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy roll must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
