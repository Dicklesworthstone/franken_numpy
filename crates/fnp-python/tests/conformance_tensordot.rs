//! Conformance tests for numpy.tensordot against NumPy oracle.
//!
//! Tests tensordot (tensor dot product).

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

#[test]
fn tensordot_axes_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(60.).reshape(3, 4, 5)
b = np.arange(24.).reshape(4, 3, 2)
result = fnp.tensordot(a, b, axes=([1, 0], [0, 1]))
expected = np.tensordot(a, b, axes=([1, 0], [0, 1]))
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes tuple should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(2, 3)
result = fnp.tensordot(a, b, axes=0)
expected = np.tensordot(a, b, axes=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes=0 (outer product) should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
result = fnp.tensordot(a, b, axes=1)
expected = np.tensordot(a, b, axes=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes=1 (matrix product) should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_axes_2() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(24).reshape(3, 4, 2)
result = fnp.tensordot(a, b, axes=2)
expected = np.tensordot(a, b, axes=2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot with axes=2 should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_scalar_result() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
fnp_result = fnp.tensordot(a, b, axes=1)
np_result = np.tensordot(a, b, axes=1)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "tensordot scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn tensordot_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2]], dtype=np.complex128)
b = np.array([[1], [2+1j]], dtype=np.complex128)
fnp_result = fnp.tensordot(a, b, axes=1)
np_result = np.tensordot(a, b, axes=1)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot complex should match numpy"
    );
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
        if stderr.contains("ValueError") || stderr.contains("shape") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn tensordot_axes_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(12).reshape(4, 3)
fnp.tensordot(a, b, axes=1)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(4, 3)
np.tensordot(a, b, axes=1)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "tensordot axes mismatch should raise same error as numpy"
    );
}

#[test]
fn tensordot_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [np.nan, 2.0]])
b = np.array([[1.0], [1.0]])
fnp_result = fnp.tensordot(a, b, axes=1)
np_result = np.tensordot(a, b, axes=1)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot special values should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_empty_result() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3]])
b = np.array([[4], [5], [6]])
fnp_result = fnp.tensordot(a, b, axes=0)
np_result = np.tensordot(a, b, axes=0)
print(fnp_result.shape == np_result.shape and np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot axes=0 shape should match numpy"
    );
    Ok(())
}

#[test]
fn tensordot_negative_axes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(12).reshape(3, 4)
fnp_result = fnp.tensordot(a, b, axes=([-1, -2], [-1, -2]))
np_result = np.tensordot(a, b, axes=([-1, -2], [-1, -2]))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot negative axes should match numpy"
    );
    Ok(())
}

#[test]
#[ignore = "PARITY GAP: fnp accumulator returns -0.0, NumPy returns 0.0. See DISC-011."]
fn tensordot_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Tensordot signed-zero parity (generalized dot product)
# tensordot contracts specified axes via multiplication and summation
a = np.array([[1.0, -0.0], [-0.0, 1.0]])
b = np.array([[1.0, 0.0], [0.0, 1.0]])
fnp_result = fnp.tensordot(a, b, axes=1)
np_result = np.tensordot(a, b, axes=1)
fnp_signs = np.signbit(fnp_result)
np_signs = np.signbit(np_result)
if np.array_equal(fnp_signs, np_signs):
    print("True")
else:
    print(f"FAIL: tensordot signbit mismatch")
    print(f"  fnp signbit={fnp_signs.tolist()}")
    print(f"  np signbit={np_signs.tolist()}")
    print("False")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "tensordot signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_tensordot_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy integer tensordot flattens to a no-BLAS matmul (slow). The native route
    // (reshape -> int GEMM -> reshape) must be byte-identical incl. overflow wrap and
    // multi-axis contraction.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(23)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32]:
    info = np.iinfo(dt)
    # axes=1 over 3-D (the common contraction)
    a = rng.integers(info.min // 4, info.max // 4, (40, 50, 24)).astype(dt)
    b = rng.integers(info.min // 4, info.max // 4, (24, 18, 9)).astype(dt)
    r = fnp.tensordot(a, b, 1); e = np.tensordot(a, b, 1)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    # axes=2 (multi-axis contraction)
    a2 = rng.integers(info.min // 8, info.max // 8, (20, 22, 30)).astype(dt)
    b2 = rng.integers(info.min // 8, info.max // 8, (22, 30, 17)).astype(dt)
    r2 = fnp.tensordot(a2, b2, 2); e2 = np.tensordot(a2, b2, 2)
    ok = ok and r2.dtype == e2.dtype and r2.shape == e2.shape and r2.tobytes() == e2.tobytes()
# 2-D axes=1 == matmul + overflow wrap
a = np.full((130, 130), 5_000_000_000, dtype=np.int64)
b = np.full((130, 130), 5_000_000_000, dtype=np.int64)
ok = ok and fnp.tensordot(a, b, 1).tobytes() == np.tensordot(a, b, 1).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer tensordot must be bit-identical to numpy: {result}"
    );
    Ok(())
}
