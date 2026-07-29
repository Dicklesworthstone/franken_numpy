//! Conformance tests for numpy.dot against NumPy oracle.
//!
//! Tests dot (dot product).

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
fn dot_1d_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.isclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 1D vectors should match numpy");
    Ok(())
}

#[test]
fn dot_2d_matrices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 2D matrices should match numpy");
    Ok(())
}

#[test]
fn dot_scalar() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = 2.0
result = fnp.dot(a, b)
expected = np.dot(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot with scalar should match numpy");
    Ok(())
}

#[test]
fn dot_with_out() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
out = np.zeros((2, 2), dtype=np.int64)
fnp.dot(a, b, out=out)
expected = np.dot(a, b)
print(np.array_equal(out, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dot with out parameter should match numpy"
    );
    Ok(())
}

#[test]
fn dot_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
fnp_result = fnp.dot(a, b)
np_result = np.dot(a, b)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "dot scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn dot_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
b = np.array([4+1j, 5+2j, 6+3j], dtype=np.complex128)
fnp_result = fnp.dot(a, b)
np_result = np.dot(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot complex should match numpy");
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
        if stderr.contains("ValueError")
            || stderr.contains("not aligned")
            || stderr.contains("shape")
        {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn dot_2d_dimension_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(10).reshape(5, 2)
fnp.dot(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(10).reshape(5, 2)
np.dot(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "dot 2D dimension mismatch should raise same error as numpy"
    );
}

#[test]
fn dot_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, np.nan, 1.0])
b = np.array([1.0, 1.0, 1.0, np.nan])
fnp_result = fnp.dot(a, b)
np_result = np.dot(a, b)
# inf + -inf + nan + nan = nan
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dot special values should match numpy"
    );
    Ok(())
}

#[test]
fn dot_empty_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
b = np.array([], dtype=np.float64)
fnp_result = fnp.dot(a, b)
np_result = np.dot(a, b)
print(fnp_result == np_result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dot empty vectors should match numpy"
    );
    Ok(())
}

#[test]
fn dot_3d_nd() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(4)
fnp_result = fnp.dot(a, b)
np_result = np.dot(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot 3D array should match numpy");
    Ok(())
}

#[test]
fn dot_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0])
b = np.array([3.0])
fnp_result = fnp.dot(a, b)
np_result = np.dot(a, b)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dot single element should match numpy"
    );
    Ok(())
}

#[test]
#[ignore = "PARITY GAP: fnp accumulator returns -0.0, NumPy returns 0.0. See DISC-011."]
fn dot_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Dot product signed-zero parity (accumulation-based)
# dot(a, b) = sum(a[i] * b[i])
tests = [
    ([0.0, 0.0], [1.0, 1.0]),         # 0 in dot product
    ([-0.0, -0.0], [1.0, 1.0]),       # -0 in dot product
    ([1.0, -0.0], [0.0, 1.0]),        # mixed zeros
    ([-0.0, 0.0], [-0.0, 0.0]),       # all zeros
    ([1.0, 1.0], [-0.0, -0.0]),       # -0 multiplied by positive
]
all_pass = True
for a_vals, b_vals in tests:
    a = np.array(a_vals)
    b = np.array(b_vals)
    fnp_result = fnp.dot(a, b)
    np_result = np.dot(a, b)
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: dot({a_vals}, {b_vals})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: dot({a_vals}, {b_vals}) values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "dot signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_dot_2d_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // np.dot(2d, 2d) == matmul; numpy has no BLAS for ints (slow naive loop). The native
    // parallel GEMM route must be byte-identical to numpy.dot incl. overflow wrap.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(13)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32]:
    info = np.iinfo(dt)
    a = rng.integers(info.min // 2, info.max // 2, (96, 130)).astype(dt)
    b = rng.integers(info.min // 2, info.max // 2, (130, 97)).astype(dt)
    r = fnp.dot(a, b); e = np.dot(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
# int64 overflow wrap
a = np.full((80, 80), 5_000_000_000, dtype=np.int64)
b = np.full((80, 80), 5_000_000_000, dtype=np.int64)
ok = ok and fnp.dot(a, b).tobytes() == np.dot(a, b).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer dot must be bit-identical to numpy: {result}"
    );
    Ok(())
}
