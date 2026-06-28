//! Conformance tests for numpy.matmul against NumPy oracle.
//!
//! Tests matmul (matrix multiplication).

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
fn matmul_2d_matrices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul 2D matrices should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_vector_matrix() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul vector-matrix should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_matrix_vector() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.array([1.0, 2.0, 3.0])
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul matrix-vector should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_batch() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(16).reshape(2, 4, 2)
result = fnp.matmul(a, b)
expected = np.matmul(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul batch should match numpy");
    Ok(())
}

#[test]
fn matmul_operator() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(6).reshape(3, 2)
result = fnp.matmul(a, b)
expected = a @ b
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul should match @ operator");
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
            || stderr.contains("matmul:")
            || stderr.contains("not aligned")
        {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn matmul_dimension_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(6).reshape(2, 3)
b = fnp.arange(10).reshape(5, 2)
fnp.matmul(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.arange(10).reshape(5, 2)
np.matmul(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "matmul dimension mismatch should raise same error as numpy"
    );
}

#[test]
fn matmul_1d_mismatch_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(3)
b = fnp.arange(5)
fnp.matmul(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(3)
b = np.arange(5)
np.matmul(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "matmul 1D vector mismatch should raise same error as numpy"
    );
}

#[test]
fn matmul_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [np.nan, 2.0]])
b = np.array([[1.0, 2.0], [3.0, 4.0]])
fnp_result = fnp.matmul(a, b)
np_result = np.matmul(a, b)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul special values should match numpy"
    );
    Ok(())
}

#[test]
fn matmul_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
b = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex128)
fnp_result = fnp.matmul(a, b)
np_result = np.matmul(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul complex should match numpy");
    Ok(())
}

#[test]
fn matmul_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
identity = np.eye(3)
fnp_result = fnp.matmul(a, identity)
np_result = np.matmul(a, identity)
# Matmul with identity should return original (or close to it)
print(np.allclose(fnp_result, a) and np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "matmul identity should match numpy");
    Ok(())
}

#[test]
fn matmul_broadcast_batch() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(12).reshape(3, 2, 2)
b = np.arange(4).reshape(2, 2)  # broadcast this
fnp_result = fnp.matmul(a, b)
np_result = np.matmul(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul broadcast batch should match numpy"
    );
    Ok(())
}

#[test]
#[ignore = "PARITY GAP: fnp accumulator returns -0.0, NumPy returns 0.0. See DISC-011."]
fn matmul_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Matmul signed-zero parity (accumulation in matrix multiplication)
# Each output element is a dot product
tests = [
    # 1D @ 1D (dot product)
    (np.array([1.0, -0.0]), np.array([-0.0, 1.0])),
    (np.array([-0.0, -0.0]), np.array([1.0, 1.0])),
    # 2D @ 2D with signed zeros
    (np.array([[1.0, -0.0], [-0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])),
]
all_pass = True
for a, b in tests:
    fnp_result = fnp.matmul(a, b)
    np_result = np.matmul(a, b)
    fnp_signs = np.signbit(fnp_result)
    np_signs = np.signbit(np_result)
    if not np.array_equal(fnp_signs, np_signs):
        print(f"FAIL: matmul signbit mismatch")
        print(f"  fnp signbit={fnp_signs.tolist()}")
        print(f"  np signbit={np_signs.tolist()}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: matmul values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "matmul signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_matmul_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy integer matmul uses a naive serial loop (no BLAS); the native parallel ikj
    // GEMM must be byte-identical — incl. overflow WRAP (numpy accumulates in the input
    // dtype with wrapping multiply+add). Sizes straddle the 1<<18 work gate.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(11)
ok = True
# bit-exactness across widths + shapes (square, rectangular, non-square inner)
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint8]:
    for (m, k, n) in [(96, 96, 96), (128, 200, 96), (65, 130, 257)]:
        info = np.iinfo(dt)
        a = rng.integers(info.min // 2, info.max // 2, (m, k)).astype(dt)
        b = rng.integers(info.min // 2, info.max // 2, (k, n)).astype(dt)
        r = fnp.matmul(a, b); e = np.matmul(a, b)
        ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
        # @ operator routes through matmul too
        ok = ok and (a @ b).tobytes() == e.tobytes()

# explicit overflow-wrap case (int64): large values so products+sums wrap
a = np.full((80, 80), 5_000_000_000, dtype=np.int64)
b = np.full((80, 80), 5_000_000_000, dtype=np.int64)
ok = ok and fnp.matmul(a, b).tobytes() == np.matmul(a, b).tobytes()

# int8 saturating-wrap case
a = np.full((70, 70), 100, dtype=np.int8)
b = np.full((70, 70), 100, dtype=np.int8)
ok = ok and fnp.matmul(a, b).tobytes() == np.matmul(a, b).tobytes()

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer matmul must be bit-identical to numpy (incl. wrap): {result}"
    );
    Ok(())
}
