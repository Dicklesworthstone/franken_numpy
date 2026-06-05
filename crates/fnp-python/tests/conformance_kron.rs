//! Conformance tests for numpy.kron against NumPy oracle.
//!
//! Tests kron (Kronecker product).

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
fn kron_2d_matrices() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron 2D matrices should match numpy");
    Ok(())
}

#[test]
fn kron_1d_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron 1D vectors should match numpy");
    Ok(())
}

#[test]
fn kron_mixed_shapes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2], [3, 4]])
b = np.array([1, 2, 3])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron with mixed shapes should match numpy"
    );
    Ok(())
}

#[test]
fn kron_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[0.5, 1.5], [2.5, 3.5]])
result = fnp.kron(a, b)
expected = np.kron(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron float should match numpy");
    Ok(())
}

#[test]
fn kron_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1+1j, 2]], dtype=np.complex128)
b = np.array([[1], [2+1j]], dtype=np.complex128)
fnp_result = fnp.kron(a, b)
np_result = np.kron(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron complex should match numpy");
    Ok(())
}

#[test]
fn kron_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[np.inf, 1.0], [np.nan, 2.0]])
b = np.array([[1.0, 2.0]])
fnp_result = fnp.kron(a, b)
np_result = np.kron(a, b)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron special values should match numpy"
    );
    Ok(())
}

#[test]
fn kron_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[3.0]])
b = np.array([[5.0]])
fnp_result = fnp.kron(a, b)
np_result = np.kron(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron single element should match numpy"
    );
    Ok(())
}

#[test]
fn kron_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.eye(2)
b = np.eye(3)
fnp_result = fnp.kron(a, b)
np_result = np.kron(a, b)
# Kronecker product of identity matrices gives larger identity-like structure
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron identity matrices should match numpy"
    );
    Ok(())
}

#[test]
fn kron_3d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(8).reshape(2, 2, 2)
b = np.arange(4).reshape(2, 2)
fnp_result = fnp.kron(a, b)
np_result = np.kron(a, b)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kron 3D arrays should match numpy");
    Ok(())
}

#[test]
fn kron_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Kronecker product signed-zero parity (multiplication-based)
# kron(a, b)[i*len(b)+j] = a[i] * b[j]
tests = [
    ([1.0, -0.0], [1.0, 2.0]),      # -0 * positive = -0
    ([0.0, 1.0], [-0.0, 2.0]),      # positive * -0 = -0
    ([-0.0, -0.0], [1.0, -1.0]),    # -0 * negative = 0, -0 * positive = -0
    ([0.0, 0.0], [-0.0, -0.0]),     # 0 * -0 = -0
]
all_pass = True
for a_vals, b_vals in tests:
    a = np.array(a_vals)
    b = np.array(b_vals)
    fnp_result = fnp.kron(a, b)
    np_result = np.kron(a, b)
    fnp_signs = np.signbit(fnp_result).tolist()
    np_signs = np.signbit(np_result).tolist()
    if fnp_signs != np_signs:
        print(f"FAIL: kron({a_vals}, {b_vals})")
        print(f"  fnp signbit={fnp_signs} np signbit={np_signs}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: kron({a_vals}, {b_vals}) values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "kron signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

/// Locks the zero-copy 1-D Kronecker product fast path (`try_zerocopy_f64_kron1d`,
/// out[i*m+j] = a[i]*b[j]) to bit-exact parity with numpy. kron writes the
/// products verbatim, so parity must hold at the IEEE-754 bit level (signed zero,
/// nan, inf). Compares the sha256 of raw output bytes across rectangular 1-D
/// shapes and extreme values.
#[test]
fn kron_1d_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n, m in [(500, 500), (100, 1000), (777, 333)]:
    chunks.append(np.asarray(mod.kron(rng.standard_normal(n), rng.standard_normal(m))).tobytes())
xe = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e308], dtype=np.float64)
chunks.append(np.asarray(mod.kron(xe, np.array([1.0, -0.0, np.inf, 0.0, -2.0], dtype=np.float64))).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!("import numpy as np\n{}", body.replace("MODULE", "np")))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy 1-D kron must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
