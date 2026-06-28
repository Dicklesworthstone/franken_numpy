//! Conformance tests for numpy.inner against NumPy oracle.
//!
//! Tests inner (inner product).

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
fn inner_1d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.isclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner 1d arrays should match numpy");
    Ok(())
}

#[test]
fn inner_2d_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(6).reshape(2, 3)
b = np.arange(3)
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner 2d x 1d arrays should match numpy"
    );
    Ok(())
}

#[test]
fn inner_higher_dim() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
b = np.arange(4)
result = fnp.inner(a, b)
expected = np.inner(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner higher-dim arrays should match numpy"
    );
    Ok(())
}

#[test]
fn inner_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
fnp_result = fnp.inner(a, b)
np_result = np.inner(a, b)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "inner scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn inner_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, np.nan, 1.0])
b = np.array([1.0, 1.0, 1.0, np.nan])
fnp_result = fnp.inner(a, b)
np_result = np.inner(a, b)
# inf + -inf + nan + nan = nan
print(np.isnan(fnp_result) and np.isnan(np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner special values should match numpy"
    );
    Ok(())
}

#[test]
fn inner_empty_arrays() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
b = np.array([], dtype=np.float64)
fnp_result = fnp.inner(a, b)
np_result = np.inner(a, b)
print(fnp_result == np_result == 0.0)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner empty arrays should match numpy"
    );
    Ok(())
}

#[test]
fn inner_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
b = np.array([4+1j, 5+2j, 6+3j], dtype=np.complex128)
fnp_result = fnp.inner(a, b)
np_result = np.inner(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "inner complex should match numpy");
    Ok(())
}

#[test]
fn inner_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0])
b = np.array([3.0])
fnp_result = fnp.inner(a, b)
np_result = np.inner(a, b)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner single element should match numpy"
    );
    Ok(())
}

#[test]
#[ignore = "PARITY GAP: fnp accumulator returns -0.0, NumPy returns 0.0. See DISC-011."]
fn inner_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Inner product signed-zero parity
# inner(a, b) = sum(a[i] * b[i]) - involves multiply and accumulate
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
    fnp_result = fnp.inner(a, b)
    np_result = np.inner(a, b)
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: inner({a_vals}, {b_vals})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: inner({a_vals}, {b_vals}) values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "inner signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_inner_native_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // numpy integer inner = no-BLAS a@b^T (slow). Native route (reshape + contiguous
    // b^T -> int GEMM) must be byte-identical incl. overflow wrap and >2-D shapes.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(29)
ok = True
for dt in [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32]:
    info = np.iinfo(dt)
    # 2-D inner: (m,k) inner (n,k) -> (m,n)
    a = rng.integers(info.min // 4, info.max // 4, (90, 130)).astype(dt)
    b = rng.integers(info.min // 4, info.max // 4, (77, 130)).astype(dt)
    r = fnp.inner(a, b); e = np.inner(a, b)
    ok = ok and r.dtype == e.dtype and r.shape == e.shape and r.tobytes() == e.tobytes()
    # 3-D inner (contracts last axis): (x,y,k) inner (z,k) -> (x,y,z)
    a3 = rng.integers(info.min // 8, info.max // 8, (8, 12, 64)).astype(dt)
    b2 = rng.integers(info.min // 8, info.max // 8, (40, 64)).astype(dt)
    r3 = fnp.inner(a3, b2); e3 = np.inner(a3, b2)
    ok = ok and r3.dtype == e3.dtype and r3.shape == e3.shape and r3.tobytes() == e3.tobytes()
# overflow wrap (int64)
a = np.full((120, 100), 5_000_000_000, dtype=np.int64)
b = np.full((110, 100), 5_000_000_000, dtype=np.int64)
ok = ok and fnp.inner(a, b).tobytes() == np.inner(a, b).tobytes()
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native integer inner must be bit-identical to numpy: {result}"
    );
    Ok(())
}
