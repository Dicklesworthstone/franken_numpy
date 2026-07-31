//! Conformance tests for the fused `multiply_add` lever against the NumPy oracle.
//!
//! `fnp.multiply_add(a, b, c)` computes `a * b + c` in one pass. NumPy has no
//! fused public form, so the oracle is NumPy's own two-call expression. The
//! contract is BYTE equality, not closeness: the fused pass keeps both roundings
//! (multiply rounds, then add rounds) exactly as the two ufuncs do, and must
//! never contract into a single-rounding FMA.

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

/// The routed regime: above the parallel gate, so the Rayon banded path runs.
#[test]
fn multiply_add_f64_large_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(7)
a = rng.standard_normal(300_000)
b = rng.standard_normal(300_000)
c = rng.standard_normal(300_000)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Below the parallel gate, so the serial path runs. Same byte contract.
#[test]
fn multiply_add_f64_small_serial_path_is_byte_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(11)
a = rng.standard_normal(1000)
b = rng.standard_normal(1000)
c = rng.standard_normal(1000)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.dtype == theirs.dtype, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// THE FMA TRAP. These operands are chosen so that a single-rounding fused
/// multiply-add gives a DIFFERENT answer from multiply-then-add. If the kernel
/// ever contracts, this test fails and the byte-equality claim dies with it.
#[test]
fn multiply_add_does_not_contract_into_a_single_rounding_fma() -> Result<(), String> {
    let script = fnp_script(
        r#"
# a*b is inexact in f64; adding -(a*b rounded) exposes the rounding residue.
# With two roundings the result is exactly 0.0; a true FMA returns the residue.
a = np.array([1.0 + 2.0**-52, 1.0 + 2.0**-52, 0.1, 1e200], dtype=np.float64)
b = np.array([1.0 + 2.0**-52, 1.0 - 2.0**-52, 0.1, 1e200], dtype=np.float64)
c = -(a * b)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.tobytes() == theirs.tobytes(),
      bool(np.all(ours[:3] == 0.0)),
      bool(np.isnan(ours[3])))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True True");
    Ok(())
}

/// Non-finite propagation must match element for element, including NaN bits.
#[test]
fn multiply_add_non_finite_matches_numpy_bitwise() -> Result<(), String> {
    let script = fnp_script(
        r#"
inf = np.inf
nan = np.nan
a = np.array([inf, -inf, nan, 0.0, -0.0, inf, 1.0, -0.0], dtype=np.float64)
b = np.array([0.0, 0.0, 1.0, inf, inf, inf, nan, -0.0], dtype=np.float64)
c = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -inf, 0.0, 0.0], dtype=np.float64)
with np.errstate(all='ignore'):
    ours = fnp.multiply_add(a, b, c)
    theirs = a * b + c
print(ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True");
    Ok(())
}

#[test]
fn multiply_add_f32_is_byte_identical_and_keeps_dtype() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(3)
a = rng.standard_normal(200_000).astype(np.float32)
b = rng.standard_normal(200_000).astype(np.float32)
c = rng.standard_normal(200_000).astype(np.float32)
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(str(ours.dtype), ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "float32 True");
    Ok(())
}

/// Every fixed-width integer dtype takes the fused path. Full-range random
/// bytes exercise wrapping overflow rather than merely the no-overflow subset.
#[test]
fn multiply_add_all_integer_widths_match_numpy_with_overflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(17)
checks = []
for dtype in [np.int8, np.uint8, np.int16, np.uint16,
              np.int32, np.uint32, np.int64, np.uint64]:
    dt = np.dtype(dtype)
    n = 300_000
    a = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    b = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    c = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    with np.errstate(over='ignore'):
        ours = fnp.multiply_add(a, b, c)
        theirs = a * b + c
    checks.append(ours.dtype == theirs.dtype and ours.tobytes() == theirs.tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 8");
    Ok(())
}

/// Multi-dimensional input must come back with the original shape, not flat.
#[test]
fn multiply_add_preserves_multidimensional_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(5)
a = rng.standard_normal((400, 900))
b = rng.standard_normal((400, 900))
c = rng.standard_normal((400, 900))
ours = fnp.multiply_add(a, b, c)
theirs = a * b + c
print(ours.shape == theirs.shape, ours.tobytes() == theirs.tobytes())
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True True");
    Ok(())
}

/// Everything outside the routed regime must defer to NumPy's own expression
/// and therefore still agree: mixed dtypes, broadcasting, non-contiguity,
/// mixed integer widths, and complex.
#[test]
fn multiply_add_deferred_regimes_still_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(13)
checks = []

# broadcasting (shapes differ) -> deferred
a = rng.standard_normal((512, 4))
b = rng.standard_normal((4,))
c = rng.standard_normal((512, 1))
checks.append(np.array_equal(fnp.multiply_add(a, b, c), a * b + c, equal_nan=True))

# non-contiguous view -> deferred
base = rng.standard_normal((256, 8))
a2 = base[:, ::2]
b2 = base[:, 1::2]
c2 = base[:, ::2]
checks.append(np.array_equal(fnp.multiply_add(a2, b2, c2), a2 * b2 + c2, equal_nan=True))

# mixed float widths -> deferred, numpy owns promotion
a3 = rng.standard_normal(1000).astype(np.float32)
b3 = rng.standard_normal(1000)
c3 = rng.standard_normal(1000)
out3 = fnp.multiply_add(a3, b3, c3)
ref3 = a3 * b3 + c3
checks.append(out3.dtype == ref3.dtype and out3.tobytes() == ref3.tobytes())

# mixed integer widths -> deferred, numpy owns promotion
a4 = rng.integers(-1000, 1000, size=5000, dtype=np.int32)
b4 = rng.integers(-1000, 1000, size=5000, dtype=np.int64)
c4 = rng.integers(-1000, 1000, size=5000, dtype=np.int64)
out4 = fnp.multiply_add(a4, b4, c4)
ref4 = a4 * b4 + c4
checks.append(out4.dtype == ref4.dtype and out4.tobytes() == ref4.tobytes())

# complex dtype -> deferred
a5 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
b5 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
c5 = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
out5 = fnp.multiply_add(a5, b5, c5)
ref5 = a5 * b5 + c5
checks.append(out5.dtype == ref5.dtype and out5.tobytes() == ref5.tobytes())

# python lists -> deferred
checks.append(np.array_equal(fnp.multiply_add([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]),
                             np.array([8.0, 14.0])))

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 6");
    Ok(())
}
