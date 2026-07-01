//! Conformance tests for numpy.cross against NumPy oracle.
//!
//! Tests cross (cross product of two vectors).

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
fn cross_3d_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross 3D vectors should match numpy");
    Ok(())
}

#[test]
fn cross_2d_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2])
b = np.array([3, 4])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross 2D vectors should match numpy (returns scalar)"
    );
    Ok(())
}

#[test]
fn cross_batch_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross batch 3D should match numpy");
    Ok(())
}

#[test]
fn cross_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = fnp.cross(a, b)
expected = np.cross(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross float should match numpy");
    Ok(())
}

#[test]
fn cross_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
b = np.array([4+1j, 5+2j, 6+3j], dtype=np.complex128)
fnp_result = fnp.cross(a, b)
np_result = np.cross(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cross complex should match numpy");
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
            || stderr.contains("incompatible")
            || stderr.contains("dimension")
        {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn cross_4d_raises_valueerror() {
    let fnp_err = classify_error(&fnp_script(
        r#"
a = fnp.arange(4)
b = fnp.arange(4)
fnp.cross(a, b)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(4)
b = np.arange(4)
np.cross(a, b)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "cross with 4D vectors should raise same error as numpy"
    );
}

#[test]
fn cross_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, 1.0, np.nan])
b = np.array([1.0, 2.0, 3.0])
fnp_result = fnp.cross(a, b)
np_result = np.cross(a, b)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross special values should match numpy"
    );
    Ok(())
}

#[test]
fn cross_zero_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.0, 0.0, 0.0])
b = np.array([1.0, 2.0, 3.0])
fnp_result = fnp.cross(a, b)
np_result = np.cross(a, b)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross zero vector should match numpy"
    );
    Ok(())
}

#[test]
fn cross_parallel_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Cross product of parallel vectors is zero
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])
fnp_result = fnp.cross(a, b)
np_result = np.cross(a, b)
print(np.allclose(fnp_result, np_result) and np.allclose(fnp_result, 0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross parallel vectors should be zero"
    );
    Ok(())
}

#[test]
fn cross_unit_vectors() -> Result<(), String> {
    let script = fnp_script(
        r#"
# i x j = k
i = np.array([1.0, 0.0, 0.0])
j = np.array([0.0, 1.0, 0.0])
k = np.array([0.0, 0.0, 1.0])
fnp_result = fnp.cross(i, j)
np_result = np.cross(i, j)
print(np.allclose(fnp_result, k) and np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross unit vectors should follow right-hand rule"
    );
    Ok(())
}

#[test]
fn cross_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Cross product signed-zero parity
# Cross product involves multiplication and subtraction
tests = [
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),      # Standard case
    ([-0.0, -0.0, 1.0], [1.0, 0.0, 0.0]),    # Negative zeros in first
    ([0.0, 0.0, 1.0], [-0.0, 0.0, 0.0]),     # Negative zero in second
    ([0.0, -0.0, 0.0], [0.0, 0.0, 1.0]),     # Mixed zeros, cross with z-axis
]
all_pass = True
for a_vals, b_vals in tests:
    a = np.array(a_vals)
    b = np.array(b_vals)
    fnp_result = fnp.cross(a, b)
    np_result = np.cross(a, b)
    fnp_signs = np.signbit(fnp_result).tolist()
    np_signs = np.signbit(np_result).tolist()
    if fnp_signs != np_signs:
        print(f"FAIL: cross({a_vals}, {b_vals})")
        print(f"  fnp signbit={fnp_signs} np signbit={np_signs}")
        all_pass = False
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: cross({a_vals}, {b_vals}) values mismatch")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cross signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn cross_f32_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = (1 << 21) // 3 + 31
base = np.linspace(-1024.0, 1024.0, n * 3, dtype=np.float32).reshape(n, 3)
a = base.copy()
b = np.empty_like(a)
b[:, 0] = base[:, 1] * np.float32(0.5) + np.float32(3.0)
b[:, 1] = base[:, 2] * np.float32(-0.25) + np.float32(1.0)
b[:, 2] = base[:, 0] * np.float32(1.5) - np.float32(2.0)
actual = fnp.cross(a, b)
expected = np.cross(a, b)
print(
    actual.dtype == expected.dtype
    and actual.shape == expected.shape
    and actual.flags["C_CONTIGUOUS"]
    and actual.flags["WRITEABLE"]
    and actual.tobytes() == expected.tobytes()
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large f32 cross parallel path should be byte-exact"
    );
    Ok(())
}

// The native per-column parallel cross for the (3, N) axis=0 layout (components stored column-wise) must
// be byte-identical to numpy across dtypes, axis spellings (axis=0 / axisa=axisb=axisc=0 / axis=-2), and
// non-finite propagation, and every non-engaging shape/dtype/layout must still match via delegation.
#[test]
fn cross_axis0_3n_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
ok = True
rng = np.random.default_rng(20260701)
# Value equality (shape+dtype+bytes). Delegated defer paths may inherit numpy's own non-C-contiguity,
# so C-contiguity is asserted separately only for the engaged (native-output) axis=0 path below.
def bx(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return x.shape == y.shape and x.dtype == y.dtype \
        and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for dt in (np.float64, np.float32):
        for N in (1, 7, 1000, 700003, 4_000_000):
            a = rng.standard_normal((3, N)).astype(dt)
            b = rng.standard_normal((3, N)).astype(dt)
            for kw in (dict(axis=0), dict(axisa=0, axisb=0, axisc=0), dict(axis=-2)):
                r = fnp.cross(a, b, **kw)
                if not bx(r, np.cross(a, b, **kw)):
                    ok = False
                if not np.asarray(r).flags["C_CONTIGUOUS"]:  # engaged native output must be C-contiguous
                    ok = False
        # non-finite propagation
        a = rng.standard_normal((3, 1_000_000)).astype(dt); a[0, 5] = np.inf; a[1, 7] = np.nan
        b = rng.standard_normal((3, 1_000_000)).astype(dt); b[2, 9] = -np.inf
        if not bx(fnp.cross(a, b, axis=0), np.cross(a, b, axis=0)):
            ok = False
        # defer paths must still equal numpy
        a3 = rng.standard_normal((500000, 3)).astype(dt); b3 = rng.standard_normal((500000, 3)).astype(dt)
        if not bx(fnp.cross(a3, b3), np.cross(a3, b3)):  # (N,3) axis=-1 existing path
            ok = False
        c = rng.standard_normal((3, 300000)).astype(dt); d = rng.standard_normal((3, 300000)).astype(dt)
        # axisa=axisb=0 with default axisc=-1 -> (N,3) output, different code path
        if not bx(fnp.cross(c, d, axisa=0, axisb=0), np.cross(c, d, axisa=0, axisb=0)):
            ok = False
        # F-contiguous (3,N) axis=0 -> defer
        cf = np.asfortranarray(c); df = np.asfortranarray(d)
        if not bx(fnp.cross(cf, df, axis=0), np.cross(cf, df, axis=0)):
            ok = False
        # 3-D axis=0 -> defer
        e = rng.standard_normal((3, 200, 50)).astype(dt); g = rng.standard_normal((3, 200, 50)).astype(dt)
        if not bx(fnp.cross(e, g, axis=0), np.cross(e, g, axis=0)):
            ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native (3,N) axis=0 cross must be byte-exact vs numpy across dtypes/axes/defers"
    );
    Ok(())
}
