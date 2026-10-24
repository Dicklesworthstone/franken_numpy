//! Conformance tests for numpy closeness comparison functions against NumPy oracle.
//!
//! Tests allclose, isclose.

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

mod support;
use support::fnp_script;

// ─────────────────────────────────────────────────────────────────────────────
// allclose
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose equal should match numpy");
    Ok(())
}

#[test]
fn allclose_close() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose close should match numpy");
    Ok(())
}

#[test]
fn allclose_not_close() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.1, 3.1])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
print(result == expected == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "allclose not close should match numpy"
    );
    Ok(())
}

#[test]
fn allclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.05, 2.1, 3.15])
result = fnp.allclose(a, b, rtol=0.1)
expected = np.allclose(a, b, rtol=0.1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "allclose custom rtol should match numpy"
    );
    Ok(())
}

#[test]
fn allclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.01, 2.01, 3.01])
result = fnp.allclose(a, b, atol=0.1)
expected = np.allclose(a, b, atol=0.1)
print(result == expected)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "allclose custom atol should match numpy"
    );
    Ok(())
}

#[test]
fn allclose_with_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.allclose(a, b)
expected = np.allclose(a, b)
# By default, nan != nan, so allclose returns False
print(result == expected == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "allclose with nan default should match numpy"
    );
    Ok(())
}

#[test]
fn allclose_with_nan_equal_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.allclose(a, b, equal_nan=True)
expected = np.allclose(a, b, equal_nan=True)
print(result == expected == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "allclose equal_nan=True should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isclose
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isclose_equal() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose equal should match numpy");
    Ok(())
}

#[test]
fn isclose_mixed() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.5, 3.0])  # second element differs
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose mixed should match numpy");
    Ok(())
}

#[test]
fn isclose_custom_rtol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.05, 2.1, 3.5])
result = fnp.isclose(a, b, rtol=0.1)
expected = np.isclose(a, b, rtol=0.1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose custom rtol should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_custom_atol() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.001, 2.05, 3.2])
result = fnp.isclose(a, b, atol=0.1)
expected = np.isclose(a, b, atol=0.1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose custom atol should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_with_nan_default() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose with nan default should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_with_nan_equal_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0])
b = np.array([1.0, np.nan, 3.0])
result = fnp.isclose(a, b, equal_nan=True)
expected = np.isclose(a, b, equal_nan=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "isclose equal_nan=True should match numpy"
    );
    Ok(())
}

#[test]
fn isclose_with_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, -np.inf])
b = np.array([1.0, np.inf, -np.inf])
result = fnp.isclose(a, b)
expected = np.isclose(a, b)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose with inf should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn allclose_isclose_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9])
# allclose should be True iff all elements of isclose are True
allclose_result = fnp.allclose(a, b)
isclose_result = fnp.isclose(a, b)
print(allclose_result == np.all(isclose_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "allclose should equal all(isclose)");
    Ok(())
}

#[test]
fn isclose_symmetry() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.0, 3.3])
result_ab = fnp.isclose(a, b)
result_ba = fnp.isclose(b, a)
print(np.array_equal(result_ab, result_ba))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "isclose should be symmetric");
    Ok(())
}

#[test]
fn isclose_parallel_gate_size_bit_exact() -> Result<(), String> {
    // The zero-copy array-array f64/f32 isclose kernels parallelize above
    // 1<<20 elements (2026-07-12); the per-element predicate is independent,
    // so chunks are byte-identical - this locks it at gate size with planted
    // inf/nan/signed-zero/huge pairs across both dtypes and equal_nan modes.
    let script = fnp_script(
        r#"
import numpy as np
verdicts = []
rng = np.random.default_rng(20260716)
for dt in (np.float64, np.float32):
    a = rng.standard_normal(2_000_000).astype(dt)
    b = (a + rng.standard_normal(2_000_000) * 1e-7).astype(dt)
    a[5] = np.inf; b[5] = np.inf
    a[6] = np.inf; b[6] = -np.inf
    a[7] = np.nan; b[7] = np.nan
    a[8] = np.nan; b[8] = dt(1.0)
    a[10] = dt(0.0); b[10] = dt(-0.0)
    for eqn in (False, True):
        r = fnp.isclose(a, b, equal_nan=eqn); e = np.isclose(a, b, equal_nan=eqn)
        if r.dtype != e.dtype or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL {dt.__name__} eqn={eqn}")
    if fnp.allclose(a, a) != np.allclose(a, a):
        verdicts.append(f"FAIL allclose {dt.__name__}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "parallel isclose must be bit-identical at gate size: {result}"
    );
    Ok(())
}

/// The array-array kernels evaluate numpy's own expression branch-free, `(|x - y| <= atol +
/// rtol * |y|) & isfinite(y) | (x == y)` plus `isnan(x) & isnan(y)` under equal_nan: every pair
/// of ten special values (signed zeros, infinities, NaN, a subnormal-range value, 1e30) in both
/// operand orders, near-boundary pairs, six tolerance settings, float32 and float64, serial and
/// pooled sizes (72 cells).
#[test]
fn isclose_special_value_grid_matches_numpy_serial_and_pooled() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(17)
bad, cells = [], 0
special = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-8, 1e30, 5e-324 * 1e300], dtype=np.float64)
xs, ys = np.meshgrid(special, special)
for dt in (np.float32, np.float64):
    grid_x, grid_y = xs.ravel().astype(dt), ys.ravel().astype(dt)
    for n in (100, 4096, (1 << 21) + 7):
        base = rng.standard_normal(n).astype(dt)
        near = (base * (1 + rng.uniform(0.999, 1.001, n) * 1e-5)).astype(dt)
        near[: grid_x.size] = grid_y
        base[: grid_x.size] = grid_x
        for kw in ({}, {"equal_nan": True}, {"rtol": 0, "atol": 0}, {"rtol": 1e-3, "atol": 1e-6}, {"atol": 0.5},
                   {"rtol": 0.0, "atol": 1e-9, "equal_nan": True}):
            for a, b in ((base, near), (near, base)):
                cells += 1
                e, g = np.isclose(a, b, **kw), fnp.isclose(a, b, **kw)
                if e.dtype != g.dtype or e.shape != g.shape or not np.array_equal(e, g):
                    bad.append(f"{dt.__name__} {n} {kw} diff={int((e != g).sum())}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "72", "cell table drifted: {result}");
    assert_eq!(bad, "[]", "isclose must match numpy: {result}");
    Ok(())
}
