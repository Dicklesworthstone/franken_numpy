//! Metamorphic tests for array operations.
//!
//! Tests mathematical properties that should hold regardless of input:
//! - transpose(transpose(x)) == x
//! - sort(reverse(x)) == sort(x)
//! - abs(abs(x)) == abs(x) (idempotent)
//! - negative(negative(x)) == x (involution)
//! - exp(log(x)) == x for x > 0
//! - log(exp(x)) == x
//! - sin(x)**2 + cos(x)**2 == 1
//! - arcsin(sin(x)) == x for x in [-pi/2, pi/2]

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
fn transpose_transpose_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(5, 7)
result = fnp.transpose(fnp.transpose(x))
print(np.allclose(x, result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "transpose(transpose(x)) should equal x"
    );
    Ok(())
}

#[test]
fn transpose_transpose_3d_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(3, 4, 5)
result = fnp.transpose(fnp.transpose(x))
print(np.allclose(x, result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "transpose(transpose(x)) should equal x for 3D"
    );
    Ok(())
}

#[test]
fn sort_reverse_equals_sort() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
result1 = fnp.sort(x[::-1].copy())
result2 = fnp.sort(x)
print(np.allclose(result1, result2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sort(reverse(x)) should equal sort(x)"
    );
    Ok(())
}

#[test]
fn abs_is_idempotent() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100) * 100
result = fnp.abs(fnp.abs(x))
expected = fnp.abs(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "abs(abs(x)) should equal abs(x)");
    Ok(())
}

#[test]
fn negative_negative_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100) * 100
result = fnp.negative(fnp.negative(x))
print(np.allclose(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "negative(negative(x)) should equal x"
    );
    Ok(())
}

#[test]
fn exp_log_roundtrip_for_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.1, 100, 100)
result = fnp.exp(fnp.log(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "exp(log(x)) should equal x for positive x"
    );
    Ok(())
}

#[test]
fn log_exp_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
result = fnp.log(fnp.exp(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log(exp(x)) should equal x");
    Ok(())
}

#[test]
fn pythagorean_identity_sin_cos() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
result = fnp.sin(x)**2 + fnp.cos(x)**2
print(np.allclose(result, 1.0, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sin(x)**2 + cos(x)**2 should equal 1"
    );
    Ok(())
}

#[test]
fn arcsin_sin_roundtrip_in_domain() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-np.pi/2 + 0.01, np.pi/2 - 0.01, 100)
result = fnp.arcsin(fnp.sin(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arcsin(sin(x)) should equal x for x in [-pi/2, pi/2]"
    );
    Ok(())
}

#[test]
fn arccos_cos_roundtrip_in_domain() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, np.pi - 0.01, 100)
result = fnp.arccos(fnp.cos(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arccos(cos(x)) should equal x for x in [0, pi]"
    );
    Ok(())
}

#[test]
fn arctan_tan_roundtrip_in_domain() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-np.pi/2 + 0.1, np.pi/2 - 0.1, 100)
result = fnp.arctan(fnp.tan(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arctan(tan(x)) should equal x for x in (-pi/2, pi/2)"
    );
    Ok(())
}

#[test]
fn sinh_cosh_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-5, 5, 100)
# cosh(x)**2 - sinh(x)**2 = 1
result = fnp.cosh(x)**2 - fnp.sinh(x)**2
print(np.allclose(result, 1.0, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cosh(x)**2 - sinh(x)**2 should equal 1"
    );
    Ok(())
}

#[test]
fn square_sqrt_roundtrip_for_positive() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, 100, 100)
result = fnp.sqrt(fnp.square(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sqrt(square(x)) should equal x for positive x"
    );
    Ok(())
}

#[test]
fn cbrt_power3_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-100, 100, 100)
result = fnp.cbrt(x**3)
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cbrt(x**3) should equal x");
    Ok(())
}

#[test]
fn maximum_minimum_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# maximum(x, y) + minimum(x, y) = x + y
max_vals = fnp.maximum(x, y)
min_vals = fnp.minimum(x, y)
result = max_vals + min_vals
expected = x + y
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "maximum(x,y) + minimum(x,y) should equal x + y"
    );
    Ok(())
}

#[test]
fn fmax_fmin_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# fmax(x, y) + fmin(x, y) = x + y (for non-NaN)
max_vals = fnp.fmax(x, y)
min_vals = fnp.fmin(x, y)
result = max_vals + min_vals
expected = x + y
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmax(x,y) + fmin(x,y) should equal x + y for non-NaN"
    );
    Ok(())
}

#[test]
fn log2_exp2_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
result = fnp.log2(fnp.exp2(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log2(exp2(x)) should equal x");
    Ok(())
}

#[test]
fn log10_power10_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-5, 5, 100)
result = fnp.log10(10.0**x)
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log10(10**x) should equal x");
    Ok(())
}

#[test]
fn hypot_pythagorean() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# hypot(x, y)**2 = x**2 + y**2
result = fnp.hypot(x, y)**2
expected = x**2 + y**2
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot(x,y)**2 should equal x**2 + y**2"
    );
    Ok(())
}

#[test]
fn arctan2_atan_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
# For positive x, arctan2(y, x) = arctan(y/x)
y = np.random.randn(100)
x = np.random.uniform(0.1, 10, 100)  # positive x only
result = fnp.arctan2(y, x)
expected = fnp.arctan(y / x)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "arctan2(y,x) should equal arctan(y/x) for positive x"
    );
    Ok(())
}

#[test]
fn divmod_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
a = np.random.uniform(-100, 100, 100)
b = np.random.uniform(1, 10, 100)  # avoid zero
q, r = fnp.divmod(a, b)
# invariant: a == floor(a/b) * b + remainder
reconstructed = q * b + r
print(np.allclose(reconstructed, a, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "divmod invariant: a == q * b + r");
    Ok(())
}

#[test]
fn reciprocal_product_is_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, 100, 100)  # positive, non-zero
result = x * fnp.reciprocal(x)
print(np.allclose(result, 1.0, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "x * reciprocal(x) should equal 1");
    Ok(())
}

#[test]
fn expm1_log1p_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
# For small x, log1p(expm1(x)) == x
x = np.random.uniform(-0.5, 0.5, 100)
result = fnp.log1p(fnp.expm1(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "log1p(expm1(x)) should equal x for small x"
    );
    Ok(())
}

#[test]
fn deg2rad_rad2deg_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-360, 360, 100)
result = fnp.rad2deg(fnp.deg2rad(x))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "rad2deg(deg2rad(x)) should equal x");
    Ok(())
}

#[test]
fn clip_equals_min_max_composition() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-10, 10, 100)
lo, hi = -5.0, 5.0
clipped = fnp.clip(x, lo, hi)
composed = fnp.minimum(fnp.maximum(x, lo), hi)
print(np.allclose(clipped, composed, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip(x, lo, hi) == minimum(maximum(x, lo), hi)"
    );
    Ok(())
}

#[test]
fn logaddexp_with_neg_inf_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-100, 100, 100)
result = fnp.logaddexp(x, np.full_like(x, -np.inf))
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "logaddexp(x, -inf) should equal x");
    Ok(())
}

#[test]
fn logaddexp_equal_args_plus_log2() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-50, 50, 100)
result = fnp.logaddexp(x, x)
expected = x + np.log(2)
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "logaddexp(x, x) should equal x + log(2)"
    );
    Ok(())
}

#[test]
fn sign_abs_product_is_identity_for_nonzero() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, 100, 100) * np.random.choice([-1, 1], 100)
result = fnp.sign(x) * fnp.abs(x)
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sign(x) * abs(x) should equal x for non-zero x"
    );
    Ok(())
}

#[test]
fn copysign_abs_restores_original() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(0.01, 100, 100) * np.random.choice([-1, 1], 100)
result = fnp.copysign(fnp.abs(x), x)
print(np.allclose(result, x, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "copysign(abs(x), x) should equal x for non-zero x"
    );
    Ok(())
}

#[test]
fn hypot_with_zero_is_abs() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-100, 100, 100)
result1 = fnp.hypot(x, np.zeros_like(x))
result2 = fnp.hypot(np.zeros_like(x), x)
expected = fnp.abs(x)
print(np.allclose(result1, expected) and np.allclose(result2, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hypot(x, 0) and hypot(0, x) should equal abs(x)"
    );
    Ok(())
}

#[test]
fn floor_le_x_le_ceil() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-100, 100, 100)
floored = fnp.floor(x)
ceiled = fnp.ceil(x)
print(np.all(floored <= x) and np.all(x <= ceiled))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "floor(x) <= x <= ceil(x)");
    Ok(())
}

#[test]
fn floor_plus_one_gt_x_unless_integer() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-100, 100, 100)
is_integer = (x == fnp.floor(x))
floored_plus_one = fnp.floor(x) + 1
result = np.logical_or(is_integer, floored_plus_one > x)
print(np.all(result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "floor(x) + 1 > x unless x is integer"
    );
    Ok(())
}

#[test]
fn cumsum_of_ones_is_arange() -> Result<(), String> {
    let script = fnp_script(
        r#"
ones = np.ones(100)
result = fnp.cumsum(ones)
expected = np.arange(1, 101, dtype=float)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumsum([1,1,1,...]) should equal [1,2,3,...]"
    );
    Ok(())
}

#[test]
fn prod_of_exp_is_exp_of_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.uniform(-3, 3, 20)
result = fnp.prod(fnp.exp(x))
expected = np.exp(fnp.sum(x))
print(np.allclose(result, expected, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "prod(exp(x)) should equal exp(sum(x))"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// where metamorphic relationships
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn where_inverted_condition_swaps_x_y() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
cond = np.random.randint(0, 2, 50).astype(bool)
x = np.random.randn(50)
y = np.random.randn(50)
result1 = fnp.where(cond, x, y)
result2 = fnp.where(~cond, y, x)
print(np.allclose(result1, result2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "where(cond, x, y) == where(~cond, y, x)"
    );
    Ok(())
}

#[test]
fn where_all_true_returns_x() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(30)
y = np.random.randn(30)
cond = np.ones(30, dtype=bool)
result = fnp.where(cond, x, y)
print(np.allclose(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where(all_true, x, y) == x");
    Ok(())
}

#[test]
fn where_all_false_returns_y() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(30)
y = np.random.randn(30)
cond = np.zeros(30, dtype=bool)
result = fnp.where(cond, x, y)
print(np.allclose(result, y))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where(all_false, x, y) == y");
    Ok(())
}

#[test]
fn where_same_x_y_returns_that_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
cond = np.random.randint(0, 2, 40).astype(bool)
v = np.random.randn(40)
result = fnp.where(cond, v, v)
print(np.allclose(result, v))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "where(cond, v, v) == v");
    Ok(())
}

#[test]
fn where_1arg_count_equals_count_nonzero() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
indices = fnp.where(x > 0)
count = fnp.count_nonzero(x > 0)
print(len(indices[0]) == count)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "len(where(cond)[0]) == count_nonzero(cond)"
    );
    Ok(())
}

#[test]
fn where_2d_inverted_condition_swaps() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
cond = np.random.randint(0, 2, (5, 7)).astype(bool)
x = np.random.randn(5, 7)
y = np.random.randn(5, 7)
result1 = fnp.where(cond, x, y)
result2 = fnp.where(~cond, y, x)
print(np.allclose(result1, result2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "2D where(cond, x, y) == where(~cond, y, x)"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// einsum / matmul metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn einsum_trace_equals_diag_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(5, 5)
trace_einsum = fnp.einsum('ii', A)
trace_diag = fnp.sum(fnp.diag(A))
print(np.allclose(trace_einsum, trace_diag))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum('ii', A) == sum(diag(A))");
    Ok(())
}

#[test]
fn einsum_transpose_equals_transpose() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 6)
via_einsum = fnp.einsum('ij->ji', A)
via_transpose = fnp.transpose(A)
print(np.allclose(via_einsum, via_transpose))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "einsum('ij->ji', A) == transpose(A)");
    Ok(())
}

#[test]
fn einsum_matmul_equals_matmul() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
via_einsum = fnp.einsum('ij,jk->ik', A, B)
via_matmul = fnp.matmul(A, B)
print(np.allclose(via_einsum, via_matmul))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "einsum('ij,jk->ik', A, B) == matmul(A, B)"
    );
    Ok(())
}

#[test]
fn matmul_associativity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.random.randn(5, 2)
left = fnp.matmul(fnp.matmul(A, B), C)
right = fnp.matmul(A, fnp.matmul(B, C))
print(np.allclose(left, right))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "(A @ B) @ C == A @ (B @ C)");
    Ok(())
}

#[test]
fn matmul_transpose_reverses_order() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
left = fnp.transpose(fnp.matmul(A, B))
right = fnp.matmul(fnp.transpose(B), fnp.transpose(A))
print(np.allclose(left, right))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "(A @ B).T == B.T @ A.T");
    Ok(())
}

#[test]
fn sum_along_axes_equals_total_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 5, 6)
total_direct = fnp.sum(A)
total_via_axes = fnp.sum(fnp.sum(fnp.sum(A, axis=0), axis=0), axis=0)
print(np.allclose(total_direct, total_via_axes))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum(A) == sum(sum(sum(A, axis=0), axis=0), axis=0)"
    );
    Ok(())
}

#[test]
fn mean_is_sum_over_size() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 5, 6)
mean_direct = fnp.mean(A)
mean_via_sum = fnp.sum(A) / A.size
print(np.allclose(mean_direct, mean_via_sum))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "mean(A) == sum(A) / size");
    Ok(())
}

#[test]
fn dot_inner_equivalence_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
a = np.random.randn(10)
b = np.random.randn(10)
via_dot = fnp.dot(a, b)
via_einsum = fnp.einsum('i,i->', a, b)
print(np.allclose(via_dot, via_einsum))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "dot(a, b) == einsum('i,i->', a, b)");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// around/round metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn around_is_idempotent() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
a = np.random.randn(100) * 1000
for decimals in [-2, -1, 0, 1, 2, 3]:
    once = fnp.around(a, decimals=decimals)
    twice = fnp.around(once, decimals=decimals)
    if not np.array_equal(once, twice):
        print("False")
        break
else:
    print("True")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "around(around(x, n), n) == around(x, n)"
    );
    Ok(())
}

#[test]
fn around_zero_decimals_matches_rint() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
a = np.random.randn(100) * 10
around_result = fnp.around(a, decimals=0)
rint_result = fnp.rint(a)
print(np.array_equal(around_result, rint_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "around(x, 0) == rint(x)");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diff/cumsum metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn diff_cumsum_telescoping() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(50)
# sum of differences equals last - first
diffs = fnp.diff(x)
total_diff = fnp.sum(diffs)
expected = x[-1] - x[0]
print(np.allclose(total_diff, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum(diff(x)) == x[-1] - x[0]");
    Ok(())
}

#[test]
fn ediff1d_telescoping() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(50)
diffs = fnp.ediff1d(x)
total_diff = fnp.sum(diffs)
expected = x[-1] - x[0]
print(np.allclose(total_diff, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum(ediff1d(x)) == x[-1] - x[0]");
    Ok(())
}

#[test]
fn cumsum_diff_recovers_original_shifted() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(50)
cs = fnp.cumsum(x)
d = fnp.diff(cs)
# diff(cumsum(x)) == x[1:]
print(np.allclose(d, x[1:]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff(cumsum(x)) == x[1:]");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// searchsorted metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn searchsorted_left_right_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 2, 3, 4, 5])
v = 2
left = fnp.searchsorted(a, v, side='left')
right = fnp.searchsorted(a, v, side='right')
# All elements in a[left:right] should equal v
segment = a[left:right]
all_equal = np.all(segment == v)
# Count should match
count_in_segment = len(segment)
count_total = np.sum(a == v)
print(all_equal and count_in_segment == count_total)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "searchsorted left/right brackets all equal elements"
    );
    Ok(())
}

#[test]
fn searchsorted_insertion_preserves_sortedness() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
a = np.sort(np.random.randn(50))
v = np.random.randn(10)
indices = fnp.searchsorted(a, v)
# Inserting at these indices should preserve sortedness
for i, idx in enumerate(indices):
    inserted = np.insert(a, idx, v[i])
    if not np.all(inserted[:-1] <= inserted[1:]):
        print("False")
        break
else:
    print("True")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "insertion at searchsorted index preserves sortedness"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// unique metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unique_has_no_duplicates() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randint(0, 20, size=100)
u = fnp.unique(x)
# All elements in unique result should be distinct
has_no_dups = len(u) == len(set(u.tolist()))
print(has_no_dups)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique(x) has no duplicates");
    Ok(())
}

#[test]
fn unique_is_subset() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randint(0, 20, size=100)
u = fnp.unique(x)
# All elements in unique should be in original
is_subset = all(elem in x for elem in u)
print(is_subset)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "unique(x) is subset of x");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// partition metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn partition_kth_is_sorted_position() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
for kth in [0, 10, 50, 90, 99]:
    p = fnp.partition(x, kth)
    s = fnp.sort(x)
    # Element at kth position should be same as in sorted array
    if not np.isclose(p[kth], s[kth]):
        print("False")
        break
else:
    print("True")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "partition(x, k)[k] == sort(x)[k]");
    Ok(())
}

#[test]
fn partition_left_right_ordering() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
kth = 50
p = fnp.partition(x, kth)
pivot = p[kth]
# All elements left of kth should be <= pivot
# All elements right of kth should be >= pivot
left_ok = np.all(p[:kth] <= pivot)
right_ok = np.all(p[kth+1:] >= pivot)
print(left_ok and right_ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "partition maintains left <= pivot <= right"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// linear algebra metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matrix_inverse_product_is_identity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(5, 5)
A = A + np.eye(5) * 5  # Make well-conditioned
A_inv = fnp.linalg.inv(A)
product = fnp.dot(A, A_inv)
identity = np.eye(5)
print(np.allclose(product, identity, atol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "A @ inv(A) == I");
    Ok(())
}

#[test]
fn transpose_matmul_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 5)
B = np.random.randn(5, 3)
# (A @ B).T == B.T @ A.T
AB = fnp.matmul(A, B)
AB_T = fnp.transpose(AB)
BT_AT = fnp.matmul(fnp.transpose(B), fnp.transpose(A))
print(np.allclose(AB_T, BT_AT))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "(A @ B).T == B.T @ A.T");
    Ok(())
}

#[test]
fn det_multiplicative_property() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)
# det(A @ B) == det(A) * det(B)
det_AB = fnp.linalg.det(fnp.matmul(A, B))
det_A_times_det_B = fnp.linalg.det(A) * fnp.linalg.det(B)
print(np.allclose(det_AB, det_A_times_det_B, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det(A @ B) == det(A) * det(B)");
    Ok(())
}

#[test]
fn det_transpose_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(5, 5)
# det(A.T) == det(A)
det_A = fnp.linalg.det(A)
det_AT = fnp.linalg.det(fnp.transpose(A))
print(np.allclose(det_A, det_AT))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "det(A.T) == det(A)");
    Ok(())
}

#[test]
fn trace_additive_property() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(5, 5)
B = np.random.randn(5, 5)
# trace(A + B) == trace(A) + trace(B)
trace_sum = fnp.trace(A + B)
sum_traces = fnp.trace(A) + fnp.trace(B)
print(np.allclose(trace_sum, sum_traces))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "trace(A + B) == trace(A) + trace(B)");
    Ok(())
}

#[test]
fn norm_homogeneity() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 5)
k = 3.7
# norm(k * A) == |k| * norm(A)
norm_kA = fnp.linalg.norm(k * A)
k_norm_A = abs(k) * fnp.linalg.norm(A)
print(np.allclose(norm_kA, k_norm_A))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "norm(k * A) == |k| * norm(A)");
    Ok(())
}

#[test]
fn svd_reconstruction() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 6)
# A == U @ diag(s) @ Vh
U, s, Vh = fnp.linalg.svd(A, full_matrices=False)
reconstructed = U @ np.diag(s) @ Vh
print(np.allclose(A, reconstructed))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "A == U @ diag(s) @ Vh (SVD reconstruction)"
    );
    Ok(())
}

#[test]
fn eig_transpose_same_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 4)
# eigenvalues of A and A.T should be the same (possibly different order)
eigvals_A = np.sort(fnp.linalg.eigvals(A))
eigvals_AT = np.sort(fnp.linalg.eigvals(fnp.transpose(A)))
print(np.allclose(eigvals_A, eigvals_AT))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "eigvals(A) == eigvals(A.T) (up to ordering)"
    );
    Ok(())
}

#[test]
fn qr_reconstruction() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(5, 4)
# A == Q @ R
Q, R = fnp.linalg.qr(A)
reconstructed = Q @ R
print(np.allclose(A, reconstructed))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "A == Q @ R (QR reconstruction)");
    Ok(())
}

#[test]
fn cholesky_reconstruction() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(4, 4)
A = A @ A.T + np.eye(4)  # Make positive definite
# A == L @ L.T
L = fnp.linalg.cholesky(A)
reconstructed = L @ L.T
print(np.allclose(A, reconstructed))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "A == L @ L.T (Cholesky reconstruction)"
    );
    Ok(())
}

#[test]
fn solve_verification() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(5, 5)
A = A + np.eye(5) * 3  # Make well-conditioned
b = np.random.randn(5)
# A @ solve(A, b) == b
x = fnp.linalg.solve(A, b)
Ax = A @ x
print(np.allclose(Ax, b))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "A @ solve(A, b) == b");
    Ok(())
}

#[test]
fn lstsq_normal_equations() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
A = np.random.randn(6, 4)
b = np.random.randn(6)
# For overdetermined system, lstsq minimizes ||Ax - b||
x, residuals, rank, s = fnp.linalg.lstsq(A, b, rcond=None)
# A.T @ A @ x == A.T @ b (normal equations, approximately)
lhs = A.T @ A @ x
rhs = A.T @ b
print(np.allclose(lhs, rhs, rtol=1e-8))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "lstsq satisfies normal equations");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistical metamorphic properties
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn var_shift_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
c = 12345.6789
# var(x + c) == var(x) (shift invariance)
v1 = fnp.var(x)
v2 = fnp.var(x + c)
print(np.allclose(v1, v2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "var(x + c) == var(x)");
    Ok(())
}

#[test]
fn var_scale_property() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
c = 3.5
# var(c * x) == c^2 * var(x)
v1 = fnp.var(c * x)
v2 = c**2 * fnp.var(x)
print(np.allclose(v1, v2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "var(c * x) == c^2 * var(x)");
    Ok(())
}

#[test]
fn std_shift_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
c = 99999.0
# std(x + c) == std(x)
s1 = fnp.std(x)
s2 = fnp.std(x + c)
print(np.allclose(s1, s2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "std(x + c) == std(x)");
    Ok(())
}

#[test]
fn std_scale_property() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
c = 2.5
# std(c * x) == |c| * std(x)
s1 = fnp.std(c * x)
s2 = abs(c) * fnp.std(x)
print(np.allclose(s1, s2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "std(c * x) == |c| * std(x)");
    Ok(())
}

#[test]
fn std_is_sqrt_var() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
# std(x) == sqrt(var(x))
s = fnp.std(x)
v = fnp.sqrt(fnp.var(x))
print(np.allclose(s, v))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "std(x) == sqrt(var(x))");
    Ok(())
}

#[test]
fn cov_self_equals_var() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
# cov(x, x)[0,0] == var(x, ddof=1)
c = fnp.cov(x)
v = fnp.var(x, ddof=1)
print(np.allclose(c, v))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov(x, x) == var(x, ddof=1)");
    Ok(())
}

#[test]
fn corrcoef_self_is_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
# corrcoef(x, x)[0,1] == 1
r = fnp.corrcoef(x, x)
print(np.allclose(r[0, 1], 1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "corrcoef(x, x)[0,1] == 1");
    Ok(())
}

#[test]
fn corrcoef_negation_is_minus_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
# corrcoef(x, -x)[0,1] == -1
r = fnp.corrcoef(x, -x)
print(np.allclose(r[0, 1], -1.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "corrcoef(x, -x)[0,1] == -1");
    Ok(())
}

#[test]
fn corrcoef_shift_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
c, d = 1000.0, 2000.0
# corrcoef(x + c, y + d) == corrcoef(x, y)
r1 = fnp.corrcoef(x, y)
r2 = fnp.corrcoef(x + c, y + d)
print(np.allclose(r1, r2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "corrcoef is shift invariant");
    Ok(())
}

#[test]
fn corrcoef_scale_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
# corrcoef(a*x, b*y) == corrcoef(x, y) for a, b > 0
r1 = fnp.corrcoef(x, y)
r2 = fnp.corrcoef(5.0 * x, 3.0 * y)
print(np.allclose(r1, r2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "corrcoef is scale invariant for positive scales"
    );
    Ok(())
}

#[test]
fn cov_shift_invariant() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
c, d = 50000.0, 60000.0
# cov(x + c, y + d) == cov(x, y)
xy = np.vstack([x, y])
xy_shifted = np.vstack([x + c, y + d])
c1 = fnp.cov(xy)
c2 = fnp.cov(xy_shifted)
print(np.allclose(c1, c2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cov is shift invariant");
    Ok(())
}

#[test]
fn mean_shift_property() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
c = 12345.6789
# mean(x + c) == mean(x) + c
m1 = fnp.mean(x + c)
m2 = fnp.mean(x) + c
print(np.allclose(m1, m2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "mean(x + c) == mean(x) + c");
    Ok(())
}

#[test]
fn mean_scale_property() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
c = 3.14159
# mean(c * x) == c * mean(x)
m1 = fnp.mean(c * x)
m2 = c * fnp.mean(x)
print(np.allclose(m1, m2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "mean(c * x) == c * mean(x)");
    Ok(())
}
