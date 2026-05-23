//! Conformance tests for numpy exponential and logarithmic functions against NumPy oracle.
//!
//! Tests exp, expm1, log, log1p, log10, log2, sqrt, square, power.

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
// exp
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn exp_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, -1, -2])
result = fnp.exp(x)
expected = np.exp(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp basic should match numpy");
    Ok(())
}

#[test]
fn exp_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0])
result = fnp.exp(x)
# exp(0) = 1
print(np.allclose(result, [1.0]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp(0) should equal 1");
    Ok(())
}

#[test]
fn exp_large() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([10, 50, 100])
result = fnp.exp(x)
expected = np.exp(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp large values should match numpy");
    Ok(())
}

#[test]
fn exp_subnormal_inputs_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([
    np.nextafter(0.0, 1.0),
    np.finfo(np.float64).tiny / 2.0,
    -np.nextafter(0.0, 1.0),
    -np.finfo(np.float64).tiny / 2.0,
], dtype=np.float64)
result = fnp.exp(x)
expected = np.exp(x)
print(result.dtype == expected.dtype and np.allclose(result, expected, rtol=1e-15, atol=0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "exp subnormal inputs should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// expm1
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn expm1_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, -1, 0.001, -0.001])
result = fnp.expm1(x)
expected = np.expm1(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "expm1 basic should match numpy");
    Ok(())
}

#[test]
fn expm1_small_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e-10, 1e-15, -1e-10, -1e-15])
result = fnp.expm1(x)
expected = np.expm1(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "expm1 small values should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// log
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn log_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, np.e, np.e**2, 10, 100])
result = fnp.log(x)
expected = np.log(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log basic should match numpy");
    Ok(())
}

#[test]
fn log_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0])
result = fnp.log(x)
# log(1) = 0
print(np.allclose(result, [0.0]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log(1) should equal 0");
    Ok(())
}

#[test]
fn log_subnormal_inputs_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([
    np.nextafter(0.0, 1.0),
    np.finfo(np.float64).tiny / 2.0,
    np.finfo(np.float64).tiny,
], dtype=np.float64)
result = fnp.log(x)
expected = np.log(x)
print(result.dtype == expected.dtype and np.allclose(result, expected, rtol=1e-15, atol=0.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "log subnormal inputs should match numpy"
    );
    Ok(())
}

#[test]
fn log_zero_and_subnormal_boundary_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, np.nextafter(0.0, 1.0)], dtype=np.float64)
result = fnp.log(x)
expected = np.log(x)
finite = np.isfinite(expected)
ok = (
    result.dtype == expected.dtype
    and np.array_equal(np.isneginf(result), np.isneginf(expected))
    and np.allclose(result[finite], expected[finite], rtol=1e-15, atol=0.0)
)
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "log zero/subnormal boundary should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// log1p
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn log1p_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, -0.5, 0.001, -0.001])
result = fnp.log1p(x)
expected = np.log1p(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log1p basic should match numpy");
    Ok(())
}

#[test]
fn log1p_small_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1e-10, 1e-15, -1e-10, -1e-15])
result = fnp.log1p(x)
expected = np.log1p(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "log1p small values should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// log10
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn log10_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 10, 100, 1000])
result = fnp.log10(x)
expected = np.log10(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log10 basic should match numpy");
    Ok(())
}

#[test]
fn log10_powers_of_ten() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([10**i for i in range(6)])
result = fnp.log10(x)
expected = np.arange(6, dtype=float)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "log10 of powers of 10 should be integers"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// log2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn log2_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 4, 8, 16])
result = fnp.log2(x)
expected = np.log2(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log2 basic should match numpy");
    Ok(())
}

#[test]
fn log2_powers_of_two() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([2**i for i in range(10)])
result = fnp.log2(x)
expected = np.arange(10, dtype=float)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "log2 of powers of 2 should be integers"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// sqrt
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sqrt_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 4, 9, 16, 25])
result = fnp.sqrt(x)
expected = np.sqrt(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt basic should match numpy");
    Ok(())
}

#[test]
fn sqrt_perfect_squares() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([i**2 for i in range(10)])
result = fnp.sqrt(x)
expected = np.arange(10, dtype=float)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sqrt of perfect squares should be integers"
    );
    Ok(())
}

#[test]
fn sqrt_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.5, 1.5, 2.5, 3.5])
result = fnp.sqrt(x)
expected = np.sqrt(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// square
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn square_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
result = fnp.square(x)
expected = np.square(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "square basic should match numpy");
    Ok(())
}

#[test]
fn square_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.5, -0.5, 0.5, 1.5])
result = fnp.square(x)
expected = np.square(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "square float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// power
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn power_int_exponent() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([2, 3, 4, 5])
result = fnp.power(x, 2)
expected = np.power(x, 2)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "power int exponent should match numpy"
    );
    Ok(())
}

#[test]
fn power_float_exponent() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0])
result = fnp.power(x, 0.5)
expected = np.power(x, 0.5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "power float exponent should match numpy"
    );
    Ok(())
}

#[test]
fn power_array_exponent() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([2, 3, 4, 5])
y = np.array([1, 2, 3, 4])
result = fnp.power(x, y)
expected = np.power(x, y)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "power array exponent should match numpy"
    );
    Ok(())
}

#[test]
fn power_subnormal_inputs_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([
    np.nextafter(0.0, 1.0),
    np.finfo(np.float64).tiny / 2.0,
    -np.nextafter(0.0, 1.0),
    -np.finfo(np.float64).tiny / 2.0,
], dtype=np.float64)
exponents = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float64)
result = fnp.power(x, exponents)
expected = np.power(x, exponents)
print(result.dtype == expected.dtype and np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "power subnormal inputs should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn exp_log_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
# log(exp(x)) = x
result = fnp.log(fnp.exp(x))
print(np.allclose(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log(exp(x)) should equal x");
    Ok(())
}

#[test]
fn sqrt_square_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
# sqrt(square(x)) = x for positive x
result = fnp.sqrt(fnp.square(x))
print(np.allclose(result, x))
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
fn expm1_log1p_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.001, 0.01, 0.1, 1])
# log1p(expm1(x)) = x
result = fnp.log1p(fnp.expm1(x))
print(np.allclose(result, x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log1p(expm1(x)) should equal x");
    Ok(())
}

#[test]
fn power_sqrt_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 4, 9, 16, 25])
sqrt_result = fnp.sqrt(x)
power_result = fnp.power(x, 0.5)
print(np.allclose(sqrt_result, power_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt should equal power(x, 0.5)");
    Ok(())
}

#[test]
fn exp_log_scalar_return_type_matches_numpy() -> Result<(), String> {
    let funcs = [
        "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
        "sqrt", "square", "positive", "negative", "absolute",
    ];
    for func in funcs {
        let script = fnp_script(format!(
            r#"
x = np.float64(2.0)
fnp_result = fnp.{func}(x)
np_result = np.{func}(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        ));
        let result = numpy_oracle(&script)?;
        assert!(
            result.trim().starts_with("True"),
            "{func} scalar return type should match numpy: {result}"
        );
    }
    Ok(())
}

#[test]
fn power_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.0)
y = np.float64(3.0)
fnp_result = fnp.power(x, y)
np_result = np.power(x, y)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "power scalar return type should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex dtype tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn exp_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 0+1j, -1+0j, 1j*np.pi], dtype=np.complex128)
fnp_result = fnp.exp(z)
np_result = np.exp(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp complex should match numpy");
    Ok(())
}

#[test]
fn log_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, -1+0j, 0+1j, 2+3j], dtype=np.complex128)
fnp_result = fnp.log(z)
np_result = np.log(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log complex should match numpy");
    Ok(())
}

#[test]
fn sqrt_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, -1+0j, 0+1j, 4+0j, -4+0j], dtype=np.complex128)
fnp_result = fnp.sqrt(z)
np_result = np.sqrt(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sqrt complex should match numpy");
    Ok(())
}

#[test]
fn square_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+3j, -1-1j, 0+2j], dtype=np.complex128)
fnp_result = fnp.square(z)
np_result = np.square(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "square complex should match numpy");
    Ok(())
}

#[test]
fn power_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z1 = np.array([2+1j, 1+1j, 3+0j], dtype=np.complex128)
z2 = np.array([2+0j, 1+1j, 0.5+0j], dtype=np.complex128)
fnp_result = fnp.power(z1, z2)
np_result = np.power(z1, z2)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "power complex should match numpy");
    Ok(())
}

#[test]
fn log10_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([10+0j, 1+1j, 100+0j, -10+0j], dtype=np.complex128)
fnp_result = fnp.log10(z)
np_result = np.log10(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log10 complex should match numpy");
    Ok(())
}

#[test]
fn log2_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([2+0j, 4+0j, 1+1j, -2+0j], dtype=np.complex128)
fnp_result = fnp.log2(z)
np_result = np.log2(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log2 complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Special value tests (nan/inf/negative)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn exp_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0])
fnp_result = fnp.exp(x)
np_result = np.exp(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp special values should match numpy");
    Ok(())
}

#[test]
fn log_negative_returns_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, -10.0, -100.0])
fnp_result = fnp.log(x)
np_result = np.log(x)
print(np.all(np.isnan(fnp_result)) and np.all(np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log of negative should return nan");
    Ok(())
}

#[test]
fn log_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([np.inf, 0.0, np.nan])
fnp_result = fnp.log(x)
np_result = np.log(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "log special values should match numpy");
    Ok(())
}

#[test]
fn power_zero_exponent() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, 1.0, 5.0, -3.0, np.inf, np.nan])
fnp_result = fnp.power(x, 0)
np_result = np.power(x, 0)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "x^0 should equal 1 for all x");
    Ok(())
}

#[test]
fn power_negative_base_fractional_exp() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, -2.0, -4.0])
fnp_result = fnp.power(x, 0.5)
np_result = np.power(x, 0.5)
print(np.all(np.isnan(fnp_result)) and np.all(np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "negative base with fractional exp should return nan");
    Ok(())
}

#[test]
fn exp_overflow() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1000.0, 710.0])
fnp_result = fnp.exp(x)
np_result = np.exp(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "exp overflow should match numpy");
    Ok(())
}
