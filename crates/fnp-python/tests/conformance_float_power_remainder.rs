//! Conformance tests for numpy.float_power, numpy.remainder, numpy.mod, and numpy.fmod
//! against NumPy oracle.
//!
//! Tests power and modulo operations:
//! - float_power(x1, x2): x1**x2, always returns float
//! - remainder(x1, x2): Python-style remainder (sign follows divisor)
//! - mod(x1, x2): alias for remainder
//! - fmod(x1, x2): C-style remainder (sign follows dividend)

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

fn parse_float_list(s: &str) -> Result<Vec<f64>, String> {
    if s.is_empty() || s == "[]" {
        return Ok(vec![]);
    }
    let trimmed = s
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| format!("expected bracketed float list, got {s:?}"))?;

    let mut values = Vec::new();
    for token in trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
    {
        let t = token.trim().trim_end_matches('.');
        let value = if t == "nan" || t == "NaN" {
            f64::NAN
        } else if t == "inf" || t == "Inf" {
            f64::INFINITY
        } else if t == "-inf" || t == "-Inf" {
            f64::NEG_INFINITY
        } else {
            t.parse::<f64>()
                .map_err(|error| format!("invalid float token {token:?} in {s:?}: {error}"))?
        };
        values.push(value);
    }
    Ok(values)
}

fn floats_close(a: &[f64], b: &[f64], rel_tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        if x.is_nan() && y.is_nan() {
            true
        } else if x.is_infinite() && y.is_infinite() {
            x.signum() == y.signum()
        } else if *x == 0.0 && *y == 0.0 {
            true
        } else {
            let diff = (x - y).abs();
            let max_val = x.abs().max(y.abs()).max(1e-15);
            diff <= rel_tol * max_val
        }
    })
}

#[test]
fn float_power_basic_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([2.0, 3.0, 4.0])", "np.array([2.0, 2.0, 2.0])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([0.0, 0.0, 0.0])"),
        ("np.array([4.0, 9.0, 16.0])", "np.array([0.5, 0.5, 0.5])"),
        (
            "np.array([10.0, 100.0, 1000.0])",
            "np.array([1.0, 2.0, 3.0])",
        ),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.float_power({x1_expr}, {x2_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.float_power({x1_expr}, {x2_expr}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "float_power mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn float_power_negative_base_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([-2.0, -3.0, -4.0])
x2 = np.array([2.0, 3.0, 4.0])
print(np.float_power(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([-2.0, -3.0, -4.0])
x2 = np.array([2.0, 3.0, 4.0])
print(fnp.float_power(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "float_power negative base mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn remainder_basic_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        (
            "np.array([5.0, 6.0, 7.0, 8.0])",
            "np.array([2.0, 2.0, 2.0, 2.0])",
        ),
        ("np.array([10.0, 20.0, 30.0])", "np.array([3.0, 7.0, 11.0])"),
        ("np.array([1.5, 2.5, 3.5])", "np.array([1.0, 1.0, 1.0])"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.remainder({x1_expr}, {x2_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.remainder({x1_expr}, {x2_expr}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "remainder mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn remainder_negative_sign_follows_divisor_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
# Python remainder: sign follows divisor
x1 = np.array([-5.0, 5.0, -5.0, 5.0])
x2 = np.array([3.0, -3.0, -3.0, 3.0])
print(np.remainder(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([-5.0, 5.0, -5.0, 5.0])
x2 = np.array([3.0, -3.0, -3.0, 3.0])
print(fnp.remainder(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "remainder negative sign mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn fmod_basic_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        (
            "np.array([5.0, 6.0, 7.0, 8.0])",
            "np.array([2.0, 2.0, 2.0, 2.0])",
        ),
        ("np.array([10.0, 20.0, 30.0])", "np.array([3.0, 7.0, 11.0])"),
        ("np.array([1.5, 2.5, 3.5])", "np.array([1.0, 1.0, 1.0])"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script = format!("import numpy as np; print(np.fmod({x1_expr}, {x2_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.fmod({x1_expr}, {x2_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "fmod mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn fmod_negative_sign_follows_dividend_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
# C fmod: sign follows dividend
x1 = np.array([-5.0, 5.0, -5.0, 5.0])
x2 = np.array([3.0, -3.0, -3.0, 3.0])
print(np.fmod(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([-5.0, 5.0, -5.0, 5.0])
x2 = np.array([3.0, -3.0, -3.0, 3.0])
print(fnp.fmod(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "fmod negative sign mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn remainder_vs_fmod_difference_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([-7.0, 7.0])
x2 = np.array([3.0, 3.0])
print('remainder:', np.remainder(x1, x2).tolist())
print('fmod:', np.fmod(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([-7.0, 7.0])
x2 = np.array([3.0, 3.0])
print('remainder:', fnp.remainder(x1, x2).tolist())
print('fmod:', fnp.fmod(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "remainder vs fmod difference mismatch"
    );

    Ok(())
}

#[test]
fn remainder_divide_by_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([5.0, 10.0, 15.0])
x2 = np.array([0.0, 0.0, 0.0])
result = np.remainder(x1, x2)
print([np.isnan(v) for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
import warnings
warnings.filterwarnings('ignore')
x1 = np.array([5.0, 10.0, 15.0])
x2 = np.array([0.0, 0.0, 0.0])
result = fnp.remainder(x1, x2)
print([np.isnan(v) for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "remainder divide by zero mismatch"
    );

    Ok(())
}

#[test]
fn float_power_50_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x1 = np.random.uniform(0.1, 10, 50)
x2 = np.random.uniform(-2, 2, 50)
print(np.float_power(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x1 = np.random.uniform(0.1, 10, 50)
x2 = np.random.uniform(-2, 2, 50)
print(fnp.float_power(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "float_power 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn remainder_50_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x1 = np.random.uniform(-100, 100, 50)
x2 = np.random.uniform(1, 10, 50)  # Avoid zeros
print(np.remainder(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x1 = np.random.uniform(-100, 100, 50)
x2 = np.random.uniform(1, 10, 50)
print(fnp.remainder(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "remainder 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn fmod_50_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x1 = np.random.uniform(-100, 100, 50)
x2 = np.random.uniform(1, 10, 50)  # Avoid zeros
print(np.fmod(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x1 = np.random.uniform(-100, 100, 50)
x2 = np.random.uniform(1, 10, 50)
print(fnp.fmod(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "fmod 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn mod_is_alias_for_remainder_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([-7.0, 7.0, 10.0, -10.0])
x2 = np.array([3.0, 3.0, 3.0, 3.0])
print(np.mod(x1, x2).tolist())
print(np.remainder(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([-7.0, 7.0, 10.0, -10.0])
x2 = np.array([3.0, 3.0, 3.0, 3.0])
print(getattr(fnp, 'mod')(x1, x2).tolist())
print(fnp.remainder(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "mod alias mismatch"
    );

    Ok(())
}

#[test]
fn float_power_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([[2.0], [3.0], [4.0]])
x2 = np.array([1.0, 2.0, 3.0])
print(np.float_power(x1, x2).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([[2.0], [3.0], [4.0]])
x2 = np.array([1.0, 2.0, 3.0])
print(fnp.float_power(x1, x2).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "float_power broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn float_power_empty_array_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.float_power(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.float_power(np.array([]), np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "float_power empty array mismatch"
    );

    Ok(())
}

#[test]
fn float_power_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.0)
y = np.float64(3.0)
fnp_result = fnp.float_power(x, y)
np_result = np.float_power(x, y)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "float_power scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn remainder_fmod_scalar_return_type_matches_numpy() -> Result<(), String> {
    for func in &["remainder", "fmod"] {
        let script = fnp_script(format!(
            r#"
x = np.float64(7.0)
y = np.float64(3.0)
fnp_result = fnp.{func}(x, y)
np_result = np.{func}(x, y)
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
fn remainder_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# remainder signed-zero: 0 % x = 0 with sign of dividend
# IEEE 754: remainder sign follows dividend (x1)
tests = [
    (0.0, 1.0), (0.0, -1.0),
    (-0.0, 1.0), (-0.0, -1.0),
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.remainder(np.float64(x1), np.float64(x2))
    np_result = np.remainder(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: remainder({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "remainder signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn fmod_signed_zero_parity() -> Result<(), String> {
    let script = fnp_script(
        r#"
# fmod signed-zero: fmod(0, x) = 0 with sign of dividend
tests = [
    (0.0, 1.0), (0.0, -1.0),
    (-0.0, 1.0), (-0.0, -1.0),
]
all_pass = True
for x1, x2 in tests:
    fnp_result = fnp.fmod(np.float64(x1), np.float64(x2))
    np_result = np.fmod(np.float64(x1), np.float64(x2))
    fnp_sign = np.signbit(fnp_result)
    np_sign = np.signbit(np_result)
    if fnp_sign != np_sign:
        print(f"FAIL: fmod({x1}, {x2})")
        print(f"  fnp result={fnp_result} signbit={fnp_sign}")
        print(f"  np result={np_result} signbit={np_sign}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "fmod signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn float_power_ieee_special_cases_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
# IEEE 754 power special cases
# Note: NumPy follows C99 conventions
import warnings
warnings.filterwarnings('ignore')

cases = [
    # 0**0 = 1 by convention
    (0.0, 0.0),
    (-0.0, 0.0),
    # x**0 = 1 for all x including inf/nan
    (np.inf, 0.0),
    (-np.inf, 0.0),
    (np.nan, 0.0),
    # 1**y = 1 for all y including inf/nan
    (1.0, np.inf),
    (1.0, -np.inf),
    (1.0, np.nan),
    # 0**positive = 0
    (0.0, 2.0),
    (-0.0, 2.0),
    # 0**negative = inf
    (0.0, -1.0),
    (-0.0, -1.0),
    # inf**positive = inf
    (np.inf, 2.0),
    # inf**negative = 0
    (np.inf, -1.0),
    # (-inf)**int = +/-inf based on parity
    (-np.inf, 2.0),  # even -> inf
    (-np.inf, 3.0),  # odd -> -inf
]

all_pass = True
for x, y in cases:
    fnp_result = fnp.float_power(np.float64(x), np.float64(y))
    np_result = np.float_power(np.float64(x), np.float64(y))
    # Compare with allclose for nan handling
    if np.isnan(fnp_result) and np.isnan(np_result):
        pass  # both nan is ok
    elif np.isinf(fnp_result) and np.isinf(np_result):
        if np.signbit(fnp_result) != np.signbit(np_result):
            print(f"FAIL: float_power({x}, {y}) sign mismatch: fnp={fnp_result} np={np_result}")
            all_pass = False
    elif not np.allclose(fnp_result, np_result, equal_nan=True):
        print(f"FAIL: float_power({x}, {y}): fnp={fnp_result} np={np_result}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "float_power IEEE special cases should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn float_power_negative_fractional_returns_nan() -> Result<(), String> {
    let script = fnp_script(
        r#"
# negative base with non-integer exponent yields nan in real domain
import warnings
warnings.filterwarnings('ignore')

x = np.array([-2.0, -3.0, -4.0])
y = np.array([0.5, 1.5, 2.5])  # fractional exponents
fnp_result = fnp.float_power(x, y)
np_result = np.float_power(x, y)
# Both should return nan for negative base with fractional exponent
print(np.all(np.isnan(fnp_result)) and np.all(np.isnan(np_result)))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "float_power negative fractional should return nan"
    );
    Ok(())
}
