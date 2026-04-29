//! Conformance tests for trigonometric and math functions against NumPy oracle.
//!
//! Tests sin, cos, sqrt, exp, log, sinh, cosh, tanh, arcsin, arccos, arctan,
//! arcsinh, arccosh, arctanh, and positive.

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

fn test_unary_function(func: &str, test_cases: &[&str], rel_tol: f64) -> Result<(), String> {
    for arr_expr in test_cases {
        let script = format!("import numpy as np; print(np.{func}({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.{func}({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, rel_tol),
            "{func} mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn sin_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([np.pi])",
        "np.array([np.pi / 2])",
        "np.array([np.pi / 4])",
        "np.array([-np.pi])",
        "np.array([-np.pi / 2])",
        "np.array([2 * np.pi])",
        "np.linspace(-np.pi, np.pi, 21)",
        "np.linspace(0, 2 * np.pi, 17)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([[0.0, np.pi / 4], [np.pi / 2, np.pi]])",
    ];
    test_unary_function("sin", &test_cases, 1e-10)
}

#[test]
fn cos_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([np.pi])",
        "np.array([np.pi / 2])",
        "np.array([np.pi / 4])",
        "np.array([-np.pi])",
        "np.array([2 * np.pi])",
        "np.linspace(-np.pi, np.pi, 21)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("cos", &test_cases, 1e-10)
}

#[test]
fn sqrt_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([4.0])",
        "np.array([9.0, 16.0, 25.0])",
        "np.array([0.25, 0.5, 0.75])",
        "np.array([2.0, 3.0, 5.0, 7.0])",
        "np.array([1e10, 1e-10])",
        "np.linspace(0, 100, 11)",
        "np.array([np.inf])",
        "np.array([np.nan])",
        "np.array([-1.0])",
    ];
    test_unary_function("sqrt", &test_cases, 1e-10)
}

#[test]
fn exp_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([2.0, 3.0, 4.0])",
        "np.array([-2.0, -3.0, -4.0])",
        "np.linspace(-5, 5, 11)",
        "np.array([0.1, 0.5, 0.9])",
        "np.array([700.0])",
        "np.array([-700.0])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("exp", &test_cases, 1e-10)
}

#[test]
fn log_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([np.e])",
        "np.array([np.e ** 2])",
        "np.array([0.5, 1.0, 2.0])",
        "np.array([10.0, 100.0, 1000.0])",
        "np.linspace(0.1, 10, 10)",
        "np.array([1e-10, 1e10])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([np.inf, np.nan])",
    ];
    test_unary_function("log", &test_cases, 1e-10)
}

#[test]
fn sinh_cosh_tanh_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-3, 3, 13)",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("sinh", &test_cases, 1e-10)?;
    test_unary_function("cosh", &test_cases, 1e-10)?;
    test_unary_function("tanh", &test_cases, 1e-10)?;
    Ok(())
}

#[test]
fn arcsin_arccos_arctan_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-1, 1, 11)",
        "np.array([0.1, 0.2, 0.3, 0.4])",
        "np.array([np.nan])",
    ];
    test_unary_function("arcsin", &test_cases, 1e-10)?;
    test_unary_function("arccos", &test_cases, 1e-10)?;

    let arctan_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-10, 10, 21)",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("arctan", &arctan_cases, 1e-10)?;
    Ok(())
}

#[test]
fn arcsinh_arccosh_arctanh_match_numpy() -> Result<(), String> {
    let arcsinh_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-5, 5, 11)",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("arcsinh", &arcsinh_cases, 1e-10)?;

    let arccosh_cases = vec![
        "np.array([1.0])",
        "np.array([2.0])",
        "np.array([10.0])",
        "np.linspace(1, 10, 10)",
        "np.array([np.inf, np.nan])",
        "np.array([0.5])",
    ];
    test_unary_function("arccosh", &arccosh_cases, 1e-10)?;

    let arctanh_cases = vec![
        "np.array([0.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.linspace(-0.99, 0.99, 11)",
        "np.array([1.0, -1.0])",
        "np.array([np.nan])",
    ];
    test_unary_function("arctanh", &arctanh_cases, 1e-10)?;
    Ok(())
}

#[test]
fn positive_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([1.0, -1.0, 2.0, -2.0])",
        "np.array([0.5, -0.5, 1.5, -1.5])",
        "np.array([1e10, -1e10])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([[1.0, -1.0], [2.0, -2.0]])",
        "np.array([1, -1, 2, -2], dtype=np.float64)",
    ];
    test_unary_function("positive", &test_cases, 1e-14)
}

#[test]
fn trig_math_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &[
        "sin", "cos", "sqrt", "exp", "log", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
        "arcsinh", "arccosh", "arctanh", "positive",
    ] {
        let script = format!(
            "import numpy as np; print(np.{func}(np.array([], dtype=np.float64)).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "print(fnp.{func}(np.array([], dtype=np.float64)).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} empty array mismatch"
        );
    }
    Ok(())
}

#[test]
fn trig_integer_input_promotes_to_float() -> Result<(), String> {
    for func in &["sin", "cos", "sqrt", "exp", "log"] {
        let script = format!(
            "import numpy as np; r = np.{func}(np.array([1, 2, 3], dtype=np.int32)); print(r.dtype)"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.{func}(np.array([1, 2, 3], dtype=np.int32)); print(r.dtype)"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} dtype promotion mismatch"
        );
    }
    Ok(())
}

#[test]
fn promoting_math_bool_inputs_match_numpy() -> Result<(), String> {
    for func in &[
        "sin", "cos", "sqrt", "exp", "log", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
        "arcsinh", "arccosh", "arctanh", "expm1", "log1p",
    ] {
        let script = format!(
            "import numpy as np; r = np.{func}(np.array([True, False], dtype=np.bool_)); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.{func}(np.array([True, False], dtype=np.bool_)); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} bool input mismatch"
        );
    }
    Ok(())
}

#[test]
fn positive_bool_input_matches_numpy_error() -> Result<(), String> {
    let script = r#"
import numpy as np
try:
    np.positive(np.array([True, False], dtype=np.bool_))
    print("no_error")
except Exception as exc:
    print(type(exc).__name__)
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
try:
    fnp.positive(np.array([True, False], dtype=np.bool_))
    print("no_error")
except Exception as exc:
    print(type(exc).__name__)
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "positive bool error mismatch"
    );
    Ok(())
}
