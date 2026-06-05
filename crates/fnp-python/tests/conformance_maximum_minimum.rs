//! Conformance tests for numpy.maximum and numpy.minimum against NumPy oracle.
//!
//! Tests element-wise maximum/minimum with NaN propagation:
//! - maximum(x1, x2): element-wise maximum, NaN-propagating
//! - minimum(x1, x2): element-wise minimum, NaN-propagating

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
fn maximum_basic_arrays_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3, 4, 5])", "np.array([5, 4, 3, 2, 1])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([3.0, 2.0, 1.0])"),
        ("np.array([-1, -2, -3])", "np.array([-3, -2, -1])"),
        ("np.array([0.0, 0.0, 0.0])", "np.array([1.0, -1.0, 0.0])"),
        ("np.linspace(-10, 10, 20)", "np.zeros(20)"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.maximum({x1_expr}, {x2_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.maximum({x1_expr}, {x2_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "maximum mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn minimum_basic_arrays_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3, 4, 5])", "np.array([5, 4, 3, 2, 1])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([3.0, 2.0, 1.0])"),
        ("np.array([-1, -2, -3])", "np.array([-3, -2, -1])"),
        ("np.array([0.0, 0.0, 0.0])", "np.array([1.0, -1.0, 0.0])"),
        ("np.linspace(-10, 10, 20)", "np.zeros(20)"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.minimum({x1_expr}, {x2_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.minimum({x1_expr}, {x2_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "minimum mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn maximum_nan_propagating_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
x2 = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
result = np.maximum(x1, x2)
print([np.isnan(v) for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
x2 = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
result = fnp.maximum(x1, x2)
print([np.isnan(v) for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "maximum nan-propagating mismatch"
    );

    Ok(())
}

#[test]
fn minimum_nan_propagating_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
x2 = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
result = np.minimum(x1, x2)
print([np.isnan(v) for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
x2 = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
result = fnp.minimum(x1, x2)
print([np.isnan(v) for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "minimum nan-propagating mismatch"
    );

    Ok(())
}

#[test]
fn maximum_vs_fmax_nan_behavior() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([np.nan, 2.0, np.nan])
max_result = np.maximum(x1, x2)
fmax_result = np.fmax(x1, x2)
print('maximum nans:', sum(np.isnan(max_result)))
print('fmax nans:', sum(np.isnan(fmax_result)))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([np.nan, 2.0, np.nan])
max_result = fnp.maximum(x1, x2)
fmax_result = fnp.fmax(x1, x2)
print('maximum nans:', sum(np.isnan(max_result)))
print('fmax nans:', sum(np.isnan(fmax_result)))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "maximum vs fmax nan behavior mismatch"
    );

    Ok(())
}

#[test]
fn minimum_vs_fmin_nan_behavior() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([np.nan, 2.0, np.nan])
min_result = np.minimum(x1, x2)
fmin_result = np.fmin(x1, x2)
print('minimum nans:', sum(np.isnan(min_result)))
print('fmin nans:', sum(np.isnan(fmin_result)))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([1.0, np.nan, 3.0])
x2 = np.array([np.nan, 2.0, np.nan])
min_result = fnp.minimum(x1, x2)
fmin_result = fnp.fmin(x1, x2)
print('minimum nans:', sum(np.isnan(min_result)))
print('fmin nans:', sum(np.isnan(fmin_result)))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "minimum vs fmin nan behavior mismatch"
    );

    Ok(())
}

#[test]
fn maximum_inf_handling_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([np.inf, -np.inf, 1.0, -1.0, np.inf])
x2 = np.array([1.0, 1.0, np.inf, -np.inf, -np.inf])
print(np.maximum(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, 1.0, -1.0, np.inf])
x2 = np.array([1.0, 1.0, np.inf, -np.inf, -np.inf])
print(fnp.maximum(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "maximum inf handling mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn minimum_inf_handling_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([np.inf, -np.inf, 1.0, -1.0, np.inf])
x2 = np.array([1.0, 1.0, np.inf, -np.inf, -np.inf])
print(np.minimum(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([np.inf, -np.inf, 1.0, -1.0, np.inf])
x2 = np.array([1.0, 1.0, np.inf, -np.inf, -np.inf])
print(fnp.minimum(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "minimum inf handling mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn maximum_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
print(np.maximum(x1, x2).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
print(fnp.maximum(x1, x2).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "maximum broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn minimum_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
print(np.minimum(x1, x2).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
print(fnp.minimum(x1, x2).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "minimum broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn maximum_50_random_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x1 = np.random.randn(50) * 100
x2 = np.random.randn(50) * 100
print(np.maximum(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x1 = np.random.randn(50) * 100
x2 = np.random.randn(50) * 100
print(fnp.maximum(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "maximum random 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn minimum_50_random_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x1 = np.random.randn(50) * 100
x2 = np.random.randn(50) * 100
print(np.minimum(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x1 = np.random.randn(50) * 100
x2 = np.random.randn(50) * 100
print(fnp.minimum(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "minimum random 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn maximum_empty_array_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.maximum(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.maximum(np.array([]), np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "maximum empty array mismatch"
    );

    Ok(())
}

#[test]
fn minimum_empty_array_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.minimum(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.minimum(np.array([]), np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "minimum empty array mismatch"
    );

    Ok(())
}

#[test]
fn maximum_scalar_broadcast_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(np.maximum(x, 3.0).tolist())
print(np.maximum(3.0, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(fnp.maximum(x, 3.0).tolist())
print(fnp.maximum(3.0, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "maximum scalar broadcast mismatch"
    );

    Ok(())
}

#[test]
fn minimum_scalar_broadcast_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(np.minimum(x, 3.0).tolist())
print(np.minimum(3.0, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(fnp.minimum(x, 3.0).tolist())
print(fnp.minimum(3.0, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "minimum scalar broadcast mismatch"
    );

    Ok(())
}

#[test]
fn maximum_minimum_scalar_return_type_matches_numpy() -> Result<(), String> {
    for func in &["maximum", "minimum"] {
        let script = fnp_script(format!(
            r#"
x = np.float64(3.0)
y = np.float64(5.0)
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
