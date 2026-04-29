//! Conformance tests for numpy.arctan2 against NumPy oracle.
//!
//! Tests two-argument arctangent (quadrant-aware atan):
//! - arctan2(y, x): angle in radians between positive x-axis and point (x, y)

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
fn arctan2_basic_quadrants_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("1.0", "1.0"),   // Q1: 45 degrees
        ("1.0", "-1.0"),  // Q2: 135 degrees
        ("-1.0", "-1.0"), // Q3: -135 degrees
        ("-1.0", "1.0"),  // Q4: -45 degrees
        ("0.0", "1.0"),   // positive x-axis
        ("0.0", "-1.0"),  // negative x-axis (pi)
        ("1.0", "0.0"),   // positive y-axis (pi/2)
        ("-1.0", "0.0"),  // negative y-axis (-pi/2)
    ];

    for (y, x) in &test_cases {
        let script = format!("import numpy as np; print(np.arctan2({y}, {x}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val: f64 = numpy_result.parse().map_err(|e| format!("{e}"))?;

        let rust_script = fnp_script(format!("print(fnp.arctan2({y}, {x}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val: f64 = rust_result.parse().map_err(|e| format!("{e}"))?;

        assert!(
            (numpy_val - rust_val).abs() < 1e-15,
            "arctan2({y}, {x}) mismatch: numpy={numpy_val}, rust={rust_val}"
        );
    }

    Ok(())
}

#[test]
fn arctan2_arrays_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3, 4, 5])", "np.array([5, 4, 3, 2, 1])"),
        (
            "np.array([1.0, -1.0, 1.0, -1.0])",
            "np.array([1.0, 1.0, -1.0, -1.0])",
        ),
        ("np.array([0.0, 0.0, 1.0, -1.0])", "np.array([1.0, -1.0, 0.0, 0.0])"),
        ("np.linspace(-1, 1, 10)", "np.linspace(-1, 1, 10)"),
        ("np.linspace(-10, 10, 20)", "np.ones(20)"),
    ];

    for (y_expr, x_expr) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.arctan2({y_expr}, {x_expr}).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.arctan2({y_expr}, {x_expr}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "arctan2 array mismatch for ({y_expr}, {x_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn arctan2_special_values_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([0.0, 0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, np.inf, -np.inf])
x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, -np.inf])
print(np.arctan2(y, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
y = np.array([0.0, 0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, np.inf, -np.inf])
x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, -np.inf])
print(fnp.arctan2(y, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 special values mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn arctan2_2d_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([[1], [2], [3]])
x = np.array([1, 2, 3])
print(np.arctan2(y, x).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
y = np.array([[1], [2], [3]])
x = np.array([1, 2, 3])
print(fnp.arctan2(y, x).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 2d broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn arctan2_negative_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([0.0, -0.0, 0.0, -0.0])
x = np.array([1.0, 1.0, -1.0, -1.0])
print(np.arctan2(y, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
y = np.array([0.0, -0.0, 0.0, -0.0])
x = np.array([1.0, 1.0, -1.0, -1.0])
print(fnp.arctan2(y, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 negative zero mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn arctan2_pi_values_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("0.0", "1.0", 0.0),           // 0
        ("1.0", "0.0", std::f64::consts::FRAC_PI_2), // pi/2
        ("0.0", "-1.0", std::f64::consts::PI),       // pi
        ("-1.0", "0.0", -std::f64::consts::FRAC_PI_2), // -pi/2
    ];

    for (y, x, expected) in &test_cases {
        let script = format!("import numpy as np; print(np.arctan2({y}, {x}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val: f64 = numpy_result.parse().map_err(|e| format!("{e}"))?;

        let rust_script = fnp_script(format!("print(fnp.arctan2({y}, {x}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val: f64 = rust_result.parse().map_err(|e| format!("{e}"))?;

        assert!(
            (numpy_val - rust_val).abs() < 1e-15,
            "arctan2({y}, {x}) mismatch: numpy={numpy_val}, rust={rust_val}, expected={expected}"
        );
        assert!(
            (numpy_val - expected).abs() < 1e-15,
            "arctan2({y}, {x}) expected {expected}, got numpy={numpy_val}"
        );
    }

    Ok(())
}

#[test]
fn arctan2_50_random_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
y = np.random.randn(50) * 100
x = np.random.randn(50) * 100
print(np.arctan2(y, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
y = np.random.randn(50) * 100
x = np.random.randn(50) * 100
print(fnp.arctan2(y, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 random 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn arctan2_empty_array_match_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.arctan2(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "print(fnp.arctan2(np.array([]), np.array([])).tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "arctan2 empty array mismatch"
    );

    Ok(())
}

#[test]
fn arctan2_dtype_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.arctan2(np.array([1, 2, 3]), np.array([1, 2, 3])).dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.arctan2(np.array([1, 2, 3]), np.array([1, 2, 3])).dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "arctan2 dtype mismatch"
    );

    Ok(())
}

#[test]
fn arctan2_scalar_broadcast_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([1, 2, 3, 4, 5])
print(np.arctan2(y, 1.0).tolist())
print(np.arctan2(1.0, y).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
y = np.array([1, 2, 3, 4, 5])
print(fnp.arctan2(y, 1.0).tolist())
print(fnp.arctan2(1.0, y).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "arctan2 scalar broadcast mismatch"
    );

    Ok(())
}
