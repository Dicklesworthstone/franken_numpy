//! Conformance tests for numpy.heaviside against NumPy oracle.
//!
//! Tests the Heaviside step function:
//! - heaviside(x1, x2): 0 for x1 < 0, x2 for x1 == 0, 1 for x1 > 0

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
fn heaviside_basic_step_function_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside basic mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_different_h0_values_match_numpy() -> Result<(), String> {
    let test_cases = vec![0.0, 0.5, 1.0, 0.25, 0.75, -0.5, 2.0];

    for h0 in &test_cases {
        let script =
            format!("import numpy as np; print(np.heaviside(np.array([-1, 0, 1]), {h0}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.heaviside(np.array([-1, 0, 1]), {h0}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "heaviside h0={h0} mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn heaviside_nan_input_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([np.nan, -1, 0, 1, np.nan])
result = np.heaviside(x, 0.5)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([np.nan, -1, 0, 1, np.nan])
result = fnp.heaviside(x, 0.5)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside nan input mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_nan_h0_at_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-1, 0, 1])
result = np.heaviside(x, np.nan)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-1, 0, 1])
result = fnp.heaviside(x, np.nan)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside nan h0 mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_inf_input_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-np.inf, np.inf, 0])
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-np.inf, np.inf, 0])
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside inf input mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[-1], [0], [1]])
h0 = np.array([0.0, 0.5, 1.0])
print(np.heaviside(x, h0).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[-1], [0], [1]])
h0 = np.array([0.0, 0.5, 1.0])
print(fnp.heaviside(x, h0).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_negative_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-0.0, 0.0])
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-0.0, 0.0])
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside negative zero mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_50_random_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x = np.random.randn(50) * 10
h0 = np.random.rand(50)
print(np.heaviside(x, h0).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(50) * 10
h0 = np.random.rand(50)
print(fnp.heaviside(x, h0).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside random 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn heaviside_empty_array_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.heaviside(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.heaviside(np.array([]), np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside empty array mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_scalar_h0_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(np.heaviside(x, 0.0).tolist())
print(np.heaviside(x, 0.5).tolist())
print(np.heaviside(x, 1.0).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(fnp.heaviside(x, 0.0).tolist())
print(fnp.heaviside(x, 0.5).tolist())
print(fnp.heaviside(x, 1.0).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside scalar h0 mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_linspace_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.linspace(-5, 5, 50)
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.linspace(-5, 5, 50)
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside linspace mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn heaviside_dtype_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.heaviside(np.array([-1, 0, 1]), 0.5).dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.heaviside(np.array([-1, 0, 1]), 0.5).dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside dtype mismatch"
    );

    Ok(())
}
