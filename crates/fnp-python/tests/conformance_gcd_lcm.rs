//! Conformance tests for numpy.gcd and numpy.lcm against NumPy oracle.
//!
//! Tests integer operations:
//! - gcd(x1, x2): greatest common divisor, element-wise
//! - lcm(x1, x2): least common multiple, element-wise

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
fn gcd_basic_arrays_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([12, 24, 36, 48, 60])", "np.array([8, 16, 24, 32, 40])"),
        ("np.array([100, 200, 300])", "np.array([50, 75, 100])"),
        ("np.array([7, 11, 13, 17])", "np.array([3, 5, 7, 11])"),
        ("np.array([0, 5, 10, 15])", "np.array([5, 5, 5, 5])"),
        ("np.array([1, 2, 3, 4, 5])", "np.array([1, 1, 1, 1, 1])"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.gcd({x1_expr}, {x2_expr}).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.gcd({x1_expr}, {x2_expr}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "gcd mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn lcm_basic_arrays_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([4, 6, 8, 10])", "np.array([6, 8, 10, 12])"),
        ("np.array([3, 5, 7, 9])", "np.array([2, 4, 6, 8])"),
        ("np.array([12, 15, 18])", "np.array([8, 10, 12])"),
        ("np.array([1, 2, 3, 4])", "np.array([5, 6, 7, 8])"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.lcm({x1_expr}, {x2_expr}).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.lcm({x1_expr}, {x2_expr}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "lcm mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn gcd_with_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([0, 5, 10, 0, 15])
b = np.array([5, 0, 0, 0, 5])
print(np.gcd(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([0, 5, 10, 0, 15])
b = np.array([5, 0, 0, 0, 5])
print(fnp.gcd(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "gcd with zero mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn lcm_with_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([0, 5, 10, 0, 15])
b = np.array([5, 0, 0, 0, 5])
print(np.lcm(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([0, 5, 10, 0, 15])
b = np.array([5, 0, 0, 0, 5])
print(fnp.lcm(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "lcm with zero mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn gcd_negative_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([-12, 12, -12, 12])
b = np.array([8, -8, -8, 8])
print(np.gcd(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([-12, 12, -12, 12])
b = np.array([8, -8, -8, 8])
print(fnp.gcd(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "gcd negative inputs mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn lcm_negative_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([-4, 4, -4, 4])
b = np.array([6, -6, -6, 6])
print(np.lcm(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([-4, 4, -4, 4])
b = np.array([6, -6, -6, 6])
print(fnp.lcm(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "lcm negative inputs mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn gcd_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([[12], [24], [36]])
b = np.array([4, 6, 8])
print(np.gcd(a, b).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([[12], [24], [36]])
b = np.array([4, 6, 8])
print(fnp.gcd(a, b).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "gcd broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn lcm_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([[3], [4], [5]])
b = np.array([2, 3, 4])
print(np.lcm(a, b).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([[3], [4], [5]])
b = np.array([2, 3, 4])
print(fnp.lcm(a, b).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "lcm broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn gcd_50_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
a = np.random.randint(1, 1000, 50)
b = np.random.randint(1, 1000, 50)
print(np.gcd(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
a = np.random.randint(1, 1000, 50)
b = np.random.randint(1, 1000, 50)
print(fnp.gcd(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "gcd 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn lcm_50_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
a = np.random.randint(1, 100, 50)
b = np.random.randint(1, 100, 50)
print(np.lcm(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
a = np.random.randint(1, 100, 50)
b = np.random.randint(1, 100, 50)
print(fnp.lcm(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "lcm 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn gcd_empty_array_match_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.gcd(np.array([], dtype=int), np.array([], dtype=int)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "print(fnp.gcd(np.array([], dtype=int), np.array([], dtype=int)).tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "gcd empty array mismatch"
    );

    Ok(())
}

#[test]
fn lcm_empty_array_match_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.lcm(np.array([], dtype=int), np.array([], dtype=int)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "print(fnp.lcm(np.array([], dtype=int), np.array([], dtype=int)).tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "lcm empty array mismatch"
    );

    Ok(())
}

#[test]
fn gcd_scalar_broadcast_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([12, 24, 36, 48, 60])
print(np.gcd(a, 6).tolist())
print(np.gcd(6, a).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
a = np.array([12, 24, 36, 48, 60])
print(fnp.gcd(a, 6).tolist())
print(fnp.gcd(6, a).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "gcd scalar broadcast mismatch"
    );

    Ok(())
}

#[test]
fn lcm_scalar_broadcast_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([3, 4, 5, 6, 7])
print(np.lcm(a, 2).tolist())
print(np.lcm(2, a).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
a = np.array([3, 4, 5, 6, 7])
print(fnp.lcm(a, 2).tolist())
print(fnp.lcm(2, a).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "lcm scalar broadcast mismatch"
    );

    Ok(())
}

#[test]
fn gcd_coprime_numbers_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([7, 11, 13, 17, 19, 23])
b = np.array([3, 5, 9, 4, 6, 8])
print(np.gcd(a, b).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([7, 11, 13, 17, 19, 23])
b = np.array([3, 5, 9, 4, 6, 8])
print(fnp.gcd(a, b).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "gcd coprime mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn gcd_same_numbers_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
a = np.array([5, 10, 15, 20, 25])
print(np.gcd(a, a).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
a = np.array([5, 10, 15, 20, 25])
print(fnp.gcd(a, a).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "gcd same numbers mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}
