//! Conformance tests for numpy floating-point introspection functions.
//!
//! Tests spacing, nextafter, ldexp against NumPy oracle:
//! - spacing(x): ULP distance between x and next representable value
//! - nextafter(x1, x2): next float from x1 toward x2
//! - ldexp(x1, x2): x1 * 2**x2

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
            x.to_bits() == y.to_bits()
        } else {
            let diff = (x - y).abs();
            let max_val = x.abs().max(y.abs()).max(1e-15);
            diff <= rel_tol * max_val
        }
    })
}

#[test]
fn spacing_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([2.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.array([1e10])",
        "np.array([1e-10])",
        "np.array([1e100])",
        "np.array([1e-100])",
        "np.array([1e308])",
        "np.array([1e-308])",
        "np.array([2.2250738585072014e-308])",
        "np.array([1.0, 2.0, 3.0, 4.0])",
        "np.array([-1.0, -2.0, -3.0, -4.0])",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([1.5, 2.5, 3.5, 4.5])",
        "np.linspace(1, 100, 10)",
        "np.linspace(0.001, 1, 10)",
        "np.array([[1.0, 2.0], [3.0, 4.0]])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([-0.0])",
        "np.array([np.finfo(float).eps])",
        "np.array([np.finfo(float).tiny])",
        "np.array([np.finfo(float).max])",
    ];

    for arr_expr in &test_cases {
        let script =
            format!("import numpy as np; print(np.spacing({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.spacing({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "spacing mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn nextafter_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([0.0])", "np.array([1.0])"),
        ("np.array([0.0])", "np.array([-1.0])"),
        ("np.array([1.0])", "np.array([2.0])"),
        ("np.array([1.0])", "np.array([0.0])"),
        ("np.array([-1.0])", "np.array([0.0])"),
        ("np.array([-1.0])", "np.array([-2.0])"),
        ("np.array([0.0])", "np.array([np.inf])"),
        ("np.array([0.0])", "np.array([-np.inf])"),
        ("np.array([np.inf])", "np.array([0.0])"),
        ("np.array([-np.inf])", "np.array([0.0])"),
        ("np.array([1.0])", "np.array([1.0])"),
        ("np.array([np.nan])", "np.array([1.0])"),
        ("np.array([1.0])", "np.array([np.nan])"),
        ("np.array([np.nan])", "np.array([np.nan])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([2.0, 3.0, 4.0])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([0.0, 1.0, 2.0])"),
        ("np.array([0.5])", "np.array([1.0])"),
        ("np.array([0.5])", "np.array([0.0])"),
        ("np.array([1e10])", "np.array([1e11])"),
        ("np.array([1e-10])", "np.array([1e-9])"),
        ("np.array([1e308])", "np.array([np.inf])"),
        ("np.array([1e-308])", "np.array([0.0])"),
        ("np.array([-0.0])", "np.array([0.0])"),
        ("np.array([0.0])", "np.array([-0.0])"),
        (
            "np.array([[1.0, 2.0], [3.0, 4.0]])",
            "np.array([[2.0, 3.0], [4.0, 5.0]])",
        ),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.nextafter({x1_expr}, {x2_expr}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.nextafter({x1_expr}, {x2_expr}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 0.0),
            "nextafter mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn ldexp_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1.0])", "np.array([0])"),
        ("np.array([1.0])", "np.array([1])"),
        ("np.array([1.0])", "np.array([2])"),
        ("np.array([1.0])", "np.array([10])"),
        ("np.array([1.0])", "np.array([-1])"),
        ("np.array([1.0])", "np.array([-10])"),
        ("np.array([0.5])", "np.array([1])"),
        ("np.array([0.5])", "np.array([2])"),
        ("np.array([2.0])", "np.array([3])"),
        ("np.array([1.5])", "np.array([4])"),
        ("np.array([0.0])", "np.array([10])"),
        ("np.array([-1.0])", "np.array([5])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([1, 2, 3])"),
        ("np.array([0.5, 0.25, 0.125])", "np.array([1, 2, 3])"),
        ("np.array([1.0])", "np.array([1023])"),
        ("np.array([1.0])", "np.array([-1022])"),
        ("np.array([1.0])", "np.array([1024])"),
        ("np.array([1.0])", "np.array([-1074])"),
        ("np.array([np.inf])", "np.array([1])"),
        ("np.array([-np.inf])", "np.array([1])"),
        ("np.array([np.nan])", "np.array([1])"),
        ("np.array([1.0])", "np.array([0, 1, 2, 3, 4])"),
        (
            "np.array([[1.0, 2.0], [3.0, 4.0]])",
            "np.array([[1, 2], [3, 4]])",
        ),
        ("np.array([0.5])", "np.array([-1])"),
        ("np.array([0.25])", "np.array([-2])"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.ldexp({x1_expr}, {x2_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.ldexp({x1_expr}, {x2_expr}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-14),
            "ldexp mismatch for ({x1_expr}, {x2_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn spacing_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.spacing(np.array([], dtype=np.float64)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.spacing(np.array([], dtype=np.float64)).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "spacing empty array mismatch"
    );

    Ok(())
}

#[test]
fn nextafter_broadcast_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([2.0])
print(np.nextafter(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([2.0])
print(fnp.nextafter(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 0.0),
        "nextafter broadcast mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn ldexp_broadcast_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([2])
print(np.ldexp(x1, x2).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([2])
print(fnp.ldexp(x1, x2).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-14),
        "ldexp broadcast mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn spacing_integer_promotes_to_float() -> Result<(), String> {
    let script =
        "import numpy as np; r = np.spacing(np.array([1, 2, 3], dtype=np.int32)); print(r.dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("r = fnp.spacing(np.array([1, 2, 3], dtype=np.int32)); print(r.dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "spacing integer promotion mismatch"
    );

    Ok(())
}

#[test]
fn nextafter_signed_zero_behavior() -> Result<(), String> {
    let script = r#"
import numpy as np
print(np.nextafter(np.array([0.0]), np.array([1.0])).tolist())
print(np.nextafter(np.array([-0.0]), np.array([1.0])).tolist())
print(np.nextafter(np.array([0.0]), np.array([-1.0])).tolist())
print(np.nextafter(np.array([-0.0]), np.array([-1.0])).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
print(fnp.nextafter(np.array([0.0]), np.array([1.0])).tolist())
print(fnp.nextafter(np.array([-0.0]), np.array([1.0])).tolist())
print(fnp.nextafter(np.array([0.0]), np.array([-1.0])).tolist())
print(fnp.nextafter(np.array([-0.0]), np.array([-1.0])).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "nextafter signed zero mismatch"
    );

    Ok(())
}

#[test]
fn ldexp_integer_mantissa_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "np.array([1, 2, 3])"),
        ("np.array([1, 2, 3], dtype=np.int64)", "np.array([4, 5, 6])"),
    ];

    for (x1_expr, x2_expr) in &test_cases {
        let script = format!(
            "import numpy as np; r = np.ldexp({x1_expr}, {x2_expr}); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.ldexp({x1_expr}, {x2_expr}); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "ldexp integer mantissa mismatch for {x1_expr}"
        );
    }

    Ok(())
}

#[test]
fn spacing_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.0)
fnp_result = fnp.spacing(x)
np_result = np.spacing(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "spacing scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn nextafter_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x1 = np.float64(1.0)
x2 = np.float64(2.0)
fnp_result = fnp.nextafter(x1, x2)
np_result = np.nextafter(x1, x2)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "nextafter scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn ldexp_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.5)
exp = np.int32(3)
fnp_result = fnp.ldexp(x, exp)
np_result = np.ldexp(x, exp)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "ldexp scalar return type should match numpy: {result}"
    );
    Ok(())
}
