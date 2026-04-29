//! Conformance tests for numpy.cbrt and numpy.fabs against NumPy oracle.
//!
//! Tests the native Rust implementations against NumPy for:
//! - cbrt(x): cube root, promotes integers to float64
//! - fabs(x): absolute value for real arrays, rejects complex with TypeError

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
fn cbrt_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([8.0])",
        "np.array([-8.0])",
        "np.array([27.0])",
        "np.array([-27.0])",
        "np.array([0.0, 1.0, 8.0, 27.0])",
        "np.array([-1.0, -8.0, -27.0])",
        "np.array([0.125])",
        "np.array([0.001])",
        "np.array([1e-9])",
        "np.array([1e9])",
        "np.array([1e-15])",
        "np.array([1e15])",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([-0.5, -1.5, -2.5])",
        "np.array([2.0, 3.0, 4.0, 5.0])",
        "np.array([64.0, 125.0, 216.0])",
        "np.array([-64.0, -125.0, -216.0])",
        "np.array([[1.0, 8.0], [27.0, 64.0]])",
        "np.array([[[8.0]]])",
        "np.array([0.0, 0.0, 0.0])",
        "np.array([1.0, 1.0, 1.0])",
        "np.array([-0.0])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.linspace(0, 100, 11)",
        "np.linspace(-100, 100, 21)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([10.0, 100.0, 1000.0])",
        "np.array([1e-10, 1e-5, 1e5, 1e10])",
        "np.array([0.123456789])",
        "np.array([9.87654321])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])",
        "np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])",
        "np.array([0.001, 0.008, 0.027, 0.064, 0.125])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.cbrt({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.cbrt({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "cbrt mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn cbrt_integer_input_promotes_to_float() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 8, 27], dtype=np.int32)",
        "np.array([1, 8, 27], dtype=np.int64)",
        "np.array([-1, -8, -27], dtype=np.int32)",
        "np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.cbrt({arr_expr}); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.cbrt({arr_expr}); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.split_whitespace().next(),
            rust_result.split_whitespace().next(),
            "cbrt dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn fabs_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([0.0, 1.0, -1.0])",
        "np.array([-0.0, 0.0])",
        "np.array([0.5, -0.5])",
        "np.array([1.5, -1.5, 2.5, -2.5])",
        "np.array([100.0, -100.0])",
        "np.array([1e10, -1e10])",
        "np.array([1e-10, -1e-10])",
        "np.array([1e308, -1e308])",
        "np.array([1e-308, -1e-308])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([np.nan, -np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([[1.0, -1.0], [2.0, -2.0]])",
        "np.array([[[1.0, -1.0]]])",
        "np.linspace(-10, 10, 21)",
        "np.linspace(-1, 1, 11)",
        "np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([-1.0, -2.0, -3.0, -4.0, -5.0])",
        "np.array([0.123456789, -0.123456789])",
        "np.array([9.87654321, -9.87654321])",
        "np.array([1.1, -2.2, 3.3, -4.4, 5.5])",
        "np.array([0.001, -0.001, 0.0001, -0.0001])",
        "np.array([1000.0, -1000.0, 10000.0, -10000.0])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.fabs({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.fabs({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-14),
            "fabs mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn fabs_integer_input_promotes_to_float() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, -1, 2, -2], dtype=np.int32)",
        "np.array([1, -1, 2, -2], dtype=np.int64)",
        "np.array([0, 1, -1, 100, -100], dtype=np.int32)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.fabs({arr_expr}); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.fabs({arr_expr}); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.split_whitespace().next(),
            rust_result.split_whitespace().next(),
            "fabs dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn fabs_rejects_complex_input() -> Result<(), String> {
    let script = r#"
import numpy as np
try:
    np.fabs(np.array([1+2j]))
    print("no_error")
except TypeError:
    print("TypeError")
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
try:
    fnp.fabs(np.array([1+2j]))
    print("no_error")
except TypeError:
    print("TypeError")
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "fabs complex rejection mismatch"
    );

    Ok(())
}

#[test]
fn cbrt_fabs_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &["cbrt", "fabs"] {
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
fn cbrt_fabs_bool_inputs_promote_like_numpy() -> Result<(), String> {
    for func in &["cbrt", "fabs"] {
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
            "{func} bool input promotion mismatch"
        );
    }

    Ok(())
}
