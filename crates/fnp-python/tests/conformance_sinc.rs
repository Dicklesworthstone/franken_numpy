//! Conformance tests for numpy.sinc against NumPy oracle.
//!
//! Tests the native Rust sinc implementation against NumPy.
//! sinc(x) = sin(pi*x) / (pi*x) with sinc(0) = 1

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

fn parse_float_list(s: &str) -> Vec<f64> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .filter_map(|token| {
            let t = token.trim().trim_end_matches('.');
            if t == "nan" || t == "NaN" {
                Some(f64::NAN)
            } else if t == "inf" || t == "Inf" {
                Some(f64::INFINITY)
            } else if t == "-inf" || t == "-Inf" {
                Some(f64::NEG_INFINITY)
            } else {
                t.parse().ok()
            }
        })
        .collect()
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
fn sinc_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.array([2.0])",
        "np.array([-2.0])",
        "np.array([0.0, 1.0, 2.0])",
        "np.array([-2.0, -1.0, 0.0, 1.0, 2.0])",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([-0.1, -0.2, -0.3, -0.4, -0.5])",
        "np.array([0.25, 0.5, 0.75, 1.0])",
        "np.array([1.5, 2.5, 3.5])",
        "np.array([-1.5, -2.5, -3.5])",
        "np.array([0.001])",
        "np.array([-0.001])",
        "np.array([0.0001])",
        "np.array([1e-10])",
        "np.array([10.0])",
        "np.array([-10.0])",
        "np.array([100.0])",
        "np.array([0.3, 0.6, 0.9])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([[0.0, 1.0], [2.0, 3.0]])",
        "np.array([[[0.5]]])",
        "np.array([0.0, 0.0, 0.0])",
        "np.array([1.0, 1.0, 1.0])",
        "np.array([-0.0])",
        "np.array([0.99, 1.01])",
        "np.array([1.99, 2.01])",
        "np.linspace(-5, 5, 11)",
        "np.linspace(-1, 1, 21)",
        "np.linspace(0, 10, 11)",
        "np.array([np.pi])",
        "np.array([np.pi / 2])",
        "np.array([np.pi / 4])",
        "np.array([2 * np.pi])",
        "np.array([-np.pi])",
        "np.array([0.123456789])",
        "np.array([1.234567890])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.sinc({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.sinc({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "sinc mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn sinc_special_values_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([0.0, np.inf, np.nan])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.sinc({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.sinc({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "sinc special values mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn sinc_integer_zeros_match_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.sinc(np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)).tolist())";
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result);

    let rust_script = fnp_script(
        "print(fnp.sinc(np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)).tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result);

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "sinc integer zeros mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn sinc_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.sinc(np.array([], dtype=np.float64)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.sinc(np.array([], dtype=np.float64)).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "sinc empty array mismatch"
    );

    Ok(())
}
