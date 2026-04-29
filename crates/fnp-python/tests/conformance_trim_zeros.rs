//! Conformance tests for numpy.trim_zeros against NumPy oracle.
//!
//! Tests the native Rust trim_zeros implementation against NumPy across various
//! input arrays and trim modes.

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

fn fnp_trim_zeros_script(body: String) -> String {
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
            t.parse().ok()
        })
        .collect()
}

#[test]
fn trim_zeros_default_mode_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic cases
        "np.array([0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 1, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5])",
        "np.array([0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1])",
        "np.array([1])",
        "np.array([0])",
        // Floating point
        "np.array([0.0, 0.0, 1.5, 2.5, 0.0])",
        "np.array([0.0, 1.0, 0.0, 0.0])",
        "np.array([1.1, 2.2, 3.3])",
        "np.array([0.0, 0.0, 0.0])",
        "np.array([0.0, 0.5, 0.0])",
        "np.array([0.001, 0.0, 0.0])",
        "np.array([0.0, 0.0, 0.999])",
        // Negative values
        "np.array([0, 0, -1, -2, 0, 0])",
        "np.array([-1, 0, 0, 0])",
        "np.array([0, 0, 0, -1])",
        "np.array([-1, -2, -3])",
        "np.array([0, -5, 0, -10, 0])",
        // Mixed signs
        "np.array([0, 1, -1, 2, -2, 0])",
        "np.array([0, 0, 1, -1, 0, 0])",
        "np.array([-1, 0, 1, 0, -1])",
        "np.array([0, 0, -0.5, 0.5, 0, 0])",
        // Single element arrays
        "np.array([5])",
        "np.array([-5])",
        "np.array([0.5])",
        "np.array([-0.5])",
        // Larger arrays
        "np.array([0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])",
        "np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
        // Different dtypes
        "np.array([0, 0, 1, 0], dtype=np.int32)",
        "np.array([0, 0, 1, 0], dtype=np.int64)",
        "np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)",
        "np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)",
        "np.array([0, 1, 0], dtype=np.uint8)",
        "np.array([0, 1, 0], dtype=np.int16)",
        // Edge patterns
        "np.array([0, 1, 0, 1, 0])",
        "np.array([1, 0, 1, 0, 1])",
        "np.array([0, 0, 1, 1, 0, 0])",
        "np.array([1, 1, 0, 0, 1, 1])",
        "np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])",
        // More variety
        "np.array([0, 0, 100, 200, 0, 0])",
        "np.array([0, 0, -100, -200, 0, 0])",
        "np.array([0.0, 0.0, 1e-10, 0.0])",
        "np.array([0.0, 1e10, 0.0, 0.0])",
        "np.array([0, 0, 42, 0, 0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trim_zeros({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_trim_zeros_script(format!("print(fnp.trim_zeros({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "trim_zeros default mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_front_mode_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 1, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5])",
        "np.array([0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1])",
        "np.array([0.0, 0.0, 1.5, 2.5, 0.0])",
        "np.array([0, 0, -1, -2, 0, 0])",
        "np.array([-1, 0, 0, 0])",
        "np.array([0, 0, 0, -1])",
        "np.array([0, 0, 0, 0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trim_zeros({arr_expr}, 'f').tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_trim_zeros_script(format!("print(fnp.trim_zeros({arr_expr}, 'f').tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "trim_zeros front mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_back_mode_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 1, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5])",
        "np.array([0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1])",
        "np.array([0.0, 0.0, 1.5, 2.5, 0.0])",
        "np.array([0, 0, -1, -2, 0, 0])",
        "np.array([-1, 0, 0, 0])",
        "np.array([0, 0, 0, -1])",
        "np.array([0, 0, 1, 2, 3, 0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trim_zeros({arr_expr}, 'b').tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_trim_zeros_script(format!("print(fnp.trim_zeros({arr_expr}, 'b').tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "trim_zeros back mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_empty_result_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 0, 0])",
        "np.array([0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])",
        "np.array([0.0, 0.0, 0.0])",
    ];

    for arr_expr in &test_cases {
        for trim in &["fb", "f", "b"] {
            let script =
                format!("import numpy as np; print(np.trim_zeros({arr_expr}, '{trim}').tolist())");
            let numpy_result = numpy_oracle(&script)?;
            let numpy_vals = parse_float_list(&numpy_result);

            let rust_script = fnp_trim_zeros_script(format!(
                "print(fnp.trim_zeros({arr_expr}, '{trim}').tolist())"
            ));
            let rust_result = numpy_oracle(&rust_script)?;
            let rust_vals = parse_float_list(&rust_result);

            assert_eq!(
                numpy_vals, rust_vals,
                "trim_zeros empty result mismatch for {arr_expr} trim='{trim}'\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
            );
        }
    }

    Ok(())
}

#[test]
fn trim_zeros_no_zeros_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 3])",
        "np.array([1])",
        "np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
        "np.array([-1, -2, -3])",
        "np.array([0.5, 1.5, 2.5])",
    ];

    for arr_expr in &test_cases {
        for trim in &["fb", "f", "b"] {
            let script =
                format!("import numpy as np; print(np.trim_zeros({arr_expr}, '{trim}').tolist())");
            let numpy_result = numpy_oracle(&script)?;
            let numpy_vals = parse_float_list(&numpy_result);

            let rust_script = fnp_trim_zeros_script(format!(
                "print(fnp.trim_zeros({arr_expr}, '{trim}').tolist())"
            ));
            let rust_result = numpy_oracle(&rust_script)?;
            let rust_vals = parse_float_list(&rust_result);

            assert_eq!(
                numpy_vals, rust_vals,
                "trim_zeros no zeros mismatch for {arr_expr} trim='{trim}'\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
            );
        }
    }

    Ok(())
}
