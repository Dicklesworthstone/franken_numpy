//! Conformance tests for numpy rounding functions.
//!
//! Tests sign, floor, ceil, rint, trunc against NumPy oracle.

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

fn floats_match(a: &[f64], b: &[f64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        if x.is_nan() && y.is_nan() {
            true
        } else if *x == 0.0 && *y == 0.0 {
            x.to_bits() == y.to_bits()
        } else {
            x == y
        }
    })
}

#[test]
fn sign_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([-0.0, 0.0, 1.0])",
        "np.array([1.0, -1.0, 2.0, -2.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([0.5, -0.5, 0.0])",
        "np.array([1e10, -1e10, 0.0])",
        "np.array([1e-10, -1e-10, 0.0])",
        "np.array([np.inf, -np.inf, 0.0])",
        "np.array([np.nan, 0.0, 1.0])",
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([[1.0, -1.0], [0.0, 2.0]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([-1, -2, -3], dtype=np.float64)",
        "np.array([0.001, -0.001, 0.0])",
        "np.array([100.5, -100.5, 0.0])",
        "np.array([[[1.0, -1.0]]])",
        "np.array([0.0, -0.0])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.sign({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.sign({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_match(&numpy_vals, &rust_vals),
            "sign mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn floor_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([-0.5, -1.5, -2.5])",
        "np.array([0.1, 0.9, 1.1, 1.9])",
        "np.array([-0.1, -0.9, -1.1, -1.9])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([0.0])",
        "np.array([0.999999])",
        "np.array([-0.999999])",
        "np.array([1e10, -1e10])",
        "np.array([1.5e10, -1.5e10])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([[1.5, 2.5], [3.5, 4.5]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([0.25, 0.5, 0.75, 1.0])",
        "np.array([-0.25, -0.5, -0.75, -1.0])",
        "np.array([[[1.1, 2.2]]])",
        "np.array([2.5, -2.5, 3.5, -3.5])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.floor({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.floor({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_match(&numpy_vals, &rust_vals),
            "floor mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn ceil_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([-0.5, -1.5, -2.5])",
        "np.array([0.1, 0.9, 1.1, 1.9])",
        "np.array([-0.1, -0.9, -1.1, -1.9])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([0.0])",
        "np.array([0.000001])",
        "np.array([-0.000001])",
        "np.array([1e10, -1e10])",
        "np.array([1.5e10, -1.5e10])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([[1.5, 2.5], [3.5, 4.5]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([0.25, 0.5, 0.75, 1.0])",
        "np.array([-0.25, -0.5, -0.75, -1.0])",
        "np.array([[[1.1, 2.2]]])",
        "np.array([2.5, -2.5, 3.5, -3.5])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.ceil({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.ceil({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_match(&numpy_vals, &rust_vals),
            "ceil mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn rint_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([-0.5, -1.5, -2.5])",
        "np.array([0.4, 0.6, 1.4, 1.6])",
        "np.array([-0.4, -0.6, -1.4, -1.6])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([0.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.array([1e10, -1e10])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([[1.5, 2.5], [3.5, 4.5]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([0.25, 0.75, 1.25, 1.75])",
        "np.array([-0.25, -0.75, -1.25, -1.75])",
        "np.array([[[1.1, 2.9]]])",
        "np.array([2.5, -2.5, 3.5, -3.5])",
        "np.array([0.49, 0.51, 1.49, 1.51])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.rint({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.rint({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_match(&numpy_vals, &rust_vals),
            "rint mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trunc_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([-0.5, -1.5, -2.5])",
        "np.array([0.9, 1.9, 2.9])",
        "np.array([-0.9, -1.9, -2.9])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([0.0])",
        "np.array([0.999999])",
        "np.array([-0.999999])",
        "np.array([1e10, -1e10])",
        "np.array([1.5e10, -1.5e10])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([[1.5, 2.5], [3.5, 4.5]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([0.1, 0.5, 0.9, 1.0])",
        "np.array([-0.1, -0.5, -0.9, -1.0])",
        "np.array([[[1.7, 2.3]]])",
        "np.array([2.9, -2.9, 3.1, -3.1])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trunc({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!("print(fnp.trunc({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_match(&numpy_vals, &rust_vals),
            "trunc mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn rounding_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &["sign", "floor", "ceil", "rint", "trunc"] {
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
