//! Conformance tests for numpy floating-point classification functions.
//!
//! Tests isnan, isinf, isfinite, signbit against NumPy oracle.

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

fn parse_bool_list(s: &str) -> Vec<bool> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|token| token.trim() == "True")
        .collect()
}

#[test]
fn isnan_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([np.nan, 0.0, 1.0])",
        "np.array([0.0, np.nan, 1.0])",
        "np.array([0.0, 1.0, np.nan])",
        "np.array([np.nan, np.nan, np.nan])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([np.nan])",
        "np.array([0.0])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([1e308, -1e308, 1e-308])",
        "np.array([0.0, -0.0, 0.0])",
        "np.array([[1.0, np.nan], [np.nan, 2.0]])",
        "np.array([[np.nan, np.nan], [np.nan, np.nan]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([1, 2, 3], dtype=np.float32)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([-0.1, -0.2, -0.3])",
        "np.array([1e-100, 1e100, np.nan])",
        "np.array([np.nan, 1e-100, 1e100])",
        "np.array([float('nan'), 0.0, 1.0])",
        "np.array([0.0, float('nan'), 1.0])",
        "np.array([0.0, 1.0, float('nan')])",
        "np.array([[[np.nan]]])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.isnan({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script =
            fnp_script(format!("print(fnp.isnan({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "isnan mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn isinf_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([np.inf, 0.0, 1.0])",
        "np.array([0.0, np.inf, 1.0])",
        "np.array([0.0, 1.0, np.inf])",
        "np.array([-np.inf, 0.0, 1.0])",
        "np.array([0.0, -np.inf, 1.0])",
        "np.array([0.0, 1.0, -np.inf])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([np.inf, np.inf, np.inf])",
        "np.array([-np.inf, -np.inf, -np.inf])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0])",
        "np.array([1e308, -1e308, 1e-308])",
        "np.array([1e309, -1e309])",
        "np.array([[1.0, np.inf], [-np.inf, 2.0]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([1, 2, 3], dtype=np.float32)",
        "np.array([float('inf'), 0.0, 1.0])",
        "np.array([0.0, float('inf'), 1.0])",
        "np.array([float('-inf'), 0.0, 1.0])",
        "np.array([[[np.inf]]])",
        "np.array([0.0, -0.0, np.inf, -np.inf])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.isinf({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script =
            fnp_script(format!("print(fnp.isinf({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "isinf mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn isfinite_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([np.inf, 0.0, 1.0])",
        "np.array([0.0, np.inf, 1.0])",
        "np.array([-np.inf, 0.0, 1.0])",
        "np.array([np.nan, 0.0, 1.0])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([1e308, -1e308, 1e-308])",
        "np.array([[1.0, np.inf], [np.nan, 2.0]])",
        "np.array([[1.0, 2.0], [3.0, 4.0]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([1, 2, 3], dtype=np.float32)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([-0.1, -0.2, -0.3])",
        "np.array([1e-100, 1e100, 0.0])",
        "np.array([0.0, -0.0, 1.0])",
        "np.array([[[1.0, np.nan], [np.inf, 2.0]]])",
        "np.array([float('inf'), float('-inf'), float('nan'), 1.0])",
        "np.array([1e-308, 2e-308, 3e-308])",
        "np.array([1.7976931348623157e+308])",
    ];

    for arr_expr in &test_cases {
        let script =
            format!("import numpy as np; print(np.isfinite({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script =
            fnp_script(format!("print(fnp.isfinite({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "isfinite mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn signbit_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 1.0, -1.0])",
        "np.array([-0.0, 0.0, 1.0])",
        "np.array([0.0, -0.0, -1.0])",
        "np.array([1.0, -1.0, 2.0, -2.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([np.inf, -np.inf])",
        "np.array([-np.inf, np.inf])",
        "np.array([np.nan, -np.nan])",
        "np.array([0.0])",
        "np.array([-0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([1e308, -1e308])",
        "np.array([-1e308, 1e308])",
        "np.array([1e-308, -1e-308])",
        "np.array([[1.0, -1.0], [-2.0, 2.0]])",
        "np.array([1, 2, 3], dtype=np.float64)",
        "np.array([-1, -2, -3], dtype=np.float64)",
        "np.array([0.5, -0.5, 0.25, -0.25])",
        "np.array([-0.001, 0.001, -0.0001])",
        "np.array([[[1.0, -1.0]]])",
        "np.array([float('-0.0'), float('0.0')])",
    ];

    for arr_expr in &test_cases {
        let script =
            format!("import numpy as np; print(np.signbit({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script =
            fnp_script(format!("print(fnp.signbit({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "signbit mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn fp_classify_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &["isnan", "isinf", "isfinite", "signbit"] {
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
