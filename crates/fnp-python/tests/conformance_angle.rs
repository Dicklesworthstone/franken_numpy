//! Conformance tests for numpy angle conversion functions.
//!
//! Tests degrees and radians against NumPy oracle.

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
            let max_val = x.abs().max(y.abs());
            diff <= rel_tol * max_val || diff < 1e-15
        }
    })
}

#[test]
fn degrees_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([np.pi])",
        "np.array([np.pi / 2])",
        "np.array([np.pi / 4])",
        "np.array([np.pi / 6])",
        "np.array([np.pi / 3])",
        "np.array([2 * np.pi])",
        "np.array([-np.pi])",
        "np.array([-np.pi / 2])",
        "np.array([np.pi, -np.pi])",
        "np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([0.1, 0.2, 0.3])",
        "np.array([10.0, 100.0, 1000.0])",
        "np.array([np.pi / 180])",
        "np.array([np.pi * 2 / 3])",
        "np.array([np.pi * 3 / 4])",
        "np.array([np.pi * 5 / 6])",
        "np.array([[np.pi / 4, np.pi / 2], [np.pi, 2 * np.pi]])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.nan, np.pi])",
        "np.array([[[np.pi]]])",
        "np.array([1e-10, 1e10])",
    ];

    for arr_expr in &test_cases {
        let script =
            format!("import numpy as np; print(np.degrees({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_script(format!("print(fnp.degrees({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "degrees mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn radians_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([180.0])",
        "np.array([90.0])",
        "np.array([45.0])",
        "np.array([30.0])",
        "np.array([60.0])",
        "np.array([360.0])",
        "np.array([-180.0])",
        "np.array([-90.0])",
        "np.array([180.0, -180.0])",
        "np.array([0.0, 90.0, 180.0, 270.0, 360.0])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([0.1, 0.2, 0.3])",
        "np.array([10.0, 100.0, 1000.0])",
        "np.array([1.0])",
        "np.array([120.0])",
        "np.array([135.0])",
        "np.array([150.0])",
        "np.array([[45.0, 90.0], [180.0, 360.0]])",
        "np.array([np.inf, -np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.nan, 180.0])",
        "np.array([[[90.0]]])",
        "np.array([1e-10, 1e10])",
    ];

    for arr_expr in &test_cases {
        let script =
            format!("import numpy as np; print(np.radians({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_script(format!("print(fnp.radians({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "radians mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn degrees_radians_roundtrip_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0, 45.0, 90.0, 135.0, 180.0])",
        "np.array([-45.0, -90.0, -180.0])",
        "np.array([30.0, 60.0, 120.0, 150.0])",
        "np.array([1.0, 10.0, 100.0])",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; print(np.degrees(np.radians({arr_expr})).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_script(format!(
            "print(fnp.degrees(fnp.radians({arr_expr})).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "degrees(radians()) roundtrip mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn angle_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &["degrees", "radians"] {
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
