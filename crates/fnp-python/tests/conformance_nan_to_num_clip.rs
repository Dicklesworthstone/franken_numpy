//! Conformance tests for numpy.nan_to_num and numpy.clip against NumPy oracle.
//!
//! Tests:
//! - nan_to_num(x, nan=0.0, posinf=None, neginf=None): replace NaN/inf values
//! - clip(a, a_min, a_max): limit array values to [a_min, a_max]

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
fn nan_to_num_default_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([np.nan])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([1.0, np.nan, 3.0])",
        "np.array([np.inf, 2.0, -np.inf])",
        "np.array([np.nan, np.inf, -np.inf])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([0.0])",
        "np.array([-0.0])",
        "np.array([1e308, np.inf, -1e308, -np.inf])",
        "np.array([[np.nan, 1.0], [np.inf, -np.inf]])",
        "np.array([[[np.nan]]])",
        "np.array([1.5, np.nan, 2.5, np.nan, 3.5])",
    ];

    for arr_expr in &test_cases {
        let script =
            format!("import numpy as np; print(np.nan_to_num({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script =
            fnp_script(format!("print(fnp.nan_to_num({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "nan_to_num default mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn nan_to_num_custom_nan_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([np.nan])", "999.0"),
        ("np.array([np.nan, 1.0, np.nan])", "-1.0"),
        ("np.array([1.0, np.nan, 3.0])", "0.0"),
        ("np.array([np.nan, np.nan, np.nan])", "42.0"),
    ];

    for (arr_expr, nan_val) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.nan_to_num({arr_expr}, nan={nan_val}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.nan_to_num({arr_expr}, nan={nan_val}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "nan_to_num custom nan mismatch for {arr_expr} with nan={nan_val}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn nan_to_num_custom_posinf_neginf_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([np.inf])", "1e10", "-1e10"),
        ("np.array([-np.inf])", "1e10", "-1e10"),
        ("np.array([np.inf, -np.inf])", "100.0", "-100.0"),
        ("np.array([1.0, np.inf, -np.inf, 2.0])", "999.0", "-999.0"),
    ];

    for (arr_expr, posinf, neginf) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.nan_to_num({arr_expr}, posinf={posinf}, neginf={neginf}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.nan_to_num({arr_expr}, posinf={posinf}, neginf={neginf}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "nan_to_num custom inf mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn nan_to_num_all_params_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0])
result = np.nan_to_num(x, nan=-1.0, posinf=1000.0, neginf=-1000.0)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0])
result = fnp.nan_to_num(x, nan=-1.0, posinf=1000.0, neginf=-1000.0)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "nan_to_num all params mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn nan_to_num_empty_array_matches_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.nan_to_num(np.array([], dtype=np.float64)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.nan_to_num(np.array([], dtype=np.float64)).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "nan_to_num empty array mismatch"
    );

    Ok(())
}

#[test]
fn clip_basic_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3, 4, 5])", "2", "4"),
        ("np.array([0.0, 0.5, 1.0, 1.5, 2.0])", "0.25", "1.75"),
        ("np.array([-5, -3, 0, 3, 5])", "-2", "2"),
        ("np.array([1.0, 2.0, 3.0, 4.0, 5.0])", "None", "3.0"),
        ("np.array([1.0, 2.0, 3.0, 4.0, 5.0])", "3.0", "None"),
        ("np.array([[1, 2, 3], [4, 5, 6]])", "2", "5"),
        ("np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])", "3", "6"),
    ];

    for (arr_expr, a_min, a_max) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.clip({arr_expr}, {a_min}, {a_max}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.clip({arr_expr}, {a_min}, {a_max}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "clip mismatch for {arr_expr} with ({a_min}, {a_max})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn clip_nan_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([np.nan, 1.0, 2.0, np.nan])", "0.5", "1.5"),
        ("np.array([1.0, np.nan, 3.0])", "0.0", "2.0"),
        ("np.array([np.nan, np.nan, np.nan])", "0.0", "1.0"),
    ];

    for (arr_expr, a_min, a_max) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.clip({arr_expr}, {a_min}, {a_max}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.clip({arr_expr}, {a_min}, {a_max}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "clip nan handling mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn clip_inf_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([np.inf, 1.0, 2.0, -np.inf])", "0.0", "1.5"),
        ("np.array([1.0, np.inf, -np.inf])", "-10.0", "10.0"),
        ("np.array([np.inf, -np.inf])", "0.0", "0.0"),
    ];

    for (arr_expr, a_min, a_max) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.clip({arr_expr}, {a_min}, {a_max}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.clip({arr_expr}, {a_min}, {a_max}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "clip inf handling mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn clip_empty_array_matches_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.clip(np.array([], dtype=np.float64), 0, 1).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.clip(np.array([], dtype=np.float64), 0, 1).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "clip empty array mismatch"
    );

    Ok(())
}

#[test]
fn clip_broadcast_bounds_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a_min = np.array([2, 3, 4])
a_max = np.array([5, 6, 7])
print(np.clip(x, a_min, a_max).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a_min = np.array([2, 3, 4])
a_max = np.array([5, 6, 7])
print(fnp.clip(x, a_min, a_max).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "clip broadcast bounds mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn clip_dtype_preserved_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 3, 4, 5], dtype=np.int32)",
        "np.array([1.0, 2.0, 3.0], dtype=np.float32)",
        "np.array([1.0, 2.0, 3.0], dtype=np.float64)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.clip({arr_expr}, 2, 4); print(r.dtype)"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.clip({arr_expr}, 2, 4); print(r.dtype)"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "clip dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn nan_to_num_dtype_preserved_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([np.nan, 1.0, np.inf], dtype=np.float32)",
        "np.array([np.nan, 1.0, np.inf], dtype=np.float64)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.nan_to_num({arr_expr}); print(r.dtype)"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.nan_to_num({arr_expr}); print(r.dtype)"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "nan_to_num dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}
