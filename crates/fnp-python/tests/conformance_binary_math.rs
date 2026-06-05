//! Conformance tests for numpy binary math functions.
//!
//! Tests hypot and copysign against NumPy oracle.

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
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|token| {
            let t = token
                .trim()
                .trim_matches(|c| c == '[' || c == ']')
                .trim_end_matches('.');
            if t == "nan" || t == "NaN" {
                Ok(f64::NAN)
            } else if t == "inf" || t == "Inf" {
                Ok(f64::INFINITY)
            } else if t == "-inf" || t == "-Inf" {
                Ok(f64::NEG_INFINITY)
            } else {
                t.parse()
                    .map_err(|error| format!("failed to parse float token {token:?}: {error}"))
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
            x.to_bits() == y.to_bits()
        } else {
            let diff = (x - y).abs();
            let max_val = x.abs().max(y.abs());
            diff <= rel_tol * max_val || diff < 1e-14
        }
    })
}

#[test]
fn hypot_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([3.0])", "np.array([4.0])"),
        ("np.array([0.0])", "np.array([0.0])"),
        ("np.array([1.0])", "np.array([0.0])"),
        ("np.array([0.0])", "np.array([1.0])"),
        ("np.array([1.0])", "np.array([1.0])"),
        ("np.array([3.0, 5.0, 8.0])", "np.array([4.0, 12.0, 15.0])"),
        ("np.array([-3.0])", "np.array([4.0])"),
        ("np.array([3.0])", "np.array([-4.0])"),
        ("np.array([-3.0])", "np.array([-4.0])"),
        ("np.array([1e10])", "np.array([1e10])"),
        ("np.array([1e-10])", "np.array([1e-10])"),
        ("np.array([1e308])", "np.array([1e308])"),
        ("np.array([np.inf])", "np.array([1.0])"),
        ("np.array([1.0])", "np.array([np.inf])"),
        ("np.array([np.inf])", "np.array([np.inf])"),
        ("np.array([-np.inf])", "np.array([1.0])"),
        ("np.array([np.nan])", "np.array([1.0])"),
        ("np.array([1.0])", "np.array([np.nan])"),
        ("np.array([np.nan])", "np.array([np.nan])"),
        (
            "np.array([[1.0, 2.0], [3.0, 4.0]])",
            "np.array([[1.0, 2.0], [3.0, 4.0]])",
        ),
        ("np.array([0.6])", "np.array([0.8])"),
        ("np.array([5.0, 7.0])", "np.array([12.0, 24.0])"),
        ("np.array([1.0, 2.0, 3.0])", "np.array([1.0, 2.0, 3.0])"),
        ("np.array([0.0, 0.0, 0.0])", "np.array([0.0, 0.0, 0.0])"),
        ("np.array([1e-300])", "np.array([1e-300])"),
    ];

    for (x_expr, y_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.hypot({x_expr}, {y_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.hypot({x_expr}, {y_expr}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "hypot mismatch for ({x_expr}, {y_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn copysign_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1.0])", "np.array([1.0])"),
        ("np.array([1.0])", "np.array([-1.0])"),
        ("np.array([-1.0])", "np.array([1.0])"),
        ("np.array([-1.0])", "np.array([-1.0])"),
        ("np.array([0.0])", "np.array([1.0])"),
        ("np.array([0.0])", "np.array([-1.0])"),
        ("np.array([-0.0])", "np.array([1.0])"),
        ("np.array([-0.0])", "np.array([-1.0])"),
        ("np.array([1.0])", "np.array([0.0])"),
        ("np.array([1.0])", "np.array([-0.0])"),
        ("np.array([1.0, -1.0, 2.0])", "np.array([1.0, 1.0, -1.0])"),
        ("np.array([5.0])", "np.array([-3.0])"),
        ("np.array([-5.0])", "np.array([3.0])"),
        ("np.array([1e10])", "np.array([-1.0])"),
        ("np.array([1e-10])", "np.array([-1.0])"),
        ("np.array([np.inf])", "np.array([1.0])"),
        ("np.array([np.inf])", "np.array([-1.0])"),
        ("np.array([-np.inf])", "np.array([1.0])"),
        ("np.array([-np.inf])", "np.array([-1.0])"),
        ("np.array([np.nan])", "np.array([1.0])"),
        ("np.array([np.nan])", "np.array([-1.0])"),
        ("np.array([1.0])", "np.array([np.nan])"),
        (
            "np.array([[1.0, -1.0], [2.0, -2.0]])",
            "np.array([[-1.0, 1.0], [-1.0, 1.0]])",
        ),
        ("np.array([1.0, 2.0, 3.0])", "np.array([-1.0, -1.0, -1.0])"),
        ("np.array([100.5, -100.5])", "np.array([-1.0, 1.0])"),
    ];

    for (x_expr, y_expr) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.copysign({x_expr}, {y_expr}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.copysign({x_expr}, {y_expr}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "copysign mismatch for ({x_expr}, {y_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn hypot_broadcast_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([3.0, 5.0, 8.0])", "np.array([4.0])"),
        ("np.array([3.0])", "np.array([4.0, 12.0, 15.0])"),
        ("np.array([[1.0], [2.0]])", "np.array([1.0, 2.0])"),
    ];

    for (x_expr, y_expr) in &test_cases {
        let script =
            format!("import numpy as np; print(np.hypot({x_expr}, {y_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.hypot({x_expr}, {y_expr}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "hypot broadcast mismatch for ({x_expr}, {y_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn binary_math_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &["hypot", "copysign"] {
        let script = format!(
            "import numpy as np; print(np.{func}(np.array([], dtype=np.float64), np.array([], dtype=np.float64)).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "print(fnp.{func}(np.array([], dtype=np.float64), np.array([], dtype=np.float64)).tolist())"
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
fn parse_float_list_rejects_unparseable_tokens() {
    let error = parse_float_list("[1.0, broken, 2.0]")
        .expect_err("malformed oracle output must fail the harness");
    assert!(error.contains("broken"), "unexpected parse error: {error}");
}

#[test]
fn binary_math_scalar_return_type_matches_numpy() -> Result<(), String> {
    for func in &["hypot", "copysign"] {
        let script = format!(
            "import numpy as np; x = np.float64(3.0); y = np.float64(4.0); r = np.{func}(x, y); print(type(r).__name__, r)"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "x = np.float64(3.0); y = np.float64(4.0); r = fnp.{func}(x, y); print(type(r).__name__, r)"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} scalar return type mismatch\nnumpy: {numpy_result}\nfnp: {rust_result}"
        );
    }

    Ok(())
}
