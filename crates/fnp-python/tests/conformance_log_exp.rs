//! Conformance tests for numpy.exp2, numpy.log2, numpy.log10 against NumPy oracle.
//!
//! Tests the implementations against NumPy for:
//! - exp2(x): 2**x element-wise
//! - log2(x): base-2 logarithm
//! - log10(x): base-10 logarithm

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

fn test_unary_function(func: &str, test_cases: &[&str], rel_tol: f64) -> Result<(), String> {
    for arr_expr in test_cases {
        let script = format!("import numpy as np; print(np.{func}({arr_expr}).flatten().tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.{func}({arr_expr}).flatten().tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, rel_tol),
            "{func} mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn exp2_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([2.0])",
        "np.array([3.0])",
        "np.array([-1.0])",
        "np.array([-2.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.array([10.0])",
        "np.array([-10.0])",
        "np.array([0.0, 1.0, 2.0, 3.0])",
        "np.array([-3.0, -2.0, -1.0, 0.0])",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([-0.1, -0.2, -0.3, -0.4, -0.5])",
        "np.linspace(-5, 5, 11)",
        "np.linspace(0, 10, 11)",
        "np.array([[1.0, 2.0], [3.0, 4.0]])",
        "np.array([[[0.0, 1.0]]])",
        "np.array([1023.0])",
        "np.array([-1074.0])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([1.5, 2.5, 3.5])",
        "np.array([-1.5, -2.5, -3.5])",
        "np.array([0.25, 0.5, 0.75, 1.0])",
        "np.array([8.0, 9.0, 10.0])",
        "np.array([-8.0, -9.0, -10.0])",
    ];
    test_unary_function("exp2", &test_cases, 1e-10)
}

#[test]
fn log2_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([2.0])",
        "np.array([4.0])",
        "np.array([8.0])",
        "np.array([0.5])",
        "np.array([0.25])",
        "np.array([0.125])",
        "np.array([1.0, 2.0, 4.0, 8.0])",
        "np.array([0.5, 0.25, 0.125, 0.0625])",
        "np.array([1.5, 2.5, 3.5])",
        "np.linspace(0.1, 10, 10)",
        "np.linspace(1, 1000, 10)",
        "np.array([[1.0, 2.0], [4.0, 8.0]])",
        "np.array([[[2.0, 4.0]]])",
        "np.array([1e10])",
        "np.array([1e-10])",
        "np.array([1e100])",
        "np.array([1e-100])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([3.0, 5.0, 7.0])",
        "np.array([16.0, 32.0, 64.0])",
        "np.array([1024.0, 2048.0, 4096.0])",
        "np.array([0.001, 0.01, 0.1])",
        "np.array([10.0, 100.0, 1000.0])",
        "np.array([np.e])",
        "np.array([np.pi])",
    ];
    test_unary_function("log2", &test_cases, 1e-10)
}

#[test]
fn log10_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([10.0])",
        "np.array([100.0])",
        "np.array([1000.0])",
        "np.array([0.1])",
        "np.array([0.01])",
        "np.array([0.001])",
        "np.array([1.0, 10.0, 100.0, 1000.0])",
        "np.array([0.1, 0.01, 0.001, 0.0001])",
        "np.array([1.5, 2.5, 3.5])",
        "np.linspace(0.1, 10, 10)",
        "np.linspace(1, 1000, 10)",
        "np.array([[1.0, 10.0], [100.0, 1000.0]])",
        "np.array([[[10.0, 100.0]]])",
        "np.array([1e10])",
        "np.array([1e-10])",
        "np.array([1e100])",
        "np.array([1e-100])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([2.0, 5.0, 7.0])",
        "np.array([50.0, 500.0, 5000.0])",
        "np.array([1e6, 1e9, 1e12])",
        "np.array([1e-6, 1e-9, 1e-12])",
        "np.array([np.e])",
        "np.array([np.pi])",
        "np.array([10.0 ** 0.5])",
    ];
    test_unary_function("log10", &test_cases, 1e-10)
}

#[test]
fn exp2_integer_input_promotes_to_float() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 1, 2, 3], dtype=np.int32)",
        "np.array([0, 1, 2, 3], dtype=np.int64)",
        "np.array([-3, -2, -1, 0], dtype=np.int32)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.exp2({arr_expr}); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.exp2({arr_expr}); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.split_whitespace().next(),
            rust_result.split_whitespace().next(),
            "exp2 dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn log2_integer_input_promotes_to_float() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 4, 8], dtype=np.int32)",
        "np.array([1, 2, 4, 8], dtype=np.int64)",
        "np.array([16, 32, 64, 128], dtype=np.int32)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.log2({arr_expr}); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.log2({arr_expr}); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.split_whitespace().next(),
            rust_result.split_whitespace().next(),
            "log2 dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn log10_integer_input_promotes_to_float() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 10, 100, 1000], dtype=np.int32)",
        "np.array([1, 10, 100, 1000], dtype=np.int64)",
        "np.array([10000, 100000], dtype=np.int64)",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; r = np.log10({arr_expr}); print(r.dtype, r.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.log10({arr_expr}); print(r.dtype, r.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.split_whitespace().next(),
            rust_result.split_whitespace().next(),
            "log10 dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn exp2_log2_log10_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &["exp2", "log2", "log10"] {
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
fn exp2_log2_log10_bool_inputs_promote_like_numpy() -> Result<(), String> {
    for func in &["exp2", "log2", "log10"] {
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

#[test]
fn exp2_log2_roundtrip_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
roundtrip = np.log2(np.exp2(x))
print(np.allclose(x, roundtrip))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
roundtrip = fnp.log2(fnp.exp2(x))
print(np.allclose(x, roundtrip))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "exp2/log2 roundtrip mismatch"
    );

    Ok(())
}

#[test]
fn log10_powers_of_ten() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
result = np.log10(x)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
result = fnp.log10(x)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-14),
        "log10 powers of ten mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn log2_powers_of_two() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
result = np.log2(x)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
result = fnp.log2(x)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-14),
        "log2 powers of two mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}
