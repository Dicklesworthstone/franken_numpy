//! Conformance tests for numpy.frexp and numpy.modf against NumPy oracle.
//!
//! Tests tuple-returning decomposition functions:
//! - frexp(x): returns (mantissa, exponent) where x = mantissa * 2**exponent
//! - modf(x): returns (fractional, integral) parts

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
fn frexp_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([2.0])",
        "np.array([4.0])",
        "np.array([0.5])",
        "np.array([0.25])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([-2.0])",
        "np.array([-0.5])",
        "np.array([3.0])",
        "np.array([1.5])",
        "np.array([1e10])",
        "np.array([1e-10])",
        "np.array([1e100])",
        "np.array([1e-100])",
        "np.array([1e308])",
        "np.array([1e-308])",
        "np.array([1.0, 2.0, 3.0, 4.0])",
        "np.array([-1.0, -2.0, -3.0, -4.0])",
        "np.array([0.5, 1.0, 1.5, 2.0])",
        "np.linspace(0.1, 10, 10)",
        "np.array([[1.0, 2.0], [4.0, 8.0]])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([-0.0])",
        "np.array([np.finfo(float).eps])",
        "np.array([np.finfo(float).tiny])",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; m, e = np.frexp({arr_expr}); print(m.flatten().tolist(), e.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "m, e = fnp.frexp({arr_expr}); print(m.flatten().tolist(), e.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "frexp mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn modf_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.5])",
        "np.array([2.7])",
        "np.array([0.5])",
        "np.array([0.0])",
        "np.array([-1.5])",
        "np.array([-2.7])",
        "np.array([-0.5])",
        "np.array([3.14159])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([10.25])",
        "np.array([-10.25])",
        "np.array([0.999])",
        "np.array([-0.999])",
        "np.array([1e10 + 0.5])",
        "np.array([1.5, 2.5, 3.5, 4.5])",
        "np.array([-1.5, -2.5, -3.5, -4.5])",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.linspace(0, 5, 6)",
        "np.array([[1.5, 2.5], [3.5, 4.5]])",
        "np.array([np.inf])",
        "np.array([-np.inf])",
        "np.array([np.nan])",
        "np.array([0.0, np.inf, np.nan])",
        "np.array([-0.0])",
        "np.array([123.456])",
        "np.array([-123.456])",
        "np.array([0.123456789])",
    ];

    for arr_expr in &test_cases {
        let script = format!(
            "import numpy as np; f, i = np.modf({arr_expr}); print(f.flatten().tolist(), i.flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "f, i = fnp.modf({arr_expr}); print(f.flatten().tolist(), i.flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "modf mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn frexp_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; m, e = np.frexp(np.array([], dtype=np.float64)); print(m.tolist(), e.tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "m, e = fnp.frexp(np.array([], dtype=np.float64)); print(m.tolist(), e.tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp empty array mismatch"
    );

    Ok(())
}

#[test]
fn modf_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; f, i = np.modf(np.array([], dtype=np.float64)); print(f.tolist(), i.tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "f, i = fnp.modf(np.array([], dtype=np.float64)); print(f.tolist(), i.tolist())".into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf empty array mismatch"
    );

    Ok(())
}

#[test]
fn frexp_dtype_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; m, e = np.frexp(np.array([1.0, 2.0])); print(m.dtype, e.dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("m, e = fnp.frexp(np.array([1.0, 2.0])); print(m.dtype, e.dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp dtype mismatch"
    );

    Ok(())
}

#[test]
fn modf_dtype_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; f, i = np.modf(np.array([1.5, 2.5])); print(f.dtype, i.dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("f, i = fnp.modf(np.array([1.5, 2.5])); print(f.dtype, i.dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf dtype mismatch"
    );

    Ok(())
}

#[test]
fn frexp_roundtrip_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.25])
m, e = np.frexp(x)
reconstructed = np.ldexp(m, e)
print(np.allclose(x, reconstructed))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 0.25])
m, e = fnp.frexp(x)
reconstructed = fnp.ldexp(m, e)
print(np.allclose(x, reconstructed))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp roundtrip mismatch"
    );

    Ok(())
}

#[test]
fn modf_sum_matches_original() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1.5, 2.7, -3.2, 4.9, -0.5])
f, i = np.modf(x)
print(np.allclose(f + i, x))
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1.5, 2.7, -3.2, 4.9, -0.5])
f, i = fnp.modf(x)
print(np.allclose(f + i, x))
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf sum mismatch"
    );

    Ok(())
}

#[test]
fn frexp_shape_preserved() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
m, e = np.frexp(x)
print(m.shape, e.shape)
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
m, e = fnp.frexp(x)
print(m.shape, e.shape)
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "frexp shape mismatch"
    );

    Ok(())
}

#[test]
fn modf_shape_preserved() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
f, i = np.modf(x)
print(f.shape, i.shape)
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
f, i = fnp.modf(x)
print(f.shape, i.shape)
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "modf shape mismatch"
    );

    Ok(())
}
