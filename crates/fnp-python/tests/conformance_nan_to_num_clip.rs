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

        let rust_script = fnp_script(format!(
            "print(fnp.nan_to_num({arr_expr}).flatten().tolist())"
        ));
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
fn nan_to_num_custom_nan_preserves_dtype_inf_defaults() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
r = np.nan_to_num(x, nan=7.0)
print(r.dtype, r.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
r = fnp.nan_to_num(x, nan=7.0)
print(r.dtype, r.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "nan_to_num custom nan should use dtype-specific default inf bounds"
    );

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
        let script = format!("import numpy as np; r = np.clip({arr_expr}, 2, 4); print(r.dtype)");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!("r = fnp.clip({arr_expr}, 2, 4); print(r.dtype)"));
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
        let script = format!("import numpy as np; r = np.nan_to_num({arr_expr}); print(r.dtype)");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!("r = fnp.nan_to_num({arr_expr}); print(r.dtype)"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "nan_to_num dtype mismatch for {arr_expr}"
        );
    }

    Ok(())
}

#[test]
fn nan_to_num_clip_scalar_return_type_matches_numpy() -> Result<(), String> {
    let clip_script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.clip(x, 2.0, 4.0)
np_result = np.clip(x, 2.0, 4.0)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&clip_script)?;
    assert!(
        result.trim().starts_with("True"),
        "clip scalar return type should match numpy: {result}"
    );

    let nan_to_num_script = fnp_script(
        r#"
x = np.float64(np.nan)
fnp_result = fnp.nan_to_num(x)
np_result = np.nan_to_num(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&nan_to_num_script)?;
    assert!(
        result.trim().starts_with("True"),
        "nan_to_num scalar return type should match numpy: {result}"
    );

    Ok(())
}

/// Locks nan_to_num on COMPLEX input (the parity fix that delegates complex to
/// numpy): complex128/complex64 with NaN/+-inf in real and/or imag parts, default
/// and custom nan/posinf/neginf, must be BYTE-identical to numpy.nan_to_num plus a
/// sha256 golden over the result bytes.
#[test]
fn nan_to_num_complex_matches_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
c128 = np.array([1+np.nan*1j, np.inf+2j, -np.inf-np.inf*1j, 3+4j,
                 np.nan+np.nan*1j, 5-6j], dtype=np.complex128)
c64 = c128.astype(np.complex64)
big = np.empty((40, 30), dtype=np.complex128)
s = 0x2545F4914F6CDD1D
for x in np.ndindex(40, 30):
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    re = ((s >> 11) / (1 << 53)) * 8.0 - 4.0
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    im = ((s >> 11) / (1 << 53)) * 8.0 - 4.0
    big[x] = complex(re, im)
big.flat[::53] = np.nan
big.flat[7::101] = np.inf + 1j * np.nan
big.flat[11::97] = (-np.inf) * (1 + 1j)
h = hashlib.sha256()
allmatch = True
specs = [
    (c128, {}), (c128, dict(nan=7.0, posinf=100.0, neginf=-100.0)), (c128, dict(nan=-1.5)),
    (c64, {}), (c64, dict(nan=2.0, posinf=9.0)),
    (big, dict(nan=0.5)), (big, {}),
]
for arr, kw in specs:
    r = np.asarray(fnp.nan_to_num(arr, **kw))
    e = np.asarray(np.nan_to_num(arr, **kw))
    if r.shape != e.shape or r.dtype != e.dtype or r.tobytes() != e.tobytes():
        allmatch = False
    h.update(r.tobytes())
print(allmatch)
print(h.hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "True",
        "complex nan_to_num must be byte-identical to numpy.nan_to_num"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "afec0558d929a14467c727f8aa576d4156f88a360e19e32d3725157550ba94a9",
        "complex nan_to_num golden sha256 drifted"
    );
    Ok(())
}
