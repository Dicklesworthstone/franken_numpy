//! Conformance tests for numpy.prod against NumPy oracle.
//!
//! Tests the native Rust prod implementation against NumPy across various
//! input shapes, axis parameters, keepdims, and data types.

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

fn fnp_prod_script(body: String) -> String {
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

fn parse_float(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(f64::NAN)
}

fn parse_float_list(s: &str) -> Vec<f64> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn floats_close(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    let abs_diff = (a - b).abs();
    let rel_tol = tol * a.abs().max(b.abs()).max(1.0);
    abs_diff < rel_tol
}

fn arrays_close(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| floats_close(*x, *y, tol))
}

#[test]
fn prod_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic arrays
        "[1, 2, 3]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[2, 2, 2]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2]",
        "[2, 3, 4, 5]",
        // Floating point
        "[0.5, 1.5, 2.5]",
        "[1.1, 2.2, 3.3]",
        "[0.1, 0.2, 0.5]",
        "[1.5, 2.0, 2.5]",
        "[0.9, 0.8, 0.7]",
        // Negatives
        "[-1, 2, -3]",
        "[-0.5, -0.5, 4]",
        "[1, -1, 1, -1]",
        "[-2, -3, -4]",
        "[2, -2, 2, -2]",
        // Larger arrays
        "[1, 2, 1, 2, 1, 2]",
        "[2, 2, 2, 2, 2]",
        "[1, 1, 1, 1, 1, 1]",
        "[1.5, 1.5, 1.5]",
        "[0.5, 0.5, 0.5]",
        // Mixed
        "[0.5, 2, 0.5, 2]",
        "[-1, 0.5, -2, 0.5]",
        "[1, 2, 0.5, 4]",
        "[3, 0.333333, 3]",
        "[2, 0.5, 2, 0.5]",
        // Edge values
        "[1.0, 1.0]",
        "[2.0, 2.0, 2.0]",
        "[-1, -1, -1, -1]",
        "[0.1, 10]",
        "[10, 0.1]",
        // More variety
        "[2, 3, 5, 7]",
        "[1, 2, 3, 2, 1]",
        "[1.1, 1.2, 1.3]",
        "[0.9, 0.9, 0.9]",
        "[2, 1, 2, 1, 2]",
        // Small values
        "[1.0, 1.01, 1.02]",
        "[0.99, 0.99, 0.99]",
        "[1.001, 1.001, 1.001]",
        "[0.999, 0.999, 0.999]",
        "[1.1, 0.9, 1.1]",
        // Additional cases
        "[3, 3, 3]",
        "[1, 3, 1, 3]",
        "[2, 4, 8]",
        "[0.25, 4, 0.25]",
        "[1, 2, 4, 8]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.prod(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_prod_script(format!("print(fnp.prod(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "prod flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn prod_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "0"),
        ("[[2, 2, 2], [2, 2, 2]]", "0"),
        ("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 2, 3], [4, 5, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "1"),
        ("[[2, 2, 2], [3, 3, 3]]", "1"),
        ("[[1, 2, 4], [1, 3, 9]]", "1"),
        // Negative axis
        ("[[1, 2, 3], [4, 5, 6]]", "-1"),
        ("[[1, 2, 3], [4, 5, 6]]", "-2"),
        ("[[2, 2], [2, 2]]", "-1"),
        ("[[2, 2], [2, 2]]", "-2"),
        // Single row/column
        ("[[1, 2, 3, 4]]", "0"),
        ("[[1, 2, 3, 4]]", "1"),
        ("[[2], [2], [2], [2]]", "0"),
        ("[[2], [2], [2], [2]]", "1"),
        // Floating point 2D
        ("[[0.5, 1.5], [2.0, 3.0]]", "0"),
        ("[[0.5, 1.5], [2.0, 3.0]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.prod(np.array({arr_str}), axis={axis}).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_prod_script(format!(
            "print(fnp.prod(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "prod axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn prod_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "0"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-3"),
        // Different shapes
        ("[[[1, 2, 3]], [[4, 5, 6]]]", "0"),
        ("[[[1, 2, 3]], [[4, 5, 6]]]", "1"),
        ("[[[1, 2, 3]], [[4, 5, 6]]]", "2"),
        ("[[[1], [2], [3]], [[4], [5], [6]]]", "0"),
        ("[[[1], [2], [3]], [[4], [5], [6]]]", "1"),
        ("[[[1], [2], [3]], [[4], [5], [6]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.prod(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_prod_script(format!(
            "print(fnp.prod(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "prod 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn prod_keepdims_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 1D with keepdims
        ("[1, 2, 3, 4, 5]", "None", true),
        // 2D with keepdims axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0", true),
        ("[[1, 2, 3], [4, 5, 6]]", "1", true),
        // 3D with keepdims
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "0", true),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "1", true),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "2", true),
        // Compare keepdims=False (default)
        ("[[1, 2, 3], [4, 5, 6]]", "0", false),
        ("[[1, 2, 3], [4, 5, 6]]", "1", false),
    ];

    for (arr_str, axis, keepdims) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!(
            "import numpy as np; print(np.prod(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_prod_script(format!(
            "print(fnp.prod(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "prod keepdims={keepdims} shape mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn prod_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 2, 3], dtype=np.int64)", "None"),
        ("np.array([1, 2, 3], dtype=np.uint8)", "None"),
        ("np.array([2, 2, 2], dtype=np.int16)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int32)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.int64)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.float32)", "None"),
        ("np.array([[1, 2], [3, 4]], dtype=np.float64)", "None"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(float(np.prod({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_prod_script(format!("print(float(fnp.prod({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-6),
            "prod dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn prod_nan_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1.0, np.nan, 3.0]",
        "[np.nan, 2.0, 3.0]",
        "[1.0, 2.0, np.nan]",
        "[np.nan, np.nan, np.nan]",
        "[1.0, np.nan, np.nan, 4.0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.prod(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_prod_script(format!("print(fnp.prod(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "prod NaN mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn prod_empty_array_matches_numpy() -> Result<(), String> {
    let test_cases = vec![("[]", "None"), ("[[]]", "None")];

    for (arr_str, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script =
            format!("import numpy as np; print(float(np.prod(np.array({arr_str}){axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_prod_script(format!(
            "print(float(fnp.prod(np.array({arr_str}){axis_arg})))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "prod empty array mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn prod_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_prod_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.prod(x)
np_result = np.prod(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "prod scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn prod_complex() -> Result<(), String> {
    let script = fnp_prod_script(
        r#"
z = np.array([1+1j, 2+0j, 0+1j], dtype=np.complex128)
fnp_result = fnp.prod(z)
np_result = np.prod(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "prod complex should match numpy");
    Ok(())
}

#[test]
fn prod_inf_handling_matches_numpy() -> Result<(), String> {
    let inf_cases = [
        "[1.0, np.inf, 3.0]",
        "[np.inf, 2.0, 3.0]",
        "[0.0, np.inf]",
        "[np.inf, np.inf]",
        "[-np.inf, np.inf]",
    ];

    for arr_str in &inf_cases {
        let np_script = format!("import numpy as np; print(repr(np.prod(np.array({arr_str}))))");
        let np_output = numpy_oracle(&np_script)?;

        let fnp_script = fnp_prod_script(format!("print(repr(fnp.prod(np.array({arr_str}))))"));
        let fnp_output = numpy_oracle(&fnp_script)?;

        assert_eq!(
            fnp_output.trim(),
            np_output.trim(),
            "prod inf mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn prod_with_out_parameter() -> Result<(), String> {
    let script = fnp_prod_script(
        r#"
a = np.array([[1, 2], [3, 4]])
out = np.empty((2,), dtype=np.int64)
fnp_result = fnp.prod(a, axis=0, out=out)
np_out = np.empty((2,), dtype=np.int64)
np_result = np.prod(a, axis=0, out=np_out)
print(np.array_equal(fnp_result, np_result) and np.array_equal(out, np_out))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "prod with out parameter should match numpy"
    );
    Ok(())
}

#[test]
fn prod_with_where_parameter() -> Result<(), String> {
    let script = fnp_prod_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp_result = fnp.prod(a, where=mask)
np_result = np.prod(a, where=mask)
print(fnp_result == np_result == 15)  # 1 * 3 * 5
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "prod with where parameter should match numpy"
    );
    Ok(())
}

#[test]
fn prod_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for parallel operation safety proofs.
    // Product of signed zeros follows XOR sign rule.
    let script = fnp_prod_script(
        r#"
# Signed-zero product semantics
# 0.0 * 0.0 = 0.0, -0.0 * 0.0 = -0.0, 0.0 * -0.0 = -0.0, -0.0 * -0.0 = 0.0
tests = [
    ([0.0, 1.0], False),      # 0.0 * 1.0 = 0.0 (positive)
    ([-0.0, 1.0], True),      # -0.0 * 1.0 = -0.0 (negative)
    ([0.0, -0.0], True),      # 0.0 * -0.0 = -0.0 (negative)
    ([-0.0, -0.0], False),    # -0.0 * -0.0 = 0.0 (positive)
    ([1.0, -0.0, 1.0], True), # Product with -0.0 in middle
]
all_pass = True
for values, expected_signbit in tests:
    arr = np.array(values)
    fnp_result = fnp.prod(arr)
    np_result = np.prod(arr)
    if np.signbit(fnp_result) != np.signbit(np_result):
        print(f"FAIL: prod({values}) fnp signbit={np.signbit(fnp_result)} np signbit={np.signbit(np_result)}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "prod signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn prod_overflow_underflow_parity() -> Result<(), String> {
    // Test overflow/underflow edge cases
    let script = fnp_prod_script(
        r#"
import warnings
warnings.filterwarnings('ignore')

# Overflow case
big = np.array([1e200, 1e200, 1e200])
fnp_big = fnp.prod(big)
np_big = np.prod(big)

# Underflow case
small = np.array([1e-200, 1e-200, 1e-200])
fnp_small = fnp.prod(small)
np_small = np.prod(small)

big_match = (np.isinf(fnp_big) == np.isinf(np_big))
small_match = ((fnp_small == 0.0) == (np_small == 0.0))
print(big_match and small_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "prod overflow/underflow should match numpy: {result}"
    );
    Ok(())
}

/// Locks the zero-copy sequential product reduction (`try_zerocopy_f64_prod`) to
/// bit-exact parity with numpy. numpy.prod multiplies sequentially, so a
/// left-to-right product matches at the IEEE-754 bit level. Compares the sha256
/// of raw output bytes across every axis of 2-D and 3-D inputs, the full
/// reduction, and signed-zero/inf/nan extremes.
#[test]
fn prod_reduction_zerocopy_f64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for shp in [(100, 50), (5, 5, 5)]:
    x = rng.standard_normal(shp) * 1.0005
    for axis in range(len(shp)):
        chunks.append(np.asarray(mod.prod(x, axis=axis)).tobytes())
    chunks.append(np.asarray(mod.prod(x)).tobytes())
xe = np.array([[-0.0, 2.0, np.inf], [3.0, -0.0, np.nan]], dtype=np.float64)
chunks.append(np.asarray(mod.prod(xe, axis=0)).tobytes())
chunks.append(np.asarray(mod.prod(xe, axis=1)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_prod_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!("import numpy as np\n{}", body.replace("MODULE", "np")))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy prod reduction must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}
