//! Conformance tests for numpy.diff, numpy.gradient, numpy.ediff1d against NumPy oracle.
//!
//! Tests array difference operations:
//! - diff(a, n=1, axis=-1): n-th discrete difference along axis
//! - gradient(f, *varargs): N-D gradient with uniform/non-uniform spacing
//! - ediff1d(ary, to_end=None, to_begin=None): consecutive element differences

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
fn diff_basic_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 3, 4, 5])",
        "np.array([1.0, 2.0, 4.0, 7.0, 11.0])",
        "np.array([0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0])",
        "np.array([1, 1, 1, 1, 1])",
        "np.array([5, 4, 3, 2, 1])",
        "np.array([-3, -1, 0, 2, 5])",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([1e10, 2e10, 3e10])",
        "np.array([1e-10, 2e-10, 3e-10])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.diff({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.diff({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "diff mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn diff_n_parameter_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 4, 7, 11, 16])", 1),
        ("np.array([1, 2, 4, 7, 11, 16])", 2),
        ("np.array([1, 2, 4, 7, 11, 16])", 3),
        ("np.array([1.0, 4.0, 9.0, 16.0, 25.0])", 2),
        ("np.array([0, 1, 8, 27, 64, 125])", 3),
    ];

    for (arr_expr, n) in &test_cases {
        let script = format!("import numpy as np; print(np.diff({arr_expr}, n={n}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.diff({arr_expr}, n={n}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "diff n={n} mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn diff_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([[1, 2, 3], [4, 5, 6]])", 0),
        ("np.array([[1, 2, 3], [4, 5, 6]])", 1),
        ("np.array([[1, 2, 3], [4, 5, 6]])", -1),
        ("np.array([[1, 4, 9], [16, 25, 36]])", 0),
        ("np.array([[1, 4, 9], [16, 25, 36]])", 1),
    ];

    for (arr_expr, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.diff({arr_expr}, axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.diff({arr_expr}, axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "diff axis={axis} mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn diff_prepend_append_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1, 2, 4, 7, 11])
print(np.diff(x, prepend=0).tolist())
print(np.diff(x, append=16).tolist())
print(np.diff(x, prepend=0, append=16).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1, 2, 4, 7, 11])
print(fnp.diff(x, prepend=0).tolist())
print(fnp.diff(x, append=16).tolist())
print(fnp.diff(x, prepend=0, append=16).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "diff prepend/append mismatch"
    );

    Ok(())
}

#[test]
fn gradient_basic_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 4, 7, 11])",
        "np.array([1.0, 4.0, 9.0, 16.0, 25.0])",
        "np.array([0, 1, 8, 27, 64, 125])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([5, 4, 3, 2, 1])",
        "np.array([0.0, 0.5, 1.0, 1.5, 2.0])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.gradient({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.gradient({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "gradient mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn gradient_custom_spacing_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1, 4, 9, 16, 25])
print(np.gradient(x, 2).tolist())
print(np.gradient(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1, 4, 9, 16, 25])
print(fnp.gradient(x, 2).tolist())
print(fnp.gradient(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "gradient custom spacing mismatch"
    );

    Ok(())
}

#[test]
fn ediff1d_basic_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 4, 7, 11])",
        "np.array([1.0, 2.0, 3.0, 4.0, 5.0])",
        "np.array([5, 4, 3, 2, 1])",
        "np.array([[1, 2, 3], [4, 5, 6]])",
        "np.array([0.1, 0.2, 0.3])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.ediff1d({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.ediff1d({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "ediff1d mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn ediff1d_to_begin_to_end_matches_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1, 2, 4, 7, 11])
print(np.ediff1d(x, to_begin=-1).tolist())
print(np.ediff1d(x, to_end=5).tolist())
print(np.ediff1d(x, to_begin=0, to_end=10).tolist())
print(np.ediff1d(x, to_begin=[0, -1]).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1, 2, 4, 7, 11])
print(fnp.ediff1d(x, to_begin=-1).tolist())
print(fnp.ediff1d(x, to_end=5).tolist())
print(fnp.ediff1d(x, to_begin=0, to_end=10).tolist())
print(fnp.ediff1d(x, to_begin=[0, -1]).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "ediff1d to_begin/to_end mismatch"
    );

    Ok(())
}

#[test]
fn diff_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.diff(np.array([], dtype=np.float64)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.diff(np.array([], dtype=np.float64)).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "diff empty array mismatch"
    );

    Ok(())
}

#[test]
fn gradient_empty_array_raises_valueerror() -> Result<(), String> {
    let numpy_script = "import numpy as np\ntry:\n    np.gradient(np.array([], dtype=np.float64))\n    print('no error')\nexcept ValueError:\n    print('ValueError')";
    let numpy_result = numpy_oracle(numpy_script)?;

    let rust_script = fnp_script(
        "try:\n    fnp.gradient(np.array([], dtype=np.float64))\n    print('no error')\nexcept ValueError:\n    print('ValueError')"
            .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "gradient empty array error behavior mismatch"
    );
    assert_eq!(
        numpy_result.trim(),
        "ValueError",
        "numpy should raise ValueError"
    );

    Ok(())
}

#[test]
fn ediff1d_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.ediff1d(np.array([], dtype=np.float64)).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.ediff1d(np.array([], dtype=np.float64)).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "ediff1d empty array mismatch"
    );

    Ok(())
}

#[test]
fn diff_single_element_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.diff(np.array([5])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.diff(np.array([5])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "diff single element mismatch"
    );

    Ok(())
}

#[test]
fn gradient_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for gradient (central difference)
    let script = fnp_script(
        r#"
# Signed-zero gradient semantics
# gradient computes differences divided by spacing
tests = [
    ([0.0, 0.0, 0.0], False),       # gradient: [0, 0, 0] (forward/backward/central)
    ([-0.0, -0.0, -0.0], False),    # gradient: [0, 0, 0]
    ([1.0, -0.0, 1.0], False),      # gradient with -0 in middle
]
all_pass = True
for values, check_signs in tests:
    arr = np.array(values)
    fnp_result = fnp.gradient(arr)
    np_result = np.gradient(arr)
    if not np.allclose(fnp_result, np_result):
        print(f"FAIL: gradient({values}) value mismatch")
        all_pass = False
    fnp_signs = np.signbit(fnp_result).tolist()
    np_signs = np.signbit(np_result).tolist()
    if fnp_signs != np_signs:
        print(f"FAIL: gradient({values}) fnp signbit={fnp_signs} np signbit={np_signs}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn diff_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 4.0])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
# NaN should propagate through differences
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn diff_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.inf, 2.0, -np.inf])
fnp_result = fnp.diff(a)
np_result = np.diff(a)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff inf handling should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_nan_propagation() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
fnp_result = fnp.gradient(a)
np_result = np.gradient(a)
# NaN should propagate through gradient calculation
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient nan propagation should match numpy"
    );
    Ok(())
}

#[test]
fn diff_higher_order_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 4, 9, 16, 25, 36])  # squares
# First difference: [3, 5, 7, 9, 11]
# Second difference: [2, 2, 2, 2]
fnp_d1 = fnp.diff(a, n=1)
fnp_d2 = fnp.diff(a, n=2)
np_d1 = np.diff(a, n=1)
np_d2 = np.diff(a, n=2)
print(np.array_equal(fnp_d1, np_d1) and np.array_equal(fnp_d2, np_d2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "diff higher order should match numpy"
    );
    Ok(())
}

#[test]
fn gradient_non_uniform_spacing() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, 1.0, 3.0, 6.0])  # non-uniform spacing
y = np.array([0.0, 1.0, 4.0, 9.0])  # y = (x/3)**2 approximately
fnp_result = fnp.gradient(y, x)
np_result = np.gradient(y, x)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "gradient non-uniform spacing should match numpy"
    );
    Ok(())
}
