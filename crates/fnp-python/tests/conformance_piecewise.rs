//! Conformance tests for numpy.piecewise against NumPy oracle.
//!
//! Tests piecewise function evaluation where different functions are applied
//! based on condition masks. This is used for implementing piecewise-defined
//! mathematical functions.

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

fn outcome_body(setup: &str, call_expr: &str) -> String {
    format!(
        "{setup}\n\
         def outcome(op):\n\
         {I4}try:\n\
         {I8}value = {call_expr}\n\
         {I8}arr = np.asarray(value)\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(str(arr.dtype))\n\
         {I8}print(tuple(arr.shape))\n\
         {I8}print(repr(arr.tolist()))\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         outcome(op)",
        I4 = "    ",
        I8 = "        ",
    )
}

fn numpy_outcome_script(setup: &str, call_expr: &str) -> String {
    format!(
        "import numpy as np\nop = np.piecewise\n{}",
        outcome_body(setup, call_expr)
    )
}

fn fnp_outcome_script(setup: &str, call_expr: &str) -> String {
    fnp_script(format!(
        "op = fnp.piecewise\n{}",
        outcome_body(setup, call_expr)
    ))
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
fn piecewise_python_container_error_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list x and list masks with scalar default",
            "",
            "op([0, 1, 2, 3], [[True, False, True, False], [False, True, False, True]], [10, 20, 99])",
        ),
        (
            "tuple x with callable and scalar fallback",
            "",
            "op((0.0, 1.0, 2.0), [np.array([True, False, True])], [lambda x: x + 0.5, -1.0])",
        ),
        (
            "args kwargs forwarded to callable",
            "x = np.array([1.0, 2.0, 3.0])\ncond = [np.array([True, False, True])]",
            "op(x, cond, [lambda x, scale, offset=0.0: x * scale + offset, -1.0], 10.0, offset=5.0)",
        ),
        (
            "nested Python list shape preserved",
            "",
            "op([[1, 2], [3, 4]], [[[True, False], [False, True]]], [lambda x: x * 3, 0])",
        ),
        (
            "funclist length mismatch error type",
            "",
            "op([1, 2], [[True, False], [False, True]], [10])",
        ),
        (
            "condition shape mismatch error type",
            "",
            "op([1, 2, 3], [[True, False]], [10, 0])",
        ),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script(setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "piecewise Python-container error surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn piecewise_basic_two_conditions() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-2, -1, 0, 1, 2, 3, 4, 5])
condlist = [x < 0, x >= 0]
funclist = [lambda x: -x, lambda x: x**2]
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-2, -1, 0, 1, 2, 3, 4, 5])
condlist = [x < 0, x >= 0]
funclist = [lambda x: -x, lambda x: x**2]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise basic mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_three_regions() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.linspace(-5, 5, 11)
condlist = [x < -1, (x >= -1) & (x <= 1), x > 1]
funclist = [lambda x: 0, lambda x: x, lambda x: 1]
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.linspace(-5, 5, 11)
condlist = [x < -1, (x >= -1) & (x <= 1), x > 1]
funclist = [lambda x: 0, lambda x: x, lambda x: 1]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise three regions mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_with_default_value() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Only specify condition for x < 5, rest gets default
condlist = [x < 5]
funclist = [lambda x: x * 2, 99]  # 99 is default for uncovered elements
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
condlist = [x < 5]
funclist = [lambda x: x * 2, 99]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise with default mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_scalar_functions() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-3, -2, -1, 0, 1, 2, 3])
condlist = [x < 0, x == 0, x > 0]
funclist = [-1, 0, 1]  # scalar values instead of functions
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-3, -2, -1, 0, 1, 2, 3])
condlist = [x < 0, x == 0, x > 0]
funclist = [-1, 0, 1]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise scalar functions mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_floating_point_input() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
condlist = [x < 0, x >= 0]
funclist = [np.abs, np.sqrt]
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
condlist = [x < 0, x >= 0]
funclist = [np.abs, np.sqrt]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise float input mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_overlapping_conditions() -> Result<(), String> {
    // When conditions overlap, first matching condition wins
    let script = r#"
import numpy as np
x = np.array([0, 1, 2, 3, 4, 5])
# Both conditions are True for x >= 2, first one wins
condlist = [x >= 2, x >= 4]
funclist = [10, 100, 0]  # default 0 for uncovered
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4, 5])
condlist = [x >= 2, x >= 4]
funclist = [10, 100, 0]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise overlapping conditions mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_2d_array() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
condlist = [x <= 3, x > 3]
funclist = [lambda x: x * 2, lambda x: x + 10]
result = np.piecewise(x, condlist, funclist)
print(result.flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[1, 2, 3], [4, 5, 6]])
condlist = [x <= 3, x > 3]
funclist = [lambda x: x * 2, lambda x: x + 10]
result = fnp.piecewise(x, condlist, funclist)
print(result.flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise 2D array mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_empty_array() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([], dtype=np.float64)
condlist = [x < 0, x >= 0]
funclist = [-1, 1]
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
condlist = [x < 0, x >= 0]
funclist = [-1, 1]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "piecewise empty array mismatch"
    );

    Ok(())
}

#[test]
fn piecewise_all_false_conditions() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([1, 2, 3, 4, 5])
# No condition is True, all elements get default (last funclist entry)
condlist = [x < 0, x > 10]
funclist = [-1, 99, 0]  # 0 is the default
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4, 5])
condlist = [x < 0, x > 10]
funclist = [-1, 99, 0]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise all false mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn piecewise_with_nan_and_inf() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-np.inf, -1, 0, 1, np.inf, np.nan])
condlist = [np.isfinite(x) & (x < 0), np.isfinite(x) & (x >= 0)]
funclist = [lambda x: x * 2, lambda x: x + 1, 999]  # 999 for inf/nan
result = np.piecewise(x, condlist, funclist)
print(result.tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-np.inf, -1, 0, 1, np.inf, np.nan])
condlist = [np.isfinite(x) & (x < 0), np.isfinite(x) & (x >= 0)]
funclist = [lambda x: x * 2, lambda x: x + 1, 999]
result = fnp.piecewise(x, condlist, funclist)
print(result.tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "piecewise nan/inf mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}
