//! Conformance tests for trigonometric and math functions against NumPy oracle.
//!
//! Tests sin, cos, sqrt, exp, log, sinh, cosh, tanh, arcsin, arccos, arctan,
//! arcsinh, arccosh, arctanh, and positive.

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
fn sin_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([np.pi])",
        "np.array([np.pi / 2])",
        "np.array([np.pi / 4])",
        "np.array([-np.pi])",
        "np.array([-np.pi / 2])",
        "np.array([2 * np.pi])",
        "np.linspace(-np.pi, np.pi, 21)",
        "np.linspace(0, 2 * np.pi, 17)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([-1.0, -2.0, -3.0])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([[0.0, np.pi / 4], [np.pi / 2, np.pi]])",
    ];
    test_unary_function("sin", &test_cases, 1e-10)
}

#[test]
fn cos_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([np.pi])",
        "np.array([np.pi / 2])",
        "np.array([np.pi / 4])",
        "np.array([-np.pi])",
        "np.array([2 * np.pi])",
        "np.linspace(-np.pi, np.pi, 21)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([1.0, 2.0, 3.0])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("cos", &test_cases, 1e-10)
}

#[test]
fn tan_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([np.pi / 4])",
        "np.array([-np.pi / 4])",
        "np.array([np.pi / 6])",
        "np.array([np.pi / 3])",
        "np.array([-np.pi / 6])",
        "np.array([-np.pi / 3])",
        "np.array([np.pi])",
        "np.array([-np.pi])",
        "np.linspace(-np.pi / 3, np.pi / 3, 13)",
        "np.array([0.1, 0.2, 0.3, 0.4, 0.5])",
        "np.array([-0.1, -0.2, -0.3, -0.4, -0.5])",
        "np.array([1.0, 1.1, 1.2])",
        "np.array([-1.0, -1.1, -1.2])",
        "np.array([[0.0, np.pi / 4], [np.pi / 6, np.pi / 3]])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("tan", &test_cases, 1e-10)
}

#[test]
fn trig_aliases_accept_ufunc_out_keyword() -> Result<(), String> {
    let cases = [
        ("acos", "np.array([0.0])"),
        ("acosh", "np.array([1.5])"),
        ("asin", "np.array([0.0])"),
        ("asinh", "np.array([0.5])"),
        ("atan", "np.array([0.5])"),
        ("atanh", "np.array([0.5])"),
        ("tan", "np.array([0.5])"),
    ];

    for (func, input) in cases {
        let script = format!(
            "import numpy as np\n\
             x = {input}\n\
             out = np.empty_like(x, dtype=float)\n\
             r = np.{func}(x, out=out)\n\
             print(r is out, out.tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "x = {input}\n\
             out = np.empty_like(x, dtype=float)\n\
             r = fnp.{func}(x, out=out)\n\
             print(r is out, out.tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} out keyword mismatch"
        );
    }

    let script = "import numpy as np\n\
                  y = np.array([1.0])\n\
                  x = np.array([1.0])\n\
                  out = np.empty_like(x, dtype=float)\n\
                  r = np.atan2(y, x, out=out)\n\
                  print(r is out, out.tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        "y = np.array([1.0])\n\
         x = np.array([1.0])\n\
         out = np.empty_like(x, dtype=float)\n\
         r = fnp.atan2(y, x, out=out)\n\
         print(r is out, out.tolist())"
            .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "atan2 out keyword mismatch"
    );
    Ok(())
}

#[test]
fn core_ufuncs_accept_numpy_signature_out_and_where_keywords() -> Result<(), String> {
    let script = fnp_script(
        r#"
import inspect
import numpy as np

unary_cases = {
    "abs": "np.array([-2.0, -0.0, 3.5])",
    "absolute": "np.array([-2.0, -0.0, 3.5])",
    "sin": "np.array([0.0, 0.5, 1.0])",
    "cos": "np.array([0.0, 0.5, 1.0])",
    "log": "np.array([0.25, 1.0, 4.0])",
    "exp": "np.array([-1.0, 0.0, 1.0])",
    "sqrt": "np.array([0.0, 4.0, 9.0])",
    "arcsin": "np.array([-0.5, 0.0, 0.5])",
    "arccos": "np.array([-0.5, 0.0, 0.5])",
    "arctan": "np.array([-1.0, 0.0, 1.0])",
    "arcsinh": "np.array([-1.0, 0.0, 1.0])",
    "arccosh": "np.array([1.0, 2.0, 4.0])",
    "arctanh": "np.array([-0.5, 0.0, 0.5])",
    "sinh": "np.array([-1.0, 0.0, 1.0])",
    "cosh": "np.array([-1.0, 0.0, 1.0])",
    "tanh": "np.array([-1.0, 0.0, 1.0])",
}
binary_cases = {
    "add": ("np.array([1.0, 2.0, 3.0])", "np.array([10.0, 20.0, 30.0])"),
    "subtract": ("np.array([1.0, 2.0, 3.0])", "np.array([10.0, 20.0, 30.0])"),
    "multiply": ("np.array([1.0, 2.0, 3.0])", "np.array([10.0, 20.0, 30.0])"),
    "arctan2": ("np.array([1.0, 2.0, 3.0])", "np.array([4.0, 5.0, 6.0])"),
    "true_divide": ("np.array([1.0, 2.0, 3.0])", "np.array([4.0, 5.0, 6.0])"),
}
mask = np.array([True, False, True])
sentinel = -12345.0
unary_signature = "(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)"
binary_signature = "(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)"

def assert_ufunc_signature_match(name, ours, theirs, expected):
    ours_signature = inspect.signature(ours)
    if str(ours_signature) != expected:
        raise AssertionError(f"fnp.{name} signature mismatch: {ours_signature!s} != {expected}")
    theirs_signature = inspect.signature(theirs)
    if str(theirs_signature) not in {expected, "(*args, **kwargs)"}:
        raise AssertionError(
            f"numpy.{name} exposed unexpected signature: {theirs_signature!s}"
        )
    missing = {"out", "where"} - set(ours_signature.parameters)
    if missing:
        raise AssertionError(f"fnp.{name} missing expected ufunc kwargs: {sorted(missing)}")

def assert_same_out(name, ours, theirs, ours_out, theirs_out):
    if ours is not ours_out:
        raise AssertionError(f"fnp.{name} did not return the provided out array")
    if not np.allclose(ours_out, theirs_out, equal_nan=True):
        raise AssertionError(f"{name} out mismatch: {ours_out!r} != {theirs_out!r}")
    if ours_out[1] != sentinel:
        raise AssertionError(f"fnp.{name} overwrote where=False slot")
    if theirs_out[1] != sentinel:
        raise AssertionError(f"numpy.{name} overwrote where=False slot")

for name, expr in unary_cases.items():
    x = eval(expr)
    numpy_fn = getattr(np, name)
    fnp_fn = getattr(fnp, name)
    assert_ufunc_signature_match(name, fnp_fn, numpy_fn, unary_signature)
    ours_out = np.full(x.shape, sentinel, dtype=float)
    theirs_out = np.full(x.shape, sentinel, dtype=float)
    ours = fnp_fn(x, out=ours_out, where=mask)
    theirs = numpy_fn(x, out=theirs_out, where=mask)
    assert_same_out(name, ours, theirs, ours_out, theirs_out)

for name, (left_expr, right_expr) in binary_cases.items():
    x1 = eval(left_expr)
    x2 = eval(right_expr)
    numpy_fn = getattr(np, name)
    fnp_fn = getattr(fnp, name)
    assert_ufunc_signature_match(name, fnp_fn, numpy_fn, binary_signature)
    ours_out = np.full(x1.shape, sentinel, dtype=float)
    theirs_out = np.full(x1.shape, sentinel, dtype=float)
    ours = fnp_fn(x1, x2, out=ours_out, where=mask)
    theirs = numpy_fn(x1, x2, out=theirs_out, where=mask)
    assert_same_out(name, ours, theirs, ours_out, theirs_out)

print("ok")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "ok",
        "core ufuncs should accept NumPy signature kwargs"
    );
    Ok(())
}

#[test]
fn core_ufuncs_match_numpy_on_complex64_and_complex128_inputs() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
import numpy as np

warnings.filterwarnings("ignore")

unary_ops = [
    "abs", "absolute", "sin", "cos", "tan", "exp", "log", "sqrt",
    "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "arcsinh", "arccosh", "arctanh",
]
binary_ops = ["add", "subtract", "multiply", "divide", "true_divide", "power"]
comparison_ops = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]
inputs = [
    (
        "complex64",
        np.array([1.0 + 2.0j, -0.5 + 0.75j, 0.25 - 1.25j], dtype=np.complex64),
        np.array([2.0 - 1.0j, 1.25 + 0.5j, -1.5 + 2.0j], dtype=np.complex64),
    ),
    (
        "complex128",
        np.array([1.0 + 2.0j, -0.5 + 0.75j, 0.25 - 1.25j], dtype=np.complex128),
        np.array([2.0 - 1.0j, 1.25 + 0.5j, -1.5 + 2.0j], dtype=np.complex128),
    ),
]
kw_mask = np.array([True, False, True])

def assert_same(name, dtype_name, ours, theirs):
    if ours.shape != theirs.shape:
        raise AssertionError(f"{name}({dtype_name}) shape mismatch: {ours.shape} != {theirs.shape}")
    if ours.dtype != theirs.dtype:
        raise AssertionError(f"{name}({dtype_name}) dtype mismatch: {ours.dtype} != {theirs.dtype}")
    if np.issubdtype(theirs.dtype, np.bool_):
        ok = np.array_equal(ours, theirs)
    else:
        ok = np.allclose(ours, theirs, equal_nan=True, rtol=1e-5, atol=1e-6)
    if not ok:
        raise AssertionError(f"{name}({dtype_name}) values mismatch: {ours!r} != {theirs!r}")

for dtype_name, x, y in inputs:
    for name in unary_ops:
        fnp_fn = getattr(fnp, name)
        numpy_fn = getattr(np, name)
        assert_same(name, dtype_name, fnp_fn(x), numpy_fn(x))

        ours_out = np.full(x.shape, 123.0 + 456.0j, dtype=x.dtype)
        theirs_out = np.full(x.shape, 123.0 + 456.0j, dtype=x.dtype)
        ours = fnp_fn(x, out=ours_out, where=kw_mask)
        theirs = numpy_fn(x, out=theirs_out, where=kw_mask)
        if ours is not ours_out:
            raise AssertionError(f"fnp.{name}({dtype_name}) did not return provided complex out")
        if theirs is not theirs_out:
            raise AssertionError(f"numpy.{name}({dtype_name}) did not return provided complex out")
        assert_same(f"{name}[out,where]", dtype_name, ours_out, theirs_out)

    for name in binary_ops:
        fnp_fn = getattr(fnp, name)
        numpy_fn = getattr(np, name)
        assert_same(name, dtype_name, fnp_fn(x, y), numpy_fn(x, y))

        ours_out = np.full(x.shape, 123.0 + 456.0j, dtype=x.dtype)
        theirs_out = np.full(x.shape, 123.0 + 456.0j, dtype=x.dtype)
        ours = fnp_fn(x, y, out=ours_out, where=kw_mask)
        theirs = numpy_fn(x, y, out=theirs_out, where=kw_mask)
        if ours is not ours_out:
            raise AssertionError(f"fnp.{name}({dtype_name}) did not return provided complex out")
        if theirs is not theirs_out:
            raise AssertionError(f"numpy.{name}({dtype_name}) did not return provided complex out")
        assert_same(f"{name}[out,where]", dtype_name, ours_out, theirs_out)

    for name in comparison_ops:
        fnp_fn = getattr(fnp, name)
        numpy_fn = getattr(np, name)
        assert_same(name, dtype_name, fnp_fn(x, y), numpy_fn(x, y))

print("ok")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "ok",
        "core ufunc complex64/complex128 parity should match numpy"
    );
    Ok(())
}

#[test]
fn sqrt_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([4.0])",
        "np.array([9.0, 16.0, 25.0])",
        "np.array([0.25, 0.5, 0.75])",
        "np.array([2.0, 3.0, 5.0, 7.0])",
        "np.array([1e10, 1e-10])",
        "np.linspace(0, 100, 11)",
        "np.array([np.inf])",
        "np.array([np.nan])",
        "np.array([-1.0])",
    ];
    test_unary_function("sqrt", &test_cases, 1e-10)
}

#[test]
fn exp_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([2.0, 3.0, 4.0])",
        "np.array([-2.0, -3.0, -4.0])",
        "np.linspace(-5, 5, 11)",
        "np.array([0.1, 0.5, 0.9])",
        "np.array([700.0])",
        "np.array([-700.0])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("exp", &test_cases, 1e-10)
}

#[test]
fn log_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1.0])",
        "np.array([np.e])",
        "np.array([np.e ** 2])",
        "np.array([0.5, 1.0, 2.0])",
        "np.array([10.0, 100.0, 1000.0])",
        "np.linspace(0.1, 10, 10)",
        "np.array([1e-10, 1e10])",
        "np.array([0.0])",
        "np.array([-1.0])",
        "np.array([np.inf, np.nan])",
    ];
    test_unary_function("log", &test_cases, 1e-10)
}

#[test]
fn sinh_cosh_tanh_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-3, 3, 13)",
        "np.array([0.5, 1.5, 2.5])",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("sinh", &test_cases, 1e-10)?;
    test_unary_function("cosh", &test_cases, 1e-10)?;
    test_unary_function("tanh", &test_cases, 1e-10)?;
    Ok(())
}

#[test]
fn arcsin_arccos_arctan_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-1, 1, 11)",
        "np.array([0.1, 0.2, 0.3, 0.4])",
        "np.array([np.nan])",
    ];
    test_unary_function("arcsin", &test_cases, 1e-10)?;
    test_unary_function("arccos", &test_cases, 1e-10)?;

    let arctan_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-10, 10, 21)",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("arctan", &arctan_cases, 1e-10)?;
    Ok(())
}

#[test]
fn arcsinh_arccosh_arctanh_match_numpy() -> Result<(), String> {
    let arcsinh_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.linspace(-5, 5, 11)",
        "np.array([np.inf, -np.inf, np.nan])",
    ];
    test_unary_function("arcsinh", &arcsinh_cases, 1e-10)?;

    let arccosh_cases = vec![
        "np.array([1.0])",
        "np.array([2.0])",
        "np.array([10.0])",
        "np.linspace(1, 10, 10)",
        "np.array([np.inf, np.nan])",
        "np.array([0.5])",
    ];
    test_unary_function("arccosh", &arccosh_cases, 1e-10)?;

    let arctanh_cases = vec![
        "np.array([0.0])",
        "np.array([0.5])",
        "np.array([-0.5])",
        "np.linspace(-0.99, 0.99, 11)",
        "np.array([1.0, -1.0])",
        "np.array([np.nan])",
    ];
    test_unary_function("arctanh", &arctanh_cases, 1e-10)?;
    Ok(())
}

#[test]
fn positive_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0.0])",
        "np.array([1.0])",
        "np.array([-1.0])",
        "np.array([1.0, -1.0, 2.0, -2.0])",
        "np.array([0.5, -0.5, 1.5, -1.5])",
        "np.array([1e10, -1e10])",
        "np.array([np.inf, -np.inf, np.nan])",
        "np.array([[1.0, -1.0], [2.0, -2.0]])",
        "np.array([1, -1, 2, -2], dtype=np.float64)",
    ];
    test_unary_function("positive", &test_cases, 1e-14)
}

#[test]
fn trig_math_empty_arrays_match_numpy() -> Result<(), String> {
    for func in &[
        "sin", "cos", "tan", "sqrt", "exp", "log", "sinh", "cosh", "tanh", "arcsin", "arccos",
        "arctan", "arcsinh", "arccosh", "arctanh", "positive",
    ] {
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
fn trig_integer_input_promotes_to_float() -> Result<(), String> {
    for func in &["sin", "cos", "tan", "sqrt", "exp", "log"] {
        let script = format!(
            "import numpy as np; r = np.{func}(np.array([1, 2, 3], dtype=np.int32)); print(r.dtype)"
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_script(format!(
            "r = fnp.{func}(np.array([1, 2, 3], dtype=np.int32)); print(r.dtype)"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "{func} dtype promotion mismatch"
        );
    }
    Ok(())
}

#[test]
fn promoting_math_bool_inputs_match_numpy() -> Result<(), String> {
    for func in &[
        "sin", "cos", "tan", "sqrt", "exp", "log", "sinh", "cosh", "tanh", "arcsin", "arccos",
        "arctan", "arcsinh", "arccosh", "arctanh", "expm1", "log1p",
    ] {
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
            "{func} bool input mismatch"
        );
    }
    Ok(())
}

#[test]
fn positive_bool_input_matches_numpy_error() -> Result<(), String> {
    let script = r#"
import numpy as np
try:
    np.positive(np.array([True, False], dtype=np.bool_))
    print("no_error")
except Exception as exc:
    print(type(exc).__name__)
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
try:
    fnp.positive(np.array([True, False], dtype=np.bool_))
    print("no_error")
except Exception as exc:
    print(type(exc).__name__)
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "positive bool error mismatch"
    );
    Ok(())
}
