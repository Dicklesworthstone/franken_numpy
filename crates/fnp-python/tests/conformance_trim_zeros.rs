//! Conformance tests for numpy.trim_zeros against NumPy oracle.
//!
//! Tests the native Rust trim_zeros implementation against NumPy across various
//! input arrays and trim modes.

use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| format!("python3 stdin should be available\nScript: {script}"))?;
    stdin.write_all(script.as_bytes()).map_err(|error| {
        format!("failed to write Python oracle script: {error}\nScript: {script}")
    })?;
    drop(stdin);

    let output = child.wait_with_output().map_err(|error| {
        format!("failed to read Python oracle output: {error}\nScript: {script}")
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("NumPy oracle failed: {stderr}\nScript: {script}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

mod support;

fn fnp_trim_zeros_script(body: String) -> String {
    support::fnp_script(body)
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
            t.parse().ok()
        })
        .collect()
}

fn trim_zeros_outcome_body(function_expr: &str, input_expr: &str, trim_arg: &str) -> String {
    format!(
        "def outcome(fn):\n\
         {I4}try:\n\
         {I8}value = fn({input_expr}{trim_arg})\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(repr(value))\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         {I8}print(str(exc))\n\
         outcome({function_expr})",
        I4 = "    ",
        I8 = "        ",
    )
}

fn numpy_trim_zeros_outcome_script(input_expr: &str, trim_arg: &str) -> String {
    format!(
        "import numpy as np\n{}",
        trim_zeros_outcome_body("np.trim_zeros", input_expr, trim_arg)
    )
}

fn fnp_trim_zeros_outcome_script(input_expr: &str, trim_arg: &str) -> String {
    fnp_trim_zeros_script(trim_zeros_outcome_body(
        "fnp.trim_zeros",
        input_expr,
        trim_arg,
    ))
}

#[test]
fn trim_zeros_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list default", "[0, 0, 1, 2, 0, 0]", ""),
        ("tuple default", "(0, 0, 1, 2, 0, 0)", ""),
        ("tuple front", "(0, 0, 1, 2, 0, 0)", ", 'f'"),
        ("tuple back", "(0, 0, 1, 2, 0, 0)", ", 'b'"),
        ("list uppercase trim", "[0, 0, 1, 2, 0, 0]", ", 'FB'"),
        (
            "array uppercase trim native path",
            "np.array([0, 0, 1, 2, 0, 0])",
            ", 'FB'",
        ),
        (
            "array reversed uppercase trim native path",
            "np.array([0, 0, 1, 2, 0, 0])",
            ", 'BF'",
        ),
        ("scalar input error", "0", ""),
        ("tuple invalid trim error", "(0, 1, 0)", ", 'ff'"),
        ("list invalid trim error", "[0, 1, 0]", ", 'x'"),
        ("array invalid trim error", "np.array([0, 1, 0])", ", 'x'"),
    ];

    for (label, input_expr, trim_arg) in cases {
        let numpy_script = numpy_trim_zeros_outcome_script(input_expr, trim_arg);
        let numpy_result = numpy_oracle(&numpy_script)?;

        let rust_script = fnp_trim_zeros_outcome_script(input_expr, trim_arg);
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result, rust_result,
            "trim_zeros Python-container surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_default_mode_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic cases
        "np.array([0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 1, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5])",
        "np.array([0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1])",
        "np.array([1])",
        "np.array([0])",
        // Floating point
        "np.array([0.0, 0.0, 1.5, 2.5, 0.0])",
        "np.array([0.0, 1.0, 0.0, 0.0])",
        "np.array([1.1, 2.2, 3.3])",
        "np.array([0.0, 0.0, 0.0])",
        "np.array([0.0, 0.5, 0.0])",
        "np.array([0.001, 0.0, 0.0])",
        "np.array([0.0, 0.0, 0.999])",
        // Negative values
        "np.array([0, 0, -1, -2, 0, 0])",
        "np.array([-1, 0, 0, 0])",
        "np.array([0, 0, 0, -1])",
        "np.array([-1, -2, -3])",
        "np.array([0, -5, 0, -10, 0])",
        // Mixed signs
        "np.array([0, 1, -1, 2, -2, 0])",
        "np.array([0, 0, 1, -1, 0, 0])",
        "np.array([-1, 0, 1, 0, -1])",
        "np.array([0, 0, -0.5, 0.5, 0, 0])",
        // Single element arrays
        "np.array([5])",
        "np.array([-5])",
        "np.array([0.5])",
        "np.array([-0.5])",
        // Larger arrays
        "np.array([0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])",
        "np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
        // Different dtypes
        "np.array([0, 0, 1, 0], dtype=np.int32)",
        "np.array([0, 0, 1, 0], dtype=np.int64)",
        "np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)",
        "np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)",
        "np.array([0, 1, 0], dtype=np.uint8)",
        "np.array([0, 1, 0], dtype=np.int16)",
        // Edge patterns
        "np.array([0, 1, 0, 1, 0])",
        "np.array([1, 0, 1, 0, 1])",
        "np.array([0, 0, 1, 1, 0, 0])",
        "np.array([1, 1, 0, 0, 1, 1])",
        "np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])",
        // More variety
        "np.array([0, 0, 100, 200, 0, 0])",
        "np.array([0, 0, -100, -200, 0, 0])",
        "np.array([0.0, 0.0, 1e-10, 0.0])",
        "np.array([0.0, 1e10, 0.0, 0.0])",
        "np.array([0, 0, 42, 0, 0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trim_zeros({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_trim_zeros_script(format!("print(fnp.trim_zeros({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "trim_zeros default mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_front_mode_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 1, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5])",
        "np.array([0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1])",
        "np.array([0.0, 0.0, 1.5, 2.5, 0.0])",
        "np.array([0, 0, -1, -2, 0, 0])",
        "np.array([-1, 0, 0, 0])",
        "np.array([0, 0, 0, -1])",
        "np.array([0, 0, 0, 0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trim_zeros({arr_expr}, 'f').tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_trim_zeros_script(format!("print(fnp.trim_zeros({arr_expr}, 'f').tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "trim_zeros front mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_back_mode_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 0, 1, 2, 3, 0, 0])",
        "np.array([0, 0, 0, 1, 0, 0, 0])",
        "np.array([1, 2, 3, 4, 5])",
        "np.array([0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0])",
        "np.array([0, 0, 0, 0, 1])",
        "np.array([0.0, 0.0, 1.5, 2.5, 0.0])",
        "np.array([0, 0, -1, -2, 0, 0])",
        "np.array([-1, 0, 0, 0])",
        "np.array([0, 0, 0, -1])",
        "np.array([0, 0, 1, 2, 3, 0, 0, 0, 0, 0])",
        "np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.trim_zeros({arr_expr}, 'b').tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script =
            fnp_trim_zeros_script(format!("print(fnp.trim_zeros({arr_expr}, 'b').tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "trim_zeros back mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn trim_zeros_empty_result_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 0, 0])",
        "np.array([0])",
        "np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])",
        "np.array([0.0, 0.0, 0.0])",
    ];

    for arr_expr in &test_cases {
        for trim in &["fb", "f", "b"] {
            let script =
                format!("import numpy as np; print(np.trim_zeros({arr_expr}, '{trim}').tolist())");
            let numpy_result = numpy_oracle(&script)?;
            let numpy_vals = parse_float_list(&numpy_result);

            let rust_script = fnp_trim_zeros_script(format!(
                "print(fnp.trim_zeros({arr_expr}, '{trim}').tolist())"
            ));
            let rust_result = numpy_oracle(&rust_script)?;
            let rust_vals = parse_float_list(&rust_result);

            assert_eq!(
                numpy_vals, rust_vals,
                "trim_zeros empty result mismatch for {arr_expr} trim='{trim}'\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
            );
        }
    }

    Ok(())
}

#[test]
fn trim_zeros_no_zeros_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1, 2, 3])",
        "np.array([1])",
        "np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
        "np.array([-1, -2, -3])",
        "np.array([0.5, 1.5, 2.5])",
    ];

    for arr_expr in &test_cases {
        for trim in &["fb", "f", "b"] {
            let script =
                format!("import numpy as np; print(np.trim_zeros({arr_expr}, '{trim}').tolist())");
            let numpy_result = numpy_oracle(&script)?;
            let numpy_vals = parse_float_list(&numpy_result);

            let rust_script = fnp_trim_zeros_script(format!(
                "print(fnp.trim_zeros({arr_expr}, '{trim}').tolist())"
            ));
            let rust_result = numpy_oracle(&rust_script)?;
            let rust_vals = parse_float_list(&rust_result);

            assert_eq!(
                numpy_vals, rust_vals,
                "trim_zeros no zeros mismatch for {arr_expr} trim='{trim}'\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
            );
        }
    }

    Ok(())
}

// The remaining MISSING-keyword class from the fnp-python signature audit: nine
// parameters numpy declares that fnp did not, so the documented call raised
// TypeError here and answered there. Two of them change the RESULT, not just the
// accepted spelling - indices(sparse=True) returns a TUPLE of broadcastable
// arrays instead of one stacked grid, and trim_zeros(axis=) accepts 2-D input
// the 1-D path cannot express - so neither can be accepted and quietly ignored.
// Each case is computed for fnp and numpy in the same interpreter, and the
// omitted-keyword spelling is asserted alongside so the default path is proven
// unchanged.
#[test]
fn audit_missing_keywords_match_numpy() -> Result<(), String> {
    let script = fnp_trim_zeros_script(
        r#"
import platform
import io

def described(value):
    if isinstance(value, tuple):
        return ("tuple",) + tuple(described(item) for item in value)
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), array.tolist())

def trim_zeros_axis0(module):
    return module.trim_zeros(np.array([[0, 1], [0, 0]]), "fb", 0)

def trim_zeros_axis_keyword(module):
    return module.trim_zeros(np.array([[0, 0, 0], [0, 1, 0]]), axis=1)

def trim_zeros_axis_none(module):
    return module.trim_zeros(np.array([0, 1, 2, 0]), axis=None)

def trim_zeros_no_axis(module):
    return module.trim_zeros(np.array([0, 0, 1, 2, 0]))

def trim_zeros_front_only(module):
    return module.trim_zeros(np.array([0, 0, 1, 2, 0]), "f")

def indices_sparse(module):
    return module.indices((2, 3), sparse=True)

def indices_sparse_dtype(module):
    return module.indices((2, 3), np.int32, True)

def indices_dense(module):
    return module.indices((2, 3))

# cov's native Gram path is allclose to numpy but not bit-identical - an
# FMA/BLAS-dependent last-ULP difference established by
# deadlock-audit-keqog, which is why the cov goldens elsewhere use a relative
# bound rather than byte pins. So the cov cases here report dtype and shape
# exactly (dtype IS what proves dtype= took effect: float32 vs float64) and
# round the values, rather than asserting bits this bead does not own.
def cov_described(result):
    return f"{result.dtype}|{result.shape}|{np.round(result, 9).tolist()}"

def cov_dtype(module):
    return cov_described(module.cov(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]]), dtype=np.float32))

def cov_no_dtype(module):
    return cov_described(module.cov(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]])))

def identity_like(module):
    return module.identity(3, like=np.array([1.0]))

def identity_plain(module):
    return module.identity(3)

def tri_like(module):
    return module.tri(3, 2, like=np.array([1.0]))

def ascontiguousarray_like(module):
    return module.ascontiguousarray([[1.0, 2.0], [3.0, 4.0]], like=np.array([1.0]))

def asfortranarray_like(module):
    return module.asfortranarray([[1.0, 2.0], [3.0, 4.0]], like=np.array([1.0]))

def chkfinite_order_f(module):
    result = module.asarray_chkfinite([[1.0, 2.0], [3.0, 4.0]], order="F")
    return (bool(result.flags.f_contiguous), described(result))

def chkfinite_order_c(module):
    result = module.asarray_chkfinite([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="C")
    return (bool(result.flags.c_contiguous), described(result))

def chkfinite_no_order(module):
    return described(module.asarray_chkfinite([[1.0, 2.0], [3.0, 4.0]]))

def loadtxt_quotechar(module):
    return module.loadtxt(io.StringIO('"a,b",2\n'), delimiter=",", quotechar='"', dtype=object)

def loadtxt_no_quotechar(module):
    return module.loadtxt(io.StringIO("1,2\n3,4\n"), delimiter=",")

cases = [
    ("trim_zeros positional axis=0", trim_zeros_axis0),
    ("trim_zeros axis= keyword", trim_zeros_axis_keyword),
    ("trim_zeros axis=None", trim_zeros_axis_none),
    ("trim_zeros without axis", trim_zeros_no_axis),
    ("trim_zeros trim='f'", trim_zeros_front_only),
    ("indices sparse=True", indices_sparse),
    ("indices positional sparse", indices_sparse_dtype),
    ("indices dense", indices_dense),
    ("cov dtype=float32", cov_dtype),
    ("cov without dtype", cov_no_dtype),
    ("identity like=", identity_like),
    ("identity without like", identity_plain),
    ("tri like=", tri_like),
    ("ascontiguousarray like=", ascontiguousarray_like),
    ("asfortranarray like=", asfortranarray_like),
    ("asarray_chkfinite order='F'", chkfinite_order_f),
    ("asarray_chkfinite order='C' + dtype", chkfinite_order_c),
    ("asarray_chkfinite without order", chkfinite_no_order),
    ("loadtxt quotechar=", loadtxt_quotechar),
    ("loadtxt without quotechar", loadtxt_no_quotechar),
]

def outcome(module, call):
    try:
        result = call(module)
        if isinstance(result, tuple) and result and isinstance(result[0], bool):
            return ("ok",) + result
        return ("ok", described(result))
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for label, call in cases:
    actual = outcome(fnp, call)
    expected = outcome(np, call)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False
print(ok)
print("oracle", platform.node(), np.__version__)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.trim().lines().rev();
    let provenance = lines.next().unwrap_or("").trim();
    let verdict = lines.next().unwrap_or("").trim();
    assert_eq!(
        verdict, "True",
        "audit missing keywords should match numpy ({provenance}): {result}"
    );
    Ok(())
}
