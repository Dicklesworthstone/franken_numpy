//! Conformance tests for numpy.argwhere against NumPy oracle.
//!
//! Tests the native Rust argwhere implementation against NumPy across various
//! input shapes, data types, and edge cases.

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

fn fnp_argwhere_script(body: String) -> String {
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

fn parse_nested_int_list(s: &str) -> Vec<Vec<i64>> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let s = s.trim();
    if !s.starts_with('[') {
        return vec![];
    }
    let inner = &s[1..s.len() - 1];
    if inner.is_empty() {
        return vec![];
    }
    let mut result = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, c) in inner.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let sub = &inner[start..=i];
                    let nums: Vec<i64> = sub
                        .trim_start_matches('[')
                        .trim_end_matches(']')
                        .split(|c: char| c.is_whitespace() || c == ',')
                        .filter(|t| !t.is_empty())
                        .filter_map(|t| t.parse().ok())
                        .collect();
                    result.push(nums);
                    start = i + 1;
                }
            }
            _ => {}
        }
    }
    result
}

fn argwhere_outcome_body(function_expr: &str, input_expr: &str) -> String {
    format!(
        "def outcome(fn):\n\
         {I4}try:\n\
         {I8}value = fn({input_expr})\n\
         {I8}arr = np.asarray(value)\n\
         {I8}print('ok')\n\
         {I8}print(type(value).__name__)\n\
         {I8}print(str(arr.dtype))\n\
         {I8}print(tuple(arr.shape))\n\
         {I8}print(arr.tolist())\n\
         {I4}except Exception as exc:\n\
         {I8}print('err')\n\
         {I8}print(type(exc).__name__)\n\
         {I8}print(str(exc))\n\
         outcome({function_expr})",
        I4 = "    ",
        I8 = "        ",
    )
}

fn numpy_argwhere_outcome_script(input_expr: &str) -> String {
    format!(
        "import numpy as np\n{}",
        argwhere_outcome_body("np.argwhere", input_expr)
    )
}

fn fnp_argwhere_outcome_script(input_expr: &str) -> String {
    fnp_argwhere_script(argwhere_outcome_body("fnp.argwhere", input_expr))
}

#[test]
fn argwhere_python_container_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list fallback", "[0, 2, 0, 3]"),
        ("tuple fallback", "(0, 0, 5, 0)"),
        ("bool list", "[False, True, False, True]"),
        ("nested list", "[[0, 1], [2, 0]]"),
        ("scalar nonzero", "7"),
        ("scalar zero", "0"),
        ("zero-d ndarray nonzero", "np.array(7)"),
        ("zero-d ndarray zero", "np.array(0)"),
        (
            "bool ndarray fast path",
            "np.array([False, True, False, True], dtype=np.bool_)",
        ),
        (
            "uint16 ndarray fast path",
            "np.array([0, 4, 0, 5], dtype=np.uint16)",
        ),
        (
            "signed-zero nan float path",
            "np.array([-0.0, np.nan, 2.5, 0.0])",
        ),
        (
            "empty two-dimensional ndarray",
            "np.zeros((0, 2), dtype=np.int64)",
        ),
        (
            "object truthiness",
            "np.array(['', 'x', '0'], dtype=object)",
        ),
        ("ragged list error", "[[1], [0, 2]]"),
    ];

    for (label, input_expr) in cases {
        let numpy_script = numpy_argwhere_outcome_script(input_expr);
        let numpy_result = numpy_oracle(&numpy_script)?;

        let rust_script = fnp_argwhere_outcome_script(input_expr);
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result, rust_result,
            "argwhere Python-container surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn argwhere_1d_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // All nonzero
        "[1, 2, 3]",
        "[1, 1, 1, 1, 1]",
        "[100, 200, 300]",
        "[-1, -2, -3]",
        "[1]",
        // All zero
        "[0, 0, 0]",
        "[0, 0, 0, 0, 0]",
        "[0]",
        "[0, 0]",
        "[0, 0, 0, 0, 0, 0, 0]",
        // Mixed positions
        "[1, 0, 0]",
        "[0, 1, 0]",
        "[0, 0, 1]",
        "[1, 0, 1]",
        "[0, 1, 1]",
        "[1, 1, 0]",
        "[1, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 1]",
        "[0, 0, 1, 0, 0]",
        "[1, 0, 1, 0, 1]",
        // Larger arrays
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]",
        "[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]",
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        // Negative values (nonzero)
        "[-1, 0, 0]",
        "[0, -1, 0]",
        "[0, 0, -1]",
        "[-1, -2, -3]",
        "[0, -5, 0, -10, 0]",
        // Floating point
        "[0.0, 0.0, 0.0]",
        "[1.0, 0.0, 0.0]",
        "[0.0, 1.0, 0.0]",
        "[0.0, 0.0, 1.0]",
        "[0.5, 0.0, 0.5]",
        "[0.0, 0.0, 0.001]",
        "[-0.5, 0.0, 0.5]",
        "[1.1, 2.2, 3.3]",
        "[0.0, 0.0, 0.0, 0.0]",
        "[0.0, 0.1, 0.0, 0.2, 0.0]",
        // Edge cases
        "[0, 1]",
        "[1, 0]",
        "[0, 0, 0, 1, 0]",
        "[5, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 5]",
        // More variety
        "[0, 8, 0, 0, 0]",
        "[0, 0, 0, 0, 0, 66]",
        "[1, 4, 9, 16, 25, 36, 49]",
        "[0, 0, 25, 0, 0]",
        "[-10, 0, 10, 0, -20]",
    ];

    for arr_str in &test_cases {
        let script =
            format!("import numpy as np; print(np.argwhere(np.array({arr_str})).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_nested_int_list(&numpy_result);

        let rust_script =
            fnp_argwhere_script(format!("print(fnp.argwhere(np.array({arr_str})).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_nested_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argwhere mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argwhere_2d_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays - returns Nx2 array of [row, col] indices
        "[[0, 0], [0, 0]]",
        "[[1, 0], [0, 0]]",
        "[[0, 1], [0, 0]]",
        "[[0, 0], [1, 0]]",
        "[[0, 0], [0, 1]]",
        "[[1, 1], [1, 1]]",
        "[[1, 0], [0, 1]]",
        "[[0, 1], [1, 0]]",
        // Different shapes
        "[[0, 0, 0], [0, 0, 0]]",
        "[[1, 0, 0], [0, 0, 0]]",
        "[[0, 0, 1], [0, 0, 0]]",
        "[[0, 0, 0], [1, 0, 0]]",
        "[[0, 0, 0], [0, 0, 1]]",
        "[[1, 2, 3], [4, 5, 6]]",
        "[[0, 0], [0, 0], [0, 0]]",
        "[[1, 0], [0, 1], [1, 0]]",
        // Single row/column
        "[[0, 0, 0, 0, 0]]",
        "[[0, 0, 1, 0, 0]]",
        "[[1, 2, 3, 4, 5]]",
        "[[0], [0], [0], [0]]",
        "[[0], [1], [0], [0]]",
        "[[1], [2], [3], [4]]",
    ];

    for arr_str in &test_cases {
        let script =
            format!("import numpy as np; print(np.argwhere(np.array({arr_str})).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_nested_int_list(&numpy_result);

        let rust_script =
            fnp_argwhere_script(format!("print(fnp.argwhere(np.array({arr_str})).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_nested_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argwhere 2D mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argwhere_3d_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]",
        "[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]",
        "[[[0, 0], [0, 0]], [[0, 0], [0, 1]]]",
        "[[[0, 1], [0, 0]], [[0, 0], [1, 0]]]",
        "[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]",
        "[[[0, 0, 0]], [[0, 0, 0]]]",
        "[[[1, 0, 0]], [[0, 0, 1]]]",
        "[[[0], [0], [0]], [[0], [1], [0]]]",
        "[[[1], [0], [1]], [[0], [1], [0]]]",
    ];

    for arr_str in &test_cases {
        let script =
            format!("import numpy as np; print(np.argwhere(np.array({arr_str})).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_nested_int_list(&numpy_result);

        let rust_script =
            fnp_argwhere_script(format!("print(fnp.argwhere(np.array({arr_str})).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_nested_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argwhere 3D mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argwhere_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([0, 1, 0, 2], dtype=np.int32)",
        "np.array([0, 0, 0, 0], dtype=np.int64)",
        "np.array([1, 0, 0], dtype=np.uint8)",
        "np.array([0, 0, 1], dtype=np.int16)",
        "np.array([0.0, 1.0, 0.0], dtype=np.float32)",
        "np.array([0.0, 0.0, 0.5], dtype=np.float64)",
        "np.array([True, False, True], dtype=np.bool_)",
        "np.array([False, False, False], dtype=np.bool_)",
        "np.array([[0, 1], [0, 0]], dtype=np.int32)",
        "np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.argwhere({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_nested_int_list(&numpy_result);

        let rust_script = fnp_argwhere_script(format!("print(fnp.argwhere({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_nested_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argwhere dtype mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argwhere_empty_array_matches_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.argwhere(np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_nested_int_list(&numpy_result);

    let rust_script = fnp_argwhere_script("print(fnp.argwhere(np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_nested_int_list(&rust_result);

    assert_eq!(
        numpy_vals, rust_vals,
        "argwhere empty array mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn argwhere_large_sparse_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.zeros(100)",
        "np.ones(100)",
        "np.eye(10)",
        "(np.arange(100) % 7 == 0)",
        "(np.arange(50) > 25)",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.argwhere({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_nested_int_list(&numpy_result);

        let rust_script = fnp_argwhere_script(format!("print(fnp.argwhere({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_nested_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argwhere large/sparse mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argwhere_complex_dtype_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "np.array([1+1j, 0+0j, 3+2j, 0+0j, 5-1j], dtype=np.complex128)",
        "np.array([0+0j, 0+0j, 0+0j], dtype=np.complex128)",
        "np.array([1+1j, 2-1j, 3+2j], dtype=np.complex128)",
        "np.array([[0+0j, 1+1j], [2-1j, 0+0j]], dtype=np.complex128)",
        "np.array([[1+1j, 0+0j], [0+0j, 2-1j]], dtype=np.complex128)",
    ];

    for arr_expr in &test_cases {
        let script = format!("import numpy as np; print(np.argwhere({arr_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_nested_int_list(&numpy_result);

        let rust_script = fnp_argwhere_script(format!("print(fnp.argwhere({arr_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_nested_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argwhere complex mismatch for {arr_expr}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn parallel_argwhere_bit_exact_matches_numpy() -> Result<(), String> {
    // Large contiguous inputs take the blocked parallel two-pass with
    // per-block odometer seeding (one divmod chain per block); coordinate
    // rows must be byte-identical to numpy across ndim, dtypes, densities,
    // non-block-multiple sizes, and degenerate shapes. Small and
    // non-contiguous inputs keep the serial native / delegate.
    let script = fnp_argwhere_script(
        r#"
import time
rng = np.random.default_rng(71)
verdicts = []
cases = [
    ("1-D int64", rng.integers(-5, 5, 4_000_003)),
    ("2-D int64 dense", rng.integers(-5, 5, (2048, 1024))),
    ("2-D bool 10%", rng.random((2048, 1024)) > 0.9),
    ("2-D f64", np.where(rng.random((1024, 1024)) > 0.5, rng.standard_normal((1024, 1024)), 0.0)),
    ("2-D odd", rng.integers(-2, 2, (999, 1237))),
    ("3-D int64", rng.integers(-3, 3, (128, 128, 128))),
    ("4-D int32", rng.integers(-2, 2, (16, 32, 64, 32)).astype(np.int32)),
    ("tall", rng.integers(-2, 2, (1_000_000, 3))),
    ("wide", rng.integers(-2, 2, (3, 1_000_000))),
    ("all-zero", np.zeros((1024, 1024), dtype=np.int64)),
    ("all-nonzero", np.ones((1024, 1024), dtype=np.uint16)),
]
for name, arr in cases:
    r = fnp.argwhere(arr); e = np.argwhere(arr)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {name}")
# small + non-contiguous stay byte-identical via serial/delegate
s = rng.integers(-2, 2, (40, 40))
if fnp.argwhere(s).tobytes() != np.argwhere(s).tobytes():
    verdicts.append("FAIL small serial")
nc = rng.integers(-2, 2, (2048, 2048))[:, ::2]
if fnp.argwhere(nc).tobytes() != np.argwhere(nc).tobytes():
    verdicts.append("FAIL non-contig delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W2 = rng.integers(-5, 5, (4096, 4096))
tn = best(lambda: np.argwhere(W2)); tf = best(lambda: fnp.argwhere(W2))
print(f"ARGWHERE_INT64_2D_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
W3 = rng.integers(-5, 5, (256, 256, 256))
tn = best(lambda: np.argwhere(W3)); tf = best(lambda: fnp.argwhere(W3))
print(f"ARGWHERE_INT64_3D_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
W2b = rng.random((4096, 4096)) > 0.9
tn = best(lambda: np.argwhere(W2b)); tf = best(lambda: fnp.argwhere(W2b))
print(f"ARGWHERE_BOOL_2D_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ARGWHERE_*_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "parallel argwhere must be bit-identical to numpy: {result}"
    );
    Ok(())
}
