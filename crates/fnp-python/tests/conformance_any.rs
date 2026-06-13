//! Conformance tests for numpy.any against NumPy oracle.
//!
//! Tests the native Rust any implementation against NumPy across various
//! input shapes, axis parameters, and data types.

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

fn fnp_any_script(body: String) -> String {
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

fn parse_bool(s: &str) -> bool {
    s.trim() == "True"
}

fn parse_bool_list(s: &str) -> Vec<bool> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|token| token.trim() == "True")
        .collect()
}

#[test]
fn any_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Has true elements
        "[1, 2, 3]",
        "[1, 0, 0, 0, 0]",
        "[True, False, False]",
        "[1]",
        "[100, 0, 0]",
        // All false
        "[0, 0, 0]",
        "[False, False, False]",
        "[0]",
        "[0, 0, 0, 0, 0]",
        "[0.0, 0.0, 0.0]",
        // Mixed boolean/int
        "[1, 2, 3, 4, 5]",
        "[0, 0, 0, 0, 1]",
        "[-1, 0, 0]",
        "[0, 0, -1]",
        "[1, -1, 2, -2, 3, -3]",
        // Floating point
        "[0.5, 0.0, 0.0]",
        "[0.0, 0.0, 1.1]",
        "[0.0, 0.0, 0.0]",
        "[1.0, 0.0, 2.0]",
        "[0.001, 0.0, 0.0]",
        // Zeros mixed
        "[0, 1, 0, 0, 0]",
        "[0, 0, 0, 0, 1]",
        "[0, 0, 0, 0, 0]",
        "[1, 0, 0, 0, 0]",
        "[0, 0, 1, 0, 0]",
        // Larger arrays
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]",
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]",
        "[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
        // Edge cases
        "[0.0, 0.0]",
        "[0.0, 1.0, 0.0, 0.0, 0.0]",
        "[-999, 0]",
        "[0, 999]",
        "[1, 2]",
        // More variety
        "[0, 0, 0, 1, 0]",
        "[0, 8, 0, 0, 0]",
        "[0, 0, 0, 0, 0, 66]",
        "[0, 0, 0, 0, 0, 0, 0]",
        "[1, 4, 9, 16, 25, 36, 49]",
        // Small values
        "[0.0, 0.0, 0.0, 1.3]",
        "[0.0, 1.0, 0.0]",
        "[0.0, 0.0, 0.0]",
        "[0.0, 0.0, 101.0]",
        "[1000, 0, 0, 0]",
        // Additional cases
        "[0, 0, 25, 0, 0]",
        "[0, 0, 0, 0, 0, 10]",
        "[-10, 0, 0, 0, 0]",
        "[0.0, 0.0, 0.3, 0.0, 0.0]",
        "[0, 0, 0, 4, 0, 0, 0, 0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.any(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_bool(&numpy_result);

        let rust_script = fnp_any_script(format!("print(fnp.any(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_bool(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "any flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn any_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[0, 0, 0], [0, 0, 0]]", "0"),
        ("[[0, 1, 0], [0, 0, 0]]", "0"),
        ("[[0, 0], [1, 0], [0, 0]]", "0"),
        ("[[0, 0], [0, 0], [0, 0], [1, 0]]", "0"),
        ("[[0, 0, 0], [1, 1, 1], [0, 0, 0]]", "0"),
        // 2D arrays with axis=1
        ("[[0, 0, 0], [0, 1, 0]]", "1"),
        ("[[0, 0], [1, 0], [0, 0]]", "1"),
        ("[[0, 1], [0, 0], [1, 0], [0, 0]]", "1"),
        ("[[0, 0, 0], [1, 1, 1]]", "1"),
        ("[[0, 0, 1], [0, 0, 0], [1, 0, 0]]", "1"),
        // Negative axis
        ("[[0, 0, 0], [0, 1, 0]]", "-1"),
        ("[[0, 0, 0], [0, 0, 0]]", "-2"),
        ("[[0, 1, 0], [0, 0, 0], [0, 0, 1]]", "-1"),
        ("[[0, 1, 0], [0, 0, 0], [0, 0, 1]]", "-2"),
        // Single row/column
        ("[[0, 0, 0, 0, 0]]", "0"),
        ("[[0, 0, 1, 0, 0]]", "1"),
        ("[[0], [0], [0], [0]]", "0"),
        ("[[0], [1], [0], [0]]", "1"),
        // All true
        ("[[1, 1], [1, 1]]", "0"),
        ("[[1, 1], [1, 1]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script =
            format!("import numpy as np; print(np.any(np.array({arr_str}), axis={axis}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script = fnp_any_script(format!(
            "print(fnp.any(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "any axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn any_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays
        ("[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]", "0"),
        ("[[[0, 0], [1, 0]], [[0, 0], [0, 0]]]", "1"),
        ("[[[0, 1], [0, 0]], [[0, 0], [0, 0]]]", "2"),
        ("[[[0, 0], [0, 0]], [[0, 0], [0, 1]]]", "-1"),
        ("[[[0, 0], [0, 0]], [[1, 0], [0, 0]]]", "-2"),
        ("[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]", "-3"),
        // Different shapes
        ("[[[0, 0, 0]], [[0, 0, 0]]]", "0"),
        ("[[[0, 0, 1]], [[0, 0, 0]]]", "1"),
        ("[[[0, 1, 0]], [[0, 0, 0]]]", "2"),
        ("[[[0], [0], [0]], [[0], [1], [0]]]", "0"),
        ("[[[0], [0], [1]], [[0], [0], [0]]]", "1"),
        ("[[[1], [0], [0]], [[0], [0], [0]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.any(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_bool_list(&numpy_result);

        let rust_script = fnp_any_script(format!(
            "print(fnp.any(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_bool_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "any 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn any_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([0, 0, 3], dtype=np.int32)", "None"),
        ("np.array([0, 0, 0], dtype=np.int64)", "None"),
        ("np.array([1, 0, 0], dtype=np.uint8)", "None"),
        ("np.array([0, 0, 0], dtype=np.int16)", "None"),
        ("np.array([[0, 0], [0, 1]], dtype=np.int32)", "None"),
        ("np.array([[0, 0], [0, 0]], dtype=np.int64)", "None"),
        (
            "np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)",
            "None",
        ),
        (
            "np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)",
            "None",
        ),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(np.any({arr_expr}{axis_arg}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_bool(&numpy_result);

        let rust_script = fnp_any_script(format!("print(fnp.any({arr_expr}{axis_arg}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_bool(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "any dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn any_empty_array_matches_numpy() -> Result<(), String> {
    let test_cases = vec![("[]", "None")];

    for (arr_str, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(np.any(np.array({arr_str}){axis_arg}))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_any_script(format!("print(fnp.any(np.array({arr_str}){axis_arg}))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "any empty array mismatch for {arr_str} axis={axis}"
        );
    }

    Ok(())
}

#[test]
fn any_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_any_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.any(x)
np_result = np.any(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "any scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn any_complex_dtype_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // All nonzero complex
        "[1+1j, 2-1j, 3+2j]",
        "[0.5+0.5j, 1.5-1.5j]",
        // Contains zero complex
        "[1+1j, 0+0j, 3+2j]",
        "[0+0j, 0+0j]",
        // Single element
        "[1+1j]",
        "[0+0j]",
        // One nonzero among zeros
        "[0+0j, 0+0j, 1+1j]",
    ];

    for arr_str in &test_cases {
        let script =
            format!("import numpy as np; print(np.any(np.array({arr_str}, dtype=np.complex128)))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_bool(&numpy_result);

        let rust_script = fnp_any_script(format!(
            "print(fnp.any(np.array({arr_str}, dtype=np.complex128)))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_bool(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "any complex mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn any_special_values_match_numpy() -> Result<(), String> {
    let script = fnp_any_script(
        r#"
tests = []
# NaN is truthy (nonzero)
a = np.array([np.nan, 0.0, 0.0])
tests.append(fnp.any(a) == np.any(a))

# inf is truthy
a = np.array([np.inf, 0.0])
tests.append(fnp.any(a) == np.any(a))

# -inf is truthy
a = np.array([-np.inf, 0.0])
tests.append(fnp.any(a) == np.any(a))

# All zeros (falsy)
a = np.array([0.0, 0.0, 0.0])
tests.append(fnp.any(a) == np.any(a))

# Mixed special values
a = np.array([np.nan, np.inf, -np.inf])
tests.append(fnp.any(a) == np.any(a))

# Only nan
a = np.array([np.nan])
tests.append(fnp.any(a) == np.any(a))

# Only inf
a = np.array([np.inf])
tests.append(fnp.any(a) == np.any(a))

print(all(tests))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "any special values should match numpy"
    );
    Ok(())
}

#[test]
fn any_negative_zero_matches_numpy() -> Result<(), String> {
    let script = fnp_any_script(
        r#"
# -0.0 is falsy (zero) - array of all negative zeros should return False
a = np.array([-0.0, -0.0])
fnp_result = fnp.any(a)
np_result = np.any(a)
print(fnp_result == np_result and fnp_result == False)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "any negative zero should match numpy"
    );
    Ok(())
}

/// Locks the block-folded full-reduction any/all fast path (block_any/all_u8/f64):
/// a battery of bool and float64 arrays — all-false, all-true, scattered-true,
/// NaN (truthy), -0.0 (falsy), single-element, large — with both any() and all(),
/// each result must equal numpy. The concatenated truth string is sha256-pinned.
#[test]
fn any_all_block_fold_full_reduction_matches_numpy_and_golden() -> Result<(), String> {
    let script = fnp_any_script(
        r#"
import hashlib
def mk(kind):
    if kind == 'b_allfalse': return np.zeros(70001, bool)
    if kind == 'b_alltrue':  return np.ones(70001, bool)
    if kind == 'b_scatter':  a=np.zeros(70001,bool); a[40000]=True; return a
    if kind == 'b_early':    a=np.zeros(70001,bool); a[3]=True; return a
    if kind == 'f_allzero':  return np.zeros(70001)
    if kind == 'f_nan':      a=np.zeros(70001); a[50000]=np.nan; return a
    if kind == 'f_negzero':  a=np.full(70001,-0.0); a[10]=1.0; return a
    if kind == 'f_mixed':    a=np.zeros(70001); a[12345]=2.5; return a
    if kind == 'one_true':   return np.array([True])
    if kind == 'one_zero':   return np.array([0.0])
bits = []
for kind in ('b_allfalse','b_alltrue','b_scatter','b_early','f_allzero','f_nan','f_negzero','f_mixed','one_true','one_zero'):
    a = mk(kind)
    for fn, nf in ((fnp.any, np.any), (fnp.all, np.all)):
        r = bool(fn(a)); e = bool(nf(a))
        bits.append('1' if r == e else '0')
        bits.append('T' if r else 'F')
s = ''.join(bits)
print(s.count('0') == 0)        # every fnp result matched numpy
print(hashlib.sha256(s.encode()).hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "True",
        "block-folded any/all must match numpy on every case"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "197f8d6fd5d3fab69542db049a4bfd91d454a35cdb964cbdbdd3142788eecc73",
        "any/all block-fold golden sha256 drifted"
    );
    Ok(())
}

/// Locks the per-axis any/all fast path (axis_any_all_fold): bool and float64
/// arrays reduced over axis 0 (non-last, row-sequential accumulator) and the
/// contiguous last axis (per-lane block-fold), 2-D and 3-D, with all-false /
/// all-true / scattered / NaN (truthy) / -0.0 (falsy) — each result byte-identical
/// to numpy plus a sha256 golden over all the result bytes.
#[test]
fn any_all_axis_fast_path_matches_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_any_script(
        r#"
import hashlib
s = 0x9E3779B97F4A7C15
def nxt():
    global s
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return s
F = np.empty((130, 71), dtype=np.float64)
for i in range(130):
    for j in range(71):
        F[i, j] = ((nxt() >> 11) / (1 << 53)) * 8.0 - 4.0
F[::13, 4] = 0.0        # falsy column
F[7, ::9] = np.nan      # nan (truthy)
F[9, ::11] = -0.0       # falsy
B = (F > 0)
B[:, 20] = False        # all-false column (axis 0)
B[40, :] = True         # all-true row (axis 1)
T3 = np.empty((11, 13, 9), dtype=np.float64)
for x in np.ndindex(11, 13, 9):
    T3[x] = ((nxt() >> 11) / (1 << 53)) * 4.0 - 2.0
B3 = (T3 > 0)
h = hashlib.sha256()
allmatch = True
for arr in (F, B, T3, B3):
    for ax in range(arr.ndim):
        for fn, nf in ((fnp.any, np.any), (fnp.all, np.all)):
            r = np.asarray(fn(arr, axis=ax))
            e = np.asarray(nf(arr, axis=ax))
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
        "per-axis any/all must be byte-identical to numpy"
    );
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "1d2281e04ffe332c20dec574941526b2aa0fc20c49a20880684df650a84a9b55",
        "per-axis any/all golden sha256 drifted"
    );
    Ok(())
}
