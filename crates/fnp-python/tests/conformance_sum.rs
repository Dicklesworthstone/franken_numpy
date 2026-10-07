//! Conformance tests for numpy.sum against NumPy oracle.
//!
//! Tests the native Rust sum implementation against NumPy across various
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

mod support;

fn fnp_sum_script(body: String) -> String {
    support::fnp_script(body)
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
    (a - b).abs() < tol
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
fn sum_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic arrays
        "[1, 2, 3]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[0, 0, 0]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2, 3, -3]",
        "[100, 200, 300, 400, 500]",
        // Floating point
        "[0.5, 1.5, 2.5]",
        "[1.1, 2.2, 3.3, 4.4]",
        "[0.001, 0.002, 0.003]",
        "[1e10, 2e10, 3e10]",
        "[1e-10, 2e-10, 3e-10]",
        // Negatives and zeros
        "[-100, 0, 100]",
        "[-1.5, -0.5, 0.5, 1.5]",
        "[0, 1, 0, 1, 0]",
        "[-5, -4, -3, -2, -1, 0]",
        "[0, -1, -2, -3, -4, -5]",
        // Larger arrays
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        "[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]",
        "[1, 3, 5, 7, 9, 11, 13, 15]",
        "[2, 4, 6, 8, 10, 12, 14, 16]",
        "[1, 1, 2, 3, 5, 8, 13, 21]",
        // Mixed
        "[0.5, 1, 1.5, 2, 2.5, 3]",
        "[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]",
        "[1, 10, 100, 1000, 10000]",
        "[10000, 1000, 100, 10, 1]",
        "[3.14159, 2.71828, 1.41421]",
        // Edge values
        "[0.0, 0.0]",
        "[1.0, 1.0, 1.0, 1.0, 1.0]",
        "[-999, 999]",
        "[0.123456789, 0.987654321]",
        "[1, 2]",
        // More variety
        "[7, 3, 9, 1, 5]",
        "[2, 8, 4, 6, 0]",
        "[11, 22, 33, 44, 55, 66]",
        "[99, 88, 77, 66, 55, 44, 33]",
        "[1, 4, 9, 16, 25, 36, 49]",
        // Small ranges
        "[1.0, 1.1, 1.2, 1.3]",
        "[0.99, 1.0, 1.01]",
        "[-0.01, 0.0, 0.01]",
        "[100.0, 100.5, 101.0]",
        "[1000, 1001, 1002, 1003]",
        // Additional cases
        "[5, 15, 25, 35, 45]",
        "[0, 2, 4, 6, 8, 10]",
        "[-10, -5, 0, 5, 10]",
        "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]",
        "[1, 3, 2, 4, 3, 5, 4, 6]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.sum(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_sum_script(format!("print(fnp.sum(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-9),
            "sum flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn sum_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 2, 3], [4, 5, 6]]", "0"),
        ("[[1, 4], [2, 5], [3, 6]]", "0"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "0"),
        ("[[10, 20, 30], [5, 15, 25]]", "0"),
        ("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 2, 3], [4, 5, 6]]", "1"),
        ("[[1, 4], [2, 5], [3, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "1"),
        ("[[10, 20, 30], [5, 15, 25]]", "1"),
        ("[[1, 5, 9], [2, 6, 10], [3, 7, 11]]", "1"),
        // Negative axis
        ("[[1, 2, 3], [4, 5, 6]]", "-1"),
        ("[[1, 2, 3], [4, 5, 6]]", "-2"),
        ("[[1, 4, 7], [2, 5, 8], [3, 6, 9]]", "-1"),
        ("[[1, 4, 7], [2, 5, 8], [3, 6, 9]]", "-2"),
        // Single row/column
        ("[[1, 2, 3, 4, 5]]", "0"),
        ("[[1, 2, 3, 4, 5]]", "1"),
        ("[[1], [2], [3], [4]]", "0"),
        ("[[1], [2], [3], [4]]", "1"),
        // Floating point 2D
        ("[[0.5, 1.5], [2.5, 3.5]]", "0"),
        ("[[0.5, 1.5], [2.5, 3.5]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script =
            format!("import numpy as np; print(np.sum(np.array({arr_str}), axis={axis}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_sum_script(format!(
            "print(fnp.sum(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "sum axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn sum_3d_axis_matches_numpy() -> Result<(), String> {
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
            "import numpy as np; print(np.sum(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result);

        let rust_script = fnp_sum_script(format!(
            "print(fnp.sum(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result);

        assert!(
            arrays_close(&numpy_vals, &rust_vals, 1e-9),
            "sum 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }
    Ok(())
}

#[test]
fn sum_keepdims_matches_numpy() -> Result<(), String> {
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
            "import numpy as np; print(np.sum(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        );
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_sum_script(format!(
            "print(fnp.sum(np.array({arr_str}){axis_arg}, keepdims={}).shape)",
            if *keepdims { "True" } else { "False" }
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "sum keepdims={keepdims} shape mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn sum_unknown_keyword_matches_numpy_error() -> Result<(), String> {
    let numpy_script = r#"import numpy as np
try:
    print(np.sum(np.array([1, 2, 3]), unexpected_kw=1))
except Exception as exc:
    print(f'{type(exc).__name__}:{exc}')"#;
    let numpy_result = numpy_oracle(numpy_script)?;

    let rust_script = fnp_sum_script(
        r#"try:
    print(fnp.sum(np.array([1, 2, 3]), unexpected_kw=1))
except Exception as exc:
    print(f'{type(exc).__name__}:{exc}')"#
            .to_string(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert!(
        numpy_result.starts_with("TypeError:"),
        "NumPy should reject unknown sum keyword, got {numpy_result}"
    );
    assert_eq!(numpy_result, rust_result);
    Ok(())
}

#[test]
fn sum_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3], dtype=np.int32)", "None"),
        ("np.array([1, 2, 3], dtype=np.int64)", "None"),
        ("np.array([1, 2, 3], dtype=np.uint8)", "None"),
        ("np.array([100, 200, 300], dtype=np.int16)", "None"),
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
        let script = format!("import numpy as np; print(float(np.sum({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_float(&numpy_result);

        let rust_script = fnp_sum_script(format!("print(float(fnp.sum({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_float(&rust_result);

        assert!(
            floats_close(numpy_val, rust_val, 1e-6),
            "sum dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }
    Ok(())
}

#[test]
fn sum_large_integer_flat_parallel_is_bit_exact() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
rng = np.random.default_rng(20260742)
checks = []
for dtype in [np.int8, np.uint8, np.int16, np.uint16,
              np.int32, np.uint32, np.int64, np.uint64]:
    dt = np.dtype(dtype)
    # 4- and 8-byte widths reach the pool from 2^22 elements; 1- and 2-byte widths run the
    # serial 32-bit-lane fold here (their pool floor, 32 MiB, is covered below).
    n = max(8 * 1024 * 1024, (1 << 22) * dt.itemsize) // dt.itemsize
    a = np.frombuffer(rng.bytes(n * dt.itemsize), dtype=dt).copy()
    ours = fnp.sum(a)
    theirs = np.sum(a)
    checks.append(type(ours) is type(theirs))
    checks.append(ours.dtype == theirs.dtype)
    checks.append(ours.tobytes() == theirs.tobytes())

    shaped = a.reshape(8, -1)
    ours_keep = fnp.sum(shaped, keepdims=True)
    theirs_keep = np.sum(shaped, keepdims=True)
    checks.append(ours_keep.shape == theirs_keep.shape)
    checks.append(ours_keep.dtype == theirs_keep.dtype)
    checks.append(ours_keep.tobytes() == theirs_keep.tobytes())

# Explicit wraparound witnesses for both promoted accumulator classes.
signed = np.full(4_200_000, np.iinfo(np.int64).max, dtype=np.int64)
unsigned = np.full(4_200_000, np.iinfo(np.uint64).max, dtype=np.uint64)
checks.append(fnp.sum(signed).tobytes() == np.sum(signed).tobytes())
checks.append(fnp.sum(unsigned).tobytes() == np.sum(unsigned).tobytes())

# Every unsupported boundary remains delegated.
base = np.arange(4_000_000, dtype=np.int64)
strided = base[::2]
checks.append(fnp.sum(strided).tobytes() == np.sum(strided).tobytes())
checks.append(fnp.sum(base, dtype=np.float64).tobytes() == np.sum(base, dtype=np.float64).tobytes())
checks.append(fnp.sum(base, initial=17).tobytes() == np.sum(base, initial=17).tobytes())
checks.append(fnp.sum(base.astype('>i8')).tobytes() == np.sum(base.astype('>i8')).tobytes())
checks.append(fnp.sum(np.ones(9_000_000, dtype=np.bool_)).tobytes() ==
              np.sum(np.ones(9_000_000, dtype=np.bool_)).tobytes())

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 55");
    Ok(())
}

/// 1- and 2-byte integer sums fold 2^15-value blocks in 32-bit lanes, serially from 2^12
/// elements and on the pool from 32 MiB. All-extreme operands longer than one lane block are the
/// negative case: a fold that kept 32-bit lanes across the whole slice overflows at int16 max x
/// 2^20, and one that mis-sized a block overflows at uint16 max. Sizes straddle the serial floor
/// and the lane-block edge; strided, byte-swapped and bool operands stay numpy's.
#[test]
fn sum_narrow_integer_lane_blocks_match_numpy() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
rng = np.random.default_rng(20260928)
bad = []
count = 0
def check(label, a, **kw):
    global count
    count += 1
    ours, theirs = fnp.sum(a, **kw), np.sum(a, **kw)
    if type(ours) is not type(theirs) or np.asarray(ours).dtype != np.asarray(theirs).dtype \
            or np.asarray(ours).tobytes() != np.asarray(theirs).tobytes():
        bad.append((label, a.dtype.name, a.size))
for dtype in [np.int8, np.uint8, np.int16, np.uint16]:
    info = np.iinfo(dtype)
    for n in [4095, 4096, 32767, 32768, 32769, 1 << 20]:
        check("random", rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True))
        check("min", np.full(n, info.min, dtype=dtype))
        check("max", np.full(n, info.max, dtype=dtype))
    check("2-D keepdims", np.full((64, 1024), info.max, dtype=dtype), keepdims=True)
    # The pool route: 32 MiB of input.
    pool_n = (32 << 20) // np.dtype(dtype).itemsize
    check("pool max", np.full(pool_n, info.max, dtype=dtype))
    check("pool random", rng.integers(info.min, info.max, pool_n, dtype=dtype, endpoint=True))
    wide = rng.integers(info.min, info.max, 20000, dtype=dtype, endpoint=True)
    check("strided", wide[::2])
    check("byte-swapped", wide.astype(np.dtype(dtype).newbyteorder()))
check("bool", rng.random(100000) < 0.5)
# bool sums count NONZERO bytes (numpy's bool -> int64 cast), so a view holding 2 or 255 counts 1.
check("bool view of raw bytes", rng.integers(0, 256, 40000, dtype=np.uint8).view(np.bool_))
check("bool pool", rng.integers(0, 256, 32 << 20, dtype=np.uint8).view(np.bool_))
print(bad if bad else True, count)
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 95");
    Ok(())
}

/// sum / mean over one contiguous run of axes of 1- and 2-byte integers and bool
/// (`try_narrow_integer_axis_reduction`): exact totals, so numpy's reduction order cannot
/// matter. Covers a 3-channel image (numpy's slowest shape), every axis spelling, keepdims, the
/// all-axes scalar, raw-byte bools, and the 32-bit lane flush: a (1_500_000, 3) int16 of maxima
/// over axis 0 puts ~68k values in each lane, which overflows without the flush. Scattered,
/// repeated and out-of-range axes, dtype= / initial= / where=, and non-C layouts are numpy's.
#[test]
fn sum_mean_narrow_integer_axis_runs_match_numpy() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
import itertools
rng = np.random.default_rng(20261001)
bad = []
count = 0
def outcome(fn, a, kw):
    try:
        return fn(a, **kw)
    except Exception as ex:
        return (type(ex).__name__, str(ex))
def check(fname, a, **kw):
    global count
    count += 1
    ours, theirs = outcome(getattr(fnp, fname), a, kw), outcome(getattr(np, fname), a, kw)
    if isinstance(ours, tuple) or isinstance(theirs, tuple):
        ok = isinstance(ours, tuple) and isinstance(theirs, tuple) and ours == theirs
    else:
        o, t = np.asarray(ours), np.asarray(theirs)
        ok = type(ours) is type(theirs) and o.dtype == t.dtype and o.shape == t.shape \
            and o.tobytes() == t.tobytes()
    if not ok:
        bad.append((fname, a.dtype.name, a.shape, kw))
img = rng.integers(0, 256, (270, 480, 3), dtype=np.uint8)
for ax in [(0, 1), 2, -1, 0, (1, 2), (0, 1, 2), (1, 0)]:
    check("mean", img, axis=ax); check("sum", img, axis=ax)
    check("mean", img, axis=ax, keepdims=True)
for dtype in [np.int8, np.uint8, np.int16, np.uint16]:
    info = np.iinfo(dtype)
    for shape in [(64, 65), (700, 7, 3), (33, 64, 5), (3, 5000), (70000, 2)]:
        a = np.where(rng.random(shape) < 0.5, info.min, info.max).astype(dtype)
        axes = list(range(a.ndim)) + [tuple(c) for k in range(2, a.ndim + 1)
                                      for c in itertools.combinations(range(a.ndim), k)]
        for ax in axes:
            check("sum", a, axis=ax); check("mean", a, axis=ax)
for a in [rng.random((300, 40)) < 0.4, rng.integers(0, 256, (300, 40), dtype=np.uint8).view(np.bool_)]:
    for ax in [0, 1, (0, 1)]:
        check("sum", a, axis=ax); check("mean", a, axis=ax)
flush = np.full((1_500_000, 3), np.iinfo(np.int16).max, dtype=np.int16)
check("sum", flush, axis=0); check("mean", flush, axis=0)
check("sum", flush.view(np.uint16), axis=0)
x = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
for ax in [(0, 2), (0, 0), 3, -4]:
    check("sum", x, axis=ax); check("mean", x, axis=ax)
check("sum", x, axis=0, dtype=np.int32); check("sum", x, axis=0, initial=5)
check("mean", x, axis=0, where=np.ones(x.shape, bool))
check("sum", np.asfortranarray(x), axis=0); check("mean", x[:, ::2], axis=1)
expected = np.mean(img, axis=(0, 1))

def poisoned_mean(*args, **kwargs):
    raise AssertionError("narrow axis mean route unexpectedly delegated")

np.mean = poisoned_mean
routed = fnp.mean(img, axis=(0, 1)).tobytes() == expected.tobytes()
print(bad if bad else True, count, routed)
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 233 True");
    Ok(())
}

#[test]
fn sum_nan_handling_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1.0, np.nan, 3.0]",
        "[np.nan, 2.0, 3.0]",
        "[1.0, 2.0, np.nan]",
        "[np.nan, np.nan, np.nan]",
        "[1.0, np.nan, np.nan, 4.0]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.sum(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_sum_script(format!("print(fnp.sum(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "sum NaN mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn sum_empty_array_matches_numpy() -> Result<(), String> {
    let test_cases = vec![("[]", "None"), ("[[]]", "None")];

    for (arr_str, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script =
            format!("import numpy as np; print(float(np.sum(np.array({arr_str}){axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;

        let rust_script = fnp_sum_script(format!(
            "print(float(fnp.sum(np.array({arr_str}){axis_arg})))"
        ));
        let rust_result = numpy_oracle(&rust_script)?;

        assert_eq!(
            numpy_result.trim(),
            rust_result.trim(),
            "sum empty array mismatch for {arr_str} axis={axis}"
        );
    }
    Ok(())
}

#[test]
fn sum_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.sum(x)
np_result = np.sum(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "sum scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_complex() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
fnp_result = fnp.sum(z)
np_result = np.sum(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "sum complex should match numpy");
    Ok(())
}

#[test]
fn sum_complex_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
z = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
fnp_result = fnp.sum(z, axis=0)
np_result = np.sum(z, axis=0)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum complex axis=0 should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error behavior tests
// ─────────────────────────────────────────────────────────────────────────────

fn classify_error(script: &str) -> String {
    let output = std::process::Command::new("python3")
        .args(["-c", script])
        .output()
        .expect("python3 should be available");
    if output.status.success() {
        "ok".to_string()
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("AxisError") || stderr.contains("axis") {
            "AxisError".to_string()
        } else if stderr.contains("ValueError") {
            "ValueError".to_string()
        } else {
            format!("other: {}", stderr.lines().last().unwrap_or(""))
        }
    }
}

#[test]
fn sum_axis_out_of_bounds_raises_axiserror() {
    let fnp_err = classify_error(&fnp_sum_script(
        r#"
a = fnp.arange(12).reshape(3, 4)
fnp.sum(a, axis=5)
"#
        .into(),
    ));
    let np_err = classify_error(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
np.sum(a, axis=5)
"#,
    );
    assert_eq!(
        fnp_err, np_err,
        "sum with out-of-bounds axis should raise same error as numpy"
    );
}

#[test]
fn sum_inf_handling_matches_numpy() -> Result<(), String> {
    let inf_cases = [
        "[1.0, np.inf, 3.0]",
        "[np.inf, 2.0, 3.0]",
        "[-np.inf, np.inf]",
        "[np.inf, np.inf]",
        "[-np.inf, -np.inf]",
    ];

    for arr_str in &inf_cases {
        let np_script = format!("import numpy as np; print(repr(np.sum(np.array({arr_str}))))");
        let np_output = numpy_oracle(&np_script)?;

        let fnp_script = fnp_sum_script(format!("print(repr(fnp.sum(np.array({arr_str}))))"));
        let fnp_output = numpy_oracle(&fnp_script)?;

        assert_eq!(
            fnp_output.trim(),
            np_output.trim(),
            "sum inf mismatch for {arr_str}"
        );
    }
    Ok(())
}

#[test]
fn sum_with_out_parameter() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2], [3, 4]])
out = np.empty((2,), dtype=np.int64)
fnp_result = fnp.sum(a, axis=0, out=out)
np_out = np.empty((2,), dtype=np.int64)
np_result = np.sum(a, axis=0, out=np_out)
# Check both result and that out was modified
print(np.array_equal(fnp_result, np_result) and np.array_equal(out, np_out))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum with out parameter should match numpy"
    );
    Ok(())
}

#[test]
fn sum_with_where_parameter() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])
fnp_result = fnp.sum(a, where=mask)
np_result = np.sum(a, where=mask)
print(fnp_result == np_result == 9)  # 1 + 3 + 5
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum with where parameter should match numpy"
    );
    Ok(())
}

#[test]
fn sum_with_initial_parameter() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
fnp_result = fnp.sum(a, initial=10)
np_result = np.sum(a, initial=10)
print(fnp_result == np_result == 25)  # 10 + 1+2+3+4+5
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum with initial parameter should match numpy"
    );
    Ok(())
}

#[test]
fn sum_signed_zero_parity() -> Result<(), String> {
    // Test signed-zero behavior for parallel operation safety proofs.
    // IEEE 754: 0.0 + 0.0 = 0.0, -0.0 + -0.0 = -0.0, 0.0 + -0.0 = 0.0
    let script = fnp_sum_script(
        r#"
# Signed-zero sum semantics
tests = [
    ([0.0, 0.0], False),      # 0.0 + 0.0 = 0.0 (positive)
    ([-0.0, -0.0], True),     # -0.0 + -0.0 = -0.0 (negative)
    ([0.0, -0.0], False),     # 0.0 + -0.0 = 0.0 (positive - IEEE 754 rule)
    ([-0.0, 0.0], False),     # -0.0 + 0.0 = 0.0 (positive)
    ([-0.0, -0.0, -0.0], True), # Multiple -0.0 sum
]
all_pass = True
for values, expected_signbit in tests:
    arr = np.array(values)
    fnp_result = fnp.sum(arr)
    np_result = np.sum(arr)
    if np.signbit(fnp_result) != np.signbit(np_result):
        print(f"FAIL: sum({values}) fnp signbit={np.signbit(fnp_result)} np signbit={np.signbit(np_result)}")
        all_pass = False
print(all_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum signed-zero parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_accumulation_stability() -> Result<(), String> {
    // Test that sum accumulation order matches NumPy
    let script = fnp_sum_script(
        r#"
# Large values that could suffer from accumulation order issues
a = np.array([1e16, 1.0, -1e16])
fnp_result = fnp.sum(a)
np_result = np.sum(a)

# Also test with axis reduction
b = np.array([[1e16, 1.0], [-1e16, 2.0]])
fnp_axis = fnp.sum(b, axis=0)
np_axis = np.sum(b, axis=0)

axis_match = np.allclose(fnp_axis, np_axis)
scalar_match = np.isclose(fnp_result, np_result) or (fnp_result == np_result)
print(scalar_match and axis_match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum accumulation stability should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_negative_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
fnp_result_m1 = fnp.sum(a, axis=-1)
np_result_m1 = np.sum(a, axis=-1)
fnp_result_m2 = fnp.sum(a, axis=-2)
np_result_m2 = np.sum(a, axis=-2)
print(np.array_equal(fnp_result_m1, np_result_m1) and np.array_equal(fnp_result_m2, np_result_m2))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum with negative axis should match numpy"
    );
    Ok(())
}

#[test]
fn sum_tuple_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.arange(24).reshape(2, 3, 4)
fnp_result_02 = fnp.sum(a, axis=(0, 2))
np_result_02 = np.sum(a, axis=(0, 2))
fnp_result_12 = fnp.sum(a, axis=(1, 2))
np_result_12 = np.sum(a, axis=(1, 2))
print(
    fnp_result_02.shape == np_result_02.shape,
    fnp_result_12.shape == np_result_12.shape,
    np.array_equal(fnp_result_02, np_result_02),
    np.array_equal(fnp_result_12, np_result_12)
)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True True True"),
        "sum with tuple axis should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_axis_none_flatten() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
fnp_result = fnp.sum(a, axis=None)
np_result = np.sum(a, axis=None)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum with axis=None should flatten and match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// dtype parameter tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sum_dtype_parameter_int_to_float() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
fnp_result = fnp.sum(a, dtype=np.float64)
np_result = np.sum(a, dtype=np.float64)
print(fnp_result.dtype == np_result.dtype, fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True"),
        "sum with dtype=float64 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_dtype_parameter_int_to_int64() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([1, 2, 3], dtype=np.int32)
fnp_result = fnp.sum(a, dtype=np.int64)
np_result = np.sum(a, dtype=np.int64)
print(fnp_result.dtype == np_result.dtype, fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True"),
        "sum with dtype=int64 should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_dtype_parameter_with_axis() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
a = np.array([[1, 2], [3, 4]], dtype=np.int32)
fnp_result = fnp.sum(a, axis=0, dtype=np.float64)
np_result = np.sum(a, axis=0, dtype=np.float64)
print(fnp_result.dtype == np_result.dtype, np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True True"),
        "sum with axis and dtype should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_large_values_overflow_matches_numpy() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
# Test overflow to inf behavior
large = np.finfo(np.float64).max / 2
a = np.array([large, large, large], dtype=np.float64)
fnp_result = fnp.sum(a)
np_result = np.sum(a)
both_inf = np.isinf(fnp_result) and np.isinf(np_result)
same_sign = fnp_result > 0 and np_result > 0
print(both_inf and same_sign)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum large value overflow should match numpy"
    );
    Ok(())
}

#[test]
fn sum_subnormal_values_matches_numpy() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
tiny = np.finfo(np.float64).tiny
subnormal = tiny / 2.0
a = np.array([subnormal, subnormal, subnormal], dtype=np.float64)
fnp_result = fnp.sum(a)
np_result = np.sum(a)
print(np.allclose(fnp_result, np_result, rtol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "sum subnormal values should match numpy"
    );
    Ok(())
}

#[test]
fn sum_lastaxis_native_pairwise_bitexact_matches_numpy() -> Result<(), String> {
    // Exercises the native last-axis pairwise sum fast path against numpy bit-exactly
    // (atol=0, equal_nan=True) incl dtype/shape: 2-D and 3-D last axis, negative axis,
    // keepdims, a NaN/Inf lane (propagate), and a non-last axis fallthrough (axis=0).
    let script = fnp_sum_script(
        r#"
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

rng = np.random.default_rng(41)
m2 = rng.standard_normal((4096, 1023))
m3 = rng.standard_normal((64, 50, 41))
nanm = m2.copy(); nanm[7, 3] = np.nan; nanm[9, 0] = np.inf
ok = True
cases = [
    (m2, -1, False),
    (m2, 1, True),
    (m3, -1, False),
    (m3, 2, True),
    (nanm, -1, False),
    (m2, 0, False),   # non-last axis -> fallthrough to numpy, must still match
]
for arr, axis, keepdims in cases:
    f = fnp.sum(arr, axis=axis, keepdims=keepdims)
    n = np.sum(arr, axis=axis, keepdims=keepdims)
    if not same(f, n):
        print("FAIL", axis, keepdims, np.asarray(f).ravel()[:4], np.asarray(n).ravel()[:4]); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native last-axis sum parity should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn sum_flat_parallel_float_pairwise_is_bitexact() -> Result<(), String> {
    // Both arrays clear the 16 MiB native-route gate.  The adversarial values
    // exercise cancellation, signed zero, infinities, NaN propagation, scalar
    // dtype/type, and the keepdims reconstruction around the parallel tree.
    let script = fnp_sum_script(
        r#"
def same(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    return (
        type(a) is type(b)
        and aa.shape == bb.shape
        and aa.dtype == bb.dtype
        and aa.tobytes() == bb.tobytes()
    )

rng = np.random.default_rng(90210)
# Above the f64 route's 2^22-element floor, so the native tree is the one compared.
f64 = rng.standard_normal((1 << 22) + 21, dtype=np.float64)
f64[7:15] = [1e300, -1e300, 1.0, -0.0, np.inf, -np.inf, 3.0, -3.0]
f32 = rng.standard_normal(4095 * 1025, dtype=np.float32).reshape(4095, 1025)
f32.flat[9:17] = np.array([1e30, -1e30, 1.0, -0.0, np.inf, -np.inf, 7.0, -7.0], dtype=np.float32)

cases = [
    (f64, False),
    (f64, True),
    (f32, False),
    (f32, True),
]
ok = True
for arr, keepdims in cases:
    got = fnp.sum(arr, keepdims=keepdims)
    want = np.sum(arr, keepdims=keepdims)
    if not same(got, want):
        print("FAIL", arr.dtype, keepdims, type(got), type(want), np.asarray(got).tobytes().hex(), np.asarray(want).tobytes().hex())
        ok = False

# A native-endian contiguous NaN payload delegates to NumPy so ISA-specific
# SIMD NaN selection cannot alter the observable payload bits.
with_nan = f64.copy()
with_nan[1_048_589] = np.float64(np.nan)
ok = ok and same(fnp.sum(with_nan), np.sum(with_nan))

# Explicit dtype and non-contiguous inputs deliberately fall through unchanged.
view = f32[:, ::2]
ok = ok and same(fnp.sum(f32, dtype=np.float64), np.sum(f32, dtype=np.float64))
ok = ok and same(fnp.sum(view), np.sum(view))
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "parallel flat float sum must be byte-exact with NumPy: {result}"
    );
    Ok(())
}

/// 2^22 f64 elements is the native-admission boundary (it was 1,000,000 until 2026-09-27: a
/// parallel sum that follows serial work loses below 2^22, bead deadlock-audit-vc4p4). Capture
/// NumPy's exact result first, then poison only the module-level fallback callable: the route's
/// tree probe uses `ndarray.sum`, while a fallback through `numpy.sum` must fail. This proves the
/// boundary really executes the SIMD pairwise tree rather than merely comparing two delegated
/// calls - and, just below it, that the call IS numpy's.
#[test]
fn sum_f64_at_the_floor_native_pairwise_path_survives_numpy_sum_poison() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
rng = np.random.default_rng(1_000_003)
a = rng.standard_normal(1 << 22, dtype=np.float64)
a[:8] = [1e300, -1e300, 1.0, -0.0, 3.0, -3.0, 2.0**-53, -2.0**-53]
below = a[: (1 << 22) - 8].copy()
expected = np.sum(a)

def poisoned_sum(*args, **kwargs):
    raise AssertionError("native f64 sum route unexpectedly delegated")

np.sum = poisoned_sum
got = fnp.sum(a)
native = type(got) is type(expected) and got.tobytes() == expected.tobytes()

# A plain exact-ndarray delegation calls `numpy.add.reduce` directly, not `numpy.sum`.
class PoisonedAdd:
    def __getattr__(self, name):
        raise AssertionError("delegated through numpy.add." + name)

np.add = PoisonedAdd()
try:
    fnp.sum(below)
    delegated_below = False
except AssertionError:
    delegated_below = True
print(native, delegated_below)
"#
        .into(),
    );
    assert_eq!(
        numpy_oracle(&script)?,
        "True True",
        "2^22 f64 sum must use the native exact-tree route and remain bit-exact; below it, numpy's"
    );
    Ok(())
}

/// REGRESSION: `add.reduce` folds in a `+0.0` identity, so summing an all-`-0.0`
/// buffer yields `+0.0` in NumPy while a bare pairwise tree yields `-0.0`
/// (because `-0.0 + -0.0 == -0.0`). Sized above the parallel gate so the native
/// route is the one under test. One degenerate input, silently wrong without the
/// identity fold, and invisible to every random-data test.
#[test]
fn sum_float_all_negative_zero_matches_numpy_sign() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
checks = []
for dtype in (np.float64, np.float32):
    n = 4_400_000
    a = np.full(n, dtype(-0.0), dtype=dtype)
    ours = dtype(fnp.sum(a))
    theirs = dtype(a.sum())
    checks.append(ours.tobytes() == theirs.tobytes())
    # sanity: the discriminating condition is real — NumPy really does return +0.0
    checks.append(theirs.tobytes() == dtype(0.0).tobytes())
    checks.append(dtype(0.0).tobytes() != dtype(-0.0).tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 6");
    Ok(())
}

/// The same `+0.0` identity across every native sum-family route, not only the flat parallel sum:
/// numpy's `np.sum` / `np.nansum` (a sum of the NaN -> 0 copy) / `np.mean` / `np.nanmean` of an
/// all-`-0.0` operand are `+0.0` along every axis and size. The row sum, the f64 flat nansum and
/// nanmean, the nanmean lane kernels and the four parallel float16 routes summed a bare pairwise
/// tree and returned `-0.0` (179 cells of a 5,960-cell probe). Sizes span the serial routes and the
/// float16 / float64 parallel floors (2^17-2^22); prod / nanprod pin the sign of a product.
#[test]
fn sum_family_of_all_negative_zero_is_positive_zero_on_every_route() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
import warnings
warnings.simplefilter("ignore")
def res(fn, a, kw):
    try:
        x = np.asarray(fn(a, **kw))
        return (x.dtype.str, x.shape, x.tobytes())
    except Exception as e:
        return (type(e).__name__, str(e))
cells, bad = 0, []
for dt in ("f8", "f4", "f2"):
    for n in (1, 8, 9, 129, 5000, (1 << 19) + 1, (1 << 22) + 3):
        for shape in ((n,), (1, n), (n, 1)):
            a = np.full(shape, -0.0, dtype=dt)
            for name in ("sum", "nansum", "mean", "nanmean", "prod", "nanprod"):
                for kw in ({}, {"axis": 0}, {"axis": -1}, {"keepdims": True}):
                    cells += 1
                    if res(getattr(fnp, name), a, kw) != res(getattr(np, name), a, kw):
                        bad.append((name, dt, shape, kw))
print(cells, bad[:8])
"#
        .to_string(),
    );
    let out = numpy_oracle(&script)?;
    let (cells, bad) = out.trim().split_once(' ').unwrap_or(("0", &out));
    assert_eq!(bad, "[]", "an all -0.0 reduction must carry numpy's sign: {out}");
    assert_eq!(cells, "1512", "cell table drifted: {out}");
    Ok(())
}

/// The parallel float route reproduces ONE specific NumPy reduction tree, and
/// NumPy changed that tree between 2.2.4 and 2.4.2 (same buffer, different last
/// ULP). Whatever NumPy is live here, `fnp.sum` must agree with it bitwise —
/// either because the tree matches and we route, or because the runtime
/// self-check rejected it and we delegated.
#[test]
fn sum_float_flat_agrees_with_whatever_numpy_tree_is_live() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
rng = np.random.default_rng(20260731)
bad = []
for n in [2_200_000, 3_000_007, 8_388_608]:
    a = rng.standard_normal(n)
    if np.float64(fnp.sum(a)).tobytes() != np.float64(a.sum()).tobytes():
        bad.append(n)
    f = rng.standard_normal(2 * n).astype(np.float32)
    if np.float32(fnp.sum(f)).tobytes() != np.float32(f.sum()).tobytes():
        bad.append(('f32', n))
print(len(bad) == 0, bad[:3])
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True []");
    Ok(())
}

/// An explicit `initial=None` is not the omitted default for a reduction with an identity: numpy
/// then starts `add.reduce` / `multiply.reduce` from the FIRST element (`sum([-0.0])` is `-0.0`,
/// a longer float operand sums a different tree) and raises ValueError on an empty operand.
/// pyo3's `Option` folded it into "omitted", so `sum` / `prod` / `nansum` / `nanprod` answered
/// with the identity (76 of these cells). `max` / `min` and their aliases have no identity - None
/// IS their default - and are pinned here so a conversion cannot invent a divergence there.
#[test]
fn explicit_initial_none_matches_numpy_for_every_reduction() -> Result<(), String> {
    let script = fnp_sum_script(
        r#"
import warnings
def outcome(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*args, **kw); a = np.asarray(r)
            res = ("ok", type(r).__name__, a.dtype.str, a.shape,
                   a.tobytes() if a.dtype != object else repr(r))
        except Exception as e:
            res = ("raise", type(e).__name__, str(e))
    return res + (tuple((x.category.__name__, str(x.message)) for x in w),)
rng = np.random.default_rng(20261007)
ops = {"[-0.0]": np.array([-0.0]), "[]": np.array([]), "-0.0 x9": np.full(9, -0.0),
       "2x0": np.zeros((2, 0)), "0x3": np.zeros((0, 3)), "f8 100": rng.standard_normal(100),
       "i8 []": np.array([], np.int64), "i8": np.arange(10), "nan": np.array([np.nan, -0.0]),
       "f4 -0.0": np.full(5, -0.0, np.float32), "big": rng.standard_normal((1 << 22) + 5),
       "obj": np.array([1, 2], object), "2x3": rng.standard_normal((2, 3)),
       "bool": np.array([True, False])}
names = ["sum", "prod", "nansum", "nanprod", "max", "min", "amax", "amin", "nanmax", "nanmin"]
cells, bad = 0, []
for name in names:
    for label, a in ops.items():
        for kw in ({"initial": None}, {"initial": None, "axis": 0}, {"initial": None, "keepdims": True},
                   {}, {"initial": 2}):
            cells += 1
            if outcome(getattr(fnp, name), a, **kw) != outcome(getattr(np, name), a, **kw):
                bad.append((name, label, kw))
print(cells, bad[:8])
"#
        .to_string(),
    );
    let out = numpy_oracle(&script)?;
    let (cells, bad) = out.trim().split_once(' ').unwrap_or(("0", &out));
    assert_eq!(bad, "[]", "initial=None must match numpy: {out}");
    assert_eq!(cells, "700", "cell table drifted: {out}");
    Ok(())
}
