//! Conformance tests for numpy.argmax against NumPy oracle.
//!
//! Tests the native Rust argmax implementation against NumPy across various
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

mod support;

fn fnp_argmax_script(body: String) -> String {
    support::fnp_script(body)
}

fn parse_int(s: &str) -> i64 {
    s.trim().parse::<i64>().unwrap_or(-1)
}

fn parse_int_list(s: &str) -> Vec<i64> {
    if s.is_empty() || s == "[]" {
        return vec![];
    }
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .filter_map(|token| token.parse::<i64>().ok())
        .collect()
}

#[test]
fn argmax_flat_matches_numpy_across_50_cases() -> Result<(), String> {
    let test_cases = vec![
        // Basic arrays - max at various positions
        "[1, 3, 2]",
        "[1, 2, 3, 4, 5]",
        "[5, 4, 3, 2, 1]",
        "[1]",
        "[1, 1, 1, 1]",
        "[0, 0, 0]",
        "[-1, -2, -3]",
        "[-3, -2, -1]",
        "[1, -1, 2, -2, 3, -3]",
        "[100, 500, 200, 400, 300]",
        // Floating point
        "[0.5, 2.5, 1.5]",
        "[1.1, 4.4, 2.2, 3.3]",
        "[0.001, 0.003, 0.002]",
        "[1e10, 3e10, 2e10]",
        "[1e-10, 3e-10, 2e-10]",
        // Negatives and zeros
        "[100, 0, -100]",
        "[-1.5, -0.5, 1.5, 0.5]",
        "[0, 1, 0, 1, 0]",
        "[-5, -4, 0, -2, -1, -3]",
        "[0, -1, -2, -3, -4, -5]",
        // Larger arrays
        "[1, 2, 3, 4, 10, 6, 7, 8, 9, 5]",
        "[1, 9, 8, 7, 6, 5, 4, 3, 2, 10]",
        "[1, 3, 5, 7, 9, 15, 13, 11]",
        "[2, 4, 16, 8, 10, 12, 14, 6]",
        "[1, 21, 2, 3, 5, 8, 13, 0]",
        // Mixed
        "[0.5, 1, 3.0, 2, 2.5, 1.5]",
        "[-2.5, -1.5, -0.5, 2.5, 1.5, 0.5]",
        "[1, 10, 10000, 1000, 100]",
        "[1, 1000, 100, 10, 10000]",
        "[3.14159, 2.71828, 1.41421]",
        // Edge values
        "[0.0, 0.0]",
        "[1.0, 1.5, 1.0, 1.0, 1.0]",
        "[999, -999]",
        "[0.123456789, 0.987654321]",
        "[1, 2]",
        // More variety
        "[7, 3, 9, 1, 5]",
        "[2, 8, 4, 6, 0]",
        "[11, 66, 33, 44, 55, 22]",
        "[33, 88, 77, 66, 55, 44, 99]",
        "[1, 4, 49, 16, 25, 36, 9]",
        // Small ranges
        "[1.0, 1.3, 1.1, 1.2]",
        "[0.99, 1.01, 1.00]",
        "[-0.01, 0.01, 0.0]",
        "[100.0, 100.5, 101.0]",
        "[1000, 1003, 1001, 1002]",
        // Additional cases
        "[5, 15, 25, 45, 35]",
        "[0, 2, 10, 6, 8, 4]",
        "[-10, 10, 0, 5, -5]",
        "[0.1, 0.2, 1.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3]",
        "[1, 3, 2, 6, 3, 5, 4, 3]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.argmax(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script = fnp_argmax_script(format!("print(fnp.argmax(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmax flat mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmax_2d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 2D arrays with axis=0
        ("[[1, 5, 3], [4, 2, 6]]", "0"),
        ("[[4, 1], [2, 5], [3, 6]]", "0"),
        ("[[1, 8], [3, 4], [5, 6], [7, 2]]", "0"),
        ("[[20, 5, 30], [10, 15, 25]]", "0"),
        ("[[1, 3, 2], [3, 1, 2], [2, 2, 3]]", "0"),
        // 2D arrays with axis=1
        ("[[1, 3, 2], [4, 6, 5]]", "1"),
        ("[[1, 4], [5, 2], [3, 6]]", "1"),
        ("[[1, 2], [3, 4], [5, 6], [7, 8]]", "1"),
        ("[[10, 30, 20], [25, 15, 5]]", "1"),
        ("[[1, 9, 5], [2, 10, 6], [3, 11, 7]]", "1"),
        // Negative axis
        ("[[1, 3, 2], [4, 6, 5]]", "-1"),
        ("[[1, 3, 2], [4, 6, 5]]", "-2"),
        ("[[1, 7, 4], [2, 8, 5], [3, 9, 6]]", "-1"),
        ("[[1, 7, 4], [2, 8, 5], [3, 9, 6]]", "-2"),
        // Single row/column
        ("[[1, 5, 3, 2, 4]]", "0"),
        ("[[1, 5, 3, 2, 4]]", "1"),
        ("[[3], [1], [4], [2]]", "0"),
        ("[[3], [1], [4], [2]]", "1"),
        // Floating point 2D
        ("[[0.5, 3.5], [2.5, 1.5]]", "0"),
        ("[[0.5, 3.5], [2.5, 1.5]]", "1"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.argmax(np.array({arr_str}), axis={axis}).tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_int_list(&numpy_result);

        let rust_script = fnp_argmax_script(format!(
            "print(fnp.argmax(np.array({arr_str}), axis={axis}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argmax axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argmax_3d_axis_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        // 3D arrays
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "0"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-1"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-2"),
        ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "-3"),
        // Different shapes
        ("[[[1, 5, 3]], [[4, 2, 6]]]", "0"),
        ("[[[4, 2, 6]], [[1, 5, 3]]]", "1"),
        ("[[[1, 3, 2]], [[4, 6, 5]]]", "2"),
        ("[[[6], [4], [5]], [[1], [3], [2]]]", "0"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "1"),
        ("[[[1], [3], [2]], [[6], [4], [5]]]", "2"),
    ];

    for (arr_str, axis) in &test_cases {
        let script = format!(
            "import numpy as np; print(np.argmax(np.array({arr_str}), axis={axis}).flatten().tolist())"
        );
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_int_list(&numpy_result);

        let rust_script = fnp_argmax_script(format!(
            "print(fnp.argmax(np.array({arr_str}), axis={axis}).flatten().tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_int_list(&rust_result);

        assert_eq!(
            numpy_vals, rust_vals,
            "argmax 3D axis={axis} mismatch for {arr_str}\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn argmax_integer_dtypes_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 3, 2], dtype=np.int32)", "None"),
        ("np.array([1, 3, 2], dtype=np.int64)", "None"),
        ("np.array([1, 3, 2], dtype=np.uint8)", "None"),
        ("np.array([100, 300, 200], dtype=np.int16)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.int32)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.int64)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.float32)", "None"),
        ("np.array([[1, 4], [3, 2]], dtype=np.float64)", "None"),
    ];

    for (arr_expr, axis) in &test_cases {
        let axis_arg = if *axis == "None" {
            String::new()
        } else {
            format!(", axis={axis}")
        };
        let script = format!("import numpy as np; print(int(np.argmax({arr_expr}{axis_arg})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script =
            fnp_argmax_script(format!("print(int(fnp.argmax({arr_expr}{axis_arg})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmax dtype mismatch for {arr_expr} axis={axis}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmax_first_occurrence_matches_numpy() -> Result<(), String> {
    let test_cases = vec![
        "[1, 1, 1]",
        "[1, 2, 2, 1]",
        "[1, 3, 2, 3, 1]",
        "[5, 5, 5, 5, 5]",
        "[[3, 2], [3, 1]]",
    ];

    for arr_str in &test_cases {
        let script = format!("import numpy as np; print(np.argmax(np.array({arr_str})))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val = parse_int(&numpy_result);

        let rust_script = fnp_argmax_script(format!("print(fnp.argmax(np.array({arr_str})))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val = parse_int(&rust_result);

        assert_eq!(
            numpy_val, rust_val,
            "argmax first occurrence mismatch for {arr_str}\nnumpy: {numpy_val}\nrust: {rust_val}"
        );
    }

    Ok(())
}

#[test]
fn argmax_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.argmax(x)
np_result = np.argmax(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "argmax scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn argmax_complex() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
a = np.array([1+1j, 3-1j, 2+2j], dtype=np.complex128)
fnp_result = fnp.argmax(a)
np_result = np.argmax(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argmax complex should match numpy");
    Ok(())
}

#[test]
fn argmax_with_nan() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
# NaN handling - numpy returns index of NaN as "max"
a = np.array([1.0, np.nan, 3.0])
fnp_result = fnp.argmax(a)
np_result = np.argmax(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argmax nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn argmax_all_nan() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
a = np.array([np.nan, np.nan, np.nan])
fnp_result = fnp.argmax(a)
np_result = np.argmax(a)
print(fnp_result == np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "argmax all-nan should match numpy");
    Ok(())
}

#[test]
fn argmax_empty_array_raises_valueerror() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
empty = np.array([])
fnp_raised = False
np_raised = False
try:
    fnp.argmax(empty)
except ValueError:
    fnp_raised = True
except Exception:
    pass
try:
    np.argmax(empty)
except ValueError:
    np_raised = True
except Exception:
    pass
print(fnp_raised == np_raised == True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "argmax of empty array should raise ValueError"
    );
    Ok(())
}

/// The int arg-extreme kernels (bead `deadlock-audit-vc4p4`): `first_argextreme_blocked` takes
/// each 256-element block's extreme and rescans only the earliest block holding the overall one;
/// int64 lanes are native from 2^22 elements (parallel, plus one serial single-row lane), and the
/// flat route scans >= 2 MiB bands in parallel from 64 MiB, combining bands left to right. A wrong
/// block or band combine returns a LATER tie: rows 0 and 1 below hold their extreme in three
/// blocks, and the flat runs in four of their bands, so only the first-occurrence index passes.
/// The flat run must be native above its floor and numpy's own call below it.
#[test]
fn argmax_argmin_int_blocks_and_bands_keep_the_first_occurrence() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
import os
rng = np.random.default_rng(5)
bad = []
def same(label, ours, theirs):
    o, t = np.asarray(ours), np.asarray(theirs)
    if o.dtype != t.dtype or o.shape != t.shape or o.tobytes() != t.tobytes():
        bad.append(label)
lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
for rows, lane in ((1 << 22, 1), (599187, 7), (16452, 255), (16387, 256), (16323, 257),
                   (8180, 513), (1027, 4096), (1, (1 << 22) + 5)):
    for gname, x in (("ties", rng.integers(0, 3, (rows, lane))),
                     ("full", rng.integers(lo, hi, (rows, lane), endpoint=True))):
        same(f"argmax lane={lane} {gname}", fnp.argmax(x, axis=1), np.argmax(x, axis=1))
        same(f"argmin lane={lane} {gname}", fnp.argmin(x, axis=-1), np.argmin(x, axis=-1))
x = np.zeros((4096, 1024), dtype=np.int64)
x[:, 200] = x[:, 600] = x[:, 900] = 9
x[1::2, 300] = x[1::2, 800] = 10
got = fnp.argmax(x, axis=1)
same("cross-block argmax", got, np.argmax(x, axis=1))
if (got[0], got[1]) != (200, 300):
    bad.append(f"cross-block argmax rows 0/1 -> {got[0]}, {got[1]}")
got = fnp.argmin(-x, axis=1)
same("cross-block argmin", got, np.argmin(-x, axis=1))
if (got[0], got[1]) != (200, 300):
    bad.append(f"cross-block argmin rows 0/1 -> {got[0]}, {got[1]}")
for dt in (np.int64, np.int32, np.uint64):
    size = np.dtype(dt).itemsize
    n = (64 << 20) // size + 13
    x = np.full(n, 5, dtype=dt)
    x[[200, (2 << 20) // size + 3, n // 2, n - 1]] = 9
    x[[150, n // 3, n - 2]] = 1
    got_max, got_min = int(fnp.argmax(x)), int(fnp.argmin(x))
    if (got_max, got_min) != (200, 150):
        bad.append(f"flat {np.dtype(dt).name} cross-band -> {got_max}, {got_min}")
    r = rng.integers(0, 3, n).astype(dt)
    same(f"flat argmax {np.dtype(dt).name} ties", fnp.argmax(r), np.argmax(r))
    same(f"flat argmin {np.dtype(dt).name} ties", fnp.argmin(r), np.argmin(r))
big = rng.integers(-1000, 1000, 1 << 22).astype(">i8")
same("flat argmax >i8", fnp.argmax(big), np.argmax(big))
real = np.argmax
def poisoned(*args, **kwargs):
    raise LookupError("numpy.argmax")
np.argmax = poisoned
try:
    if (os.cpu_count() or 1) >= 2:
        try:
            fnp.argmax(np.arange((64 << 20) // 8 + 1))
        except LookupError:
            bad.append("int64 at 64 MiB went to numpy")
    try:
        fnp.argmax(np.arange(1 << 23, dtype=np.int32))
        bad.append("int32 at 32 MiB stayed native")
    except LookupError:
        pass
finally:
    np.argmax = real
print("OK" if not bad else bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "OK",
        "int arg-extreme blocks / bands: {result}"
    );
    Ok(())
}

/// argmax / argmin along a SHORT contiguous last axis (2-16 elements; `try_small_lane_argextreme`):
/// numpy's first-occurrence tie rule, the FIRST NaN winning for both argmax and argmin, signed
/// zeros comparing equal, every integer width, serial and pooled (a 32 MiB operand). The negative
/// cases a naive scan gets wrong: a NaN after a larger value (argmax must still return the NaN),
/// -0.0 before 0.0 (argmax must keep index 0), and a tie at the lane end. The delegate for these
/// calls is the ndarray METHOD, which cannot be poisoned, so this pins parity of whatever route
/// is live; the route's engagement is the timing in its ledger row.
#[test]
fn argextreme_short_last_axis_matches_numpy() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
rng = np.random.default_rng(20261008)
bad = []
count = 0
def check(fname, a, **kw):
    global count
    count += 1
    ours, theirs = getattr(fnp, fname)(a, **kw), getattr(np, fname)(a, **kw)
    o, t = np.asarray(ours), np.asarray(theirs)
    if type(ours) is not type(theirs) or o.dtype != t.dtype or o.shape != t.shape or o.tobytes() != t.tobytes():
        bad.append((fname, a.dtype.name, a.shape))
for lane in [2, 3, 8, 16, 17]:
    for dtype in [np.float32, np.float64]:
        x = rng.standard_normal((4100, lane)).astype(dtype)
        x[rng.random(x.shape) < 0.1] = np.nan
        z = np.where(rng.random((4100, lane)) < 0.5, -0.0, 0.0).astype(dtype)
        r = np.round(rng.standard_normal((4100, lane))).astype(dtype)
        for a in [x, z, r]:
            check("argmax", a, axis=-1); check("argmin", a, axis=1)
    for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
        info = np.iinfo(dtype)
        a = rng.integers(info.min, info.max, (4100, lane), dtype=dtype, endpoint=True)
        check("argmax", a, axis=-1); check("argmin", a, axis=-1)
witness = np.tile(np.array([[1.0, 5.0, np.nan], [-0.0, 0.0, -1.0], [2.0, 1.0, 2.0]]), (1400, 1))
check("argmax", witness, axis=-1); check("argmin", witness, axis=-1)
pooled = rng.integers(0, 256, (1 << 22, 8), dtype=np.uint8)
check("argmax", pooled, axis=-1); check("argmin", pooled, axis=-1)
print(bad if bad else True, count)
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 144");
    Ok(())
}

/// Flat bool argmax / argmin (`np.argmax(mask)`, find the first True or False). The native scan
/// tests 64-byte blocks whole and walks only the block that holds the hit, so the hits sit on
/// both sides of block edges, the masks carry non-canonical True bytes (2 and 255 made through a
/// uint8 view, which numpy reads as True), a second hit follows the first, and a mask with no hit
/// answers 0 as numpy does.
#[test]
fn argmax_argmin_flat_bool_find_the_first_true_and_first_false() -> Result<(), String> {
    let script = fnp_argmax_script(
        r#"
checks = []
def same(ours, theirs, expected):
    return type(ours) is type(theirs) and int(ours) == int(theirs) == expected
for n in [1, 63, 64, 65, 1000, (1 << 22) + 17]:
    for pos in sorted({0, 1, 63, 64, 65, n // 2, n - 1}):
        if pos >= n:
            continue
        raw = np.zeros(n, dtype=np.uint8); raw[pos] = 2
        if pos + 70 < n:
            raw[pos + 70] = 1
        mask = raw.view(bool)
        checks.append(same(fnp.argmax(mask), np.argmax(mask), pos))
        raw = np.full(n, 255, dtype=np.uint8); raw[pos] = 0
        mask = raw.view(bool)
        checks.append(same(fnp.argmin(mask), np.argmin(mask), pos))
    none_true, none_false = np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    checks.append(same(fnp.argmax(none_true), np.argmax(none_true), 0))
    checks.append(same(fnp.argmin(none_false), np.argmin(none_false), 0))
grid = np.zeros((300, 700), dtype=bool); grid[123, 456] = True
checks.append(same(fnp.argmax(grid), np.argmax(grid), 123 * 700 + 456))
print(all(checks), len(checks))
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 69");
    Ok(())
}
