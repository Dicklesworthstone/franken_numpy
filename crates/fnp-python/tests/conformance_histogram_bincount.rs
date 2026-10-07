//! Conformance tests for numpy histogram and bincount functions against NumPy oracle.
//!
//! Tests histogram, histogram_bin_edges, bincount, digitize.

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
use support::fnp_script;

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

fn numpy_outcome_script(function_expr: &str, setup: &str, call_expr: &str) -> String {
    format!(
        "import numpy as np\nop = {function_expr}\n{}",
        outcome_body(setup, call_expr)
    )
}

fn fnp_outcome_script(function_name: &str, setup: &str, call_expr: &str) -> String {
    fnp_script(format!(
        "op = fnp.{function_name}\n{}",
        outcome_body(setup, call_expr)
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// histogram
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a)
np_hist, np_edges = np.histogram(a)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram basic should match numpy");
    Ok(())
}

#[test]
fn histogram_with_bins() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a, bins=5)
np_hist, np_edges = np.histogram(a, bins=5)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram with bins should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_with_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
hist, edges = fnp.histogram(a, bins=5, range=(2, 8))
np_hist, np_edges = np.histogram(a, bins=5, range=(2, 8))
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram with range should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_with_explicit_edges() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
bin_edges = np.array([1, 2, 3, 4, 5])
hist, edges = fnp.histogram(a, bins=bin_edges)
np_hist, np_edges = np.histogram(a, bins=bin_edges)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram with explicit edges should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_density() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, edges = fnp.histogram(a, density=True)
np_hist, np_edges = np.histogram(a, density=True)
print(np.allclose(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram density should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bincount
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bincount_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount basic should match numpy");
    Ok(())
}

#[test]
fn bincount_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
w = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
result = fnp.bincount(a, weights=w)
expected = np.bincount(a, weights=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount with weights should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_with_minlength() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2])
result = fnp.bincount(a, minlength=5)
expected = np.bincount(a, minlength=5)
print(np.array_equal(result, expected) and len(result) == 5)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount with minlength should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=int)
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bincount empty should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// digitize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_bin_edges_python_container_keyword_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        ("list data with int bins", "", "op([0, 1, 2, 3], bins=4)"),
        (
            "tuple data with explicit edge list",
            "",
            "op((0.2, 0.8, 1.4), bins=[0.0, 0.5, 1.0, 1.5])",
        ),
        (
            "range and weights keywords",
            "",
            "op([0, 1, 2, 3], bins=3, range=(0, 3), weights=[1, 2, 3, 4])",
        ),
        (
            "auto bin estimator fallback",
            "",
            "op([0.0, 0.5, 1.0, 1.5, 2.0], bins='auto')",
        ),
        (
            "invalid range error type",
            "",
            "op([0, 1], bins=3, range=(2, 1))",
        ),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script(
            "np.histogram_bin_edges",
            setup,
            call_expr,
        ))?;
        let rust_result =
            numpy_oracle(&fnp_outcome_script("histogram_bin_edges", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "histogram_bin_edges Python-container keyword surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn digitize_python_container_keyword_surfaces_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list x with tuple bins",
            "",
            "op([0.2, 1.5, 2.3], (1.0, 2.0, 3.0))",
        ),
        (
            "tuple x with right keyword",
            "",
            "op((1, 2, 3), [1, 2, 3], right=True)",
        ),
        ("scalar x output", "", "op(np.float64(2.5), [1, 2, 3, 4])"),
        (
            "decreasing bins with right keyword",
            "",
            "op([0.5, 1.5, 3.5], [4, 3, 2, 1], right=True)",
        ),
        ("nonmonotonic bins error type", "", "op([1, 2], [0, 2, 1])"),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script("np.digitize", setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script("digitize", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "digitize Python-container keyword surface mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn digitize_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 0.8, 1.5, 2.3, 3.8, 5.0])
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize basic should match numpy");
    Ok(())
}

#[test]
fn digitize_right() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4])
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins, right=True)
expected = np.digitize(x, bins, right=True)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize right should match numpy");
    Ok(())
}

#[test]
fn digitize_decreasing() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 0.8, 1.5, 2.3, 3.8, 5.0])
bins = np.array([4, 3, 2, 1])  # decreasing
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize decreasing bins should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_parallel_large_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = (1 << 21) + 4096
shape = (2052, 1024)

def check(values, bins, right):
    result = fnp.digitize(values, bins, right=right)
    expected = np.digitize(values, bins, right=right)
    return (
        result.dtype == expected.dtype
        and result.shape == expected.shape
        and result.flags.c_contiguous
        and result.tobytes() == expected.tobytes()
    )

ok = True

x64 = np.linspace(-5.0, 5.0, n, dtype=np.float64).reshape(shape)
x64_flat = x64.ravel()
x64_flat[0] = np.nan
x64_flat[97] = np.inf
x64_flat[211] = -np.inf
x64_flat[4096] = -0.0
bins64 = np.linspace(-4.0, 4.0, 50, dtype=np.float64)
ok = ok and check(x64, bins64, False)
ok = ok and check(x64, bins64, True)

x32 = x64.astype(np.float32, copy=True)
bins32 = bins64.astype(np.float32)
ok = ok and check(x32, bins32, False)
ok = ok and check(x32, bins32, True)

i64 = ((np.arange(n, dtype=np.int64) % 1000) - 500).reshape(shape)
ibins = np.array([-500, -1, 0, 0, 1, 499], dtype=np.int64)
ok = ok and check(i64, ibins, False)
ok = ok and check(i64, ibins, True)

u16 = (np.arange(n, dtype=np.uint32) % 1000).astype(np.uint16).reshape(shape)
ubins = np.array([0, 1, 1, 500, 999], dtype=np.uint16)
ok = ok and check(u16, ubins, False)
ok = ok and check(u16, ubins, True)

dec_x = np.array([0.5, 1.5, 3.5], dtype=np.float64)
dec_bins = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
ok = ok and check(dec_x, dec_bins, False)
ok = ok and check(dec_x, dec_bins, True)

print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "large digitize parallel path should match numpy bit-exactly"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
hist, _ = fnp.histogram(a)
print(np.sum(hist) == len(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram sum should equal array length"
    );
    Ok(())
}

#[test]
fn bincount_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = fnp.bincount(a)
print(np.sum(result) == len(a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount sum should equal array length"
    );
    Ok(())
}

#[test]
fn digitize_searchsorted_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.2, 1.5, 2.8, 4.2])
bins = np.array([1, 2, 3, 4])
digitize_result = fnp.digitize(x, bins)
searchsorted_result = fnp.searchsorted(bins, x, side='right')
print(np.array_equal(digitize_result, searchsorted_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize should equal searchsorted for increasing bins"
    );
    Ok(())
}

#[test]
fn digitize_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(2.5)
bins = np.array([1, 2, 3, 4])
fnp_result = fnp.digitize(x, bins)
np_result = np.digitize(x, bins)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "digitize scalar return type should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
hist, edges = fnp.histogram(a, bins=5)
np_hist, np_edges = np.histogram(a, bins=5)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogram empty should match numpy");
    Ok(())
}

#[test]
fn histogram_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0])
hist, edges = fnp.histogram(a, bins=3)
np_hist, np_edges = np.histogram(a, bins=3)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram single element should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_all_same_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
hist, edges = fnp.histogram(a)
np_hist, np_edges = np.histogram(a)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram all same value should match numpy"
    );
    Ok(())
}

#[test]
fn histogram_edge_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
hist, edges = fnp.histogram(a, bins=bins)
np_hist, np_edges = np.histogram(a, bins=bins)
print(np.array_equal(hist, np_hist) and np.allclose(edges, np_edges))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram edge values should match numpy"
    );
    Ok(())
}

/// Locks the typed uniform-bin histogram fast path to NumPy's raw output bytes:
/// int64 counts, f32 edges for f32 inputs, f64 edges for f64 and supported
/// integer inputs, and fallback/error parity for unsupported cases.
#[test]
fn histogram_typed_uniform_bins_bit_exact_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
chunks = []

def capture(func):
    try:
        hist, edges = func()
    except Exception as exc:
        return ("E", type(exc).__name__, None, None)
    return ("O", None, np.asarray(hist), np.asarray(edges))

def record(label, values, bins=10, **kwargs):
    got_kind, got_exc, got_hist, got_edges = capture(
        lambda: fnp.histogram(values, bins=bins, **kwargs)
    )
    exp_kind, exp_exc, exp_hist, exp_edges = capture(
        lambda: np.histogram(values, bins=bins, **kwargs)
    )
    assert got_kind == exp_kind, (label, got_kind, exp_kind)
    chunks.append(label.encode())
    chunks.append(b'\0')
    if got_kind == "E":
        assert got_exc == exp_exc, (label, got_exc, exp_exc)
        chunks.append(b'E')
        chunks.append(got_exc.encode())
        return
    assert got_hist.dtype == exp_hist.dtype, (label, got_hist.dtype, exp_hist.dtype)
    assert got_edges.dtype == exp_edges.dtype, (label, got_edges.dtype, exp_edges.dtype)
    assert got_hist.shape == exp_hist.shape, (label, got_hist.shape, exp_hist.shape)
    assert got_edges.shape == exp_edges.shape, (label, got_edges.shape, exp_edges.shape)
    assert got_hist.tobytes() == exp_hist.tobytes(), (
        label,
        got_hist.tolist(),
        exp_hist.tolist(),
    )
    assert got_edges.tobytes() == exp_edges.tobytes(), (
        label,
        got_edges.tolist(),
        exp_edges.tolist(),
    )
    chunks.append(b'O')
    chunks.append(got_hist.dtype.str.encode())
    chunks.append(str(got_hist.shape).encode())
    chunks.append(got_hist.tobytes())
    chunks.append(got_edges.dtype.str.encode())
    chunks.append(str(got_edges.shape).encode())
    chunks.append(got_edges.tobytes())

for dtype in (np.float32, np.float64):
    record(f'{dtype.__name__}:empty', np.array([], dtype=dtype), bins=5)
    record(f'{dtype.__name__}:same', np.array([7, 7, 7], dtype=dtype), bins=5)
    record(f'{dtype.__name__}:linear', np.linspace(-1000, 1000, 10000, dtype=dtype), bins=50)
    record(f'{dtype.__name__}:edges',
           np.array([-1, -0.8, -0.2, 0, 0.2, 0.8, 1], dtype=dtype), bins=5)

for dtype in (np.int8, np.int16, np.int32, np.int64):
    record(f'{dtype.__name__}:signed', ((np.arange(1000) % 37) - 13).astype(dtype), bins=11)

for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
    record(f'{dtype.__name__}:unsigned', (np.arange(1000) % 37).astype(dtype), bins=11)

record('int64:exact-boundary', np.array([-2**53, -1, 0, 2**53], dtype=np.int64), bins=4)
record('uint64:exact-boundary', np.array([0, 1, 2**32, 2**53], dtype=np.uint64), bins=4)
record('int64:large-error', np.array([2**60, 2**60 + 3], dtype=np.int64), bins=5)
record('float32:nonfinite-error', np.array([1, np.inf], dtype=np.float32), bins=5)
record('float32:strided-defer', np.arange(20, dtype=np.float32)[::2], bins=5)
record('float32:range-defer', np.arange(20, dtype=np.float32), bins=5, range=(2, 18))
record('float32:density-defer', np.arange(20, dtype=np.float32), bins=5, density=True)

print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#
        .into(),
    );
    let hash = numpy_oracle(&script)?;
    assert_eq!(
        hash, "dca1ab4a9b56fc672a88e951bcd68f25b8db593ee67dfd5364f17214b39f5739",
        "typed uniform-bin histogram must be bit-identical to numpy (sha256 of dtype/shape/raw output bytes)"
    );
    Ok(())
}

#[test]
fn bincount_zeros_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 0, 0, 0, 0])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount zeros only should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_single_large_value() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([100])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected) and len(result) == 101)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount single large value should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_sparse_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 50, 100])
result = fnp.bincount(a)
expected = np.bincount(a)
print(np.array_equal(result, expected) and result[0] == 1 and result[50] == 1 and result[100] == 1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount sparse values should match numpy"
    );
    Ok(())
}

#[test]
fn bincount_with_zero_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0, 1, 2, 3])
w = np.array([1.0, 0.0, 0.0, 1.0])
result = fnp.bincount(a, weights=w)
expected = np.bincount(a, weights=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bincount with zero weights should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_empty() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([], dtype=np.float64)
bins = np.array([1, 2, 3, 4])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected) and result.shape == expected.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "digitize empty should match numpy");
    Ok(())
}

#[test]
fn digitize_single_bin() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
bins = np.array([1.0])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize single bin should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_exact_matches() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0, 4.0])
bins = np.array([1.0, 2.0, 3.0, 4.0])
result_left = fnp.digitize(x, bins, right=False)
result_right = fnp.digitize(x, bins, right=True)
expected_left = np.digitize(x, bins, right=False)
expected_right = np.digitize(x, bins, right=True)
print(np.array_equal(result_left, expected_left) and np.array_equal(result_right, expected_right))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize exact matches should match numpy"
    );
    Ok(())
}

#[test]
fn digitize_inf_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.inf, 0.0, np.inf])
bins = np.array([1.0, 2.0, 3.0])
result = fnp.digitize(x, bins)
expected = np.digitize(x, bins)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "digitize inf values should match numpy"
    );
    Ok(())
}

/// Locks the zero-copy bincount fast path (`try_zerocopy_bincount`, the 1-D
/// non-negative int64 no-weights case that tallies the buffer directly into an
/// int64 output) to bit-exact parity with numpy, including the int64 result
/// dtype and the minlength-driven output length. Compares the sha256 of raw
/// output bytes across sparse and dense ranges and explicit minlength.
#[test]
fn bincount_zerocopy_int64_bit_exact_matches_numpy() -> Result<(), String> {
    let body = r#"
import hashlib
mod = MODULE
rng = np.random.default_rng(20260605)
chunks = []
for n in [1000, 100003]:
    x = rng.integers(0, 500, n)
    out = np.asarray(mod.bincount(x))
    chunks.append(bytes([1 if out.dtype == np.int64 else 0]))
    chunks.append(out.tobytes())
    chunks.append(np.asarray(mod.bincount(x, minlength=1000)).tobytes())
chunks.append(np.asarray(mod.bincount(np.array([0, 5, 5, 2, 9, 0], dtype=np.int64))).tobytes())
chunks.append(np.asarray(mod.bincount(np.array([], dtype=np.int64), minlength=10)).tobytes())
print(hashlib.sha256(b''.join(chunks)).hexdigest())
"#;

    let fnp_hash = numpy_oracle(&fnp_script(body.replace("MODULE", "fnp")))?;
    let numpy_hash = numpy_oracle(&format!(
        "import numpy as np\n{}",
        body.replace("MODULE", "np")
    ))?;

    assert_eq!(
        fnp_hash, numpy_hash,
        "zero-copy bincount must be bit-identical to numpy (sha256 of raw output bytes)"
    );
    Ok(())
}

/// numpy's bincount converts a SEQUENCE with an intp target, so an EMPTY sequence is an empty
/// int64 count (length `minlength`), not the float64 safe-cast TypeError fnp raised - `asarray`
/// makes `[]` float64 (numpy's own TestBincount::test_empty_list). A float LIST is
/// deprecated-but-accepted on numpy 2.1+ (DeprecationWarning, then truncation) while a float
/// ARRAY raises; the outcome, warnings included, must be whatever the live numpy does.
#[test]
fn bincount_empty_sequence_is_an_empty_int_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = fn()
            value = ("ok", type(r).__name__, str(r.dtype), r.tolist())
        except Exception as exc:
            value = ("err", type(exc).__name__)
    return value, sorted(c.category.__name__ for c in caught)
cases = [
    lambda m: m.bincount([]),
    lambda m: m.bincount([], minlength=3),
    lambda m: m.bincount((), weights=[]),
    lambda m: m.bincount([1.0, 2.0]),
    lambda m: m.bincount(np.array([1.0, 2.0])),
    lambda m: m.bincount([0, 2, 2]),
    # A list of strings, bytes or objects is numpy's per-element conversion (numpy's
    # TestBincount::test_bad_list: ['0', '1', '1'] is [1 2] with a DeprecationWarning); a str
    # ARRAY is its safe-cast TypeError. fnp raised its own TypeError for all four, so the three
    # list cells failed on a8d9a337.
    lambda m: m.bincount(["0", "1", "1"]),
    lambda m: m.bincount([b"1"]),
    lambda m: m.bincount([None]),
    lambda m: m.bincount(np.array(["0", "1"])),
]
bad = [i for i, c in enumerate(cases) if outcome(lambda: c(fnp)) != outcome(lambda: c(np))]
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.lines().last().unwrap_or("").trim(),
        "True",
        "bincount on empty sequences must match numpy: {result}"
    );
    Ok(())
}

/// The serial uniform-bin tally (below the 2^21 parallel gate) takes its ±1 edge corrections
/// branch-free against a +inf sentinel above the last bin, which is only right if a value on an
/// edge, on the last edge, or in the first / last bin lands where numpy puts it. The grid holds
/// every integer width and float64 over flat data (first and last bins each take 1/bins of the
/// values), normal data, values sitting exactly on the edges, a constant array (numpy widens the
/// range by 0.5), two values, and bins 1 / 2 / 10 / 100 / 1000, at sizes on both sides of the
/// parallel gate. Plus the ranges numpy raises on: a width that overflows (`[-1e308, 1e308]`
/// with bins=1 is numpy's IndexError; it was answered with counts by the extract path, for an
/// ndarray and a list alike), a range too narrow for the bins, >2^53 integers, inf and NaN.
/// Counts, edges, dtypes and raise type must be numpy's.
#[test]
fn histogram_uniform_bins_edge_placement_grid_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
def outcome(f):
    try:
        c, e = f()
        c, e = np.asarray(c), np.asarray(e)
        return ("ok", c.dtype.str, c.tobytes(), e.dtype.str, e.tobytes())
    except BaseException as ex:
        return (type(ex).__name__, str(ex)[:80])
rng = np.random.default_rng(37)
def data_for(dt, n, kind):
    if kind == "flat":
        if np.dtype(dt).kind == "f":
            return rng.uniform(-3, 3, n).astype(dt)
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, n, endpoint=True, dtype=dt)
    if kind == "normal":
        v = rng.standard_normal(n) * 50
        if np.dtype(dt).kind == "f":
            return v.astype(dt)
        return np.clip(v, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
    if kind == "const":
        return np.full(n, 7, dtype=dt)
    if kind == "two":
        return np.where(rng.integers(0, 2, n) == 0, 0, 100).astype(dt)
    return (rng.integers(0, 11, n) * 10).astype(dt)
cells = 0
bad = []
for dt in ("f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"):
    for n in (1, 2, 17, 4096, (1 << 21) + 5):
        for kind in ("flat", "normal", "const", "two", "on_edges"):
            if n > 4096 and kind in ("const", "two"):
                continue
            a = data_for(dt, n, kind)
            for bins in (1, 2, 10, 100, 1000):
                if n > 4096 and bins not in (10, 1000):
                    continue
                cells += 1
                ours = outcome(lambda: fnp.histogram(a, bins=bins))
                theirs = outcome(lambda: np.histogram(a, bins=bins))
                if ours != theirs:
                    bad.append(f"{dt} n={n} {kind} bins={bins}: fnp={ours[:2]} numpy={theirs[:2]}")
specials = {
    "huge range": np.array([-1e308, 1e308]),
    "huge range list": [-1e308, 1e308],
    "tiny range": np.array([1.0, 1.0 + 2**-50, 1.0 + 2**-49]),
    "signed zeros": np.array([0.0, -0.0, -0.0, 0.0, 1.0, -1.0]),
    "i8 above 2^53": np.array([0, 2**53 + 1], dtype=np.int64),
    "u8 extremes": np.array([0, 2**64 - 1], dtype=np.uint64),
    "inf": np.array([1.0, np.inf, 2.0]),
    "nan": np.array([1.0, np.nan, 2.0]),
}
for name, a in specials.items():
    for bins in (1, 3, 10):
        cells += 1
        ours = outcome(lambda: fnp.histogram(a, bins=bins))
        theirs = outcome(lambda: np.histogram(a, bins=bins))
        if ours != theirs:
            bad.append(f"{name} bins={bins}: fnp={ours[:2]} numpy={theirs[:2]}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "978",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "histogram must place every value in numpy's bin: {result}"
    );
    Ok(())
}

/// The uniform-bin kernel (`uniform_bin_range` + `tally_uniform_bins`): a SIMD range scan whose
/// choice between equal zeros is unspecified, and a tally that computes linspace's edges as
/// `k * step + first` four lanes at a time through 512-element blocks. numpy's range is
/// `a.min()` / `a.max()`, whose signed zero follows its own lane order: the max of
/// `[0.0, -0.0, -1.0]` is -0.0, which the former first-met scan answered 0.0 (4 of these zero
/// cells failed on 022b485a). The tails cover every remainder of the 4-lane quads and the block
/// boundaries; 4096 / 4097 / 100000 bins straddle the four-row tally and its one-row form; the
/// subnormal ranges underflow linspace's step, so its edges are not `k * step + first` and the
/// call must come out as numpy's own.
#[test]
fn histogram_signed_zero_extremes_and_kernel_tails_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
def outcome(f):
    try:
        c, e = f()
        c, e = np.asarray(c), np.asarray(e)
        return ("ok", c.dtype.str, c.tobytes(), e.dtype.str, e.tobytes())
    except BaseException as ex:
        return (type(ex).__name__, str(ex)[:80])
cells = 0
bad = []
def check(label, a, bins):
    global cells
    cells += 1
    ours = outcome(lambda: fnp.histogram(a, bins=bins))
    theirs = outcome(lambda: np.histogram(a, bins=bins))
    if ours != theirs:
        bad.append(f"{label} bins={bins}: fnp={ours[:2]} numpy={theirs[:2]}")
# (case, fill, [(index, zero), ...] applied in order); a slice index sets every third element.
zero_cases = (
    ("neg_first", 1.0, [(0, -0.0), ("half", 0.0)]),
    ("pos_first", 1.0, [(0, 0.0), ("half", -0.0)]),
    ("neg_last", 1.0, [("last", -0.0), (1, 0.0)]),
    ("pos_last", 1.0, [("last", 0.0), (1, -0.0)]),
    ("mixed_min", 1.0, [(slice(0, None, 3), -0.0), (slice(1, None, 3), 0.0)]),
    ("mixed_max", -1.0, [(slice(0, None, 3), 0.0), (slice(1, None, 3), -0.0)]),
)
for n in (2, 3, 7, 16, 33, 1000, 1 << 21, (1 << 21) + 5):
    for case, fill, zeros in zero_cases:
        a = np.full(n, fill)
        for index, zero in zeros:
            a[{"half": n // 2, "last": n - 1}[index] if isinstance(index, str) else index] = zero
        for bins in (1, 10):
            check(f"zeros n={n} {case}", a, bins)
rng = np.random.default_rng(1004)
for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 511, 512, 513, 1027):
    for kind in ("uniform", "on_edges"):
        if kind == "uniform":
            a = rng.uniform(-2.5, 7.5, n)
        else:
            a = np.linspace(-2.5, 7.5, 41)[rng.integers(0, 41, n)]
        for bins in (3, 10, 4096, 4097, 100000):
            check(f"tail n={n} {kind}", a, bins)
for name, a in {
    "subnormal pair": np.array([0.0, 2.5e-323]),
    "subnormal run": np.arange(8) * 5e-324,
    "subnormal offset": np.array([1e-320, 1e-320 + 2.5e-323, 1e-320 + 1e-323]),
    "near-equal normals": np.array([1.0, np.nextafter(1.0, 2.0), np.nextafter(np.nextafter(1.0, 2.0), 2.0)]),
}.items():
    for bins in (1, 2, 3, 10):
        check(name, a, bins)
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "242",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "histogram must take numpy's range, edges and bins: {result}"
    );
    Ok(())
}

/// `histogram` of 1- and 2-byte integers (`try_narrow_integer_histogram`): fnp counts each
/// distinct value and hands numpy only the occurring values with int64 weights, so numpy's own bin
/// arithmetic, auto range, edges and density decide the answer. Covers integer / explicit / float
/// edges, range incl. degenerate and invalid ones (numpy's errors), density, images, and the
/// declines (estimator strings, weights=, low compression). numpy.histogram is wrapped to prove the
/// route hands numpy the COMPRESSED operand rather than the data.
#[test]
fn histogram_narrow_integer_counts_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(20261003)
bad = []
count = 0
def outcome(fn, a, kw):
    try:
        return fn(a, **kw)
    except Exception as ex:
        return (type(ex).__name__, str(ex))
def check(a, **kw):
    global count
    count += 1
    ours, theirs = outcome(fnp.histogram, a, kw), outcome(np.histogram, a, kw)
    if isinstance(ours[0], str) or isinstance(theirs[0], str):
        ok = ours == theirs
    else:
        ok = all(np.asarray(o).dtype == np.asarray(t).dtype and np.asarray(o).tobytes() == np.asarray(t).tobytes()
                 for o, t in zip(ours, theirs))
    if not ok:
        bad.append((a.dtype.name, a.shape, str(kw)[:60]))
datas = [rng.integers(0, 256, (120, 160, 3), dtype=np.uint8), rng.integers(-128, 128, 20000).astype(np.int8),
         rng.integers(0, 4096, (128, 256)).astype(np.uint16), rng.integers(-1000, 1000, 1 << 16).astype(np.int16),
         np.full(20000, 7, dtype=np.uint8)]
for a in datas:
    for kw in [{}, {"bins": 256}, {"bins": 256, "range": (0, 256)}, {"bins": 7, "range": (10, 200)},
               {"bins": 3, "range": (0.5, 255.5)}, {"bins": 4, "range": (5, 5)}, {"bins": 4, "range": (200, 10)},
               {"bins": np.arange(257)}, {"bins": np.array([-5.5, 0.25, 99.9, 300.0])},
               {"bins": 256, "range": (0, 256), "density": True}, {"bins": "auto"}, {"bins": 16, "weights": np.ones(a.shape)}]:
        check(a, **kw)
wide = rng.integers(-32768, 32767, 20000).astype(np.int16)
check(wide, bins=64)
seen = []
real_histogram = np.histogram

def spy(a, *args, **kwargs):
    seen.append(np.asarray(a).size)
    return real_histogram(a, *args, **kwargs)

np.histogram = spy
gray = rng.integers(0, 256, (400, 500), dtype=np.uint8)
fnp.histogram(gray, bins=256, range=(0, 256))
compressed = seen == [256]
print(bad if bad else True, count, compressed)
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 61 True");
    Ok(())
}

/// Weighted `bincount` natively for every integer `x` numpy casts safely to intp (int8..int64,
/// uint8..uint32), and numpy's own call for everything else - values, dtype, AND its exact errors.
/// The negative cases the former float64 extract tail got wrong: a longdouble `weights` (numpy's
/// safe-cast TypeError; the tail answered in float64), a negative int8 `x` and a 2-D `weights`
/// (the tail raised with its own messages).
#[test]
fn bincount_weighted_integer_widths_and_numpy_errors() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(20261011)
bad = []
count = 0
def outcome(fn, args, kw):
    try:
        r = fn(*args, **kw)
        return ("ok", r.dtype.name, r.shape, r.tobytes())
    except Exception as ex:
        return ("raise", type(ex).__name__, str(ex))
def check(*args, **kw):
    global count
    count += 1
    theirs, ours = outcome(np.bincount, args, kw), outcome(fnp.bincount, args, kw)
    if theirs != ours:
        bad.append((str([getattr(a, "dtype", type(a).__name__) for a in args]), kw, theirs[:3], ours[:3]))
w8 = rng.standard_normal(3000)
for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]:
    x = rng.integers(0, 2, 3000).astype(dtype) if dtype is np.bool_ else rng.integers(0, 120, 3000).astype(dtype)
    for w in [w8, w8.astype(np.float32), rng.integers(0, 9, 3000).astype(np.uint8), rng.random(3000) < 0.5,
              np.where(rng.random(3000) < 0.1, np.nan, w8)]:
        check(x, w)
    check(x, w8, minlength=400)
x8 = rng.integers(0, 120, 3000).astype(np.uint8)
for w in [w8.astype(np.longdouble), w8 + 1j, np.array(["1"] * 3000), np.ones((30, 100)), np.ones(2999),
          np.arange(3000).astype("m8[s]")]:
    check(x8, w)
check(np.array([1, -1, 2], dtype=np.int8)); check(np.array([1, -1, 2], dtype=np.int8), np.ones(3))
check([1, 2, 3], [0.5, 0.5, 0.5])
print(bad if bad else True, count)
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 63");
    Ok(())
}

/// float32 `histogram` with integer bins (`histogram_f32`): a vectorised range pass and numpy's
/// float32 bin arithmetic, counted on the pool from 2^20 elements while bins <= 4096. Counts and
/// edges must be numpy's bytes: signed zeros at either end of the range (the first zero met sets
/// the edge's sign bit), a constant array, huge / tiny scales, bins above the parallel cap, and the
/// negative case the former route got wrong - an EMPTY float32 array, whose edges numpy computes
/// from the Python ints (0, 1) in float64 (fnp computed them in float32, last bits differed).
#[test]
fn histogram_float32_counts_and_edges_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(20261014)
bad = []
count = 0
def outcome(fn, a, kw):
    try:
        c, e = fn(a, **kw)
        return ("ok", c.dtype.name, c.tobytes(), e.dtype.name, e.tobytes())
    except Exception as ex:
        return ("raise", type(ex).__name__, str(ex))
def check(a, label, **kw):
    global count
    count += 1
    if outcome(np.histogram, a, kw) != outcome(fnp.histogram, a, kw):
        bad.append((label, a.size, kw))
for n in [9, 65537, (1 << 20) + 3]:
    x = rng.standard_normal(n).astype(np.float32)
    for bins in [1, 10, 4096, 5000]:
        check(x, "normal", bins=bins)
    check(np.sort(x), "sorted", bins=10)
    check(np.full(n, 2.5, dtype=np.float32), "constant", bins=10)
    y = np.abs(x); y[n // 3] = -0.0; y[n // 2] = 0.0; check(y, "zero minimum", bins=10)
    y = -np.abs(x); y[0] = -0.0; check(y, "negative zero maximum", bins=10)
    check((x * 1e30).astype(np.float32), "huge", bins=10)
    y = x.copy(); y[n // 2] = np.nan; check(y, "nan", bins=10)
check(np.array([], dtype=np.float32), "empty", bins=10)
check(np.array([], dtype=np.float32), "empty", bins=3)
print(bad if bad else True, count)
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 32");
    Ok(())
}

#[test]
fn digitize_matches_numpy_for_every_key_order() -> Result<(), String> {
    // The native digitize searches an ascending run of keys from the previous key's bracket
    // (numpy's own binsearch) and any other order with a branchless search. Both must give
    // numpy's insertion points for every order, dtype, side and bin direction - including NaN
    // keys inside, before and after an ascending run (a NaN breaks the ascending scan), ties
    // with the bins, duplicate bins, and runs split across the parallel chunks (2^21 + 3).
    let script = fnp_script(
        r#"
rng = np.random.default_rng(29)
cells, bad = 0, []
def orders(n, dtype, ties):
    base = rng.integers(-50, 2050, n).astype(dtype)
    asc = np.sort(base)
    out = {"ascending": asc, "descending": asc[::-1].copy(), "random": base}
    if n > 2:
        bump = asc.copy(); bump[n // 2], bump[n // 2 + 1] = bump[n // 2 + 1], bump[n // 2]
        out["one descent"] = bump
        out["ties with bins"] = np.repeat(ties, n // len(ties) + 1)[:n]
    if np.dtype(dtype).kind == "f":
        for where in ("start", "middle", "end"):
            k = asc.copy(); k[{"start": 0, "middle": n // 2, "end": n - 1}[where]] = np.nan
            out["nan " + where] = k
        out["lone nan"] = np.array([np.nan], dtype=dtype)
    return out
for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64, np.float32, np.float64):
    if np.dtype(dtype).itemsize == 1:
        bins_sets = {"increasing": np.array([5, 10, 10, 60, 100], dtype=dtype)}
    else:
        bins_sets = {"increasing": np.array([10, 100, 100, 1000], dtype=dtype),
                     "64 bins": np.arange(0, 2000, 31).astype(dtype)}
    bins_sets["decreasing"] = bins_sets["increasing"][::-1].copy()
    bins_sets["single"] = bins_sets["increasing"][:1].copy()
    for n in (1, 2, 7, 4096) + (((1 << 21) + 3,) if dtype in (np.int64, np.float64) else ()):
        for label, keys in orders(n, dtype, bins_sets["increasing"]).items():
            if np.dtype(dtype).kind == "u":
                keys = np.clip(keys, 0, None) if keys.dtype.kind != "f" else keys
            for bl, bins in bins_sets.items():
                for right in (False, True):
                    cells += 1
                    got = fnp.digitize(keys, bins, right=right)
                    want = np.digitize(keys, bins, right=right)
                    if got.dtype != want.dtype or got.shape != want.shape or not np.array_equal(got, want):
                        bad.append(f"{np.dtype(dtype).name} n={n} {label} {bl} right={right}")
print(cells, bad[:10])
"#
        .into(),
    );
    let out = numpy_oracle(&script)?;
    let (cells, bad) = out.trim().split_once(' ').unwrap_or(("0", &out));
    assert_eq!(
        bad, "[]",
        "digitize must match numpy for every key order: {out}"
    );
    assert!(
        cells.parse::<usize>().unwrap_or(0) > 600,
        "cell table shrank: {out}"
    );
    Ok(())
}

/// bincount's int64 tally reads its operand off the object layout before any dtype is read
/// (`try_zerocopy_bincount`), and a negative `minlength` is numpy's to raise - after it has
/// converted `x`, so a float array, a 2-D or 0-d `x` is numpy's own error and a float list its
/// deprecation warning first. Every observable must stay numpy's (type, dtype and its char,
/// shape, bytes, warnings, errors) across int64 / longlong / int32 / uint8 / uint64 / bool /
/// float / byte-swapped operands, lists, empty, negative, strided, reversed, misaligned,
/// matrix inputs, minlength and weights.
#[test]
fn bincount_int64_layout_route_and_negative_minlength_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = fn(*a, **k); x = np.asarray(r)
            res = ("ok", type(r).__name__, x.dtype.str, x.dtype.char, x.shape, x.tobytes())
        except Exception as e:
            res = ("raise", type(e).__name__, str(e))
    return res + (tuple(sorted((x.category.__name__, str(x.message)) for x in w)),)
def misaligned(x):
    buf = np.zeros(x.nbytes + 1, np.uint8)
    buf[1:] = np.ascontiguousarray(x).view(np.uint8).ravel()
    return np.frombuffer(buf.data, dtype=x.dtype, count=x.size, offset=1).reshape(x.shape)
rng = np.random.default_rng(20261007)
i = rng.integers(0, 50, 64)
ops = {"i8": i, "i8 one": np.array([5]), "i8 zeros": np.zeros(7, np.int64), "q": i.astype("q"), "i4": i.astype("i4"),
       "u1": i.astype("u1"), "u8": i.astype("u8"), "bool": i > 20, "f8": i.astype("f8"), "list": [1, 2, 2, 5],
       "float list": [1.0, 2.0], "empty": np.array([], np.int64), "negative": np.array([1, -1, 2]), "2-D": i.reshape(8, 8),
       "0-d": np.array(3), "strided": i[::2], "reversed": i[::-1], "misaligned": misaligned(i), "big": np.array([0, 100000]),
       "8000": rng.integers(0, 1000, 8000), ">i8": i.astype(">i8"), "matrix": np.matrix(i[:4])}
cells, bad = 0, []
for name, x in ops.items():
    for kw in ({}, {"minlength": 60}, {"minlength": 0}, {"minlength": -1}, {"weights": np.ones(np.size(x))},
               {"weights": None}):
        cells += 1
        if outcome(fnp.bincount, x, **kw) != outcome(np.bincount, x, **kw):
            bad.append((name, kw))
print(cells, bad[:8])
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "132 []");
    Ok(())
}
