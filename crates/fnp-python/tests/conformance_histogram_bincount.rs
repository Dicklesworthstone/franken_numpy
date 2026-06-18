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
             try:\n\
                 value = {call_expr}\n\
                 arr = np.asarray(value)\n\
                 print('ok')\n\
                 print(type(value).__name__)\n\
                 print(str(arr.dtype))\n\
                 print(tuple(arr.shape))\n\
                 print(repr(arr.tolist()))\n\
             except Exception as exc:\n\
                 print('err')\n\
                 print(type(exc).__name__)\n\
         outcome(op)"
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
        let numpy_result =
            numpy_oracle(&numpy_outcome_script("np.histogram_bin_edges", setup, call_expr))?;
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
        (
            "nonmonotonic bins error type",
            "",
            "op([1, 2], [0, 2, 1])",
        ),
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
