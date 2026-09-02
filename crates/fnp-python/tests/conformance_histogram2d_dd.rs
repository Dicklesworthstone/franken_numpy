//! Conformance tests for numpy histogram2d and histogramdd functions against NumPy oracle.
//!
//! Tests histogram2d, histogramdd for 2D, 3D, and higher-dimensional histograms.

mod common;

use common::with_fnp_and_numpy;
use pyo3::types::{PyDict, PyDictMethods};
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
         def normalize(value):\n\
         {I4}if isinstance(value, tuple):\n\
         {I8}return ('tuple', [normalize(item) for item in value])\n\
         {I4}if isinstance(value, list):\n\
         {I8}return ('list', [normalize(item) for item in value])\n\
         {I4}arr = np.asarray(value)\n\
         {I4}return ('array', str(arr.dtype), tuple(arr.shape), repr(arr.tolist()))\n\
         def outcome(op):\n\
         {I4}try:\n\
         {I8}print(('ok', normalize({call_expr})))\n\
         {I4}except Exception as exc:\n\
         {I8}print(('err', type(exc).__name__))\n\
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
// histogram2d
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram2d_tuple_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list tuple inputs with per-axis bins",
            "",
            "op([0.2, 1.2, 2.2, 3.2], (0.4, 1.4, 2.4, 3.4), bins=[2, 3])",
        ),
        (
            "weights density keywords",
            "",
            "op([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], bins=2, weights=[1.0, 2.0, 3.0], density=True)",
        ),
        (
            "invalid bins length error type",
            "",
            "op([0.0, 1.0], [0.0, 1.0], bins=[2])",
        ),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script("np.histogram2d", setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script("histogram2d", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "histogram2d tuple outcome mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn histogram2d_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 2, 3, 4, 5])
H, xedges, yedges = fnp.histogram2d(x, y)
np_H, np_xedges, np_yedges = np.histogram2d(x, y)
match = np.array_equal(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d basic should match numpy"
    );
    Ok(())
}

#[test]
fn histogram2d_with_bins_int() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
y = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
H, xedges, yedges = fnp.histogram2d(x, y, bins=3)
np_H, np_xedges, np_yedges = np.histogram2d(x, y, bins=3)
match = np.array_equal(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d with int bins should match numpy"
    );
    Ok(())
}

#[test]
fn histogram2d_with_bins_tuple() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
y = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
H, xedges, yedges = fnp.histogram2d(x, y, bins=[3, 4])
np_H, np_xedges, np_yedges = np.histogram2d(x, y, bins=[3, 4])
match = np.array_equal(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d with tuple bins should match numpy"
    );
    Ok(())
}

#[test]
fn histogram2d_with_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H, xedges, yedges = fnp.histogram2d(x, y, bins=5, range=[[2, 7], [3, 8]])
np_H, np_xedges, np_yedges = np.histogram2d(x, y, bins=5, range=[[2, 7], [3, 8]])
match = np.array_equal(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d with range should match numpy"
    );
    Ok(())
}

#[test]
fn histogram2d_density() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
H, xedges, yedges = fnp.histogram2d(x, y, bins=5, density=True)
np_H, np_xedges, np_yedges = np.histogram2d(x, y, bins=5, density=True)
match = np.allclose(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d density should match numpy"
    );
    Ok(())
}

#[test]
fn histogram2d_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 2, 3, 4])
weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
H, xedges, yedges = fnp.histogram2d(x, y, bins=3, weights=weights)
np_H, np_xedges, np_yedges = np.histogram2d(x, y, bins=3, weights=weights)
match = np.allclose(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d with weights should match numpy"
    );
    Ok(())
}

#[test]
fn histogram2d_explicit_bin_edges() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.5, 1.5, 2.5, 3.5])
y = np.array([0.5, 1.5, 2.5, 3.5])
xbins = np.array([0, 1, 2, 3, 4])
ybins = np.array([0, 2, 4])
H, xedges, yedges = fnp.histogram2d(x, y, bins=[xbins, ybins])
np_H, np_xedges, np_yedges = np.histogram2d(x, y, bins=[xbins, ybins])
match = np.array_equal(H, np_H) and np.allclose(xedges, np_xedges) and np.allclose(yedges, np_yedges)
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d with explicit edges should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// histogramdd
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogramdd_tuple_outcomes_match_numpy() -> Result<(), String> {
    let cases = [
        (
            "list sample with per-axis bins",
            "",
            "op([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0]], bins=[2, 3])",
        ),
        (
            "range weights density keywords",
            "",
            "op([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]], bins=2, range=[[0.0, 2.0], [0.0, 4.0]], weights=[1.0, 2.0, 3.0], density=True)",
        ),
        (
            "invalid bins length error type",
            "",
            "op([[0.0, 0.0], [1.0, 1.0]], bins=[2])",
        ),
    ];

    for (label, setup, call_expr) in cases {
        let numpy_result = numpy_oracle(&numpy_outcome_script("np.histogramdd", setup, call_expr))?;
        let rust_result = numpy_oracle(&fnp_outcome_script("histogramdd", setup, call_expr))?;

        assert_eq!(
            numpy_result, rust_result,
            "histogramdd tuple outcome mismatch for {label}"
        );
    }

    Ok(())
}

#[test]
fn histogramdd_one_dimensional_ndarray_sample_matches_numpy() -> Result<(), String> {
    // REGRESSION GUARD for the 1332x cell (`deadlock-audit-9tk4m`). A 1-D ndarray sample
    // is NumPy's `(N, D)` spelling normalised by `np.atleast_2d(sample).T`, i.e. ONE axis
    // of N samples - it is not a sequence of N single-sample axes. The dispatcher used to
    // reach its iterate-the-sequence branch with an ndarray and read a 65536-element array
    // as 65536 axes, calling `histogram_bin_edges` once per element.
    //
    // The three shapes below are the whole decision: 1-D must histogram as D = 1, 2-D
    // keeps the `(N, D)` reading, and 3-D must RAISE exactly as NumPy does rather than be
    // silently reinterpreted. `histogramdd` returns a tuple, so shape, dtype, counts and
    // every edge array are compared.
    with_fnp_and_numpy(|py, fnp, np| {
        let globals = PyDict::new(py);
        globals.set_item("np", np)?;
        globals.set_item("fnp", fnp)?;
        py.run(
            pyo3::ffi::c_str!(
                r#"
rng = np.random.default_rng(20260826)

for n, bins in ((7, 3), (256, 10), (4096, 17)):
    x = rng.standard_normal(n)
    H, edges = fnp.histogramdd(x, bins=bins)
    nH, nedges = np.histogramdd(x, bins=bins)
    assert H.dtype == nH.dtype
    assert H.shape == nH.shape
    assert np.array_equal(H, nH)
    assert len(edges) == len(nedges) == 1
    assert all(e.dtype == ne.dtype and np.array_equal(e, ne) for e, ne in zip(edges, nedges))

# the default-bins spelling, and the one with an explicit range
x = rng.standard_normal(1024)
for kwargs in ({}, {"bins": 5}, {"bins": 5, "range": [(-1.0, 1.0)]}):
    H, edges = fnp.histogramdd(x, **kwargs)
    nH, nedges = np.histogramdd(x, **kwargs)
    assert np.array_equal(H, nH)
    assert all(np.array_equal(e, ne) for e, ne in zip(edges, nedges))

# a 2-D sample keeps the (N, D) reading
s2 = rng.standard_normal((512, 2))
H, edges = fnp.histogramdd(s2, bins=4)
nH, nedges = np.histogramdd(s2, bins=4)
assert np.array_equal(H, nH)
assert all(np.array_equal(e, ne) for e, ne in zip(edges, nedges))

# a 3-D ndarray sample must raise the SAME error type NumPy raises
def outcome(fn):
    try:
        fn(rng.standard_normal((4, 3, 2)), bins=2)
        return "no-raise"
    except Exception as exc:
        return type(exc).__name__
assert outcome(fnp.histogramdd) == outcome(np.histogramdd)
"#
            ),
            Some(&globals),
            None,
        )?;

        Ok(())
    });
    Ok(())
}

#[test]
fn histogramdd_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
sample = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
H, edges = fnp.histogramdd(sample, bins=3)
np_H, np_edges = np.histogramdd(sample, bins=3)
match = np.array_equal(H, np_H)
for e, ne in zip(edges, np_edges):
    if not np.allclose(e, ne):
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogramdd 2D should match numpy");
    Ok(())
}

#[test]
fn histogramdd_3d() -> Result<(), String> {
    let script = fnp_script(
        r#"
sample = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
H, edges = fnp.histogramdd(sample, bins=2)
np_H, np_edges = np.histogramdd(sample, bins=2)
match = np.array_equal(H, np_H)
for e, ne in zip(edges, np_edges):
    if not np.allclose(e, ne):
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "histogramdd 3D should match numpy");
    Ok(())
}

#[test]
fn histogramdd_with_range() -> Result<(), String> {
    let script = fnp_script(
        r#"
sample = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
H, edges = fnp.histogramdd(sample, bins=3, range=[[1, 4], [1, 4]])
np_H, np_edges = np.histogramdd(sample, bins=3, range=[[1, 4], [1, 4]])
match = np.array_equal(H, np_H)
for e, ne in zip(edges, np_edges):
    if not np.allclose(e, ne):
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogramdd with range should match numpy"
    );
    Ok(())
}

#[test]
fn histogramdd_density() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
sample = np.random.randn(50, 2)
H, edges = fnp.histogramdd(sample, bins=3, density=True)
np_H, np_edges = np.histogramdd(sample, bins=3, density=True)
match = np.allclose(H, np_H)
for e, ne in zip(edges, np_edges):
    if not np.allclose(e, ne):
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogramdd density should match numpy"
    );
    Ok(())
}

#[test]
fn histogramdd_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
sample = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
weights = np.array([1.0, 2.0, 3.0, 4.0])
H, edges = fnp.histogramdd(sample, bins=2, weights=weights)
np_H, np_edges = np.histogramdd(sample, bins=2, weights=weights)
match = np.allclose(H, np_H)
for e, ne in zip(edges, np_edges):
    if not np.allclose(e, ne):
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogramdd with weights should match numpy"
    );
    Ok(())
}

#[test]
fn histogramdd_different_bins_per_dim() -> Result<(), String> {
    let script = fnp_script(
        r#"
sample = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
H, edges = fnp.histogramdd(sample, bins=[2, 4])
np_H, np_edges = np.histogramdd(sample, bins=[2, 4])
match = np.array_equal(H, np_H)
for e, ne in zip(edges, np_edges):
    if not np.allclose(e, ne):
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogramdd with different bins per dimension should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn histogram2d_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
H, _, _ = fnp.histogram2d(x, y)
print(np.sum(H) == len(x))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d sum should equal sample count"
    );
    Ok(())
}

#[test]
fn histogramdd_sum_equals_count() -> Result<(), String> {
    let script = fnp_script(
        r#"
sample = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
H, _ = fnp.histogramdd(sample)
print(np.sum(H) == len(sample))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogramdd sum should equal sample count"
    );
    Ok(())
}

#[test]
fn histogram2d_density_integrates_to_one() -> Result<(), String> {
    let script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(200)
y = np.random.randn(200)
H, xedges, yedges = fnp.histogram2d(x, y, bins=10, density=True)
dx = np.diff(xedges)
dy = np.diff(yedges)
integral = np.sum(H * np.outer(dx, dy))
print(np.isclose(integral, 1.0, atol=0.1))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogram2d density should approximately integrate to 1"
    );
    Ok(())
}

#[test]
fn histogramdd_matches_histogram2d_for_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
y = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
sample = np.column_stack([x, y])

H2d, xedges, yedges = fnp.histogram2d(x, y, bins=3)
Hdd, edges = fnp.histogramdd(sample, bins=3)

# histogramdd returns transposed compared to histogram2d
match = np.allclose(H2d, Hdd.T) and np.allclose(xedges, edges[0]) and np.allclose(yedges, edges[1])
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "histogramdd should be consistent with histogram2d for 2D data"
    );
    Ok(())
}
