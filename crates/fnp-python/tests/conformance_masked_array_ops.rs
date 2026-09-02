//! Conformance tests for numpy.ma masked array operations against NumPy oracle.
//!
//! Tests: compress_rows, compress_cols, clump_masked, clump_unmasked,
//! flatnotmasked_edges, flatnotmasked_contiguous

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

fn fnp_script(body: String) -> String {
    support::fnp_script_with("import numpy.ma as ma\n", false, body)
}

// ─────────────────────────────────────────────────────────────────────────────
// compress_rows
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_rows_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4], [5, 6]], mask=[[0, 0], [1, 0], [0, 0]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows basic mismatch");
    Ok(())
}

#[test]
fn compress_rows_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4], [5, 6]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows no masked mismatch");
    Ok(())
}

#[test]
fn compress_rows_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2], [3, 4]], mask=[[1, 1], [1, 1]])
fnp_result = fnp.compress_rows(x)
np_result = ma.compress_rows(x)
print(fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_rows all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// compress_cols
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_cols_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 0, 0]])
fnp_result = fnp.compress_cols(x)
np_result = ma.compress_cols(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_cols basic mismatch");
    Ok(())
}

#[test]
fn compress_cols_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]])
fnp_result = fnp.compress_cols(x)
np_result = ma.compress_cols(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "compress_cols no masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// clump_masked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clump_masked_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 1])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked basic mismatch");
    Ok(())
}

#[test]
fn clump_masked_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
print(len(fnp_result) == len(np_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked no masked mismatch");
    Ok(())
}

#[test]
fn clump_masked_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.clump_masked(a)
np_result = ma.clump_masked(a)
match = len(fnp_result) == len(np_result) == 1
if match:
    match = fnp_result[0] == np_result[0]
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_masked all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// clump_unmasked
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clump_unmasked_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[1, 0, 0, 1, 0])
fnp_result = fnp.clump_unmasked(a)
np_result = ma.clump_unmasked(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_unmasked basic mismatch");
    Ok(())
}

#[test]
fn clump_unmasked_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.clump_unmasked(a)
np_result = ma.clump_unmasked(a)
print(len(fnp_result) == len(np_result) == 0)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "clump_unmasked all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flatnotmasked_edges
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatnotmasked_edges_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[1, 0, 0, 0, 1])
fnp_result = fnp.flatnotmasked_edges(a)
np_result = ma.flatnotmasked_edges(a)
# flatnotmasked_edges returns a tuple or None
if np_result is None:
    print(fnp_result is None)
else:
    print(fnp_result[0] == np_result[0] and fnp_result[1] == np_result[1])
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_edges basic mismatch");
    Ok(())
}

#[test]
fn flatnotmasked_edges_all_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3], mask=[1, 1, 1])
fnp_result = fnp.flatnotmasked_edges(a)
np_result = ma.flatnotmasked_edges(a)
print(fnp_result is None and np_result is None)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_edges all masked mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// flatnotmasked_contiguous
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flatnotmasked_contiguous_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 0])
fnp_result = fnp.flatnotmasked_contiguous(a)
np_result = ma.flatnotmasked_contiguous(a)
match = len(fnp_result) == len(np_result)
if match:
    for f, n in zip(fnp_result, np_result):
        if f != n:
            match = False
            break
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "flatnotmasked_contiguous basic mismatch");
    Ok(())
}

#[test]
fn flatnotmasked_contiguous_no_masked() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = ma.array([1, 2, 3, 4, 5])
fnp_result = fnp.flatnotmasked_contiguous(a)
np_result = ma.flatnotmasked_contiguous(a)
match = len(fnp_result) == len(np_result) == 1
if match:
    match = fnp_result[0] == np_result[0]
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(
        output, "True",
        "flatnotmasked_contiguous no masked mismatch"
    );
    Ok(())
}

// numpy.ma re-export lock.
//
// Several numpy.ma names collide with an fnp TOP-LEVEL wrapper while being a
// different, masked-aware function: ma.cov, ma.corrcoef, ma.allclose, ma.ptp,
// ma.choose, ma.put. fnp registers them by pulling the objects straight out of
// numpy.ma (`np_ma.getattr(name)`), which is correct - but nothing asserted it,
// and swapping in the same-named top-level wrapper is exactly the mistake that
// was live in the linalg namespace until
// deadlock-audit-linalg-aliases-toplevel-wrappers-mi3k6. There it returned the
// wrong array for ndim > 2; here it would silently drop the MASK.
//
// Each case is compared against numpy.ma, and the test additionally asserts the
// PRECONDITIONS that make the comparison meaningful: ma.cov and np.cov must
// actually disagree on masked input, and the ma-only keywords must be rejected
// by the top-level functions. Without those, an alias regression could leave
// every case still passing.
#[test]
fn ma_namespace_is_numpy_ma_not_the_toplevel_wrappers() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform
import numpy.ma as npma

def masked():
    return npma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 1], [0, 0, 0]])

def described(value):
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape), np.round(array, 9).tolist())

def ma_cov(module):
    return described(module.ma.cov(masked()))

def ma_cov_allow_masked(module):
    return described(module.ma.cov(masked(), allow_masked=True))

def ma_corrcoef(module):
    return described(module.ma.corrcoef(masked()))

def ma_allclose_masked_equal(module):
    return bool(module.ma.allclose(masked(), masked(), masked_equal=True))

def ma_ptp(module):
    return described(module.ma.ptp(masked()))

def ma_count(module):
    return described(module.ma.count(masked()))

def ma_average_returned(module):
    value, weight = module.ma.average(masked(), axis=0, returned=True)
    return (described(value), described(weight))

def ma_masked_is_singleton(module):
    return module.ma.masked is npma.masked

def ma_maskedarray_is_type(module):
    return module.ma.MaskedArray is npma.MaskedArray

cases = [
    ("ma.cov on masked input", ma_cov),
    ("ma.cov allow_masked=", ma_cov_allow_masked),
    ("ma.corrcoef on masked input", ma_corrcoef),
    ("ma.allclose masked_equal=", ma_allclose_masked_equal),
    ("ma.ptp on masked input", ma_ptp),
    ("ma.count on masked input", ma_count),
    ("ma.average returned=", ma_average_returned),
    ("ma.masked singleton identity", ma_masked_is_singleton),
    ("ma.MaskedArray type identity", ma_maskedarray_is_type),
]

def outcome(module, call):
    try:
        return ("ok", call(module))
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

# Preconditions. If these ever stop holding, the cases above can no longer
# distinguish numpy.ma's function from the same-named top-level one, and this
# test has quietly stopped testing what it exists for.
x = masked()
if np.array_equal(np.asarray(npma.cov(x)), np.asarray(np.cov(x))):
    print("PRECONDITION LOST: ma.cov and np.cov agree on masked input")
    ok = False
for name, kwargs in [("cov", {"allow_masked": True}), ("allclose", {"masked_equal": True})]:
    try:
        if name == "cov":
            getattr(np, name)(x, **kwargs)
        else:
            getattr(np, name)(x, x, **kwargs)
        print(f"PRECONDITION LOST: top-level np.{name} accepted {list(kwargs)[0]}")
        ok = False
    except TypeError:
        pass

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
        "ma namespace should be numpy.ma's own objects ({provenance}): {result}"
    );
    Ok(())
}
