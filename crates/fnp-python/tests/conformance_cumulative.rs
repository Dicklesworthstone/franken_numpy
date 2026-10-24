//! Conformance tests for numpy cumulative operations against NumPy oracle.
//!
//! Tests cumsum, cumprod, diff (diff is in here for completeness with incremental ops).

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

// ─────────────────────────────────────────────────────────────────────────────
// cumsum
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 1d should match numpy");
    Ok(())
}

#[test]
fn cumsum_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumsum 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn cumsum_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a, axis=0)
expected = np.cumsum(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 2d axis=0 should match numpy");
    Ok(())
}

#[test]
fn cumsum_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumsum(a, axis=1)
expected = np.cumsum(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum 2d axis=1 should match numpy");
    Ok(())
}

#[test]
fn cumsum_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.1, 0.2, 0.3, 0.4])
result = fnp.cumsum(a)
expected = np.cumsum(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumprod
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumprod_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod 1d should match numpy");
    Ok(())
}

#[test]
fn cumprod_2d_no_axis() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d no axis should flatten and match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_2d_axis0() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a, axis=0)
expected = np.cumprod(a, axis=0)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d axis=0 should match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_2d_axis1() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
result = fnp.cumprod(a, axis=1)
expected = np.cumprod(a, axis=1)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod 2d axis=1 should match numpy"
    );
    Ok(())
}

#[test]
fn cumprod_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.1, 1.2, 1.3, 1.4])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod float should match numpy");
    Ok(())
}

#[test]
fn cumprod_with_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 0, 4, 5])
result = fnp.cumprod(a)
expected = np.cumprod(a)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumprod with zero should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumsum_last_equals_sum() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
cumsum_result = fnp.cumsum(a)
sum_result = fnp.sum(a)
print(cumsum_result[-1] == sum_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "last cumsum element should equal sum"
    );
    Ok(())
}

#[test]
fn cumprod_last_equals_prod() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
cumprod_result = fnp.cumprod(a)
prod_result = fnp.prod(a)
print(cumprod_result[-1] == prod_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "last cumprod element should equal prod"
    );
    Ok(())
}

#[test]
fn cumsum_diff_relationship() -> Result<(), String> {
    let script = fnp_script(
        r#"
# For cumulative sum: diff(cumsum(a)) gives a[1:] (the original array without first element)
a = np.array([1, 2, 3, 4, 5])
cumsum_a = fnp.cumsum(a)
diff_cumsum = fnp.diff(cumsum_a)
# diff of cumsum gives the original elements (except first)
print(np.array_equal(diff_cumsum, a[1:]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "diff(cumsum(a)) should equal a[1:]");
    Ok(())
}

#[test]
fn cumsum_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
fnp_result = fnp.cumsum(z)
np_result = np.cumsum(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumsum complex should match numpy");
    Ok(())
}

#[test]
fn cumprod_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 2+0j, 0+1j], dtype=np.complex128)
fnp_result = fnp.cumprod(z)
np_result = np.cumprod(z)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "cumprod complex should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cumulative_sum / cumulative_prod (NumPy 2.0 Array-API names, native-wired)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cumulative_sum_prod_match_numpy_across_dtype_axis_and_include_initial() -> Result<(), String> {
    let script = fnp_script(
        r#"
ok = True
rng = np.random.default_rng(7)
for op in ["cumulative_sum", "cumulative_prod"]:
    ffn = getattr(fnp, op); nfn = getattr(np, op)
    for dt in [np.float64, np.float32, np.int8, np.int32, np.uint8, np.int64, np.bool_]:
        for shape in [(20,), (6, 5), (4, 3, 2)]:
            if dt == np.bool_:
                a = rng.integers(0, 2, shape).astype(dt)
            elif np.issubdtype(dt, np.integer):
                a = rng.integers(0, 4, shape).astype(dt)
            else:
                a = rng.standard_normal(shape).astype(dt)
            axes = [None] if len(shape) == 1 else list(range(len(shape)))
            for ax in axes:
                for inc in (False, True):
                    kw = {"include_initial": inc}
                    if ax is not None:
                        kw["axis"] = ax
                    f = np.asarray(ffn(a, **kw)); n = np.asarray(nfn(a, **kw))
                    if f.dtype != n.dtype or f.shape != n.shape or not np.allclose(f, n, rtol=1e-6, atol=1e-6, equal_nan=True):
                        ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumulative_sum/prod must match numpy across dtype/axis/include_initial"
    );
    Ok(())
}

#[test]
fn cumulative_sum_axis_none_on_nd_raises_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
ok = True
for op in ["cumulative_sum", "cumulative_prod"]:
    try:
        getattr(fnp, op)(np.arange(12).reshape(3, 4))
        ok = False  # numpy raises ValueError here
    except ValueError:
        pass
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "cumulative_sum/prod with axis=None on ndim>1 must raise ValueError like numpy"
    );
    Ok(())
}

/// The per-axis kernels (`cumulative_axis`, `interleaved_lane_scans`, `interleaved_lane_products`):
/// slab rows stepped as whole vectors, last-axis lanes scanned eight at a time, prod lanes
/// multiplied eight at a time. The shapes cover:
/// - lane counts that are not multiples of eight, and lanes of length one;
/// - middle axes of 3-D arrays;
/// - sizes past the 2^18 parallel floor on both branches.
///
/// The data holds NaN, -0.0 leading a slab, inf, products that overflow and integer sums that
/// wrap. A lane order or a slab order that differs from numpy's changes the bytes.
///
/// NaNs compare by class: where an input NaN meets the default NaN of `inf * 0`, which payload
/// survives depends on operand order in numpy's own compiled loop (its vector lanes and scalar
/// tail differ), not on numpy's semantics. Everything else, the zero signs included, compares as
/// bytes. On the former build, nancumsum (40, 7000) along axis 0 kept a running -0.0 where numpy
/// adds the +0.0 it substitutes for a NaN.
#[test]
fn axis_cumulatives_and_prod_are_byte_identical_to_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(31)
def data(dtype, shape):
    if dtype in ("f8", "f4"):
        a = rng.uniform(0.5, 1.6, shape).astype(dtype)
        flat = a.reshape(-1)
        flat[::97] = np.nan
        flat[1::131] = -0.0
        flat[2::173] = np.inf
        flat[3::59] = 1e30
        return a
    if dtype == "?":
        return rng.integers(0, 2, shape).astype(bool)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, shape, endpoint=True, dtype=dtype)
def outcome(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = np.array(fn())
            if r.dtype.kind == "f":
                r[np.isnan(r)] = np.nan
            got = ("ok", r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__)
    return got, sorted({str(w.message) for w in caught})
shapes = [(1, 1), (3, 1), (1, 7), (5, 3), (17, 9), (8, 16), (9, 8), (2, 3, 4), (3, 17, 5),
          (300, 257), (513, 512), (40, 7000), (7000, 40)]
cells = 0
bad = []
for dtype in ("f8", "f4", "i1", "i2", "i4", "i8", "u1", "u4", "u8", "?"):
    ops = ["cumsum", "cumprod"]
    if dtype in ("f8", "f4"):
        ops += ["nancumsum", "nancumprod", "prod"]
    if dtype == "i8":
        ops += ["prod"]
    for shape in shapes:
        a = data(dtype, shape)
        for op in ops:
            for axis in range(-len(shape), len(shape)):
                cells += 1
                ours = outcome(lambda: getattr(fnp, op)(a, axis=axis))
                theirs = outcome(lambda: getattr(np, op)(a, axis=axis))
                if ours != theirs:
                    bad.append(f"{op} {dtype} {shape} axis={axis}: {ours[0][:2]} {ours[1]} vs {theirs[0][:2]} {theirs[1]}")
print(cells, bad[:10])
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "1512",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "per-axis cumulatives and prod must be numpy's bytes: {result}"
    );
    Ok(())
}
