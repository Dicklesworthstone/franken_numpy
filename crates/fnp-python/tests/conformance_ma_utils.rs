//! Conformance tests for numpy.ma utility functions against NumPy oracle.
//!
//! Tests: getdata, is_mask, isMA, isMaskedArray, isarray, harden_mask,
//! soften_mask, cov, corrcoef for masked arrays.

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
         import numpy.ma as ma\n\
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// getdata
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn getdata_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
fnp_result = fnp.ma.getdata(x)
np_result = ma.getdata(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "getdata basic mismatch");
    Ok(())
}

#[test]
fn getdata_no_mask() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3, 4])
fnp_result = fnp.ma.getdata(x)
np_result = ma.getdata(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "getdata no mask mismatch");
    Ok(())
}

#[test]
fn getdata_regular_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3, 4])
fnp_result = fnp.ma.getdata(x)
np_result = ma.getdata(x)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "getdata regular array mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// is_mask
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn is_mask_true_cases() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Boolean array is a valid mask
m1 = np.array([True, False, True])
fnp_r1 = fnp.ma.is_mask(m1)
np_r1 = ma.is_mask(m1)
print(fnp_r1 == np_r1)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "is_mask boolean array mismatch");
    Ok(())
}

#[test]
fn is_mask_false_cases() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Integer array is not a valid mask
m = np.array([1, 0, 1])
fnp_r = fnp.ma.is_mask(m)
np_r = ma.is_mask(m)
print(fnp_r == np_r)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "is_mask integer array mismatch");
    Ok(())
}

#[test]
fn is_mask_nomask() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_r = fnp.ma.is_mask(ma.nomask)
np_r = ma.is_mask(ma.nomask)
print(fnp_r == np_r)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "is_mask nomask mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// isMA / isMaskedArray / isarray
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn isma_masked_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3], mask=[0, 1, 0])
fnp_r = fnp.ma.isMA(x)
np_r = ma.isMA(x)
print(fnp_r == np_r == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "isMA masked array mismatch");
    Ok(())
}

#[test]
fn isma_regular_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1, 2, 3])
fnp_r = fnp.ma.isMA(x)
np_r = ma.isMA(x)
print(fnp_r == np_r == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "isMA regular array mismatch");
    Ok(())
}

#[test]
fn ismaskedarray_equivalence() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3], mask=[0, 1, 0])
# isMaskedArray is an alias for isMA
fnp_r1 = fnp.ma.isMaskedArray(x)
fnp_r2 = fnp.ma.isMA(x)
np_r1 = ma.isMaskedArray(x)
np_r2 = ma.isMA(x)
print(fnp_r1 == fnp_r2 == np_r1 == np_r2 == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "isMaskedArray equivalence mismatch");
    Ok(())
}

#[test]
fn isarray_masked_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3], mask=[0, 1, 0])
fnp_r = fnp.ma.isarray(x)
np_r = ma.isarray(x)
print(fnp_r == np_r)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "isarray mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// harden_mask / soften_mask
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn harden_mask_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3], mask=[0, 1, 0])
fnp_result = fnp.ma.harden_mask(x)
np_result = ma.harden_mask(x)
# After hardening, hardmask should be True
print(fnp_result.hardmask == np_result.hardmask == True)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "harden_mask mismatch");
    Ok(())
}

#[test]
fn soften_mask_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3], mask=[0, 1, 0], hard_mask=True)
fnp_result = fnp.ma.soften_mask(x)
np_result = ma.soften_mask(x)
# After softening, hardmask should be False
print(fnp_result.hardmask == np_result.hardmask == False)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "soften_mask mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// cov (masked covariance)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ma_cov_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 0], [0, 1, 0]])
fnp_result = fnp.ma.cov(x)
np_result = ma.cov(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ma.cov basic mismatch");
    Ok(())
}

#[test]
fn ma_cov_no_mask() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3], [4, 5, 6]])
fnp_result = fnp.ma.cov(x)
np_result = ma.cov(x)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ma.cov no mask mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// corrcoef (masked correlation)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ma_corrcoef_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3, 4], [4, 3, 2, 1]], mask=[[0, 0, 0, 0], [0, 1, 0, 0]])
fnp_result = fnp.ma.corrcoef(x)
np_result = ma.corrcoef(x)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ma.corrcoef basic mismatch");
    Ok(())
}

#[test]
fn ma_corrcoef_no_mask() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([[1, 2, 3, 4], [4, 3, 2, 1]])
fnp_result = fnp.ma.corrcoef(x)
np_result = ma.corrcoef(x)
print(np.allclose(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ma.corrcoef no mask mismatch");
    Ok(())
}

#[test]
fn ma_corrcoef_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
y = ma.array([5, 4, 3, 2, 1], mask=[0, 1, 0, 0, 0])
fnp_result = fnp.ma.corrcoef(x, y)
np_result = ma.corrcoef(x, y)
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "ma.corrcoef 1d mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// make_mask_none
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn make_mask_none_1d() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ma.make_mask_none((5,))
np_result = ma.make_mask_none((5,))
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "make_mask_none 1d mismatch");
    Ok(())
}

#[test]
fn make_mask_none_2d() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ma.make_mask_none((3, 4))
np_result = ma.make_mask_none((3, 4))
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (3, 4))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "make_mask_none 2d mismatch");
    Ok(())
}

#[test]
fn make_mask_none_scalar_shape() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ma.make_mask_none(5)
np_result = ma.make_mask_none(5)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == (5,))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "make_mask_none scalar shape mismatch");
    Ok(())
}

#[test]
fn make_mask_none_dtype_bool() -> Result<(), String> {
    let script = fnp_script(
        r#"
fnp_result = fnp.ma.make_mask_none((2, 3))
print(fnp_result.dtype == np.bool_ and np.all(fnp_result == False))
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(output, "True", "make_mask_none dtype/values mismatch");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level masked helper export coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn top_level_mask_construction_and_fill_helpers_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def same_surface(left, right):
    if isinstance(left, ma.MaskedArray) or isinstance(right, ma.MaskedArray):
        return (
            isinstance(left, ma.MaskedArray)
            and isinstance(right, ma.MaskedArray)
            and np.array_equal(ma.getdata(left), ma.getdata(right))
            and np.array_equal(ma.getmaskarray(left), ma.getmaskarray(right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return repr(left) == repr(right)

structured_dtype = np.dtype([("a", np.int16), ("b", np.float32)])
structured_mask = np.array(
    [(True, False), (False, True)],
    dtype=[("a", bool), ("b", bool)],
)
structured_values = ma.array(
    np.array([(1, 2.5), (3, 4.5)], dtype=structured_dtype),
    mask=structured_mask,
)

common_left = ma.array([1, 2], mask=[0, 1], fill_value=-7)
common_right = ma.array([3, 4], mask=[0, 0], fill_value=-7)
fill_result = ma.array([1, 2, 3], mask=[0, 1, 0])
fill_expected = fill_result.copy()
fnp_set_result = fnp.set_fill_value(fill_result, -123)
np_set_result = ma.set_fill_value(fill_expected, -123)

match = (
    fnp.common_fill_value(common_left, common_right)
    == ma.common_fill_value(common_left, common_right)
    and fnp.default_fill_value(np.array([1], dtype=np.int16))
    == ma.default_fill_value(np.array([1], dtype=np.int16))
    and fnp_set_result == np_set_result
    and fill_result.fill_value == fill_expected.fill_value
    and same_surface(
        fnp.masked_object(["x", "sentinel", "y"], "sentinel"),
        ma.masked_object(["x", "sentinel", "y"], "sentinel"),
    )
    and same_surface(fnp.make_mask_none((2, 3)), ma.make_mask_none((2, 3)))
    and fnp.make_mask_descr(structured_dtype) == ma.make_mask_descr(structured_dtype)
    and same_surface(fnp.flatten_mask(structured_mask), ma.flatten_mask(structured_mask))
    and same_surface(
        fnp.flatten_structured_array(structured_values),
        ma.flatten_structured_array(structured_values),
    )
)
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(
        output, "True",
        "top-level masked construction/fill helpers mismatch"
    );
    Ok(())
}

#[test]
fn top_level_mask_rowcol_compress_and_edge_helpers_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def same_surface(left, right):
    if isinstance(left, ma.MaskedArray) or isinstance(right, ma.MaskedArray):
        return (
            isinstance(left, ma.MaskedArray)
            and isinstance(right, ma.MaskedArray)
            and np.array_equal(ma.getdata(left), ma.getdata(right))
            and np.array_equal(ma.getmaskarray(left), ma.getmaskarray(right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return repr(left) == repr(right)

x = ma.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    mask=[[False, True, False], [False, False, False], [True, True, True]],
)
match = (
    same_surface(fnp.mask_rows(x), ma.mask_rows(x))
    and same_surface(fnp.mask_cols(x), ma.mask_cols(x))
    and same_surface(fnp.mask_rowcols(x), ma.mask_rowcols(x))
    and same_surface(fnp.compress_nd(x), ma.compress_nd(x))
    and same_surface(fnp.compress_rowcols(x), ma.compress_rowcols(x))
    and same_surface(fnp.notmasked_edges(x, axis=1), ma.notmasked_edges(x, axis=1))
    and same_surface(
        fnp.notmasked_contiguous(x, axis=1),
        ma.notmasked_contiguous(x, axis=1),
    )
)
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(
        output, "True",
        "top-level masked row/col/compress/edge helpers mismatch"
    );
    Ok(())
}

#[test]
fn top_level_masked_apply_and_arg_helpers_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
def same_surface(left, right):
    if isinstance(left, ma.MaskedArray) or isinstance(right, ma.MaskedArray):
        return (
            isinstance(left, ma.MaskedArray)
            and isinstance(right, ma.MaskedArray)
            and np.array_equal(ma.getdata(left), ma.getdata(right))
            and np.array_equal(ma.getmaskarray(left), ma.getmaskarray(right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return repr(left) == repr(right)

def spread(row):
    return row.max() - row.min()

x = ma.array(
    [[1, 2, 3], [4, 5, 6]],
    mask=[[False, True, False], [False, False, True]],
)
match = (
    same_surface(
        fnp.ma_apply_along_axis(spread, 1, x),
        ma.apply_along_axis(spread, 1, x),
    )
    and same_surface(
        fnp.ma_apply_over_axes(ma.sum, x, [0, 1]),
        ma.apply_over_axes(ma.sum, x, [0, 1]),
    )
    and np.array_equal(fnp.ma_argmax(x, axis=1), ma.argmax(x, axis=1))
    and np.array_equal(fnp.ma_argmin(x, axis=0), ma.argmin(x, axis=0))
)
print(match)
"#
        .into(),
    );
    let output = numpy_oracle(&script)?;
    assert_eq!(
        output, "True",
        "top-level masked apply/arg helpers mismatch"
    );
    Ok(())
}
