//! Conformance tests for numpy.spacing against NumPy oracle.
//!
//! Tests spacing (distance to nearest representable value).

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

#[test]
fn spacing_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 1e10, 1e-10])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "spacing basic should match numpy");
    Ok(())
}

#[test]
fn spacing_negative() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-1.0, -2.0, -1e10])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing with negative values should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, np.inf, np.nan])
result = fnp.spacing(x)
expected = np.spacing(x)
# Check that non-NaN values match and NaN positions align
match = True
for r, e in zip(result.flat, expected.flat):
    if np.isnan(e):
        if not np.isnan(r):
            match = False
    elif r != e:
        match = False
print(match)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing with special values should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.0)
fnp_result = fnp.spacing(x)
np_result = np.spacing(x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "spacing scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn spacing_subnormal() -> Result<(), String> {
    let script = fnp_script(
        r#"
tiny = np.finfo(np.float64).tiny
x = np.array([tiny / 2, tiny / 4, tiny / 8])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing subnormal should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0.0, -0.0])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_large_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Test large but not maximum values
x = np.array([1e100, -1e100, 1e200])
result = fnp.spacing(x)
expected = np.spacing(x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing large values should match numpy"
    );
    Ok(())
}

#[test]
fn spacing_negative_inf() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([-np.inf])
result = fnp.spacing(x)
expected = np.spacing(x)
# spacing of -inf should be nan
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "spacing negative inf should match numpy"
    );
    Ok(())
}

/// The native float64 / float32 spacing route: serial (4,096) and pooled (2^18 + 37, a ragged
/// last chunk) sizes, both byte orders, contiguous and strided, under errstate(all=) warn /
/// raise / ignore. Special values sit in the LAST chunk: +-MAX (numpy's "overflow"), a
/// signaling NaN ("invalid"), nonzero subnormals ("underflow", which the default errstate
/// ignores), NaN payloads of both signs, +-inf and +-0, so a route that folds only some chunks'
/// flags, or reads a '>f8' buffer as native, fails. Bytes, dtype, shape and every warning are
/// compared. The route before this test missed underflow everywhere and every event on strided
/// or big-endian float64 operands: 42 of the 240 cells.
#[test]
fn spacing_native_route_matches_numpy_bytes_and_warnings_at_every_size() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
bad, cells = [], 0
def outcome(m, x, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = m.spacing(x)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
rng = np.random.default_rng(11)
for code, bits in (("f8", np.uint64), ("f4", np.uint32)):
    ftype = np.dtype(code).type
    info = np.finfo(ftype)
    snan = np.array([0x7FF0000000000001 if code == "f8" else 0x7F800001], dtype=bits).view(ftype)[0]
    qnan_neg = np.array([0xFFF8000000000123 if code == "f8" else 0xFFC00123], dtype=bits).view(ftype)[0]
    specials = {
        "plain": [],
        "inf zero subnormal nan": [np.inf, -np.inf, 0.0, -0.0, info.smallest_subnormal, -info.smallest_subnormal, np.nan, qnan_neg],
        "max": [info.max],
        "-max": [-info.max],
        "snan": [snan],
    }
    for n in (4096, (1 << 18) + 37):
        base = (rng.standard_normal(n) * 10.0 ** rng.integers(-30, 30, n)).astype(code)
        for label, tail in specials.items():
            x = base.copy()
            if tail:
                x[-len(tail):] = np.array(tail, dtype=code)
            for order in ("<", ">"):
                arr = x.astype(order + code)
                for layout, view in (("contiguous", arr), ("strided", arr[::2])):
                    for mode in ("warn", "raise", "ignore"):
                        cells += 1
                        if outcome(fnp, view, mode) != outcome(np, view, mode):
                            bad.append(f"{code} n={n} {label} {order} {layout} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut fields = result.trim().splitn(2, ' ');
    assert_eq!(
        fields.next().unwrap_or("0"),
        "240",
        "cell table drifted: {result}"
    );
    assert_eq!(
        fields.next().unwrap_or(""),
        "[]",
        "spacing must match numpy's bytes and warnings: {result}"
    );
    Ok(())
}
