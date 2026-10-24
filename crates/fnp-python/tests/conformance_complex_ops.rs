//! Conformance tests for numpy complex number operations against NumPy oracle.
//!
//! Tests real, imag, conj (complex number operations).

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
fn real_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.real(z)
expected = np.real(z)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real complex array should match numpy"
    );
    Ok(())
}

#[test]
fn imag_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.imag(z)
expected = np.imag(z)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "imag complex array should match numpy"
    );
    Ok(())
}

#[test]
fn conj_complex_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j, 5+6j])
result = fnp.conj(z)
expected = np.conj(z)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "conj complex array should match numpy"
    );
    Ok(())
}

#[test]
fn real_real_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([1.0, 2.0, 3.0])
result = fnp.real(x)
expected = np.real(x)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real on real array should match numpy"
    );
    Ok(())
}

#[test]
fn real_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+2j)
fnp_result = fnp.real(z)
np_result = np.real(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "real scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn imag_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+2j)
fnp_result = fnp.imag(z)
np_result = np.imag(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "imag scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn conj_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.complex128(1+2j)
fnp_result = fnp.conj(z)
np_result = np.conj(z)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "conj scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn real_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([np.inf + 0j, -np.inf + 1j, np.nan + 2j])
result = fnp.real(z)
expected = np.real(z)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "real special values should match numpy"
    );
    Ok(())
}

#[test]
fn imag_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1 + np.inf*1j, 2 + -np.inf*1j])
result = fnp.imag(z)
expected = np.imag(z)
# inf * 1j produces nan in imag part due to multiplication
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "imag special values should match numpy"
    );
    Ok(())
}

#[test]
fn conj_special_values() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([np.inf + 0j, -np.inf + 0j, 0 + np.nan*1j])
result = fnp.conj(z)
expected = np.conj(z)
print(np.allclose(result, expected, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "conj special values should match numpy"
    );
    Ok(())
}

#[test]
fn conjugate_alias() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+2j, 3+4j])
fnp_conj = fnp.conj(z)
fnp_conjugate = fnp.conjugate(z)
print(np.array_equal(fnp_conj, fnp_conjugate))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "conjugate should be alias for conj");
    Ok(())
}

#[test]
fn real_imag_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([0+0j, -0+0j, 0-0j, -0-0j])
fnp_real = fnp.real(z)
fnp_imag = fnp.imag(z)
np_real = np.real(z)
np_imag = np.imag(z)
print(np.allclose(fnp_real, np_real) and np.allclose(fnp_imag, np_imag))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "real/imag zero should match numpy");
    Ok(())
}

/// complex64 / complex128 ufuncs and dispatched functions either side of their small-call
/// thresholds (numpy's call below, native above): 20 ufuncs and 18 functions (dot / inner /
/// matmul / correlate / convolve / ediff1d are numpy's at every size) x two widths x eight sizes
/// 16 .. 2^18, operands carrying NaN, inf and zero - result type, dtype, shape, bytes and the
/// warnings raised.
#[test]
fn complex_ufuncs_and_functions_match_numpy_either_side_of_the_gates() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(111)
bad, cells = [], 0
def outcome(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            v = fn()
        except Exception as exc:
            return ("raise", type(exc).__name__, str(exc)[:80])
        a = np.asarray(v)
        return (type(v).__name__, a.dtype.str, a.shape, a.tobytes(), sorted({str(w.message)[:50] for w in caught}))
def check(label, call):
    global cells
    cells += 1
    if outcome(lambda: call(np)) != outcome(lambda: call(fnp)):
        bad.append(label)
def cx(dt, shape):
    v = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    if v.size > 8:
        v.flat[3] = complex(np.nan, 1); v.flat[5] = complex(np.inf, 0); v.flat[7] = 0
    return v
for dt in ("complex64", "complex128"):
    for n in (16, 127, 128, 2047, 2048, 8192, 32768, 1 << 18):
        a, b = cx(dt, n), cx(dt, n)
        for name in ("absolute", "add", "multiply", "divide", "sqrt", "exp", "log", "isnan", "isinf", "equal",
                     "less", "maximum", "minimum", "fmax", "sign", "square", "conjugate", "reciprocal", "power", "tanh"):
            uf = getattr(np, name)
            args = (a,) if uf.nin == 1 else (a, b)
            check(f"{name} {dt} {n}", lambda m, name=name, args=args: getattr(m, name)(*args))
        side = max(2, int(n ** 0.5))
        sq = cx(dt, (side, side))
        for name, call in (("trace", lambda m: m.trace(sq)), ("tril", lambda m: m.tril(sq)), ("sort", lambda m: m.sort(a)),
                           ("isin", lambda m: m.isin(a, b[:50])), ("append", lambda m: m.append(a, b)),
                           ("nanargmax", lambda m: m.nanargmax(a)), ("median", lambda m: m.median(a)),
                           ("ptp", lambda m: m.ptp(a)), ("diff", lambda m: m.diff(a)), ("cumsum", lambda m: m.cumsum(a)),
                           ("argmax", lambda m: m.argmax(a)), ("max", lambda m: m.max(a)),
                           ("dot", lambda m: m.dot(a, b)), ("inner", lambda m: m.inner(a, b)),
                           ("matmul", lambda m: m.matmul(a, b)), ("correlate", lambda m: m.correlate(a, b[:50])),
                           ("convolve", lambda m: m.convolve(a, b[:50])), ("ediff1d", lambda m: m.ediff1d(a))):
            check(f"{name} {dt} {n}", call)
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "608", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "complex ufuncs and functions must match numpy: {result}"
    );
    Ok(())
}

/// The native complex route that calls the system libm's own complex functions (clog, csqrt,
/// ctan, ctanh, catan, casin, cacos, casinh, cacosh, catanh and their float twins), as numpy's
/// loops do: complex128 and complex64 at 4,095 (numpy's call), 4,096 (the smallest native size)
/// and 2^16 + 37 (pooled, ragged). Each runs plain and with one special set in the LAST chunk -
/// signed zeros (log's divide-by-zero), non-finite pairs (C99 Annex G), huge and subnormal parts
/// (overflow / underflow) and branch-cut points with signed-zero imaginary parts - under
/// errstate(all=) warn / raise / ignore. Bytes, dtype, shape and every warning are compared.
#[test]
fn complex_libm_ops_match_numpy_bytes_and_events_on_both_sides_of_the_floor() -> Result<(), String>
{
    let script = fnp_script(
        r#"
import warnings
inf, nan = np.inf, np.nan
ops = ["log", "sqrt", "tan", "tanh", "arctan", "arcsin", "arccos", "arcsinh", "arccosh", "arctanh"]
def outcome(m, name, x, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = getattr(m, name)(x)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
rng = np.random.default_rng(31)
cells, bad = 0, []
for dt in (np.complex128, np.complex64):
    real = np.finfo(np.float64 if dt == np.complex128 else np.float32)
    big, sub = real.max / 4, real.smallest_subnormal
    specials = {
        "zeros": [complex(0.0, 0.0), complex(-0.0, 0.0), complex(0.0, -0.0), complex(-0.0, -0.0)],
        "nonfinite": [complex(inf, nan), complex(nan, inf), complex(-inf, 0.0), complex(0.0, inf),
                      complex(inf, -inf), complex(nan, nan), complex(-inf, nan)],
        "huge tiny": [complex(big, big), complex(-big, 5.0), complex(sub, sub), complex(sub, 1.0)],
        "cuts": [complex(-1.0, 0.0), complex(-1.0, -0.0), complex(2.0, 0.0), complex(2.0, -0.0),
                 complex(0.0, 2.0), complex(-0.0, 2.0), complex(0.5, 0.0), complex(1.0, 0.0)],
    }
    for n in (4095, 4096, (1 << 16) + 37):
        base = (rng.standard_normal(n) * 2 + 1j * rng.standard_normal(n) * 2).astype(dt)
        for name in ops:
            for label, tail in {"plain": [], **specials}.items():
                x = base.copy()
                if tail:
                    x[-len(tail):] = np.array(tail, dtype=dt)
                for mode in ("warn", "raise", "ignore"):
                    cells += 1
                    if outcome(fnp, name, x, mode) != outcome(np, name, x, mode):
                        bad.append(f"{np.dtype(dt).name} {name} n={n} {label} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "900", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "the libm complex route must match numpy's bytes and events: {result}"
    );
    Ok(())
}
