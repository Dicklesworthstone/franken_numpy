//! Conformance tests for numpy.heaviside against NumPy oracle.
//!
//! Tests the Heaviside step function:
//! - heaviside(x1, x2): 0 for x1 < 0, x2 for x1 == 0, 1 for x1 > 0

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

fn parse_float_list(s: &str) -> Result<Vec<f64>, String> {
    if s.is_empty() || s == "[]" {
        return Ok(vec![]);
    }
    let trimmed = s
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| format!("expected bracketed float list, got {s:?}"))?;

    let mut values = Vec::new();
    for token in trimmed
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
    {
        let t = token.trim().trim_end_matches('.');
        let value = if t == "nan" || t == "NaN" {
            f64::NAN
        } else if t == "inf" || t == "Inf" {
            f64::INFINITY
        } else if t == "-inf" || t == "-Inf" {
            f64::NEG_INFINITY
        } else {
            t.parse::<f64>()
                .map_err(|error| format!("invalid float token {token:?} in {s:?}: {error}"))?
        };
        values.push(value);
    }
    Ok(values)
}

fn floats_close(a: &[f64], b: &[f64], rel_tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        if x.is_nan() && y.is_nan() {
            true
        } else if x.is_infinite() && y.is_infinite() {
            x.signum() == y.signum()
        } else if *x == 0.0 && *y == 0.0 {
            true
        } else {
            let diff = (x - y).abs();
            let max_val = x.abs().max(y.abs()).max(1e-15);
            diff <= rel_tol * max_val
        }
    })
}

#[test]
fn heaviside_basic_step_function_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside basic mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_different_h0_values_match_numpy() -> Result<(), String> {
    let test_cases = vec![0.0, 0.5, 1.0, 0.25, 0.75, -0.5, 2.0];

    for h0 in &test_cases {
        let script =
            format!("import numpy as np; print(np.heaviside(np.array([-1, 0, 1]), {h0}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!(
            "print(fnp.heaviside(np.array([-1, 0, 1]), {h0}).tolist())"
        ));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "heaviside h0={h0} mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn heaviside_nan_input_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([np.nan, -1, 0, 1, np.nan])
result = np.heaviside(x, 0.5)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([np.nan, -1, 0, 1, np.nan])
result = fnp.heaviside(x, 0.5)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside nan input mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_nan_h0_at_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-1, 0, 1])
result = np.heaviside(x, np.nan)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-1, 0, 1])
result = fnp.heaviside(x, np.nan)
print([np.isnan(v) if np.isnan(v) else v for v in result])
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside nan h0 mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_inf_input_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-np.inf, np.inf, 0])
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-np.inf, np.inf, 0])
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside inf input mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([[-1], [0], [1]])
h0 = np.array([0.0, 0.5, 1.0])
print(np.heaviside(x, h0).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([[-1], [0], [1]])
h0 = np.array([0.0, 0.5, 1.0])
print(fnp.heaviside(x, h0).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_negative_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-0.0, 0.0])
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-0.0, 0.0])
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside negative zero mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn heaviside_50_random_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
x = np.random.randn(50) * 10
h0 = np.random.rand(50)
print(np.heaviside(x, h0).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
x = np.random.randn(50) * 10
h0 = np.random.rand(50)
print(fnp.heaviside(x, h0).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside random 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn heaviside_empty_array_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.heaviside(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.heaviside(np.array([]), np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside empty array mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_scalar_h0_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(np.heaviside(x, 0.0).tolist())
print(np.heaviside(x, 0.5).tolist())
print(np.heaviside(x, 1.0).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(fnp.heaviside(x, 0.0).tolist())
print(fnp.heaviside(x, 0.5).tolist())
print(fnp.heaviside(x, 1.0).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside scalar h0 mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_linspace_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
x = np.linspace(-5, 5, 50)
print(np.heaviside(x, 0.5).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
x = np.linspace(-5, 5, 50)
print(fnp.heaviside(x, 0.5).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "heaviside linspace mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn heaviside_dtype_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.heaviside(np.array([-1, 0, 1]), 0.5).dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.heaviside(np.array([-1, 0, 1]), 0.5).dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "heaviside dtype mismatch"
    );

    Ok(())
}

#[test]
fn heaviside_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(1.0)
h0 = np.float64(0.5)
fnp_result = fnp.heaviside(x, h0)
np_result = np.heaviside(x, h0)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "heaviside scalar return type should match numpy: {result}"
    );
    Ok(())
}

/// The native float32 route runs one select pass per element, serially or in tasks of 2^20, and
/// must give numpy's bytes and events: a NaN `x` becomes numpy's canonical 0x7fc00000 whatever
/// its payload, a zero of either sign returns the step value bit for bit (a NaN or signaling
/// one included, with no event), and a signaling NaN `x` raises "invalid". A spy counting
/// `np.heaviside`'s array calls proves the route answers the plain 1,024 (serial), 2^16 + 37
/// and 2^21 + 3 (two tasks) cells itself; the ufunc's small-call gate hands calls below 128
/// elements to numpy's ufunc object directly, which the spy cannot see.
#[test]
fn heaviside_float32_route_matches_numpy_bytes_and_events() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(f, a, b, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = f(a, b)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
def delegations(a, b):
    real, calls = np.heaviside, []
    def spy(*args, **kwargs):
        calls.append(isinstance(args[0], np.ndarray))
        return real(*args, **kwargs)
    np.heaviside = spy
    try:
        fnp.heaviside(a, b)
    finally:
        np.heaviside = real
    return sum(calls)
def bits(x):
    return x if isinstance(x, int) else int(np.array([x], np.float32).view(np.uint32)[0])
inf = np.inf
specials = {
    "nan x payloads": [(0x7fc00001, 0.5), (0xffc00002, 0.25), (0x7fc00000, 0x7fc00003)],
    "signaling x": [(0x7fa00000, 0.5)],
    "zero x special h": [
        (-0.0, 0x7fc00003), (0.0, 0xffc00004), (0.0, 0x7fa00004), (-0.0, -0.0), (0.0, inf),
    ],
    "infinities subnormals": [(inf, 0.5), (-inf, 0.5), (0x00000001, 0.5), (0x80000001, 0.5)],
}
rng = np.random.default_rng(47)
cells, bad = 0, []
for n in (1, 17, 1024, (1 << 16) + 37, (1 << 21) + 3):
    a0 = rng.standard_normal(n).astype(np.float32)
    a0[::5] = 0
    b0 = rng.uniform(0, 1, n).astype(np.float32)
    for label, pairs in {"plain": [], **specials}.items():
        if len(pairs) > n:
            continue
        a, b = a0.copy(), b0.copy()
        if pairs:
            a.view(np.uint32)[-len(pairs):] = [bits(p[0]) for p in pairs]
            b.view(np.uint32)[-len(pairs):] = [bits(p[1]) for p in pairs]
        for mode in ("warn", "raise", "ignore"):
            cells += 1
            if outcome(fnp.heaviside, a, b, mode) != outcome(np.heaviside, a, b, mode):
                bad.append(f"n={n} {label} {mode}")
    if n >= 1024 and delegations(a0, b0) != 0:
        bad.append(f"n={n} delegated")
a = rng.standard_normal(1 << 17).astype(np.float32)
a[::3] = 0
b = rng.uniform(0, 1, 1 << 17).astype(np.float32)
layouts = {
    "2-D": (a.reshape(256, 512), b.reshape(256, 512)),
    "broadcast row": (a.reshape(256, 512), b[:512]),
    "strided": (a[::2], b[::2]),
    "mixed float64": (a, b.astype(np.float64)),
    "big-endian": (a.astype(">f4"), b.astype(">f4")),
    "fortran": (np.asfortranarray(a.reshape(256, 512)), np.asfortranarray(b.reshape(256, 512))),
    "python float step": (a, 0.5),
}
for label, (x, y) in layouts.items():
    for mode in ("warn", "raise", "ignore"):
        cells += 1
        if outcome(fnp.heaviside, x, y, mode) != outcome(np.heaviside, x, y, mode):
            bad.append(f"{label} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "87", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "float32 heaviside must match numpy's bytes and events: {result}"
    );
    Ok(())
}

/// float64 heaviside with a SCALAR step value runs its own native kernel, which must give
/// numpy's bytes and events: a NaN `x` becomes numpy's canonical 0x7ff8000000000000 whatever its
/// payload (the kernel used to return the operand itself), and a signaling NaN `x` raises
/// numpy's "invalid" (the kernel used to return it unquieted and silent). Python-float,
/// float64-scalar, negative-zero and NaN step values, at 1,000 elements (serial) and 2^17 + 3
/// (parallel), under errstate warn / raise / ignore.
#[test]
fn heaviside_float64_scalar_step_matches_numpy_bytes_and_events() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(f, a, h, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = f(a, h)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
specials = {
    "nan payloads": [0x7ff8000000000001, 0xfff8000000000000],
    "signaling nan": [0x7ff0000000000001],
    "zeros": [0x0000000000000000, 0x8000000000000000],
    "infinities subnormal": [0x7ff0000000000000, 0xfff0000000000000, 0x0000000000000001],
}
steps = [0.5, np.float64(0.5), -0.0, float("nan")]
rng = np.random.default_rng(73)
cells, bad = 0, []
for n in (1000, (1 << 17) + 3):
    a0 = rng.standard_normal(n)
    a0[::7] = 0
    for label, values in {"plain": [], **specials}.items():
        a = a0.copy()
        if values:
            a.view(np.uint64)[-len(values):] = values
        for h in steps:
            for mode in ("warn", "raise", "ignore"):
                cells += 1
                if outcome(fnp.heaviside, a, h, mode) != outcome(np.heaviside, a, h, mode):
                    bad.append(f"n={n} {label} h={h!r} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "120", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "float64 heaviside with a scalar step must match numpy's bytes and events: {result}"
    );
    Ok(())
}

/// float32 heaviside with a Python-float step value runs a native select pass: numpy narrows the
/// step to float32 (an underflowing one silently, an overflowing one with "overflow encountered in
/// cast", which stays numpy's), returns its canonical NaN for a NaN `x` and raises "invalid" for a
/// signaling one. numpy float32 / float64 and integer steps stay numpy's (float64 makes the result
/// float64). Every cell must match numpy's dtype, bytes and events; a spy proves the route answers
/// a plain 2^17 + 3 call with step 0.5 itself.
#[test]
fn heaviside_float32_scalar_step_matches_numpy_bytes_and_events() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
def outcome(f, a, h, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(all=mode):
                r = f(a, h)
            got = (r.dtype.str, r.shape, r.tobytes())
        except Exception as exc:
            got = ("raise", type(exc).__name__, str(exc))
    return got, sorted(str(w.message) for w in caught)
def delegations(a, h):
    real, calls = np.heaviside, []
    def spy(*args, **kwargs):
        calls.append(isinstance(args[0], np.ndarray))
        return real(*args, **kwargs)
    np.heaviside = spy
    try:
        fnp.heaviside(a, h)
    finally:
        np.heaviside = real
    return sum(calls)
specials = {
    "nan payloads": [0x7fc00001, 0xffc00002],
    "signaling nan": [0x7fa00000],
    "zeros": [0x00000000, 0x80000000],
}
steps = [0.5, 0.1, -0.0, float("nan"), 1e-40, 3.5e38, np.float32(0.5), np.float64(0.5), 1]
rng = np.random.default_rng(79)
cells, bad = 0, []
for n in (1024, (1 << 17) + 3, (1 << 21) + 3):
    a0 = rng.standard_normal(n).astype(np.float32)
    a0[::5] = 0
    for label, values in {"plain": [], **specials}.items():
        a = a0.copy()
        if values:
            a.view(np.uint32)[-len(values):] = values
        for h in steps:
            for mode in ("warn", "raise", "ignore"):
                cells += 1
                if outcome(fnp.heaviside, a, h, mode) != outcome(np.heaviside, a, h, mode):
                    bad.append(f"n={n} {label} h={h!r} {mode}")
    if n > 1 << 16 and delegations(a0, 0.5) != 0:
        bad.append(f"n={n} delegated")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "324", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "float32 heaviside with a scalar step must match numpy's bytes and events: {result}"
    );
    Ok(())
}
