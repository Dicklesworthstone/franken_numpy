//! Conformance tests for numpy.arctan2 against NumPy oracle.
//!
//! Tests two-argument arctangent (quadrant-aware atan):
//! - arctan2(y, x): angle in radians between positive x-axis and point (x, y)

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
fn arctan2_basic_quadrants_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("1.0", "1.0"),   // Q1: 45 degrees
        ("1.0", "-1.0"),  // Q2: 135 degrees
        ("-1.0", "-1.0"), // Q3: -135 degrees
        ("-1.0", "1.0"),  // Q4: -45 degrees
        ("0.0", "1.0"),   // positive x-axis
        ("0.0", "-1.0"),  // negative x-axis (pi)
        ("1.0", "0.0"),   // positive y-axis (pi/2)
        ("-1.0", "0.0"),  // negative y-axis (-pi/2)
    ];

    for (y, x) in &test_cases {
        let script = format!("import numpy as np; print(np.arctan2({y}, {x}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val: f64 = numpy_result.parse().map_err(|e| format!("{e}"))?;

        let rust_script = fnp_script(format!("print(fnp.arctan2({y}, {x}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val: f64 = rust_result.parse().map_err(|e| format!("{e}"))?;

        assert!(
            (numpy_val - rust_val).abs() < 1e-15,
            "arctan2({y}, {x}) mismatch: numpy={numpy_val}, rust={rust_val}"
        );
    }

    Ok(())
}

#[test]
fn arctan2_arrays_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("np.array([1, 2, 3, 4, 5])", "np.array([5, 4, 3, 2, 1])"),
        (
            "np.array([1.0, -1.0, 1.0, -1.0])",
            "np.array([1.0, 1.0, -1.0, -1.0])",
        ),
        (
            "np.array([0.0, 0.0, 1.0, -1.0])",
            "np.array([1.0, -1.0, 0.0, 0.0])",
        ),
        ("np.linspace(-1, 1, 10)", "np.linspace(-1, 1, 10)"),
        ("np.linspace(-10, 10, 20)", "np.ones(20)"),
    ];

    for (y_expr, x_expr) in &test_cases {
        let script = format!("import numpy as np; print(np.arctan2({y_expr}, {x_expr}).tolist())");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_vals = parse_float_list(&numpy_result)?;

        let rust_script = fnp_script(format!("print(fnp.arctan2({y_expr}, {x_expr}).tolist())"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_vals = parse_float_list(&rust_result)?;

        assert!(
            floats_close(&numpy_vals, &rust_vals, 1e-10),
            "arctan2 array mismatch for ({y_expr}, {x_expr})\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
        );
    }

    Ok(())
}

#[test]
fn arctan2_special_values_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([0.0, 0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, np.inf, -np.inf])
x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, -np.inf])
print(np.arctan2(y, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
y = np.array([0.0, 0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, np.inf, -np.inf])
x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, -np.inf])
print(fnp.arctan2(y, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 special values mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn arctan2_2d_broadcasting_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([[1], [2], [3]])
x = np.array([1, 2, 3])
print(np.arctan2(y, x).flatten().tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
y = np.array([[1], [2], [3]])
x = np.array([1, 2, 3])
print(fnp.arctan2(y, x).flatten().tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 2d broadcasting mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn arctan2_negative_zero_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([0.0, -0.0, 0.0, -0.0])
x = np.array([1.0, 1.0, -1.0, -1.0])
print(np.arctan2(y, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
y = np.array([0.0, -0.0, 0.0, -0.0])
x = np.array([1.0, 1.0, -1.0, -1.0])
print(fnp.arctan2(y, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 negative zero mismatch\nnumpy: {numpy_vals:?}\nrust: {rust_vals:?}"
    );

    Ok(())
}

#[test]
fn arctan2_pi_values_match_numpy() -> Result<(), String> {
    let test_cases = vec![
        ("0.0", "1.0", 0.0),                           // 0
        ("1.0", "0.0", std::f64::consts::FRAC_PI_2),   // pi/2
        ("0.0", "-1.0", std::f64::consts::PI),         // pi
        ("-1.0", "0.0", -std::f64::consts::FRAC_PI_2), // -pi/2
    ];

    for (y, x, expected) in &test_cases {
        let script = format!("import numpy as np; print(np.arctan2({y}, {x}))");
        let numpy_result = numpy_oracle(&script)?;
        let numpy_val: f64 = numpy_result.parse().map_err(|e| format!("{e}"))?;

        let rust_script = fnp_script(format!("print(fnp.arctan2({y}, {x}))"));
        let rust_result = numpy_oracle(&rust_script)?;
        let rust_val: f64 = rust_result.parse().map_err(|e| format!("{e}"))?;

        assert!(
            (numpy_val - rust_val).abs() < 1e-15,
            "arctan2({y}, {x}) mismatch: numpy={numpy_val}, rust={rust_val}, expected={expected}"
        );
        assert!(
            (numpy_val - expected).abs() < 1e-15,
            "arctan2({y}, {x}) expected {expected}, got numpy={numpy_val}"
        );
    }

    Ok(())
}

#[test]
fn arctan2_50_random_inputs_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
np.random.seed(42)
y = np.random.randn(50) * 100
x = np.random.randn(50) * 100
print(np.arctan2(y, x).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;
    let numpy_vals = parse_float_list(&numpy_result)?;

    let rust_script = fnp_script(
        r#"
np.random.seed(42)
y = np.random.randn(50) * 100
x = np.random.randn(50) * 100
print(fnp.arctan2(y, x).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;
    let rust_vals = parse_float_list(&rust_result)?;

    assert!(
        floats_close(&numpy_vals, &rust_vals, 1e-10),
        "arctan2 random 50 inputs mismatch\nnumpy len: {}\nrust len: {}",
        numpy_vals.len(),
        rust_vals.len()
    );

    Ok(())
}

#[test]
fn arctan2_empty_array_match_numpy() -> Result<(), String> {
    let script = "import numpy as np; print(np.arctan2(np.array([]), np.array([])).tolist())";
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script("print(fnp.arctan2(np.array([]), np.array([])).tolist())".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "arctan2 empty array mismatch"
    );

    Ok(())
}

#[test]
fn arctan2_dtype_match_numpy() -> Result<(), String> {
    let script =
        "import numpy as np; print(np.arctan2(np.array([1, 2, 3]), np.array([1, 2, 3])).dtype)";
    let numpy_result = numpy_oracle(script)?;

    let rust_script =
        fnp_script("print(fnp.arctan2(np.array([1, 2, 3]), np.array([1, 2, 3])).dtype)".into());
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "arctan2 dtype mismatch"
    );

    Ok(())
}

#[test]
fn arctan2_scalar_broadcast_match_numpy() -> Result<(), String> {
    let script = r#"
import numpy as np
y = np.array([1, 2, 3, 4, 5])
print(np.arctan2(y, 1.0).tolist())
print(np.arctan2(1.0, y).tolist())
"#;
    let numpy_result = numpy_oracle(script)?;

    let rust_script = fnp_script(
        r#"
y = np.array([1, 2, 3, 4, 5])
print(fnp.arctan2(y, 1.0).tolist())
print(fnp.arctan2(1.0, y).tolist())
"#
        .into(),
    );
    let rust_result = numpy_oracle(&rust_script)?;

    assert_eq!(
        numpy_result.trim(),
        rust_result.trim(),
        "arctan2 scalar broadcast mismatch"
    );

    Ok(())
}

#[test]
fn arctan2_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
y = np.float64(1.0)
x = np.float64(2.0)
fnp_result = fnp.arctan2(y, x)
np_result = np.arctan2(y, x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "arctan2 scalar return type should match numpy: {result}"
    );
    Ok(())
}

/// Locks the zero-copy PARALLEL arctan2 path (len >= 16384): a large deterministic
/// f64 array (above the parallel gate) with NaN/+-inf/+-0 must be byte-identical to
/// numpy.arctan2, plus a sha256 golden over the fnp bytes. op.apply (parallel) is
/// the same per-element atan2, so this is bit-identical to the serial path.
#[test]
fn arctan2_zerocopy_parallel_matches_numpy_bytes_and_golden() -> Result<(), String> {
    let script = fnp_script(
        r#"
import hashlib
s = 0x2545F4914F6CDD1D
def nxt():
    global s
    s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return s
n = 40000
y = np.empty(n, dtype=np.float64)
x = np.empty(n, dtype=np.float64)
for i in range(n):
    y[i] = ((nxt() >> 11) / (1 << 53)) * 12.0 - 6.0
    x[i] = ((nxt() >> 11) / (1 << 53)) * 12.0 - 6.0
for j, v in ((3, np.nan), (5, np.inf), (7, -np.inf), (9, 0.0), (11, -0.0)):
    y[j] = v
for j, v in ((13, 0.0), (17, -0.0), (19, np.inf), (23, -np.inf)):
    x[j] = v
# The oracle arm must really be numpy's, or the byte-equality line would be vacuous.
assert np.arctan2 is not fnp.arctan2, "oracle arm is fnp"
r = np.asarray(fnp.arctan2(y, x))
e = np.arctan2(y, x)
print(r.shape == e.shape and r.tobytes() == e.tobytes())
print(hashlib.sha256(r.tobytes()).hexdigest())
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let mut lines = result.lines();
    // Byte-equality with the LIVE numpy is the contract. The digest (reported on line 2, not
    // asserted) is host-dependent: 77a1486c... where it was pinned, 59ad6f07... on CI G2's GitHub
    // runner (numpy 2.4.6) with the byte-equality line still True there, so pinning it failed
    // G2 on every host but the pinning one.
    assert_eq!(
        lines.next().unwrap_or("").trim(),
        "True",
        "zero-copy parallel arctan2 must be byte-identical to numpy.arctan2: {result}"
    );
    Ok(())
}

/// The float32 route calls atan2f per element in parallel from 2^16 elements, but only where
/// numpy's own float32 loop is the scalar baseline that calls atan2f (its byte probe decides; an
/// avx512f host's SVML loop is not libm's). Either way every cell must match numpy's bytes and
/// events: atan2f's underflow (a tiny over a huge operand) is replayed through numpy, a
/// signaling NaN defers, signed zeros keep their quadrants. A spy counting `np.arctan2`'s array
/// calls checks the route answers 2^16 + 37 and 2^20 + 3 itself exactly where numpy's loop is the
/// baseline on a host without avx512f, and leaves 2^15 to numpy.
#[test]
fn arctan2_float32_route_matches_numpy_bytes_and_events() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
from numpy.lib.introspect import opt_func_info
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
    real, calls = np.arctan2, []
    def spy(*args, **kwargs):
        calls.append(isinstance(args[0], np.ndarray))
        return real(*args, **kwargs)
    np.arctan2 = spy
    try:
        fnp.arctan2(a, b)
    finally:
        np.arctan2 = real
    return sum(calls)
def bits(x):
    return x if isinstance(x, int) else int(np.array([x], np.float32).view(np.uint32)[0])
try:
    avx512f = "avx512f" in open("/proc/cpuinfo").read().split()
except OSError:
    avx512f = True
loops = opt_func_info(func_name="arctan2", signature="float32")["arctan2"].values()
native = not avx512f and all(loop["current"].startswith("baseline") for loop in loops)
inf = np.inf
specials = {
    "signed zero axes": [
        (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0), (-0.0, 0.0), (1.0, 0.0), (-1.0, -0.0),
    ],
    "underflow": [(1e-30, 1e30), (0x00000001, 1.0), (-3e-39, 1.0)],
    "infinities": [(inf, inf), (-inf, inf), (inf, -inf), (1.0, inf), (1.0, -inf)],
    "nan payloads": [(0x7fc00001, 1.0), (1.0, 0xffc00002), (0x7fc00001, 0xffc00002)],
    "signaling nan": [(0x7fa00000, 1.0)],
}
rng = np.random.default_rng(53)
cells, bad = 0, []
for n in (1 << 15, (1 << 16) + 37, (1 << 20) + 3):
    a0 = (rng.standard_normal(n) * 10).astype(np.float32)
    b0 = (rng.standard_normal(n) * 10).astype(np.float32)
    for label, pairs in {"plain": [], **specials}.items():
        a, b = a0.copy(), b0.copy()
        if pairs:
            a.view(np.uint32)[-len(pairs):] = [bits(p[0]) for p in pairs]
            b.view(np.uint32)[-len(pairs):] = [bits(p[1]) for p in pairs]
        for mode in ("warn", "raise", "ignore"):
            cells += 1
            if outcome(fnp.arctan2, a, b, mode) != outcome(np.arctan2, a, b, mode):
                bad.append(f"n={n} {label} {mode}")
    expected = 0 if native and n > 1 << 16 else 1
    if delegations(a0, b0) != expected:
        bad.append(f"n={n} delegations != {expected} (native={native})")
a = (rng.standard_normal(1 << 17) * 10).astype(np.float32)
b = (rng.standard_normal(1 << 17) * 10).astype(np.float32)
layouts = {
    "2-D": (a.reshape(256, 512), b.reshape(256, 512)),
    "broadcast row": (a.reshape(256, 512), b[:512]),
    "strided": (a[::2], b[::2]),
    "mixed float64": (a, b.astype(np.float64)),
    "big-endian": (a.astype(">f4"), b.astype(">f4")),
}
for label, (x, y) in layouts.items():
    for mode in ("warn", "raise", "ignore"):
        cells += 1
        if outcome(fnp.arctan2, x, y, mode) != outcome(np.arctan2, x, y, mode):
            bad.append(f"{label} {mode}")
print(cells, bad)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    let (cells, bad) = result.trim().split_once(' ').unwrap_or(("0", &result));
    assert_eq!(cells, "69", "cell table drifted: {result}");
    assert_eq!(
        bad, "[]",
        "float32 arctan2 must match numpy's bytes and events: {result}"
    );
    Ok(())
}
