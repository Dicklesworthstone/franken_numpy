//! Conformance tests for numpy window functions against NumPy oracle.
//!
//! Tests bartlett, blackman, hamming, hanning, kaiser.

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
// bartlett
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bartlett_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.bartlett(10)
expected = np.bartlett(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bartlett basic should match numpy");
    Ok(())
}

#[test]
fn bartlett_small() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.bartlett(3)
expected = np.bartlett(3)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "bartlett small should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// blackman
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn blackman_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.blackman(10)
expected = np.blackman(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "blackman basic should match numpy");
    Ok(())
}

#[test]
fn blackman_large() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.blackman(100)
expected = np.blackman(100)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "blackman large should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hamming
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hamming_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hamming(10)
expected = np.hamming(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hamming basic should match numpy");
    Ok(())
}

#[test]
fn hamming_small() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hamming(5)
expected = np.hamming(5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hamming small should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// hanning
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hanning_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hanning(10)
expected = np.hanning(10)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hanning basic should match numpy");
    Ok(())
}

#[test]
fn hanning_symmetry() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.hanning(11)  # odd length for symmetry
# Check that the window is symmetric
mid = len(result) // 2
print(np.allclose(result[:mid], result[-1:-mid-1:-1]))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "hanning should be symmetric");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// kaiser
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn kaiser_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.kaiser(10, 14)  # M=10, beta=14
expected = np.kaiser(10, 14)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kaiser basic should match numpy");
    Ok(())
}

#[test]
fn kaiser_low_beta() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.kaiser(10, 0)  # beta=0 should give rectangular window
expected = np.kaiser(10, 0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kaiser low beta should match numpy");
    Ok(())
}

#[test]
fn kaiser_high_beta() -> Result<(), String> {
    let script = fnp_script(
        r#"
result = fnp.kaiser(20, 30)  # high beta for more attenuation
expected = np.kaiser(20, 30)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "kaiser high beta should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn window_endpoints_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Bartlett, Blackman, Hanning should start and end near 0
bart = fnp.bartlett(10)
black = fnp.blackman(10)
hann = fnp.hanning(10)
# Check first and last elements are small
print(bart[0] < 0.1 and bart[-1] < 0.1)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "bartlett endpoints should be near zero"
    );
    Ok(())
}

#[test]
fn hamming_vs_hanning_center() -> Result<(), String> {
    let script = fnp_script(
        r#"
# Both should peak in the center
ham = fnp.hamming(11)
hann = fnp.hanning(11)
mid = len(ham) // 2
# Both should have maximum at center
print(np.argmax(ham) == mid and np.argmax(hann) == mid)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "hamming and hanning should peak at center"
    );
    Ok(())
}

// numpy spells the window length `M`, and its docs write np.hamming(M=51). A
// wrapper that names the parameter `m` turns that documented call into a
// TypeError, so this pins the keyword spelling for all five windows alongside
// the positional form and the negative/zero lengths numpy special-cases.
//
// Everything here is compared on BYTES, not tolerance: dtype, shape, error
// class and the raw output buffer, plus an exact fnp(M=k) == fnp(k) check for
// the keyword binding itself and an explicit window == window[::-1] symmetry
// assertion. The tolerance version of this test is what caught
// deadlock-audit-window-ulp-asymmetry-sdcoh (the kernels evaluated
// 0.5 - 0.5*cos(2*pi*i/(M-1)) where numpy evaluates 0.5 + 0.5*cos(pi*n/(M-1))
// over n = arange(1-M, M, 2), leaving them 1 ULP off and asymmetric); the byte
// comparison is what keeps it closed.
#[test]
fn window_length_keyword_is_capital_m_like_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import platform

def keyword_call(fn, length):
    return fn(M=length)

def positional_call(fn, length):
    return fn(length)

def lowercase_keyword_call(fn, length):
    return fn(m=length)

cases = []
for name in ["bartlett", "blackman", "hamming", "hanning"]:
    for length in [-3, 0, 1, 2, 7, 12]:
        cases.append((f"{name} M={length}", name, keyword_call, length))
        cases.append((f"{name} positional {length}", name, positional_call, length))
    cases.append((f"{name} lowercase m=", name, lowercase_keyword_call, 7))

def kaiser_keyword(fn, length):
    return fn(M=length, beta=8.6)

def kaiser_positional(fn, length):
    return fn(length, 8.6)

def kaiser_lowercase(fn, length):
    return fn(m=length, beta=8.6)

for length in [-3, 0, 1, 2, 7, 12]:
    cases.append((f"kaiser M={length}", "kaiser", kaiser_keyword, length))
    cases.append((f"kaiser positional {length}", "kaiser", kaiser_positional, length))
cases.append(("kaiser lowercase m=", "kaiser", kaiser_lowercase, 7))

def outcome(module, name, call, length):
    try:
        result = call(getattr(module, name), length)
        return ("ok", str(result.dtype), tuple(result.shape), result.tobytes().hex())
    except Exception as exc:
        return ("err", type(exc).__name__)

ok = True
for label, name, call, length in cases:
    actual = outcome(fnp, name, call, length)
    expected = outcome(np, name, call, length)
    if actual != expected:
        print(label)
        print(actual)
        print(expected)
        ok = False

# The keyword binding itself is exact: passing the length by keyword must
# produce the identical bytes to passing it positionally.
for name in ["bartlett", "blackman", "hamming", "hanning"]:
    for length in [-3, 0, 1, 2, 7, 12, 33]:
        by_keyword = getattr(fnp, name)(M=length).tobytes()
        by_position = getattr(fnp, name)(length).tobytes()
        if by_keyword != by_position:
            print(f"{name} M={length} keyword/positional byte mismatch")
            ok = False
for length in [-3, 0, 1, 2, 7, 12, 33]:
    if fnp.kaiser(M=length, beta=8.6).tobytes() != fnp.kaiser(length, 8.6).tobytes():
        print(f"kaiser M={length} keyword/positional byte mismatch")
        ok = False

# numpy's windows are symmetric BY CONSTRUCTION (it evaluates them over the
# symmetric index arange(1-M, M, 2), and cos is even). A window that is merely
# close to numpy can still be asymmetric, which is what this catches - and
# asymmetry is a property loss in its own right for overlap-add reconstruction.
for name in ["bartlett", "blackman", "hamming", "hanning"]:
    for length in [2, 3, 4, 5, 7, 8, 12, 17, 33, 64, 101, 255]:
        window = getattr(fnp, name)(length)
        if window.tobytes() != window[::-1].copy().tobytes():
            print(f"{name}({length}) is not symmetric")
            ok = False
for length in [2, 3, 4, 5, 7, 8, 12, 17, 33, 64, 101, 255]:
    window = fnp.kaiser(length, 8.6)
    if window.tobytes() != window[::-1].copy().tobytes():
        print(f"kaiser({length}) is not symmetric")
        ok = False

# Byte parity across a wider size sweep than the keyword grid above, since the
# ULP divergence only showed up from M >= 4 and grew with M.
for name in ["bartlett", "blackman", "hamming", "hanning"]:
    for length in [2, 3, 4, 5, 7, 8, 12, 17, 33, 64, 101, 255, 1024]:
        if getattr(fnp, name)(length).tobytes() != getattr(np, name)(length).tobytes():
            print(f"{name}({length}) bytes differ from numpy")
            ok = False
for length in [2, 3, 4, 5, 7, 8, 12, 17, 33, 64, 101, 255, 1024]:
    for beta in [0.0, 2.5, 8.6, 14.0]:
        if fnp.kaiser(length, beta).tobytes() != np.kaiser(length, beta).tobytes():
            print(f"kaiser({length}, {beta}) bytes differ from numpy")
            ok = False

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
        "window length keyword should match numpy ({provenance}): {result}"
    );
    Ok(())
}
