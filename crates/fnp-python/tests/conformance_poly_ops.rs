//! Conformance tests for numpy polynomial operations against NumPy oracle.
//!
//! Tests polyval, polyder, polyint, polyfit, polyadd, polysub, polymul, polydiv.

use std::io::Write;
use std::process::{Command, Stdio};

fn numpy_oracle(script: &str) -> Result<String, String> {
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("python3 should be available: {error}\nScript: {script}"))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| format!("python3 stdin should be available\nScript: {script}"))?
        .write_all(script.as_bytes())
        .map_err(|error| {
            format!("failed to write NumPy oracle script: {error}\nScript: {script}")
        })?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for NumPy oracle: {error}\nScript: {script}"))?;
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

// ─────────────────────────────────────────────────────────────────────────────
// polyval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyval_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # 1*x^2 + 2*x + 3
x = 2
result = fnp.polyval(p, x)
expected = np.polyval(p, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval basic should match numpy");
    Ok(())
}

#[test]
fn polyval_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, -2, 1]  # (x-1)^2
x = np.array([0, 1, 2, 3])
result = fnp.polyval(p, x)
expected = np.polyval(p, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval array should match numpy");
    Ok(())
}

#[test]
fn polyval_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1.5, -2.5, 0.5]
x = np.array([0.5, 1.5, 2.5])
result = fnp.polyval(p, x)
expected = np.polyval(p, x)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyval float should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polyder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyder_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # x^2 + 2x + 3 -> 2x + 2
result = fnp.polyder(p)
expected = np.polyder(p)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyder basic should match numpy");
    Ok(())
}

#[test]
fn polyder_with_m() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3, 4]  # x^3 + 2x^2 + 3x + 4
result = fnp.polyder(p, m=2)  # second derivative
expected = np.polyder(p, m=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyder with m should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polyint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyint_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [2, 2]  # 2x + 2 -> x^2 + 2x + C
result = fnp.polyint(p)
expected = np.polyint(p)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyint basic should match numpy");
    Ok(())
}

#[test]
fn polyint_with_k() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [2, 2]
result = fnp.polyint(p, k=5)  # constant of integration
expected = np.polyint(p, k=5)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyint with k should match numpy");
    Ok(())
}

#[test]
fn polyint_with_m() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [6]  # 6 -> 3x^2 -> x^3 (double integration)
result = fnp.polyint(p, m=2)
expected = np.polyint(p, m=2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyint with m should match numpy");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Relationship tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyder_polyint_inverse() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # x^2 + 2x + 3
integrated = fnp.polyint(p)
derived = fnp.polyder(integrated)
# Should get back original (up to precision)
print(np.allclose(derived, p))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyder(polyint(p)) should equal p");
    Ok(())
}

#[test]
fn polyval_at_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = [1, 2, 3]  # x^2 + 2x + 3
# At x=0, should equal constant term
result = fnp.polyval(p, 0)
print(np.allclose(result, 3))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polyval at 0 should equal constant term"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polyadd
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyadd_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]  # x^2 + 2x + 3
b = [4, 5]     # 4x + 5
result = fnp.polyadd(a, b)
expected = np.polyadd(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyadd basic should match numpy");
    Ok(())
}

#[test]
fn polyadd_same_degree() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [3, 2, 1]
result = fnp.polyadd(a, b)
expected = np.polyadd(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polyadd same degree should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polysub
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polysub_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [4, 5]
result = fnp.polysub(a, b)
expected = np.polysub(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polysub basic should match numpy");
    Ok(())
}

#[test]
fn polysub_self_gives_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
result = fnp.polysub(a, a)
expected = np.polysub(a, a)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polysub self should give zero");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polymul
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polymul_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 1]  # x + 1
b = [1, -1]  # x - 1
result = fnp.polymul(a, b)  # x^2 - 1
expected = np.polymul(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polymul basic should match numpy");
    Ok(())
}

#[test]
fn polymul_by_constant() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 2, 3]
b = [2]  # multiply by constant 2
result = fnp.polymul(a, b)
expected = np.polymul(a, b)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polymul by constant should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polydiv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polydiv_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 0, -1]  # x^2 - 1
b = [1, -1]     # x - 1
q, r = fnp.polydiv(a, b)
expected_q, expected_r = np.polydiv(a, b)
print(np.allclose(q, expected_q) and np.allclose(r, expected_r))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polydiv basic should match numpy");
    Ok(())
}

#[test]
fn polydiv_no_remainder() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = [1, 3, 2]  # x^2 + 3x + 2 = (x+1)(x+2)
b = [1, 1]     # x + 1
q, r = fnp.polydiv(a, b)
expected_q, expected_r = np.polydiv(a, b)
print(np.allclose(q, expected_q) and np.allclose(r, expected_r, atol=1e-10))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polydiv no remainder should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// polyfit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyfit_linear() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 5, 7, 9])  # y = 2x + 1
result = fnp.polyfit(x, y, 1)
expected = np.polyfit(x, y, 1)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "polyfit linear should match numpy");
    Ok(())
}

#[test]
fn polyfit_quadratic() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 2, 5, 10, 17])  # y = x^2 + 1
result = fnp.polyfit(x, y, 2)
expected = np.polyfit(x, y, 2)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polyfit quadratic should match numpy"
    );
    Ok(())
}

#[test]
fn polyfit_with_weights() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 5, 7, 9])
w = np.array([1, 1, 2, 1, 1])  # weight middle point more
result = fnp.polyfit(x, y, 1, w=w)
expected = np.polyfit(x, y, 1, w=w)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polyfit with weights should match numpy"
    );
    Ok(())
}

#[test]
fn polyfit_roundtrip() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 5, 7, 9])
coeffs = fnp.polyfit(x, y, 1)
y_pred = fnp.polyval(coeffs, x)
print(np.allclose(y, y_pred))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polyfit then polyval should recover y"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar return type tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn polyval_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
p = np.array([1, 2, 3], dtype=np.float64)
x = np.float64(2.0)
fnp_result = fnp.polyval(p, x)
np_result = np.polyval(p, x)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "polyval scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn polyval_native_horner_bitexact_matches_numpy() -> Result<(), String> {
    // Exercises the native fused-Horner polyval fast path against numpy bit-exactly
    // (atol=0, equal_nan=True) incl dtype/shape: various degrees, 1-D and 2-D x,
    // integer coefficients (promote to f64), a single-coefficient (degree 0) poly,
    // negative/large x, and a +-inf/NaN x (numpy's first 0*x+p[0] step makes 0*inf=NaN).
    let script = fnp_script(
        r#"
def same(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)

rng = np.random.default_rng(29)
x1 = rng.standard_normal(100000) * 3.0
x2 = rng.standard_normal((512, 64))
xspec = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0, -2.5, 1e8], dtype=np.float64)
cases = [
    (rng.standard_normal(6), x1),
    (rng.standard_normal(13), x1),
    (rng.standard_normal(4), x2),
    (np.array([3, -2, 5, 1], dtype=np.int64), x1),     # int coeffs -> f64
    (np.array([7.0]), x1),                              # degree 0
    (rng.standard_normal(5), xspec),                    # inf/nan x
]
ok = True
for p, x in cases:
    f = fnp.polyval(p, x)
    n = np.polyval(p, x)
    if not same(f, n):
        print("FAIL", p.shape, x.shape, np.asarray(f).ravel()[:5], np.asarray(n).ravel()[:5]); ok = False
print(ok)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "native Horner polyval parity should match numpy: {result}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty-coefficient contract for the polynomial families (deadlock-audit-fb171)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn family_add_sub_mul_reject_empty_coefficient_arrays_like_numpy() -> Result<(), String> {
    // deadlock-audit-fb171: the fnp-ufunc Rust kernels behind the five polynomial
    // families return an EMPTY series where numpy raises ValueError("Coefficient
    // array is empty"). Today that divergence is unreachable from Python because
    // all fifteen bindings are numpy passthroughs, so the user-visible behaviour is
    // already correct — and that is exactly the problem this test exists to solve.
    // The risk is not the current code, it is the next person who wires one of these
    // names to the native kernel for speed and silently inherits the divergence.
    // That is not hypothetical: hermeder/hermeint regressed this way before mlz0s,
    // where the remedy was to make the PYTHON layer defer on empty so numpy still
    // raises. This is the tripwire for that class, and it fails the moment a native
    // wiring stops deferring.
    //
    // It asks numpy on every case rather than transcribing what numpy is believed to
    // do (the deadlock-audit-m3gbv finding): the oracle call and the fnp call are in
    // the same script, on the same inputs, in the same interpreter.
    let script = fnp_script(
        r#"
import numpy.polynomial as NP

# (fnp attribute, numpy submodule, function name). The five *basis* families only:
# the legacy top-level polyadd/polysub/polymul are a DIFFERENT contract and are
# checked separately below, precisely so a guard that lumps all eighteen together
# cannot pass.
FAMILIES = [
    ("cheb",  NP.chebyshev),
    ("herm",  NP.hermite),
    ("herme", NP.hermite_e),
    ("lag",   NP.laguerre),
    ("leg",   NP.legendre),
]
OPS = ["add", "sub", "mul"]

# Empty lhs, empty rhs, and both empty: numpy's _as_series rejects any empty input,
# so all three must raise. A wrong implementation that only guards the first
# argument passes the first shape and fails the second.
SHAPES = {
    "empty_lhs":  ([], [1.0, 2.0]),
    "empty_rhs":  ([1.0, 2.0], []),
    "both_empty": ([], []),
}

def outcome(fn, a, b):
    """'ok:<repr>' on success, 'raise:<ExcType>' on failure."""
    try:
        return "ok:" + repr(np.asarray(fn(a, b)).tolist())
    except Exception as exc:
        return "raise:" + type(exc).__name__

failures = []

for prefix, module in FAMILIES:
    for op in OPS:
        name = prefix + op
        ours = getattr(fnp, name)
        theirs = getattr(module, name)

        for label, (a, b) in SHAPES.items():
            got = outcome(ours, a, b)
            want = outcome(theirs, a, b)
            # Pin the oracle's own behaviour too. If numpy ever stops raising here
            # this test must report THAT, not quietly re-baseline onto it.
            if want != "raise:ValueError":
                failures.append(
                    "%s %s: numpy no longer raises ValueError on empty input, it gave %s"
                    % (name, label, want)
                )
            if got != want:
                failures.append("%s %s: fnp %s vs numpy %s" % (name, label, got, want))

        # The negative case a naive "always raise" implementation fails: non-empty
        # coefficients must still compute, and agree with numpy elementwise.
        got = outcome(ours, [1.0, 2.0, 3.0], [4.0, 5.0])
        want = outcome(theirs, [1.0, 2.0, 3.0], [4.0, 5.0])
        if got != want:
            failures.append("%s nonempty: fnp %s vs numpy %s" % (name, got, want))

# Contrast: np.polyadd/polysub/polymul are the legacy poly1d-order helpers and they
# do NOT raise on empty input — np.polyadd([], [1.0]) is array([1.]). They are in
# this test as the discriminator: a guard that made "empty coefficients raise" a
# blanket rule across every add/sub/mul name would be wrong, and would fail here.
for name in ["polyadd", "polysub", "polymul"]:
    ours = getattr(fnp, name)
    theirs = getattr(np, name)
    for label, (a, b) in SHAPES.items():
        got = outcome(ours, a, b)
        want = outcome(theirs, a, b)
        if got != want:
            failures.append("%s %s: fnp %s vs numpy %s" % (name, label, got, want))
    if outcome(theirs, [], [1.0]).startswith("raise:"):
        failures.append(
            "legacy polyadd now raises on empty input; the contrast case in this "
            "test is stale and the families/legacy split must be re-derived"
        )

if failures:
    print("FAILURES\n" + "\n".join(failures))
else:
    print("True")
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "polynomial family add/sub/mul must match numpy's empty-coefficient contract \
         (deadlock-audit-fb171):\n{result}"
    );
    Ok(())
}
