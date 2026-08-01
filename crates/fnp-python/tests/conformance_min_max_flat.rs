//! Conformance tests for the parallel flat `float64` min/max fast path.
//!
//! Selection is order-independent, so unlike the sum route there is no
//! accumulation tree to reproduce. The entire risk is TIE SEMANTICS, and the
//! ties are observable because `-0.0` and `+0.0` compare EQUAL while differing
//! in bits. Verified against live NumPy: `minimum(a, b)` is
//! `(a < b || isnan(a)) ? a : b`, which returns the SECOND operand on a tie, so
//! a left-to-right reduce keeps the LAST tied element.
//!
//! These assert BIT equality, not closeness — a min that returns the correctly
//! valued but wrongly signed zero is wrong here.

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
         spec = importlib.util.spec_from_file_location('fnp_python', {module_literal})\n\
         fnp = importlib.util.module_from_spec(spec)\n\
         spec.loader.exec_module(fnp)\n\
         {body}"
    )
}

/// Ordinary random data across the routed regime, bit-identical.
#[test]
fn min_max_flat_random_is_bit_identical() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(2026)
bad = []
for n in [2_200_000, 3_000_007, 8_388_608]:
    a = rng.standard_normal(n)
    for name, ours, theirs in (("min", fnp.min(a), a.min()), ("max", fnp.max(a), a.max())):
        if np.float64(ours).tobytes() != np.float64(theirs).tobytes():
            bad.append((name, n))
print(len(bad) == 0, bad[:3])
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True []");
    Ok(())
}

/// THE TIE TEST. `-0.0` and `+0.0` compare equal, so which one comes back is
/// determined purely by the reduction's tie rule — NumPy keeps the LAST. The
/// signed zero is placed at several positions, including as the final element
/// and spanning band boundaries, so a parallel split that combines out of index
/// order is caught. Also asserts the two zeros really do differ in bits, so the
/// test cannot pass vacuously.
#[test]
fn min_max_flat_signed_zero_tie_keeps_numpys_choice() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
band = 1 << 16
checks = []
checks.append(np.float64(0.0).tobytes() != np.float64(-0.0).tobytes())
positions = [0, 1, band - 1, band, band + 1, 2 * band, n // 2, n - 2, n - 1]
for pos in positions:
    for first, second in ((0.0, -0.0), (-0.0, 0.0)):
        a = np.ones(n)
        a[pos] = first
        a[(pos + 1) % n] = second
        checks.append(np.float64(fnp.min(a)).tobytes() == np.float64(a.min()).tobytes())
    # all-zero buffers with a single opposite-signed zero somewhere
    z = np.zeros(n)
    z[pos] = -0.0
    checks.append(np.float64(fnp.min(z)).tobytes() == np.float64(z.min()).tobytes())
    checks.append(np.float64(fnp.max(z)).tobytes() == np.float64(z.max()).tobytes())
    w = np.full(n, -0.0)
    w[pos] = 0.0
    checks.append(np.float64(fnp.min(w)).tobytes() == np.float64(w.min()).tobytes())
    checks.append(np.float64(fnp.max(w)).tobytes() == np.float64(w.max()).tobytes())
fails = [i for i, ok in enumerate(checks) if not ok]
print(all(checks), len(checks), fails[:5])
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 55 []");
    Ok(())
}

/// Uniform buffers where EVERY element ties. The returned bits must still match
/// NumPy exactly, which pins the tie rule end to end rather than at one index.
#[test]
fn min_max_flat_all_tied_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
checks = []
for fill in (-0.0, 0.0, 1.0, -1.0, np.inf, -np.inf):
    a = np.full(n, fill)
    with np.errstate(all='ignore'):
        checks.append(np.float64(fnp.min(a)).tobytes() == np.float64(a.min()).tobytes())
        checks.append(np.float64(fnp.max(a)).tobytes() == np.float64(a.max()).tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 12");
    Ok(())
}

/// NaN propagates and its PAYLOAD is preserved: the first NaN encountered takes
/// over and sticks. A parallel scan must not lose the payload or pick a
/// different NaN, so several payloads are planted at different bands.
#[test]
fn min_max_flat_nan_propagates_with_payload() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
band = 1 << 16
checks = []
def nan_with(payload):
    return np.frombuffer(np.uint64(0x7ff8000000000000 | payload).tobytes(), dtype=np.float64)[0]
for pos in [0, 1, band + 7, n // 3, n - 1]:
    for payload in (1, 0x2a, 0xdead):
        a = np.ones(n)
        a[pos] = nan_with(payload)
        with np.errstate(all='ignore'):
            checks.append(np.float64(fnp.min(a)).tobytes() == np.float64(a.min()).tobytes())
            checks.append(np.float64(fnp.max(a)).tobytes() == np.float64(a.max()).tobytes())
# several NaNs: the FIRST one wins
b = np.ones(n)
b[10] = nan_with(1); b[band + 3] = nan_with(2); b[n - 5] = nan_with(3)
with np.errstate(all='ignore'):
    checks.append(np.float64(fnp.min(b)).tobytes() == np.float64(b.min()).tobytes())
    checks.append(np.float64(fnp.max(b)).tobytes() == np.float64(b.max()).tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 32");
    Ok(())
}

/// Infinities and mixed extremes, plus N-D C-contiguous input (flattened by
/// NumPy, so it is routed rather than deferred).
#[test]
fn min_max_flat_infinities_and_ndim() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(7)
n = 2_200_000
checks = []
a = rng.standard_normal(n)
a[5] = np.inf; a[n - 9] = -np.inf
with np.errstate(all='ignore'):
    checks.append(np.float64(fnp.min(a)).tobytes() == np.float64(a.min()).tobytes())
    checks.append(np.float64(fnp.max(a)).tobytes() == np.float64(a.max()).tobytes())
for shape in [(1500, 1500), (3, 800_000), (2, 2, 600_000)]:
    m = rng.standard_normal(shape)
    checks.append(np.float64(fnp.min(m)).tobytes() == np.float64(m.min()).tobytes())
    checks.append(np.float64(fnp.max(m)).tobytes() == np.float64(m.max()).tobytes())
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 8");
    Ok(())
}

/// Everything outside the routed regime must defer and still agree: below the
/// size gate, non-contiguous, non-f64, keepdims, axis, out, initial, and empty
/// (which must raise exactly as NumPy does).
#[test]
fn min_max_flat_deferred_regimes_still_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(11)
checks = []
small = rng.standard_normal(1000)
checks.append(np.float64(fnp.min(small)).tobytes() == np.float64(small.min()).tobytes())

n = 2_200_000
nc = rng.standard_normal(2 * n)[::2]
checks.append(np.float64(fnp.min(nc)).tobytes() == np.float64(nc.min()).tobytes())

f32 = rng.standard_normal(n).astype(np.float32)
checks.append(np.float32(fnp.min(f32)).tobytes() == np.float32(f32.min()).tobytes())

i64 = rng.integers(-10**9, 10**9, n, dtype=np.int64)
checks.append(int(fnp.min(i64)) == int(i64.min()))

a = rng.standard_normal(n)
checks.append(np.array_equal(fnp.min(a, keepdims=True), a.min(keepdims=True)))

m = rng.standard_normal((1500, 1500))
checks.append(np.array_equal(fnp.min(m, axis=0), m.min(axis=0)))
checks.append(np.array_equal(fnp.max(m, axis=1), m.max(axis=1)))

o1 = np.zeros((), dtype=np.float64); o2 = np.zeros((), dtype=np.float64)
fnp.min(a, out=o1); np.min(a, out=o2)
checks.append(o1.tobytes() == o2.tobytes())

checks.append(np.float64(fnp.min(a, initial=-1e9)).tobytes()
              == np.float64(a.min(initial=-1e9)).tobytes())

def raised(fn):
    try:
        fn(); return False
    except Exception:
        return True
empty = np.zeros(0)
checks.append(raised(lambda: fnp.min(empty)) and raised(lambda: empty.min()))

print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 10");
    Ok(())
}
