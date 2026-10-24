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

mod support;
use support::fnp_script;

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

/// Every NaN is the canonical `np.nan` (`0x7ff8000000000000`): the route answers it natively, as
/// a numpy.float64 with numpy's bits, under `errstate(all='raise')` (numpy raises nothing) and
/// with keepdims. One NaN of another sign or payload beside it hands the call back to numpy,
/// whose answer still matches.
#[test]
fn min_max_flat_canonical_nan_is_answered_with_numpys_bits() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
band = 1 << 16
checks = []
def same(ours, theirs):
    return type(ours) is type(theirs) and np.asarray(ours).tobytes() == np.asarray(theirs).tobytes() \
        and np.shape(ours) == np.shape(theirs)
def nan_bits(bits):
    return np.frombuffer(np.uint64(bits).tobytes(), dtype=np.float64)[0]
rng = np.random.default_rng(41)
base = rng.standard_normal(n)
for positions in ([0], [5], [band + 7], [n - 1], [3, band * 2, n // 2]):
    a = base.copy()
    a[positions] = np.nan
    with np.errstate(all='raise'):
        for name in ("max", "min", "amax", "amin"):
            checks.append(same(getattr(fnp, name)(a), getattr(np, name)(a)))
        checks.append(same(fnp.max(a, keepdims=True), np.max(a, keepdims=True)))
for other in (0xfff8000000000000, 0x7ff8000000000001, 0x7ff0000000000001):
    a = base.copy()
    a[5] = np.nan
    a[n // 2] = nan_bits(other)
    with np.errstate(all='ignore'):
        checks.append(same(fnp.max(a), np.max(a)))
        checks.append(same(fnp.min(a), np.min(a)))
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 31");
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

// ---------------------------------------------------------------------------
// argmin / argmax. The tie convention is the OPPOSITE of the value routes above
// (FIRST index, not a later element), and unlike them there is no signed-zero
// hazard and no NaN deferral: NumPy returns the FIRST NaN's index for both.
// ---------------------------------------------------------------------------

/// Ordinary random data across the routed regime.
#[test]
fn argmin_argmax_flat_random_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(4242)
bad = []
for n in [2_200_000, 3_000_007, 8_388_608]:
    a = rng.standard_normal(n)
    if int(fnp.argmin(a)) != int(a.argmin()): bad.append(("min", n))
    if int(fnp.argmax(a)) != int(a.argmax()): bad.append(("max", n))
print(len(bad) == 0, bad[:3])
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True []");
    Ok(())
}

/// THE TIE TEST, and it is the MIRROR of the value-route tie test: duplicated
/// extrema must return the FIRST index, where `min`/`max` keep a later element.
/// Duplicates are planted straddling band boundaries so a parallel combine that
/// prefers a later band is caught.
#[test]
fn argmin_argmax_flat_ties_return_first_index() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
band = 1 << 16
checks = []
for pos in [0, 1, band - 1, band, band + 1, 2 * band, n // 2, n - 2]:
    a = np.ones(n); a[pos] = -5.0; a[pos + 1] = -5.0
    checks.append(int(fnp.argmin(a)) == int(a.argmin()) == pos)
    b = np.ones(n); b[pos] = 5.0; b[pos + 1] = 5.0
    checks.append(int(fnp.argmax(b)) == int(b.argmax()) == pos)
# a duplicate spanning a band boundary exactly
c = np.ones(n); c[band - 1] = -7.0; c[band] = -7.0
checks.append(int(fnp.argmin(c)) == int(c.argmin()) == band - 1)
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 17");
    Ok(())
}

/// Signed zeros are NOT a hazard for arg*: the answer is an index and ties go to
/// the lower one regardless of sign, so both plant orders give the same index.
/// This is exactly where the value routes must defer and these must not.
#[test]
fn argmin_argmax_flat_signed_zero_is_index_based() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
band = 1 << 16
checks = []
for lo, hi in [(100, 200), (0, n - 1), (band - 1, band), (band, 2 * band)]:
    for first, second in ((-0.0, 0.0), (0.0, -0.0)):
        z = np.ones(n); z[lo] = first; z[hi] = second
        checks.append(int(fnp.argmin(z)) == int(z.argmin()) == lo)
# all zeros, mixed signs everywhere: still the first index
w = np.zeros(n); w[1::2] = -0.0
checks.append(int(fnp.argmin(w)) == int(w.argmin()) == 0)
checks.append(int(fnp.argmax(w)) == int(w.argmax()) == 0)
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 10");
    Ok(())
}

/// A NaN anywhere makes BOTH argmin and argmax return the FIRST NaN's index,
/// not the extremum's — so these routes resolve NaN directly rather than
/// deferring the way the value routes do.
#[test]
fn argmin_argmax_flat_nan_returns_first_nan_index() -> Result<(), String> {
    let script = fnp_script(
        r#"
n = 2_200_000
band = 1 << 16
checks = []
for pos in [0, 7, band + 3, n // 2, n - 1]:
    a = np.ones(n); a[5] = -99.0; a[n - 3] = 99.0; a[pos] = np.nan
    with np.errstate(all='ignore'):
        checks.append(int(fnp.argmin(a)) == int(a.argmin()))
        checks.append(int(fnp.argmax(a)) == int(a.argmax()))
# several NaNs across bands: the earliest index wins
b = np.ones(n); b[10] = np.nan; b[band + 3] = np.nan; b[n - 5] = np.nan
with np.errstate(all='ignore'):
    checks.append(int(fnp.argmin(b)) == int(b.argmin()) == 10)
    checks.append(int(fnp.argmax(b)) == int(b.argmax()) == 10)
# all NaN
c = np.full(n, np.nan)
with np.errstate(all='ignore'):
    checks.append(int(fnp.argmin(c)) == int(c.argmin()) == 0)
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 13");
    Ok(())
}

/// Deferred regimes must still agree: below the gate, non-contiguous, non-f64,
/// axis, keepdims, out, N-D flattened, and empty (which raises).
#[test]
fn argmin_argmax_flat_deferred_regimes_still_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(31)
checks = []
n = 2_200_000
small = rng.standard_normal(1000)
checks.append(int(fnp.argmin(small)) == int(small.argmin()))
nc = rng.standard_normal(2 * n)[::2]
checks.append(int(fnp.argmin(nc)) == int(nc.argmin()))
f32 = rng.standard_normal(n).astype(np.float32)
checks.append(int(fnp.argmin(f32)) == int(f32.argmin()))
i64 = rng.integers(-10**9, 10**9, n, dtype=np.int64)
checks.append(int(fnp.argmax(i64)) == int(i64.argmax()))
m = rng.standard_normal((1500, 1500))
checks.append(int(fnp.argmin(m)) == int(m.argmin()))
checks.append(np.array_equal(fnp.argmin(m, axis=0), m.argmin(axis=0)))
checks.append(np.array_equal(fnp.argmax(m, axis=1), m.argmax(axis=1)))
def raised(fn):
    try:
        fn(); return False
    except Exception:
        return True
empty = np.zeros(0)
checks.append(raised(lambda: fnp.argmin(empty)) and raised(lambda: empty.argmin()))
print(all(checks), len(checks))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 8");
    Ok(())
}

/// max / min of integers over a contiguous axis run with a small innermost extent
/// (`try_small_extent_integer_extremum`): a 3-channel image over (0, 1) or its per-pixel channel
/// extremes, every integer width, keepdims, the all-axes scalar, and numpy's ValueError for an empty
/// reduction. Other layouts (a (2000, 2000) column max) stay numpy's. A numpy.maximum.reduce spy
/// proves the image route does not delegate.
#[test]
fn integer_extremes_over_small_extent_axis_runs_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(20261005)
bad = []
count = 0
def outcome(fn, a, kw):
    try:
        return fn(a, **kw)
    except Exception as ex:
        return (type(ex).__name__, str(ex))
def check(fname, a, **kw):
    global count
    count += 1
    ours, theirs = outcome(getattr(fnp, fname), a, kw), outcome(getattr(np, fname), a, kw)
    if isinstance(ours, tuple) or isinstance(theirs, tuple):
        ok = isinstance(ours, tuple) and isinstance(theirs, tuple) and ours == theirs
    else:
        o, t = np.asarray(ours), np.asarray(theirs)
        ok = type(ours) is type(theirs) and o.dtype == t.dtype and o.shape == t.shape and o.tobytes() == t.tobytes()
    if not ok:
        bad.append((fname, a.dtype.name, a.shape, kw))
for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
    info = np.iinfo(dtype)
    img = rng.integers(info.min, info.max, (60, 80, 3), dtype=dtype, endpoint=True)
    for ax in [(0, 1), 2, -1, (1, 2), (0, 1, 2)]:
        check("max", img, axis=ax); check("min", img, axis=ax)
        check("max", img, axis=ax, keepdims=True)
    tall = np.where(rng.random((5000, 5)) < 0.5, info.min, info.max).astype(dtype)
    check("min", tall, axis=0); check("amax", tall, axis=0)
check("max", np.zeros((4096, 0, 3), np.uint8), axis=(0, 1))
check("max", np.zeros((0, 5000), np.uint8), axis=0)
gray = rng.integers(0, 256, (400, 500), dtype=np.uint8)
check("max", gray, axis=0); check("min", gray, axis=1)
img = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
expected = np.max(img, axis=(0, 1))

class Poison:
    def reduce(self, *args, **kwargs):
        raise AssertionError("small-extent integer max route unexpectedly delegated")

real_maximum = np.maximum
np.maximum = Poison()
try:
    got = fnp.max(img, axis=(0, 1))
    routed = got.tobytes() == expected.tobytes() and got.dtype == expected.dtype
except AssertionError:
    routed = False
np.maximum = real_maximum
print(bad if bad else True, count, routed)
"#
        .into(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 140 True");
    Ok(())
}
