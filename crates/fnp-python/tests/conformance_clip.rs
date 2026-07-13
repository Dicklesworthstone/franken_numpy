//! Conformance tests for numpy.clip against NumPy oracle.
//!
//! Tests clip (clip array values to range).

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

#[test]
fn clip_basic() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
result = fnp.clip(a, 3, 7)
expected = np.clip(a, 3, 7)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip basic should match numpy");
    Ok(())
}

#[test]
fn clip_float() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
result = fnp.clip(a, 1.0, 4.0)
expected = np.clip(a, 1.0, 4.0)
print(np.allclose(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip float should match numpy");
    Ok(())
}

#[test]
fn clip_min_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.clip(a, 3, None)
expected = np.clip(a, 3, None)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip with min only should match numpy"
    );
    Ok(())
}

#[test]
fn clip_max_only() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1, 2, 3, 4, 5])
result = fnp.clip(a, None, 3)
expected = np.clip(a, None, 3)
print(np.array_equal(result, expected))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip with max only should match numpy"
    );
    Ok(())
}

#[test]
fn clip_scalar_return_type_matches_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
x = np.float64(5.0)
fnp_result = fnp.clip(x, 2.0, 8.0)
np_result = np.clip(x, 2.0, 8.0)
print(type(fnp_result).__name__ == type(np_result).__name__, fnp_result, np_result)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert!(
        result.trim().starts_with("True"),
        "clip scalar return type should match numpy: {result}"
    );
    Ok(())
}

#[test]
fn clip_complex() -> Result<(), String> {
    let script = fnp_script(
        r#"
z = np.array([1+1j, 5+5j, 2+2j], dtype=np.complex128)
fnp_result = fnp.clip(z, 0+0j, 3+3j)
np_result = np.clip(z, 0+0j, 3+3j)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip complex should match numpy");
    Ok(())
}

#[test]
fn clip_nan_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, np.nan, 3.0, 5.0])
fnp_result = fnp.clip(a, 2.0, 4.0)
np_result = np.clip(a, 2.0, 4.0)
# NaN should propagate through clip
print(np.allclose(fnp_result, np_result, equal_nan=True))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip nan handling should match numpy"
    );
    Ok(())
}

#[test]
fn clip_inf_handling() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([np.inf, -np.inf, 0.0])
fnp_result = fnp.clip(a, -1.0, 1.0)
np_result = np.clip(a, -1.0, 1.0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip inf handling should match numpy"
    );
    Ok(())
}

#[test]
fn clip_negative_zero() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([-0.0, 0.0, -1.0, 1.0])
fnp_result = fnp.clip(a, -0.5, 0.5)
np_result = np.clip(a, -0.5, 0.5)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip negative zero should match numpy"
    );
    Ok(())
}

#[test]
fn clip_inf_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.clip(a, -np.inf, np.inf)
np_result = np.clip(a, -np.inf, np.inf)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip inf bounds should match numpy");
    Ok(())
}

#[test]
fn clip_broadcasting() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([[1, 2, 3], [4, 5, 6]])
a_min = np.array([2, 2, 2])
a_max = np.array([5, 5, 5])
fnp_result = fnp.clip(a, a_min, a_max)
np_result = np.clip(a, a_min, a_max)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip broadcasting should match numpy"
    );
    Ok(())
}

#[test]
fn clip_out_parameter() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
out = np.empty_like(a)
fnp_result = fnp.clip(a, 2.0, 4.0, out=out)
np_out = np.empty_like(a)
np_result = np.clip(a, 2.0, 4.0, out=np_out)
print(np.array_equal(out, np_out) and fnp_result is out)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip out parameter should match numpy"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn clip_empty_array() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([], dtype=np.float64)
fnp_result = fnp.clip(a, 2.0, 4.0)
np_result = np.clip(a, 2.0, 4.0)
print(np.array_equal(fnp_result, np_result) and fnp_result.shape == np_result.shape)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(result.trim(), "True", "clip empty array should match numpy");
    Ok(())
}

#[test]
fn clip_single_element() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([5.0])
fnp_result = fnp.clip(a, 2.0, 4.0)
np_result = np.clip(a, 2.0, 4.0)
print(np.array_equal(fnp_result, np_result))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip single element should match numpy"
    );
    Ok(())
}

#[test]
fn clip_equal_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.clip(a, 3.0, 3.0)
np_result = np.clip(a, 3.0, 3.0)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 3.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip equal bounds should match numpy"
    );
    Ok(())
}

#[test]
fn clip_integer_dtypes() -> Result<(), String> {
    let script = fnp_script(
        r#"
a_int32 = np.array([1, 5, 10, 15, 20], dtype=np.int32)
a_int64 = np.array([1, 5, 10, 15, 20], dtype=np.int64)
tests_pass = True
for a in [a_int32, a_int64]:
    fnp_result = fnp.clip(a, 5, 15)
    np_result = np.clip(a, 5, 15)
    tests_pass = tests_pass and np.array_equal(fnp_result, np_result)
print(tests_pass)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip integer dtypes should match numpy"
    );
    Ok(())
}

#[test]
fn clip_all_within_bounds() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
fnp_result = fnp.clip(a, 1.0, 10.0)
np_result = np.clip(a, 1.0, 10.0)
print(np.array_equal(fnp_result, np_result) and np.array_equal(fnp_result, a))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip all within bounds should match numpy"
    );
    Ok(())
}

#[test]
fn clip_all_below_min() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
fnp_result = fnp.clip(a, 10.0, 20.0)
np_result = np.clip(a, 10.0, 20.0)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 10.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip all below min should match numpy"
    );
    Ok(())
}

#[test]
fn clip_all_above_max() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
fnp_result = fnp.clip(a, 1.0, 5.0)
np_result = np.clip(a, 1.0, 5.0)
print(np.array_equal(fnp_result, np_result) and np.all(fnp_result == 5.0))
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "clip all above max should match numpy"
    );
    Ok(())
}

#[test]
fn clip_array_bounds_f64_zerocopy_parallel_bit_exact_matches_numpy() -> Result<(), String> {
    // 3-operand array-bounds clip: verbatim npy_maximum/minimum select rules,
    // byte-exact with no hazard defers (NaN payloads and signed zeros ride the
    // operand-copy semantics). Batteries: random over the 1<<20 gate, specials
    // injected into ALL THREE operands (NaN/inf/-0.0), lo > hi inversion, 2-D,
    // below-gate, scalar-bounds regression, broadcast-shape delegate. Prints a
    // coarse interleaved best-of-7 A/B for the ship record.
    let script = fnp_script(
        r#"
import time
verdicts = []
rng = np.random.default_rng(20260713)
n = 1 << 21
a = rng.standard_normal(n)
lo = rng.standard_normal(n) - 1.0
hi = rng.standard_normal(n) + 1.0
for arr in (a, lo, hi):
    idx = rng.integers(0, n, 6000)
    arr[idx[:2000]] = np.nan
    arr[idx[2000:3000]] = np.inf
    arr[idx[3000:4000]] = -np.inf
    arr[idx[4000:]] = -0.0
r, e = fnp.clip(a, lo, hi), np.clip(a, lo, hi)
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL specials bytes")
inv_lo, inv_hi = hi.copy(), lo.copy()  # lo > hi in ~half the lanes
if fnp.clip(a, inv_lo, inv_hi).tobytes() != np.clip(a, inv_lo, inv_hi).tobytes():
    verdicts.append("FAIL inverted-bounds bytes")
m2, l2, h2 = a[: 1 << 20].reshape(1024, 1024), lo[: 1 << 20].reshape(1024, 1024), hi[: 1 << 20].reshape(1024, 1024)
r, e = fnp.clip(m2, l2, h2), np.clip(m2, l2, h2)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL 2-D bytes")
if fnp.clip(a[:4096], lo[:4096], hi[:4096]).tobytes() != np.clip(a[:4096], lo[:4096], hi[:4096]).tobytes():
    verdicts.append("FAIL below-gate bytes")
if fnp.clip(a, -0.5, 0.5).tobytes() != np.clip(a, -0.5, 0.5).tobytes():
    verdicts.append("FAIL scalar-bounds regression bytes")
if fnp.clip(m2, l2[0], h2).tobytes() != np.clip(m2, l2[0], h2).tobytes():
    verdicts.append("FAIL broadcast-delegate bytes")
def best(fn, reps=7):
    fn(); best_s = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best_s = min(best_s, time.perf_counter() - t0)
    return best_s * 1000
a8 = rng.standard_normal(8_000_000)
l8 = rng.standard_normal(8_000_000) - 1.0
h8 = rng.standard_normal(8_000_000) + 1.0
tn = best(lambda: np.clip(a8, l8, h8))
tf = best(lambda: fnp.clip(a8, l8, h8))
print(f"CLIP_ARRAYS_COARSE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces CLIP_ARRAYS_COARSE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "f64 array-bounds clip must be bit-identical incl NaN-payload/signed-zero rules: {result}"
    );
    Ok(())
}

#[test]
fn clip_f64_broadcast_row_bounds_bit_exact_matches_numpy() -> Result<(), String> {
    // (n,) lo/hi row vectors against (.., n) a: the broadcast sibling of the
    // same-shape array-bounds kernel. Same verbatim maximum/minimum select
    // rules (NaN in a propagates; NaN bounds poison their element; lo > hi
    // resolves min-after-max), bounds indexed by the last axis. Other
    // broadcast forms, small, and non-contiguous inputs keep the delegate.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(73)
verdicts = []
for shape in [(2048, 1024), (999, 1237), (16, 64, 2048)]:
    n = shape[-1]
    a = rng.standard_normal(shape) * 3
    lo = rng.standard_normal(n) - 1
    hi = rng.standard_normal(n) + 1
    r = fnp.clip(a, lo, hi); e = np.clip(a, lo, hi)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {shape}")
# NaN in a / NaN bounds / lo > hi pin the select rules
a = rng.standard_normal((2048, 1024))
a[rng.random((2048, 1024)) < 0.01] = np.nan
lo = rng.standard_normal(1024) - 1
hi = rng.standard_normal(1024) + 1
lo[7] = np.nan; hi[13] = np.nan
lo[100] = 2.0; hi[100] = -2.0  # inverted bounds: min-after-max
if fnp.clip(a, lo, hi).tobytes() != np.clip(a, lo, hi).tobytes():
    verdicts.append("FAIL NaN/inverted rules")
# +-0.0 select identity: STRICT comparisons - ties copy the BOUND operand
z = np.zeros((1024, 1024)); z[::2] = -0.0
zl = np.full(1024, -0.0); zh = np.full(1024, 0.0)
if fnp.clip(z, zl, zh).tobytes() != np.clip(z, zl, zh).tobytes():
    verdicts.append("FAIL signed-zero rules")
zl2 = np.full(1024, 0.0); zh2 = np.full(1024, -0.0)
if fnp.clip(z, zl2, zh2).tobytes() != np.clip(z, zl2, zh2).tobytes():
    verdicts.append("FAIL signed-zero rules swapped")
# the same tie rule through the SAME-SHAPE array-bounds kernel (shared
# contract; its original battery only scattered -0.0 in a, never a tie)
zlf = np.full((1024, 1024), -0.0); zhf = np.full((1024, 1024), 0.0)
if fnp.clip(z, zlf, zhf).tobytes() != np.clip(z, zlf, zhf).tobytes():
    verdicts.append("FAIL signed-zero same-shape kernel")
# (m,1) COLUMN bounds now engage the per-row-scalar kernel (same rules)
col = rng.standard_normal((2048, 1))
if fnp.clip(a, col, col + 2).tobytes() != np.clip(a, col, col + 2).tobytes():
    verdicts.append("FAIL column-bounds")
colz = np.zeros((1024, 1)); colz[7] = np.nan
zc = np.zeros((1024, 1024)); zc[::2] = -0.0
if fnp.clip(zc, np.full((1024, 1), -0.0), colz).tobytes() != np.clip(zc, np.full((1024, 1), -0.0), colz).tobytes():
    verdicts.append("FAIL column signed-zero/NaN rules")
sm = rng.standard_normal((64, 64))
if fnp.clip(sm, lo[:64], hi[:64]).tobytes() != np.clip(sm, lo[:64], hi[:64]).tobytes():
    verdicts.append("FAIL below-gate delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.standard_normal((4096, 4096))
wl = rng.standard_normal(4096) - 1
wh = rng.standard_normal(4096) + 1
tn = best(lambda: np.clip(W, wl, wh)); tf = best(lambda: fnp.clip(W, wl, wh))
print(f"CLIP_F64_BCAST_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces CLIP_F64_BCAST_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "broadcast-row-bounds f64 clip must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn clip_int_array_bounds_bit_exact_matches_numpy() -> Result<(), String> {
    // Integer array bounds (same-shape and (n,) row vectors) across all
    // widths: pure strict-comparison selection, no NaN/signed-zero surface.
    // Inverted bounds pin min-after-max; extremes pin no-wrap; mixed dtypes,
    // column bounds, and below-gate keep the delegate.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(79)
verdicts = []
for dt in [np.int64, np.int32, np.int8, np.uint64, np.uint8]:
    info = np.iinfo(dt)
    A = rng.integers(info.min // 2, info.max // 2, (2048, 1024)).astype(dt)
    Ls = rng.integers(info.min // 2, 0 if info.min < 0 else info.max // 4, (2048, 1024)).astype(dt)
    Hs = rng.integers(0 if info.min < 0 else info.max // 4, info.max // 2, (2048, 1024)).astype(dt)
    r = fnp.clip(A, Ls, Hs); e = np.clip(A, Ls, Hs)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL same-shape {dt.__name__}")
    lo = Ls[0].copy(); hi = Hs[0].copy()
    r = fnp.clip(A, lo, hi); e = np.clip(A, lo, hi)
    if r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL row-bounds {dt.__name__}")
    # inverted bounds: min-after-max order
    if fnp.clip(A, hi, lo).tobytes() != np.clip(A, hi, lo).tobytes():
        verdicts.append(f"FAIL inverted {dt.__name__}")
# dtype extremes as bounds (no arithmetic, pure compares)
A = rng.integers(-2**62, 2**62, (2048, 1024))
lo = np.full(1024, np.iinfo(np.int64).min)
hi = np.full(1024, np.iinfo(np.int64).max)
if fnp.clip(A, lo, hi).tobytes() != np.clip(A, lo, hi).tobytes():
    verdicts.append("FAIL extremes")
# 3-D row bounds
A3 = rng.integers(-1000, 1000, (16, 64, 2048))
l3 = rng.integers(-1200, -800, 2048)
h3 = rng.integers(800, 1200, 2048)
if fnp.clip(A3, l3, h3).tobytes() != np.clip(A3, l3, h3).tobytes():
    verdicts.append("FAIL 3-D row bounds")
# delegates: mixed dtype, column bounds, below gate
if fnp.clip(A3, l3.astype(np.int32), h3).tobytes() != np.clip(A3, l3.astype(np.int32), h3).tobytes():
    verdicts.append("FAIL mixed-dtype delegate")
Ac = rng.integers(-1000, 1000, (2048, 1024))
cl = rng.integers(-1200, -800, (2048, 1))
ch = rng.integers(800, 1200, (2048, 1))
if fnp.clip(Ac, cl, ch).tobytes() != np.clip(Ac, cl, ch).tobytes():
    verdicts.append("FAIL column-bounds")
# 3-D a with (m,1) bounds is a DIFFERENT broadcast (leading axes) - delegate
if fnp.clip(A3, l3[:64].reshape(64, 1), h3[:64].reshape(64, 1)).tobytes() != np.clip(A3, l3[:64].reshape(64, 1), h3[:64].reshape(64, 1)).tobytes():
    verdicts.append("FAIL 3-D col-bounds delegate")
sm = rng.integers(-100, 100, (64, 64))
if fnp.clip(sm, sm - 10, sm + 10).tobytes() != np.clip(sm, sm - 10, sm + 10).tobytes():
    verdicts.append("FAIL below-gate delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.integers(-1000, 1000, (4096, 4096))
Wl = rng.integers(-1200, -800, (4096, 4096))
Wh = rng.integers(800, 1200, (4096, 4096))
tn = best(lambda: np.clip(W, Wl, Wh)); tf = best(lambda: fnp.clip(W, Wl, Wh))
print(f"CLIP_INT64_ARRAYS_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
wl = Wl[0].copy(); wh = Wh[0].copy()
tn = best(lambda: np.clip(W, wl, wh)); tf = best(lambda: fnp.clip(W, wl, wh))
print(f"CLIP_INT64_BCAST_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
wcl = Wl[:, :1].copy(); wch = Wh[:, :1].copy()
tn = best(lambda: np.clip(W, wcl, wch)); tf = best(lambda: fnp.clip(W, wcl, wch))
print(f"CLIP_INT64_COLS_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
Wf = rng.standard_normal((4096, 4096))
fcl = rng.standard_normal((4096, 1)) - 1
fch = rng.standard_normal((4096, 1)) + 1
tn = best(lambda: np.clip(Wf, fcl, fch)); tf = best(lambda: fnp.clip(Wf, fcl, fch))
print(f"CLIP_F64_COLS_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces CLIP_INT64_*_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "int array-bounds clip must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn clip_f32_array_bounds_bit_exact_matches_numpy() -> Result<(), String> {
    // f32 arms of the consolidated float array-bounds kernel: all three
    // broadcast forms with the same strict select rules (ties copy the bound,
    // NaN in a propagates, NaN bounds poison) in f32 precision.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(139)
verdicts = []
A = (rng.standard_normal((2048, 1024)) * 3).astype(np.float32)
A[rng.random((2048, 1024)) < 0.01] = np.nan
Ls = (rng.standard_normal((2048, 1024)) - 1).astype(np.float32)
Hs = (rng.standard_normal((2048, 1024)) + 1).astype(np.float32)
if fnp.clip(A, Ls, Hs).tobytes() != np.clip(A, Ls, Hs).tobytes():
    verdicts.append("FAIL f32 same-shape")
lo = Ls[0].copy(); hi = Hs[0].copy()
lo[7] = np.float32(np.nan); hi[13] = np.float32(np.nan)
lo[100] = np.float32(2.0); hi[100] = np.float32(-2.0)
r = fnp.clip(A, lo, hi); e = np.clip(A, lo, hi)
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL f32 row bounds NaN/inverted")
cl = Ls[:, :1].copy(); ch = Hs[:, :1].copy()
if fnp.clip(A, cl, ch).tobytes() != np.clip(A, cl, ch).tobytes():
    verdicts.append("FAIL f32 col bounds")
# f32 signed-zero ties copy the bound operand
z = np.zeros((1024, 1024), dtype=np.float32); z[::2] = np.float32(-0.0)
zl = np.full(1024, -0.0, dtype=np.float32); zh = np.full(1024, 0.0, dtype=np.float32)
if fnp.clip(z, zl, zh).tobytes() != np.clip(z, zl, zh).tobytes():
    verdicts.append("FAIL f32 signed-zero ties")
# mixed f64/f32 bounds promote - delegate
if fnp.clip(A, lo.astype(np.float64), hi).tobytes() != np.clip(A, lo.astype(np.float64), hi).tobytes():
    verdicts.append("FAIL mixed-precision delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W = rng.standard_normal((4096, 4096)).astype(np.float32)
Wl = (rng.standard_normal((4096, 4096)) - 1).astype(np.float32)
Wh = (rng.standard_normal((4096, 4096)) + 1).astype(np.float32)
# f32 same-shape stays a DELEGATE (numpy's 2x-lane SIMD wins there; measured
# 0.748x loss before the scope cut) - parity row only, no perf claim.
if fnp.clip(W, Wl, Wh).tobytes() != np.clip(W, Wl, Wh).tobytes():
    verdicts.append("FAIL f32 same-shape delegate")
wl = Wl[0].copy(); wh = Wh[0].copy()
tn = best(lambda: np.clip(W, wl, wh)); tf = best(lambda: fnp.clip(W, wl, wh))
print(f"CLIP_F32_BCAST_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
wcl = Wl[:, :1].copy(); wch = Wh[:, :1].copy()
tn = best(lambda: np.clip(W, wcl, wch)); tf = best(lambda: fnp.clip(W, wcl, wch))
print(f"CLIP_F32_COLS_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces CLIP_F32_*_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last,
        "True",
        "f32 array-bounds clip must be bit-identical to numpy: {result}"
    );
    Ok(())
}
