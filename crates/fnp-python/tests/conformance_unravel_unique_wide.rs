//! Wide-array companion shard for `conformance_unravel_unique`.
//!
//! Holds the probes whose array sizes are LOAD-BEARING: they exist to cross fnp's native
//! parallel dispatch gates, so shrinking them would move the probe onto the serial path and
//! stop it testing what it exists to test. They are separated rather than shrunk or
//! ignored — coverage is unchanged and every test still runs by default. Split under bead
//! `deadlock-audit-syi8e`.
//!
//!   unique_nd_flat_view_matches_numpy
//!   setops_nd_flat_view_matches_numpy
//!   int_sort_class_stale_basis_probe_and_parity
//!   add_at_i64_large_target_parallel_matches_numpy
//!   wide_int_setops_bit_exact_match_numpy
//!
//! RUNTIME IS HOST-SCOPED — do not quote a number without naming the host that observed it.
//! The first four were each measured at over a minute on CreamGlen's box on 2026-08-09.
//! `wide_int_setops_bit_exact_match_numpy` was on nobody's list and turned out to be the
//! largest single cost in the family — 115.372s of a 146.26s parent wall on rch worker hz2,
//! 2026-08-15 — which is why moving only the named four left the parent over 120s. A
//! sibling shard measured 9.03s, 183.46s and 617.62s for identical tests on three different
//! hosts, so absolute budgets and per-test rankings do not transfer between workers. Each
//! test shells out to `python3`, so a contended worker inflates every one of them.
//!
//! When running under a time cap, CHECK THAT THE BINARY REPORTED — a cap kills a shard
//! mid-execution and prints no `test result:` line for it, which reads exactly like a pass
//! if you are only grepping for failures.
//!
//! The `numpy_oracle` / `fnp_script` helpers below are copied from the parent shard,
//! matching how every other `conformance_*` shard carries its own.

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
fn unique_nd_flat_view_matches_numpy() -> Result<(), String> {
    // np.unique's default axis=None operates on the FLATTENED array; the
    // dispatch now normalizes C-contiguous N-D input to a zero-copy
    // reshape(-1) view so the ndim==1-gated flat kernels (string, c128,
    // c64, datetime, struct) serve N-D input that previously delegated
    // wholesale. int/f64 kernels read the flat buffer either way
    // (regression rows). Byte-exact: np.unique(a) IS np.unique(a.ravel());
    // F-contig / defer cases keep delegate parity.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(313)
verdicts = []
def ab(name, a):
    ours = fnp.unique(a)
    theirs = np.unique(a)
    if ours.tobytes() != theirs.tobytes() or str(ours.dtype) != str(theirs.dtype):
        verdicts.append(f"FAIL {name}")
# unlocked arms: 2-D datetime64, 2-D complex128, 2-D fixed-width unicode
D = rng.integers(0, 5_000_000, (1414, 1414)).astype("datetime64[s]")
ab("2-D datetime64", D)
C = (rng.standard_normal((1414, 1414)) + 1j * rng.standard_normal((1414, 1414))).astype(np.complex128)
ab("2-D complex128", C)
S = np.array([f"k{v:07d}" for v in rng.integers(0, 400_000, 300_000)], dtype="U8").reshape(600, 500)
ab("2-D unicode", S)
# already-covered arms keep working through the view (regression)
ab("3-D small-range int", rng.integers(0, 300, (128, 128, 128)))
ab("2-D f64", np.round(rng.standard_normal((1500, 1400)), 3))
# mixed-sign-zero f64 defers (signed-zero-tie parity fix: which zero survives
# dedup is the sort's algorithm-specific tie choice) - 1-D row pins the fix
# for the pre-existing flat path, 2-D covered by the rounded row above
z1 = rng.standard_normal(2_000_000)
z1[::3] = 0.0
z1[1::3] = -0.0
ab("1-D mixed-zero f64", z1)
# defer/delegate parity: F-contig, NaN complex (kernel defers), small, 1-D
ab("F-contig datetime", np.asfortranarray(D[:500, :500]))
Cn = C[:800, :800].copy(); Cn[3, 5] = complex(np.nan, 1.0)
ab("2-D c128 nan parity", Cn)
ab("small 2-D", rng.integers(0, 10, (8, 9)))
ab("1-D unchanged", rng.integers(0, 5_000_000, 2_000_000).astype("datetime64[s]"))

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.unique(C)); tf = best(lambda: fnp.unique(C))
print(f"UNIQUE_ND_C128_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tnd = best(lambda: np.unique(D)); tfd = best(lambda: fnp.unique(D))
print(f"UNIQUE_ND_DT64_AB numpy_ms={tnd:.3f} fnp_ms={tfd:.3f} ratio={tnd / tfd:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces UNIQUE_ND_C128_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "N-D flat-view unique must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn setops_nd_flat_view_matches_numpy() -> Result<(), String> {
    // intersect1d/union1d/setdiff1d/setxor1d FLATTEN their inputs by
    // contract; the entry points now normalize C-contiguous N-D input to
    // zero-copy reshape(-1) views so the whole 1-D-gated arm chain
    // (narrow-int, wide-int, string, c128, datetime, struct) serves N-D
    // forms that previously delegated wholesale. Defers keep delegate
    // parity with ORIGINAL args.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(347)
verdicts = []
def ab(op, name, a, b, **kw):
    ours = getattr(fnp, op)(a, b, **kw)
    theirs = getattr(np, op)(a, b, **kw)
    if ours.tobytes() != theirs.tobytes() or str(ours.dtype) != str(theirs.dtype):
        verdicts.append(f"FAIL {op} {name}")
def strs(n, hi):
    return np.array([f"id{v:06d}" for v in rng.integers(0, hi, n)], dtype="U8")
S1 = strs(1_200_000, 400_000).reshape(1200, 1000)
S2 = strs(1_200_000, 400_000).reshape(1000, 1200)
ab("intersect1d", "2-D string", S1, S2)
ab("union1d", "2-D string", S1, S2)
ab("setdiff1d", "2-D string", S1, S2)
ab("setxor1d", "2-D string", S1[:600, :500], S2[:500, :600])
W1 = rng.integers(-2**62, 2**62, (1500, 1400))
W2 = rng.integers(-2**62, 2**62, (1400, 1500))
ab("intersect1d", "2-D wide int", W1, W2)
ab("setdiff1d", "2-D wide int", W1, W2)
ab("union1d", "2-D narrow int", rng.integers(0, 200, (1200, 1100)).astype(np.int16), rng.integers(0, 200, (1100, 1200)).astype(np.int16))
D1 = rng.integers(0, 3_000_000, (1200, 1100)).astype("datetime64[s]")
D2 = rng.integers(0, 3_000_000, (1100, 1200)).astype("datetime64[s]")
ab("setdiff1d", "2-D datetime", D1, D2)
# mixed 2-D vs 1-D, delegate-parity forms
ab("intersect1d", "2-D vs 1-D string", S1, S2.ravel()[:300_000])
ab("intersect1d", "F-contig parity", np.asfortranarray(W1[:500, :500]), W2[:400, :400])
ab("setdiff1d", "assume_unique parity", np.unique(W1)[:200_000].reshape(400, 500), np.unique(W2)[:200_000], assume_unique=True)
ab("union1d", "small 2-D", rng.integers(0, 50, (8, 9)), rng.integers(0, 50, (7, 6)))
ab("intersect1d", "1-D unchanged", W1.ravel()[:2_000_000], W2.ravel()[:2_000_000])

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.union1d(S1, S2)); tf = best(lambda: fnp.union1d(S1, S2))
print(f"UNION1D_ND_STRING_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tni = best(lambda: np.intersect1d(W1, W2)); tfi = best(lambda: fnp.intersect1d(W1, W2))
print(f"INTERSECT1D_ND_WIDEINT_AB numpy_ms={tni:.3f} fnp_ms={tfi:.3f} ratio={tni / tfi:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces UNION1D_ND_STRING_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "N-D flat-view setops must be bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn int_sort_class_stale_basis_probe_and_parity() -> Result<(), String> {
    // Stale-basis follow-through (the flat-f64-sort regate rule: numpy 2.x
    // x86-simd-sort ships int qsort/argsort on avx2+ hosts too — re-run
    // sort-class ABs after any numpy upgrade). Parity rows pin byte-exactness
    // of whatever routing is live; the AB rows measure the int flat sort /
    // default argsort arms against the current worker's numpy for the regate
    // decision.
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(353)
verdicts = []
def ab(fn, name, a, **kw):
    if getattr(fnp, fn)(a, **kw).tobytes() != getattr(np, fn)(a, **kw).tobytes():
        verdicts.append(f"FAIL {fn} {name}")
W = rng.integers(-2**62, 2**62, 8_000_000)
I32 = rng.integers(-2**31, 2**31 - 1, 8_000_000).astype(np.int32)
D = rng.integers(0, 5_000_000_000, 8_000_000).astype("datetime64[ns]")
ab("sort", "i64 flat", W)
ab("sort", "i32 flat", I32)
ab("sort", "dt64 flat", D)
ab("argsort", "i64 default distinct", np.random.default_rng(354).permutation(8_000_000).astype(np.int64) * 1099511627776 + np.arange(8_000_000))
ab("sort", "i64 small", W[:1000])

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

tn = best(lambda: np.sort(W)); tf = best(lambda: fnp.sort(W))
print(f"SORT_I64_FLAT_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn2 = best(lambda: np.sort(I32)); tf2 = best(lambda: fnp.sort(I32))
print(f"SORT_I32_FLAT_AB numpy_ms={tn2:.3f} fnp_ms={tf2:.3f} ratio={tn2 / tf2:.3f}")
tn3 = best(lambda: np.sort(D)); tf3 = best(lambda: fnp.sort(D))
print(f"SORT_DT64_FLAT_AB numpy_ms={tn3:.3f} fnp_ms={tf3:.3f} ratio={tn3 / tf3:.3f}")
AD = np.random.default_rng(355).permutation(8_000_000).astype(np.int64) * 1099511627776 + np.arange(8_000_000)
tn4 = best(lambda: np.argsort(AD)); tf4 = best(lambda: fnp.argsort(AD))
print(f"ARGSORT_I64_DEFAULT_AB numpy_ms={tn4:.3f} fnp_ms={tf4:.3f} ratio={tn4 / tf4:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces SORT_I64_FLAT_AB etc. under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "int sort-class arms must stay bit-identical to numpy: {result}"
    );
    Ok(())
}

#[test]
fn add_at_i64_large_target_parallel_matches_numpy() -> Result<(), String> {
    // np.add.at(i64, idx, vals) large-target regime: numpy's ufunc.at is a
    // DRAM-latency-bound serial scatter there (probe: 136ms for 8M into 8M);
    // the parallel atomic fetch_add arm is byte-exact because wrapping i64
    // addition commutes (duplicate-index order unobservable) and fetch_add
    // wraps exactly like numpy's i64 overflow. Histogram-style small targets
    // (numpy fast path, 7.5ms), floats, scalar vals, other dtypes, 2-D, and
    // OOB indices keep the delegate (parity / numpy's exact errors).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(367)
verdicts = []
def ab(name, n, idx, vals, dtype=np.int64):
    a1 = np.zeros(n, dtype=dtype); a2 = np.zeros(n, dtype=dtype)
    fnp.add.at(a1, idx, vals)
    np.add.at(a2, idx, vals)
    if a1.tobytes() != a2.tobytes():
        verdicts.append(f"FAIL {name}")
n = 4_000_000
idx = rng.integers(0, n, 4_000_000)
vals = rng.integers(-10**9, 10**9, 4_000_000)
ab("large target dup-heavy", n, idx, vals)
# negative indices wrap once
idxn = idx.copy(); idxn[::3] -= n
ab("negative indices", n, idxn, vals)
# wrapping overflow parity
big = rng.integers(2**62, 2**63 - 1, 4_000_000)
ab("i64 wrap overflow", n, idx, big)
# delegate-parity forms: histogram-regime small target, f64, scalar vals, i32, 2-D target
ab("small target delegate", 1024, rng.integers(0, 1024, 4_000_000), vals)
af1 = np.zeros(n); af2 = np.zeros(n)
fv = rng.standard_normal(4_000_000)
fnp.add.at(af1, idx, fv); np.add.at(af2, idx, fv)
if af1.tobytes() != af2.tobytes():
    verdicts.append("FAIL f64 delegate")
s1 = np.zeros(n, dtype=np.int64); s2 = np.zeros(n, dtype=np.int64)
fnp.add.at(s1, idx, 7); np.add.at(s2, idx, 7)
if s1.tobytes() != s2.tobytes():
    verdicts.append("FAIL scalar delegate")
ab("i32 delegate", n, idx.astype(np.int32), vals.astype(np.int32), dtype=np.int32)
t1 = np.zeros((2000, 2000), dtype=np.int64); t2 = np.zeros((2000, 2000), dtype=np.int64)
r = rng.integers(0, 2000, 3_000_000)
fnp.add.at(t1, r, 1); np.add.at(t2, r, 1)
if t1.tobytes() != t2.tobytes():
    verdicts.append("FAIL 2-D delegate")
# OOB raises identically through the delegate
try:
    fnp.add.at(np.zeros(n, dtype=np.int64), np.array([0, n], dtype=np.int64), np.array([1, 1], dtype=np.int64))
    verdicts.append("FAIL oob no-raise")
except IndexError:
    pass

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

A = np.zeros(8_000_000, dtype=np.int64)
IDX = rng.integers(0, 8_000_000, 8_000_000)
V = rng.integers(-1000, 1000, 8_000_000)
tn = best(lambda: np.add.at(A, IDX, V)); tf = best(lambda: fnp.add.at(A, IDX, V))
print(f"ADD_AT_I64_LARGE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces ADD_AT_I64_LARGE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "parallel i64 add.at must be bit-identical to numpy: {result}"
    );
    Ok(())
}
#[test]
fn wide_int_setops_bit_exact_match_numpy() -> Result<(), String> {
    // Wide (4/8-byte) int set ops via par_sort+dedup+merge: sorted unique
    // integer outputs are value-deterministic -> byte-exact across the FULL
    // dtype range (incl. > 2^53, where the old extract-precise route raised
    // and fell back to numpy's multi-second sort path).
    let script = fnp_script(
        r#"
import time
rng = np.random.default_rng(257)
verdicts = []
for dt in [np.int64, np.uint64, np.int32, np.uint32]:
    info = np.iinfo(dt)
    a = rng.integers(info.min, info.max, 400_000, dtype=dt, endpoint=True)
    b = rng.integers(info.min, info.max, 400_000, dtype=dt, endpoint=True)
    b[:5000] = a[:5000]  # guaranteed overlap
    for fname in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
        ff = getattr(fnp, fname); nf = getattr(np, fname)
        r = ff(a, b); e = nf(a, b)
        if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
            verdicts.append(f"FAIL {fname} {dt.__name__}")
# duplicates-heavy, disjoint, empty, N-D ravel
d1 = rng.integers(0, 100, 300_000)
d2 = rng.integers(50, 150, 300_000)
for fname in ("intersect1d", "union1d", "setdiff1d", "setxor1d"):
    ff = getattr(fnp, fname); nf = getattr(np, fname)
    if ff(d1, d2).tobytes() != nf(d1, d2).tobytes():
        verdicts.append(f"FAIL dup-heavy {fname}")
lo = rng.integers(-2**62, -2**61, 200_000)
hi = rng.integers(2**61, 2**62, 200_000)
if fnp.intersect1d(lo, hi).tobytes() != np.intersect1d(lo, hi).tobytes():
    verdicts.append("FAIL disjoint")
e_ = np.array([], dtype=np.int64)
big = rng.integers(-2**62, 2**62, 200_000)
if fnp.union1d(e_, big).tobytes() != np.union1d(e_, big).tobytes():
    verdicts.append("FAIL empty operand")
M2 = rng.integers(-2**62, 2**62, (600, 500))
if fnp.intersect1d(M2, big).tobytes() != np.intersect1d(M2, big).tobytes():
    verdicts.append("FAIL N-D ravel")
# assume_unique keeps the delegate
au1 = np.unique(rng.integers(-2**62, 2**62, 200_000))
au2 = np.unique(rng.integers(-2**62, 2**62, 200_000))
if fnp.intersect1d(au1, au2, assume_unique=True).tobytes() != np.intersect1d(au1, au2, assume_unique=True).tobytes():
    verdicts.append("FAIL assume_unique delegate")

def best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)

W1 = rng.integers(-2**62, 2**62, 8_000_000)
W2 = rng.integers(-2**62, 2**62, 8_000_000)
W2[:100_000] = W1[:100_000]
tn = best(lambda: np.intersect1d(W1, W2)); tf = best(lambda: fnp.intersect1d(W1, W2))
print(f"INTERSECT_INT64_WIDE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
tn = best(lambda: np.setdiff1d(W1, W2)); tf = best(lambda: fnp.setdiff1d(W1, W2))
print(f"SETDIFF_INT64_WIDE_AB numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces INTERSECT/UNION_INT64_WIDE_AB under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "wide int setops must be bit-identical to numpy: {result}"
    );
    Ok(())
}
