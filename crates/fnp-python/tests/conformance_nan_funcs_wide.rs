//! Wide-array companion shard for `conformance_nan_funcs`.
//!
//! Holds the order-statistics parity probe whose array sizes are LOAD-BEARING:
//! it builds two 8M-element arrays, a 2896x2896 matrix and a 64x512x256 tensor
//! specifically to cross fnp's native parallel dispatch gates. Shrinking them
//! would move the test onto the serial path and stop exercising what it exists
//! to exercise, so it cannot be made fast — it is separated instead.
//!
//! RUNTIME: this shard alone takes >110s. Its parent `conformance_nan_funcs`
//! runs its other 41 tests in ~9s. Split under bead `deadlock-audit-syi8e`;
//! coverage is unchanged, every test still runs by default.
//!
//! If you run this under a time cap, CHECK THAT THE BINARY REPORTED. A cap kills
//! a shard mid-execution and the run then prints no `test result:` line for it at
//! all, which reads exactly like a pass if you are only grepping for failures.
//!
//! The `numpy_oracle` / `fnp_script` helpers below are copied from the parent
//! shard, matching how every other `conformance_*` shard carries its own.

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
fn flat_multi_quantile_and_weighted_average_track_numpy() -> Result<(), String> {
    // Convergence-sweep probe (2026-07-12): the last two ambiguous wide-rank rows.
    // quantile(array-q, flat) and percentile(list-q, flat) ride the shipped native
    // order-statistics path - assert byte parity; its coarse A/B is opt-in.
    // average(weights=) rides the extract path whose serial sum is NOT numpy's
    // pairwise order - assert allclose (documented mean-family tolerance) and
    // keep the A/B available for explicit profiling without extending CI.
    let script = fnp_script(
        r#"
import os
import time
import warnings
verdicts = []
rng = np.random.default_rng(20260712)
a = rng.standard_normal(8_000_000)
w = np.abs(rng.standard_normal(8_000_000)) + 0.01
qs = np.linspace(0.1, 0.9, 9)
# Byte-exact since the numpy_quantile_lerp fix (bead deadlock-audit-19jv4): the
# linear method now runs numpy's two-sided _lerp, so multi-q flat is tobytes-equal.
r, e = fnp.quantile(a, qs), np.quantile(a, qs)
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL quantile9 bytes")
r, e = fnp.percentile(a, [25, 50, 75]), np.percentile(a, [25, 50, 75])
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL percentile-trio bytes")
ra, ea = fnp.average(a, weights=w), np.average(a, weights=w)
if not np.allclose(ra, ea, rtol=1e-12):
    verdicts.append("FAIL average allclose")
# multi-q LAST-axis native path (fractions_last_axis): byte parity + q-first layout
m = rng.standard_normal((2896, 2896))
r, e = fnp.percentile(m, [25, 50, 75], axis=1), np.percentile(m, [25, 50, 75], axis=1)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL percentile3-ax1 bytes")
r, e = fnp.quantile(m, qs, axis=1), np.quantile(m, qs, axis=1)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL quantile9-ax1 bytes")
r, e = fnp.quantile(m, [0.5], axis=-1), np.quantile(m, [0.5], axis=-1)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL single-q-list ax-1 bytes")
mn = m.copy(); mn[7, 123] = np.nan
r, e = fnp.percentile(mn, [25, 75], axis=1), np.percentile(mn, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-lane bytes")
r, e = fnp.percentile(m, [25, 75], axis=0), np.percentile(m, [25, 75], axis=0)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL percentile-ax0 bytes")
r, e = fnp.quantile(m, qs, axis=0), np.quantile(m, qs, axis=0)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL quantile9-ax0 bytes")
mc = m.copy(); mc[123, 7] = np.nan
r, e = fnp.percentile(mc, [25, 75], axis=0), np.percentile(mc, [25, 75], axis=0)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-column-ax0 bytes")
# N-D non-last-axis multi-q via the generalized strided-lane kernel
t3 = rng.standard_normal((64, 512, 256))
for tag, ax in (("3d-ax1", 1), ("3d-ax0", 0)):
    r, e = fnp.percentile(t3, [25, 50, 75], axis=ax), np.percentile(t3, [25, 50, 75], axis=ax)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
r, e = fnp.quantile(t3, qs, axis=1, keepdims=True), np.quantile(t3, qs, axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-ax1-keepdims bytes")
t3n = t3.copy(); t3n[3, 100, 7] = np.nan
r, e = fnp.percentile(t3n, [25, 75], axis=1), np.percentile(t3n, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-nan-lane bytes")
# nan N-D non-last-axis multi-q: compaction composed into the strided kernel
t3nn = t3.copy()
t3nn.ravel()[rng.integers(0, t3.size, 20000)] = np.nan
for tag, ax in (("3d-nanpct-ax1", 1), ("3d-nanpct-ax0", 0)):
    r, e = fnp.nanpercentile(t3nn, [25, 50, 75], axis=ax), np.nanpercentile(t3nn, [25, 50, 75], axis=ax)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
r, e = fnp.nanquantile(t3nn, qs, axis=1, keepdims=True), np.nanquantile(t3nn, qs, axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-nan-keepdims bytes")
t3all = t3nn.copy(); t3all[5, :, 9] = np.nan
with warnings.catch_warnings(record=True) as wf:
    warnings.simplefilter("always")
    r = fnp.nanpercentile(t3all, [25, 75], axis=1)
with warnings.catch_warnings(record=True) as wn:
    warnings.simplefilter("always")
    e = np.nanpercentile(t3all, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL 3d-all-nan-lane bytes")
if [str(w.message) for w in wf] != [str(w.message) for w in wn]:
    verdicts.append("FAIL 3d-all-nan-lane warnings")
# nanmedian MEDIAN-vs-lerp(0.5) regression battery (window 9d5d83ac..fix):
# even-count compacted lanes are where mean-of-middles differs bitwise from lerp.
me = rng.standard_normal((4096, 33))
me[:, 7] = np.nan  # 32 valid per lane = EVEN
for tag, arr, kw in (
    ("nanmedian-even-ax1", me, {"axis": 1}),
    ("nanmedian-even-ax0", me.T.copy(), {"axis": 0}),
    ("nanmedian-3d-ax1", t3nn, {"axis": 1}),
    ("nanmedian-odd-ax1", me[:, :32], {"axis": 1}),
):
    r, e = fnp.nanmedian(arr, **kw), np.nanmedian(arr, **kw)
    r, e = np.asarray(r), np.asarray(e)
    if r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
r, e = fnp.quantile(m, qs, axis=1, keepdims=True), np.quantile(m, qs, axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL keepdims-ax1 bytes")
r, e = fnp.percentile(m, [25, 75], axis=0, keepdims=True), np.percentile(m, [25, 75], axis=0, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL keepdims-ax0 bytes")
# nan multi-q native path: per-lane NaN compaction + shared plan/lerp
mn = m.copy()
mn[rng.integers(0, 2896, 20000), rng.integers(0, 2896, 20000)] = np.nan
r, e = fnp.nanpercentile(mn, [25, 50, 75], axis=1), np.nanpercentile(mn, [25, 50, 75], axis=1)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nanpercentile3-ax1 bytes")
r, e = fnp.nanquantile(mn, qs, axis=1), np.nanquantile(mn, qs, axis=1)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nanquantile9-ax1 bytes")
r, e = fnp.nanquantile(mn.ravel(), [0.1, 0.5, 0.9]), np.nanquantile(mn.ravel(), [0.1, 0.5, 0.9])
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-flat multi-q bytes")
allnan = mn.copy(); allnan[5, :] = np.nan
with warnings.catch_warnings(record=True) as wf:
    warnings.simplefilter("always")
    r = fnp.nanpercentile(allnan, [25, 75], axis=1)
with warnings.catch_warnings(record=True) as wn:
    warnings.simplefilter("always")
    e = np.nanpercentile(allnan, [25, 75], axis=1)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL all-nan-lane bytes")
if [str(w.message) for w in wf] != [str(w.message) for w in wn]:
    verdicts.append("FAIL all-nan-lane warnings")
# nan multi-q axis 0: block gather + compaction composition
r, e = fnp.nanpercentile(mn, [25, 50, 75], axis=0), np.nanpercentile(mn, [25, 50, 75], axis=0)
if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nanpercentile3-ax0 bytes")
allnan0 = mn.copy(); allnan0[:, 5] = np.nan
with warnings.catch_warnings(record=True) as wf:
    warnings.simplefilter("always")
    r = fnp.nanquantile(allnan0, [0.25, 0.75], axis=0)
with warnings.catch_warnings(record=True) as wn:
    warnings.simplefilter("always")
    e = np.nanquantile(allnan0, [0.25, 0.75], axis=0)
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL all-nan-column-ax0 bytes")
if [str(w.message) for w in wf] != [str(w.message) for w in wn]:
    verdicts.append("FAIL all-nan-column-ax0 warnings")
r, e = fnp.nanpercentile(mn, [25, 75], axis=1, keepdims=True), np.nanpercentile(mn, [25, 75], axis=1, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-keepdims-ax1 bytes")
r, e = fnp.nanquantile(mn, [0.1, 0.9], keepdims=True), np.nanquantile(mn, [0.1, 0.9], keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL nan-keepdims-flat bytes")
r, e = fnp.quantile(m, qs, keepdims=True), np.quantile(m, qs, keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL plain-keepdims-flat bytes")
r, e = fnp.percentile(a, [25, 50, 75], keepdims=True), np.percentile(a, [25, 50, 75], keepdims=True)
if r.shape != e.shape or r.tobytes() != e.tobytes():
    verdicts.append("FAIL plain-keepdims-flat-1d bytes")
# method='midpoint' native unlock: numpy _lerp(a,b,0.5), not (a+b)/2
for tag, call_f, call_n in (
    ("mid-flat", lambda: fnp.percentile(a, 37.3, method="midpoint"), lambda: np.percentile(a, 37.3, method="midpoint")),
    ("mid-ax1", lambda: fnp.percentile(m, 50, axis=1, method="midpoint"), lambda: np.percentile(m, 50, axis=1, method="midpoint")),
    ("mid-ax0", lambda: fnp.quantile(m, 0.66, axis=0, method="midpoint"), lambda: np.quantile(m, 0.66, axis=0, method="midpoint")),
    ("mid-exact-idx", lambda: fnp.percentile(a[:100001], 50, method="midpoint"), lambda: np.percentile(a[:100001], 50, method="midpoint")),
):
    rf, rn = call_f(), call_n()
    rf, rn = np.asarray(rf), np.asarray(rn)
    if rf.shape != rn.shape or rf.tobytes() != rn.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
# continuous H&F methods: clamped virtual-index plan + two-sided lerp
for meth in ("hazen", "weibull", "median_unbiased", "normal_unbiased"):
    rf = np.asarray(fnp.percentile(a, 37.3, method=meth))
    rn = np.asarray(np.percentile(a, 37.3, method=meth))
    if rf.tobytes() != rn.tobytes():
        verdicts.append(f"FAIL {meth}-flat bytes")
    r, e = fnp.quantile(m, 0.66, axis=1, method=meth), np.quantile(m, 0.66, axis=1, method=meth)
    if r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {meth}-ax1 bytes")
# clamp edges: tiny/huge q on H&F (vi < 0 and vi >= n-1)
for qv in (1e-9, 1.0 - 1e-9, 0.0, 1.0):
    rf = np.asarray(fnp.quantile(a, qv, method="weibull"))
    rn = np.asarray(np.quantile(a, qv, method="weibull"))
    if rf.tobytes() != rn.tobytes():
        verdicts.append(f"FAIL weibull-clamp q={qv} bytes")
# INTEGER array-q unlock (stale-reject reopen): int percentile == exact-f64-widened, byte-for-byte
mi64 = rng.integers(-10**9, 10**9, (1024, 1024))
mi16 = rng.integers(-30000, 30000, (1024, 1024)).astype(np.int16)
for tag, arr, args, kw in (
    ("int64-pct3-ax1", mi64, ([25, 50, 75],), {"axis": 1}),
    ("int64-q9-ax0", mi64, (qs,), {"axis": 0}),
    ("int16-flat", mi16, ([10, 90],), {}),
    ("int64-kd", mi64, (qs,), {"axis": 1, "keepdims": True}),
):
    r, e = fnp.percentile(arr, *args, **kw), np.percentile(arr, *args, **kw)
    if r.dtype != e.dtype or r.shape != e.shape or r.tobytes() != e.tobytes():
        verdicts.append(f"FAIL {tag} bytes")
if fnp.percentile(mi64, 50).tobytes() != np.asarray(np.percentile(mi64, 50)).tobytes():
    verdicts.append("FAIL int-scalar-delegate bytes")
# weights= inverted_cdf: source-exact selection kernel
wq = np.abs(rng.standard_normal(a.size)) + 0.01
wq[rng.integers(0, a.size, 200000)] = 0.0
r = fnp.quantile(a, [0.25, 0.5, 0.75], weights=wq, method="inverted_cdf")
e = np.quantile(a, [0.25, 0.5, 0.75], weights=wq, method="inverted_cdf")
if r.dtype != e.dtype or r.tobytes() != e.tobytes():
    verdicts.append("FAIL weighted-q3 bytes")
at = np.round(a[:300000], 1); at[at == 0] = 0.1
wt = wq[:300000]
r = fnp.quantile(at, [0.0, 0.1, 0.5, 0.9, 1.0], weights=wt, method="inverted_cdf")
e = np.quantile(at, [0.0, 0.1, 0.5, 0.9, 1.0], weights=wt, method="inverted_cdf")
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL weighted-ties-edges bytes")
rs = fnp.quantile(a, 0.5, weights=wq, method="inverted_cdf")
es = np.quantile(a, 0.5, weights=wq, method="inverted_cdf")
if np.asarray(rs).tobytes() != np.asarray(es).tobytes():
    verdicts.append("FAIL weighted-scalar-q bytes")
r = fnp.percentile(a, [25, 75], weights=wq, method="inverted_cdf")
e = np.percentile(a, [25, 75], weights=wq, method="inverted_cdf")
if r.tobytes() != e.tobytes():
    verdicts.append("FAIL weighted-percentile bytes")
wneg = wq.copy(); wneg[123] = -1.0
fe = ne = None
try:
    fnp.quantile(a, [0.5], weights=wneg, method="inverted_cdf")
except Exception as ex:
    fe = type(ex).__name__
try:
    np.quantile(a, [0.5], weights=wneg, method="inverted_cdf")
except Exception as ex:
    ne = type(ex).__name__
if fe != ne:
    verdicts.append(f"FAIL neg-weight error parity fnp={fe} np={ne}")
an = a.copy(); an[77] = np.nan
r = fnp.quantile(an, [0.5], weights=wq, method="inverted_cdf")
e = np.quantile(an, [0.5], weights=wq, method="inverted_cdf")
if np.asarray(r).tobytes() != np.asarray(e).tobytes():
    verdicts.append("FAIL weighted-nan-defer bytes")
if os.environ.get("FNP_SURFACE_PROBE_AB") == "1":
    def best(fn, reps=5):
        fn(); best_s = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); best_s = min(best_s, time.perf_counter() - t0)
        return best_s * 1000
    for name, nf, ff in (
        ("quantile9", lambda: np.quantile(a, qs), lambda: fnp.quantile(a, qs)),
        ("percentile3_ax1", lambda: np.percentile(m, [25, 50, 75], axis=1), lambda: fnp.percentile(m, [25, 50, 75], axis=1)),
        ("quantile9_ax1", lambda: np.quantile(m, qs, axis=1), lambda: fnp.quantile(m, qs, axis=1)),
        ("nanpct3_ax1", lambda: np.nanpercentile(mn, [25, 50, 75], axis=1), lambda: fnp.nanpercentile(mn, [25, 50, 75], axis=1)),
        ("percentile3_ax0", lambda: np.percentile(m, [25, 50, 75], axis=0), lambda: fnp.percentile(m, [25, 50, 75], axis=0)),
        ("nanpct3_ax0", lambda: np.nanpercentile(mn, [25, 50, 75], axis=0), lambda: fnp.nanpercentile(mn, [25, 50, 75], axis=0)),
        ("quantile9_ax1_kd", lambda: np.quantile(m, qs, axis=1, keepdims=True), lambda: fnp.quantile(m, qs, axis=1, keepdims=True)),
        ("quantile9_flat_kd", lambda: np.quantile(m, qs, keepdims=True), lambda: fnp.quantile(m, qs, keepdims=True)),
        ("pct50_ax1_midpoint", lambda: np.percentile(m, 50, axis=1, method="midpoint"), lambda: fnp.percentile(m, 50, axis=1, method="midpoint")),
        ("pct3_3d_ax1", lambda: np.percentile(t3, [25, 50, 75], axis=1), lambda: fnp.percentile(t3, [25, 50, 75], axis=1)),
        ("nanpct3_3d_ax1", lambda: np.nanpercentile(t3nn, [25, 50, 75], axis=1), lambda: fnp.nanpercentile(t3nn, [25, 50, 75], axis=1)),
        ("nanmedian_3d_ax1", lambda: np.nanmedian(t3nn, axis=1), lambda: fnp.nanmedian(t3nn, axis=1)),
        ("hazen_ax1", lambda: np.percentile(m, 37.3, axis=1, method="hazen"), lambda: fnp.percentile(m, 37.3, axis=1, method="hazen")),
        ("int64_pct3_ax1", lambda: np.percentile(mi64, [25, 50, 75], axis=1), lambda: fnp.percentile(mi64, [25, 50, 75], axis=1)),
        ("weighted_q3", lambda: np.quantile(a, [0.25, 0.5, 0.75], weights=wq, method="inverted_cdf"), lambda: fnp.quantile(a, [0.25, 0.5, 0.75], weights=wq, method="inverted_cdf")),
        ("percentile3", lambda: np.percentile(a, [25, 50, 75]), lambda: fnp.percentile(a, [25, 50, 75])),
        ("avg_weights", lambda: np.average(a, weights=w), lambda: fnp.average(a, weights=w)),
    ):
        tn, tf = best(nf), best(ff)
        print(f"SURFACE_PROBE_AB row={name} numpy_ms={tn:.3f} fnp_ms={tf:.3f} ratio={tn / tf:.3f}")
print(verdicts if verdicts else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    println!("{result}"); // surfaces opt-in A/B rows under --nocapture
    let last = result.lines().last().unwrap_or("").trim();
    assert_eq!(
        last, "True",
        "flat multi-quantile/percentile must stay byte-exact and weighted average allclose: {result}"
    );
    Ok(())
}

/// nanmax / nanmin along an axis at and above the streaming floor (bead `deadlock-audit-vc4p4`):
/// lanes, planes and the single-group row fold go parallel from 16 MiB with whole lanes / planes /
/// rows batched to >= 2 MiB per task. Each batched result must land in its own lane's slot, an
/// all-NaN lane must still give NaN with numpy's single warning, and a signed-zero extreme must
/// still defer - on both sides of the floor, float64 and float32.
#[test]
fn nanmax_nanmin_axis_batched_lanes_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
rng = np.random.default_rng(927)
bad = []
def same(label, ours, theirs):
    o, t = np.asarray(ours), np.asarray(theirs)
    if o.dtype != t.dtype or o.shape != t.shape or o.tobytes() != t.tobytes():
        bad.append(label)
for dt in (np.float64, np.float32):
    for rows, lane in ((1023, 1024), (2048, 1024), (16384, 257)):
        a = rng.standard_normal((rows, lane)).astype(dt)
        a[::5, 3] = np.nan
        a[11] = np.nan
        a[12] = 0.0
        a[12, ::2] = -0.0
        for red in ("nanmax", "nanmin"):
            for view, axes in ((a, (1, -1, 0)), (a.reshape(-1, 64), (0,)), (a.reshape(4, -1, 64), (1,))):
                for ax in axes:
                    with warnings.catch_warnings(record=True) as ours_w:
                        warnings.simplefilter("always")
                        ours = getattr(fnp, red)(view, axis=ax)
                    with warnings.catch_warnings(record=True) as np_w:
                        warnings.simplefilter("always")
                        theirs = getattr(np, red)(view, axis=ax)
                    label = f"{red} {np.dtype(dt).name} {view.shape} axis={ax}"
                    same(label, ours, theirs)
                    if [str(w.message) for w in ours_w] != [str(w.message) for w in np_w]:
                        bad.append(label + " warnings")
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "nanmax/nanmin batched lanes must match numpy bytes and warnings: {result}"
    );
    Ok(())
}

/// The per-lane / per-plane reductions that went parallel from 98,304 ELEMENTS (or 2^16 / 2^18 /
/// 2^20) with one rayon item per lane now fan out from 32 MiB with whole lanes, planes and rows
/// batched to >= 2 MiB per task (bead `deadlock-audit-vc4p4`): var / std / nanvar / nanstd /
/// nanmean / nansum / norm / prod / nanprod / int min-max / einsum reductions and unravel_index.
/// Each batched result must land in its own slot on both sides of the floor - a batching or
/// slot-mapping slip shows up as a wrong lane - so every cell is compared byte for byte at 2 MiB
/// (serial) and 32 MiB+ (parallel), float64 and float32, lanes, planes and single-group folds.
#[test]
fn axis_reductions_match_numpy_on_both_sides_of_the_parallel_floors() -> Result<(), String> {
    let script = fnp_script(
        r#"
import warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(1927)
bad = []
def same(label, ours, theirs):
    xs = ours if isinstance(ours, tuple) else (ours,)
    ys = theirs if isinstance(theirs, tuple) else (theirs,)
    for x, y in zip(xs, ys):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            bad.append(label)
            return
for dt in (np.float64, np.float32):
    for rows, lane in ((512, 512), (4096, 2048) if dt == np.float32 else (2048, 2048)):
        a = rng.standard_normal((rows, lane)).astype(dt)
        an = a.copy()
        an[::7, 3] = np.nan
        an[5] = np.nan
        a3 = a.reshape(4, -1, 64)
        an3 = an.reshape(4, -1, 64)
        near1 = (1.0 + a * 1e-4).astype(dt)
        near1n = near1.copy()
        near1n[::7, 3] = np.nan
        near1n[3, ::5] = np.nan
        a0 = a.copy()
        a0[::3] = 0.0
        a0[1, :] = -0.0
        ai = rng.integers(-(1 << 30), 1 << 30, (rows, lane)).astype(np.int64 if dt == np.float64 else np.int32)
        tag = f"{np.dtype(dt).name} {rows}x{lane}"
        for name, fn in [
            ("std ax1", lambda m: m.std(a, axis=1)),
            ("var ax1 ddof1", lambda m: m.var(a, axis=1, ddof=1)),
            ("nanvar ax1", lambda m: m.nanvar(an, axis=1)),
            ("nanmean ax1", lambda m: m.nanmean(an, axis=1)),
            ("nanmean 3d ax1", lambda m: m.nanmean(an3, axis=1)),
            ("nansum 3d ax1", lambda m: m.nansum(an3, axis=1)),
            ("nansum ax1", lambda m: m.nansum(an, axis=1)),
            ("nansum 3d signed zeros", lambda m: m.nansum(np.where(np.isnan(an3), an3, -0.0), axis=1)),
            ("nanprod 3d ax1", lambda m: m.nanprod(near1n.reshape(4, -1, 64), axis=1)),
            ("norm -inf ax0", lambda m: m.linalg.norm(an, -np.inf, axis=0)),
            ("norm 0 3d ax1", lambda m: m.linalg.norm(a0.reshape(4, -1, 64), 0, axis=1)),
            ("norm inf nan 3d ax1", lambda m: m.linalg.norm(an3, np.inf, axis=1)),
            ("var 3d ax1", lambda m: m.var(a3, axis=1)),
            ("nanvar 3d ax1", lambda m: m.nanvar(an3, axis=1)),
            ("norm ax1", lambda m: m.linalg.norm(a, axis=1)),
            ("norm inf ax0", lambda m: m.linalg.norm(a, np.inf, axis=0)),
            ("norm inf 3d ax1", lambda m: m.linalg.norm(a3, np.inf, axis=1)),
            ("norm fro 3d", lambda m: m.linalg.norm(a3.reshape(-1, 64, 64), axis=(1, 2))),
            ("prod ax1", lambda m: m.prod(near1, axis=1)),
            ("nanprod ax1", lambda m: m.nanprod(near1, axis=1)),
            ("int max ax1", lambda m: m.max(ai, axis=1)),
            ("int min flat", lambda m: m.min(ai)),
            ("int max 3d ax1", lambda m: m.max(ai.reshape(4, -1, 64), axis=1)),
            ("int min ax0", lambda m: m.min(ai, axis=0)),
            ("einsum ij->i", lambda m: m.einsum("ij->i", a)),
            ("einsum ij->j", lambda m: m.einsum("ij->j", a)),
            ("einsum ij->", lambda m: m.einsum("ij->", a)),
            ("einsum ijk->ik", lambda m: m.einsum("ijk->ik", a3)),
            ("nan_to_num", lambda m: m.nan_to_num(an)),
            ("unravel 3d", lambda m: m.unravel_index(ai.ravel().astype(np.int64) % (rows * lane), (rows // 2, 2, lane))),
        ]:
            same(f"{name} {tag}", fn(fnp), fn(np))
idx = np.array([0, 1, 2 * 3 * 5 - 1, 17, 29], dtype=np.int64)
for shape in ((30,), (2, 15), (2, 3, 5), (5, 3, 2), (1, 30, 1)):
    ours, theirs = fnp.unravel_index(idx, shape), np.unravel_index(idx, shape)
    same(f"unravel edges {shape}", ours, theirs)
    if [np.asarray(o).strides for o in ours] != [np.asarray(t).strides for t in theirs]:
        bad.append(f"unravel strides {shape}")
print(bad if bad else True)
"#
        .into(),
    );
    let result = numpy_oracle(&script)?;
    assert_eq!(
        result.trim(),
        "True",
        "axis reductions across the parallel floors must match numpy bytes: {result}"
    );
    Ok(())
}
