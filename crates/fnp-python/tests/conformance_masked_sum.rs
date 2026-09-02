//! Conformance tests for the fused masked reduction against the NumPy oracle.
//!
//! `fnp.masked_sum(a, mask)` computes `a[mask].sum()` in one pass, never
//! materialising the compacted array. NumPy has no fused public form, so the
//! oracle is NumPy's own two-step expression. The contract is BYTE equality, not
//! closeness: NumPy sums the COMPACTED sequence with its pairwise tree, and the
//! fused pass reproduces that tree exactly — same <=128 base case, same split at
//! n/2 rounded down to a multiple of 8 — so the accumulation order is identical.
//!
//! The tree shape depends only on the number of SELECTED elements, so the
//! density sweep below is the load-bearing test: it walks sizes across the
//! serial/parallel gate and the <=128 base case, at densities from empty to full.

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

/// THE LOAD-BEARING TEST. The compacted pairwise tree depends only on the
/// selected count, so sizes are walked across the parallel gate (1<<20), across
/// the <=128 base case, and off every power of two, at densities from 0 to 1.
/// Scales are deliberately mixed by 1e6/1e-6 so that ANY reordering of the
/// additions shows up as differing bits rather than being masked by rounding.
#[test]
fn masked_sum_is_byte_identical_across_sizes_and_densities() -> Result<(), String> {
    // On a mismatch, classify it instead of just reporting the pair. `ref_pw`
    // is NumPy's DOCUMENTED pairwise tree (<8 sequential, <=128 eight
    // accumulators, else split at n/2 rounded down to a multiple of 8) written
    // in pure NumPy, so it depends on no fnp code at all. That turns an
    // otherwise-unfalsifiable "the bits differ" into a verdict:
    //   OURS_WRONG   -- the reference agrees with this NumPy build, we do not,
    //                   so the fused kernel genuinely mis-orders its additions.
    //   NUMPY_DIFFERS-- we agree with the documented tree and this NumPy build
    //                   does not, i.e. the installed build reduces in a
    //                   different order and byte-identity is per-build.
    //   BOTH_DIFFER  -- neither matches; treat as OURS_WRONG plus a build note.
    // The reference is only evaluated for failing cases, so the common path
    // costs nothing.
    let script = fnp_script(
        r#"
rng = np.random.default_rng(20260802)

def ref_pw(a, off, n):
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[off + i]
        return r
    if n <= 128:
        body = n - (n % 8)
        r = a[off:off + 8].copy()
        for i in range(8, body, 8):
            r += a[off + i:off + i + 8]
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for i in range(body, n):
            res += a[off + i]
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return ref_pw(a, off, n2) + ref_pw(a, off + n2, n - n2)

def bits(x):
    return np.float64(x).view(np.uint64)

bad = []
for n in [0, 1, 7, 8, 127, 128, 129, 1000, 4096, 100_000, 1_048_575, 1_048_576, 1_500_000]:
    for dens in [0.0, 0.001, 0.25, 0.5, 0.75, 1.0]:
        a = rng.standard_normal(n) * rng.choice([1.0, 1e6, 1e-6], n)
        m = rng.random(n) < dens
        ours = fnp.masked_sum(a, m)
        theirs = a[m].sum()
        if bits(ours) != bits(theirs):
            picked = np.ascontiguousarray(a[m])
            reference = ref_pw(picked, 0, picked.shape[0])
            if bits(reference) == bits(theirs):
                verdict = "OURS_WRONG"
            elif bits(reference) == bits(ours):
                verdict = "NUMPY_DIFFERS"
            else:
                verdict = "BOTH_DIFFER"
            bad.append((n, dens, int(picked.shape[0]), verdict))
if bad:
    print(f"numpy={np.__version__} verdicts={sorted(set(v for *_, v in bad))}")
print("CASES_OK" if not bad else f"MISMATCH {bad}")
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "CASES_OK");
    Ok(())
}

/// Above the parallel gate the banded Rayon path runs and must still agree
/// bit-for-bit; a skewed mask (all selected elements clustered at one end)
/// exercises the split-point locator's uneven-band handling.
#[test]
fn masked_sum_parallel_path_handles_skewed_masks() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(5)
n = 2_000_000
a = rng.standard_normal(n) * 1e5
out = []
for name, m in [
    ("front", np.arange(n) < n // 10),
    ("back", np.arange(n) >= n - n // 10),
    ("middle", (np.arange(n) > n // 3) & (np.arange(n) < n // 2)),
    ("stride7", (np.arange(n) % 7) == 0),
    ("single", np.arange(n) == n - 1),
]:
    ours = fnp.masked_sum(a, m)
    theirs = a[m].sum()
    out.append(np.float64(ours).view(np.uint64) == np.float64(theirs).view(np.uint64))
print(all(out), len(out))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 5");
    Ok(())
}

/// Empty selections, signed zeros, NaN and infinity must match NumPy's bits.
/// `a[mask].sum()` of an empty selection is `+0.0`, and summing only `-0.0`
/// stays `-0.0` — a zero-substitution shortcut would get that wrong.
#[test]
fn masked_sum_edge_values_match_numpy_bitwise() -> Result<(), String> {
    let script = fnp_script(
        r#"
cases = [
    (np.array([1.0, 2.0]), np.array([False, False])),
    (np.array([-0.0, -0.0]), np.array([True, True])),
    (np.array([-0.0, 1.0]), np.array([True, False])),
    (np.array([0.0, -0.0]), np.array([True, True])),
    (np.array([np.nan, 1.0]), np.array([True, True])),
    (np.array([np.nan, 1.0]), np.array([False, True])),
    (np.array([np.inf, -np.inf]), np.array([True, True])),
    (np.array([np.inf, 1.0]), np.array([True, True])),
    (np.array([1e308, 1e308]), np.array([True, True])),
]
ok = []
for a, m in cases:
    ours = np.float64(fnp.masked_sum(a, m))
    theirs = np.float64(a[m].sum())
    ok.append(ours.view(np.uint64) == theirs.view(np.uint64))
print(all(ok), len(ok))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 9");
    Ok(())
}

/// Inputs the native route does not claim must defer and still equal NumPy:
/// non-f64 arrays, non-bool masks, shape mismatch, non-contiguous slices, and
/// multi-dimensional arrays (whose boolean index flattens in C order).
#[test]
fn masked_sum_deferred_inputs_still_match_numpy() -> Result<(), String> {
    let script = fnp_script(
        r#"
rng = np.random.default_rng(13)
ok = []
# float32 array (native route is f64-only)
a32 = rng.standard_normal(5000).astype(np.float32)
m = rng.random(5000) < 0.4
ok.append(np.float32(fnp.masked_sum(a32, m)) == a32[m].sum())
# non-contiguous view
base = rng.standard_normal(10_000)
a_nc, m_nc = base[::2], (rng.random(10_000) < 0.5)[::2]
ok.append(np.float64(fnp.masked_sum(a_nc, m_nc)).view(np.uint64)
          == np.float64(a_nc[m_nc].sum()).view(np.uint64))
# 2-D, C order
a2 = rng.standard_normal((300, 700))
m2 = rng.random((300, 700)) < 0.3
ok.append(np.float64(fnp.masked_sum(a2, m2)).view(np.uint64)
          == np.float64(a2[m2].sum()).view(np.uint64))
# integer array
ai = rng.integers(-50, 50, 4000)
mi = rng.random(4000) < 0.5
ok.append(fnp.masked_sum(ai, mi) == ai[mi].sum())
print(all(ok), len(ok))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 4");
    Ok(())
}

/// A mask that is not bool (int8 of 0/1) is NOT a boolean index in NumPy — it is
/// an integer fancy-index with completely different semantics. The route must
/// defer rather than silently treating nonzero as selected.
#[test]
fn masked_sum_integer_mask_keeps_numpy_fancy_index_semantics() -> Result<(), String> {
    let script = fnp_script(
        r#"
a = np.array([10.0, 20.0, 30.0, 40.0])
idx = np.array([1, 1, 0], dtype=np.int8)
ours = fnp.masked_sum(a, idx)
theirs = a[idx].sum()
print(np.float64(ours).view(np.uint64) == np.float64(theirs).view(np.uint64),
      float(theirs))
"#
        .to_string(),
    );
    assert_eq!(numpy_oracle(&script)?, "True 50.0");
    Ok(())
}
