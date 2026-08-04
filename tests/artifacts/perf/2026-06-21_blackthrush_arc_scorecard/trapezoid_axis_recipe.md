# Ready-to-apply: native trapezoid along an axis for N-D (BlackThrush)

Status: QUEUED — `crates/fnp-python/src/lib.rs` is actively reserved by YellowElk
(fresh exclusive lock, not stale). Land when free. Two parts:

## Part A — last axis (axis=-1/last), the common integration axis  [enhance 0.60x -> ~0.1x]

Measured: `np.trapezoid(M, axis=1)` 2-D 2000x2000 = 0.60x (still on the slow extract
kernel). The 1-D zero-copy path (`try_zerocopy_f64_trapezoid_flat`, f091be6b) handles
only 1-D. Generalize to the last contiguous axis exactly like gradient N-D (87bd6403):
each contiguous row of length L → `dx*(rowsum - (r[0]+r[-1])/2)`, parallel over rows.

Formula VALIDATED bit-allclose vs numpy (pure-Python, no build): shapes (2000,2000)/
(50,200,30)/(4,1e6) × dx∈{1.0,2.5} → allclose True, maxreldiff ≤4.2e-16.

Sketch (generalize the flat fn; result has shape[:-1]):
```rust
let l = shape[ndim-1]; let outer = total / l;   // 1-D => outer==1
// out = numpy.empty(outer); o: &mut [f64]
let rowtrap = |r: &[f64]| -> f64 {
    let s: f64 = r.iter().sum();           // serial per-row sum (allclose, like 1-D path)
    dx * (s - (r[0] + r[r.len()-1]) / 2.0)
};
if total >= TRAPEZOID_PARALLEL_MIN && threads >= 2 {
    o.par_iter_mut().zip(data.par_chunks(l)).for_each(|(s, r)| *s = rowtrap(r));
} else {
    for (s, r) in o.iter_mut().zip(data.chunks(l)) { *s = rowtrap(r); }
}
// reshape o (len outer) to shape[:-1]; if ndim==1 return scalar (existing 1-D path)
```
Dispatch: native when `x.is_none()` AND target axis == last (axis=None for 1-D, or
axis normalizes to ndim-1). Non-last axis, x-coord, non-f64, non-contig defer.

## Part B — axis=0 / first axis  [FIX loss 1.23x]

Measured: `np.trapezoid(M, axis=0)` 2-D = 1.23x LOSS. = `dx*(colsum - (row0+rowlast)/2)`
per column, where colsum = sum over the first axis (stride = inner = product(shape[1:])).
numpy single-threaded. Privatized parallel column sum (bincount pattern): each thread
sums its row-chunk into an inner-length accumulator, combine; then endpoint-correct.
Gate inner <= 1<<16 (privatized memory) AND total >= ~1<<19. allclose (trapezoid is not
bit-exact — naive vs pairwise ~1e-13). result shape = shape[1:].

```rust
let r = shape[0]; let inner = total / r;
// privatized: data.par_chunks(chunk*inner).fold(|| vec![0.0;inner], |mut acc, block| {
//   for row in block.chunks(inner) { for k in 0..inner { acc[k]+=row[k]; } } acc
// }).reduce(|| vec![0.0;inner], |mut a,b| { for k {a[k]+=b[k]} a });
// result[k] = dx*(acc[k] - (data[k] + data[(r-1)*inner+k])/2.0)
```

## Verify before commit (both parts)
- conformance_interp_trapz (currently 16/16).
- allclose vs numpy: 2-D axis=0/1, 3-D axis=-1/0, dx=1/2.5, edge L=2.
- timing: trapezoid(2000x2000, axis=1) and (axis=0).
- GOTCHA: insert helper fns ABOVE the `#[pyfunction]`/`#[pyo3]` attrs (else E0433).
- If lib.rs carries a peer's STALE uncommitted WIP: stash-PUSH/POP (never DROP),
  verify peer diff byte-identical after pop (see 87bd6403 ledger entry). If the peer
  holds a FRESH/active reservation, do NOT touch — wait.


## STATUS 2026-06-21: DONE/LANDED — trapezoid axis now 0.02-0.43x WIN (re-measured). No longer a loss.
