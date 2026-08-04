# Ready-to-apply: native gradient along the LAST axis for N-D (BlackThrush)

Status: QUEUED — blocked on `crates/fnp-python/src/lib.rs` (YellowElk uncommitted
compress WIP in the shared tree; messaged 2026-06-21). Land when the tree is clean.

Formula VALIDATED bit-exact vs numpy (pure-Python prototype, no build):
shapes (1000,1000)/(50,200,30)/(4,1_000_000) × dx∈{1.0,2.5} → array_equal True, maxdiff 0.0.
`np.gradient(M, axis=-1)`: per contiguous row `r` of length L,
`out[0]=(r[1]-r[0])/dx`, `out[L-1]=(r[L-1]-r[L-2])/dx`, `out[1:L-1]=(r[2:]-r[:-2])/(2*dx)`.

## Plan

Generalize the existing 1-D path (`try_zerocopy_f64_gradient_1d`, ~line 20585) to the
last (contiguous) axis. 1-D is the `outer==1` case. numpy.gradient(M, axis=-1) is
single-threaded; parallelize over the `outer = total/L` rows (par_chunks(L)). Keep the
1-D interior-parallel path (par_chunks(L) gives only 1 chunk when outer==1).

Rename/extend the fn to `try_zerocopy_f64_gradient_lastaxis(py, numpy, f, dx, edge_order)`:
- after the f64 + c_contiguous checks, use `let shape = buffer.shape().to_vec(); let ndim = shape.len();`
  drop the `shape.len()!=1` gate.
- `let l = shape[ndim-1]; if l < 2 { return Ok(None); } let total = cells.len(); let outer = total / l;`
- allocate `numpy.empty(total)`, get `o: &mut [f64]`.
- compute:
```rust
use rayon::prelude::*;
const GRADIENT_PARALLEL_MIN: usize = 1 << 18;
let stencil = |orow: &mut [f64], r: &[f64]| {
    let l = r.len();
    orow[0] = (r[1] - r[0]) / dx;
    orow[l - 1] = (r[l - 1] - r[l - 2]) / dx;
    for j in 1..l - 1 { orow[j] = (r[j + 1] - r[j - 1]) / (2.0 * dx); }
};
if total >= GRADIENT_PARALLEL_MIN && rayon::current_num_threads() >= 2 {
    if outer == 1 {
        // 1-D: parallel over interior (rows give no parallelism)
        o[0] = (data[1] - data[0]) / dx;
        o[l - 1] = (data[l - 1] - data[l - 2]) / dx;
        o[1..l - 1].par_iter_mut().enumerate().for_each(|(j, s)| {
            let i = j + 1; *s = (data[i + 1] - data[i - 1]) / (2.0 * dx);
        });
    } else {
        o.par_chunks_mut(l).zip(data.par_chunks(l)).for_each(|(orow, r)| stencil(orow, r));
    }
} else {
    for (orow, r) in o.chunks_mut(l).zip(data.chunks(l)) { stencil(orow, r); }
}
```
- reshape: `if shape.len()==1 { out.unbind() } else { out.reshape(shape) }`.

## Dispatch (in `fn gradient`)

Determine the target axis and only take the native path when it is the LAST axis
(contiguous) — otherwise defer (strided columns / list-return). `axis=None` is native
ONLY for ndim==1 (numpy returns a list for N-D no-axis). For `axis=Some(a)`: normalize
`a` (handle negative) and require `a == ndim-1`.

```rust
let uniform_dx: Option<f64> = match varargs.len() {
    0 => Some(1.0),
    1 => varargs.get_item(0)?.extract::<f64>().ok(),
    _ => None,
};
// ndim from f.getattr("ndim")? or shape len; axis_is_last = axis None&&ndim==1, or normalized(axis)==ndim-1
if let Some(dx) = uniform_dx
    && axis_is_last_contiguous
    && let Some(out) = try_zerocopy_f64_gradient_lastaxis(py, &numpy, f.bind(py), dx, edge_order)?
{ return Ok(out); }
```

## Expected
gradient(M, axis=-1) 2-D 1.08x parity → ~0.3-0.5x (parallel per-row stencil, numpy
single-threaded). 1-D unchanged (already 3-20x). Defers: axis=0/non-last, coord-array
spacing, edge_order=2, non-f64, non-contiguous, N-D no-axis.

## Verify before commit
- conformance_diff_gradient (currently 23/23).
- array_equal vs numpy for 2-D axis=-1/1, 3-D axis=-1, dx=1/2.5, edge cases (L=2).
- timing: gradient(2000x2000, axis=1) and (4, 1e6, axis=-1).
- GOTCHA: insert helper fn ABOVE the `#[pyfunction]`/`#[pyo3]` attrs, never between
  them and `fn gradient` (steals the attribute → E0433 wrap_pyfunction).
