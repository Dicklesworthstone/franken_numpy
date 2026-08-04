# roll / compress 1.36x profiling + exact fix recipe (cod-b / BlackThrush, 2026-06-21)

Read-only profiling of the two real vs-NumPy losses found this session (see
NEGATIVE_EVIDENCE "LEADS: roll + compress 1.36x"). Could not apply the fix this turn:
`crates/fnp-python/src/lib.rs` is peer-dirty (YellowElk's uncommitted work) and
agent-mail is corrupt (no reservation/messaging). This is the ready-to-apply recipe.

## Measurement (SERIAL RAYON=1, stable on a loaded box; parallel is noise at load~27)
- np.roll(4M, 1000): fnp/numpy = **1.36-1.43x** (LOSS)
- np.compress(mask, 4M): fnp/numpy = **1.36x** (LOSS)
Both already have zero-copy fast paths, so the loss is the fast path itself.

## roll — ROOT CAUSE (clear, high-confidence fix)
`try_zerocopy_f64_roll` rotates with TWO ELEMENT-BY-ELEMENT Cell loops:
```rust
let split = n - s;
for i in 0..s     { output[i].set(input[split + i].get()); }   // tail -> front
for i in 0..split { output[s + i].set(input[i].get()); }       // head -> back
```
`output: &[Cell<f64>]`, `input: &[ReadOnlyCell<f64>]` — per-element `.get()/.set()`
does NOT lower to `memcpy`; numpy's roll is two bulk memmoves (concatenate of two
contiguous slices). That gap == the 1.36x.

FIX (fnp-python permits unsafe — convolve/cov already do this): read both as plain
slices via `from_raw_parts` and use two `copy_from_slice` (each is a memcpy):
```rust
// input_f64: &[f64] from in_buffer (from_raw_parts on the ReadOnlyCell ptr),
// output_f64: &mut [f64] from the numpy.empty buffer (from_raw_parts_mut).
output_f64[..s].copy_from_slice(&input_f64[split..]);   // tail -> front
output_f64[s..].copy_from_slice(&input_f64[..split]);   // head -> back
```
Expect parity-or-win (2 memcpy == numpy). Verify: bit-exact vs np.roll across
s∈{0,1,n-1,n,n+1, ±}, n not a multiple of anything, empty, 1-elem; conformance green.
Also worth doing for `try_zerocopy_any_roll` (same pattern, byte copy via u8 view).

## compress — ROOT CAUSE (marginal, lower priority)
`compress` fast path is a two-pass branchy gather:
```rust
let count = cond_in.iter().filter(|c| c.get() != 0).count();   // pass 1
for (c, a) in cond_in.iter().zip(arr_in.iter()) {              // pass 2
    if c.get() != 0 { output[w].set(a.get()); w += 1; }
}
```
The select is inherently data-dependent (numpy's is too), so the ceiling is lower.
The only safe lever is removing Cell overhead: read cond/arr as `&[u8]`/`&[f64]`
(from_raw_parts) so the count + gather are over plain slices (better codegen,
no interior-mutability load/store). Likely shaves part of the 1.36x but not all;
revert if neutral. Lower priority than roll.

## Apply when unblocked
When `crates/fnp-python/src/lib.rs` is free (no peer-dirty) and agent-mail is back:
reserve it, apply the roll memcpy fix, build (rch or local), python head-to-head +
conformance, record win/loss in NEGATIVE_EVIDENCE, commit only your file.
