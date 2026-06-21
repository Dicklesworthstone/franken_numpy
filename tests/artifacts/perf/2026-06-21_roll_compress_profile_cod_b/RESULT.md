# RESULT: roll LANDED (win), compress NO-SHIP (already optimized) — cod-b 2026-06-21

Resolution of the two leads from this dir's recipe.md.

## roll — WIN, LANDED (commit 84f52074)
`try_zerocopy_f64_roll` + `try_zerocopy_any_roll`: element-by-element Cell .get/.set
rotate loops -> `from_raw_parts` &[f64]/&[u8] + two `copy_from_slice` (memcpy).
VERIFIED (built fnp-python release, .probe): np.roll(4M) **1.36-1.43x -> 0.96-0.99x
parity**; bit-exact across shapes/shifts (0,1,n-1,n,±,empty) + int64 + complex
any_roll. Byte-identical to prior element-copy -> conformance preserved. (Impl was
applied in the shared working tree and orphaned 2+ turns by agent-mail corruption;
I verified + landed it.)

## compress — NO-SHIP (near-floor; already optimized)
The bool-condition path (the measured 1.36x case) routes
`compress -> try_zerocopy_any_compact -> compact_typed::<u64>` (f64 viewed u64).
That kernel is ALREADY a branchless 8-lane mask compaction:
```rust
while base + 8 <= m {
    let mut mask = 0u8;
    for lane in 0..8 { mask |= ((cond_in[base+lane].get() != 0) as u8) << lane; }
    while mask != 0 { let lane = mask.trailing_zeros(); output[w].set(arr_in[base+lane].get()); w+=1; mask &= mask-1; }
}
```
(explicitly built to avoid branch mispredicts + speculative stores for unkept lanes.)
The only remaining lever is dropping Cell overhead via raw &[T] — but `Cell::<T>::get()`
on a `Copy` T already lowers to a plain load, so that is NEUTRAL. The residual 1.36x
is the INHERENT cost of data-dependent compaction (numpy's compress does the same
count+gather in tighter C). No clean lever. NO-SHIP without a new algorithmic idea.
Did not spend a build on a code-inspected-neutral change (discipline: revert neutral).

## Net
roll: 1 WIN shipped (loss->parity). compress: no-ship (already optimized). The
remaining vs-numpy losses are now only documented no-ships (sqrt forbid-unsafe;
batched inv/solve + spectral kernel walls). fnp dominates the rest.
