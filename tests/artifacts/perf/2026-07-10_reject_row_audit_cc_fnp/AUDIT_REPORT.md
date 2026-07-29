# Ledger-integrity audit: the three "numpy SIMD wall" reject rows (cc_fnp, 2026-07-10, no-commit window)

Ordered check: profile-verify each row's bench executed the rejected code with non-zero
self-time. Constraint reality: all three candidates were REVERTED (none is in HEAD), so
profiling them requires rebuilding candidate binaries; local builds are forbidden (disk),
and `perf` cannot be routed through rch (rch classifies it as a non-compilation command —
prior artifact evidence). Full profile-verification is therefore BLOCKED; this audit uses
execution FINGERPRINTS from the recorded measurements plus in-tree code analysis. Where a
fingerprint is decisive, the verdict is final; the blocked profile pass is noted per row.

## Row 1 — 2026-07-07 "FLOAT np.median via RADIX-SELECT — 0.68x LOSS" : MEASUREMENT VALID, GENERALIZATION INVALID (amendment required)

Execution fingerprint (decisive): candidate arm read 233.8 ms where pure delegation would
read ≈ numpy's 160 ms — a dead-code arm cannot exceed the delegation bound by 46%. The
spread-vs-clustered contrast in one binary (0.68x standard_normal vs 1.22x round2) is the
radix-narrowing mechanism's signature (slow multi-byte narrow near 0 for spread data).
The rejected code ran. REJECT of that candidate stands.

HOWEVER the row's ceiling claim — "Do not re-attempt radix-select for float order
statistics… only a SINGLE-pass in-place parallel partition (parallel introselect /
Floyd-Rivest) could beat it" — is FALSIFIED BY IN-TREE SHIPPED CODE that predates the row:
`fnp_ufunc::par_select_two` (lib.rs ~29235, used by `par_select_median` /
`par_select_percentile` / `par_select_ranks`, engaged from the python surface for flat f64
n >= 1<<19) IS a multi-pass MSD radix narrow (8-bit digits of `f64_sortable_key`) and its
own gate comment records it ~2x FASTER than numpy's introselect at >= 512K. The rejected
candidate lost for three implementation-specific reasons par_select_two does not share:
  (a) it MATERIALIZED an O(n) u64 key buffer (par_select_two recomputes keys inline);
  (b) it COMPACTED the candidate set each level ("filter to it, recurse") — an O(n)
      allocation+copy per level (par_select_two histograms the full array with a cheap
      prefix test, no compaction);
  (c) it ran TWO independent selects + a keys clone for even n (par_select_two extracts
      both order statistics in ONE descent).
The row as written closes a family in which the repo already ships a win. AMENDMENT TEXT
(to be appended to the ledger in the next commit window):

> AMENDMENT (2026-07-10 audit, cc_fnp): scope-correct this REJECT. It rejects THAT
> candidate (key-materializing, per-level-compacting, even-n-doubling, python-layer
> re-implementation). It does NOT establish "radix-select loses to introselect for float
> order statistics": the shipped `fnp_ufunc::par_select_two` is a multi-pass radix narrow
> that beats introselect ~2x at >= 512K and already backs np.median/percentile/quantile
> flat-f64 at the surface. The retry-predicate ("only a single-pass in-place partition
> could beat introselect") was already satisfied in-tree when the row was written. Future
> float order-statistic levers should extend par_select_two's no-copy pattern, never the
> rejected candidate's materialize+compact pattern.

## Row 2 — 2026-07-04 "np.sort(float16) via f32 widening — 0.75x" : STANDS (fingerprint decisive)

fnp arm 132.0 ms vs numpy 99.1 ms: a dead-code (delegating) arm reads ≈ 99 ms + call
overhead; the +33 ms excess equals the astype(8 MB) + astype(16 MB) round-trip the root
cause names. The generalization ("lossless-upcast wins only where an fnp KERNEL advantage
exists; plain sort has none") is consistent with, not contradicted by, the shipped f16
unique/isin/searchsorted/setops wins (all kernel-advantage cases). No amendment needed.
Profile pass blocked (candidate was reverted via a BlackThrush stash that is not
recoverable with certainty; rebuilding requires a local build).

## Row 3 — 2026-07-02 "f32 flat argsort on TIE data — irreducible 1.05-1.2x" : STANDS (fingerprint decisive)

The measured loss RESPONDED to permuting the code under test (reordering the tie-oracle
before the NaN scan moved 1.11x -> 1.07x) — dead code cannot move a timing. The same path
wins 3.4x on distinct data, proving the fast path engages and the loss is the defer
pre-check on tie data, exactly as the row says. The "irreducible" claim is scoped to a
semantic constraint (numpy's f32 SIMD tie order is unreproducible, so byte-exactness
forces the defer) — not a perf ceiling; no in-tree code contradicts it. No amendment.

## Summary
- 1 of 3 rows requires a scope amendment (family-ceiling claim falsified by shipped code);
  its measurement and revert decision remain valid. 0 of 3 rows are dead-code-benched.
- Full profile-with-self-time verification of rows 1-2 requires rebuilding reverted
  candidates and running perf locally: BLOCKED under the no-local-build constraint (and
  perf is not routable through rch). Named blocker; fingerprints above are the strongest
  evidence obtainable this window.
