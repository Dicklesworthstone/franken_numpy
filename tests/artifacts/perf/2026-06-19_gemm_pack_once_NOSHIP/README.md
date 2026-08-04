# 2026-06-19 GEMM pack-once / A-reuse reorder — NO-SHIP (negative evidence)

Agent: CobaltForge (claude-opus-4-8). Host: thinkstation1 (Threadripper PRO 5975WX,
32c/64t, L2 512KB/core, L3 128MB). NumPy 2.4.3 scipy-openblas64 (MAX_THREADS=64).

## Motivation
Broadest measured fnp-vs-numpy gap found this session: `np.matmul`/`np.dot` on 2-D
f64 in the native-GEMM gate window (~320..1024). On the UNLOADED box numpy's 64-thread
OpenBLAS hits 260-500 GFLOP/s; the native register-tiled kernel
(`matmul_accumulate`) saturates ~76-90 GFLOP/s → matmul 2.3-6.5x slower than numpy.
Native GEMM only wins when numpy is thread-starved (OPENBLAS_NUM_THREADS=1: native
0.59-1.03x); with >=2 BLAS threads numpy wins. The gate is therefore a deliberate
load-aware design that loses on an unloaded multithreaded-BLAS box.

## Hypothesis tested
Native parallel GEMM thread-scaling plateaus (RAYON 1→4→16→64 = 27→58→73→76 GFLOP/s)
→ memory-bandwidth bound. Suspected cause: the parallel driver calls the serial
kernel per band, and the serial kernel re-packs ALL of B per call → with threads*4
bands, B is repacked O(threads) times (~2 GB redundant memcpy at n=1024). Fix tried:
pack B once into a shared NR-micropanel buffer; bands read the shared pack.
Two loop orders tried for the band kernel:
  (A) i0-outer / panel-inner  (A micro-tile L1-reuse)
  (B) panel-outer / i0-inner  (B micropanel L1-reuse; "hybrid")
Bit-exactness preserved (full-K increasing-kk accumulation per MR×NR tile); new test
`matmul_parallel_prepacked_matches_serial_bits` + golden `matmul_row_blocking_...`
both green.

## Measurement (RIGOROUS: OLD vs NEW timed back-to-back in ONE binary)
Cross-invocation rch A/B was unreliable (shared workers, 152-293 GFLOP/s variance for
the SAME build). Interleaved same-binary A/B (controls machine+load):

Variant (A) i0-outer:    n768 0.73-0.89x  n1024 0.82-1.00x  n1536 0.80-0.87x  n2048 0.74-0.91x
Variant (B) hybrid:      n768 0.68-0.69x  n1024 0.82-0.85x  n1536 0.92-0.96x  n2048 1.17-1.19x

NEW is SLOWER than OLD at the gated sizes (768/1024) in every interleaved run; only
n=2048 (BEYOND the gate, internal-use only) improves ~1.18x. The OLD per-band B-pack
is already well-amortized and prefetch-friendly; the 8MB shared-pack allocation +
upfront pack adds memory traffic that is not recovered at the relevant sizes.

## Verdict: REVERTED (git stash). Ledger row: LOSS (regression) — do not ship.
The "pack-once eliminates 2GB redundant memcpy" hypothesis is empirically false: the
redundant pack is not the bottleneck; the OLD kernel is bandwidth-bound on the data
streams themselves, which pack-once does not reduce at kc=K (bit-exactness forbids
K-blocking, the only lever that would cut B re-streaming).

## Standing recommendation (human decision)
The native-GEMM gate (`PY_NATIVE_GEMM_MIN_FLOPS`/`MAX_DIM`) is a NET LOSS vs numpy on
an unloaded box with multithreaded OpenBLAS, and a WIN only when numpy is thread-
starved. Whether to make it load/thread-adaptive (e.g. defer when OPENBLAS_NUM_THREADS
unset/high) is a design call left to maintainers — not unilaterally flipped here.
