# Contract verify for f16 einsum 'bji,bjl->bil' (batched a.T@b): hypothesis -
# the 2-op gram per-step muladd chain per batch slice, chunk-IMMUNE (per-step
# narrows make buffering invisible). Method rule: verify k straddling 8192,
# B=1 vs B>1, and n>8192 (row-dim chunking) anyway.
import numpy as np

rng = np.random.default_rng(20260713)
fails = total = 0
cases = [(2, 5, 7, 4), (3, 96, 130, 80), (2, 8193, 3, 4), (1, 8193, 3, 4),
         (2, 9000, 2, 3), (1, 2001, 4, 5), (2, 130, 3, 9000), (2, 33, 17, 29)]
for (B, k, m, n) in cases:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((B, k, m)) * scale).astype(np.float16)
        x = (rng.standard_normal((B, k, n)) * scale).astype(np.float16)
        want = np.einsum('bji,bjl->bil', a, x)
        af, xf = a.astype(np.float32), x.astype(np.float32)
        acc = np.zeros((B, m, n), dtype=np.float16)
        for j in range(k):
            acc = (af[:, j, :, None] * xf[:, j, None, :]
                   + acc.astype(np.float32)).astype(np.float16)
        if want.tobytes() != acc.tobytes():
            fails += 1
            print(f"MISMATCH B={B} k={k} m={m} n={n} scale={scale}")
print(f"BATCHED_GRAM_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
