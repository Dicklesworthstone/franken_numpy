# Refined model for 'bij,blj->bil': with B>1 and m,n>1 (no coalescing to the
# ndim-3 unbuffered loop), EVERY output element folds per-8192-chunk wide
# trees through f16 store/reload; k<=8192 degenerates to the single tree.
import numpy as np

def tree_f32(av, bv):
    k = av.shape[0]; acc = np.float32(0.0); j = 0
    while j + 4 <= k:
        ab = av[j:j+4] * bv[j:j+4]
        acc = acc + (((ab[0] + ab[1]) + ab[2]) + ab[3]); j += 4
    while j < k:
        acc = acc + av[j] * bv[j]; j += 1
    return acc

def elem(av, bv, CH=8192):
    out = np.float16(0.0)
    for s in range(0, av.shape[0], CH):
        out = np.float16(np.float32(out) + tree_f32(av[s:s+CH], bv[s:s+CH]))
    return out

rng = np.random.default_rng(20260713)
fails = total = 0
for (B, m, k, n) in [(2, 3, 8193, 2), (3, 4, 9000, 3), (2, 2, 16389, 2), (2, 2, 24576, 2),
                     (4, 5, 100_000, 2), (2, 6, 513, 5), (3, 2, 2001, 3), (2, 3, 8192, 4)]:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((B, m, k)) * scale).astype(np.float16)
        x = (rng.standard_normal((B, n, k)) * scale).astype(np.float16)
        want = np.einsum('bij,blj->bil', a, x)
        af, xf = a.astype(np.float32), x.astype(np.float32)
        got = np.empty((B, m, n), dtype=np.float16)
        for b in range(B):
            for i in range(m):
                for l in range(n):
                    got[b, i, l] = elem(af[b, i], xf[b, l])
        if want.tobytes() != got.tobytes():
            fails += 1
            print(f"MISMATCH B={B} m={m} k={k} n={n} scale={scale}")
print(f"BATCHED_T_MODEL numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
