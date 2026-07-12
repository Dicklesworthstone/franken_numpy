# Output-swapped batched specs, discriminators re-run per layout:
# 'bij,blj->bli' (transposed-swapped): B>1,m>1,n>1 -> per-8192-chunk wide-tree
#   fold per element (buffered ndim-4); B==1 -> single tree (ndim-3 coalesce).
# 'bji,bjl->bli' (gram-swapped): per-step chain per element, chunk-immune.
import numpy as np

def tree_f32(av, bv):
    k = av.shape[0]; acc = np.float32(0.0); j = 0
    while j + 4 <= k:
        ab = av[j:j+4] * bv[j:j+4]
        acc = acc + (((ab[0] + ab[1]) + ab[2]) + ab[3]); j += 4
    while j < k:
        acc = acc + av[j] * bv[j]; j += 1
    return acc

def elem_fold(av, bv, CH=8192):
    out = np.float16(0.0)
    for s in range(0, av.shape[0], CH):
        out = np.float16(np.float32(out) + tree_f32(av[s:s+CH], bv[s:s+CH]))
    return out

rng = np.random.default_rng(20260713)
fails = total = 0
# transposed-swapped: a (B,m,k), x (B,n,k), out (B,n,m)
for (B, m, k, n) in [(2, 3, 8193, 4), (3, 4, 9000, 3), (2, 6, 513, 5), (1, 4, 9000, 3), (2, 2, 16389, 2)]:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((B, m, k)) * scale).astype(np.float16)
        x = (rng.standard_normal((B, n, k)) * scale).astype(np.float16)
        want = np.einsum('bij,blj->bli', a, x)
        af, xf = a.astype(np.float32), x.astype(np.float32)
        got = np.empty((B, n, m), dtype=np.float16)
        for b in range(B):
            for l in range(n):
                for i in range(m):
                    if B == 1:
                        got[b, l, i] = np.float16(np.float32(0.0) + tree_f32(af[b, i], xf[b, l]))
                    else:
                        got[b, l, i] = elem_fold(af[b, i], xf[b, l])
        if want.tobytes() != got.tobytes():
            fails += 1
            print(f"T-SWAP MISMATCH B={B} m={m} k={k} n={n} scale={scale}")
# gram-swapped: a (B,k,m), x (B,k,n), out (B,n,m)
for (B, k, m, n) in [(2, 8193, 3, 4), (1, 9000, 4, 3), (2, 130, 5, 6), (3, 33, 17, 9)]:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((B, k, m)) * scale).astype(np.float16)
        x = (rng.standard_normal((B, k, n)) * scale).astype(np.float16)
        want = np.einsum('bji,bjl->bli', a, x)
        af, xf = a.astype(np.float32), x.astype(np.float32)
        acc = np.zeros((B, n, m), dtype=np.float16)
        for j in range(k):
            acc = (xf[:, j, :, None] * af[:, j, None, :]
                   + acc.astype(np.float32)).astype(np.float16)
        if want.tobytes() != acc.tobytes():
            fails += 1
            print(f"G-SWAP MISMATCH B={B} k={k} m={m} n={n} scale={scale}")
print(f"BATCHED_SWAPPED_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
