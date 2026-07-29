import numpy as np

def tree_f32(av, bv):
    k = av.shape[0]; acc = np.float32(0.0); j = 0
    while j + 4 <= k:
        ab = av[j:j+4] * bv[j:j+4]
        acc = acc + (((ab[0] + ab[1]) + ab[2]) + ab[3]); j += 4
    while j < k:
        acc = acc + av[j] * bv[j]; j += 1
    return acc

def elem_chunked(av, bv, CH):
    out = np.float16(0.0)
    for s in range(0, av.shape[0], CH):
        out = np.float16(np.float32(out) + tree_f32(av[s:s+CH], bv[s:s+CH]))
    return out

rng = np.random.default_rng(20260712)
for (B, m, k, n) in [(2, 2, 2001, 3), (1, 4, 9000, 3), (2, 3, 8193, 2), (1, 2, 16389, 2), (1, 2, 24576, 2), (1, 2, 8192, 2), (1, 1, 9000, 1)]:
    a = (rng.standard_normal((B, m, k)) * 0.3).astype(np.float16)
    x = (rng.standard_normal((B, n, k)) * 0.3).astype(np.float16)
    want = np.einsum('bij,blj->bil', a, x)
    af, xf = a.astype(np.float32), x.astype(np.float32)
    matches = []
    for name, ch in [("single", None), ("ch8192", 8192), ("ch4096", 4096), ("ch2048", 2048), ("ch16384", 16384)]:
        got = np.empty((B, m, n), dtype=np.float16)
        for b in range(B):
            for i in range(m):
                for l in range(n):
                    if ch is None:
                        got[b, i, l] = np.float16(np.float32(0.0) + tree_f32(af[b, i], xf[b, l]))
                    else:
                        got[b, i, l] = elem_chunked(af[b, i], xf[b, l], ch)
        if want.tobytes() == got.tobytes():
            matches.append(name)
    print(f"B={B} m={m} k={k} n={n}: {matches}")
print(f"numpy={np.__version__}")
