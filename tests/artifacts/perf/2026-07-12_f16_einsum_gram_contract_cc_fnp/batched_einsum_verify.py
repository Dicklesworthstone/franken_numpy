# Contract verify for f16 einsum 'bij,bjk->bik' (batched matmul spec):
# hypothesis - each batch slice follows the plain-spec per-step-narrow chain
# acc = f16(f32(acc) + f32(a[b,i,j])*f32(x[b,j,l])) over ascending j, acc0=f16(0).
import numpy as np

rng = np.random.default_rng(20260712)
fails = total = 0
for (B, m, k, n) in [(3, 5, 7, 4), (2, 8, 96, 6), (4, 3, 513, 5), (2, 2, 2001, 3),
                     (1, 6, 9000, 4), (5, 4, 8192, 3), (2, 9, 8193, 2)]:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((B, m, k)) * scale).astype(np.float16)
        x = (rng.standard_normal((B, k, n)) * scale).astype(np.float16)
        want = np.einsum('bij,bjk->bik', a, x)
        af, xf = a.astype(np.float32), x.astype(np.float32)
        got = np.empty((B, m, n), dtype=np.float16)
        for b in range(B):
            for i in range(m):
                for l in range(n):
                    acc = np.float16(0.0)
                    for j in range(k):
                        acc = np.float16(np.float32(acc) + af[b, i, j] * xf[b, j, l])
                    got[b, i, l] = acc
        if want.tobytes() != got.tobytes():
            fails += 1
            if fails <= 3:
                bad = int(np.count_nonzero(want.view(np.int16) != got.view(np.int16)))
                print(f"MISMATCH B={B} m={m} k={k} n={n} scale={scale} nbad={bad}/{want.size}")
print(f"BATCHED_EINSUM_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
