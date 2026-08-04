# Contract verification for f16 einsum 'ji,jl->il' (a.T@b idiom):
# hypothesis (from einsum_sumprod.c.src stride0_contig_outcontig_two ->
# sum_of_products_muladd, half scalar path): out[i,l] accumulates over
# ascending j as acc = f16(f32(a[j,i]) * f32(b[j,l]) + f32(acc)), acc0 = f16(0).
import numpy as np

rng = np.random.default_rng(20260712)
fails = total = 0
shapes = [(k, m, n) for k in list(range(1, 24)) + [64, 100, 333, 1000, 2000]
          for (m, n) in [(3, 5), (8, 4)]]
for k, m, n in shapes:
    for scale in (1.0, 100.0):
        total += 1
        a = (rng.standard_normal((k, m)) * scale).astype(np.float16)
        b = (rng.standard_normal((k, n)) * scale).astype(np.float16)
        want = np.einsum('ji,jl->il', a, b)
        acc = np.zeros((m, n), dtype=np.float16)
        af32 = a.astype(np.float32); bf32 = b.astype(np.float32)
        for j in range(k):
            acc = (af32[j, :, None] * bf32[j, None, :]
                   + acc.astype(np.float32)).astype(np.float16)
        if want.tobytes() != acc.tobytes():
            fails += 1
            if fails <= 3:
                bad = np.nonzero(want.view(np.int16) != acc.view(np.int16))
                print(f"MISMATCH k={k} m={m} n={n} scale={scale} nbad={len(bad[0])}")
print(f"CONTRACT_VERIFY gram_muladd numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
