# 'ijk->ik' (middle-axis sum): hypothesis - per i-slice, per-step chains over
# ascending j (the col-mode contract per slice), f16 narrowing / f64 plain adds.
import numpy as np

rng = np.random.default_rng(20260715)
fails = total = 0
for (p, q, r) in [(7, 9, 11), (16, 100, 65), (3, 8193, 5), (4, 5, 2049), (2, 9000, 3)]:
    a64 = rng.standard_normal((p, q, r))
    a16 = (rng.standard_normal((p, q, r)) * 0.3).astype(np.float16)
    total += 1
    want = np.einsum('ijk->ik', a64)
    got = np.zeros((p, r))
    for i in range(p):
        for j in range(q):
            got[i] = got[i] + a64[i, j]
    if want.tobytes() != got.tobytes():
        fails += 1; print(f"f64 MISMATCH {p}x{q}x{r}")
    total += 1
    want = np.einsum('ijk->ik', a16)
    af = a16.astype(np.float32)
    got16 = np.zeros((p, r), dtype=np.float16)
    for i in range(p):
        acc = np.zeros(r, dtype=np.float16)
        for j in range(q):
            acc = (acc.astype(np.float32) + af[i, j]).astype(np.float16)
        got16[i] = acc
    if want.tobytes() != got16.tobytes():
        fails += 1; print(f"f16 MISMATCH {p}x{q}x{r}")
    total += 1
    a32 = a64.astype(np.float32)
    want = np.einsum('ijk->ik', a32)
    got32 = np.zeros((p, r), dtype=np.float32)
    for i in range(p):
        for j in range(q):
            got32[i] = got32[i] + a32[i, j]
    if want.tobytes() != got32.tobytes():
        fails += 1; print(f"f32 MISMATCH {p}x{q}x{r}")
print(f"MIDAXIS_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
