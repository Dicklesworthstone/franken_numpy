import numpy as np

rng = np.random.default_rng(20260715)
fails_a = fails_b = total = 0
for n in (1000, 100000, 8193):
    for scale in (0.3, 30.0):
        a = (rng.standard_normal(n) * scale).astype(np.float16)
        af = a.astype(np.float32)
        want = np.cumsum(a)
        # A: per-step f16 narrow
        got_a = np.empty(n, dtype=np.float16)
        acc = np.float16(0.0)
        # note: first element = a[0] itself? cumsum[0] = a[0] (no seed add?) test both
        acc = a[0]
        got_a[0] = acc
        for i in range(1, n):
            acc = np.float16(np.float32(acc) + af[i])
            got_a[i] = acc
        # B: wide f32 accumulator, narrow per output
        got_b = np.empty(n, dtype=np.float16)
        accw = np.float32(af[0])
        got_b[0] = np.float16(accw)
        for i in range(1, n):
            accw = accw + af[i]
            got_b[i] = np.float16(accw)
        total += 1
        if want.tobytes() != got_a.tobytes(): fails_a += 1
        if want.tobytes() != got_b.tobytes(): fails_b += 1
print(f"per-step-narrow fails: {fails_a}/{total}; wide-acc fails: {fails_b}/{total}")
print(f"numpy={np.__version__}")
