# f16 einsum full contraction ('ij,ij->', 'ijk,ijk->'): C-contiguous same-shape
# operands coalesce to 1-D, so the contract should equal the shipped dot1d
# chunk-fold with k = prod(shape).
import numpy as np

def tree_f32(av, bv):
    k = av.shape[0]; acc = np.float32(0.0); j = 0
    while j + 4 <= k:
        ab = av[j:j+4] * bv[j:j+4]
        acc = acc + (((ab[0] + ab[1]) + ab[2]) + ab[3]); j += 4
    while j < k:
        acc = acc + av[j] * bv[j]; j += 1
    return acc

def dot_fold(af, bf, CH=8192):
    out = np.float16(0.0)
    for s in range(0, af.shape[0], CH):
        out = np.float16(np.float32(out) + tree_f32(af[s:s+CH], bf[s:s+CH]))
    return out

rng = np.random.default_rng(20260713)
fails = total = 0
cases_2d = [(64, 100), (127, 65), (96, 86), (3, 2731), (500, 40)]
for (m, n) in cases_2d:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((m, n)) * scale).astype(np.float16)
        b = (rng.standard_normal((m, n)) * scale).astype(np.float16)
        want = np.float16(np.einsum('ij,ij->', a, b))
        got = dot_fold(a.astype(np.float32).ravel(), b.astype(np.float32).ravel())
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"2D MISMATCH m={m} n={n} scale={scale}")
for (p, q, r) in [(7, 41, 33), (4, 64, 65), (2, 3, 4000)]:
    for scale in (0.3, 30.0):
        total += 1
        a = (rng.standard_normal((p, q, r)) * scale).astype(np.float16)
        b = (rng.standard_normal((p, q, r)) * scale).astype(np.float16)
        want = np.float16(np.einsum('ijk,ijk->', a, b))
        got = dot_fold(a.astype(np.float32).ravel(), b.astype(np.float32).ravel())
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"3D MISMATCH {p}x{q}x{r} scale={scale}")
print(f"FULLCONTRACT_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
