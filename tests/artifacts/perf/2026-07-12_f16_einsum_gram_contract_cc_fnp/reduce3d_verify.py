# 3-D einsum reductions mapped onto the 2-D kernel modes (f64 lane2/f16 tree refs).
import numpy as np

def tree_f16(av):
    acc = np.float32(0.0); i = 0; n = len(av)
    while n - i > 4:
        acc = acc + ((av[i] + av[i+1]) + (av[i+2] + av[i+3])); i += 4
    while i < n:
        acc = acc + av[i]; i += 1
    return acc

def lane2(av):
    n = len(av); v = np.zeros(2); i = 0
    while n - i >= 8:
        v = ((av[i:i+2] + av[i+2:i+4]) + (av[i+4:i+6] + av[i+6:i+8])) + v; i += 8
    while i < n:
        c = np.zeros(2); t = min(2, n - i); c[:t] = av[i:i+t]; v = c + v; i += t
    return (0.0 + v[0]) + v[1]

rng = np.random.default_rng(20260715)
fails = total = 0
for (p, q, r) in [(7, 9, 11), (16, 100, 65), (3, 5, 8193), (4, 2049, 5)]:
    a64 = rng.standard_normal((p, q, r))
    a16 = (rng.standard_normal((p, q, r)) * 0.3).astype(np.float16)
    a16f = a16.astype(np.float32)
    cases = [
        # (spec, ref64, ref16) via flattening onto row/col/full modes
        ('ijk->ij', lambda a: np.array([lane2(row) for row in a.reshape(-1, a.shape[2])]).reshape(a.shape[0], a.shape[1]),
                    lambda a: np.array([np.float16(np.float32(0.0) + tree_f16(row)) for row in a.reshape(-1, a.shape[2])], dtype=np.float16).reshape(a.shape[0], a.shape[1])),
        ('ijk->i',  lambda a: np.array([lane2(a[i].ravel()) for i in range(a.shape[0])]),
                    lambda a: np.array([np.float16(np.float32(0.0) + tree_f16(a[i].ravel())) for i in range(a.shape[0])], dtype=np.float16)),
        ('ijk->k',  lambda a: (lambda f: [f.__setitem__(slice(None), 0) or None, [f.__iadd__(row) for row in a.reshape(-1, a.shape[2])], f][2])(np.zeros(a.shape[2])),
                    None),
        ('ijk->jk', lambda a: (lambda f: [ [f.__iadd__(sl) for sl in a.reshape(a.shape[0], -1)], f][1])(np.zeros(a.shape[1]*a.shape[2])).reshape(a.shape[1], a.shape[2]),
                    None),
    ]
    for spec, ref64, ref16 in cases:
        total += 1
        want = np.einsum(spec, a64)
        got = np.asarray(ref64(a64))
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"f64 MISMATCH {spec} {p}x{q}x{r}")
        if ref16 is not None:
            total += 1
            want = np.einsum(spec, a16)
            got = np.asarray(ref16(a16f))
            if want.tobytes() != got.tobytes():
                fails += 1; print(f"f16 MISMATCH {spec} {p}x{q}x{r}")
    # f16 col-style specs: per-step narrow chains over flattened leading axes
    for spec, reshape_mn in (('ijk->k', lambda a: (a.shape[0]*a.shape[1], a.shape[2])), ('ijk->jk', lambda a: (a.shape[0], a.shape[1]*a.shape[2]))):
        total += 1
        m, n = reshape_mn(a16)
        flat = a16f.reshape(m, n)
        acc = np.zeros(n, dtype=np.float16)
        for i in range(m):
            acc = (acc.astype(np.float32) + flat[i]).astype(np.float16)
        want = np.einsum(spec, a16)
        if want.tobytes() != acc.reshape(want.shape).tobytes():
            fails += 1; print(f"f16 col MISMATCH {spec} {p}x{q}x{r}")
print(f"REDUCE3D_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
