# 'ijk->j': hypothesis - out[j] = fold over ascending i of
#   dt(out[j] + tree(a[i,j,:]))  (tree = the dtype's pinned sum_of_arr form).
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
for (p, q, r) in [(7, 9, 11), (16, 40, 65), (3, 5, 8193), (9, 2049, 5)]:
    a64 = rng.standard_normal((p, q, r))
    total += 1
    want = np.einsum('ijk->j', a64)
    got = np.zeros(q)
    for i in range(p):
        for j in range(q):
            got[j] = got[j] + lane2(a64[i, j])
    if want.tobytes() != got.tobytes():
        fails += 1; print(f"f64 MISMATCH {p}x{q}x{r}")
    a16 = (rng.standard_normal((p, q, r)) * 0.3).astype(np.float16)
    af = a16.astype(np.float32)
    total += 1
    want = np.einsum('ijk->j', a16)
    got16 = np.zeros(q, dtype=np.float16)
    for i in range(p):
        for j in range(q):
            got16[j] = np.float16(np.float32(got16[j]) + tree_f16(af[i, j]))
    if want.tobytes() != got16.tobytes():
        fails += 1; print(f"f16 MISMATCH {p}x{q}x{r}")
print(f"MIDKEEP_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
