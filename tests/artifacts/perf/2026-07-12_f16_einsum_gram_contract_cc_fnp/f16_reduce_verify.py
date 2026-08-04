# f16 einsum reduction contracts ('ij->i', 'ij->j', 'ij->'):
# sum_of_arr half tree: while count>4: acc += (d0+d1)+(d2+d3); tail 1-at-a-time
# (NOTE: strictly >4 - a trailing 4 goes through the tail).
import numpy as np

def sum_tree(av):
    acc = np.float32(0.0); i = 0; n = len(av)
    while n - i > 4:
        acc = acc + ((av[i] + av[i+1]) + (av[i+2] + av[i+3])); i += 4
    while i < n:
        acc = acc + av[i]; i += 1
    return acc

def fold_1d(av, CH=8192):
    out = np.float16(0.0)
    for s in range(0, len(av), CH):
        out = np.float16(np.float32(out) + sum_tree(av[s:s+CH]))
    return out

rng = np.random.default_rng(20260714)
fails = total = 0
for (m, n) in [(37, 41), (64, 100), (5, 8193), (3, 9000), (200, 64), (7, 4), (2, 8192)]:
    for scale in (0.3, 30.0):
        a = (rng.standard_normal((m, n)) * scale).astype(np.float16)
        af = a.astype(np.float32)
        # row-sum 'ij->i': per-row chunk-fold (single tree when n<=8192)
        total += 1
        want = np.einsum('ij->i', a)
        got = np.array([np.float16(np.float32(0.0) + sum_tree(af[i])) for i in range(m)], dtype=np.float16)
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"ROW MISMATCH m={m} n={n} scale={scale}")
        # col-sum 'ij->j': per-step narrow rows over ascending i
        total += 1
        want = np.einsum('ij->j', a)
        acc = np.zeros(n, dtype=np.float16)
        for i in range(m):
            acc = (acc.astype(np.float32) + af[i]).astype(np.float16)
        if want.tobytes() != acc.tobytes():
            fails += 1; print(f"COL MISMATCH m={m} n={n} scale={scale}")
        # full-sum 'ij->': coalesced chunk-fold
        total += 1
        want = np.float16(np.einsum('ij->', a))
        got = fold_1d(af.ravel())
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"FULL MISMATCH m={m} n={n} scale={scale}")
print(f"F16_REDUCE_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
