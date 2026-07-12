# f64 einsum reduction contracts: row-sum = L=2 lane tree per row (npyv SSE2
# baseline, einsum_sumprod NOT runtime-dispatched); col-sum = per-step f64
# adds over ascending i; full-sum = per-8192-chunk L=2 trees folded serially.
import numpy as np

def lane_tree(av):
    n = len(av); L = 4
    v = np.zeros(L, dtype=np.float32); i = 0
    while n - i >= 4 * L:
        a0 = av[i:i+L]; a1 = av[i+L:i+2*L]; a2 = av[i+2*L:i+3*L]; a3 = av[i+3*L:i+4*L]
        v = ((a0 + a1) + (a2 + a3)) + v
        i += 4 * L
    while i < n:
        chunk = np.zeros(L, dtype=np.float32); take = min(L, n - i)
        chunk[:take] = av[i:i+take]
        v = chunk + v; i += take
    acc = np.float32(0.0)
    for x in v:
        acc += x
    return acc

def fold_full(av, CH=8192):
    out = np.float32(0.0)
    for s in range(0, len(av), CH):
        out = out + lane_tree(av[s:s+CH])
    return np.float32(out)

rng = np.random.default_rng(20260714)
fails = total = 0
for (m, n) in [(37, 41), (5, 8193), (3, 9000), (200, 64), (7, 4), (64, 1000)]:
    for scale in (0.3, 1e30):
        a = (rng.standard_normal((m, n)) * scale).astype(np.float32)
        total += 1
        want = np.einsum('ij->i', a)
        got = np.array([lane_tree(a[i]) for i in range(m)])
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"ROW MISMATCH m={m} n={n} scale={scale}")
        total += 1
        want = np.einsum('ij->j', a)
        acc = np.zeros(n, dtype=np.float32)
        for i in range(m):
            acc = acc + a[i]
        if want.tobytes() != acc.tobytes():
            fails += 1; print(f"COL MISMATCH m={m} n={n} scale={scale}")
        total += 1
        want = np.float32(np.einsum('ij->', a))
        got = fold_full(a.ravel())
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"FULL MISMATCH m={m} n={n} scale={scale}")
print(f"F32_REDUCE_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
