import numpy as np

def lane_tree(av, L):
    n = len(av)
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

def seq(av):
    acc = np.float32(0.0)
    for x in av:
        acc += x
    return acc

rng = np.random.default_rng(20260714)
for n in (41, 64, 1000, 8193):
    a = (rng.standard_normal((3, n)) * 0.3).astype(np.float32)
    want = np.einsum('ij->i', a)
    cands = {f"L{L}": np.array([lane_tree(a[i], L) for i in range(3)], dtype=np.float32) for L in (2, 4, 8, 16)}
    cands["seq"] = np.array([seq(a[i]) for i in range(3)], dtype=np.float32)
    cands["pairwise"] = np.add.reduce(a, axis=1)
    print(f"n={n}: {[k for k, v in cands.items() if v.tobytes() == want.tobytes()]}")
print(f"numpy={np.__version__}")
