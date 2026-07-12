import numpy as np

def lane_tree(av, fold):
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
    return fold(v)

folds = {
    "seq": lambda v: np.float32(np.float32(np.float32(np.float32(0.0) + v[0]) + v[1]) + v[2]) + v[3],
    "shuf_02_13": lambda v: np.float32(np.float32(v[0] + v[2]) + np.float32(v[1] + v[3])),
    "hadd_01_23": lambda v: np.float32(np.float32(v[0] + v[1]) + np.float32(v[2] + v[3])),
}
rng = np.random.default_rng(20260714)
for n in (41, 64, 1000, 8193, 16389):
    a = (rng.standard_normal((3, n)) * 0.3).astype(np.float32)
    want = np.einsum('ij->i', a)
    matches = []
    for fname, ffn in folds.items():
        got = np.array([lane_tree(a[i], ffn) for i in range(3)], dtype=np.float32)
        if got.tobytes() == want.tobytes():
            matches.append(fname)
    print(f"n={n}: {matches}")
print(f"numpy={np.__version__}")
