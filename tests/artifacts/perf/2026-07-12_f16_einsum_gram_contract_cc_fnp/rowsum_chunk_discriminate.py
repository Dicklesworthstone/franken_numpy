import numpy as np

def sum_tree(av):
    acc = np.float32(0.0); i = 0; n = len(av)
    while n - i > 4:
        acc = acc + ((av[i] + av[i+1]) + (av[i+2] + av[i+3])); i += 4
    while i < n:
        acc = acc + av[i]; i += 1
    return acc

def fold(av, bounds):
    out = np.float16(0.0)
    for s, e in bounds:
        out = np.float16(np.float32(out) + sum_tree(av[s:e]))
    return out

rng = np.random.default_rng(20260714)
for n in (8193, 9000, 16389, 20000):
    a = (rng.standard_normal((3, n)) * 0.3).astype(np.float16)
    af = a.astype(np.float32)
    want = np.einsum('ij->i', a)
    models = {
        "single": [[(0, n)]],
        "ch8192": [[(s, min(s+8192, n)) for s in range(0, n, 8192)]],
        "equal_ceil": None,
        "ch4096": [[(s, min(s+4096, n)) for s in range(0, n, 4096)]],
    }
    import math
    parts = math.ceil(n / 8192)
    base = n // parts
    rem = n % parts
    bounds = []
    pos = 0
    for p in range(parts):
        sz = base + (1 if p < rem else 0)
        bounds.append((pos, pos + sz)); pos += sz
    models["equal_ceil"] = [bounds]
    matches = []
    for name, blist in models.items():
        got = np.array([fold(af[i], blist[0]) for i in range(3)], dtype=np.float16)
        if want.tobytes() == got.tobytes():
            matches.append(name)
    print(f"n={n}: {matches}")
print(f"numpy={np.__version__}")
