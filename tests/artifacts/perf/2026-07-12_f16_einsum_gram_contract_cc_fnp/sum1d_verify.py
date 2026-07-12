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
        a0 = av[i:i+2]; a1 = av[i+2:i+4]; a2 = av[i+4:i+6]; a3 = av[i+6:i+8]
        v = ((a0 + a1) + (a2 + a3)) + v; i += 8
    while i < n:
        c = np.zeros(2); t = min(2, n - i); c[:t] = av[i:i+t]; v = c + v; i += t
    return (0.0 + v[0]) + v[1]

def lane4(av):
    n = len(av); v = np.zeros(4, dtype=np.float32); i = 0
    while n - i >= 16:
        a0 = av[i:i+4]; a1 = av[i+4:i+8]; a2 = av[i+8:i+12]; a3 = av[i+12:i+16]
        v = ((a0 + a1) + (a2 + a3)) + v; i += 16
    while i < n:
        c = np.zeros(4, dtype=np.float32); t = min(4, n - i); c[:t] = av[i:i+t]; v = c + v; i += t
    return np.float32(np.float32(v[0] + v[1]) + np.float32(v[2] + v[3]))

rng = np.random.default_rng(20260714)
fails = total = 0
for n in (100000, 8193, 2_000_003):
    a16 = (rng.standard_normal(n) * 0.3).astype(np.float16)
    af = a16.astype(np.float32)
    out = np.float16(0.0)
    for s in range(0, n, 8192):
        out = np.float16(np.float32(out) + tree_f16(af[s:s+8192]))
    total += 1
    if np.float16(np.einsum('i->', a16)).tobytes() != out.tobytes():
        fails += 1; print(f"f16 MISMATCH n={n}")
    a64 = rng.standard_normal(n)
    o = 0.0
    for s in range(0, n, 8192):
        o = o + lane2(a64[s:s+8192])
    total += 1
    if np.float64(np.einsum('i->', a64)).tobytes() != np.float64(o).tobytes():
        fails += 1; print(f"f64 MISMATCH n={n}")
    a32 = a64.astype(np.float32)
    o32 = np.float32(0.0)
    for s in range(0, n, 8192):
        o32 = o32 + lane4(a32[s:s+8192])
    total += 1
    if np.float32(np.einsum('i->', a32)).tobytes() != np.float32(o32).tobytes():
        fails += 1; print(f"f32 MISMATCH n={n}")
print(f"SUM1D_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
