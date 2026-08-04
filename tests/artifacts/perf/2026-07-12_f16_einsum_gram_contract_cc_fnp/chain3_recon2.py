import numpy as np
rng = np.random.default_rng(20260715)
ok = 0; total = 0
for (m, k1, k2, n) in [(8, 9, 10, 11), (16, 16, 16, 16), (5, 30, 4, 7), (24, 8, 40, 6), (64, 64, 64, 64)]:
    a = (rng.standard_normal((m, k1)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((k1, k2)) * 0.3).astype(np.float16)
    c = (rng.standard_normal((k2, n)) * 0.3).astype(np.float16)
    path, _ = np.einsum_path('ij,jk,kl->il', a, b, c, optimize=True)
    w = np.einsum('ij,jk,kl->il', a, b, c, optimize=True)
    mm = a @ (b @ c) if path[1] == (1, 2) else (a @ b) @ c
    total += 1; ok += mm.tobytes() == w.tobytes()
    print(f"({m},{k1},{k2},{n}) path={path[1:]} matmul-pairs={mm.tobytes() == w.tobytes()}")
print(f"numpy={np.__version__} ok={ok}/{total}")
