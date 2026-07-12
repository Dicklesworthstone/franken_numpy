# Rank the remaining unmeasured einsum surface by numpy self-time.
import time
import numpy as np

rng = np.random.default_rng(20260714)
def t(fn, reps=5):
    fn()  # warm
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1000

f16 = np.float16
a2 = (rng.standard_normal((512, 512)) * 0.3).astype(f16)
b2 = (rng.standard_normal((512, 512)) * 0.3).astype(f16)
c2 = (rng.standard_normal((512, 512)) * 0.3).astype(f16)
big = (rng.standard_normal((2896, 2896)) * 0.3).astype(f16)
sq = (rng.standard_normal((2896, 2896)) * 0.3).astype(f16)
d64_3 = [rng.standard_normal((512, 512)) for _ in range(3)]
rows = [
    ("3op chain f16 'ij,jk,kl->il' 128^3 (O(n^4) UNOPT - 512 took >2min)", lambda: np.einsum('ij,jk,kl->il', a2[:128,:128], b2[:128,:128], c2[:128,:128])),
    ("3op chain f16 optimize=True", lambda: np.einsum('ij,jk,kl->il', a2, b2, c2, optimize=True)),
    ("3op hadamard f64 'ij,ij,ij->ij' 512^2", lambda: np.einsum('ij,ij,ij->ij', *d64_3)),
    ("f16 row-sum 'ij->i' 2896^2", lambda: np.einsum('ij->i', big)),
    ("f16 col-sum 'ij->j' 2896^2", lambda: np.einsum('ij->j', big)),
    ("f16 full-sum 'ij->' 2896^2", lambda: np.einsum('ij->', big)),
    ("f16 transpose 'ij->ji' 2896^2", lambda: np.einsum('ij->ji', big)),
    ("f16 diag 'ii->i' 2896^2", lambda: np.einsum('ii->i', sq)),
    ("f16 trace 'ii->' 2896^2", lambda: np.einsum('ii->', sq)),
    ("f64 row-sum 'ij->i' 2896^2", lambda: np.einsum('ij->i', big.astype(np.float64))),
]
for name, fn in rows:
    try:
        print(f"{t(fn):10.2f} ms  {name}")
    except Exception as e:
        print(f"      ERR   {name}: {e}")
print(f"numpy={np.__version__}")
