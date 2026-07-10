# Perf worker: one process = one library (np126 venv numpy, or fnp .so dir via argv).
# Prints min-of-7 ms per probe plus an untouched matmul-2048^2 gauge.
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

tag = sys.argv[1]
if tag.startswith("fnp"):
    sys.path.insert(0, sys.argv[2])
    import fnp_python as fnp
    covfn, ccfn = fnp.cov, fnp.corrcoef
else:
    covfn, ccfn = np.cov, np.corrcoef

rng = np.random.default_rng(0)

def bench(fn, reps=7):
    fn(); b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1e3

shapes = [(2000, 500), (1000, 1000), (500, 5000), (200, 1000)]
for nv, no in shapes:
    X = rng.standard_normal((nv, no))
    print(f"{tag} cov {nv}x{no}: {bench(lambda: covfn(X)):.2f}")
    print(f"{tag} corrcoef {nv}x{no}: {bench(lambda: ccfn(X)):.2f}")
A = rng.standard_normal((2048, 2048)); B = rng.standard_normal((2048, 2048))
print(f"{tag} gauge matmul2048: {bench(lambda: A @ B, 3):.1f}")
