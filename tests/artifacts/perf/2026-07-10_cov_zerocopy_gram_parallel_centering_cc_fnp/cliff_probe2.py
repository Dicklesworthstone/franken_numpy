import sys, time
import numpy as np
sys.path.insert(0, sys.argv[2])
import fnp_python as fnp
rng = np.random.default_rng(0)
def bench(fn, reps=9):
    fn(); b=1e9
    for _ in range(reps):
        t=time.perf_counter(); fn(); b=min(b,time.perf_counter()-t)
    return b*1e3
for nv, no in [(2000,50),(2000,128),(2000,500)]:
    X = rng.standard_normal((nv,no))
    print(f"{sys.argv[1]} cov {nv}x{no}: {bench(lambda: fnp.cov(X)):.2f} ms")
