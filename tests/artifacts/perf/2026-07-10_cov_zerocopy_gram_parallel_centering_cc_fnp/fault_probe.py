import sys, resource, time
import numpy as np
sys.path.insert(0, sys.argv[2])
import fnp_python as fnp
rng = np.random.default_rng(0)
X = rng.standard_normal((2000,50))
fnp.cov(X)
f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
t0 = time.perf_counter()
for _ in range(20): fnp.cov(X)
dt = (time.perf_counter()-t0)/20*1e3
f1 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
print(f"{sys.argv[1]}: {dt:.2f} ms/call, {(f1-f0)//20} minor faults/call")
