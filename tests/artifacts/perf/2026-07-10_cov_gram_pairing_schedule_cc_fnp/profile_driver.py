# Profile driver: loops fnp.cov / fnp.corrcoef on one shape so perf record can sample
# the steady state. argv: <so_dir> <n_vars> <n_obs> <seconds> [corrcoef]
import sys, time, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
import fnp_python as fnp

nv, no, secs = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
fn = fnp.corrcoef if (len(sys.argv) > 5 and sys.argv[5] == "corrcoef") else fnp.cov
rng = np.random.default_rng(0)
X = rng.standard_normal((nv, no))
fn(X)  # warm
t0 = time.perf_counter()
calls = 0
while time.perf_counter() - t0 < secs:
    fn(X)
    calls += 1
print(f"{calls} calls in {time.perf_counter() - t0:.2f}s")
