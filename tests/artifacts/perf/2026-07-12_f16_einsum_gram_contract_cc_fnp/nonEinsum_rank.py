import time
import numpy as np

rng = np.random.default_rng(20260715)
def t(fn, reps=5):
    fn(); best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1000

f16 = np.float16
v16 = (rng.standard_normal(8_000_000) * 0.3).astype(f16)
w16 = (rng.standard_normal(8_000_000) * 0.3).astype(f16)
m16 = (rng.standard_normal((512, 512)) * 0.3).astype(f16)
n16a = (rng.standard_normal((256, 256)) * 0.3).astype(f16)
n16b = (rng.standard_normal((256, 256)) * 0.3).astype(f16)
n16c = (rng.standard_normal((256, 256)) * 0.3).astype(f16)
v64 = rng.standard_normal(8_000_000)
mask = v64 > 0.5
v64n = v64.copy(); v64n[::7] = np.nan
rows = [
    ("f16 vdot 8M", lambda: np.vdot(v16, w16)),
    ("f16 inner 8M", lambda: np.inner(v16, w16)),
    ("f16 dot 1D 8M", lambda: np.dot(v16, w16)),
    ("f16 cumsum 8M", lambda: np.cumsum(v16)),
    ("f16 multi_dot 3x256", lambda: np.linalg.multi_dot([n16a, n16b, n16c])),
    ("f16 outer 2896x2896", lambda: np.outer(v16[:2896], w16[:2896])),
    ("f16 trace 512^2", lambda: np.trace(m16)),
    ("f16 sort 8M", lambda: np.sort(v16)),
    ("f64 nansum 8M", lambda: np.nansum(v64n)),
    ("f64 nanmean 8M", lambda: np.nanmean(v64n)),
    ("f64 average weighted 8M", lambda: np.average(v64, weights=np.abs(v64))),
    ("f64 median 8M", lambda: np.median(v64)),
    ("f64 ptp 8M", lambda: np.ptp(v64)),
    ("f64 compress 8M", lambda: np.compress(mask, v64)),
    ("f16 nansum 8M", lambda: np.nansum(v16)),
    ("f16 mean 8M", lambda: np.mean(v16)),
]
for name, fn in rows:
    try:
        print(f"{t(fn):10.2f} ms  {name}")
    except Exception as e:
        print(f"      ERR   {name}: {type(e).__name__}")
print(f"numpy={np.__version__}")
