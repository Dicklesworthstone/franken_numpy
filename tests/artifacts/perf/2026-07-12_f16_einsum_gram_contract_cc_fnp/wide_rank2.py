import time
import numpy as np

rng = np.random.default_rng(20260716)
def t(fn, reps=5):
    fn(); best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1000

f16 = np.float16
v16 = (rng.standard_normal(8_000_000) * 0.3).astype(f16)
v64 = rng.standard_normal(8_000_000)
sorted64 = np.sort(rng.standard_normal(1_000_000))
q64 = rng.standard_normal(1_000_000)
m64 = rng.standard_normal((2896, 2896))
bins = np.linspace(-3, 3, 101)
rows = [
    ("f16 std 8M", lambda: np.std(v16)),
    ("f16 var 8M", lambda: np.var(v16)),
    ("f16 prod 8M", lambda: np.prod(v16)),
    ("f16 clip 8M", lambda: np.clip(v16, -0.5, 0.5)),
    ("f16 round 8M", lambda: np.round(v16, 2)),
    ("f16 diff 8M", lambda: np.diff(v16)),
    ("f16 unique 8M", lambda: np.unique(v16)),
    ("f16 nanargmax 8M", lambda: np.nanargmax(v16)),
    ("f16 maximum.reduce 8M", lambda: np.maximum.reduce(v16)),
    ("f16 searchsorted 1M into 1M", lambda: np.searchsorted(np.sort(v16[:1_000_000]), v16[1_000_000:2_000_000])),
    ("f64 percentile-25/50/75 8M", lambda: np.percentile(v64, [25, 50, 75])),
    ("f64 quantile9 8M", lambda: np.quantile(v64, np.linspace(0.1, 0.9, 9))),
    ("f64 digitize 1M/100bins", lambda: np.digitize(q64, bins)),
    ("f64 histogram2d 1M", lambda: np.histogram2d(q64, q64[::-1], bins=64)),
    ("f64 gradient 2d 2896^2", lambda: np.gradient(m64)),
    ("f64 isclose 8M", lambda: np.isclose(v64, v64)),
    ("f64 allclose 8M", lambda: np.allclose(v64, v64)),
    ("f64 tile 2896^2 x2", lambda: np.tile(m64[:1024], (2, 2))),
]
for name, fn in rows:
    try:
        print(f"{t(fn):10.2f} ms  {name}")
    except Exception as e:
        print(f"      ERR   {name}: {type(e).__name__}")
print(f"numpy={np.__version__}")
