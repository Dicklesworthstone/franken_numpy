# Parity worker: loads fnp_python.so from sys.argv[1], runs a cov/corrcoef battery,
# prints one "case sha256 allclose" line per case. Run under old and new .so and
# diff the outputs: identical output = bit-identical results (the MR=4 bar).
import sys, hashlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
import fnp_python as fnp

rng = np.random.default_rng(42)

def report(name, r, ref):
    if np.isscalar(r) or getattr(r, "ndim", 1) == 0:
        b = np.asarray(r).tobytes()
    else:
        b = r.tobytes()
    h = hashlib.sha256(b).hexdigest()[:20]
    ok = np.allclose(r, ref, rtol=1e-12, atol=1e-14, equal_nan=True)
    print(f"{name:>42} {h} allclose={ok}")

# single-operand: (n_vars, n_obs) incl. n_obs<8 (scalar-tail-only), non-multiples
# of 8, non-multiples of MR=4, and the big bench shapes.
shapes = [(3, 7), (3, 100), (5, 503), (16, 1000), (31, 999), (33, 1001),
          (64, 1000), (127, 2000), (129, 3000), (500, 500), (1000, 1000),
          (2000, 500), (500, 5000)]
for nv, no in shapes:
    X = rng.standard_normal((nv, no))
    report(f"cov({nv}x{no})", fnp.cov(X), np.cov(X))
    report(f"cov({nv}x{no},ddof=0)", fnp.cov(X, ddof=0), np.cov(X, ddof=0))
    report(f"corrcoef({nv}x{no})", fnp.corrcoef(X), np.corrcoef(X))

# 1-D input (n_vars == 1 squeeze-to-scalar path)
x = rng.standard_normal(1000)
report("cov(1d)", fnp.cov(x), np.cov(x))
report("corrcoef(1d)", fnp.corrcoef(x), np.corrcoef(x))

# two-operand forms (m rows + y rows), incl. 1-D pairs and mixed row counts
a1 = rng.standard_normal(2000); b1 = rng.standard_normal(2000)
report("cov(1d,1d)", fnp.cov(a1, b1), np.cov(a1, b1))
report("corrcoef(1d,1d)", fnp.corrcoef(a1, b1), np.corrcoef(a1, b1))
A = rng.standard_normal((37, 1500)); B = rng.standard_normal((14, 1500))
report("cov(37x1500,14x1500)", fnp.cov(A, B), np.cov(A, B))
report("corrcoef(37x1500,14x1500)", fnp.corrcoef(A, B), np.corrcoef(A, B))
A2 = rng.standard_normal((900, 700)); B2 = rng.standard_normal((600, 700))
report("cov(900x700,600x700)", fnp.cov(A2, B2), np.cov(A2, B2))
report("corrcoef(900x700,600x700)", fnp.corrcoef(A2, B2), np.corrcoef(A2, B2))

# corrcoef zero-variance row (normalize defers -> fallback); NaN-tolerant compare
Z = rng.standard_normal((40, 5000)); Z[7] = 3.0
report("corrcoef(const-row 40x5000)", fnp.corrcoef(Z), np.corrcoef(Z))
# cov with a NaN in the data (values propagate, no defer semantics change)
N = rng.standard_normal((50, 5000)); N[3, 17] = np.nan
report("cov(nan 50x5000)", fnp.cov(N), np.cov(N))

# fortran-order (non-C-contiguous 2-D) must defer identically
F = np.asfortranarray(rng.standard_normal((64, 1000)))
report("cov(fortran 64x1000)", fnp.cov(F), np.cov(F))
