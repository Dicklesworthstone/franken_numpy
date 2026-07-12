# f64 isclose(a, b_array) contract: finite pair -> |a-b| <= atol + rtol*|b|;
# non-finite -> (a == b) [inf==inf True]; nan -> False unless equal_nan and both nan.
import warnings
import numpy as np

rng = np.random.default_rng(20260716)
fails = total = 0
def ref(a, b, rtol, atol, equal_nan):
    out = np.empty(a.shape, dtype=bool)
    fin = np.isfinite(a) & np.isfinite(b)
    out[fin] = np.abs(a[fin] - b[fin]) <= (atol + rtol * np.abs(b[fin]))
    nf = ~fin
    out[nf] = a[nf] == b[nf]
    if equal_nan:
        out[np.isnan(a) & np.isnan(b)] = True
    return out

for n in (10000, 100001):
    a = rng.standard_normal(n)
    b = a + rng.standard_normal(n) * 1e-7
    a[5] = np.inf; b[5] = np.inf
    a[6] = np.inf; b[6] = -np.inf
    a[7] = np.nan; b[7] = np.nan
    a[8] = np.nan; b[8] = 1.0
    a[9] = 1e300; b[9] = -1e300
    a[10] = 0.0; b[10] = -0.0
    for (rtol, atol, eqn) in ((1e-5, 1e-8, False), (1e-5, 1e-8, True), (1e-3, 0.0, False), (0.0, 1e-6, False)):
        total += 1
        with warnings.catch_warnings(record=True) as nw:
            warnings.simplefilter("always")
            want = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=eqn)
        got = ref(a, b, rtol, atol, eqn)
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"MISMATCH n={n} rtol={rtol} atol={atol} eqn={eqn}")
        if nw:
            print(f"WARNINGS from numpy isclose: {[str(w.message) for w in nw]}")
        total += 1
        want_all = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=eqn)
        if want_all != bool(got.all()):
            fails += 1; print(f"ALLCLOSE MISMATCH")
print(f"ISCLOSE_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
