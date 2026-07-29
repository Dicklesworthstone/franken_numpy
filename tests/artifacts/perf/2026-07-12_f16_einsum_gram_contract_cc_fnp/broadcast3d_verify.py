# 3-D broadcast no-contraction einsum: first/last-axis vec forms.
import numpy as np

rng = np.random.default_rng(20260714)
fails = total = 0
for dt in (np.float64, np.float16):
    p, q, r = 20, 30, 40
    full = (rng.standard_normal((p, q, r)) * 0.3).astype(dt)
    lastv = (rng.standard_normal(r) * 0.3).astype(dt)
    firstv = (rng.standard_normal(p) * 0.3).astype(dt)
    midv = (rng.standard_normal(q) * 0.3).astype(dt)
    full[0, 0, 0] = dt(-0.0); lastv[0] = dt(0.0); firstv[0] = dt(0.0); midv[0] = dt(0.0)
    if dt == np.float16:
        ref = lambda f, v: (np.float32(0.0) + f.astype(np.float32) * v.astype(np.float32)).astype(dt)
    else:
        ref = lambda f, v: (dt(0.0) + f * v).astype(dt)
    cases = [
        ('ijk,k->ijk', (full, lastv), ref(full, lastv[None, None, :])),
        ('ijk,i->ijk', (full, firstv), ref(full, firstv[:, None, None])),
        ('k,ijk->ijk', (lastv, full), ref(full, lastv[None, None, :])),
        ('i,ijk->ijk', (firstv, full), ref(full, firstv[:, None, None])),
        ('ijk,j->ijk', (full, midv), ref(full, midv[None, :, None])),
    ]
    for spec, ops, want_ref in cases:
        total += 1
        want = np.einsum(spec, *ops)
        if want.tobytes() != want_ref.tobytes():
            fails += 1; print(f"MISMATCH {dt.__name__} {spec}")
print(f"BCAST3D_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
