# Broadcast no-contraction einsum ('ij,j->ij', 'ij,i->ij', swapped operand
# orders): hypothesis - zero-seeded product per element in the input dtype:
#   out[i,j] = dt(0 + full[i,j] * vec[axis_idx])
# (multiply commutative -> operand order irrelevant to bytes).
import numpy as np

rng = np.random.default_rng(20260714)
fails = total = 0
for dt in (np.float64, np.float32, np.float16):
    m, n = 300, 400
    full = (rng.standard_normal((m, n)) * 0.3).astype(dt)
    row = (rng.standard_normal(n) * 0.3).astype(dt)
    col = (rng.standard_normal(m) * 0.3).astype(dt)
    # plant signed zeros
    full[0, 0] = dt(-0.0); row[0] = dt(0.0)
    full[1, 1] = dt(0.0); row[1] = dt(-0.0)
    if dt == np.float16:
        ref = lambda f, v: (np.float32(0.0) + f.astype(np.float32) * v.astype(np.float32)).astype(dt)
    else:
        ref = lambda f, v: (dt(0.0) + f * v).astype(dt)
    cases = [
        ('ij,j->ij', (full, row), ref(full, row[None, :])),
        ('ij,i->ij', (full, col), ref(full, col[:, None])),
        ('j,ij->ij', (row, full), ref(full, row[None, :])),
        ('i,ij->ij', (col, full), ref(full, col[:, None])),
    ]
    for spec, ops, want_ref in cases:
        total += 1
        want = np.einsum(spec, *ops)
        if want.tobytes() != want_ref.tobytes():
            fails += 1; print(f"MISMATCH {dt.__name__} {spec}")
print(f"BROADCAST_EW_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
