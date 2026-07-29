# f64/f32 einsum elementwise ('X,X->X'): zero-seeded per-element product in
# the input dtype: out = dt(0 + a*b). Signed zeros: 0*-0 -> +0.0 via the seed.
import numpy as np

rng = np.random.default_rng(20260713)
fails = total = 0
for dt in (np.float64, np.float32):
    for shape, spec in [((100000,), 'j,j->j'), ((300, 400), 'ij,ij->ij'), ((8193,), 'j,j->j')]:
        for scale in (0.3, 1e150 if dt == np.float64 else 1e30):
            total += 1
            a = (rng.standard_normal(shape) * scale).astype(dt)
            b = (rng.standard_normal(shape) * scale).astype(dt)
            want = np.einsum(spec, a, b)
            got = (dt(0.0) + a * b).astype(dt)
            if want.tobytes() != got.tobytes():
                fails += 1; print(f"MISMATCH {dt.__name__} {spec} {shape} scale={scale}")
    # signed zeros + specials
    a = np.array([0.0, -0.0, -0.0, np.inf, np.nan, -2.0], dtype=dt)
    b = np.array([-0.0, -0.0, 0.0, np.nan, 3.0, 0.0], dtype=dt)
    total += 1
    want = np.einsum('j,j->j', a, b)
    got = (dt(0.0) + a * b).astype(dt)
    if want.tobytes() != got.tobytes():
        fails += 1; print(f"MISMATCH {dt.__name__} specials: einsum={want!r} seeded={got!r}")
print(f"F64F32_ELEMENTWISE_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
