# f16 einsum elementwise ('j,j->j', 'ij,ij->ij'): hypothesis - per element
# out = f16(f32(a)*f32(b)), independent per element (chunk/order-free).
import numpy as np

rng = np.random.default_rng(20260713)
fails = total = 0
for shape, spec in [((100,), 'j,j->j'), ((8193,), 'j,j->j'), ((100000,), 'j,j->j'),
                    ((64, 100), 'ij,ij->ij'), ((3, 2731), 'ij,ij->ij'), ((300, 400), 'ij,ij->ij')]:
    for scale in (0.3, 300.0):
        total += 1
        a = (rng.standard_normal(shape) * scale).astype(np.float16)
        b = (rng.standard_normal(shape) * scale).astype(np.float16)
        want = np.einsum(spec, a, b)
        got = (np.float32(0.0) + a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
        if want.tobytes() != got.tobytes():
            fails += 1; print(f"MISMATCH {spec} {shape} scale={scale}")
# specials: inf/nan/overflow-to-inf
a = np.array([np.inf, -np.inf, np.nan, 60000, -60000, 0.0, -0.0], dtype=np.float16)
b = np.array([2.0, np.nan, 3.0, 60000, 60000, -0.0, -0.0], dtype=np.float16)
total += 1
want = np.einsum('j,j->j', a, b)
got = (np.float32(0.0) + a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
if want.tobytes() != got.tobytes():
    fails += 1; print("MISMATCH specials")
print(f"ELEMENTWISE_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
