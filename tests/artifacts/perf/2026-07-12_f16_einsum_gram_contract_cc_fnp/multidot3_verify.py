# f16 multi_dot 3-array: numpy = _multi_dot_three cost rule + dot pairs.
# cost1 = p0*p1*p2 + p0*p2*p3  ((AB)C) ; cost2 = p1*p2*p3 + p0*p1*p3 (A(BC)).
import numpy as np

rng = np.random.default_rng(20260715)
fails = total = 0
for (p0, p1, p2, p3) in [(8, 9, 10, 11), (64, 64, 64, 64), (5, 300, 4, 7), (24, 8, 400, 6), (128, 32, 32, 128)]:
    a = (rng.standard_normal((p0, p1)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((p1, p2)) * 0.3).astype(np.float16)
    c = (rng.standard_normal((p2, p3)) * 0.3).astype(np.float16)
    want = np.linalg.multi_dot([a, b, c])
    cost1 = p0 * p1 * p2 + p0 * p2 * p3
    cost2 = p1 * p2 * p3 + p0 * p1 * p3
    got = (a @ b) @ c if cost1 < cost2 else a @ (b @ c)
    total += 1
    if want.tobytes() != got.tobytes() or want.dtype != got.dtype:
        fails += 1; print(f"MISMATCH {p0},{p1},{p2},{p3} cost1={cost1} cost2={cost2}")
print(f"MULTIDOT3_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
