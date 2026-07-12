# Contract verify for f16 einsum 'j,j->': per-8192-chunk blocked-4 f32 tree,
# folded through f16 store/reload: out = f16(f32(out) + tree(chunk)), out0=f16(0).
import numpy as np

def tree_f32(av, bv):
    k = av.shape[0]
    acc = np.float32(0.0)
    j = 0
    while j + 4 <= k:
        ab = av[j:j+4] * bv[j:j+4]
        acc = acc + (((ab[0] + ab[1]) + ab[2]) + ab[3])
        j += 4
    while j < k:
        acc = acc + av[j] * bv[j]
        j += 1
    return acc

rng = np.random.default_rng(20260712)
CH = 8192
fails = total = 0
for k in (5, 100, 8191, 8192, 8193, 8196, 16384, 16389, 24576, 100_000, 500_003):
    for scale in (1.0, 30.0):
        a = (rng.standard_normal(k) * scale).astype(np.float16)
        b = (rng.standard_normal(k) * scale).astype(np.float16)
        want = np.einsum('j,j->', a, b)
        af, bf = a.astype(np.float32), b.astype(np.float32)
        out = np.float16(0.0)
        for c0 in range(0, k, CH):
            out = np.float16(np.float32(out) + tree_f32(af[c0:c0+CH], bf[c0:c0+CH]))
        total += 1
        if np.float16(want).tobytes() != out.tobytes():
            fails += 1
            if fails <= 3:
                print(f"MISMATCH k={k} scale={scale} want={want!r} got={out!r}")
print(f"return_type={type(np.einsum('j,j->', a, b)).__name__} ndim={np.ndim(np.einsum('j,j->', a, b))}")
print(f"DOT1D_CONTRACT_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
