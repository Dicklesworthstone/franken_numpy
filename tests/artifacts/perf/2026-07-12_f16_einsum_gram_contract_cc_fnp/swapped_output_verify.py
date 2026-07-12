# Output-transposed einsum specs as operand swaps of the two verified contracts:
# 'ij,lj->li' : out[l,i] = wide blocked-4 tree over j of f32(a[i,j])*f32(b[l,j])
#               == transposed_kernel(b, a) elementwise (multiply commuted).
# 'ji,jl->li' : out[l,i] = per-step chain acc=f16(f32(a[j,i])*f32(b[j,l])+f32(acc))
#               == gram_kernel(b, a) elementwise (multiply commuted).
import numpy as np

rng = np.random.default_rng(20260712)
fails = total = 0

def wide_tree(av, bv):  # blocked-4 left-assoc tree, one final narrow
    k = av.shape[0]; acc = np.float32(0.0)
    j = 0
    while j + 4 <= k:
        acc = acc + (((av[j]*bv[j] + av[j+1]*bv[j+1]) + av[j+2]*bv[j+2]) + av[j+3]*bv[j+3])
        j += 4
    while j < k:
        acc = acc + av[j]*bv[j]; j += 1
    return np.float16(np.float32(0.0) + acc)

for k, m, n in [(7, 5, 4), (96, 9, 6), (513, 3, 5), (2001, 2, 3)]:
    for scale in (1.0, 100.0):
        a = (rng.standard_normal((m, k)) * scale).astype(np.float16)
        b = (rng.standard_normal((n, k)) * scale).astype(np.float16)
        want = np.einsum('ij,lj->li', a, b)
        af, bf = a.astype(np.float32), b.astype(np.float32)
        got = np.empty((n, m), dtype=np.float16)
        for l in range(n):
            for i in range(m):
                got[l, i] = wide_tree(bf[l], af[i])  # swapped-operand order
        total += 1; fails += want.tobytes() != got.tobytes()
    for scale in (1.0, 100.0):
        a = (rng.standard_normal((k, m)) * scale).astype(np.float16)
        b = (rng.standard_normal((k, n)) * scale).astype(np.float16)
        want = np.einsum('ji,jl->li', a, b)
        af, bf = a.astype(np.float32), b.astype(np.float32)
        acc = np.zeros((n, m), dtype=np.float16)
        for j in range(k):  # swapped-operand muladd rows: out[l,i] += b[j,l]*a[j,i]
            acc = (bf[j, :, None] * af[j, None, :] + acc.astype(np.float32)).astype(np.float16)
        total += 1; fails += want.tobytes() != acc.tobytes()
print(f"SWAPPED_OUTPUT_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
