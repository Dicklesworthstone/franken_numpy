# vfbef recon: numpy optimize=True 3-op f16 chain = einsum_path plan +
# sequential pairwise c_einsum with f16 intermediates?
import numpy as np

def pairstep(a, b):  # the verified 'ij,jk->ik' per-step-narrow contract
    m, k = a.shape; k2, n = b.shape
    af, bf = a.astype(np.float32), b.astype(np.float32)
    out = np.zeros((m, n), dtype=np.float16)
    for kk in range(k):
        out = (out.astype(np.float32) + np.outer(af[:, kk], bf[kk, :])).astype(np.float16)
    return out

rng = np.random.default_rng(20260715)
fails = total = 0
for (m, k1, k2, n) in [(8, 9, 10, 11), (16, 16, 16, 16), (5, 30, 4, 7), (24, 8, 40, 6)]:
    a = (rng.standard_normal((m, k1)) * 0.3).astype(np.float16)
    b = (rng.standard_normal((k1, k2)) * 0.3).astype(np.float16)
    c = (rng.standard_normal((k2, n)) * 0.3).astype(np.float16)
    path, info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize=True)
    want = np.einsum('ij,jk,kl->il', a, b, c, optimize=True)
    # candidate plans
    left = pairstep(pairstep(a, b), c)   # (ab)c
    right = pairstep(a, pairstep(b, c))  # a(bc)
    lm = want.tobytes() == left.tobytes()
    rm = want.tobytes() == right.tobytes()
    total += 1
    if not (lm or rm):
        fails += 1
    print(f"shapes=({m},{k1},{k2},{n}) path={path} left(ab)c={lm} right a(bc)={rm}")
# intermediate dtype check
a = (rng.standard_normal((8, 9)) * 0.3).astype(np.float16)
b = (rng.standard_normal((9, 10)) * 0.3).astype(np.float16)
c = (rng.standard_normal((10, 11)) * 0.3).astype(np.float16)
w_opt = np.einsum('ij,jk,kl->il', a, b, c, optimize=True)
manual_f16 = np.einsum('kl,lm->km'.replace('k','i').replace('l','j').replace('m','k'), a, b)  # just np pairwise
step1 = np.einsum('ij,jk->ik', a, b)
manual = np.einsum('ik,kl->il', step1, c)
print(f"pairwise-via-np-einsum == optimize=True: {manual.tobytes() == w_opt.tobytes()} (step1 dtype={step1.dtype})")
print(f"numpy={np.__version__} fails={fails}/{total}")
