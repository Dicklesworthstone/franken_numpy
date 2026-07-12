"""Verify the pinned wide-accumulate-once contract for f16 einsum transposed specs.

Contract from legacy_numpy_code einsum_sumprod.c.src half variant
(half_sum_of_products_contig_contig_outstride0_two, NPYV_CHK=0 scalar path):

    npy_float accum = 0;
    for (; count >= 4; count -= 4, ...) {
        ab0..ab3 = f32(a[i])*f32(b[i]);           # exact products
        accum += ((ab0 + ab1) + ab2) + ab3;       # left-assoc block tree
    }
    for (; count > 0; ...) accum += f32(a)*f32(b) # tail, one at a time
    out = f16(f32(out) + accum)                   # out zero-initialized

All arithmetic strictly float32.
"""
import numpy as np
import sys

f32 = np.float32


def contract_dot(a16_row, b16_row):
    """Blocked-4 f32 dot per the pinned contract. Returns f16 scalar."""
    a = a16_row.astype(np.float32)
    b = b16_row.astype(np.float32)
    k = a.shape[0]
    accum = f32(0.0)
    i = 0
    while k - i >= 4:
        ab0 = f32(a[i] * b[i])
        ab1 = f32(a[i + 1] * b[i + 1])
        ab2 = f32(a[i + 2] * b[i + 2])
        ab3 = f32(a[i + 3] * b[i + 3])
        accum = f32(accum + f32(f32(f32(ab0 + ab1) + ab2) + ab3))
        i += 4
    while i < k:
        accum = f32(accum + f32(a[i] * b[i]))
        i += 1
    return np.float16(f32(f32(0.0) + accum))


def main():
    rng = np.random.default_rng(20260711)
    print("numpy", np.__version__)

    # Phase 1: 'j,j->' family, the recon's 400-trial sweep (k=9..2000).
    # Same inner kernel; buffered path but single chunk for k<8192.
    mismatches = 0
    trials = 400
    for t in range(trials):
        k = int(rng.integers(9, 2001))
        a = (rng.standard_normal(k) * rng.choice([0.01, 1.0, 100.0])).astype(np.float16)
        b = (rng.standard_normal(k) * rng.choice([0.01, 1.0, 100.0])).astype(np.float16)
        want = np.einsum("j,j->", a, b)
        got = contract_dot(a, b)
        if want.tobytes() != got.tobytes():
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH t={t} k={k} want={want!r} got={got!r} "
                      f"bits {want.view(np.uint16)} vs {got.view(np.uint16)}")
    print(f"phase1 'j,j->': {trials - mismatches}/{trials} match")

    # Phase 1b: k=7 discriminating case from the recon, many trials.
    mm = 0
    for t in range(200):
        a = rng.standard_normal(7).astype(np.float16)
        b = rng.standard_normal(7).astype(np.float16)
        if np.einsum("j,j->", a, b).tobytes() != contract_dot(a, b).tobytes():
            mm += 1
    print(f"phase1b k=7: {200 - mm}/200 match")

    # Phase 2: the actual transposed matrix spec 'ij,lj->il', full matrices,
    # byte-compare whole outputs. Includes tails (k%4 != 0), scale mix, inf/nan.
    shapes = [(5, 7, 6), (16, 9, 16), (33, 130, 17), (64, 257, 48), (128, 96, 40),
              (1, 5, 3), (3, 4, 1), (7, 2000, 5)]
    ok = True
    for (m, k, n) in shapes:
        a = (rng.standard_normal((m, k)) * 4.0).astype(np.float16)
        b = (rng.standard_normal((n, k)) * 4.0).astype(np.float16)
        want = np.einsum("ij,lj->il", a, b)
        got = np.empty((m, n), dtype=np.float16)
        for i in range(m):
            for l in range(n):
                got[i, l] = contract_dot(a[i], b[l])
        same = want.tobytes() == got.tobytes()
        ndiff = int((want.view(np.uint16) != got.view(np.uint16)).sum())
        print(f"phase2 ij,lj->il ({m},{k})x({n},{k}): {'BYTE-IDENTICAL' if same else f'{ndiff} bytes differ'}")
        ok = ok and same

    # Phase 2b: special values — inf/nan propagation through the chain.
    a = np.zeros((2, 9), dtype=np.float16)
    b = np.zeros((3, 9), dtype=np.float16)
    a[0, 2] = np.float16(np.inf); a[0, 5] = np.float16(60000)
    a[1, 0] = np.float16(np.nan)
    b[0, 2] = np.float16(-2.0); b[1, 5] = np.float16(60000); b[2, 0] = np.float16(1.0)
    want = np.einsum("ij,lj->il", a, b)
    got = np.empty((2, 3), dtype=np.float16)
    for i in range(2):
        for l in range(3):
            got[i, l] = contract_dot(a[i], b[l])
    same = want.tobytes() == got.tobytes()
    print(f"phase2b inf/nan: {'BYTE-IDENTICAL' if same else 'DIFFER'} want={want.ravel()} got={got.ravel()}")
    ok = ok and same

    # Phase 3: alternate letter spellings of the same transposed class.
    a = (rng.standard_normal((12, 21)) * 2).astype(np.float16)
    b = (rng.standard_normal((10, 21)) * 2).astype(np.float16)
    for spec in ["ij,lj->il", "ab,cb->ac", "xy,zy->xz"]:
        w = np.einsum(spec, a, b)
        g = np.empty((12, 10), dtype=np.float16)
        for i in range(12):
            for l in range(10):
                g[i, l] = contract_dot(a[i], b[l])
        s = w.tobytes() == g.tobytes()
        print(f"phase3 {spec}: {'BYTE-IDENTICAL' if s else 'DIFFER'}")
        ok = ok and s

    if mismatches == 0 and mm == 0 and ok:
        print("CONTRACT CONFIRMED: blocked-4 left-assoc f32 tree + tail + f16(0+accum)")
        return 0
    print("CONTRACT NOT CONFIRMED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
