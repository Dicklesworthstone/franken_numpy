# f16 searchsorted via a 65536-entry cumulative table over the order bijection
# key = bits^0x8000 if sign clear else !bits (nan-free haystack only).
# NaN query rule (bisect with all-false compares): left -> 0, right -> n.
import numpy as np

def key(bits):
    bits = int(bits)
    return (bits | 0x8000) if bits < 0x8000 else (0xFFFF - bits)

rng = np.random.default_rng(20260716)
fails = total = 0
for hn, qn in ((1000, 500), (100000, 1000)):
    hay = np.sort((rng.standard_normal(hn) * 2).astype(np.float16))
    q = (rng.standard_normal(qn) * 3).astype(np.float16)
    q[0] = np.float16(np.nan); q[1] = np.float16(-np.nan)
    q[2] = np.float16(0.0); q[3] = np.float16(-0.0)
    q[4] = np.float16(np.inf); q[5] = np.float16(-np.inf)
    q[6] = hay[0]; q[7] = hay[-1]
    # cumulative counts per key
    counts = np.zeros(65537, dtype=np.int64)
    for b in hay.view(np.uint16):
        counts[key(b) + 1] += 1
    cum = np.cumsum(counts)
    for side in ('left', 'right'):
        want = np.searchsorted(hay, q, side=side)
        got = np.empty(qn, dtype=np.int64)
        for i, v in enumerate(q.view(np.uint16)):
            fv = q[i]
            if np.isnan(fv):
                got[i] = hn
            else:
                k = key(v)
                got[i] = cum[k] if side == 'left' else cum[k + 1]
        total += 1
        if want.tobytes() != got.astype(want.dtype).tobytes():
            fails += 1
            bad = np.nonzero(want != got)[0][:4]
            print(f"MISMATCH {side} hn={hn}: idx {bad} q={q[bad]} want={want[bad]} got={got[bad]}")
print(f"SEARCHSORTED_VERIFY numpy={np.__version__} total={total} fails={fails}")
print("VERIFY_OK" if fails == 0 else "VERIFY_FAIL")
