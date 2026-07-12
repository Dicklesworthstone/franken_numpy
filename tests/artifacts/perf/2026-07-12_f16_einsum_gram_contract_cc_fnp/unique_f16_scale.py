import numpy as np
rng = np.random.default_rng(20260716)
# 1M array, many +0.0 with ONE -0.0 planted late; many nan(0x7e00) with one 0xfe00 late
for trial in range(3):
    a = (rng.standard_normal(1_000_000) * 2).astype(np.float16)
    a[a == 0] = np.float16(1.0)
    idx = rng.integers(0, 1_000_000, 5000)
    a[idx] = np.float16(0.0)          # many +0
    a[999_999] = np.float16(-0.0)     # one -0 at the END
    nan_idx = rng.integers(0, 999_998, 3000)
    a[nan_idx] = np.float16(np.nan)   # many 0x7e00
    a[999_998] = np.uint16(0xfe00).view(np.float16)  # one -nan near end
    u = np.unique(a)
    zbits = hex(u.view(np.uint16)[np.where(u == 0)[0][0]])
    nbits = hex(u.view(np.uint16)[-1])
    print(f"trial{trial}: zero={zbits} nan={nbits}")
# reversed: -0 flood with one late +0
a = (rng.standard_normal(1_000_000) * 2).astype(np.float16)
a[a == 0] = np.float16(1.0)
a[rng.integers(0, 1_000_000, 5000)] = np.float16(-0.0)
a[999_999] = np.float16(0.0)
u = np.unique(a)
print("flood-neg0:", hex(u.view(np.uint16)[np.where(u == 0)[0][0]]))
print(np.__version__)
