# exp2 sibling of the gkznn probe (same seed/domain class): numpy f64 exp2 vs
# system libm exp2 (math.exp2 = CPython's direct libm call), byte relation per host.
import math
import platform
import socket

import numpy as np

cpu = getattr(np._core._multiarray_umath, '__cpu_features__', {})
rng = np.random.default_rng(20260711)
e = rng.standard_normal(200_000)
np_out = np.exp2(e)
ref = np.array([math.exp2(float(v)) for v in e])
diff = int(np.count_nonzero(np_out.view(np.int64) != ref.view(np.int64)))
print(f"EXP2_HOST_PROBE host={socket.gethostname()} numpy={np.__version__} "
      f"avx512f={bool(cpu.get('AVX512F', False))} avx512_skx={bool(cpu.get('AVX512_SKX', False))} "
      f"op=exp2 byte_equal={diff == 0} diff_elems={diff}")
if diff:
    bitdiff = np.abs(np_out.view(np.int64) - ref.view(np.int64))
    idx = int(np.argmax(bitdiff))
    print(f"EXP2_HOST_PROBE_DETAIL max_bitdiff={int(bitdiff.max())} "
          f"frac_diff={diff / len(e):.6f} example_x={e[idx]!r} "
          f"np={np_out[idx]!r} libm={ref[idx]!r}")
print(f"EXP2_HOST_PROBE_ENV glibc={platform.libc_ver()[1]} python={platform.python_version()}")
print("PROBE_OK")
