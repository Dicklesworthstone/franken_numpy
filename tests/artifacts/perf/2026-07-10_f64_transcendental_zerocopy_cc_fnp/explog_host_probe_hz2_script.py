# Verbatim probe body from conformance_exp_log::f64_exp_log_numpy_vs_system_libm_byte_probe
# (bead deadlock-audit-gkznn). The in-tree test loads fnp_python but the probe body does
# not use it; running the body directly on the host yields the identical data point.
import math
import platform
import socket

import numpy as np

cpu = getattr(np._core._multiarray_umath, '__cpu_features__', {})
rng = np.random.default_rng(20260711)
e = rng.standard_normal(200_000)
l = np.abs(rng.standard_normal(200_000)) + 0.5
for name, ref_fn, data in [
    ("exp", math.exp, e), ("log", math.log, l),
    ("log2", math.log2, l), ("log10", math.log10, l),
]:
    np_out = getattr(np, name)(data)
    ref = np.array([ref_fn(float(v)) for v in data])
    diff = int(np.count_nonzero(np_out.view(np.int64) != ref.view(np.int64)))
    print(f"EXPLOG_HOST_PROBE host={socket.gethostname()} numpy={np.__version__} "
          f"avx512f={bool(cpu.get('AVX512F', False))} avx512_skx={bool(cpu.get('AVX512_SKX', False))} "
          f"op={name} byte_equal={diff == 0} diff_elems={diff}")
    if diff:
        bitdiff = np.abs(np_out.view(np.int64) - ref.view(np.int64))
        idx = int(np.argmax(bitdiff))
        print(f"EXPLOG_HOST_PROBE_DETAIL op={name} max_bitdiff={int(bitdiff.max())} "
              f"frac_diff={diff / len(data):.6f} example_x={data[idx]!r} "
              f"np={np_out[idx]!r} libm={ref[idx]!r}")
print(f"EXPLOG_HOST_PROBE_ENV glibc={platform.libc_ver()[1]} "
      f"machine={platform.machine()} python={platform.python_version()}")
print("PROBE_OK")
