# AVX-512 byte-relation probe for the SHIPPED 15-op f64 transcendental set
# (e54e3195) + sqrt: numpy vs system libm per host. On AVX2 workers the whole
# set was proven byte-equal through the wired route; numpy vectorizes f64
# sin/cos (and friends) only on AVX512-class hosts, where the relation may
# differ - same class as the exp/log family divergence that gated gkznn.
import math
import platform
import socket

import numpy as np

cpu = getattr(np._core._multiarray_umath, '__cpu_features__', {})
rng = np.random.default_rng(20260711)
base = rng.standard_normal(200_000)
unit = rng.uniform(-0.999, 0.999, 200_000)
geq1 = 1.0 + np.abs(rng.standard_normal(200_000))
pos = np.abs(rng.standard_normal(200_000)) + 0.5
ops = [
    ("sin", math.sin, base), ("cos", math.cos, base), ("tan", math.tan, base),
    ("arctan", math.atan, base), ("arcsinh", math.asinh, base),
    ("tanh", math.tanh, base), ("cbrt", math.cbrt, base),
    ("expm1", math.expm1, base), ("log1p", math.log1p, pos),
    ("sinh", math.sinh, base), ("cosh", math.cosh, base),
    ("arcsin", math.asin, unit), ("arccos", math.acos, unit),
    ("arctanh", math.atanh, unit), ("arccosh", math.acosh, geq1),
    ("sqrt", math.sqrt, pos),
]
host = socket.gethostname()
for name, ref_fn, data in ops:
    np_out = getattr(np, name)(data)
    ref = np.array([ref_fn(float(v)) for v in data])
    diff = int(np.count_nonzero(np_out.view(np.int64) != ref.view(np.int64)))
    line = (f"TRANS_HOST_PROBE host={host} numpy={np.__version__} "
            f"avx512f={bool(cpu.get('AVX512F', False))} op={name} "
            f"byte_equal={diff == 0} diff_elems={diff}")
    if diff:
        bitdiff = np.abs(np_out.view(np.int64) - ref.view(np.int64))
        line += f" max_bitdiff={int(bitdiff.max())} frac={diff / len(data):.6f}"
    print(line)
print(f"TRANS_HOST_PROBE_ENV glibc={platform.libc_ver()[1]} python={platform.python_version()}")
print("PROBE_OK")
