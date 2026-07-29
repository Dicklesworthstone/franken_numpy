import warnings
import numpy as np
a = np.full(2_000_000, np.float16(60000)); b = a.copy()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.einsum('j,j->j', a, b)
print(f"elemwise_1d: {[str(x.message) for x in w]}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.einsum('ij,ij->ij', a.reshape(2000, 1000), b.reshape(2000, 1000))
print(f"elemwise_2d: {[str(x.message) for x in w]}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.einsum('j,j->', a, b)
print(f"dot: {[str(x.message) for x in w]}")
sm = np.full(100, np.float16(60000))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.einsum('j,j->j', sm, sm)
print(f"elemwise_small: {[str(x.message) for x in w]}")
print(f"numpy={np.__version__}")
