import time
import numpy as np

rng = np.random.default_rng(20260714)
a = rng.standard_normal((2896, 2896))

def t(fn, reps=5):
    fn(); best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1000

print(f"einsum ij->i : {t(lambda: np.einsum('ij->i', a)):8.2f} ms")
print(f"sum axis=1   : {t(lambda: a.sum(axis=1)):8.2f} ms")
print(f"add.reduce a1: {t(lambda: np.add.reduce(a, axis=1)):8.2f} ms")
print(f"einsum ij->  : {t(lambda: np.einsum('ij->', a)):8.2f} ms")
print(f"einsum ij->j : {t(lambda: np.einsum('ij->j', a)):8.2f} ms")

# contract candidates for one row
def scalar_tree(av):
    acc = 0.0; i = 0; n = len(av)
    while n - i > 4:
        acc += (av[i] + av[i+1]) + (av[i+2] + av[i+3]); i += 4
    while i < n:
        acc += av[i]; i += 1
    return acc

def lane_tree(av, L):
    # npyv: v_accum = sum of 4-blocks (a01=(a0+a1),a23=(a2+a3),v+=(a01+a23)) per 4*L,
    # then tillz remainder lanes, then npyv_sum (sequential lane fold assumed)
    n = len(av)
    v = np.zeros(L)
    i = 0
    while n - i >= 4 * L:
        a0 = av[i:i+L]; a1 = av[i+L:i+2*L]; a2 = av[i+2*L:i+3*L]; a3 = av[i+3*L:i+4*L]
        v = ((a0 + a1) + (a2 + a3)) + v
        i += 4 * L
    while i < n:
        chunk = np.zeros(L)
        take = min(L, n - i)
        chunk[:take] = av[i:i+take]
        v = chunk + v
        i += take
    acc = 0.0
    for x in v:
        acc += x
    return acc

row = a[0]
want = np.einsum('ij->i', a)[0]
print(f"want            {want!r}")
print(f"scalar_tree     {scalar_tree(row)!r}  match={scalar_tree(row) == want}")
for L in (2, 4, 8):
    lt = lane_tree(row, L)
    print(f"lane_tree L={L}   {lt!r}  match={lt == want}")
print(f"numpy={np.__version__}")
