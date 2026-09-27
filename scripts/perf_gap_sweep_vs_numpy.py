#!/usr/bin/env python3
"""perf_gap_sweep_vs_numpy.py — reusable vs-NumPy perf-regression sweep for fnp-python.

Consolidates the ad-hoc diagnostics used during the 2026-06 BOLD-VERIFY sessions
(which found pinv 215x, eigvals/cov/corrcoef/convolve/view-op/stale-cliff wins)
into ONE tracked tool. Run it anytime after a build to catch NEW vs-NumPy losses
(e.g. the "perf size-gate goes stale after a NumPy/BLAS upgrade" class — det/inv/
solve/eigvalsh silently flipped from win to 2-6x loss when NumPy 2.4.3 removed an
OpenBLAS cliff). It does NOT build anything; point it at a built fnp_python.so.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH=.probe python3 scripts/perf_gap_sweep_vs_numpy.py [--full]
    PYTHONPATH=.probe python3 scripts/perf_gap_sweep_vs_numpy.py --surface [name ...]

Verdict: ratio = fnp/numpy.  <0.9 WIN | 0.9-1.4 ok | >1.4 LOSS (investigate).
Exit code = number of LOSS rows (0 = clean).

--surface is the whole-surface loss map: every numpy.__all__ callable fnp implements itself
(numpy's own re-exported objects are skipped), called with auto-probed arguments (1-D / 2-D,
float64 / int64, one or two operands, the first two shapes numpy accepts) at n = 4096 and 2^20,
fnp and numpy interleaved, printing every cell at >= 1.25x. Its 2026-09-27 run found percentile
90x, asarray(dtype=float64) 17x, trace 5x (NEGATIVE_EVIDENCE row of that date). Run it twice,
with RAYON_NUM_THREADS=1 and without: a cell that loses only on the full pool of a loaded host
is the contention class (bead deadlock-audit-vc4p4), not an algorithmic loss. The large size is
skipped wherever numpy's 4096-element time, extrapolated even linearly, passes 0.5 s - a
quadratic function (convolve, correlate) at 2^20 cannot be interrupted once it runs.

Measurement gotchas baked in (learned the hard way):
- median of N timed runs after warmup; set single-thread BLAS via env for stable A/B.
- binary ufuncs use DISTINCT arrays (same-array hits NumPy libm fast paths).
- cov/corrcoef use a FRESH default_rng each timed call (generator state, not perf).
- inputs are C-contiguous unless a strided case is explicitly intended.
- view-returning ops (transpose/rollaxis/ravel/...) must be checked for
  shares_memory(result,input)==True, NOT just speed — materializing a copy is the
  18000x/40000x bug class. (See --views.)
"""
import sys, time
import numpy as np

try:
    import fnp_python as f
except Exception as e:  # pragma: no cover
    print(f"cannot import fnp_python (set PYTHONPATH to the built .so dir): {e}")
    sys.exit(2)

FULL = "--full" in sys.argv
N = 4_000_000 if FULL else 1_000_000


def bench(fn, n=9):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


SURFACE_SKIP = ("save", "load", "txt", "file", "print", "memmap", "seterr", "setbuf", "getbuf",
                "show_", "info", "test", "genfrom", "fromregex", "frombuffer", "fromstring",
                "fromiter", "fromfunction", "errstate", "vectorize", "frompyfunc", "piecewise",
                "apply_", "nditer", "nested_iters", "ndindex", "ndenumerate", "busday",
                "datetime_", "einsum_path", "get_include", "typename", "mintypecode",
                "may_share", "shares_memory", "iterable", "broadcast", "require", "from_dlpack",
                "asmatrix", "bmat", "matrix", "set_", "get_")


def surface(only):
    import signal
    rng = np.random.default_rng(7)

    def operands(n):
        side = max(2, int(round(n ** 0.5)))
        a, b = rng.random(n), rng.random(n)
        ai, bi = rng.integers(0, 1000, n), rng.integers(0, 1000, n)
        A, B = rng.random((side, side)), rng.random((side, side))
        return [("f8", (a,)), ("f8,f8", (a, b)), ("i8", (ai,)), ("i8,i8", (ai, bi)),
                ("f8 2d", (A,)), ("f8 2d,2d", (A, B)), ("i8 2d", (rng.integers(0, 1000, (side, side)),)),
                ("f8,small", (a, rng.random(3))), ("f8,0.5", (a, 0.5)), ("f8 2d,ax0", (A, 0)),
                ("i8,small", (ai, np.array([1, 5, 9])))]

    def alarm(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, alarm)

    def once(fn, args, limit):
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            t = time.perf_counter(); fn(*args); return time.perf_counter() - t
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    names = [n for n in sorted(set(np.__all__))
             if (n in only if only else not any(w in n for w in SURFACE_SKIP))
             and callable(getattr(np, n, None)) and not isinstance(getattr(np, n), type)
             and getattr(f, n, None) is not None and getattr(f, n) is not getattr(np, n)]
    import warnings
    warnings.simplefilter("ignore")
    big, losses, cells = 1 << 20, 0, 0
    for name in names:
        npf, ff = getattr(np, name), getattr(f, name)
        small_time = {}
        for n in (4096, big):
            picked = 0
            for label, args in operands(n):
                if picked == 2:
                    break
                if n == big and (label not in small_time or small_time[label] * big / 4096 > 0.5):
                    continue
                try:
                    tn = once(npf, args, 3)
                except Exception:
                    continue
                small_time.setdefault(label, tn)
                picked += 1
                try:
                    once(ff, args, max(3, 50 * tn))
                except Exception as e:
                    print(f"{name:24} n={n:8d} {label:10} fnp {type(e).__name__} where numpy returns")
                    losses += 1
                    continue
                reps = max(1, int(0.004 / max(tn, 1e-7)))
                tf_all, tn_all = [], []
                for i in range(7):
                    for fn, acc in ((ff, tf_all), (npf, tn_all)) if i % 2 == 0 else ((npf, tn_all), (ff, tf_all)):
                        t = time.perf_counter()
                        for _ in range(reps):
                            fn(*args)
                        acc.append((time.perf_counter() - t) / reps)
                cells += 1
                r = sorted(tf_all)[3] / sorted(tn_all)[3]
                losses += r > 1.4
                if r >= 1.25:
                    print(f"{name:24} n={n:8d} {label:10} fnp/np {r:6.2f}  fnp {sorted(tf_all)[3]*1e6:10.1f}us"
                          f"  np {sorted(tn_all)[3]*1e6:10.1f}us  {'LOSS' if r > 1.4 else ''}")
    print(f"\nsurface: {len(names)} functions, {cells} cells, LOSS (> 1.4x) rows: {losses}")
    return losses


def main():
    if "--surface" in sys.argv:
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        sys.exit(min(surface(only), 125))
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N); y = rng.standard_normal(N)
    a2 = rng.standard_normal((1500, 1500)); b2 = rng.standard_normal((1500, 1500))
    spd = a2 @ a2.T + 1500 * np.eye(1500)
    av = rng.standard_normal(200000); bv = rng.standard_normal(200000)

    cases = []  # (name, numpy_fn, fnp_fn)
    add = lambda n, p, q: cases.append((n, p, q))

    # elementwise / reductions
    add("add", lambda: np.add(x, y), lambda: f.add(x, y))
    add("hypot", lambda: np.hypot(x, y), lambda: f.hypot(x, y))
    add("arctan2", lambda: np.arctan2(x, y), lambda: f.arctan2(x, y))
    add("atan2(alias)", lambda: np.arctan2(x, y), lambda: f.atan2(x, y))
    add("sum", lambda: np.sum(x), lambda: f.sum(x))
    add("std", lambda: np.std(x), lambda: f.std(x))
    add("median", lambda: np.median(x), lambda: f.median(x))
    add("cumsum", lambda: np.cumsum(x), lambda: f.cumsum(x))
    add("unique", lambda: np.unique(rng.integers(0, 1000, N)), lambda: f.unique(rng.integers(0, 1000, N)))
    # reductions vs cov/corrcoef two-operand (the wrapper-copy / gated-fast-path class)
    add("cov(a,b)", lambda: np.cov(av, bv), lambda: f.cov(av, bv))
    add("corrcoef(a,b)", lambda: np.corrcoef(av, bv), lambda: f.corrcoef(av, bv))
    # convolve/correlate short-kernel (zero-copy wrapper class)
    k = rng.standard_normal(16)
    add("convolve(same,k16)", lambda: np.convolve(x[:N], k, "same"), lambda: f.convolve(x[:N], k, "same"))
    add("correlate(valid,k16)", lambda: np.correlate(x[:N], k, "valid"), lambda: f.correlate(x[:N], k, "valid"))
    add("concat(alias)", lambda: np.concat([av, bv]), lambda: f.concat([av, bv]))
    # 2-D dense linalg (stale-cliff delegate class) — single 2-D should be ~parity (delegate)
    add("det", lambda: np.linalg.det(a2), lambda: f.det(a2))
    add("inv", lambda: np.linalg.inv(a2), lambda: f.inv(a2))
    add("slogdet", lambda: np.asarray(np.linalg.slogdet(a2)), lambda: np.asarray(f.slogdet(a2)))
    add("solve", lambda: np.linalg.solve(a2, b2[:, 0]), lambda: f.solve(a2, b2[:, 0]))
    add("svdvals", lambda: np.linalg.svdvals(a2), lambda: f.svdvals(a2))
    add("eigvalsh", lambda: np.linalg.eigvalsh(spd), lambda: f.eigvalsh(spd))
    add("eigh", lambda: np.linalg.eigh(spd)[0], lambda: np.asarray((f.eigh(spd))[0]))
    add("cholesky", lambda: np.linalg.cholesky(spd), lambda: f.cholesky(spd))
    add("matrix_power3", lambda: np.linalg.matrix_power(a2, 3), lambda: f.matrix_power(a2, 3))
    # batched linalg (parallel-across-lanes native WINS)
    A3 = rng.standard_normal((256, 16, 16))
    S3 = np.einsum("...ij,...kj->...ik", A3, A3) + 16 * np.eye(16)
    add("batch_inv", lambda: np.linalg.inv(A3), lambda: f.inv(A3))
    add("batch_eigvalsh", lambda: np.linalg.eigvalsh(S3), lambda: f.eigvalsh(S3))

    losses = 0
    print(f"{'op':22} {'np_us':>10} {'fnp_us':>10} {'fnp/np':>8}  verdict   (N={N})")
    for name, npf, ff in cases:
        try:
            tn = bench(npf); tf = bench(ff)
        except Exception as e:  # an op erroring is itself a regression signal
            print(f"{name:22} ERROR: {str(e)[:40]}")
            losses += 1
            continue
        r = tf / tn
        v = "WIN" if r < 0.9 else ("ok" if r <= 1.4 else "LOSS")
        if v == "LOSS":
            losses += 1
        print(f"{name:22} {tn*1e6:10.1f} {tf*1e6:10.1f} {r:8.3f}  {v}")

    # view-op aliasing contract (the 18000x/40000x materialization-bug class)
    print("\n-- view-op semantics (must share memory with input) --")
    Mv = rng.standard_normal((400, 600))
    for name, vf, base in [
        ("matrix_transpose", lambda: f.matrix_transpose(Mv), Mv),
        ("rollaxis", lambda: f.rollaxis(Mv.reshape(40, 10, 600), 2, 0), Mv),
        ("ravel", lambda: f.ravel(Mv), Mv),
        ("diagonal", lambda: f.diagonal(Mv), Mv),
    ]:
        try:
            r = np.asarray(vf())
            shares = np.shares_memory(r, base)
            print(f"{name:22} shares_memory={shares}  {'OK' if shares else 'MATERIALIZE-BUG'}")
            if not shares:
                losses += 1
        except Exception as e:
            print(f"{name:22} ERROR: {str(e)[:40]}")

    print(f"\nLOSS/regression rows: {losses}")
    sys.exit(min(losses, 125))


if __name__ == "__main__":
    main()
