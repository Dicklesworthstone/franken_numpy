#!/usr/bin/env python3
"""Measure the fusion headroom available on NumPy's elementwise chain surface.

The cc/STRUCTURAL thesis from the 2026-07-25 campaign brief is:

    "The elementwise loss is a fusion problem, not a kernel problem: NumPy
     materializes every temporary too, so a fused multi-op traversal that never
     round-trips to memory should beat it outright."

That claim is only true if NumPy's chained form actually pays for the extra
memory round-trips. This measures the ceiling of that lever *using NumPy alone*,
so the answer does not depend on any FrankenNumPy build:

  chained : d = a*b + c            2 kernel passes, 1 full-size temporary
  out=    : multiply(a,b,out=t); add(t,c,out=t)   2 passes, temporary reused
  fused   : the unreachable ideal - modelled by a single-pass op of the same
            traffic shape (3 reads + 1 write), measured as np.add(a, b) scaled

Method follows campaign section 2: arms interleaved inside one round, order
alternating per round, statistic = median of per-round ratios, and an A/A null
control (same arm against itself) reported in the same invocation. Gate on the
null, never on cv.
"""
import gc
import statistics
import sys
import time

import numpy as np

ROUNDS = 41
MIN_OF = 3


def timed(fn, *args):
    """Minimum of MIN_OF inner replicates - the dominant knob per campaign 2.4."""
    best = float("inf")
    for _ in range(MIN_OF):
        t0 = time.perf_counter_ns()
        fn(*args)
        t1 = time.perf_counter_ns()
        best = min(best, t1 - t0)
    return best


def paired(arm_a, arm_b, rounds=ROUNDS):
    """Interleaved A/B inside one round, order alternating. Returns per-round ratios."""
    ratios, a_times, b_times = [], [], []
    for r in range(rounds):
        if r % 2 == 0:
            ta = timed(arm_a)
            tb = timed(arm_b)
        else:
            tb = timed(arm_b)
            ta = timed(arm_a)
        ratios.append(ta / tb)
        a_times.append(ta)
        b_times.append(tb)
    return ratios, a_times, b_times


def ci95(xs):
    """Bootstrap-free 95% interval on the median via order statistics."""
    s = sorted(xs)
    n = len(s)
    lo = s[max(0, int(0.025 * n))]
    hi = s[min(n - 1, int(0.975 * n))]
    return lo, hi


def cv(xs):
    m = statistics.fmean(xs)
    return (statistics.pstdev(xs) / m * 100.0) if m else float("nan")


def report(name, ratios, a_times, b_times):
    med = statistics.median(ratios)
    lo, hi = ci95(ratios)
    print(
        f"  {name:34s} median={med:6.4f}x  ci95=[{lo:6.4f},{hi:6.4f}]  "
        f"cvA={cv(a_times):5.2f}%  cvB={cv(b_times):5.2f}%"
    )
    return med, lo, hi


def run(n):
    rng = np.random.default_rng(12345)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    t = np.empty_like(a)

    def chained():
        return a * b + c

    def with_out():
        np.multiply(a, b, out=t)
        np.add(t, c, out=t)
        return t

    def single_pass_proxy():
        # One kernel pass over 2 reads + 1 write. The fused ideal for a*b+c is
        # 3 reads + 1 write, so this proxy is optimistic - it bounds the lever.
        return np.add(a, b)

    # Correctness first: the two real arms must agree bit-for-bit.
    r1 = chained()
    r2 = with_out()
    assert r1.tobytes() == r2.tobytes(), "chained and out= arms diverge"

    gc.collect()
    bytes_touched = n * 8
    print(f"\nN = {n:>10,}  ({bytes_touched/2**20:.1f} MiB per array)")

    # A/A null control, same invocation, same arm.
    report("NULL  chained / chained", *paired(chained, chained))
    report("NULL  out= / out=", *paired(with_out, with_out))
    # The actual questions.
    m_out, _, _ = report("chained / out=", *paired(chained, with_out))
    m_sp, _, _ = report("chained / single-pass proxy", *paired(chained, single_pass_proxy))
    return m_out, m_sp


if __name__ == "__main__":
    print(f"numpy {np.__version__}  python {sys.version.split()[0]}")
    print(f"rounds={ROUNDS} min_of={MIN_OF}")
    for n in (1 << 20, 1 << 23, 1 << 25):
        run(n)
