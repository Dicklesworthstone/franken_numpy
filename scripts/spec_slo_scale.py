#!/usr/bin/env python3
"""Spec section 17 SLO rows at SPEC SCALE, the fnp_python route with numpy side by side.

docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENNUMPY_V1.md section 17 sets absolute budgets for
100M-element workloads; G7 (run_performance_budget_gate) applies those numbers to 65k-element
Rust-engine workloads, where they cannot fail (bead deadlock-audit-rc0923-epic-71qy3.24). This
runs the rows at the spec's scale, on the Python surface users call, and prints one line per row:
fnp's p95 (or throughput), numpy's, the budget, and fnp's PASS/FAIL. Exit status = number of fnp
rows that miss their budget. Peak RSS is printed at the end. Needs ~4 GB of memory and writes a
1 GB .npy into a temporary directory.

    PYTHONPATH=<dir holding fnp_python.so> python3 scripts/spec_slo_scale.py [rounds]
"""
import os
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import fnp_python as fnp

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
failures = 0


def p95_ms(fn):
    fn()  # warm
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[min(len(times) - 1, round(0.95 * (len(times) - 1)))] * 1e3


def latency_row(label, budget_ms, make):
    global failures
    ours, theirs = p95_ms(make(fnp)), p95_ms(make(np))
    ok = ours <= budget_ms
    failures += not ok
    print(f"{label:46s} fnp p95 {ours:8.1f} ms | numpy {theirs:8.1f} ms | budget <= {budget_ms} ms "
          f"{'PASS' if ok else 'FAIL'}")


def throughput_row(label, budget_gbps, nbytes, make):
    global failures
    ours_ms, theirs_ms = p95_ms(make(fnp)), p95_ms(make(np))
    ours, theirs = nbytes / ours_ms / 1e6, nbytes / theirs_ms / 1e6
    ok = ours >= budget_gbps
    failures += not ok
    print(f"{label:46s} fnp {ours:6.2f} GB/s | numpy {theirs:6.2f} GB/s | budget >= {budget_gbps} GB/s "
          f"{'PASS' if ok else 'FAIL'} (p95 {ours_ms:.1f} / {theirs_ms:.1f} ms)")


def main():
    print(f"host={platform.node()} cpus={os.cpu_count()} loadavg={os.getloadavg()[0]:.1f} "
          f"numpy={np.__version__} python={platform.python_version()} rounds={ROUNDS}")
    rng = np.random.default_rng(0)
    big = rng.random((10_000, 10_000))  # 1e8 float64
    row, col = rng.random(10_000), rng.random((10_000, 1))
    latency_row("broadcast add 1e8 (+ row)", 180, lambda m: lambda: m.add(big, row))
    latency_row("broadcast multiply 1e8 (* column)", 180, lambda m: lambda: m.multiply(big, col))
    for name in ("sum", "mean"):
        for axis in (0, 1):
            latency_row(f"reduction {name} axis={axis} 1e8", 210,
                        lambda m, name=name, axis=axis: lambda: getattr(m, name)(big, axis=axis))

    small = rng.random((64, 64, 8))

    def transforms(m):
        def run():
            x = small
            for _ in range(10_000 // 4):
                x = m.swapaxes(m.reshape(m.transpose(m.reshape(x, (64, 512))), (512, 8, 8)), 0, 2)
                x = m.reshape(x, (64, 64, 8))
        return run
    latency_row("reshape/view 10k transforms", 40, transforms)

    throughput_row("dtype conversion float64->float32 1e8", 1.2, big.nbytes,
                   lambda m: lambda: m.asarray(big, dtype=np.float32))
    ints = rng.integers(0, 1 << 40, 100_000_000)
    throughput_row("dtype conversion int64->float64 1e8", 1.2, ints.nbytes,
                   lambda m: lambda: m.asarray(ints, dtype=np.float64))
    del ints, big

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spec_slo_1gb.npy")
        np.save(path, rng.random(125_000_000))  # 1 GB float64
        throughput_row("npy parse+load 1 GB", 0.4, os.path.getsize(path), lambda m: lambda: m.load(path))
    print(f"peak_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} "
          f"fnp_rows_failing={failures}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
