#!/usr/bin/env python3
"""
FrankenNumPy vs NumPy Performance Comparison

Benchmark scenarios covering hot public operations.
Run with: python benchmarks/perf_comparison.py [--fnp-only|--numpy-only]
"""

import argparse
import gc
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

# Load fnp_python if available
FNP_MODULE = None
FNP_PATH = os.environ.get('FNP_MODULE_PATH')
if FNP_PATH and os.path.exists(FNP_PATH):
    spec = importlib.util.spec_from_file_location('fnp_python', FNP_PATH)
    FNP_MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FNP_MODULE)


@dataclass
class BenchmarkResult:
    name: str
    library: str
    times_ms: List[float]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    throughput: Optional[float] = None
    throughput_unit: str = "ops/sec"


def benchmark(fn: Callable, warmup: int = 3, runs: int = 20) -> List[float]:
    """Run benchmark with warmup, return times in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
        gc.collect()

    # Timed runs
    times = []
    for _ in range(runs):
        gc.collect()
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return times


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def run_benchmark(name: str, fn: Callable, library: str,
                  size: int = 0, runs: int = 20) -> BenchmarkResult:
    """Run a single benchmark and compute statistics."""
    times = benchmark(fn, runs=runs)
    p50 = percentile(times, 50)
    p95 = percentile(times, 95)
    p99 = percentile(times, 99)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0

    throughput = None
    if size > 0:
        throughput = size / (mean / 1000)  # elements/sec

    return BenchmarkResult(
        name=name,
        library=library,
        times_ms=times,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std,
        throughput=throughput,
        throughput_unit="elem/sec" if size > 0 else "ops/sec"
    )


# === BENCHMARK SCENARIOS ===

SCENARIOS = []

def scenario(name: str, size: int = 0):
    """Decorator to register a benchmark scenario."""
    def decorator(fn):
        SCENARIOS.append((name, fn, size))
        return fn
    return decorator


# 1. Array Creation
@scenario("array_creation_zeros_1M", size=1_000_000)
def bench_zeros(lib):
    return lambda: lib.zeros(1_000_000, dtype=lib.float64)

@scenario("array_creation_arange_1M", size=1_000_000)
def bench_arange(lib):
    return lambda: lib.arange(1_000_000, dtype=lib.float64)

# 2. Element-wise Math
@scenario("elementwise_add_1M", size=1_000_000)
def bench_add(lib):
    a = lib.arange(1_000_000, dtype=lib.float64)
    b = lib.arange(1_000_000, dtype=lib.float64)
    return lambda: lib.add(a, b)

@scenario("elementwise_multiply_1M", size=1_000_000)
def bench_multiply(lib):
    a = lib.arange(1_000_000, dtype=lib.float64)
    b = lib.arange(1_000_000, dtype=lib.float64)
    return lambda: lib.multiply(a, b)

@scenario("elementwise_exp_1M", size=1_000_000)
def bench_exp(lib):
    a = lib.linspace(0, 10, 1_000_000, dtype=lib.float64)
    return lambda: lib.exp(a)

# 3. Reductions
@scenario("reduction_sum_1M", size=1_000_000)
def bench_sum(lib):
    a = lib.arange(1_000_000, dtype=lib.float64)
    return lambda: lib.sum(a)

@scenario("reduction_mean_1M", size=1_000_000)
def bench_mean(lib):
    a = lib.arange(1_000_000, dtype=lib.float64)
    return lambda: lib.mean(a)

@scenario("reduction_std_1M", size=1_000_000)
def bench_std(lib):
    a = lib.arange(1_000_000, dtype=lib.float64)
    return lambda: lib.std(a)

# 4. Matrix Operations
@scenario("matmul_1000x1000", size=1_000_000)
def bench_matmul(lib):
    a = lib.random.rand(1000, 1000).astype(lib.float64) if hasattr(lib, 'random') else \
        lib.arange(1_000_000, dtype=lib.float64).reshape(1000, 1000)
    b = lib.random.rand(1000, 1000).astype(lib.float64) if hasattr(lib, 'random') else \
        lib.arange(1_000_000, dtype=lib.float64).reshape(1000, 1000)
    return lambda: lib.matmul(a, b)

@scenario("dot_10000", size=10_000)
def bench_dot(lib):
    a = lib.arange(10_000, dtype=lib.float64)
    b = lib.arange(10_000, dtype=lib.float64)
    return lambda: lib.dot(a, b)

# 5. Linear Algebra
@scenario("linalg_det_500x500", size=250_000)
def bench_det(lib):
    a = lib.arange(250_000, dtype=lib.float64).reshape(500, 500)
    a = a + lib.eye(500) * 1000  # Make well-conditioned
    return lambda: lib.linalg.det(a)

@scenario("linalg_inv_200x200", size=40_000)
def bench_inv(lib):
    a = lib.arange(40_000, dtype=lib.float64).reshape(200, 200)
    a = a + lib.eye(200) * 1000  # Make invertible
    return lambda: lib.linalg.inv(a)

# 6. FFT
@scenario("fft_1M", size=1_048_576)
def bench_fft(lib):
    a = lib.arange(1_048_576, dtype=lib.float64)  # Power of 2
    return lambda: lib.fft.fft(a)

@scenario("rfft_1M", size=1_048_576)
def bench_rfft(lib):
    a = lib.arange(1_048_576, dtype=lib.float64)
    return lambda: lib.fft.rfft(a)

# 7. Sorting
@scenario("sort_1M", size=1_000_000)
def bench_sort(lib):
    # Create a new random array each time to avoid caching
    def run():
        a = lib.random.rand(1_000_000) if hasattr(lib, 'random') else \
            lib.arange(1_000_000, dtype=lib.float64)[::-1].copy()
        return lib.sort(a)
    return run

@scenario("argsort_100K", size=100_000)
def bench_argsort(lib):
    a = lib.random.rand(100_000) if hasattr(lib, 'random') else \
        lib.arange(100_000, dtype=lib.float64)[::-1].copy()
    return lambda: lib.argsort(a)

# 8. Broadcasting
@scenario("broadcast_add_1000x1000_1000", size=1_000_000)
def bench_broadcast(lib):
    a = lib.arange(1_000_000, dtype=lib.float64).reshape(1000, 1000)
    b = lib.arange(1000, dtype=lib.float64)
    return lambda: lib.add(a, b)

# 9. Indexing
@scenario("fancy_indexing_100K", size=100_000)
def bench_fancy_index(lib):
    a = lib.arange(1_000_000, dtype=lib.float64)
    idx = lib.arange(0, 1_000_000, 10, dtype=lib.int64)[:100_000]
    return lambda: a[idx]

# 10. Type Conversion
@scenario("astype_int_to_float_1M", size=1_000_000)
def bench_astype(lib):
    a = lib.arange(1_000_000, dtype=lib.int64)
    return lambda: a.astype(lib.float64)


def get_environment_fingerprint() -> dict:
    """Capture environment info for reproducibility."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


def run_all_benchmarks(lib, lib_name: str, runs: int = 20) -> List[BenchmarkResult]:
    """Run all scenarios for a library."""
    results = []
    for name, setup_fn, size in SCENARIOS:
        try:
            bench_fn = setup_fn(lib)
            result = run_benchmark(name, bench_fn, lib_name, size=size, runs=runs)
            results.append(result)
            print(f"  {name}: p50={result.p50_ms:.3f}ms, p95={result.p95_ms:.3f}ms")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    return results


def compare_results(numpy_results: List[BenchmarkResult],
                    fnp_results: List[BenchmarkResult]) -> List[dict]:
    """Compare results and compute speedup/slowdown."""
    comparison = []
    numpy_by_name = {r.name: r for r in numpy_results}
    fnp_by_name = {r.name: r for r in fnp_results}

    for name in numpy_by_name:
        if name not in fnp_by_name:
            continue
        np_r = numpy_by_name[name]
        fnp_r = fnp_by_name[name]

        # Ratio: >1 means fnp is slower, <1 means fnp is faster
        ratio = fnp_r.p50_ms / np_r.p50_ms if np_r.p50_ms > 0 else float('inf')

        comparison.append({
            "scenario": name,
            "numpy_p50_ms": np_r.p50_ms,
            "fnp_p50_ms": fnp_r.p50_ms,
            "ratio": ratio,
            "status": "SLOWER" if ratio > 1.1 else "FASTER" if ratio < 0.9 else "EQUAL",
            "numpy_p95_ms": np_r.p95_ms,
            "fnp_p95_ms": fnp_r.p95_ms,
        })

    return sorted(comparison, key=lambda x: -x["ratio"])


def main():
    parser = argparse.ArgumentParser(description="FrankenNumPy Performance Comparison")
    parser.add_argument("--numpy-only", action="store_true", help="Only benchmark NumPy")
    parser.add_argument("--fnp-only", action="store_true", help="Only benchmark FrankenNumPy")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    args = parser.parse_args()

    print("=" * 60)
    print("FrankenNumPy vs NumPy Performance Comparison")
    print("=" * 60)

    env = get_environment_fingerprint()
    print(f"\nEnvironment: {env['platform']}")
    print(f"Python: {env['python_version']}, NumPy: {env['numpy_version']}")
    print(f"CPUs: {env['cpu_count']}")
    print(f"Runs per scenario: {args.runs}")
    print()

    results = {"environment": env, "numpy": [], "fnp": [], "comparison": []}

    # NumPy benchmarks
    if not args.fnp_only:
        print("Running NumPy benchmarks...")
        results["numpy"] = [
            {
                "name": r.name,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "p99_ms": r.p99_ms,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "throughput": r.throughput,
            }
            for r in run_all_benchmarks(np, "numpy", runs=args.runs)
        ]
        print()

    # FrankenNumPy benchmarks
    if not args.numpy_only and FNP_MODULE:
        print("Running FrankenNumPy benchmarks...")
        results["fnp"] = [
            {
                "name": r.name,
                "p50_ms": r.p50_ms,
                "p95_ms": r.p95_ms,
                "p99_ms": r.p99_ms,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "throughput": r.throughput,
            }
            for r in run_all_benchmarks(FNP_MODULE, "fnp", runs=args.runs)
        ]
        print()
    elif not args.numpy_only:
        print("FrankenNumPy not available. Set FNP_MODULE_PATH env var.")

    # Comparison
    if results["numpy"] and results["fnp"]:
        print("=" * 60)
        print("COMPARISON (ratio > 1 = fnp slower)")
        print("=" * 60)

        np_results = [BenchmarkResult(
            name=r["name"], library="numpy", times_ms=[],
            p50_ms=r["p50_ms"], p95_ms=r["p95_ms"], p99_ms=r["p99_ms"],
            mean_ms=r["mean_ms"], std_ms=r["std_ms"]
        ) for r in results["numpy"]]

        fnp_results = [BenchmarkResult(
            name=r["name"], library="fnp", times_ms=[],
            p50_ms=r["p50_ms"], p95_ms=r["p95_ms"], p99_ms=r["p99_ms"],
            mean_ms=r["mean_ms"], std_ms=r["std_ms"]
        ) for r in results["fnp"]]

        comparison = compare_results(np_results, fnp_results)
        results["comparison"] = comparison

        print(f"\n{'Scenario':<40} {'NumPy':>10} {'FNP':>10} {'Ratio':>8} {'Status':>8}")
        print("-" * 80)
        for c in comparison:
            print(f"{c['scenario']:<40} {c['numpy_p50_ms']:>10.3f} {c['fnp_p50_ms']:>10.3f} {c['ratio']:>8.2f}x {c['status']:>8}")

        # Summary
        slower = [c for c in comparison if c["status"] == "SLOWER"]
        faster = [c for c in comparison if c["status"] == "FASTER"]
        equal = [c for c in comparison if c["status"] == "EQUAL"]

        print(f"\nSummary: {len(faster)} faster, {len(equal)} equal, {len(slower)} slower")

        if slower:
            print("\nHotspots (fnp slower than numpy):")
            for c in slower[:5]:
                print(f"  - {c['scenario']}: {c['ratio']:.2f}x slower")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
