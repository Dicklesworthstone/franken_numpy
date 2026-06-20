import math
import platform
import statistics
import time

import numpy as np


def generate_invertible_matrix(n):
    a = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a[i, j] = (n * 2.0) if i == j else (((i + j) % 5) * 0.1)
    return a


def bench(label, fn, samples=21):
    for _ in range(3):
        fn()

    inner = 1
    while True:
        start = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        elapsed = time.perf_counter_ns() - start
        if elapsed >= 50_000_000 or inner >= 64:
            break
        inner *= 2

    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        values.append((time.perf_counter_ns() - start) / inner)

    values.sort()
    median = statistics.median(values)
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    cv = 0.0 if mean == 0.0 else stdev / mean * 100.0
    p95 = values[math.ceil(0.95 * len(values)) - 1]
    print(
        f"{label}: median_ns={median:.0f} min_ns={values[0]:.0f} "
        f"p95_ns={p95:.0f} max_ns={values[-1]:.0f} mean_ns={mean:.0f} "
        f"cv_pct={cv:.2f} inner={inner} samples={samples}"
    )


def main():
    print("host hz1")
    print("python", platform.python_version())
    print("numpy", np.__version__)
    for n in (64, 128, 256, 512):
        a = generate_invertible_matrix(n)
        bench(f"cond_nxn/size/{n}", lambda a=a: np.linalg.cond(a))


if __name__ == "__main__":
    main()
