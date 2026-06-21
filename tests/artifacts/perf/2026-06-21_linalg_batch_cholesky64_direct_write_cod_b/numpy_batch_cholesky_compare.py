import os
import statistics
import time

import numpy as np


def generate_batch_spd(batch, n):
    data = np.empty((batch, n, n), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    base = 1.0 / (np.abs(idx[:, None] - idx[None, :]) + 1.0)
    for b in range(batch):
        mat = base.copy()
        np.fill_diagonal(mat, (n + 1) + (b % 7) * 0.25)
        data[b] = mat
    return data


def bench(name, batch, n, reps=11):
    a = generate_batch_spd(batch, n)
    np.linalg.cholesky(a)
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        out = np.linalg.cholesky(a)
        elapsed = time.perf_counter_ns() - start
        if out.shape != (batch, n, n):
            raise SystemExit(f"bad shape for {name}: {out.shape}")
        samples.append(elapsed)
    median = int(statistics.median(samples))
    print(
        f"{name}: median_ns={median} min_ns={min(samples)} "
        f"max_ns={max(samples)} samples={samples}"
    )


print(f"python {os.popen('python3 --version').read().strip()}")
print(f"numpy {np.__version__}")
bench("batch_cholesky/shape/1000x32x32", 1000, 32)
bench("batch_cholesky/shape/500x64x64", 500, 64)
bench("batch_cholesky/shape/64x128x128", 64, 128)
