# Cross-Engine Benchmark v1

- Generated at: 2026-04-10T02:20:00Z
- Git SHA: eebce5243deba26b7db28c412dc0e22d4f37ba75
- NumPy: 2.4.3 via /data/projects/franken_numpy/.venv-numpy314/bin/python3
- BLAS backend: {
  "Compilers": {
    "c": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "14.2.1",
      "commands": "cc"
    },
    "cython": {
      "name": "cython",
      "linker": "cython",
      "version": "3.2.4",
      "commands": "cython"
    },
    "c++": {
      "name": "gcc",
      "linker": "ld.bfd",
      "version": "14.2.1",
      "commands": "c++"
    }
  },
  "Machine Information": {
    "host": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    },
    "build": {
      "cpu": "x86_64",
      "family": "x86_64",
      "endian": "little",
      "system": "linux"
    }
  },
  "Build Dependencies": {
    "blas": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.31.dev",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.14.0/lib/python3.14/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.14.0/lib/python3.14/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    },
    "lapack": {
      "name": "scipy-openblas",
      "found": true,
      "version": "0.3.31.dev",
      "detection method": "pkgconfig",
      "include directory": "/opt/_internal/cpython-3.14.0/lib/python3.14/site-packages/scipy_openblas64/include",
      "lib directory": "/opt/_internal/cpython-3.14.0/lib/python3.14/site-packages/scipy_openblas64/lib",
      "openblas configuration": "OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64",
      "pc file directory": "/project/.openblas"
    }
  },
  "Python Information": {
    "path": "/tmp/build-env-eaz9oo7u/bin/python",
    "version": "3.14"
  },
  "SIMD Extensions": {
    "baseline": [
      "X86_V2"
    ],
    "found": [
      "X86_V3"
    ],
    "not found": [
      "X86_V4",
      "AVX512_ICL",
      "AVX512_SPR"
    ]
  }
}
- Host: AMD Ryzen Threadripper PRO 5975WX 32-Cores / Linux-6.17.0-14-generic-x86_64-with-glibc2.42
- Total workloads: 37
- Median ratio: 2.10x
- Best ratio: 0.10x
- Worst ratio: 53.60x
- Band counts: green=17 yellow=9 red=11

## fft

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| fft_non_power2_medium | medium | 1.050 ms | 0.051 ms | 20.63x | red |
| fft_power2_medium | medium | 0.136 ms | 0.272 ms | 0.50x | green |
## io

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| genfromtxt_medium | medium | 0.123 ms | 0.851 ms | 0.14x | green |
| npy_roundtrip_medium | medium | 0.007 ms | 0.067 ms | 0.10x | green |
## linalg

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| eig_medium | medium | 0.093 ms | 0.090 ms | 1.02x | green |
| solve_small | small | 0.002 ms | 0.005 ms | 0.44x | green |
| svd_medium | medium | 0.174 ms | 0.062 ms | 2.78x | yellow |
## matmul

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| matmul_medium | medium | 0.039 ms | 0.012 ms | 3.40x | yellow |
| matmul_small | small | 0.001 ms | 0.001 ms | 0.86x | green |
## random

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| binomial_large | large | 3.027 ms | 4.357 ms | 0.69x | green |
| poisson_large | large | 5.760 ms | 4.812 ms | 1.20x | green |
| standard_normal_large | large | 0.796 ms | 0.793 ms | 1.00x | green |
## reductions

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| reduce_argmax_medium | medium | 0.007 ms | 0.003 ms | 2.30x | yellow |
| reduce_mean_large | large | 0.841 ms | 0.173 ms | 4.87x | yellow |
| reduce_mean_medium | medium | 0.004 ms | 0.007 ms | 0.59x | green |
| reduce_std_large | large | 1.433 ms | 1.446 ms | 0.99x | green |
| reduce_std_medium | medium | 0.012 ms | 0.026 ms | 0.46x | green |
| reduce_sum_large | large | 0.673 ms | 0.201 ms | 3.35x | yellow |
| reduce_sum_medium | medium | 0.006 ms | 0.005 ms | 1.18x | green |
## sorting

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| sort_large | large | 8.573 ms | 4.510 ms | 1.90x | green |
| sort_medium | medium | 0.075 ms | 0.049 ms | 1.53x | green |
## statistics

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| percentile_medium | medium | 0.203 ms | 0.073 ms | 2.79x | yellow |
## ufunc-broadcast

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| broadcast_add_large | large | 20.855 ms | 0.409 ms | 50.95x | red |
| broadcast_add_medium | medium | 0.053 ms | 0.004 ms | 12.05x | red |
| broadcast_add_small | small | 0.001 ms | 0.001 ms | 0.55x | green |
| broadcast_mul_large | large | 21.506 ms | 0.462 ms | 46.56x | red |
| broadcast_mul_medium | medium | 0.073 ms | 0.005 ms | 13.53x | red |
| broadcast_mul_small | small | 0.001 ms | 0.002 ms | 0.50x | green |
## ufunc-elementwise

| Workload | Size Tier | FNP p50 | NumPy p50 | Ratio | Band |
| --- | --- | --- | --- | --- | --- |
| add_f64_large | large | 20.789 ms | 0.676 ms | 30.76x | red |
| add_f64_medium | medium | 0.115 ms | 0.002 ms | 47.53x | red |
| add_f64_small | small | 0.001 ms | 0.000 ms | 2.09x | yellow |
| div_f64_large | large | 22.954 ms | 0.428 ms | 53.60x | red |
| div_f64_medium | medium | 0.075 ms | 0.003 ms | 23.40x | red |
| div_f64_small | small | 0.002 ms | 0.001 ms | 2.10x | yellow |
| mul_f64_large | large | 21.443 ms | 0.523 ms | 40.99x | red |
| mul_f64_medium | medium | 0.098 ms | 0.003 ms | 31.84x | red |
| mul_f64_small | small | 0.001 ms | 0.001 ms | 2.46x | yellow |
