# FrankenNumPy

<div align="center">
  <img src="franken_numpy_illustration.webp" alt="FrankenNumPy - memory-safe clean-room NumPy reimplementation in Rust" width="400">

  **Memory-safe, clean-room NumPy reimplementation in Rust.**<br>
  Zero unsafe code. 1,941 tests. Bit-exact RNG parity. Every feature family green.

  ![Rust](https://img.shields.io/badge/Rust-nightly%202024-orange)
  ![Tests](https://img.shields.io/badge/tests-1%2C941%20passing-brightgreen)
  ![Unsafe](https://img.shields.io/badge/unsafe-0%20blocks-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

---

## The Problem

NumPy is the backbone of scientific Python, but it carries 30 years of C/C++ legacy: buffer overflows in parsers, undefined behavior in edge cases, opaque stride semantics, and no formal compatibility contracts. Rewriting pieces in Cython or C++ perpetuates the same class of memory bugs.

## The Solution

FrankenNumPy rebuilds NumPy's semantics from scratch in safe Rust with two non-negotiable goals:

1. **Absolute behavioral compatibility** with legacy NumPy (not a subset, not "inspired by" -- the full API, edge cases and all).
2. **Rigorous architecture** with formal contracts, dual-mode runtime (strict/hardened), and differential conformance against a NumPy oracle.

## Why FrankenNumPy?

| | NumPy (C) | FrankenNumPy (Rust) |
|---|---|---|
| Memory safety | Buffer overflows possible | `#![forbid(unsafe_code)]` on all 9 crates |
| RNG parity | Reference implementation | Bit-exact match (40 oracle-verified distributions) |
| NaN semantics | Implicit C behavior | Explicit propagation verified by 20+ oracle tests |
| Stride calculus | Evolved over decades | Clean-room deterministic engine (SCE) |
| Runtime modes | Single mode | Strict (max compat) + Hardened (safety guards) |
| Conformance | Self-referential | Differential oracle against real NumPy |
| Test coverage | pytest suite | 1,941 Rust tests + 8-gate CI topology |

---

## Quick Example

```rust
use fnp_ufunc::{UFuncArray, DType, BinaryOp};

// Create arrays
let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64)?;

// Z-score normalization: (data - mean) / std
let mean = data.reduce_mean(None, false)?;
let std = data.reduce_std(None, false, 0)?;
let z = data.elementwise_binary(&mean.broadcast_to(&[5])?, BinaryOp::Sub)?
            .elementwise_binary(&std.broadcast_to(&[5])?, BinaryOp::Div)?;
// z = [-1.414, -0.707, 0.0, 0.707, 1.414]

// Sort, cumsum, percentile chain
let sorted = data.sort(None, None)?;           // [1, 2, 3, 4, 5]
let cumsum = sorted.cumsum(None)?;             // [1, 3, 6, 10, 15]
let median = data.percentile(50.0, None)?;     // 3.0

// Linear algebra
let a = UFuncArray::new(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0], DType::F64)?;
let b = UFuncArray::new(vec![2], vec![9.0, 8.0], DType::F64)?;
let x = a.solve(&b)?;                         // [2.0, 3.0]
```

```rust
use fnp_random::Generator;

// Bit-exact NumPy-compatible RNG
let mut rng = Generator::from_pcg64_dxsm(12345)?;
let normals = rng.standard_normal(1000);        // Identical to NumPy's output
let samples = rng.binomial(10, 0.5, 100);       // BTPE algorithm, same as NumPy
let perm = rng.permutation(&data)?;             // Fisher-Yates via random_interval
```

---

## Design Philosophy

**Parity debt, not feature cuts.** Every behavioral gap with NumPy is tracked as debt to be closed, not as an accepted scope reduction.

**Stride Calculus Engine (SCE).** All shape transformations -- broadcast, reshape, transpose, view aliasing -- flow through a single deterministic engine. This is the non-negotiable compatibility kernel.

**Dual-mode runtime.** Strict mode maximizes observable NumPy compatibility. Hardened mode adds safety guards and bounded defensive recovery. Both modes log decisions to an evidence ledger.

**Fail-closed by default.** Unknown wire formats, unrecognized metadata, and incompatible features cause explicit errors, not silent corruption.

**Oracle-verified.** Every RNG distribution, every linalg decomposition, and every reduction edge case is tested against NumPy's actual output from the same seed.

---

## API Surface

Over 1,000 public functions across 9 crates covering the full NumPy API:

| Category | Functions (highlights) |
|---|---|
| **Array creation** | `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `identity`, `diag`, `meshgrid`, `fromfunction`, `array` |
| **Shape manipulation** | `reshape`, `ravel`, `flatten`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `broadcast_to`, `concatenate`, `stack`, `vstack`, `hstack`, `split`, `tile`, `repeat`, `pad` |
| **Unary math** | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `sign`, `isnan`, `isinf`, `isfinite` (42 total) |
| **Binary math** | `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `power`, `fmod`, `arctan2`, `gcd`, `lcm`, `bitwise_and/or/xor`, comparisons (34 total) |
| **Reductions** | `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`, `cumsum`, `cumprod`, `all`, `any`, `count_nonzero`, `nansum`, `nanmin`, `nanmax`, `ptp` (22 total) |
| **Sorting** | `sort`, `argsort`, `searchsorted`, `partition`, `argpartition`, `unique`, `unique_all/counts/inverse/values` |
| **Set operations** | `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `in1d`, `isin` |
| **Linear algebra** | `solve`, `det`, `inv`, `eig`, `svd`, `qr`, `cholesky`, `lstsq`, `norm`, `matrix_rank`, `matrix_power`, `multi_dot`, `pinv`, `cond`, `slogdet`, `funm` |
| **Random** | 39 distributions via PCG64DXSM: `normal`, `uniform`, `binomial`, `poisson`, `gamma`, `beta`, `hypergeometric`, `multinomial`, `dirichlet`, `vonmises`, `zipf`, etc. |
| **FFT** | `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `fftfreq`, `rfftfreq`, `fftshift` |
| **Statistics** | `histogram`, `percentile`, `quantile`, `median`, `average`, `corrcoef`, `cov`, `bincount`, `digitize` |
| **Polynomials** | Chebyshev, Legendre, Hermite, Laguerre families + power series (33 functions) |
| **String arrays** | 33 `numpy.char` functions |
| **Financial** | `fv`, `pv`, `pmt`, `ppmt`, `ipmt`, `nper`, `rate`, `npv`, `irr`, `mirr` |
| **I/O** | `load`, `save`, `savez`, `savez_compressed`, `loadtxt`, `savetxt`, `genfromtxt`, `fromfile`, `tofile` |
| **Masked arrays** | `MaskedArray` with reshape, transpose, concatenate, comparison ops, `filled`, `compressed`, `anom` |
| **Datetime** | `DatetimeArray`, `TimedeltaArray` with arithmetic, `busday_count`, `busday_offset`, `is_busday` |
| **Misc** | `einsum`, `tensordot`, `kron`, `dot`, `matmul`, `outer`, `inner`, `vdot`, `convolve`, `correlate`, `gradient`, `diff`, `interp`, `clip`, `where`, `select`, `piecewise` |

---

## Installation

```bash
# Clone and build
git clone https://github.com/Dicklesworthstone/franken_numpy.git
cd franken_numpy
cargo build --workspace

# Run all tests
cargo test --workspace

# Run with all features
cargo test --workspace --all-features
```

**Requirements:** Rust nightly (pinned in `rust-toolchain.toml` to `nightly-2026-02-20`).

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              User API Layer              │
                    │  UFuncArray · MaskedArray · StringArray  │
                    └────────────────┬────────────────────────┘
                                     │
         ┌───────────────┬───────────┼───────────┬───────────────┐
         ▼               ▼           ▼           ▼               ▼
   ┌──────────┐   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ fnp-ufunc│   │fnp-linalg│ │fnp-random│ │  fnp-io  │ │  fnp-fft │
   │ 1,141    │   │  197     │ │  179     │ │  143     │ │ (in-ufunc│
   │ tests    │   │  tests   │ │  tests   │ │  tests   │ │  module) │
   └────┬─────┘   └──────────┘ └──────────┘ └──────────┘ └──────────┘
        │
   ┌────┼──────────────────────────────────────────┐
   ▼    ▼                                          ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ fnp-dtype│  │fnp-ndarray│  │ fnp-iter │  │ fnp-runtime  │
│ 114 tests│  │ 44 tests │  │ 13 tests │  │  12 tests    │
│ promotion│  │  SCE core │  │ transfer │  │ strict/      │
│ casting  │  │  strides  │  │ semantics│  │ hardened     │
└──────────┘  └──────────┘  └──────────┘  └──────────────┘
                    ▲
                    │
        Stride Calculus Engine (SCE)
        ─ shape → strides (C/F)
        ─ broadcast legality
        ─ reshape with -1 inference
        ─ alias-safe view transitions
```

The **Stride Calculus Engine** is the non-regression kernel. Every shape transformation -- broadcast, reshape, transpose, view creation -- is validated through SCE before execution.

---

## How It Works: Deep Dive

### Dtype System (`fnp-dtype`)

The type system is the foundation everything else builds on. 18 dtype variants cover the full NumPy type hierarchy:

```
Bool  I8  I16  I32  I64  U8  U16  U32  U64
F16  F32  F64  Complex64  Complex128
Str  DateTime64  TimeDelta64  Structured
```

Each dtype maps to a type-safe `ArrayStorage` variant with native Rust containers. No flattened `Vec<u8>` reinterpretation. `I64` values live in `Vec<i64>`, `Complex128` values live in `Vec<(f64, f64)>`, and `F16` values use the `half` crate's `f16` type. This means integer fidelity is preserved (no silent truncation through f64 for i64 values > 2^53), f32 identity is maintained, and complex numbers are stored as native interleaved pairs.

**Promotion table.** The `promote(lhs, rhs)` function implements NumPy's exact promotion rules as a deterministic `const fn`:

| LHS | RHS | Result | Why |
|-----|-----|--------|-----|
| Bool | anything | RHS | Bool is identity for promotion |
| U8 | I8 | I16 | Smallest signed type covering 0..255 and -128..127 |
| U16 | I16 | I32 | Smallest signed type covering both ranges |
| U64 | any signed | F64 | No integer type holds both U64.max and negative values |
| F16 | I8/U8 | F16 | NumPy preserves float16 for 8-bit ints |
| F16 | I16/U16 | F32 | 16-bit integers exceed float16 mantissa |
| F32 | I32/I64 | F64 | float32 can't represent all 32/64-bit ints |
| Complex64 | I32 | Complex128 | Mirrors the F32+I32 → F64 widening rule |

The U64+signed→F64 promotion is the most counterintuitive rule in the table, but it's what NumPy does: there simply isn't a 128-bit integer type in NumPy's type system.

**Cast policy.** `can_cast(src, dst, casting)` supports all five NumPy casting modes: `"no"`, `"equiv"`, `"safe"`, `"same_kind"`, and `"unsafe"`.

### Stride Calculus Engine (`fnp-ndarray`)

The SCE owns all shape transformation rules and is the correctness backbone of the entire project:

1. **Shape → strides.** Given a shape `[d0, d1, ..., dn]` and memory order (C or F), compute contiguous strides. C-order: rightmost dimension has stride = item_size, each dimension to the left multiplies by the next dimension's size. F-order: leftmost dimension has stride = item_size.

2. **Broadcast legality.** Two shapes are broadcast-compatible when, aligned from the right, each pair of dimensions is either equal or one is 1. The output shape takes the maximum at each position. FrankenNumPy handles mixed-rank broadcasting (different number of dimensions) by left-padding the shorter shape with 1s.

3. **Reshape with `-1` inference.** At most one dimension can be -1, meaning "infer from the total element count and the other dimensions." The engine divides the total element count by the product of known dimensions and validates that the result is exact (no remainder).

4. **View safety.** `as_strided()` and `sliding_window_view()` compute the required byte span from the minimum to maximum reachable offset, then verify it fits within the base allocation. Negative strides (for reverse slicing) are fully supported.

5. **Broadcast strides.** When a dimension has size 1 but the broadcast output has size > 1, the stride for that dimension is set to 0. This creates a "virtual repeat" without copying data.

### Dual-Mode Runtime (`fnp-runtime`)

The runtime implements a Bayesian decision engine that chooses between three actions for each compatibility check:

| Action | When Used |
|--------|-----------|
| **Allow** | KnownCompatible input in Strict mode, or low-risk KnownCompatible in Hardened mode |
| **FullValidate** | Hardened mode with elevated risk, or malformed metadata (NaN/out-of-range inputs) |
| **FailClosed** | Unknown semantics or KnownIncompatible in any mode |

**Bayesian risk scoring.** Each decision starts with class-specific prior odds (1% incompatibility for KnownCompatible, 99% for KnownIncompatible, 50% for Unknown). A risk score is computed as the log-likelihood ratio of the evidence against a threshold. The posterior incompatibility probability determines the action via a loss model.

The loss model encodes the asymmetric costs:

```
allow_if_compatible:       0.0    (correct accept: no cost)
allow_if_incompatible:   100.0    (silent corruption: catastrophic)
full_validate_if_compatible: 4.0  (unnecessary work: small cost)
full_validate_if_incompatible: 2.0 (caught by validation: acceptable)
fail_closed_if_compatible: 125.0  (false rejection: high cost)
fail_closed_if_incompatible: 1.0  (correct rejection: minimal cost)
```

Every decision is logged to an `EvidenceLedger` with timestamp, mode, class, evidence terms, and action taken. Override requests are tracked separately in `OverrideAuditEvent` records with audit references.

### Ufunc Dispatch (`fnp-ufunc`)

The ufunc engine is the largest crate (~30,000 lines) and implements the array operation layer:

**35 binary operations** — `Add`, `Sub`, `Mul`, `Div`, `Power`, `FloorDivide`, `Remainder`, `Fmod`, `Minimum`, `Maximum`, `Fmax`, `Fmin`, `Copysign`, `Heaviside`, `Nextafter`, `Arctan2`, `Hypot`, `Logaddexp`, `Logaddexp2`, `Ldexp`, `FloatPower`, `BitwiseAnd/Or/Xor`, `LeftShift`, `RightShift`, `LogicalAnd/Or/Xor`, and 6 comparison ops.

**42 unary operations** — `Abs`, `Negative`, `Sqrt`, `Exp`, `Log`, all trig and hyperbolic functions, `Floor`, `Ceil`, `Round`, `Cbrt`, `Expm1`, `Log1p`, `Spacing`, `Signbit`, `Isnan`, `Isinf`, `Isfinite`, `Invert`, and more.

Every binary operation handles broadcasting through the same code path: compute the output shape via SCE's `broadcast_shape`, map output indices to source indices using broadcast strides (0-stride for size-1 dimensions), and apply the operation elementwise.

**NaN semantics are explicit.** After fixing 18 NaN-handling bugs found through systematic oracle testing:
- `reduce_min`, `reduce_max`: propagate NaN (use custom `nan_min`/`nan_max`, not `f64::min`/`f64::max`)
- `cummin`, `cummax`: NaN propagates through the accumulator
- `sort`, `argsort`: NaN sorts to end via `nan_last_cmp` (not `partial_cmp().unwrap_or(Equal)`)
- `median`, `percentile`, `quantile`: NaN early-returns before sorting
- `ptp`: NaN propagates in both UFuncArray and MaskedArray variants

**Float error handling.** A thread-local `FloatErrorState` tracks divide-by-zero, overflow, underflow, and invalid operation events. Each mode (Ignore, Warn, Raise, Call, Print, Log) can be configured independently. `seterr()`, `geterr()`, and `errstate()` match NumPy's API.

**Generalized ufuncs.** `GufuncSignature` parses signatures like `"(n?,k),(k,m?)->(n?,m?)"` for operations that act on sub-arrays (e.g., matrix multiply). The parser normalizes signatures, extracts core dimensions, and validates operand shapes against the pattern.

**einsum.** Three implementations: basic `einsum()` for direct evaluation, `einsum_path()` for contraction path optimization, and `einsum_optimized()` for greedy or brute-force strategy selection.

### Random Number Generation (`fnp-random`)

The RNG crate achieves bit-exact parity with NumPy by porting every algorithm from NumPy's C source code. Here is how each layer works:

**Bit generators.** Three implementations:
- `Pcg64DxsmRng` — the default. State is 128-bit with 128-bit increment. Uses the DXSM output function and cheap multiplier for generation. Seeding goes through `SeedSequence` which applies Melissa O'Neill's seed_seq design with SplitMix64 mixing.
- `Mt19937Rng` — Mersenne Twister for legacy `RandomState` compatibility. 624-word state array with Matsumoto-Nishimura twist.
- `DeterministicRng` — simple SplitMix64-based generator for testing.

**SeedSequence.** Hierarchical seeding with spawn/generate_state contracts. `spawn(n)` creates child SeedSequences by incorporating a monotonic spawn counter into the entropy pool. `generate_state(words)` cycles through the pool with hash transformations to produce initialization vectors. This exactly matches NumPy's `numpy.random.SeedSequence`.

**Bounded integers.** NumPy's `random_bounded_uint64()` dispatch:
- Range fits in 32 bits → Lemire's method via buffered `next_uint32()` (each u64 is split into two u32s, low first)
- Range exceeds 32 bits → Lemire's method with 128-bit multiplication
- Range = 0 → return 0; Range = u32::MAX → raw `next_uint32()`; Range = u64::MAX → raw `next_u64()`

**Shuffle/permutation.** Uses `random_interval()` (masked bit rejection, not Lemire), matching NumPy's `_shuffle_raw` code path. For each index from n-1 down to 1, generate a uniform random integer in [0, i] by masking to the smallest bit-width covering i and rejecting values > i.

### Linear Algebra (`fnp-linalg`)

93 public functions organized into three tiers:

**2x2 fast paths.** `solve_2x2`, `det_2x2`, `inv_2x2`, `qr_2x2`, `svd_2x2`, `eigh_2x2`, `cholesky_2x2` avoid the overhead of general NxN algorithms for the smallest matrix size.

**NxN general algorithms:**
- **QR decomposition** — Householder reflections with `qr_nxn` (reduced) and `qr_mxn` (rectangular)
- **SVD** — Golub-Kahan bidiagonalization + implicit shifted QR
- **Eigenvalues** — Hessenberg reduction + implicit shifted QR iteration for real Schur form
- **Symmetric eigenvalues** — Tridiagonal reduction + implicit QL shifts (`eigvalsh_nxn`, `eigh_nxn`)
- **LU factorization** — Partial pivoting with `lu_factor_nxn` and `lu_solve`
- **Cholesky** — Column-wise lower-triangular factorization
- **Least squares** — Normal equations via A^T A for `lstsq_nxn`

**Spectral methods:** `expm_nxn` (matrix exponential via Pade approximation), `sqrtm_nxn` (matrix square root), `logm_nxn` (matrix logarithm), `funm_nxn` (general matrix function via Schur decomposition), `polar_nxn` (polar decomposition U*P), `schur_nxn` (Schur triangular form).

**Batch operations.** 14 batch functions (`batch_inv`, `batch_det`, `batch_solve`, `batch_svd`, `batch_qr`, `batch_cholesky`, `batch_eig`, `batch_eigvalsh`, `batch_eigh`, `batch_slogdet`, `batch_svd_full`, `batch_matrix_norm`, `batch_matrix_rank`, `batch_trace`) operate on stacked matrices with leading batch dimensions, matching NumPy's broadcasting semantics for linear algebra.

**Complex number support.** 15+ functions for complex-valued matrices: `complex_solve_nxn`, `complex_det_nxn`, `complex_inv_nxn`, `complex_cholesky_nxn`, `complex_qr_mxn`, `complex_matmul`, `complex_matvec`, `complex_conjugate_transpose`, `complex_matrix_norm_frobenius`, `complex_trace_nxn`.

### I/O Format Handling (`fnp-io`)

**NPY format.** Complete implementation of NumPy's `.npy` binary format:
- Magic prefix `\x93NUMPY` + version (1.0 or 2.0) + header length + Python dict header + padding to 16-byte alignment + raw data payload
- 24 supported dtype descriptors including little/big-endian variants for all integer and float types, complex types, fixed-width byte strings, Unicode strings, and object type
- Header parsing validates all three required fields (`descr`, `fortran_order`, `shape`) and rejects unknown keys
- Hardened with `MAX_HEADER_BYTES = 65,536` to prevent allocation bombs

**NPZ format.** ZIP-based container for multiple arrays:
- `savez` (stored) and `savez_compressed` (DEFLATE) via the `flate2` crate
- `MAX_ARCHIVE_MEMBERS = 4,096` and `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 GB` limits prevent archive bombs
- Each member is a complete `.npy` file within the ZIP

**Text I/O.** `loadtxt`, `savetxt`, and `genfromtxt` handle delimiter-separated text files with configurable dtypes, missing value handling, and column selection (`usecols`).

**Pickle policy.** Object dtype arrays require explicit opt-in via `allow_pickle=true`, matching NumPy's security posture against arbitrary code execution from untrusted `.npy` files.

### Conformance Infrastructure (`fnp-conformance`)

The conformance crate is the quality backbone with four layers:

1. **Differential harness.** Captures NumPy's output for a corpus of input fixtures, then runs the same inputs through FrankenNumPy and compares shapes, dtypes, and values (with configurable tolerance). Covers ufunc, linalg, FFT, polynomial, string, masked array, datetime, RNG, and I/O operations.

2. **Metamorphic testing.** Verifies algebraic identities that must hold regardless of input: `a + b = b + a`, `a * 1 = a`, `sum(a) = sum(sort(a))`, etc. 13 identities tested.

3. **Adversarial fuzzing.** Tests behavior on hostile inputs: NaN-filled arrays, extreme shapes (0-d, empty, very large), denormalized floats, integer overflow, and malformed NPY headers.

4. **Witness stability.** Hardcoded expected values for every RNG distribution ensure that code changes don't silently alter output sequences. When an algorithm is intentionally changed (e.g., porting Lemire's method), the witness values are regenerated from the new implementation.

---

## Security Model

FrankenNumPy's security stance goes beyond memory safety:

- **Zero unsafe Rust.** All 9 crates declare `#![forbid(unsafe_code)]`. The 88,000+ lines of Rust contain zero unsafe blocks.
- **Fail-closed by default.** Unknown wire formats, unrecognized dtype descriptors, and metadata schema violations cause explicit errors, not silent fallbacks.
- **Bounded resource consumption.** NPY header parsing caps at 64 KB. NPZ archives cap at 4,096 members and 2 GB uncompressed. Memmap validation retries cap at 64. These prevent denial-of-service via crafted inputs.
- **Pickle rejection.** Object dtype arrays that could execute arbitrary code during deserialization require explicit opt-in, matching NumPy's `allow_pickle` security gate.
- **Adversarial conformance.** The security gate (`run_security_gate`) tests exploit scenarios from a versioned threat matrix mapped to specific parser/IO/shape-validation boundaries.

---

## Algorithm Catalog

Beyond the core engine described above, FrankenNumPy implements dozens of specific numerical algorithms. Here is what each subsystem uses under the hood.

### FFT

Hybrid approach supporting arbitrary input lengths:

- **Power-of-two lengths:** Cooley-Tukey decimation-in-time (DIT). Recursively splits into even/odd subsequences, applies butterfly operations with twiddle factors `exp(-2*pi*i*k/N)`.
- **Non-power-of-two lengths:** Bluestein's chirp-Z transform. Rewrites the DFT as a convolution, zero-pads to the next power of two, uses Cooley-Tukey for the convolution, then extracts the result.

All transforms (`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`) build on these two primitives. Multi-dimensional transforms apply 1-D FFTs sequentially along each axis. `fftfreq` and `rfftfreq` compute the frequency bin centers; `fftshift`/`ifftshift` rearrange zero-frequency to the center.

### Signal Processing

- **`convolve(a, v)`:** Direct O(n*m) convolution, full mode (output length n+m-1). Flips the kernel and slides it across the input.
- **`correlate(a, v)`:** Cross-correlation implemented as `convolve(a, v[::-1])`.
- **`convolve2d` / `correlate2d`:** Full 2-D convolution with output shape `(h1+h2-1, w1+w2-1)`.

### Numerical Differentiation and Interpolation

**`gradient(f, *varargs)`** computes numerical derivatives with configurable edge handling:

| Position | edge_order=1 | edge_order=2 |
|----------|-------------|--------------|
| Interior | `(f[k+1] - f[k-1]) / 2h` | same (central difference) |
| First element | `(f[1] - f[0]) / h` | `(-3f[0] + 4f[1] - f[2]) / 2h` |
| Last element | `(f[n-1] - f[n-2]) / h` | `(3f[n-1] - 4f[n-2] + f[n-3]) / 2h` |

Non-uniform spacing uses Lagrange polynomial coefficients (3-point at edges, 2-point for edge_order=1).

**`diff(a, n)`** computes n-th discrete difference: `diff[i] = a[i+1] - a[i]`, applied n times. Output length decreases by n.

**`interp(x, xp, fp)`** does 1-D piecewise linear interpolation: binary search to find the enclosing interval in `xp`, then `fp[lo] * (1-t) + fp[hi] * t` where `t = (x - xp[lo]) / (xp[hi] - xp[lo])`. Values outside `xp` are clamped to the boundary values.

### Windowing Functions

Five window types for spectral analysis, all returning `[1.0]` for M <= 1:

| Window | Formula |
|--------|---------|
| Hamming | `0.54 - 0.46 * cos(2*pi*i / (M-1))` |
| Hanning | `0.5 - 0.5 * cos(2*pi*i / (M-1))` |
| Blackman | `0.42 - 0.5 * cos(2*pi*i / (M-1)) + 0.08 * cos(4*pi*i / (M-1))` |
| Bartlett | `1 - |2*(i - (M-1)/2) / (M-1)|` |
| Kaiser | `I0(beta * sqrt(1 - r^2)) / I0(beta)` (modified Bessel I0 via piecewise rational approx) |

### Histogram and Binning

**`histogram(a, bins)`** supports three automatic bin-count strategies:

| Strategy | Formula |
|----------|---------|
| `"sturges"` | `ceil(log2(n)) + 1` |
| `"sqrt"` | `ceil(sqrt(n))` |
| `"auto"` | `max(sturges, sqrt)` |

Bin edges are uniformly spaced: `min + i * (max - min) / bins` for i in 0..=bins. Element-to-bin assignment uses O(log bins) binary search per element. `histogram_bin_edges` returns just the edges without counting. `histogramdd` generalizes to N dimensions.

### Padding

11 pad modes matching NumPy's `np.pad`:

| Mode | Behavior |
|------|----------|
| `constant` | Fill with a specified value (default 0) |
| `edge` | Replicate the edge value |
| `wrap` | Modular wrapping via `rem_euclid` |
| `reflect` | Mirror at edge, not duplicating the edge element |
| `symmetric` | Mirror at edge, duplicating the edge element |
| `linear_ramp` | Linear interpolation from edge to a ramp endpoint |
| `maximum` | Fill with the maximum of a window of the array edge |
| `minimum` | Fill with the minimum of a window |
| `mean` | Fill with the mean of a window |
| `median` | Fill with the median of a window |
| `empty` | Leave padded values uninitialized (filled with 0.0) |

### Financial Mathematics

Ten time-value-of-money functions using closed-form annuity algebra where possible:

| Function | Method |
|----------|--------|
| `fv`, `pv`, `pmt` | Closed-form annuity factor `(1+rate)^nper` |
| `nper` | Logarithmic inversion of annuity formula |
| `npv` | Discounted cashflow sum `sum(cf[i] / (1+rate)^i)` |
| `irr` | Newton's method (max 100 iterations, tol 1e-12, initial guess 0.1) |
| `mirr` | Separates positive/negative cashflows, applies reinvestment/finance rates |
| `rate` | Newton's method on the annuity equation |
| `ipmt`, `ppmt` | Interest and principal portions derived from `fv` and `pmt` |

### Polynomial Systems

Five complete polynomial families, each with evaluation, arithmetic, calculus, and fitting:

| Family | Basis | Key Operations |
|--------|-------|----------------|
| Power series | `x^n` | `polyval`, `polyder`, `polyint`, `polyfit`, `polymul`, `polyadd`, `polysub`, `polydiv`, `polyroots` |
| Chebyshev | `T_n(x)` | `chebval`, `chebadd`, `chebsub`, `chebmul`, `chebdiv`, `chebder`, `chebint`, `chebroots`, `chebfromroots`, `chebfit`, `cheb2poly`, `poly2cheb` |
| Legendre | `P_n(x)` | `legval`, `legder`, `legint`, `legfit` |
| Hermite (physicist) | `H_n(x)` | `hermval`, `hermder`, `hermint` |
| Hermite (probabilist) | `He_n(x)` | `hermeval` |
| Laguerre | `L_n(x)` | `lagval`, `lagder`, `lagint` |

Chebyshev has the fullest support because its numerical conditioning makes it the recommended basis for most practical problems.

### Masked Arrays

`MaskedArray` wraps a `UFuncArray` with an optional boolean mask and fill value:

```
MaskedArray { data: UFuncArray, mask: Option<UFuncArray>, fill_value: f64, hard_mask: bool }
```

Mask convention: `1.0` = masked (excluded from computation), `0.0` = valid. This matches NumPy's `numpy.ma` module. Operations that reduce masked arrays (`sum`, `mean`, `min`, `max`, `var`, `std`, `median`, `ptp`, `argmin`, `argmax`, `cumsum`, `cumprod`, `count`) skip masked values. Comparison and arithmetic operations propagate masks through `mask_or`. `compressed()` returns a 1-D array of only the unmasked values. `filled(fill_value)` replaces masked values with a fill value.

### Datetime and Timedelta

`DatetimeArray` and `TimedeltaArray` support temporal arithmetic:
- Datetime - Datetime = Timedelta
- Datetime + Timedelta = Datetime
- Timedelta + Timedelta = Timedelta
- Scalar multiplication of Timedelta

Business day functions: `busday_count(start, end)` counts business days between dates, `busday_offset(date, offset)` adds business days to a date, `is_busday(date)` checks if a date is a business day. Weekday mask is Monday-Friday.

### String Operations

33 `numpy.char` functions operating elementwise on string arrays: `upper`, `lower`, `capitalize`, `title`, `center`, `ljust`, `rjust`, `zfill`, `strip`, `lstrip`, `rstrip`, `replace`, `find`, `rfind`, `count`, `startswith`, `endswith`, `isnumeric`, `isalpha`, `isdigit`, `isdecimal`, `str_len`, `encode`, `decode`, `translate`, `maketrans`, `partition`, `rpartition`, `split`, `rsplit`, `join`, `expandtabs`, `swapcase`. String `add` concatenates elementwise; string `multiply` repeats.

### Bit Packing

- **`packbits(axis)`:** Packs 8 boolean elements into 1 byte, MSB first. Output length = `ceil(axis_len / 8)`.
- **`unpackbits(axis, count)`:** Unpacks bytes into boolean bits, MSB to LSB. Output length = `axis_len * 8` (or `count` if specified).

### Einsum Optimization

The `einsum` implementation supports three contraction strategies:

| Strategy | Method | Best For |
|----------|--------|----------|
| `"greedy"` | At each step, contract the pair with smallest intermediate result | Default, fast for any operand count |
| `"optimal"` | Dynamic programming over all contraction orderings | Optimal for <= 10 operands |
| `"auto"` | Selects greedy | Convenience alias |

`einsum_path` returns the contraction order without executing, for inspection. `einsum_optimized` applies the selected strategy.

---

## Artifact Durability (RaptorQ)

Every conformance artifact (fixture bundles, benchmark baselines, migration manifests) is protected by erasure-coding sidecars:

**Encoding.** Source data is hashed (SHA-256) and encoded into source + repair symbols using RaptorQ fountain codes. The sidecar records the codec parameters (symbol size, block count, repair overhead) alongside the encoded symbols in base64.

**Scrubbing.** A scrub report decodes all symbols, computes the SHA-256 of the decoded payload, and verifies it matches the source hash. It then drops one symbol and verifies that recovery from the remaining symbols still produces the correct hash.

**Decode proof.** An explicit artifact recording which symbol was dropped, how many repair symbols were needed to recover, and whether recovery succeeded. This provides machine-checkable evidence that the artifact can survive single-symbol loss.

The G8 CI gate (`run_raptorq_gate.sh`) enforces that all required bundles have valid sidecars, scrub reports with `status: "ok"`, and decode proofs with `recovery_success: true`.

---

## Test Coverage

| Crate | Tests | What it covers |
|---|---|---|
| `fnp-ufunc` | 1,141 | Array ops, math, sorting, polynomials, NaN-correct reductions, 20 oracle tests, 4 workflow tests |
| `fnp-linalg` | 197 | Decompositions, solvers, norms, 16 NumPy oracle tests |
| `fnp-random` | 179 | 40 oracle-verified distributions, seeding, reproducibility |
| `fnp-io` | 143 | NPY/NPZ read/write, text formats, compression, 7 format oracle tests |
| `fnp-dtype` | 114 | Dtype taxonomy, promotion table, cast policy |
| `fnp-conformance` | 93 | Differential parity, metamorphic identities, adversarial fuzzing |
| `fnp-ndarray` | 44 | Shape legality, stride calculus, broadcast contracts |
| `fnp-iter` | 13 | Transfer semantics, overlap detection |
| `fnp-runtime` | 12 | Mode split, fail-closed decoding, override-audit gate |
| **Total** | **1,936** | **+ doc-tests = 1,941 in workspace** |

### Oracle Test Strategy

83 oracle tests verify bit-exact parity against NumPy:

- **40 RNG oracle tests** -- every distribution produces identical output from the same seed, verified against `numpy.random.Generator(PCG64DXSM(12345))`
- **20 ufunc edge-case tests** -- NaN propagation, empty arrays, Inf arithmetic, boolean dtype promotion, sort ordering
- **16 linalg oracle tests** -- det, inv, solve, eig, svd, cholesky, QR, norm, cond, slogdet, lstsq, rank
- **7 I/O format tests** -- NPY roundtrip, magic bytes, header dict format, 16-byte alignment

### RNG Algorithm Parity

Every algorithm was ported from NumPy's C source code:

| Algorithm | NumPy source | Our implementation |
|---|---|---|
| Bounded integers | `random_bounded_uint64()` | Lemire's method with 32/64-bit dispatch + buffered `next_uint32` |
| Binomial | `random_binomial_btpe()` + `_inversion()` | BTPE for large n*p, inversion for small |
| Hypergeometric | `hypergeometric_hrua()` + `_sample()` | HRUA (Stadlober 1989) + direct via `random_interval` |
| Poisson | `random_poisson_ptrs()` + `_mult()` | PTRS (Hormann 1993) for lam >= 10, multiplicative for small |
| Gamma | `random_standard_gamma()` | Marsaglia-Tsang + rejection for shape < 1 + exponential for shape = 1 |
| Zipf | `random_zipf()` | Exact rejection with Umin clamping |
| Shuffle | `_shuffle_raw()` | Fisher-Yates via `random_interval()` masked rejection |

---

## Parity Status

Every feature family is `parity_green`:

| Feature Family | Status |
|---|---|
| Shape/stride/view semantics | parity_green |
| Broadcasting legality | parity_green |
| Dtype promotion/casting | parity_green |
| Core math (ufunc) | parity_green |
| Reductions (NaN-propagating) | parity_green |
| Sorting (NaN-last) | parity_green |
| Set operations | parity_green |
| Indexing | parity_green |
| Polynomials (5 families) | parity_green |
| Statistics | parity_green |
| FFT | parity_green |
| String arrays | parity_green |
| Masked arrays | parity_green |
| Datetime/timedelta | parity_green |
| Linear algebra | parity_green |
| Random (39 distributions) | parity_green |
| I/O (npy/npz) | parity_green |
| Financial | parity_green |
| Scimath | parity_green |

See `FEATURE_PARITY.md` for the full matrix with evidence links.

---

## CI Gate Topology

Eight ordered gates run from fast to heavy:

```
G1  fmt + lint           cargo fmt --check && cargo clippy -- -D warnings
G2  unit/property        cargo test --workspace --lib
G3  oracle differential  capture_numpy_oracle → run_ufunc_differential
G4  adversarial/security run_security_policy_gate.sh
G5  test/logging contract run_test_contract_gate.sh
G6  workflow e2e         run_workflow_scenario_gate.sh
G7  performance budget   run_performance_budget_gate.sh
G8  durability/decode    run_raptorq_gate.sh
```

Run all gates:

```bash
scripts/e2e/run_ci_gate_topology.sh
```

---

## Conformance Pipeline

The oracle capture pipeline runs real NumPy, captures its output, and compares against our implementation:

```bash
# Set up NumPy oracle environment
uv venv --python 3.14 .venv-numpy314
uv pip install --python .venv-numpy314/bin/python numpy

# Capture oracle output
FNP_ORACLE_PYTHON="$(pwd)/.venv-numpy314/bin/python3" \
  cargo run -p fnp-conformance --bin capture_numpy_oracle

# Run differential comparison
cargo run -p fnp-conformance --bin run_ufunc_differential
```

| Environment Variable | Effect |
|---|---|
| `FNP_ORACLE_PYTHON` | Path to Python interpreter with NumPy (default: `python3`) |
| `FNP_REQUIRE_REAL_NUMPY_ORACLE=1` | Fail fast if NumPy is unavailable (recommended) |

---

## Repository Layout

```
franken_numpy/
├── Cargo.toml                         # Workspace root (9 crates)
├── crates/
│   ├── fnp-dtype/                     # Dtype taxonomy, promotion table, cast policy
│   ├── fnp-ndarray/                   # Stride Calculus Engine (SCE)
│   ├── fnp-iter/                      # Transfer semantics, overlap-safe iteration
│   ├── fnp-ufunc/                     # 1,000+ array operations, reductions, einsum
│   ├── fnp-linalg/                    # solve, eig, svd, qr, cholesky, lstsq, etc.
│   ├── fnp-random/                    # 39 distributions, PCG64DXSM, bit-exact parity
│   ├── fnp-io/                        # NPY/NPZ read/write, text I/O, DEFLATE
│   ├── fnp-conformance/               # Oracle capture, differential harness, benchmarks
│   └── fnp-runtime/                   # Strict/hardened mode, evidence ledger
├── legacy_numpy_code/numpy/           # Behavioral oracle (upstream NumPy source)
├── artifacts/                         # Contracts, security maps, logs, proofs
├── scripts/                           # E2E gate scripts
└── docs/                              # Specs and planning documents
```

---

## Performance

FrankenNumPy prioritizes correctness over speed in this phase, but the architecture is designed for future optimization:

- **Release profile**: `opt-level = 3`, LTO, single codegen unit, stripped symbols.
- **Contiguous reduction kernel**: Axis reductions on contiguous data avoid per-element index computation. A targeted optimization pass reduced axis-reduction latency by ~56% (p50/p95/p99 deltas of ~90% on contiguous workloads).
- **Broadcast index mapping**: Output-to-source index mapping uses incremental odometer updates instead of full unravel/remap per element.
- **2x2 fast paths**: Linear algebra has specialized 2x2 implementations that bypass general NxN overhead for the most common small matrix case.
- **Horner's method**: Polynomial evaluation and Stirling series use Horner's form for numerical stability and minimal multiplications.
- **Ziggurat sampling**: Normal and exponential random variates use the Ziggurat method (same as NumPy), which accepts ~97% of samples on the first try.

Performance budgets are enforced by the G7 gate, which measures p50/p95/p99 latencies for ufunc and reduction sentinel workloads and rejects regressions.

---

## Limitations

Honest about what is and is not done:

- **Not a Python package.** FrankenNumPy is a Rust library. There is no `pip install` or Python FFI bridge yet. You cannot `import frankennumpy` from Python today.
- **No BLAS/LAPACK backend.** Linear algebra uses pure-Rust implementations (Householder QR, Golub-Kahan SVD, implicit shifted QR for eigenvalues). Competitive with BLAS for small matrices; slower for large ones. Future BLAS linkage is planned.
- **Complex elementwise arithmetic is not specialized.** Complex64/Complex128 dtypes exist and complex linalg operations work, but elementwise `multiply` and `divide` on complex arrays treat them as interleaved real/imaginary floats rather than applying complex multiplication `(a+bi)(c+di) = (ac-bd)+(ad+bc)i`.
- **`multivariate_normal` uses Cholesky.** NumPy defaults to SVD. Adding SVD would require `fnp-linalg` as a dependency of `fnp-random` (currently zero-dependency).
- **`multivariate_hypergeometric` uses sequential draws.** NumPy uses the `random_mvhg_marginals` algorithm.
- **`frompyfunc` and `nditer`** require Python callable protocol (N/A for Rust).
- **Single-threaded.** All operations are single-threaded. The `asupersync` async runtime integration is optional and used only for conformance pipeline orchestration, not for parallel array computation.
- **f64 internal representation.** `UFuncArray` stores all numeric values as `Vec<f64>` internally (the `ArrayStorage` type-safe representation is in `fnp-dtype`). This means the ufunc layer loses integer fidelity for i64 values > 2^53. The dtype layer preserves it.

---

## FAQ

**Is this a drop-in replacement for NumPy?**
Not yet. It reimplements NumPy's semantics in Rust for correctness verification and eventual use as a Rust-native array library. A Python FFI bridge is planned but not implemented.

**How do you verify parity with NumPy?**
Oracle tests: we run the same operations with the same inputs in both NumPy and FrankenNumPy, comparing outputs to floating-point tolerance. For RNG, the comparison is bit-exact.

**Why Rust nightly?**
We use Rust Edition 2024 features. The toolchain is pinned to a specific nightly date for reproducibility.

**Why zero unsafe code?**
Memory safety is a core value. All 9 crates declare `#![forbid(unsafe_code)]`. The 88,000+ lines of Rust contain zero unsafe blocks.

**How fast is it?**
Performance is not the primary goal in this phase. Correctness and parity come first. That said, the release profile uses `opt-level = 3`, LTO, and single codegen unit.

**Can I use just the RNG crate?**
Yes. `fnp-random` has zero dependencies and produces bit-exact NumPy-compatible random sequences from a given seed.

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

MIT License (with OpenAI/Anthropic Rider). See `LICENSE`.
