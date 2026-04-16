# FrankenNumPy

<div align="center">
  <img src="franken_numpy_illustration.webp" alt="FrankenNumPy - memory-safe clean-room NumPy reimplementation in Rust" width="400">

  **Memory-safe, clean-room NumPy reimplementation in Rust.**<br>
  Zero unsafe code. Extensive conformance coverage. Bit-exact RNG parity. Every feature family green.

  ![Rust](https://img.shields.io/badge/Rust-nightly%202024-orange)
  ![Tests](https://img.shields.io/badge/tests-extensive%20coverage-brightgreen)
  ![Unsafe](https://img.shields.io/badge/unsafe-0%20blocks-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

---

## The Problem

NumPy is the backbone of scientific Python, but it carries 30 years of C/C++ legacy: buffer overflows in parsers, undefined behavior in edge cases, opaque stride semantics, and no formal compatibility contracts. Rewriting pieces in Cython or C++ perpetuates the same class of memory bugs.

## The Solution

FrankenNumPy rebuilds NumPy's semantics from scratch in safe Rust with two non-negotiable goals:

1. **Absolute behavioral compatibility** with legacy NumPy. Not a subset, not "inspired by." The full API, edge cases and all.
2. **Rigorous architecture** with formal contracts, dual-mode runtime (strict/hardened), and differential conformance against a NumPy oracle.

## Why FrankenNumPy?

| | NumPy (C) | FrankenNumPy (Rust) |
|---|---|---|
| Memory safety | Buffer overflows possible | `#![forbid(unsafe_code)]` on all 9 crates |
| RNG parity | Reference implementation | Bit-exact PCG64DXSM core stream plus broad statistical and replay conformance coverage |
| NaN semantics | Implicit C behavior | Explicit propagation verified by 20+ oracle tests |
| Stride calculus | Evolved over decades | Clean-room deterministic engine (SCE) |
| Runtime modes | Single mode | Strict (max compat) + Hardened (safety guards) |
| Conformance | Self-referential | Differential oracle against real NumPy |
| Test coverage | pytest suite | Extensive Rust test coverage + 8-gate CI topology |

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

**Stride Calculus Engine (SCE).** All shape transformations (broadcast, reshape, transpose, view aliasing) flow through a single deterministic engine. This is the non-negotiable compatibility kernel.

**Dual-mode runtime.** Strict mode maximizes observable NumPy compatibility. Hardened mode adds safety guards and bounded defensive recovery. Both modes log decisions to an evidence ledger.

**Fail-closed by default.** Unknown wire formats, unrecognized metadata, and incompatible features cause explicit errors, not silent corruption.

**Oracle-verified.** Every RNG distribution, every linalg decomposition, and every reduction edge case is tested against NumPy's actual output from the same seed.

---

## API Surface

Over 1,000 public functions across 9 crates covering the full NumPy API:

| Category | Functions (highlights) |
|---|---|
| **Array creation** | `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `identity`, `diag`, `meshgrid`, `mgrid`, `ogrid`, `fromfunction`, `array` |
| **Shape manipulation** | `reshape`, `ravel`, `flatten`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `broadcast_to`, `concatenate`, `stack`, `vstack`, `hstack`, `split`, `tile`, `repeat`, `pad`, `r_`, `c_` |
| **Unary math** | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `sign`, `isnan`, `isinf`, `isfinite` (42 total) |
| **Binary math** | `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `power`, `fmod`, `arctan2`, `gcd`, `lcm`, `bitwise_and/or/xor`, comparisons (34 total) |
| **Reductions** | `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`, `cumsum`, `cumprod`, `all`, `any`, `count_nonzero`, `nansum`, `nanmin`, `nanmax`, `ptp` (22 total) |
| **Sorting** | `sort`, `argsort`, `searchsorted`, `partition`, `argpartition`, `unique`, `unique_all/counts/inverse/values` |
| **Set operations** | `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `in1d`, `isin` |
| **Linear algebra** | `solve`, `det`, `inv`, `eig`, `svd`, `qr`, `cholesky`, `lstsq`, `norm`, `matrix_rank`, `matrix_power`, `multi_dot`, `pinv`, `cond`, `slogdet`, `funm` |
| **Random** | Statistical distributions plus permutation/state helpers via PCG64DXSM: `normal`, `uniform`, `binomial`, `poisson`, `gamma`, `beta`, `hypergeometric`, `multinomial`, `dirichlet`, `vonmises`, `zipf`, etc. |
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
           ┌──────────────────────────────────────────────┐
           │              User API Layer                  │
           │    UFuncArray · MaskedArray · StringArray    │
           └───────────────────────┬──────────────────────┘
                                  │
      ┌───────────┬───────────────┼──────────────┬───────────┐
      ▼           ▼               ▼              ▼           ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ fnp-ufunc│ │fnp-linalg│ │fnp-random│ │  fnp-io  │ │ fnp-fft  │
 │ array ops│ │ linalg   │ │ RNG      │ │ NPY/NPZ  │ │ (in-ufunc│
 │ dispatch │ │ kernels  │ │ engine   │ │ + text   │ │  module) │
 └─────┬────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
       │
 ┌─────┴────────────────────────────────────────────────────┐
 │                                                          │
 ▼                                                          ▼
 ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
 │ fnp-dtype│ │ fnp-ndarray│ │ fnp-iter │ │ fnp-runtime  │
 │ dtype    │ │ SCE core   │ │ transfer │ │ strict/      │
 │ rules    │ │ strides    │ │ semantics│ │ hardened     │
 └──────────┘ └────────────┘ └──────────┘ └──────────────┘
                    ▲
                    │
         Stride Calculus Engine (SCE)
         - shape -> strides (C/F)
         - broadcast legality
         - reshape with -1 inference
         - alias-safe view transitions
```

The **Stride Calculus Engine** is the non-regression kernel. Every shape transformation (broadcast, reshape, transpose, view creation) is validated through SCE before execution.

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

The runtime implements a risk-aware decision engine with posterior probability estimation that chooses between three actions for each compatibility check:

| Action | When Used |
|--------|-----------|
| **Allow** | KnownCompatible input in Strict mode, or low-risk KnownCompatible in Hardened mode |
| **FullValidate** | Hardened mode with elevated risk, or malformed metadata (NaN/out-of-range inputs) |
| **FailClosed** | Unknown semantics or KnownIncompatible in any mode |

**Risk scoring with posterior estimation.** Each decision starts with class-specific prior odds (1% incompatibility for KnownCompatible, 99% for KnownIncompatible, 50% for Unknown). A risk score is computed as the log-likelihood ratio of the evidence against a threshold. The posterior incompatibility probability determines the action via a loss model.

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

The RNG crate achieves bit-exact parity with NumPy by porting every algorithm from NumPy's C source code.

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

92 public functions organized into three tiers:

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

**Complex number support.** 10 functions for complex-valued matrices: `complex_solve_nxn`, `complex_det_nxn`, `complex_inv_nxn`, `complex_cholesky_nxn`, `complex_qr_mxn`, `complex_matmul`, `complex_matvec`, `complex_conjugate_transpose`, `complex_matrix_norm_frobenius`, `complex_trace_nxn`.

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

FrankenNumPy's security posture covers more than memory safety:

- **Zero unsafe Rust.** All 9 crates declare `#![forbid(unsafe_code)]`. The 92,000+ lines of Rust contain zero unsafe blocks.
- **Fail-closed by default.** Unknown wire formats, unrecognized dtype descriptors, and metadata schema violations cause explicit errors, not silent fallbacks.
- **Bounded resource consumption.** NPY header parsing caps at 64 KB. NPZ archives cap at 4,096 members and 2 GB uncompressed. Memmap validation retries cap at 64. These prevent denial-of-service via crafted inputs.
- **Pickle rejection.** Object dtype arrays that could execute arbitrary code during deserialization require explicit opt-in, matching NumPy's `allow_pickle` security gate.
- **Adversarial conformance.** The security gate (`run_security_gate`) tests exploit scenarios from a versioned threat matrix mapped to specific parser/IO/shape-validation boundaries.

---

## Algorithm Catalog

Each subsystem uses specific numerical algorithms. This section catalogs them.

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
| Legendre | `P_n(x)` | `legval`, `legadd`, `legsub`, `legmul`, `legdiv`, `legder`, `legint`, `legroots`, `legfromroots`, `legfit`, `leg2poly`, `poly2leg` |
| Hermite (physicist) | `H_n(x)` | `hermval`, `hermadd`, `hermsub`, `hermmul`, `hermdiv`, `hermder`, `hermint`, `hermroots`, `hermfromroots`, `hermfit`, `herm2poly`, `poly2herm` |
| Hermite (probabilist) | `He_n(x)` | `hermeval`, `hermeadd`, `hermesub`, `hermemul`, `hermediv`, `hermeroots`, `hermefromroots`, `herme2poly`, `poly2herme` |
| Laguerre | `L_n(x)` | `lagval`, `lagadd`, `lagsub`, `lagmul`, `lagdiv`, `lagder`, `lagint`, `lagroots`, `lagfromroots`, `lagfit`, `lag2poly`, `poly2lag` |

All five polynomial families support the full suite of evaluation, arithmetic, calculus, root-finding, basis conversion, and fitting operations.

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

## Threat Model

12 threat classes are formally mapped in `security_control_checks_v1.yaml`, each with assigned conformance suites, compatibility gates, and override audit policies:

| Threat Class | What Could Go Wrong | Control |
|---|---|---|
| `malformed_shape` | Crafted dimensions cause OOB access or allocation bomb | Shape/stride suite + runtime policy adversarial suite |
| `unsafe_cast_path` | Silent data corruption through widening/narrowing cast | Dtype promotion suite with drift gate |
| `malicious_stride_alias` | Overlapping views cause data races or corruption | Shape/stride suite with alias drift gate |
| `malformed_npy_npz` | Malicious `.npy`/`.npz` file exploits parser bugs | IO adversarial suite + parser fail-closed gate |
| `unknown_metadata_version` | Future format version silently misinterpreted | Runtime policy suite + compatibility drift hash |
| `adversarial_fixture` | Hostile test inputs cause crash or panic | Adversarial suites across IO/RNG/linalg with reproducibility gate |
| `rng_reproducibility_drift` | Code change silently alters RNG output sequences | RNG differential + metamorphic + adversarial suites |
| `linalg_shape_tolerance_abuse` | Ill-conditioned matrix causes wrong result | Linalg differential + metamorphic suites |
| `linalg_backend_bridge_tampering` | Backend produces wrong result for well-conditioned input | Linalg adversarial + crash signature suites |
| `corrupt_durable_artifact` | Bit-rot or tampering in stored conformance artifacts | RaptorQ decode proof hash gate |
| `policy_override_abuse` | Unauthorized bypass of compatibility gates | Runtime policy adversarial + explicit audited override |

Every threat log entry must include: `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`.

---

## Shared Memory and Views

`UFuncArrayView` provides NumPy-style shared-memory views with overlap detection:

```
UFuncArrayView {
    shape: Vec<usize>,
    buffer: Arc<RwLock<Vec<f64>>>,  // shared backing store
    offset: isize,                   // byte offset into buffer
    strides: Vec<isize>,            // per-dimension strides (can be negative)
    writable: bool,
    dtype: DType,
}
```

**Memory overlap detection** uses a two-tier approach:

1. **Fast path (`may_share_memory`):** Checks whether the byte-offset spans of two views overlap. This is O(ndim) and conservative (may report false positives for non-contiguous views).

2. **Exact path (`shares_memory`):** First checks `Arc` pointer equality (different backing buffers never share). If same buffer, computes the actual set of accessed byte offsets (up to 200,000) and checks for intersection. Falls back to the fast path if offset collection exceeds the limit.

This supports safe in-place operations: if two views share memory, operations that read from one and write to the other must use temporary copies to avoid data corruption.

---

## Float Error State Machine

FrankenNumPy replicates NumPy's floating-point error handling system:

```
┌───────────────┐   seterr(divide='raise')   ┌──────────────┐
│ Default       │ ────────────────────────── │ Custom       │
│ divide=Warn   │                            │ divide=Raise │
│ over=Warn     │   errstate(all='ignore')   │ over=Warn    │
│ under=Ignore  │ ────────────────────────── │ under=Ignore │
│ invalid=Warn  │   (RAII guard restores)    │ invalid=Warn │
└───────────────┘                            └──────────────┘
```

Six error modes per category: `Ignore` (suppress), `Warn` (log and continue), `Raise` (return error), `Call` (invoke user callback), `Print` (stderr), `Log` (append to event buffer).

Four error categories: divide-by-zero, overflow, underflow, invalid operation.

`errstate()` returns an RAII guard that automatically restores the previous error configuration when dropped, matching NumPy's context manager semantics:

```rust
let _guard = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
// all float errors ignored in this scope
// previous state restored when _guard drops
```

---

## Scimath: Complex-Domain Extensions

The `scimath` module provides 8 functions that extend real-valued math to the complex domain for inputs outside the real function's natural domain:

| Function | Real Domain | Extension |
|---|---|---|
| `scimath_sqrt(x)` | x >= 0 | Returns complex sqrt for x < 0 |
| `scimath_log(x)` | x > 0 | Returns complex log for x <= 0 |
| `scimath_log2(x)` | x > 0 | Complex base-2 logarithm |
| `scimath_log10(x)` | x > 0 | Complex base-10 logarithm |
| `scimath_power(x, p)` | x >= 0 (for non-integer p) | Complex result for negative base |
| `scimath_arccos(x)` | -1 <= x <= 1 | Complex arccosine for \|x\| > 1 |
| `scimath_arcsin(x)` | -1 <= x <= 1 | Complex arcsine for \|x\| > 1 |
| `scimath_arctanh(x)` | -1 < x < 1 | Complex arctanh for \|x\| >= 1 |

These match `numpy.lib.scimath` and are useful in signal processing and physics where negative square roots or out-of-range inverse trig values arise naturally.

---

## Complete Distribution List

Random-generation methods available on `Generator`, grouped by family:

**Continuous:** `beta`, `chisquare`, `exponential`, `f`/`f_distribution`, `gamma`, `gumbel`, `halfnormal`, `laplace`, `levy`, `logistic`, `lognormal`, `lomax`, `maxwell`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `power`, `rayleigh`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `vonmises`, `wald`, `weibull`

**Discrete:** `binomial`, `geometric`, `hypergeometric`, `logseries`, `negative_binomial`, `poisson`, `zipf`

**Multivariate:** `dirichlet`, `multinomial`, `multivariate_normal`

**Uniform:** `random` (float [0,1)), `uniform` (float [low,high)), `integers` (int [low,high))

**Permutation:** `shuffle` (in-place), `permutation` (copy), `permuted` (axis-aware)

**Utility:** `bytes`, `choice`/`choice_weighted`

**State:** `spawn`, `jumped`, `state`/`set_state`

---

## Array Manipulation Toolkit

FrankenNumPy implements the full set of NumPy's array construction and manipulation functions.

### Construction

| Function | What it does |
|---|---|
| `zeros(shape)` | Array filled with 0.0 |
| `ones(shape)` | Array filled with 1.0 |
| `full(shape, val)` | Array filled with arbitrary value |
| `empty(shape)` | Uninitialized array (filled with 0.0 in practice) |
| `eye(n, m, k)` | Identity-like matrix with diagonal offset `k` |
| `identity(n)` | Square identity matrix (delegates to `eye`) |
| `diag(v, k)` | 1-D input: construct diagonal matrix. 2-D input: extract diagonal. |
| `arange(start, stop, step)` | Evenly spaced values within interval |
| `linspace(start, stop, num)` | `num` evenly spaced values including endpoints |
| `logspace(start, stop, num)` | Values spaced evenly on log scale |
| `geomspace(start, stop, num)` | Values spaced evenly on geometric scale |
| `meshgrid(x, y, ...)` | Coordinate matrices from coordinate vectors (xy indexing) |
| `fromfunction(shape, f)` | Apply closure `f(&[usize]) -> f64` to each multi-index |

### Joining and Splitting

| Function | Axis behavior |
|---|---|
| `concatenate(arrays, axis)` | Join along existing axis. All arrays must match on non-concat dims. |
| `stack(arrays, axis)` | Join along new axis. All arrays must have identical shape. |
| `vstack` / `row_stack` | Stack vertically (along axis 0). Promotes 1-D to (1, N). |
| `hstack` | Stack horizontally. For 1-D: axis 0. For N-D: axis 1. |
| `dstack` | Stack along axis 2. Promotes to at least 3-D first. |
| `column_stack` | Stack 1-D arrays as columns of a 2-D array. |
| `block(grid)` | Assemble from nested grid. Concatenates within rows, then stacks. |
| `split(ary, n, axis)` | Split into `n` equal sub-arrays along axis |
| `array_split(ary, n, axis)` | Split allowing unequal sub-arrays |

### Rearranging

| Function | What it does |
|---|---|
| `transpose(axes)` | Permute dimensions |
| `moveaxis(src, dst)` | Move one axis to a new position |
| `rollaxis(axis, start)` | Roll axis backward until before `start` |
| `swapaxes(a1, a2)` | Swap two axes via permutation |
| `expand_dims(axis)` | Insert size-1 dimension |
| `squeeze(axis)` | Remove size-1 dimensions |
| `flip(axis)` | Reverse elements along axis |
| `fliplr` / `flipud` | Left-right / up-down reversal (requires ndim >= 2 / >= 1) |
| `rot90(k)` | Rotate by k*90 degrees on first two axes |
| `roll(shift, axis)` | Circular shift with wrapping |
| `tile(reps)` | Repeat array along each axis per `reps` |
| `repeat(n, axis)` | Repeat each element `n` times |
| `resize(new_shape)` | Resize with cyclic repetition if new shape is larger |

### Advanced Indexing

| Function | What it does |
|---|---|
| `take(indices, axis)` | Select elements by integer indices (supports negative) |
| `put(indices, values)` | Replace flat-indexed elements (cyclic values) |
| `compress(condition, axis)` | Select elements where boolean condition is true |
| `extract(condition, arr)` | Flat extraction by boolean mask |
| `place(mask, vals)` | In-place replacement where mask is true (cyclic values) |
| `select(condlist, choicelist, default)` | Choose from multiple arrays by first-matching condition |
| `piecewise(condlist, funclist)` | Piecewise constant function via condition list |
| `take_along_axis(indices, axis)` | Gather values along axis by index array |
| `put_along_axis(indices, values, axis)` | Scatter values along axis by index array |
| `ravel_multi_index(coords, shape)` | Convert N-D coordinates to flat indices (C-order) |
| `unravel_index(indices, shape)` | Convert flat indices to N-D coordinates |

---

## Transfer Semantics (`fnp-iter`)

The iterator crate models NumPy's internal data transfer system, which decides *how* to move data between arrays during operations:

**Transfer classes** determine the copy strategy:

| Class | When selected | Cost |
|---|---|---|
| `Contiguous` | Both src/dst have unit strides and matching alignment | Fastest (memcpy-like) |
| `Strided` | Arbitrary strides but lossless cast | Medium (per-element stride arithmetic) |
| `StridedCast` | Arbitrary strides with lossy cast | Slowest (per-element cast + stride) |

**Overlap detection** decides copy direction:

| Action | When | Why |
|---|---|---|
| `NoCopy` | Source and destination don't overlap | No precaution needed |
| `ForwardCopy` | Overlap, but forward iteration is safe | Dst starts after src start |
| `BackwardCopy` | Overlap, forward would corrupt | Must iterate in reverse |

**FlatIter indexing** supports four modes: `Single(i)` for scalar access, `Slice{start, stop, step}` for regular ranges, `Fancy(Vec<usize>)` for arbitrary index arrays, and `BoolMask(Vec<bool>)` for boolean selection. The `count_true_mask` optimization processes boolean masks in 8-element chunks for vectorizable counting.

**Stateful NDIter wrapper.** `fnp-iter` now exposes a first-class `Nditer` state machine on top of `NditerPlan`, including `iterindex`, `multi_index`, `reset`, seek-by-index/seek-by-multi-index, and `external_loop` chunk iteration. It also ships a NumPy-backed `nditer_python*` bridge for parity checks against Python-side stepping behavior, and `fnp-python` now exposes that iterator state machine as a Python extension-module `PyNditer`.

---

## Phase2C Extraction Packets

The conformance system is organized around 9 extraction packets, each covering one domain of NumPy behavior:

| Packet | Domain | Key Contracts |
|---|---|---|
| FNP-P2C-001 | **Shape/reshape** | Element-count conservation, single -1 dimension, broadcast compatibility |
| FNP-P2C-002 | **Dtype/promotion** | Promotion matrix determinism, safe-cast policy, dtype lifecycle |
| FNP-P2C-003 | **Strided transfer** | Transfer-loop selection, cast pipeline, overlap handling, where-mask assignment |
| FNP-P2C-004 | **NDIter traversal** | Iterator construction, multi-index seek, C/F tracking, external-loop mode |
| FNP-P2C-005 | **Ufunc dispatch** | Signature parsing, method selection, override precedence, gufunc reduction |
| FNP-P2C-006 | **Stride tricks/broadcast** | as_strided views, zero-stride propagation, writeability contracts |
| FNP-P2C-007 | **RNG contracts** | Seed normalization, child-stream derivation, deterministic state, jump-ahead |
| FNP-P2C-008 | **Linalg bridge** | Solver contracts, factorization modes, spectral operations, backend dispatch |
| FNP-P2C-009 | **NPY/NPZ IO** | Magic/version validation, header-length bounds, pickle policy, truncated-data detection |

Each packet produces 8 artifact files: `legacy_anchor_map.md`, `contract_table.md`, `fixture_manifest.json`, `parity_gate.yaml`, `risk_note.md`, `parity_report.json`, `parity_report.raptorq.json`, `parity_report.decode_proof.json`. The packet readiness validator checks all 8 files exist and contain required fields before a packet is marked `ready`.

---

## RNG State Serialization

The random number generator supports full state capture and restoration for reproducibility:

```rust
// Capture state
let payload = generator.to_pickle_payload();

// Restore state
let restored = Generator::from_pickle_payload(payload)?;
// restored produces identical sequence from this point
```

`GeneratorPicklePayload` captures the bit-generator state (seed, counter, algorithm tag, schema version) and optionally the `SeedSequence` snapshot for spawn lineage tracking. The `RandomState` wrapper provides legacy compatibility with NumPy's older `numpy.random.RandomState` API.

**Seed material** accepts multiple forms: `None` (random), `U64(seed)`, `U32Words(vec)`, `SeedSequence`, or direct `State { seed, counter }` for exact state restoration. This matches NumPy's flexible seeding interface where `default_rng(12345)`, `default_rng([1, 2, 3])`, and `default_rng(SeedSequence(42))` all work.

---

## Test Coverage

Current per-crate counts move over time; see `FEATURE_PARITY.md` for the live inventory. Coverage focus by crate:

| Crate | Coverage focus |
|---|---|
| `fnp-ufunc` | Array ops, math, sorting, polynomials, reductions, oracle tests, linalg bridge, FFT, gufunc validation, parameter parity, einsum, and edge-case parity |
| `fnp-linalg` | Decompositions, solvers, norms, NumPy oracle tests, extreme-scale regression, and non-finite parity |
| `fnp-random` | Oracle-verified distributions, seeding, reproducibility, and large-n binomial/multinomial behavior |
| `fnp-io` | NPY/NPZ read/write, text formats, compression, oracle format tests, and `genfromtxt`/`fromfile` coverage |
| `fnp-dtype` | Dtype taxonomy, promotion table coverage, cast policy primitives, and NumPy byte-width parsing |
| `fnp-conformance` | Differential parity, metamorphic identities, adversarial fuzzing, witness stability, and matmul conformance |
| `fnp-ndarray` | Shape legality, stride calculus, broadcast contracts, overlap detection, multi-axis negative strides, F-order, and edge cases |
| `fnp-iter` | Transfer-loop selection, NDIter traversal/broadcast/overlap contracts, flatiter indexing/assignment, and enumeration helpers |
| `fnp-runtime` | Mode split, fail-closed decoding, override-audit gates, risk-aware decisions, and evidence-ledger behavior |

### Oracle Test Strategy

The oracle suite verifies bit-exact or behaviorally equivalent parity against NumPy across the major subsystems:

- **RNG oracle tests:** every distribution produces identical output from the same seed, verified against `numpy.random.Generator(PCG64DXSM(12345))`
- **Ufunc edge-case tests:** NaN propagation, empty arrays, Inf arithmetic, boolean dtype promotion, and sort ordering
- **Linalg oracle tests:** `det`, `inv`, `solve`, `eig`, `svd`, `cholesky`, `qr`, `norm`, `cond`, `slogdet`, `lstsq`, and rank checks
- **I/O format tests:** NPY roundtrip, magic bytes, header dict format, and 16-byte alignment

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
| Random generation | parity_green |
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

For manual direct `rch` workspace checks in this repo, prefer disabling Cargo
incremental artifacts to avoid noisy remote artifact-retrieval races under
`target/debug/incremental`:

```bash
CARGO_INCREMENTAL=0 rch exec -- cargo check --workspace --all-targets
CARGO_INCREMENTAL=0 rch exec -- cargo clippy --workspace --all-targets -- -D warnings
```

---

## Conformance Pipeline

The oracle capture pipeline runs real NumPy, captures its output, and compares against our implementation:

```bash
# Recommended: require a real NumPy oracle and let the runner bootstrap
# a repo-local .venv-numpy314 with uv if needed.
FNP_REQUIRE_REAL_NUMPY_ORACLE=1 \
  cargo run -p fnp-conformance --bin capture_numpy_oracle

# Run differential comparison
cargo run -p fnp-conformance --bin run_ufunc_differential
```

If you want to manage the interpreter explicitly, this still works:

```bash
uv venv --python 3.14 .venv-numpy314
uv pip install --python .venv-numpy314/bin/python numpy
FNP_ORACLE_PYTHON="$(pwd)/.venv-numpy314/bin/python3" \
  cargo run -p fnp-conformance --bin capture_numpy_oracle
```

| Environment Variable | Effect |
|---|---|
| `FNP_ORACLE_PYTHON` | Path to Python interpreter with NumPy. Explicit non-default paths win over repo-local bootstrap. |
| `FNP_REQUIRE_REAL_NUMPY_ORACLE=1` | Require a real NumPy oracle. If `FNP_ORACLE_PYTHON` is unset (or left at `python3`), `capture_numpy_oracle` bootstraps and reuses `.venv-numpy314` automatically, preferring `uv`, then standard `venv` + `pip`, and finally a user-site `pip install numpy` fallback when the worker lacks venv tooling. |

---

## Repository Layout

```
franken_numpy/
├── Cargo.toml                         # Workspace root (9 crates)
├── FEATURE_PARITY.md                  # Live parity matrix + evidence links
├── PROPOSED_ARCHITECTURE.md           # High-level architecture notes
├── crates/
│   ├── fnp-dtype/                     # Dtype taxonomy, promotion table, cast policy
│   ├── fnp-ndarray/                   # Stride Calculus Engine (SCE)
│   ├── fnp-iter/                      # Transfer semantics, overlap-safe iteration
│   ├── fnp-ufunc/                     # 1,000+ array operations, reductions, einsum
│   ├── fnp-linalg/                    # solve, eig, svd, qr, cholesky, lstsq, etc.
│   ├── fnp-random/                    # Statistical distributions + generator helpers, PCG64DXSM, bit-exact parity
│   ├── fnp-io/                        # NPY/NPZ read/write, text I/O, DEFLATE
│   ├── fnp-conformance/               # Oracle capture, differential harness, benchmarks
│   └── fnp-runtime/                   # Strict/hardened mode, evidence ledger
├── legacy_numpy_code/numpy/           # Behavioral oracle (upstream NumPy source)
├── artifacts/                         # Contracts, security maps, logs, proofs
└── scripts/                           # E2E gate scripts
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

What works and what doesn't:

- **Not a full Python package.** FrankenNumPy is still primarily a Rust library. There is no `pip install frankennumpy` packaging story today, but the workspace now includes an early `fnp-python` extension module exposing `PyNditer`, `frompyfunc`, `vectorize`, and `digitize`.
- **No BLAS/LAPACK backend.** Linear algebra uses pure-Rust implementations (Householder QR, Golub-Kahan SVD, implicit shifted QR for eigenvalues). Competitive with BLAS for small matrices; slower for large ones. Future BLAS linkage is planned.
- **Complex elementwise arithmetic uses interleaved storage.** Complex64/Complex128 dtypes store real/imaginary parts as interleaved floats with a trailing dimension of 2. Elementwise `multiply` and `divide` apply true complex arithmetic `(a+bi)(c+di) = (ac-bd)+(ad+bc)i`, but the interleaved representation adds overhead compared to native complex types.
- **`multivariate_normal` uses Cholesky.** NumPy defaults to SVD. Adding SVD would require `fnp-linalg` as a dependency of `fnp-random` (currently zero-dependency).
- **`multivariate_hypergeometric` uses sequential draws.** NumPy uses the `random_mvhg_marginals` algorithm.
- **`frompyfunc` now has package-level Python exposure, but only for that surface.** The Rust crate still supports closure-backed numeric `frompyfunc`, object-value outputs through `frompyfunc_object`, source-evaluated Python callables through `frompyfunc_python`, and imported Python callable handles through `frompyfunc_python_import`, while `fnp-python` now exposes a `frompyfunc` wrapper that accepts live Python callable objects and returns object-array results. Broader Python packaging and FFI coverage for the rest of FrankenNumPy remain incomplete.
- **`vectorize` now has package-level Python exposure, but only for that surface.** `fnp-python` now exposes a `vectorize` wrapper that accepts live Python callable objects, handles broadcasting across positional arguments, supports positional `excluded` indices, and returns NumPy arrays or tuples of arrays with NumPy-inferred output dtypes. Broader Python packaging and FFI coverage for the rest of FrankenNumPy remain incomplete.
- **`digitize` now has package-level Python exposure, but only for that surface.** `fnp-python` now exposes a `digitize` wrapper backed by the Rust `UFuncArray::digitize_right` implementation, with a reusable NumPy bridge for bool/int/uint/float arrays and regression coverage for decreasing bins, `right=True`, and large `uint64` values. Broader Python packaging and FFI coverage for the rest of FrankenNumPy remain incomplete.
- **`nditer` now has package-level Python exposure, but only for that surface.** `fnp_iter::Nditer` has `nditer_python` / `nditer_python_with_interpreter` for NumPy-backed state and chunk bridging, and `fnp-python` now exposes a `PyNditer` class. Broader Python packaging and FFI coverage for the rest of FrankenNumPy remain incomplete.
- **Single-threaded.** All operations are single-threaded. The `asupersync` async runtime integration is optional and used only for conformance pipeline orchestration, not for parallel array computation.
- **f64 internal representation.** `UFuncArray` stores numeric values as `Vec<f64>` internally for arithmetic. For i64/u64 values > 2^53, an `IntegerSidecar` preserves exact integer values through storage round-trips (`from_storage` / `to_storage`). Arithmetic on large integers still uses f64 approximation.

---

## FAQ

**Is this a drop-in replacement for NumPy?**
Not yet. It reimplements NumPy's semantics in Rust for correctness verification and eventual use as a Rust-native array library. A Python FFI bridge is planned but not implemented.

**How do you verify parity with NumPy?**
Oracle tests: we run the same operations with the same inputs in both NumPy and FrankenNumPy, comparing outputs to floating-point tolerance. For RNG, the comparison is bit-exact.

**Why Rust nightly?**
We use Rust Edition 2024 features. The toolchain is pinned to a specific nightly date for reproducibility.

**Why zero unsafe code?**
Memory safety is a core value. All 9 crates declare `#![forbid(unsafe_code)]`. The 92,000+ lines of Rust contain zero unsafe blocks.

**How fast is it?**
Performance is not the primary goal in this phase. Correctness and parity come first. That said, the release profile uses `opt-level = 3`, LTO, and single codegen unit.

**Can I use just the RNG crate?**
Yes. `fnp-random` has zero dependencies and produces bit-exact NumPy-compatible random sequences from a given seed.

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

MIT License (with OpenAI/Anthropic Rider). See `LICENSE`.
