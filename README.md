# FrankenNumPy

<div align="center">
  <img src="franken_numpy_illustration.webp" alt="FrankenNumPy — memory-safe clean-room NumPy reimplementation in Rust" width="400">

  **A memory-safe, clean-room reimplementation of NumPy in Rust.**<br>
  100% of `numpy.__all__` (499/499) is reachable as `fnp_python.<name>`, structurally locked by a conformance test that fails CI on regression. Zero hand-written `unsafe` blocks across all 10 implementation crates (9 declare `#![forbid(unsafe_code)]`; `fnp-python` is opt-out only because PyO3 macros may expand to unsafe). 6,392 tests. Bit-exact PCG64DXSM RNG parity.

  ![Rust](https://img.shields.io/badge/Rust-nightly%202026--02--20-orange)
  ![Edition](https://img.shields.io/badge/edition-2024-blue)
  ![Tests](https://img.shields.io/badge/tests-6%2C392%20passing-brightgreen)
  ![Surface](https://img.shields.io/badge/numpy.__all__-499%2F499%20(100%25)-brightgreen)
  ![Unsafe](https://img.shields.io/badge/unsafe-0%20blocks-blue)
  ![CI Gates](https://img.shields.io/badge/CI%20gates-G1%E2%80%93G8-blueviolet)
  ![License](https://img.shields.io/badge/license-MIT%2BRider-green)
</div>

---

## Table of Contents

- [The Problem](#the-problem) · [The Solution](#the-solution) · [Who This Is For](#who-this-is-for) · [Why FrankenNumPy?](#why-frankennumpy)
- [Quick Example](#quick-example) · [More Worked Examples](#more-worked-examples)
- [Design Philosophy](#design-philosophy) · [Installation](#installation) · [API Surface](#api-surface)
- [Architecture](#architecture) · [Workspace and Crate Map](#workspace-and-crate-map)
- [How It Works: Per-Crate Deep Dive](#how-it-works-per-crate-deep-dive)
  · [Dtype](#dtype-system-fnp-dtype) · [SCE](#stride-calculus-engine-fnp-ndarray) · [Transfer/Iter](#transfer-semantics-and-iteration-fnp-iter) · [Ufunc](#ufunc-dispatch-fnp-ufunc) · [Linalg](#linear-algebra-fnp-linalg) · [Random](#random-number-generation-fnp-random) · [IO](#io-format-handling-fnp-io) · [Runtime](#dual-mode-runtime-fnp-runtime) · [Conformance](#conformance-infrastructure-fnp-conformance) · [Python](#python-bindings-fnp-python)
- [Algorithm Catalog](#algorithm-catalog) · [Numerical Stability and Precision Notes](#numerical-stability-and-precision-notes) · [Determinism Guarantees](#determinism-guarantees)
- [Complete Distribution List](#complete-distribution-list) · [Array Manipulation Toolkit](#array-manipulation-toolkit)
- [Shared Memory and Views](#shared-memory-and-views) · [Float Error State Machine](#float-error-state-machine)
- [Error Taxonomy](#error-taxonomy) · [Threading and Concurrency Model](#threading-and-concurrency-model) · [Versioning and Compatibility Promises](#versioning-and-compatibility-promises) · [Reproducibility Recipe](#reproducibility-recipe) · [Multi-Agent Development Process](#multi-agent-development-process)
- [SCE Invariants in Detail](#sce-invariants-in-detail) · [NaN Semantics Cheat Sheet](#nan-semantics-cheat-sheet) · [Memory Footprint and Allocation Patterns](#memory-footprint-and-allocation-patterns) · [How to Investigate a Parity Issue](#how-to-investigate-a-parity-issue)
- [NumPy Submodule Coverage Matrix](#numpy-submodule-coverage-matrix) · [Cookbook](#cookbook) · [Migrating from NumPy: A Checklist](#migrating-from-numpy-a-checklist) · [Common Pitfalls](#common-pitfalls) · [Reading the Evidence Ledger](#reading-the-evidence-ledger)
- [Security Model](#security-model) · [Threat Model](#threat-model) · [Phase2C Extraction Packets](#phase2c-extraction-packets)
- [Test Coverage](#test-coverage) · [Conformance Methodology Deep-Dive](#conformance-methodology-deep-dive) · [CI Gate Topology](#ci-gate-topology) · [Conformance Pipeline](#conformance-pipeline)
- [Performance](#performance) · [Artifact Durability (RaptorQ)](#artifact-durability-raptorq) · [Divergence Ledger](#divergence-ledger) · [Fuzzing](#fuzzing)
- [Parity Status](#parity-status) · [Comparison with Other Rust Array Libraries](#comparison-with-other-rust-array-libraries) · [Repository Layout](#repository-layout)
- [Limitations](#limitations) · [Roadmap (Phase 3 candidates)](#roadmap-phase-3-candidates) · [Anti-Goals](#anti-goals) · [Performance Levers Applied So Far](#performance-levers-applied-so-far)
- [What "Clean-Room" Means Here](#what-clean-room-means-here) · [F-order vs C-order in Practice](#f-order-vs-c-order-in-practice) · [Reading the Source Code](#reading-the-source-code)
- [Inspirations and Prior Art](#inspirations-and-prior-art) · [Algorithm References and Citations](#algorithm-references-and-citations) · [Glossary](#glossary) · [Project Timeline at a Glance](#project-timeline-at-a-glance) · [FAQ](#faq)
- [About Contributions](#about-contributions) · [License](#license)

---

## The Problem

NumPy is the bedrock of scientific Python. It is also 30 years of C and Cython carrying every memory bug that any of that code ever had: buffer overruns in parsers, undefined behavior in edge cases, opaque stride semantics, and no machine-checkable compatibility contract. Rewriting hot paths in more C or Cython preserves the same class of bugs and the same opaque semantics.

## The Solution

FrankenNumPy rebuilds NumPy's behavior from scratch in safe Rust with two goals:

1. **Full behavioral compatibility** with legacy NumPy. Not a subset, not "inspired by." The full API, edge cases and all.
2. **A tighter architecture** with formal contracts, a deterministic shape/stride engine, a dual-mode runtime (strict / hardened), and differential conformance against a real NumPy oracle on every CI run.

The full Python surface is reachable today through the `fnp_python` PyO3 extension, with engine fast-paths in safe Rust for performance-relevant operations and identity-preserving fallback to numpy for the rest. Coverage is structurally enforced by the `fnp_python_covers_full_numpy_all` conformance test, which iterates `numpy.__all__` at run time against the live numpy on the build host.

## Who This Is For

FrankenNumPy is not trying to replace NumPy for every workload tomorrow. It targets a specific audience:

| Audience | Why FrankenNumPy is useful |
|---|---|
| **Rust applications that want numerical arrays** without taking on `unsafe` C dependencies, calling out to BLAS, or shelling to Python. The 10 implementation crates have **zero hand-written unsafe blocks** and **zero external runtime C dependencies** in the numeric core. |
| **Reproducibility-critical pipelines** (regulatory ML, scientific publication, financial backtesting) that need **bit-exact RNG and dtype-deterministic promotion** across machines and across NumPy versions. PCG64DXSM streams are bit-exact vs upstream NumPy; the 324-pair promotion table is exhaustively explicit. |
| **Security-conscious systems** that read untrusted `.npy` / `.npz` from third parties. NumPy's C parsers are an attack surface; the `fnp-io` parsers are bounded, fail-closed, and fuzzed. |
| **Embedded / kiosk-style Rust deployments** that want NumPy-shaped APIs without dragging a Python interpreter into the build. |
| **Library authors writing differential-test oracles** against their own numerical code. The `fnp-conformance` crate is a reusable oracle harness; it captures NumPy reference output and compares against any implementation, not just FrankenNumPy. |
| **Researchers studying NumPy semantics.** Every shape transformation, dtype promotion, and ufunc dispatch decision is implemented as a small, readable, fully tested Rust function. The codebase is a legible specification of NumPy's actual behavior. |
| **NumPy users who want a drop-in Python module** with native-speed hot paths and numpy fallback everywhere else. (Caveat: no pip wheel yet; you build locally for now. See Limitations.) |

This is the wrong tool if your bottleneck is large dense matmul on >2,000×2,000 matrices with OpenBLAS already linked, or if you need GPU acceleration today. Those gaps are documented in Limitations and tracked in the Roadmap.

## Why FrankenNumPy?

| | NumPy (C / Cython) | FrankenNumPy (Rust) |
|---|---|---|
| Memory safety | Buffer overflows possible | 9 of 10 implementation crates declare `#![forbid(unsafe_code)]`; the 10th (`fnp-python`) contains zero unsafe blocks in its own source. The lint is opt-out only because PyO3 macros may expand into unsafe |
| `numpy.__all__` surface | Reference | 499/499 (100%), structurally locked by CI conformance test |
| RNG parity | Reference | Bit-exact PCG64DXSM core stream, oracle-verified distributions |
| NaN semantics | C-level behavior | Explicit propagation across reductions / sort / median / ptp |
| Stride calculus | Evolved over decades | Single deterministic engine (SCE) owning all shape transforms |
| Runtime modes | Single | Strict (max compat) + Hardened (safety guards) with evidence ledger |
| Conformance | Self-referential | Differential oracle against real NumPy on every CI build |
| Input hardening | Best-effort | Bounded resource limits + fail-closed on unknown semantics |
| Test coverage | pytest suite | 6,392 Rust tests across 10 crates + 8-gate CI topology + 27 fuzz targets |
| Format durability | None | RaptorQ erasure-coded sidecars + scrub + decode-proof for every artifact bundle |

---

## Quick Example

### Rust

```rust
use fnp_dtype::DType;
use fnp_ufunc::{BinaryOp, UFuncArray};

// Build an array and z-score normalize it.
let data = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64)?;
let mean = data.reduce_mean(None, false)?;       // shape [] (0-dim scalar)
let std  = data.reduce_std(None, false, 0)?;     // shape [] (0-dim scalar)
// NumPy auto-broadcasts scalars; the explicit Rust API requires
// broadcasting the 0-dim mean/std to [5] before elementwise ops.
let z = data
    .elementwise_binary(&mean.broadcast_to(&[5])?, BinaryOp::Sub)?
    .elementwise_binary(&std .broadcast_to(&[5])?, BinaryOp::Div)?;
// z ≈ [-1.414, -0.707, 0.0, 0.707, 1.414]

// Sort, cumsum, percentile chain.
let sorted = data.sort(None, None)?;        // [1, 2, 3, 4, 5]
let cumsum = sorted.cumsum(None)?;          // [1, 3, 6, 10, 15]
let median = data.percentile(50.0, None)?;  // 3.0

// Linear algebra.
let a = UFuncArray::new(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0], DType::F64)?;
let b = UFuncArray::new(vec![2], vec![9.0, 8.0], DType::F64)?;
let x = a.solve(&b)?;                       // [2.0, 3.0]
```

### Bit-exact NumPy-compatible RNG

```rust
use fnp_random::Generator;

let mut rng = Generator::from_pcg64_dxsm(12345)?;
let normals = rng.standard_normal(1000);             // identical to numpy.random.Generator(PCG64DXSM(12345)).standard_normal(1000)
let samples = rng.binomial(10, 0.5, 100)?;           // BTPE algorithm, NumPy-exact
let perm    = rng.permutation(&[1.0, 2.0, 3.0, 4.0, 5.0])?; // Fisher–Yates via random_interval, matches NumPy's _shuffle_raw
```

### Python (via the `fnp_python` PyO3 extension)

```python
import fnp_python as np

# Identity-equal to numpy on submodule surfaces that have no engine substitute,
# native Rust fast-paths on the hot ones.
print(np.__version__)
print(np.__numpy_version__)        # the numpy used as fallback oracle
print(np.linalg.solve([[3, 1], [1, 2]], [9, 8]))

rng = np.random.default_rng(12345)
print(rng.standard_normal(5))      # bit-exact NumPy parity
```

---

## More Worked Examples

### Stride tricks: a 1-D sliding window with zero copy

```rust
use fnp_ufunc::UFuncArray;
use fnp_dtype::DType;

let signal = UFuncArray::new(vec![10], (0..10).map(|i| i as f64).collect(), DType::F64)?;

// Build a window-of-3 view over the 10-sample signal. The result is shape (8, 3);
// `window_shape` is the per-axis window size, so a 1-D source takes `&[3]`.
let windows = signal.sliding_window_view(&[3])?;
// rows: [0,1,2], [1,2,3], [2,3,4], ..., [7,8,9]
let row_means = windows.reduce_mean(Some(1), false)?;
```

The view is bounds-checked: `as_strided` and `sliding_window_view` compute the maximum reachable byte offset and refuse to construct the view if it would escape the base allocation. Negative strides across multiple axes work the same way.

### Reproducible RNG with full state capture

```rust
use fnp_random::{BitGenerator, BitGeneratorKind, Generator, SeedSequence};

let mut rng_a = Generator::from_pcg64_dxsm(42)?;
let _ = rng_a.standard_normal(1000);          // advance the stream

let payload = rng_a.to_pickle_payload();      // in-process snapshot (typed Rust struct)
let mut rng_b = Generator::from_pickle_payload(payload)?;
assert_eq!(rng_a.standard_normal(50), rng_b.standard_normal(50));

// `SeedSequence::spawn(n)` creates independent child streams for parallel work.
// `new` takes a u32 entropy slice; `spawn` requires `&mut self`.
let mut parent = SeedSequence::new(&[42u32])?;
let children = parent.spawn(8)?;              // 8 independent, reproducible streams
let _streams: Vec<BitGenerator> = children
    .iter()
    .map(|ss| BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64Dxsm, ss))
    .collect::<Result<_, _>>()?;
```

`GeneratorPicklePayload` is a typed in-memory struct (`Debug + Clone + PartialEq + Eq`); it carries the bit-generator state schema-version plus the optional `SeedSequence` snapshot. There is no built-in byte serialization today, so cross-process persistence requires an explicit transport layer. The structured types make adding one (or layering serde derives behind a feature flag) straightforward.

### Fail-closed I/O on hostile input

```rust
use fnp_io::{load, IOError, MAX_HEADER_BYTES};

// A crafted .npy file claims a 200-MB header dict. The parser refuses without
// allocating a 200-MB buffer: MAX_HEADER_BYTES = 65,536. Header-bound and other
// schema violations surface as `IOError::HeaderSchemaInvalid` with a descriptive
// static-string reason ("header length exceeds platform usize boundary", etc.).
let result = load(crafted_bytes);
assert!(matches!(result, Err(IOError::HeaderSchemaInvalid(_))));
```

The same fail-closed treatment applies to:
- NPZ archives claiming more than `MAX_ARCHIVE_MEMBERS = 4,096` entries
- NPZ archives whose declared uncompressed size exceeds `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 GiB`
- Text input whose total element count would exceed `MAX_TEXT_ELEMENTS = 16,777,216`
- Object-dtype `.npy` files without `allow_pickle=true`
- Unknown dtype descriptors, future format versions, and malformed header keys

### Floating-point error policy via RAII

```rust
use fnp_ufunc::{errstate, FloatErrorMode, geterr};

let before = geterr();                      // { divide: Warn, over: Warn, ... }

{
    let _guard = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
    // All FP errors silenced in this scope.
    let _ = some_division_that_might_divzero();
    // _guard drops at end of scope -> state restored automatically.
}

assert_eq!(geterr(), before);
```

`errstate` returns an RAII guard; dropping the guard restores the prior configuration, matching NumPy's context-manager semantics. Six modes are available per category (`Ignore`, `Warn`, `Raise`, `Call`, `Print`, `Log`); four categories (divide, over, under, invalid) are tracked independently.

---

## Design Philosophy

1. **Parity debt, not feature cuts.** Surface parity is locked at 100% of `numpy.__all__`. Behavioral edge cases that still drift are tracked as parity debt to be closed, never as accepted scope reduction.
2. **The Stride Calculus Engine (SCE) is the compatibility kernel.** Every shape transformation (broadcast, reshape, transpose, view aliasing, sliding window) flows through a single deterministic Rust engine that owns all the legality rules. Replacing the SCE means replacing FrankenNumPy.
3. **Dual-mode runtime.** Strict mode maximizes observable NumPy compatibility with no behavior-altering repairs. Hardened mode preserves the API contract while adding safety guards and bounded defensive recovery for malformed inputs. Every decision lands in an evidence ledger with class / risk / action / loss-model context.
4. **Fail-closed by default.** Unknown wire formats, unrecognized dtype descriptors, future metadata-version markers, and semantically incompatible inputs all cause explicit errors. There are no silent fallbacks past safety boundaries.
5. **Oracle-verified, durably stored.** Every RNG distribution, every linalg decomposition, every reduction edge case is differentially compared to NumPy's actual output from the same seed. Every conformance bundle, benchmark baseline, and migration manifest is protected by a RaptorQ erasure-coded sidecar plus a machine-checked scrub and decode-proof.

---

## Installation

### Build the workspace

```bash
git clone https://github.com/Dicklesworthstone/franken_numpy.git
cd franken_numpy
cargo build --workspace
cargo test  --workspace      # 6,392 tests
cargo test  --workspace --all-features
```

Requirements: **Rust nightly `nightly-2026-02-20`**, pinned in `rust-toolchain.toml` and mirrored in `.github/workflows/ci.yml` via the `RUST_TOOLCHAIN` env var (single source of truth).

### Python access via the `fnp_python` PyO3 extension

```bash
cargo build -p fnp-python --release --features python-extension
# The compiled cdylib's filename varies by platform; rename and place on PYTHONPATH:
#   Linux:   target/release/libfnp_python.so    → fnp_python.so
#   macOS:   target/release/libfnp_python.dylib → fnp_python.so
#   Windows: target/release/fnp_python.dll      → fnp_python.pyd
```

```python
import fnp_python as np
np.__version__              # workspace version (0.1.0)
np.__numpy_version__        # version of numpy used as fallback oracle
np.linalg.solve([[3, 1], [1, 2]], [9, 8])
```

There is **no `pip install frankennumpy` wheel/PyPI flow yet**; packaging is the only residual gap (see the FAQ and the Limitations section). The Python surface itself is complete and CI-locked.

### Optional features

| Crate | Feature | Effect |
|---|---|---|
| `fnp-python` | `python-extension` | Compile as PyO3 cdylib (default for the build above) |
| `fnp-runtime` | `asupersync` | Optional structured-async integration for conformance/artifact pipelines |
| `fnp-runtime` | `frankentui` | Optional terminal-native dashboards for parity-drift and perf deltas |

---

## API Surface

**1,575 public Rust functions across the 10 library crates** (verified live count of `pub fn ` declarations under `crates/*/src/**/*.rs`, excluding the `src/bin/` binary entry-points), exposing **100% of `numpy.__all__` (499/499 names)** through the `fnp_python` Python module. Coverage is enforced by `fnp_python_covers_full_numpy_all` in `crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs`, which iterates `numpy.__all__` at run time against the live numpy on the build host. The per-function `crates/fnp-python/tests/conformance_*.rs` shards add another **133 dedicated parity files** on top of the surface lock. The companion `run_fnp_python_api_coverage` gate independently tracks the Python-visible surface (`exports=633 covered=599 missing=0` as of 2026-05-13), of which the PyO3 wrappers and identity-equal numpy re-exports together hit every name in `numpy.__all__`.

`fnp_python` also registers **12 PyO3 classes**: `Nditer` / `NditerStep` (iterator state machine), `FromPyFunc` / `Vectorize` (callable wrappers), and the `random` submodule's `SeedSequence`, `Generator`, `RandomState`, plus the 5 bit-generator classes `MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`. The `mgrid`, `ogrid`, `r_`, and `c_` NumPy-style index objects are exposed as live singleton instances.

| Category | Functions (highlights) |
|---|---|
| **Array creation** | `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `identity`, `diag`, `meshgrid`, `mgrid`, `ogrid`, `fromfunction`, `frombuffer`, `array`, `asarray`, `copy` |
| **Shape manipulation** | `reshape`, `ravel`, `flatten`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `broadcast_to`, `broadcast_arrays`, `broadcast_shapes`, `concatenate`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`, `split`, `array_split`, `vsplit`, `hsplit`, `dsplit`, `tile`, `repeat`, `pad`, `append`, `delete`, `insert`, `roll`, `flip`, `fliplr`, `flipud`, `rot90`, `moveaxis`, `rollaxis`, `resize`, `trim_zeros`, `r_`, `c_` |
| **Unary math** | `abs`, `negative`, `positive`, `sign`, `sqrt`, `square`, `cbrt`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `degrees`, `radians`, `floor`, `ceil`, `rint`, `trunc`, `round`, `reciprocal`, `spacing`, `fabs`, `signbit`, `isnan`, `isinf`, `isfinite`, `logical_not`, `bitwise_not` (43 variants of `enum UnaryOp`) |
| **Binary math** | `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `power`, `float_power`, `fmod`, `arctan2`, `copysign`, `heaviside`, `nextafter`, `fmax`, `fmin`, `logaddexp`, `logaddexp2`, `ldexp`, `hypot`, `gcd`, `lcm`, `bitwise_and/or/xor`, comparisons (35 total) |
| **Special** | `frexp`, `modf`, `divmod`, `isposinf`, `isneginf`, `bitwise_count`, `sort_complex`, `clip`, `where`, `copyto`, `sinc`, `unwrap`, `conj`, `real`, `imag`, `real_if_close`, `angle` |
| **Reductions** | `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`, `cumsum`, `cumprod`, `cumulative_sum`, `cumulative_prod`, `all`, `any`, `count_nonzero`, `nansum`, `nanprod`, `nanmin`, `nanmax`, `nanmean`, `nanstd`, `nanvar`, `nanargmin`, `nanargmax`, `nancumsum`, `nancumprod`, `nanpercentile`, `nanquantile`, `ptp` |
| **Sorting / searching** | `sort`, `argsort`, `searchsorted` (with `side`, `sorter`), `partition`, `argpartition`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`, `nonzero`, `flatnonzero`, `argwhere`, `isin` (with `invert`), `extract`, `place`, `select`, `piecewise` |
| **Set operations** | `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `isin` (NumPy 2.x replacement for `in1d`) |
| **Linear algebra** | `solve`, `det`, `inv`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `qr`, `cholesky`, `lstsq`, `norm`, `matrix_rank`, `matrix_power`, `multi_dot`, `tensorsolve`, `tensorinv`, `pinv`, `cond`, `slogdet`, `funm`, `expm`, `sqrtm`, `logm`, `polar`, `schur`, `solve_triangular`, `block`, plus batched and complex variants (~100 public functions in `fnp-linalg`) |
| **Random** | Five bit generators (`PCG64`, `PCG64DXSM`, `MT19937`, `Philox`, `SFC64`); `PCG64DXSM` is the default and the bit-exact NumPy-parity reference. 40+ oracle-verified distributions (`normal`, `uniform`, `binomial`, `poisson`, `gamma`, `beta`, `hypergeometric`, `multinomial`, `dirichlet`, `vonmises`, `zipf`, …) |
| **FFT (18 entry points)** | `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` |
| **Statistics** | `histogram`, `histogram2d`, `histogramdd`, `histogram_bin_edges`, `bincount`, `digitize`, `percentile`, `quantile`, `median`, `average`, `corrcoef`, `cov` |
| **Polynomials (5 families)** | Power series, Chebyshev, Legendre, Hermite (physicist + probabilist), Laguerre. Full evaluation / arithmetic / calculus / root-finding / fitting / basis conversion suites |
| **String arrays** | 33 elementwise `numpy.char` functions |
| **Financial** | `fv`, `pv`, `pmt`, `ppmt`, `ipmt`, `nper`, `rate`, `npv`, `irr`, `mirr` |
| **I/O** | `load`, `save`, `savez`, `savez_compressed`, `loadtxt`, `savetxt`, `genfromtxt`, `fromfile`, `tofile`, `fromstring`, `array2string`, memmap helpers |
| **Masked arrays** | `MaskedArray` with reshape, transpose, concatenate, comparison ops, `filled`, `compressed`, `shrink_mask`, `anom`, `fix_invalid`, `is_masked`, `make_mask`, `mask_or` |
| **Datetime / timedelta** | `UFuncArray` with `DType::DateTime64` / `DType::TimeDelta64`, built via `UFuncArray::from_datetime_strings` / `from_timedelta_strings`. Arithmetic, comparison, `busday_count`, `busday_offset`, `is_busday`. Units `ns`, `us`, `ms`, `s`, `min`, `h`, `D`, `W`, `M`, `Y` via `DateTimeUnit`. |
| **Tensor + general ops** | `einsum`, `einsum_path`, `tensordot`, `kron`, `dot`, `matmul`, `vdot`, `inner`, `outer`, `convolve`, `correlate`, `gradient`, `diff`, `interp`, `clip`, `where`, `select`, `piecewise` |
| **Scimath** | `scimath_sqrt`, `scimath_log`, `scimath_log2`, `scimath_log10`, `scimath_logn`, `scimath_power`, `scimath_arccos`, `scimath_arcsin`, `scimath_arctanh` (complex-aware extensions of real-domain math) |
| **NumPy 2.0+ API** | `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`, `permuted`, `matrix_transpose`, `cumulative_sum`, `cumulative_prod`, `trapezoid`, `unstack`, `vecdot` |

See [`FEATURE_PARITY.md`](FEATURE_PARITY.md) for the complete live parity matrix and per-crate test counts.

---

## Architecture

```
           ┌──────────────────────────────────────────────┐
           │              User API Layer                  │
           │   UFuncArray · MaskedArray · StringArray     │
           │ (datetime / timedelta = UFuncArray + DType)  │
           └───────────────────────┬──────────────────────┘
                                   │
       ┌───────────┬───────────────┼──────────────┬───────────┐
       ▼           ▼               ▼              ▼           ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ fnp-ufunc│ │fnp-linalg│ │fnp-random│ │  fnp-io  │ │fnp-python│
  │ array ops│ │ linalg   │ │ RNG      │ │ NPY/NPZ  │ │ PyO3     │
  │ + FFT    │ │ kernels  │ │ engine   │ │ + text   │ │ bindings │
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
                     ▲                            ▲
                     │                            │
       Stride Calculus Engine (SCE)        Evidence Ledger
       - shape → strides (C/F)             + Decision Engine
       - broadcast legality                + Override Audit
       - reshape with -1 inference
       - alias-safe view transitions
```

(The only first-class user-facing array struct types in `fnp-ufunc` are `UFuncArray`, `MaskedArray`, and `StringArray`. Datetime and timedelta arrays are not separate struct types; they are `UFuncArray` instances with `DType::DateTime64` or `DType::TimeDelta64` and `ArrayStorage::I64` backing, with the unit (`ns`, `us`, `ms`, `s`, `min`, `h`, `D`, `W`, `M`, `Y`) carried on the dtype via `DateTimeUnit`.)

`fnp-python` sits above the canonical chain as the Python-facing surface; it does not alter the layering below it.

**`fnp-conformance` is the perpendicular axis.** It consumes every crate, captures NumPy oracle output, runs differential / metamorphic / adversarial / witness suites, generates RaptorQ artifacts, and exposes 47 dedicated binaries under `crates/fnp-conformance/src/bin/` that drive the CI gates.

---

## Workspace and Crate Map

10 implementation crates, all under `crates/fnp-*`. 9 of the 10 declare `#![forbid(unsafe_code)]`; `fnp-python` is the lone exception, because PyO3's procedural macros may expand into unsafe as part of generating the cdylib entry point. In practice, the current `fnp-python` source contains zero hand-written `unsafe` blocks (verified by ripgrep); the lint is opt-out, not invoked.

| Crate | Lines (src/) | Purpose |
|---|---:|---|
| `fnp-dtype` | 3,190 | 18 dtype variants, deterministic `const fn` promotion table covering all 324 pairs, 5 cast policies, `ArrayStorage` taxonomy. (The `IntegerSidecar` mentioned later in this README lives in `fnp-ufunc`, not here, because it travels with `UFuncArray`.) |
| `fnp-ndarray` | 1,531 | Stride Calculus Engine: shape→element_count, shape+order→strides, broadcast legality, reshape with `-1` inference, alias-sensitive transitions |
| `fnp-iter` | 3,745 | Transfer-loop selector, overlap detection, `Nditer` / `NditerPlan` / `NditerStep` state machine with `iterindex` / `multi_index` / reset / seek / external-loop chunks, `nditer_python*` parity bridge |
| `fnp-ufunc` | 59,299 | Largest crate. 35 binary ops, 43 unary ops, 30+ reduction methods, masked arrays, string arrays, datetime arrays, polynomial families, FFT primitives, einsum (`einsum`, `einsum_path`, `einsum_optimized`), float error state machine, NaN-correct reductions |
| `fnp-linalg` | 10,120 | ~100 public functions: 2×2 fast paths, NxN decompositions (QR, SVD, eig, eigh, Cholesky, LU), spectral methods (`expm`, `sqrtm`, `logm`, `funm`, `polar`, `schur`), least-squares, 14 batch operations, complex variants |
| `fnp-random` | 12,204 | 5 production bit generators (PCG64, PCG64DXSM, MT19937, Philox, SFC64) + an internal `DeterministicRng` for tests, full `SeedSequence` / `SeedMaterial` hierarchy with spawn lineage, pickle payload round-trip, `RandomState` legacy wrapper, 40+ oracle-verified distributions |
| `fnp-io` | 8,002 | NPY 1.0 / 2.0 read & write, NPZ stored + DEFLATE, text I/O (`loadtxt`, `savetxt`, `genfromtxt`), binary I/O (`fromfile`, `tofile`, `fromstring`), memmap, structured dtype I/O, hardened bounds: `MAX_HEADER_BYTES = 65,536`, `MAX_ARCHIVE_MEMBERS = 4,096`, `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 GiB`, `MAX_TEXT_ELEMENTS = 16,777,216` |
| `fnp-runtime` | 1,672 | `RuntimeMode` (Strict / Hardened), `CompatibilityClass`, `DecisionAction`, risk-aware decision engine with posterior estimation, `DecisionLossModel`, `EvidenceLedger`, `OverrideAuditEvent`, optional `asupersync` / `frankentui` integration |
| `fnp-conformance` | 65,546 | Differential harness, metamorphic identities, adversarial fuzzing, witness stability, oracle capture, diagnostic-oracle harness, divergence ledger checker, cross-version drift matrix, RaptorQ sidecar / scrub / decode-proof tooling, 47 CLI binaries under `src/bin/` |
| `fnp-python` | 68,715 | PyO3 bindings exposing 100% of `numpy.__all__`. 12 registered classes, native fast-paths for hot operations, identity-equal numpy fallback for surfaces with no engine substitute, 133 dedicated `conformance_*.rs` test files |

Total: roughly **304k lines of safe Rust** in the implementation crates, plus seven fuzz crates and an extensive `tests/` tree.

---

## How It Works: Per-Crate Deep Dive

### Dtype System (`fnp-dtype`)

The type system is the foundation everything else builds on. **18 dtype variants** cover the full NumPy type hierarchy:

```
Bool  I8  I16  I32  I64  U8  U16  U32  U64
F16  F32  F64  Complex64  Complex128
Str  DateTime64  TimeDelta64  Structured
```

Each dtype maps to a type-safe `ArrayStorage` variant with native Rust containers; no flattened `Vec<u8>` reinterpretation. `I64` values live in `Vec<i64>`, `Complex128` values live in `Vec<(f64, f64)>`, `F16` uses the `half` crate's `f16` type, structured dtypes are described by typed `StructuredField` / `StructuredStorage`. Integer fidelity is preserved (no silent truncation through f64 for i64 values > 2^53), f32 identity is maintained, and complex numbers are stored as native interleaved pairs.

**Promotion table.** `promote(lhs, rhs)` is a deterministic `const fn`. All **324 dtype pairs are explicitly handled** with no catch-all fallback; adding a new dtype variant causes a compile error until promotion rules are added for it. Selected rules:

| LHS | RHS | Result | Why |
|-----|-----|--------|-----|
| Bool | anything | RHS | Bool is identity for promotion |
| U8 | I8 | I16 | Smallest signed type covering 0..255 and -128..127 |
| U16 | I16 | I32 | Smallest signed type covering both ranges |
| U64 | any signed | F64 | No NumPy integer type holds both `U64::MAX` and negative values |
| F16 | I8 / U8 | F16 | NumPy preserves float16 for 8-bit ints |
| F16 | I16 / U16 | F32 | 16-bit integers exceed float16 mantissa |
| F32 | I32 / I64 | F64 | float32 can't represent all 32/64-bit ints |
| Complex64 | I32 | Complex128 | Mirrors the F32+I32 → F64 widening rule |

The `U64 + signed → F64` promotion is the most counterintuitive rule in the table, but it is exactly what NumPy does: there simply isn't a 128-bit integer type in NumPy's type system.

**Cast policy.** `can_cast(src, dst, casting)` supports all five NumPy casting modes: `"no"`, `"equiv"`, `"safe"`, `"same_kind"`, and `"unsafe"`. A `same_kind` cast cannot silently demote signed to unsigned.

**IntegerSidecar** (defined in `fnp-ufunc`, not `fnp-dtype`; it travels with `UFuncArray`'s storage rather than with the dtype itself)**.** For arrays whose `UFuncArray` storage is `Vec<f64>` but whose dtype is i64/u64, a sidecar preserves exact integer values through storage round-trips so values above 2^53 survive `from_storage` / `to_storage`. Arithmetic on such values still uses f64 approximation (see Limitations).

**Promotion table at a glance.** All 324 (dtype × dtype) pairs are individually enumerated. The two-dimensional table is roughly symmetric and follows NumPy's "smallest type that holds both ranges, prefer floats over signed-unsigned widening that would lose precision" rule:

```
        Bool   I8    I16   I32   I64   U8    U16   U32   U64   F16   F32   F64   C64   C128
Bool    Bool   I8    I16   I32   I64   U8    U16   U32   U64   F16   F32   F64   C64   C128
I8       I8    I8    I16   I32   I64   I16   I32   I64   F64   F16   F32   F64   C64   C128
I16     I16   I16    I16   I32   I64   I16   I32   I64   F64   F32   F32   F64   C64   C128
I32     I32   I32    I32   I32   I64   I32   I32   I64   F64   F64   F64   F64  C128   C128
I64     I64   I64    I64   I64   I64   I64   I64   I64   F64   F64   F64   F64  C128   C128
U8       U8   I16    I16   I32   I64   U8    U16   U32   U64   F16   F32   F64   C64   C128
U16     U16   I32    I32   I32   I64   U16   U16   U32   U64   F32   F32   F64   C64   C128
U32     U32   I64    I64   I64   I64   U32   U32   U32   U64   F64   F64   F64  C128   C128
U64     U64   F64    F64   F64   F64   U64   U64   U64   U64   F64   F64   F64  C128   C128
F16     F16   F16    F32   F64   F64   F16   F32   F64   F64   F16   F32   F64   C64   C128
F32     F32   F32    F32   F64   F64   F32   F32   F64   F64   F32   F32   F64   C64   C128
F64     F64   F64    F64   F64   F64   F64   F64   F64   F64   F64   F64   F64  C128   C128
C64     C64   C64    C64  C128  C128   C64   C64  C128  C128   C64   C64  C128   C64   C128
C128   C128  C128   C128  C128  C128  C128  C128  C128  C128  C128  C128  C128  C128   C128
```

Read across the row for the left-hand operand and down to the column for the right-hand operand. The table is symmetric (`promote(a, b) == promote(b, a)`). Key patterns visible in the matrix:

- **Signed × unsigned cross.** The result is the smallest signed type whose range covers both operands, so `U8 × I16 = I16`, but `U8 × I8 = I16` (no single 8-bit type holds both `-128..127` and `0..255`).
- **U64 × any signed = F64.** No NumPy integer type holds both `U64::MAX` and negative values, so the result widens to float.
- **F32 × {I32, I64, U32, U64} = F64.** F32's 24-bit mantissa cannot exactly represent all 32/64-bit integers; F64 has 53 bits, which covers everything up to ±2^53.
- **F16 × {I8, U8} = F16.** F16's 11-bit mantissa is enough for 8-bit ints; F16 × 16-bit ints widens to F32; F16 × 32/64-bit ints jumps to F64.
- **Complex × small int = Complex64**, **Complex × large int = Complex128.** The same mantissa-precision logic as the float rows.

(`Str`, `DateTime64`, `TimeDelta64`, and `Structured` follow same-family-only rules and are not shown in this 14×14 numeric sub-table.) The full enumeration is in `crates/fnp-dtype/src/lib.rs`; if a new variant is added to `enum DType`, the compiler refuses to build until the new row and column are filled in. There is no catch-all arm.

### Stride Calculus Engine (`fnp-ndarray`)

SCE owns all shape transformation rules and is the correctness backbone of the entire project. Every other crate flows through it.

1. **Shape → strides.** Given a shape `[d0, d1, ..., dn]` and `MemoryOrder::C` or `F`, compute the contiguous strides. C-order: rightmost dimension has stride = item_size, each dimension to the left multiplies by the next dimension's size. F-order: leftmost dimension has stride = item_size.
2. **Broadcast legality.** Two shapes are broadcast-compatible when, aligned from the right, each pair of dimensions is either equal or one is 1. The output shape takes the maximum at each position. Mixed-rank inputs are left-padded with 1s. `broadcast_shapes(...)` reduces over a slice of shapes deterministically.
3. **Reshape with `-1` inference.** At most one dimension can be `-1`, meaning "infer from the total element count and the other dimensions." `fix_unknown_dimension` divides the total element count by the product of known dimensions and validates that the result is exact.
4. **View safety.** `as_strided()` and `sliding_window_view()` compute the required byte span from the minimum to maximum reachable offset and verify it fits within the base allocation; out-of-bounds requests are rejected instead of zero-filled. Negative strides (reverse slicing) are fully supported across multiple axes.
5. **Broadcast strides.** When a dimension has size 1 but the broadcast output has size > 1, the stride for that dimension is set to 0. This creates a virtual repeat without copying data.

### Transfer Semantics and Iteration (`fnp-iter`)

The iterator crate models NumPy's internal transfer-loop selector, the `nditer` state machine, and a parity bridge that lets us step through a real numpy nditer alongside ours.

**Transfer classes:**

| Class | When selected | Cost |
|---|---|---|
| `Contiguous` | Both src/dst have unit strides and matching alignment | Fastest (memcpy-like) |
| `Strided` | Arbitrary strides, lossless cast | Medium (per-element stride arithmetic) |
| `StridedCast` | Arbitrary strides, lossy cast | Slowest (per-element cast + stride) |

**Overlap actions:**

| Action | When | Why |
|---|---|---|
| `NoCopy` | Source and destination don't overlap | No precaution needed |
| `ForwardCopy` | Overlap, forward iteration safe | Dst starts after src start |
| `BackwardCopy` | Overlap, forward would corrupt | Must iterate in reverse |

**Stateful `Nditer`.** First-class state machine on top of `NditerPlan`: `iterindex`, `multi_index`, `reset`, `seek_by_index`, `seek_by_multi_index`, `external_loop` chunk iteration, plus a NumPy-backed `nditer_python*` bridge for parity checks. `fnp-python` re-exposes the iterator as the PyO3 class `PyNditer`.

**FlatIter indexing** supports four modes: `Single(i)`, `Slice { start, stop, step }`, `Fancy(Vec<usize>)`, and `BoolMask(Vec<bool>)`. The `count_true_mask` optimization processes boolean masks in 8-element chunks for vectorizable counting.

### Ufunc Dispatch (`fnp-ufunc`)

The largest crate (~60k LOC) and the heart of the array engine.

**35 binary operations:** `Add`, `Sub`, `Mul`, `Div`, `Power`, `FloorDivide`, `Remainder`, `Fmod`, `Minimum`, `Maximum`, `Fmax`, `Fmin`, `Copysign`, `Heaviside`, `Nextafter`, `Arctan2`, `Hypot`, `Logaddexp`, `Logaddexp2`, `Ldexp`, `FloatPower`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `LeftShift`, `RightShift`, `LogicalAnd`, `LogicalOr`, `LogicalXor`, and the six comparison operators.

**43 unary operations:** `Abs`, `Negative`, `Positive`, `Sign`, `Sqrt`, `Square`, `Exp`, `Log`, `Log2`, `Log10`, all trig and inverse trig, all hyperbolic and inverse hyperbolic, `Cbrt`, `Expm1`, `Log1p`, `Degrees`, `Radians`, `Floor`, `Ceil`, `Round`, `Rint`, `Trunc`, `Reciprocal`, `Spacing`, `Signbit`, `Isnan`, `Isinf`, `Isfinite`, `LogicalNot`, `Invert`, and more.

Every binary operation goes through the same execution skeleton: compute the output shape via SCE's `broadcast_shape`, map output indices to source indices using broadcast strides (0-stride for size-1 dimensions), and apply the operation elementwise. The output-to-source index map uses an incremental odometer instead of a full unravel/remap per element.

**NaN semantics are explicit.** A systematic correctness sweep eliminated implicit "NaN is just `partial_cmp` returns None" behaviors:

- `reduce_min`, `reduce_max` propagate NaN (custom `nan_min` / `nan_max`, not `f64::min` / `f64::max`)
- `cummin`, `cummax` propagate NaN through the accumulator
- `sort`, `argsort` sort NaN to the end via `nan_last_cmp` (not `partial_cmp().unwrap_or(Equal)`)
- `median`, `percentile`, `quantile` early-return NaN before sorting
- `ptp` propagates NaN in both `UFuncArray` and `MaskedArray`
- Mode, heapsort, set operations all propagate NaN consistently

**Float error state.** A thread-local `FloatErrorState` tracks divide-by-zero, overflow, underflow, and invalid-operation events. Each mode (`Ignore`, `Warn`, `Raise`, `Call`, `Print`, `Log`) is configurable per category. `seterr()`, `geterr()`, and `errstate()` match NumPy's API; `errstate()` returns an RAII guard that restores prior state on drop.

**Generalized ufuncs.** `GufuncSignature` parses signatures like `"(n?,k),(k,m?)->(n?,m?)"` for sub-array operations (matmul, etc.). The parser normalizes signatures, extracts core dimensions, validates operand shapes, and applies broadcasting over the remaining "loop" dimensions.

**einsum.** Three entry points: `einsum()` for direct evaluation, `einsum_path()` for contraction-path optimization, and `einsum_optimized()` for greedy or brute-force strategy selection.

| Strategy | Method | Best for |
|---|---|---|
| `"greedy"` | At each step, contract the pair with smallest intermediate result | Default, fast for any operand count |
| `"optimal"` | Dynamic programming over all contraction orderings | ≤ 10 operands |
| `"auto"` | Selects greedy | Convenience alias |

### Linear Algebra (`fnp-linalg`)

**~100 public functions** organized into four tiers:

**2×2 fast paths.** `solve_2x2`, `det_2x2`, `slogdet_2x2`, `inv_2x2`, `qr_2x2`, `svd_2x2`, `eigh_2x2`, `cholesky_2x2` bypass the general NxN overhead for the smallest matrix size.

**NxN general algorithms.**
- **QR:** Householder reflections with `qr_nxn` (reduced) and `qr_mxn` (rectangular)
- **SVD:** Golub–Kahan bidiagonalization + implicit shifted QR; `svd_full(full_matrices)` for reduced or full
- **Eigenvalues:** Hessenberg reduction + implicit shifted QR iteration; symmetric path uses tridiagonal reduction + implicit QL shifts (`eigvalsh_nxn`, `eigh_nxn`)
- **LU:** Partial pivoting with `lu_factor_nxn` and `lu_solve`
- **Cholesky:** Column-wise lower-triangular factorization
- **Least squares:** `lstsq_svd` and `lstsq_nxn`; returns the full NumPy 4-tuple `(x, residuals, rank, singular_values)`

**Spectral methods.** `expm_nxn` (matrix exponential via Padé approximation), `sqrtm_nxn`, `logm_nxn`, `funm_nxn` (general matrix function via Schur decomposition), `polar_nxn`, `schur_nxn`.

**Batch operations.** 14 batched functions (`batch_inv`, `batch_det`, `batch_solve`, `batch_svd`, `batch_qr`, `batch_cholesky`, `batch_eig`, `batch_eigvalsh`, `batch_eigh`, `batch_slogdet`, `batch_svd_full`, `batch_matrix_norm`, `batch_matrix_rank`, `batch_trace`) operate on stacked matrices with leading batch dimensions, matching NumPy's broadcasting for linear algebra.

**Complex number support.** 10+ functions for complex-valued matrices: `complex_solve_nxn`, `complex_det_nxn`, `complex_inv_nxn`, `complex_cholesky_nxn`, `complex_qr_mxn`, `complex_matmul`, `complex_matvec`, `complex_conjugate_transpose`, `complex_matrix_norm_frobenius`, `complex_trace_nxn`.

Eigenvalue sort order is ascending to match NumPy. Singularity checks use exact zero (not epsilon thresholds) where NumPy does. Non-finite inputs propagate cleanly through `cond_p`, 2D `cross_product`, and rectangular NxN norms.

### Random Number Generation (`fnp-random`)

The RNG crate achieves **bit-exact parity with NumPy** by porting every algorithm from NumPy's C source code. **`fnp-random` has zero external `crates.io` dependencies**; it depends only on `fnp-ndarray` within the workspace.

**5 production bit generators**, enumerated under `pub enum BitGeneratorKind`:

| Generator | State | Use |
|---|---|---|
| `PCG64DXSM` | 128-bit + 128-bit increment, DXSM output | **Default**; bit-exact NumPy parity reference |
| `PCG64` | 128-bit state, classic PCG output | Compatible with `numpy.random.Generator(PCG64(seed))` |
| `MT19937` | 624-word state, Matsumoto–Nishimura twist | Legacy `RandomState` compatibility |
| `Philox` | 256-bit key + 256-bit counter, 4×64 rounds | Counter-based parallel streaming |
| `SFC64` | Small Fast Counting (256-bit state) | Lightweight high-throughput generator |

An internal `DeterministicRng` (SplitMix64-based) is used in tests only.

**SeedSequence.** Hierarchical seeding with `spawn` / `generate_state` contracts. `spawn(n)` creates child SeedSequences by incorporating a monotonic spawn counter into the entropy pool; `generate_state(words)` cycles through the pool with hash transformations to produce initialization vectors. This exactly matches NumPy's `numpy.random.SeedSequence`. A `SeedSequenceSnapshot` allows snapshot/restore for lineage tracking.

**Bounded integers.** NumPy's `random_bounded_uint64()` dispatch is reproduced exactly:
- Range fits in 32 bits → Lemire's method via buffered `next_uint32()` (each `u64` is split into two `u32`s, low first)
- Range exceeds 32 bits → Lemire's method with 128-bit multiplication
- Range = 0 → return 0; range = `u32::MAX` → raw `next_uint32()`; range = `u64::MAX` → raw `next_u64()`

**Shuffle / permutation.** `random_interval()` (masked bit rejection, not Lemire), matching NumPy's `_shuffle_raw` code path: for each index from n-1 down to 1, generate a uniform integer in `[0, i]` by masking to the smallest bit-width covering `i` and rejecting values above `i`.

**Algorithm correspondence:**

| Algorithm | NumPy source | Our implementation |
|---|---|---|
| Bounded integers | `random_bounded_uint64()` | Lemire 32/64-bit dispatch + buffered `next_uint32` |
| Binomial | `random_binomial_btpe()` + `_inversion()` | BTPE for large n·p, inversion for small |
| Hypergeometric | `hypergeometric_hrua()` + `_sample()` | HRUA (Stadlober 1989) + direct via `random_interval` |
| Poisson | `random_poisson_ptrs()` + `_mult()` | PTRS (Hörmann 1993) for λ ≥ 10, multiplicative for small |
| Gamma | `random_standard_gamma()` | Marsaglia–Tsang + rejection for shape < 1 + exponential for shape = 1 |
| Zipf | `random_zipf()` | Exact rejection with `Umin` clamping |
| Shuffle | `_shuffle_raw()` | Fisher–Yates via `random_interval()` masked rejection |
| Normal | `rk_gauss` / Ziggurat | Ziggurat sampler (same as NumPy) |

**State serialization.** Full state capture and restoration for reproducibility:

```rust
let payload  = generator.to_pickle_payload();
let restored = Generator::from_pickle_payload(payload)?;
// `restored` produces an identical sequence from this point forward.
```

`GeneratorPicklePayload` captures the bit-generator state (seed, counter, algorithm tag, schema version) and optionally the `SeedSequence` snapshot for spawn-lineage tracking. The `RandomState` wrapper provides legacy `numpy.random.RandomState` API parity.

**Seed material** accepts `None` (deterministic default), `U64(seed)`, `U32Words(vec)`, `SeedSequence`, or direct `State { seed, counter }` for exact restoration. `default_rng(12345)`, `default_rng([1, 2, 3])`, and `default_rng(SeedSequence(42))` all work bit-exactly.

> ⚠️ **`SeedMaterial::None` uses a fixed deterministic seed** (`DEFAULT_RNG_SEED = 0xC0DE_CAFE_F00D_BAAD`) rather than OS entropy. NumPy's `default_rng()` sources fresh OS entropy on each call. This is tracked as the single active row in [`docs/DIVERGENCES.md`](docs/DIVERGENCES.md) (bead `franken_numpy-ucc2o`) pending a decision on whether to add `getrandom` as `fnp-random`'s first external dependency. **Explicit seeds always match NumPy bit-for-bit.**

### I/O Format Handling (`fnp-io`)

**NPY format.** Complete implementation of NumPy's `.npy` binary format:
- Magic prefix `\x93NUMPY` + version (1.0 or 2.0) + header length + Python dict header + padding to 16-byte alignment + raw data payload
- 24 supported dtype descriptors including little/big-endian variants for all integer and float types, complex types, fixed-width byte strings, Unicode strings, object type
- Header parsing validates all three required fields (`descr`, `fortran_order`, `shape`) and rejects unknown keys
- Hardened with `MAX_HEADER_BYTES = 65,536` to prevent allocation bombs

**NPZ format.** ZIP-based container for multiple arrays:
- `savez` (stored) and `savez_compressed` (DEFLATE) via the `flate2` crate
- `MAX_ARCHIVE_MEMBERS = 4,096` and `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 GiB` limits prevent archive bombs
- Each member is a complete `.npy` file inside the ZIP

**Text I/O.** `loadtxt`, `savetxt`, `genfromtxt`, plus `loadtxt_usecols` / `loadtxt_unpack` / `genfromtxt_full` handle delimiter-separated text files with configurable dtypes, missing-value policies, and column selection. `MAX_TEXT_ELEMENTS = 16,777,216` bounds parser memory.

**Binary I/O.** `fromfile`, `tofile`, `fromstring`, `tobytes`, `tostring`, plus typed variants `fromfile_text` / `tofile_text` and `fromfile_complex` / `tofile_complex`.

**Structured dtype I/O.** `parse_structured_descr`, `fromfile_structured`, `tofile_structured`, `save_structured`, `load_structured`.

**Memmap.** `memmap(path, mode)`, `memmap_npy(path, mode)`, `open_memmap` with bounded retry on validation failures.

**Pickle policy.** Object dtype arrays require `allow_pickle=true`, matching NumPy's security posture against arbitrary code execution from untrusted `.npy` files.

### Dual-Mode Runtime (`fnp-runtime`)

The runtime implements a risk-aware decision engine with posterior probability estimation. Three actions:

| Action | When |
|---|---|
| **Allow** | `KnownCompatible` input in Strict mode, or low-risk `KnownCompatible` in Hardened mode |
| **FullValidate** | Hardened mode with elevated risk, or malformed metadata (NaN / out-of-range inputs) |
| **FailClosed** | Unknown semantics or `KnownIncompatible` in any mode |

**Risk scoring with posterior estimation.** Each decision starts with class-specific prior odds (1% incompatibility for `KnownCompatible`, 99% for `KnownIncompatible`, 50% for `Unknown`). A risk score is computed as a log-likelihood ratio of evidence against a threshold. The posterior incompatibility probability selects the action via a loss model:

```
allow_if_compatible:              0.0   (correct accept: no cost)
allow_if_incompatible:          100.0   (silent corruption: catastrophic)
full_validate_if_compatible:      4.0   (unnecessary work: small cost)
full_validate_if_incompatible:    2.0   (caught by validation: acceptable)
fail_closed_if_compatible:      125.0   (false rejection: high cost)
fail_closed_if_incompatible:      1.0   (correct rejection: minimal cost)
```

Every decision is logged to an `EvidenceLedger` with timestamp, mode, class, evidence terms, and action. Override requests are tracked separately in `OverrideAuditEvent` records with audit references.

**Runtime mode matrix:**

| Input class | Strict mode | Hardened mode |
|---|---|---|
| Known compatible + low risk | allow | allow |
| Known compatible + high risk | allow | full_validate |
| Unknown semantics | fail_closed | fail_closed |
| Known incompatible semantics | fail_closed | fail_closed |

### Conformance Infrastructure (`fnp-conformance`)

The conformance crate is the quality backbone. **47 binaries** under `crates/fnp-conformance/src/bin/` drive the layered system. About 20 of the 47 are the active workflow tools (oracle capture, differential runners, drift matrices, ledger gates, API coverage gate, etc.) listed below; the remaining ~27 are one-shot phase2c evidence/optimization-report generators (`generate_packet001_final_evidence_pack` through `generate_packet009_*`, plus a few historical regenerators) that ran during the FNP-P2C-001..009 packet creation and produced checked-in artifacts under `artifacts/phase2c/`.

1. **Differential harness.** Captures NumPy's output for a fixture corpus (`capture_numpy_oracle`), then runs the same inputs through FrankenNumPy (`run_ufunc_differential`) and compares shapes, dtypes, and values with configurable tolerance. Covers ufunc, linalg, FFT, polynomial, string, masked-array, datetime, RNG, and I/O operations.
2. **Metamorphic testing.** Verifies algebraic identities that must hold regardless of input: `a + b = b + a`, `a * 1 = a`, `sum(a) = sum(sort(a))`, FFT round-trips, etc. (13+ identities).
3. **Adversarial fuzzing and seeds.** Tests behavior on hostile inputs: NaN-filled arrays, extreme shapes, denormalized floats, integer overflow, malformed NPY headers, corrupt ZIP EOCDs. **27 fuzz targets** across **7 fuzz crates** with **~200 curated seed corpus files**; see [`docs/FUZZING.md`](docs/FUZZING.md).
4. **Witness stability.** Hard-coded expected values for every RNG distribution ensure code changes don't silently alter output sequences. When an algorithm is intentionally changed, witness values are regenerated from the new implementation.
5. **Diagnostic oracle.** A structured oracle for warnings, exceptions, and printed messages: `run_diagnostic_oracle`, `run_oracle_drift_matrix` (cross-version drift), `run_io_diagnostics`, plus the divergence-ledger checker (`run_divergence_ledger --fail-on-missing`).
6. **API coverage gate.** `run_fnp_python_api_coverage --fail-on-missing` reports `exports=633 covered=599 missing=0` against the full `numpy.__all__` surface plus internal helpers.
7. **RaptorQ durability.** `generate_raptorq_sidecars`, `run_raptorq_gate`, `run_raptorq_stress_gate`. See the RaptorQ section below.
8. **Performance baseline.** `generate_benchmark_baseline`, `run_performance_budget_gate`, `run_cross_engine_benchmark`. See the Performance section.
9. **Security gate.** `run_security_gate`, `run_test_contract_gate`, `run_workflow_scenario_gate`. See the Threat Model section.

### Python Bindings (`fnp-python`)

The `fnp_python` PyO3 extension is the parity-oracle surface. The heavy lifting lives in the engine crates; `fnp_python` wires them to Python and falls back to numpy where there is no engine substitute.

**Three architectural moves drive the 100% surface coverage:**

1. **Engine vs surface separation.** Performance-relevant wrappers (`sum`, `mean`, `var`, `sort`, `partition`, `fft.*`, `linalg.*`, `polynomial.*`) implement the common shape on the Rust engine; they fall back to numpy for unusual kwarg combinations. This preserves drop-in semantics while delivering native speed on hot paths.
2. **Re-export-where-safe.** Submodules whose semantics are pure numpy state (`numpy.strings`, `numpy.char`, `numpy.rec`, `numpy.emath`, `numpy.matrixlib`, `numpy.ma`, `numpy.testing`, `numpy.typing`, `numpy.ctypeslib`, `numpy.core`, `numpy.f2py`) and class instances (`mgrid`, `ogrid`, `s_`, `True_`, …) are re-exported via `m.add(name, &numpy.getattr(name))`. This keeps `fnp_python` identity-equal to numpy on those surfaces with zero maintenance and full upstream parity (including version-gated and deprecation behaviors).
3. **Structural lock-in.** `fnp_python_covers_full_numpy_all` in `crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs` iterates `numpy.__all__` at run time against the live numpy on the build host and fails CI if any name regresses. The guarantee holds as numpy evolves: new `__all__` entries fail the test until explicitly added.

12 PyO3 classes are registered (`Nditer`, `NditerStep`, `FromPyFunc`, `Vectorize`, `SeedSequence`, `Generator`, `RandomState`, `MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`). `mgrid`, `ogrid`, `r_`, `c_` are exposed as live singleton instances. Module-level attributes include `__version__`, `__numpy_version__`, `pi`, `e`, `euler_gamma`, `inf`, `nan`, `little_endian`, the 52 dtype scalars, and all the wired submodules.

#### Python attribute resolution model

When Python code accesses `fnp_python.some_attr` or `fnp_python.submodule.some_func(...)`, lookup follows a deterministic three-tier model. The tier picked for each name is fixed at module-init time and stable across calls:

| Tier | Mechanism | Used for | Example |
|---|---|---|---|
| **1. Native Rust function** | A `#[pyfunction]` exposed through `m.add_function(...)` that drives a Rust engine path (`fnp-ufunc`, `fnp-linalg`, `fnp-random`, …) and falls back to numpy only for unusual kwarg combinations | Performance-relevant hot paths and surfaces where the engine has a real implementation | `sum`, `mean`, `var`, `sort`, `searchsorted`, `digitize`, `where`, `linalg.solve`, `fft.fft`, polynomial families |
| **2. Native PyO3 class** | A `#[pyclass]` registered via `m.add_class::<…>()` or live singleton instance via `m.add(name, instance)` | Classes that wrap Rust state (iterators, generators, seed sequences, grid objects) | `Generator`, `SeedSequence`, `Nditer`, `mgrid`, `ogrid`, `r_`, `c_` |
| **3. Identity-equal numpy re-export** | `m.add(name, &numpy.getattr(name))`; the actual numpy attribute is rebound under the same name | Submodules and attributes whose semantics are pure numpy state and have no engine substitute | `numpy.strings`, `numpy.char`, `numpy.rec`, `numpy.emath`, `numpy.matrixlib`, `numpy.ma`, `numpy.testing`, `numpy.typing`, `numpy.ctypeslib`, `numpy.core`, `numpy.f2py`, plus the various constants and dtype scalars |

Within a tier-1 wrapper, the fast-path-vs-fallback decision is itself tiered:

```
fast_path_for_common_shape(arg_dtype, common_kwargs)
  └── (on success) return Rust engine result
  └── (on shape/dtype mismatch or unsupported kwargs)
       └── call into numpy.<name>(*args, **kwargs) and return that
```

Two consequences:
1. **`fnp_python.some_name is numpy.some_name`** is literally `True` for any name handled in tier 3. There is no wrapper, no copy, no version-skew risk.
2. The fast-path-then-fallback gate is the natural place where the **conformance shards** under `crates/fnp-python/tests/conformance_*.rs` live. Each shard pins the exact kwargs that exercise the fast path, the exact kwargs that fall through, and the dtype combinations along both paths, so a regression on either side fails immediately.

The structural lock-in test `fnp_python_covers_full_numpy_all` runs against the live `numpy.__all__` from the build host, so newly-added NumPy names show up immediately as CI failures rather than silently going unsupported.

---

## Algorithm Catalog

### FFT

Hybrid approach supporting arbitrary input lengths:

- **Power-of-two lengths:** Cooley–Tukey decimation-in-time (DIT). Recursively splits into even/odd subsequences, applies butterfly operations with twiddle factors `exp(-2πi·k/N)`.
- **Non-power-of-two lengths:** Bluestein's chirp-Z transform. Rewrites the DFT as a convolution, zero-pads to the next power of two, uses Cooley–Tukey for the convolution, then extracts the result.

All 18 transforms (`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, plus `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`) build on these two primitives. Multi-dimensional transforms apply 1-D FFTs sequentially along each axis.

### Signal Processing

- **`convolve(a, v, mode)`:** Direct O(n·m) convolution; supports `full` (default, length n+m−1), `same`, and `valid` modes. Flips the kernel and slides it across the input.
- **`correlate(a, v)`:** Cross-correlation implemented as `convolve(a, v[::-1])`.
- **`convolve2d` / `correlate2d`:** Full 2-D convolution with output shape `(h1+h2−1, w1+w2−1)`. `correlate2d` is implemented as `convolve2d` with the kernel reversed along both axes, the same flip-then-convolve relationship as the 1-D pair.

### Numerical Differentiation and Interpolation

**`gradient(f, *varargs)`** computes numerical derivatives with configurable edge handling:

| Position | `edge_order=1` | `edge_order=2` |
|---|---|---|
| Interior | `(f[k+1] − f[k−1]) / 2h` | same (central difference) |
| First element | `(f[1] − f[0]) / h` | `(−3f[0] + 4f[1] − f[2]) / 2h` |
| Last element | `(f[n−1] − f[n−2]) / h` | `(3f[n−1] − 4f[n−2] + f[n−3]) / 2h` |

Non-uniform spacing uses Lagrange polynomial coefficients (3-point at edges, 2-point for `edge_order=1`).

**`diff(a, n)`** computes n-th discrete difference: `diff[i] = a[i+1] − a[i]`, applied n times. Output length decreases by n.

**`interp(x, xp, fp)`** does 1-D piecewise linear interpolation: binary search to find the enclosing interval in `xp`, then `fp[lo]·(1−t) + fp[hi]·t` with `t = (x − xp[lo]) / (xp[hi] − xp[lo])`. Values outside `xp` are clamped to the boundary values.

### Windowing Functions

Five window types for spectral analysis (all returning `[1.0]` for `M ≤ 1`):

| Window | Formula |
|---|---|
| Hamming | `0.54 − 0.46·cos(2π·i / (M−1))` |
| Hanning | `0.5 − 0.5·cos(2π·i / (M−1))` |
| Blackman | `0.42 − 0.5·cos(2π·i / (M−1)) + 0.08·cos(4π·i / (M−1))` |
| Bartlett | `1 − |2·(i − (M−1)/2) / (M−1)|` |
| Kaiser | `I0(β·sqrt(1 − r²)) / I0(β)` (modified Bessel I0 via piecewise rational approx) |

### Histogram and Binning

**`histogram(a, bins)`** supports three automatic bin-count strategies:

| Strategy | Formula |
|---|---|
| `"sturges"` | `ceil(log2(n)) + 1` |
| `"sqrt"` | `ceil(sqrt(n))` |
| `"auto"` | `max(sturges, sqrt)` |

Bin edges are uniformly spaced. Element-to-bin assignment uses O(log bins) binary search per element. `histogram_bin_edges` returns just the edges without counting. `histogramdd` generalizes to N dimensions.

### Padding

**11 pad modes** matching NumPy's `np.pad`:

| Mode | Behavior |
|---|---|
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
|---|---|
| `fv`, `pv`, `pmt` | Closed-form annuity factor `(1+rate)^nper` |
| `nper` | Logarithmic inversion of annuity formula |
| `npv` | Discounted cashflow sum `Σ cf[i] / (1+rate)^i` |
| `irr` | Newton's method (max 100 iterations, tol 1e-12, initial guess 0.1) |
| `mirr` | Separates positive/negative cashflows, applies reinvestment/finance rates |
| `rate` | Newton's method on the annuity equation |
| `ipmt`, `ppmt` | Interest and principal portions derived from `fv` and `pmt` |

### Polynomial Systems

Five polynomial families (Power series, Chebyshev, Legendre, Hermite, Laguerre), with Hermite represented in both **physicist** and **probabilist** normalizations as separate submodules. Counting Hermite's two normalizations as distinct submodules, the table below has six rows, matching NumPy's own `numpy.polynomial.{polynomial, chebyshev, legendre, hermite, hermite_e, laguerre}` enumeration. Each row carries the full evaluation, arithmetic, calculus, fitting, and basis-conversion suite:

| Family | Basis | Key operations |
|---|---|---|
| Power series | `x^n` | `polyval`, `polyder`, `polyint`, `polyfit`, `polymul`, `polyadd`, `polysub`, `polydiv`, `polyroots` |
| Chebyshev | `T_n(x)` | `chebval`, `chebadd`, `chebsub`, `chebmul`, `chebdiv`, `chebder`, `chebint`, `chebroots`, `chebfromroots`, `chebfit`, `cheb2poly`, `poly2cheb` |
| Legendre | `P_n(x)` | `legval`, `legadd`, `legsub`, `legmul`, `legdiv`, `legder`, `legint`, `legroots`, `legfromroots`, `legfit`, `leg2poly`, `poly2leg` |
| Hermite (physicist) | `H_n(x)` | `hermval`, `hermadd`, `hermsub`, `hermmul`, `hermdiv`, `hermder`, `hermint`, `hermroots`, `hermfromroots`, `hermfit`, `herm2poly`, `poly2herm` |
| Hermite (probabilist) | `He_n(x)` | `hermeval`, `hermeadd`, `hermesub`, `hermemul`, `hermediv`, `hermeroots`, `hermefromroots`, `hermefit`, `herme2poly`, `poly2herme` |
| Laguerre | `L_n(x)` | `lagval`, `lagadd`, `lagsub`, `lagmul`, `lagdiv`, `lagder`, `lagint`, `lagroots`, `lagfromroots`, `lagfit`, `lag2poly`, `poly2lag` |

### Masked Arrays

`MaskedArray` wraps a `UFuncArray` with an optional boolean mask and fill value:

```
MaskedArray { data: UFuncArray, mask: Option<UFuncArray>, fill_value: f64, hard_mask: bool }
```

Mask convention: `1.0` = masked (excluded), `0.0` = valid. Matches `numpy.ma`. Operations that reduce masked arrays (`sum`, `mean`, `min`, `max`, `var`, `std`, `median`, `ptp`, `argmin`, `argmax`, `cumsum`, `cumprod`, `count`) skip masked values. Comparison and arithmetic operations propagate masks through `mask_or`. `compressed()` returns a 1-D array of only the unmasked values. `filled(fill_value)` replaces masked values with a fill value.

### Datetime and Timedelta

Datetime and timedelta arrays are **not separate struct types**; they are `UFuncArray` instances with `DType::DateTime64` or `DType::TimeDelta64`, backed by `ArrayStorage::I64`. The time unit (one of `ns`, `us`, `ms`, `s`, `min`, `h`, `D`, `W`, `M`, `Y`) is carried on the dtype via the `DateTimeUnit` enum. Construct them via `UFuncArray::from_datetime_strings(values, unit)` or `from_timedelta_strings(values, unit)`.

Full temporal arithmetic is supported through dispatch on the dtype:

- Datetime − Datetime = Timedelta
- Datetime + Timedelta = Datetime
- Timedelta + Timedelta = Timedelta
- Scalar multiplication of Timedelta

Business-day helpers: `busday_count(start, end)`, `busday_offset(date, offset)`, `is_busday(date)` with Monday–Friday weekday mask.

### String Operations

33 `numpy.char` functions operate elementwise on string arrays: `upper`, `lower`, `capitalize`, `title`, `center`, `ljust`, `rjust`, `zfill`, `strip`, `lstrip`, `rstrip`, `replace`, `find`, `rfind`, `count`, `startswith`, `endswith`, `isnumeric`, `isalpha`, `isdigit`, `isdecimal`, `str_len`, `encode`, `decode`, `translate`, `maketrans`, `partition`, `rpartition`, `split`, `rsplit`, `join`, `expandtabs`, `swapcase`. String `add` concatenates elementwise; string `multiply` repeats.

### Bit Packing

- **`packbits(axis)`:** Packs 8 boolean elements into 1 byte, MSB first. Output length = `ceil(axis_len / 8)`.
- **`unpackbits(axis, count)`:** Unpacks bytes into boolean bits, MSB to LSB. Output length = `axis_len * 8` (or `count` if specified).

### Scimath (Complex-Domain Extensions)

8 `numpy.lib.scimath` functions extend real-valued math to the complex domain for inputs outside the real function's natural domain. Useful in signal processing and physics where negative square roots or out-of-range inverse trig naturally arise.

| Function | Real domain | Extension |
|---|---|---|
| `scimath_sqrt(x)` | x ≥ 0 | Complex sqrt for x < 0 |
| `scimath_log(x)` | x > 0 | Complex log for x ≤ 0 |
| `scimath_log2(x)` | x > 0 | Complex base-2 logarithm |
| `scimath_log10(x)` | x > 0 | Complex base-10 logarithm |
| `scimath_power(x, p)` | x ≥ 0 (non-integer p) | Complex result for negative base |
| `scimath_arccos(x)` | −1 ≤ x ≤ 1 | Complex arccos for \|x\| > 1 |
| `scimath_arcsin(x)` | −1 ≤ x ≤ 1 | Complex arcsin for \|x\| > 1 |
| `scimath_arctanh(x)` | −1 < x < 1 | Complex arctanh for \|x\| ≥ 1 |

---

## Numerical Stability and Precision Notes

"Same answer, mostly" is not enough. A library that aspires to bit-exact or behaviorally-equivalent output has to face concrete numerical-analysis choices.

**Summation strategy.** Reductions use a **two-tier compensated-summation policy** (`reduce_sum_values` in `crates/fnp-ufunc/src/lib.rs`):
- Arrays with `≤ 1,000,000` finite values (or any non-finite value at all) use a straight linear sum. At those sizes naive accumulation has bounded error and the branch prediction wins on cost.
- Arrays larger than `COMPENSATED_SUM_MIN_LEN = 1,000,000` switch to **Neumaier-compensated sum**, a Kahan variant that handles the `|value| > |running sum|` case correctly. Error stays at O(ε) per element instead of O(n·ε), which matters at the sizes where naive summation drifts.

The same selector is used for strided axis reductions (`reduce_sum_strided`), so axis-wise sums on large contiguous chunks get the same compensation treatment. The contiguous-reduction fast path that drove the ~56% p50 latency improvement landed in commit `d9cfe90` (2026-02-13); the proof bundle is under `artifacts/optimization/`.

**Polynomial evaluation.** Power-series, Chebyshev, Legendre, Hermite, and Laguerre evaluations all use Horner's method (or the Clenshaw recurrence for orthogonal bases). This minimizes multiplications and keeps the floating-point error proportional to the polynomial degree rather than to repeated powers of x.

**Catastrophic cancellation.** Where NumPy uses identities to dodge cancellation (e.g. `log1p(x) ≈ x - x²/2 + x³/3` for small x, computed via the FMA-style libc routine), we use the same `f64::ln_1p`, `f64::exp_m1`, etc.; the standard-library backers are themselves engineered to avoid cancellation. `nextafter`, `spacing`, and `signbit` route through libc-equivalent paths so denormals and signed zero are handled identically.

**NaN propagation.** A dedicated oracle-test sweep verifies that NaN propagates through every reduction, every sort, every percentile/median/quantile, every cumulative operation, and every set operation; tests live in `crates/fnp-ufunc/tests/` and the metamorphic suite at `crates/fnp-ufunc/tests/metamorphic_math.rs` (e.g. `partition_contract_binary_mul_preserves_nan_and_signed_zero_bits`, `parallel_opt_in_sum_axis_matches_serial_for_nan_signed_zero_and_empty_axes`). Where NumPy's documented behavior is "NaN sorts to the end," we use the explicit `nan_last_cmp` ordering rather than relying on `partial_cmp().unwrap_or(Equal)` (which is a silent bug).

**Signed zero.** `maximum(-0.0, 0.0)`, `heaviside(0.0, ...)`, `floor_divide` of negative operands, `remainder`, `clip`, `divmod`, and `cummin`/`cummax` all preserve signed-zero semantics matching NumPy. These are explicit fixture cases in `fnp-ufunc/tests`.

**Denormals.** No flush-to-zero. Subnormal inputs to ufuncs produce subnormal outputs, matching NumPy with denormals enabled (which is the default on most platforms). Fuzzer seeds include subnormal corners (smallest positive subnormal, largest subnormal, transition to normal) under `crates/fnp-dtype/fuzz/corpus`.

**Linear algebra tolerances.** Singularity checks for `lu_factor`, `cholesky`, and triangular solve use exact zero (not an epsilon threshold), matching NumPy. Rank determination uses the largest singular value times machine epsilon times max(m, n), the standard SVD-based numerical-rank criterion. Conditioning is computed exactly through the SVD; no rank-1 approximations.

**FFT precision.** Both Cooley–Tukey (power-of-two) and Bluestein chirp-Z (arbitrary length) implementations use `f64` throughout the recursion and synthesis steps. Twiddle and chirp factors are computed directly from `angle.cos()` / `angle.sin()` in double precision; no single-precision table tricks. The Bluestein chirp uses `exp(±i·π·k² / n)` derived inline. An `fft_mul` helper protects against the case where a near-zero twiddle (e.g. `cos(π/2)` evaluating to ~6e-17 instead of 0) multiplies an `Inf` or `NaN` input: the result is forced to `0.0` when the twiddle is within `1e-14` of zero and the other operand is non-finite. This matches NumPy's behavior at multiples of π/2.

**RNG bit-exactness vs distribution shape.** The bit-exact contract holds for the core integer-emission stream (PCG64DXSM, MT19937, Philox, SFC64) and for the eight algorithm families ported verbatim from NumPy's C source (BTPE binomial, HRUA hypergeometric, PTRS Poisson, Marsaglia–Tsang gamma, Ziggurat normal/exponential, Lemire bounded ints, Fisher–Yates shuffle, Zipf rejection). Distribution methods built on top inherit bit-exactness when their algebra is identical.

---

## Determinism Guarantees

Determinism is a graded property; the README is explicit about which kind applies where.

| Surface | Guarantee | What that means |
|---|---|---|
| Shape / stride / broadcast computation | **Bit-deterministic, platform-independent.** | Every `broadcast_shape`, `fix_unknown_dimension`, and `contiguous_strides` is a pure `const fn`-style computation. Same inputs always produce the same outputs, on every platform. |
| Dtype promotion | **Bit-deterministic, exhaustively enumerated.** | All 324 pairs explicit; no platform-dependent fallback path; no compiler-version-dependent inference. |
| RNG output (explicit seed) | **Bit-exact vs NumPy upstream**, platform-independent. | `Generator::from_pcg64_dxsm(seed)` and the eight ported distribution algorithms produce sequences identical to `numpy.random.Generator(PCG64DXSM(seed))`. |
| RNG output (no seed) | **Deterministic** but **diverges from NumPy upstream.** | `SeedMaterial::None` uses fixed default seed `0xC0DE_CAFE_F00D_BAAD`; NumPy uses OS entropy. Tracked in [`docs/DIVERGENCES.md`](docs/DIVERGENCES.md) row `franken_numpy-ucc2o`. |
| Ufunc / reduction values | **Behaviorally equivalent vs NumPy** to within the per-fixture relative tolerance recorded in each differential case. | Many ops coincide bit-for-bit with NumPy; when they don't, the differential gate compares against the explicit `rel_tol` field in the fixture. |
| Linalg outputs | **Behaviorally equivalent vs NumPy** up to tolerance; **bit-deterministic on a single platform**. | QR/SVD/Cholesky/eig values match NumPy within tolerance; on a given platform with a given Rust toolchain, repeating the call always yields the same bits. |
| FFT outputs | **Behaviorally equivalent vs NumPy** up to tolerance; **deterministic** ordering of butterfly operations. | The Cooley–Tukey and Bluestein paths are deterministic; output values match NumPy within the per-fixture relative tolerance configured for the FFT differential cases (`fft_differential_cases.json`). |
| IO round-trip | **Byte-deterministic.** | `save → load → save` is byte-for-byte identical to `numpy.save → numpy.load → numpy.save` for every supported dtype. |
| Runtime decisions / evidence ledger | **Deterministic** for a given input class, mode, and evidence vector. | The `DecisionLossModel` is a fixed set of constants; the posterior estimator is a closed-form Bayesian update; the same inputs produce the same `DecisionAction` and the same `DecisionEvent` field values (modulo the `ts_millis` wall-clock timestamp, which is the only non-determinism in the struct). Byte-determinism on disk depends on the embedder's chosen serializer, since `fnp-runtime` does not ship one (see [Reading the Evidence Ledger](#reading-the-evidence-ledger)). |
| Conformance artifacts (fixtures, oracle outputs, RaptorQ sidecars) | **SHA-256-stable** across runs. | Every artifact bundle is content-hashed; the RaptorQ scrub gate verifies `expected_hash` matches `decoded_hash` (scrub report) and `recovered_hash` (decode proof). Grep `artifacts/raptorq/*.scrub_report.json` and `*.decode_proof.json` to inspect the hash fields directly. |

**What is *not* deterministic:** the order of evaluation inside an `asupersync` task graph (when that optional feature is enabled), the wall-clock timestamp embedded in evidence-ledger entries, and the host-environment fingerprint captured at benchmark time. These are observability metadata, not numerical state.

---

## Complete Distribution List

Random-generation methods available on `Generator`:

- **Continuous:** `beta`, `chisquare`, `exponential`, `f`, `gamma`, `gumbel`, `halfnormal`, `laplace`, `levy`, `logistic`, `lognormal`, `lomax`, `maxwell`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `power`, `rayleigh`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `vonmises`, `wald`, `weibull`
- **Discrete:** `binomial`, `geometric`, `hypergeometric`, `logseries`, `negative_binomial`, `poisson`, `zipf`
- **Multivariate:** `dirichlet`, `multinomial`, `multivariate_normal`, `multivariate_hypergeometric`
- **Uniform:** `random` (float [0,1)), `uniform` (float [low, high)), `integers` (int [low, high))
- **Permutation:** `shuffle` (in-place), `permutation` (copy), `permuted` (axis-aware)
- **Utility:** `bytes`, `choice` / `choice_weighted`
- **State:** `spawn`, `jumped`, `state` / `set_state`

---

## Array Manipulation Toolkit

### Construction

| Function | What it does |
|---|---|
| `zeros(shape)` | Array filled with 0.0 |
| `ones(shape)` | Array filled with 1.0 |
| `full(shape, val)` | Array filled with arbitrary value |
| `empty(shape)` | Allocates and zero-fills (safe Rust forbids uninitialized memory; matches NumPy's observed behavior for freshly allocated arrays) |
| `eye(n, m, k)` | Identity-like matrix with diagonal offset `k` |
| `identity(n)` | Square identity matrix (delegates to `eye`) |
| `diag(v, k)` | 1-D input: construct diagonal matrix. 2-D input: extract diagonal. |
| `arange(start, stop, step)` | Evenly spaced values within an interval |
| `linspace(start, stop, num)` | `num` evenly spaced values including endpoints |
| `logspace(start, stop, num)` | Values evenly spaced on a log scale |
| `geomspace(start, stop, num)` | Values evenly spaced on a geometric scale |
| `meshgrid(x, y, ...)` | Coordinate matrices with `xy`/`ij`, `sparse`, `copy` parity |
| `mgrid[...]` | Dense grid object with NumPy slice semantics |
| `ogrid[...]` | Open grid object with sparse tuple semantics |
| `fromfunction(shape, f)` | Apply closure `f(&[usize]) -> f64` to each multi-index |

### Joining and Splitting

| Function | Axis behavior |
|---|---|
| `concatenate(arrays, axis)` | Join along existing axis |
| `stack(arrays, axis)` | Join along new axis (all arrays must have identical shape) |
| `vstack` / `row_stack` | Stack vertically (axis 0). Promotes 1-D to (1, N). |
| `hstack` | Stack horizontally. For 1-D: axis 0. For N-D: axis 1. |
| `dstack` | Stack along axis 2. Promotes to at least 3-D first. |
| `column_stack` | Stack 1-D arrays as columns of a 2-D array |
| `r_[...]` | Row-wise concatenation object with slice expansion and NumPy directive fallback |
| `c_[...]` | Column-wise concatenation object with 1-D-to-column promotion and NumPy directive fallback |
| `block(grid)` | Assemble from nested grid (concatenates within rows, then stacks) |
| `split(ary, n, axis)` | Split into `n` equal sub-arrays |
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
| `fliplr` / `flipud` | Left-right / up-down reversal |
| `rot90(k)` | Rotate by k·90° on first two axes |
| `roll(shift, axis)` | Circular shift with wrapping |
| `tile(reps)` | Repeat array along each axis per `reps` |
| `repeat(n, axis)` | Repeat each element `n` times |
| `resize(new_shape)` | Resize with cyclic repetition if new shape is larger |

### Advanced Indexing

| Function | What it does |
|---|---|
| `take(indices, axis)` | Select elements by integer indices |
| `put(indices, values)` | Replace flat-indexed elements (cyclic values) |
| `compress(condition, axis)` | Select elements where boolean condition is true |
| `extract(condition, arr)` | Flat extraction by boolean mask |
| `place(mask, vals)` | In-place replacement where mask is true |
| `putmask(mask, values)` | In-place masked scatter with cyclic value repetition |
| `indices(dimensions, dtype)` | Dense grid of coordinate indices |
| `diag(v, k)` | Construct or extract a diagonal with offset `k` |
| `diagflat(v, k)` | Flatten input and place values on a diagonal matrix |
| `diagonal(a, offset, axis1, axis2)` | Extract a diagonal across arbitrary axes |
| `fill_diagonal(a, val, wrap)` | In-place diagonal fill with NumPy `wrap` semantics |
| `ix_(*args)` | Open mesh of broadcastable index vectors |
| `diag_indices(n, ndim)` | Tuple of diagonal index arrays |
| `tril_indices(n, k, m)` | Lower-triangle row/column index tuple |
| `triu_indices(n, k, m)` | Upper-triangle row/column index tuple |
| `select(condlist, choicelist, default)` | Choose from multiple arrays by first-matching condition |
| `piecewise(condlist, funclist)` | Piecewise constant function via condition list |
| `take_along_axis(indices, axis)` | Gather values along axis by index array |
| `put_along_axis(indices, values, axis)` | Scatter values along axis by index array |
| `ravel_multi_index(coords, shape)` | Convert N-D coordinates to flat indices |
| `unravel_index(indices, shape)` | Convert flat indices to N-D coordinates |

---

## Shared Memory and Views

`UFuncArrayView` provides NumPy-style shared-memory views with overlap detection:

```
UFuncArrayView {
    shape: Vec<usize>,
    buffer: Arc<RwLock<Vec<f64>>>,  // shared backing store
    offset: isize,                   // byte offset into buffer
    strides: Vec<isize>,             // per-dimension strides (can be negative)
    writable: bool,
    dtype: DType,
}
```

**Memory overlap detection** uses a two-tier approach:

1. **Fast path (`may_share_memory`):** Checks whether the byte-offset spans of two views overlap. O(ndim) and conservative (may report false positives for non-contiguous views).
2. **Exact path (`shares_memory`):** First checks `Arc` pointer equality (different backing buffers never share). If the same buffer, computes the actual set of accessed byte offsets (up to 200,000) and checks for intersection. Falls back to the fast path if offset collection exceeds the limit.

This supports safe in-place operations: if two views share memory, operations that read from one and write to the other use temporary copies to avoid data corruption.

---

## Float Error State Machine

FrankenNumPy replicates NumPy's floating-point error handling:

```
┌───────────────┐   seterr(divide='raise')   ┌──────────────┐
│ Default       │ ────────────────────────── │ Custom       │
│ divide=Warn   │                            │ divide=Raise │
│ over=Warn     │   errstate(all='ignore')   │ over=Warn    │
│ under=Ignore  │ ────────────────────────── │ under=Ignore │
│ invalid=Warn  │   (RAII guard restores)    │ invalid=Warn │
└───────────────┘                            └──────────────┘
```

Six error modes per category: `Ignore`, `Warn`, `Raise`, `Call` (user callback), `Print` (stderr), `Log` (event buffer).

Four error categories: divide-by-zero, overflow, underflow, invalid operation.

`errstate()` returns an RAII guard that automatically restores the previous configuration when dropped:

```rust
let _guard = errstate(Some(FloatErrorMode::Ignore), None, None, None, None);
// all float errors ignored in this scope
// previous state restored when _guard drops
```

---

## Error Taxonomy

Every operation that can fail returns `Result<T, E>` with an explicit error type; there are no hidden panics on user-reachable paths. The exact variants per crate (verified against `crates/fnp-*/src/lib.rs`):

| Crate | Error type | Variants |
|---|---|---|
| `fnp-ndarray` | `ShapeError` | `InvalidItemSize`, `InvalidDimension(isize)`, `MultipleUnknownDimensions`, `Overflow`, `RankMismatch { expected, actual }`, `InvalidWindowDimension { axis, window, dim }`, `OutOfBoundsView { required_nbytes, available_nbytes }`, `IncompatibleBroadcast { lhs, rhs }`, `IncompatibleElementCount { old, new }` |
| `fnp-dtype` | `StorageError` | `IndexOutOfBounds { index, len }`, `UnsupportedCast { from, to }`, `StructuredFieldMismatch { expected, got }` |
| `fnp-iter` | `TransferError` | `SelectorInvalidContext(&'static str)`, `OverlapPolicyTriggered(&'static str)`, `WhereMaskContractViolation(&'static str)`, `SameValueCastRejected`, `StringWidthMismatch(&'static str)`, `SubarrayBroadcastContractViolation(&'static str)`, `FlatiterReadViolation(&'static str)`, `FlatiterWriteViolation(&'static str)`, `NditerOverlapPolicy(&'static str)`, `FpeCastError(&'static str)` |
| `fnp-iter` | `FlatIterContractError` | `IndexingViolation(&'static str)` |
| `fnp-iter` | `NditerError` | `InvalidConfiguration(&'static str)`, `MultiIndexViolation(&'static str)`, `NoBroadcastViolation(&'static str)`, `OverlapPolicyTriggered(&'static str)`, `NdindexShapeValidation(&'static str)`, `PythonBridgeFailure(String)` |
| `fnp-ufunc` | `UFuncError` | `Shape(ShapeError)`, `InvalidInputLength { expected, actual }`, `AxisOutOfBounds { axis, ndim }`, `EmptyReduction { op }`, `FloatingPoint { kind, detail }`, `SignatureConflict { sig, signature }`, `SignatureParse { detail }`, `FixedSignatureInvalid { detail }`, `OverridePrecedenceViolation { detail }`, `DispatchResolutionFailed { detail }`, `TypeResolutionInvalid { detail }`, `ReductionContractViolation { detail }`, `LoopRegistryInvalid { detail }`, `PolicyUnknownMetadata { detail }`, `Msg(String)` (catch-all for one-off parameter errors) |
| `fnp-ufunc` (masked) | `MAError` | `MaskShapeMismatch { data_shape, mask_shape }`, `UFunc(UFuncError)` (auto-wraps via `From`), `Msg(String)` |
| `fnp-linalg` | `LinAlgError` | `ShapeContractViolation(&'static str)`, `SolverSingularity`, `CholeskyContractViolation(&'static str)`, `QrModeInvalid`, `SvdNonConvergence`, `SpectralConvergenceFailed`, `LstsqTupleContractViolation(&'static str)`, `NormDetRankPolicyViolation(&'static str)`, `BackendBridgeInvalid(&'static str)`, `PolicyUnknownMetadata(&'static str)` |
| `fnp-random` | `RandomError` | `InvalidUpperBound`, `InvalidParameter` |
| `fnp-random` | `SeedSequenceError` | `GenerateStateContractViolation`, `SpawnContractViolation` |
| `fnp-random` | `BitGeneratorError` | `GeneratorBindingInvalid(&'static str)`, `InitFailed(&'static str)`, `SpawnContractViolation(&'static str)`, `JumpContractViolation(&'static str)`, `StateSchemaInvalid(&'static str)`, `PickleStateMismatch(&'static str)` |
| `fnp-random` | `RngConstructorError` | `SeedMetadataInvalid` |
| `fnp-random` | `RandomPolicyError` | `UnknownMetadata` |
| `fnp-io` | `IOError` | `MagicInvalid`, `HeaderSchemaInvalid(&'static str)`, `DTypeDescriptorInvalid`, `WriteContractViolation(&'static str)`, `ReadPayloadIncomplete(&'static str)`, `PicklePolicyViolation`, `MemmapContractViolation(&'static str)`, `LoadDispatchInvalid(&'static str)`, `NpzArchiveContractViolation(&'static str)`, `PolicyUnknownMetadata(&'static str)` |
| `fnp-runtime` | (return values, not a single enum) | `DecisionAction::FailClosed` is the rejection signal; the rejection reason travels alongside it on the enclosing `DecisionEvent` via `reason_code: String`. `OverrideAuditEvent` records explicit human bypass requests. |

Four design points:

1. **Each crate uses a `reason_code()` method** on its error type (where applicable) that returns a stable `&'static str` slug: `linalg_solver_singularity`, `io_pickle_policy_violation`, `transfer_nditer_overlap_policy`, etc. The runtime evidence ledger and external monitoring systems use these to alert on specific failure modes without parsing display messages.
2. **`&'static str` detail messages are the dominant carrier for context.** The variant tells you the error *category*, the static string tells you the *specific contract that was violated*. The string is stable across patches and safe to assert against in tests.
3. **`UFuncError::Msg(String)` is the catch-all for one-off parameter errors** (off-by-one ranges, malformed kwargs) that don't yet have a dedicated structured variant. Most failures hit a structured variant; only the long tail of unstructured user-input checks lands in `Msg`. Callers should match `Err(UFuncError::Shape(_))` first, then the other structured variants, then `Msg` last.
4. **Error types compose through `From` impls.** A `ShapeError` from SCE flows up into `UFuncError::Shape(...)` automatically. A `UFuncError` from a ufunc call flows up into `MAError::UFunc(...)` for masked-array operations. You don't need to write conversion glue.

No public function in the implementation crates returns `Result<T, Box<dyn Error>>` or `anyhow::Error`. The crate boundary is part of the type contract.

---

## Threading and Concurrency Model

FrankenNumPy operations are single-threaded by default. The choice merits precise description.

**`Send` / `Sync` for the core types.** `UFuncArray` owns its `Vec<f64>` and is `Send + Sync`; you can move an array between threads or share it behind an `Arc` for concurrent reads. `MaskedArray` and `StringArray` follow the same pattern. Datetime/timedelta arrays are `UFuncArray` instances and inherit the same `Send + Sync`.

**`UFuncArrayView` is `Send + Sync` only when read-only.** A view's backing store is `Arc<RwLock<Vec<f64>>>`; concurrent readers can hold a shared `RwLockReadGuard`. A writer acquires the exclusive lock for the duration of its mutation.

**`Generator` and `BitGenerator` are effectively single-thread.** Their inner state is plain integers (`u128` for PCG, the MT19937 state array, etc.), so the auto-derived `Send` and `Sync` impls apply, but every method that draws a value takes `&mut self`. Two threads cannot draw from the same generator without external locking, and you would not want them to: the draw order would become non-deterministic and the bit-exact-vs-NumPy contract would break. The right pattern for parallel RNG work is `SeedSequence::spawn(n)`, which hands each worker an independent child stream that remains individually reproducible. NumPy documents the same pattern.

**`fnp-runtime`'s `EvidenceLedger` has no interior locking.** It is `pub struct EvidenceLedger { events: Vec<DecisionEvent> }`, a plain `Vec` behind `&mut self` mutators. Multi-threaded appenders must wrap the ledger in their own `Mutex` / `RwLock`. The crate does not impose a synchronization model; that choice is left to the embedder.

**No global mutable state in numeric ops.** The one exception is `fnp-ufunc`'s thread-local `FloatErrorState`, which is, as the name implies, per-thread. Configuring `errstate(divide=Raise)` on one thread does not affect another thread.

**No parallel array kernels.** No internal Rayon / SIMD / thread-pool dispatch inside reductions, broadcasts, or matmul. The `reduce_sum_parallel` and `elementwise_binary_parallel` methods exist as opt-in entry points, but the default execution is serial. Three conformance bins help decide when to opt in: `run_parallel_calibration_matrix` (sweeps workload sizes), `run_parallel_speedup_verdict` (per-op verdict), and `run_parallel_thread_recommendations` (per-machine thread-count suggestion) — all under `crates/fnp-conformance/src/bin/`. Multi-threaded execution by default is a Phase 3 candidate (ADR-001).

**Async story is observability-only.** When the optional `asupersync` feature is enabled in `fnp-runtime`, it powers RaptorQ encoding, telemetry channels, and cancellation-safe oracle capture; it does **not** schedule numerical kernels.

---

## Versioning and Compatibility Promises

| Surface | What we promise |
|---|---|
| Workspace version | `0.1.0` pre-release, no semver promises yet. The crates.io publish-readiness metadata (description, repository, keywords, categories, license-file) is in place; the publish itself is gated on the explicit "ship a tag" decision. |
| `numpy.__all__` parity | Tracked against the **live numpy on the build host**, whatever that version is. The structural lock-in test (`fnp_python_covers_full_numpy_all`) catches any new name that numpy adds to `__all__`. New names fail CI until explicitly added to the re-export block. |
| RNG bit-exactness | Promised vs **PCG64DXSM** specifically, the algorithm NumPy 1.20+ ships as its high-quality default. Other bit generators (PCG64, MT19937, Philox, SFC64) match their upstream NumPy counterparts at the wire-stream level. |
| `.npy` / `.npz` round-trip | Promised for NPY 1.0 and 2.0 formats with every supported dtype. NumPy 3.0 will introduce a new format version; FrankenNumPy will follow once the format is finalized. |
| Rust toolchain | Pinned to `nightly-2026-02-20` in both `rust-toolchain.toml` and `.github/workflows/ci.yml` (`env.RUST_TOOLCHAIN`). Bumps are scheduled, coordinated, and CI-verified before merge. |
| Edition | Rust 2024. |
| MSRV vs MSRRust | The minimum is also the maximum. We pin a specific nightly rather than supporting a range, because some used features (`let-chains`, certain `const fn` capabilities) graduated through nightly during the project's lifetime. |
| Public Rust API | Will likely receive a major reshape before `0.2.0`. Treat 0.x as exploratory; do not load-bear on `UFuncError` variant names or on undocumented method signatures yet. |
| Python `fnp_python` API | Identical to the live numpy surface at build time. Code that runs against `numpy` will run against `fnp_python` with `import fnp_python as np`. |

---

## Reproducibility Recipe

A concrete checklist for "make my numerical pipeline bit-reproducible from a fresh checkout, on any compatible machine, today and three years from now."

1. **Pin the toolchain.** Add `rust-toolchain.toml` with a specific nightly. We use `nightly-2026-02-20`.
2. **Pin every dependency.** Use exact `=x.y.z` constraints in `Cargo.toml`, not caret/tilde. Commit `Cargo.lock`.
3. **Use an explicit RNG seed.** `Generator::from_pcg64_dxsm(seed)` for new code; never rely on `SeedMaterial::None`.
4. **Spawn child streams for parallelism, not OS entropy.** `let mut parent = SeedSequence::new(&[seed])?; let children = parent.spawn(n_workers)?;` gives each worker a child stream. The full lineage is captured in the spawn counter, so child indices reproduce.
5. **Capture the full generator state.** Before any non-deterministic side-effect, call `generator.to_pickle_payload()` to obtain a typed snapshot struct; `from_pickle_payload` reconstructs the *exact* state in-process. For cross-process persistence you currently need to layer your own transport over the payload's public fields.
6. **Use `errstate` rather than global `seterr` for short-lived overrides.** Global `seterr` is process-wide; `errstate` is RAII-scoped and restores prior state on drop.
7. **Round-trip via `save` / `load` for canonical output.** The NPY format is byte-stable across runs; pickle and JSON formats are not.
8. **Hash your inputs and outputs.** SHA-256 every fixture and every artifact. The `fnp-conformance` artifact-durability stack does this for you, but the pattern is portable.
9. **Record the environment fingerprint** (Rust toolchain, NumPy version if relevant, CPU model). Two reproducibility runs are only comparable when their fingerprints match.

Following all nine steps gives you a pipeline whose output is bit-identical between any two runs on compatible hardware, including across the project's `0.x → 1.0` evolution: the RNG, dtype, and shape contracts are stable even when the API surface moves.

---

## Multi-Agent Development Process

FrankenNumPy was built and continues to evolve under a multi-agent workflow. The tooling is documented here because it is visible in the repo (`.beads/`, `.ntm/`, `.claude/`, etc.) and because the methodology is reusable.

| Tool | Purpose |
|---|---|
| **`br` (beads_rust)** | Dependency-aware issue tracker. Every change lands as a closed bead with the issue ID in the commit subject (`[franken_numpy-XXXX] ...`). 1,337 closed beads as of 2026-05-16. Issues live in `.beads/issues.jsonl` (checked in, JSONL-formatted, mergeable). |
| **`bv`** | Graph-aware triage on top of `br`: PageRank, betweenness, critical-path, k-core, cycle detection. Used to pick "ready" work that unblocks the most downstream tasks. |
| **MCP Agent Mail** | Inter-agent messaging plus advisory file reservations. Before editing a file, an agent reserves it with a TTL; conflicts are flagged before anyone wastes work. |
| **`ubs` (Ultimate Bug Scanner)** | Pre-commit static-analysis pass. Catches common Rust bug classes (unwrap-on-Option, integer-overflow patterns, missing-error-handling) before they land. |
| **`rch` (Remote Compilation Helper)** | Offloads heavy builds to a worker fleet. Important for a 304k-LOC workspace where local `cargo build --workspace` is expensive. |
| **`ru`** | Multi-repo orchestration: sync, commit, push across related repos in one pass. |
| **CASS / `cass`** | Cross-agent session search. Lets a fresh agent find what prior agents already learned about the same code, so we avoid re-solving solved problems. |

The pattern that makes this work: every change has a bead ID, every bead has a commit, every commit links back to the bead in its subject line. The graph is complete and queryable. Combined with the four-layer conformance system (differential / metamorphic / adversarial / witness) you get a tight feedback loop: code change → CI gates → bead closure → next ready work.

None of this is required to *use* FrankenNumPy. The crates are stand-alone Rust libraries with no external `crates.io` dependencies on any of the multi-agent tooling. The methodology is documented because the repo doubles as a teaching artifact for the development process.

---

## SCE Invariants in Detail

The Stride Calculus Engine enforces five hard contracts. They are the invariants every other crate depends on, so they merit understanding in detail.

### Invariant 1: element-count conservation

For any shape `s = [d0, d1, …, dn]`, the element count is `prod(d_i)` with checked multiplication. Overflow returns `ShapeError::Overflow`. This is the **first** check in any reshape, broadcast, or view-construction path: a shape that would address more elements than `usize::MAX` is rejected before any allocation is attempted.

```
element_count([2, 3, 4])     = 24
element_count([0, 5])        = 0       (zero is legal: empty axis)
element_count([usize::MAX, 2]) = Err(Overflow)
```

### Invariant 2: stride-from-shape is deterministic

Given `shape` and `MemoryOrder`, `contiguous_strides` returns the unique stride vector for a contiguous layout. The function is a pure `const fn`-style computation: same inputs, same outputs, on every platform, regardless of compiler optimization level. C-order is the default; F-order is selected explicitly.

```
contiguous_strides([2, 3, 4], C) = [12, 4, 1]   (rightmost = 1)
contiguous_strides([2, 3, 4], F) = [1, 2, 6]    (leftmost = 1)
```

### Invariant 3: broadcast legality is symmetric and associative

Two shapes broadcast-compatible iff, aligned from the right, every pair of dimensions is equal or one is `1`. The output shape takes the max at each position. The relation is symmetric: `broadcast_shape(a, b) == broadcast_shape(b, a)`. It is associative: `broadcast_shapes([a, b, c])` reduces deterministically regardless of pairing order.

```
broadcast_shape([3, 1, 5], [4, 5])   = [3, 4, 5]   (right-align, mixed-rank)
broadcast_shape([3, 4], [3, 5])      = Err(IncompatibleBroadcast)
broadcast_shape([], [3])             = [3]         (0-D scalar broadcasts to anything)
```

### Invariant 4: `-1` inference has at most one degree of freedom

`fix_unknown_dimension(shape, total_elements)` accepts a shape with at most one `-1` and infers it from the remaining dimensions. Two `-1`s return `ShapeError::MultipleUnknownDimensions`. A `-1` that doesn't divide evenly returns `ShapeError::IncompatibleElementCount`.

```
fix_unknown_dim([-1, 4], 24)    = [6, 4]
fix_unknown_dim([-1, 5], 24)    = Err(IncompatibleElementCount { old: 24, new: 25 })
fix_unknown_dim([-1, -1], 24)   = Err(MultipleUnknownDimensions)
```

### Invariant 5: view byte-spans must fit in the backing allocation

`as_strided(shape, strides, offset)` and `sliding_window_view(window_shape)` compute the minimum and maximum reachable byte offset and verify the span fits in the buffer. A view that would address beyond the allocation is rejected (`ShapeError::OutOfBoundsView`), not silently zero-filled. Negative strides across multiple axes are handled by the same byte-span check.

These five invariants are why SCE is called the compatibility kernel. Every other crate's correctness rests on them, and the conformance gates verify them on every CI run via the `FNP-P2C-001` (shape/reshape) and `FNP-P2C-006` (stride tricks) packets.

---

## NaN Semantics Cheat Sheet

How each common operation handles NaN, verified against NumPy:

| Operation | Input contains NaN → Result |
|---|---|
| `add`, `sub`, `mul`, `div` | NaN-in → NaN-out (per-element) |
| `min`, `max` (reductions) | **NaN propagates** (uses `nan_min` / `nan_max`, not `f64::min` / `f64::max`) |
| `argmin`, `argmax` | Returns index of first NaN if any; else index of min/max |
| `sum`, `prod`, `mean`, `var`, `std` | NaN propagates (any NaN in → NaN out) |
| `nansum`, `nanmean`, `nanstd`, etc. | NaN-aware: ignores NaN, reduces over remaining |
| `cumsum`, `cumprod` | NaN propagates from first NaN forward |
| `cummin`, `cummax` | NaN propagates through accumulator from first occurrence |
| `sort`, `argsort` | **NaN sorts to end** (uses explicit `nan_last_cmp`, not `partial_cmp().unwrap_or(Equal)`) |
| `partition`, `argpartition` | NaN treated as max; ends up in the upper partition |
| `median`, `percentile`, `quantile` | **Early-return NaN** if any input is NaN |
| `ptp` (peak-to-peak) | NaN propagates (both `UFuncArray` and `MaskedArray`) |
| `unique` | NaN appears once in output (per NumPy convention; multiple NaNs collapse) |
| `isin` | NaN matches NaN (numerically, NaN != NaN, but `isin` uses element equality semantics) |
| `clip` | NaN propagates from value or bounds |
| `where(cond, x, y)` | NaN propagates from selected branch |
| `comparison ops` (==, <, >, ...) | Per IEEE 754: all comparisons with NaN are `False` |
| `logical_and`, `logical_or` | Truthiness of NaN is `True` (matches NumPy boolean coercion) |
| `equal_nan=True` parameters (in `allclose`, `array_equal`) | NaN compares equal to NaN |

The general rule: **arithmetic propagates NaN, reductions propagate NaN unless they have `nan*` variants, sort puts NaN last, and median early-returns**. These are explicit fixture cases in `crates/fnp-ufunc/tests/` and were locked in by a correctness sweep documented in the CHANGELOG ("NaN Propagation" section).

---

## Memory Footprint and Allocation Patterns

### Per-dtype storage cost

The 16 variants of `pub enum ArrayStorage` (in `crates/fnp-dtype/src/lib.rs`); the table below groups the signed/unsigned integer pairs into one row each for brevity:

| DType | `ArrayStorage` variant | Bytes per element |
|---|---|---|
| `Bool` | `Bool(Vec<bool>)` | 1 |
| `I8` / `U8` | `I8(Vec<i8>)` / `U8(Vec<u8>)` | 1 |
| `I16` / `U16` | `I16(Vec<i16>)` / `U16(Vec<u16>)` | 2 |
| `I32` / `U32` | `I32(Vec<i32>)` / `U32(Vec<u32>)` | 4 |
| `I64` / `U64` | `I64(Vec<i64>)` / `U64(Vec<u64>)` | 8 |
| `F16` | `F16(Vec<half::f16>)` | 2 |
| `F32` | `F32(Vec<f32>)` | 4 |
| `F64` | `F64(Vec<f64>)` | 8 |
| `Complex64` | `Complex64(Vec<(f32, f32)>)` interleaved | 8 |
| `Complex128` | `Complex128(Vec<(f64, f64)>)` interleaved | 16 |
| `Str` | `String(Vec<std::string::String>)` (heap-backed) | variable + 24-byte fat pointer per element |
| `Structured` | `Structured(StructuredStorage)` typed by `StructuredField` definitions | variable per record |

`DateTime64` and `TimeDelta64` dtypes are not separate `ArrayStorage` variants; they reuse `ArrayStorage::I64(Vec<i64>)` (8 bytes per element, ticks since epoch), with the time unit (`DateTimeUnit`) carried alongside the dtype. NaT is encoded as `i64::MIN`, matching NumPy.

These are the natural Rust-side storage types; no `Vec<u8>` reinterpretation anywhere. A reader of `crates/fnp-dtype/src/lib.rs` can see exactly where each dtype's bytes live.

### When operations copy vs view

| Operation | Behavior |
|---|---|
| `reshape` to a contiguously-compatible shape | View (zero copy) when possible; copy when strides cannot be expressed |
| `transpose` / `swapaxes` / `moveaxis` | Always a view (just a stride permutation) |
| `broadcast_to` | View with zero strides on the broadcast axes |
| `as_strided`, `sliding_window_view` | Always a view, bounds-checked |
| `flatten` | Always copies (the result is always a fresh contiguous 1-D array) |
| `ravel(order='C')` on a C-contiguous array | View |
| `ravel(order='C')` on a non-contiguous array | Copy |
| `copy()` / `array(a, copy=True)` | Always copies |
| Arithmetic (`a + b`, `a * b`) | Always allocates a new output |
| `reduce_*` methods | Always allocates a new output (typically a scalar or smaller array) |
| `sort` / `argsort` / `partition` | Always allocates (non-mutating) |
| In-place mutation via `MaskedArray::set_*`, `fill_diagonal`, `place`, `putmask` | No copy; mutates in place |

`UFuncArrayView` uses an `Arc<RwLock<Vec<f64>>>` backing store, so multiple views into the same array share one allocation. The overlap-detection machinery prevents an in-place operation from corrupting another view that shares memory.

### Two memory-safety guard rails

1. **`MAX_HEADER_BYTES = 65,536`, `MAX_ARCHIVE_MEMBERS = 4,096`, `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 GiB`, `MAX_TEXT_ELEMENTS = 16,777,216`.** Hard caps on `fnp-io` parser memory usage prevent untrusted inputs from triggering allocation bombs. These are public `pub const` values you can reference from your own code if you want the same bounds applied to layer-above text/binary parsers.
2. **No uninitialized memory anywhere.** `#![forbid(unsafe_code)]` makes `MaybeUninit` and unchecked `Vec::set_len` unreachable. `empty(shape)` allocates and zero-fills; there is no analogue of NumPy's `np.empty` returning uninitialized bytes. The performance cost (zeroing vs not) is small and the safety benefit (no UB from reading uninitialized) is large.

---

## How to Investigate a Parity Issue

A practical runbook for "I think FrankenNumPy and NumPy disagree on something."

1. **Reproduce minimally.** Reduce the input to the smallest array that still shows the divergence. Most parity issues survive on 4-element arrays.
2. **Run the operation under both** with the same explicit RNG seed if any randomness is involved:
   ```rust
   let mut rng = fnp_random::Generator::from_pcg64_dxsm(42)?;
   let fnp_out = rng.<op>(...);
   ```
   ```python
   rng = numpy.random.Generator(numpy.random.PCG64DXSM(42))
   np_out = rng.<op>(...)
   ```
3. **Capture the oracle** via the conformance harness:
   ```bash
   FNP_REQUIRE_REAL_NUMPY_ORACLE=1 cargo run -p fnp-conformance --bin capture_numpy_oracle
   ```
   Add a fixture row to the relevant `*_input_cases.json` so the divergence is reproducible from a clean checkout.
4. **Run the differential gate** for the relevant family:
   ```bash
   cargo run -p fnp-conformance --bin run_ufunc_differential   # for ufunc ops
   ```
   The output names the fixture, the expected value, and the observed value.
5. **Check the divergence ledger** ([`docs/DIVERGENCES.md`](docs/DIVERGENCES.md)):
   ```bash
   cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
   ```
   If the case is already known as `parity_debt` or `intentional`, the ledger row tells you why and what's planned.
6. **If new**, the divergence is one of three things:
   - **Bug in FrankenNumPy:** fix the kernel, regenerate witnesses if applicable, add a regression fixture.
   - **Behavior we want to keep but document:** add a `parity_debt` row to `docs/DIVERGENCES.md` with bead ID and follow-up.
   - **Bug in NumPy:** extremely rare, but it happens. Document with upstream issue link.
7. **Regression-guard.** Whatever the resolution, leave a fixture, a witness, or a ledger row behind so the next agent (or your future self) doesn't re-investigate from scratch.

The whole loop is built so investigation produces a permanent regression test, not just a one-off fix.

---

## NumPy Submodule Coverage Matrix

How each `numpy.*` submodule is covered by `fnp_python`. The "Engine" column says whether the call resolves through a Rust kernel or whether it is a tier-3 identity re-export of the live numpy attribute (see [Python attribute resolution model](#python-attribute-resolution-model)).

| Submodule | Engine | Notes |
|---|---|---|
| `numpy` (top-level) | Mixed | Tier-1 native wrappers for the hot surface (`sum`, `mean`, `var`, `sort`, `searchsorted`, `digitize`, `where`, `argwhere`, `take`, `put`, etc.); tier-3 re-exports for things like `np.True_`, `np.s_`, `np.newaxis` |
| `numpy.fft` | **Native** | All 18 entry points run through the `fnp-ufunc` FFT kernel (Cooley–Tukey for power-of-two, Bluestein chirp-Z otherwise) |
| `numpy.linalg` | **Native** | All decompositions, solvers, norms, batched ops run through `fnp-linalg` (Householder QR, Golub–Kahan SVD, implicit shifted QR for eig, etc.) |
| `numpy.random` | **Native** | `Generator`, `RandomState`, `SeedSequence` and all 5 bit generators are native Rust classes; distributions run through the ported NumPy algorithms with bit-exact PCG64DXSM parity |
| `numpy.polynomial` | **Native** | All 5 families (power, Chebyshev, Legendre, Hermite physicist + probabilist, Laguerre) with full eval / arithmetic / calculus / fitting suites |
| `numpy.ma` | Re-export | Identity-equal to upstream numpy.ma; our own `MaskedArray` is reachable through the engine as well |
| `numpy.char` | Re-export | Identity-equal; plus our own native `StringArray` with the 33 elementwise char functions reachable via the engine |
| `numpy.strings` | Re-export | Re-export of upstream `numpy.strings` |
| `numpy.rec` | Re-export | Re-export of upstream `numpy.rec` (record arrays) |
| `numpy.emath` | Re-export | Re-export of upstream `numpy.emath` (we also expose scimath natively under top-level) |
| `numpy.matrixlib` | Re-export | Re-export of upstream `numpy.matrixlib` |
| `numpy.testing` | Re-export | Re-export of upstream `numpy.testing` |
| `numpy.typing` | Re-export | Re-export of upstream `numpy.typing` |
| `numpy.ctypeslib` | Re-export | Re-export of upstream `numpy.ctypeslib` |
| `numpy.core` | Re-export | Re-export of upstream `numpy.core` (internal compatibility surface) |
| `numpy.f2py` | Re-export | Re-export of upstream `numpy.f2py` |
| `numpy.lib` | Mixed | Re-export by default; specific tier-1 wrappers where we have an engine path |
| `numpy.dtypes` | Re-export | Re-export of upstream `numpy.dtypes` |
| `numpy.exceptions` | Re-export | Re-export of upstream `numpy.exceptions` so exception identity is preserved |

The mixed-tier-3 strategy is what enables the 100% surface guarantee: any name that exists on `numpy.<X>` exists on `fnp_python.<X>` automatically when it is exposed through a re-exported submodule. The structural-lock-in test verifies the same property for the top-level `numpy.__all__`.

---

## Cookbook

Recipes for common Rust-side tasks. These complement the [Quick Example](#quick-example) and [More Worked Examples](#more-worked-examples) sections, focusing on idiomatic patterns rather than feature demos.

### Recipe 1: load an `.npy`, do work, save the result

```rust
use fnp_io::{load, save, IOSupportedDType};
use fnp_ufunc::UFuncArray;
use fnp_dtype::DType;

let bytes = std::fs::read("input.npy")?;
let (shape, values, _dtype) = load(&bytes)?;          // bounded parser, fail-closed

let array = UFuncArray::new(shape, values, DType::F64)?;
let result = array.reduce_mean(Some(0), false)?;     // axis-0 mean

let out_bytes = save(
    result.shape(),                                  // &[usize], borrows from result
    result.values(),                                 // &[f64], borrows from result
    IOSupportedDType::F64,
)?;
std::fs::write("output.npy", out_bytes)?;
```

`load` enforces every bound (`MAX_HEADER_BYTES`, dtype-descriptor validation, payload completeness) before allocating; `save` produces NumPy-compatible bytes that round-trip exactly.

### Recipe 2: parallel RNG with reproducible per-worker streams

```rust
use fnp_random::{BitGenerator, BitGeneratorKind, Generator, SeedSequence};
use std::thread;

let n_workers = 8;
let mut root = SeedSequence::new(&[42u32])?;
let child_seqs = root.spawn(n_workers)?;

let handles: Vec<_> = child_seqs
    .into_iter()
    .enumerate()
    .map(|(worker_id, ss)| {
        thread::spawn(move || -> Result<_, Box<dyn std::error::Error + Send + Sync>> {
            let bg = BitGenerator::from_seed_sequence(BitGeneratorKind::Pcg64Dxsm, &ss)?;
            let mut rng = Generator::from_bit_generator(bg);
            let samples = rng.standard_normal(10_000);
            Ok((worker_id, samples))
        })
    })
    .collect();

for h in handles {
    let (worker_id, samples) = h.join().unwrap()?;
    println!("worker {worker_id}: first sample = {}", samples[0]);
}
```

Each worker stream is statistically independent **and** individually reproducible; re-running with the same root seed produces the same per-worker draw sequences. This is how NumPy itself recommends parallelizing RNG-driven work.

### Recipe 3: differential test against numpy from Rust

```rust
use fnp_ufunc::UFuncArray;
use fnp_dtype::DType;

#[test]
fn fft_round_trip_matches_numpy_within_tolerance() -> Result<(), Box<dyn std::error::Error>> {
    // Build a fixture deterministically.
    let signal_values: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
    let signal = UFuncArray::new(vec![256], signal_values, DType::F64)?;

    // Forward then inverse FFT, should match the original to within FP tolerance.
    let spectrum = signal.fft(None)?;
    let recovered = spectrum.ifft()?;

    for (i, (got, want)) in recovered
        .values()
        .chunks_exact(2)
        .map(|c| c[0])
        .zip(signal.values().iter().copied())
        .enumerate()
    {
        let err = (got - want).abs();
        assert!(err < 1e-12, "fft/ifft round-trip diverged at index {i}: |Δ| = {err:.3e}");
    }
    Ok(())
}
```

The same pattern (build fixture → run kernel → run inverse → compare to original) is the basis for the metamorphic-test layer described in [Conformance Methodology Deep-Dive](#conformance-methodology-deep-dive).

### Recipe 4: call FrankenNumPy from Python (mixed Rust fast-paths + numpy fallback)

```python
import fnp_python as np

# Native Rust fast-path for sorting; identity-equal numpy.ma for masked-array
# operations that do not yet have an engine path.
data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
sorted_view = np.sort(data)                # Rust engine
masked = np.ma.masked_greater(data, 4.0)   # numpy fallback (tier-3)

# Save and round-trip via .npy. Bytes are identical to numpy.save / numpy.load.
np.save("data.npy", data)
restored = np.load("data.npy")
assert (data == restored).all()
```

Hot operations land on the Rust engine. Surfaces with no engine substitute fall back to the same `numpy.ma`, `numpy.matrixlib`, etc. you would have called directly. The boundary is invisible to your code.

### Recipe 5: capture a generator's state mid-stream and resume it later

```rust
use fnp_random::Generator;

let mut rng = Generator::from_pcg64_dxsm(2026)?;
let _first_batch = rng.standard_normal(100);

// Snapshot the exact state into an in-memory payload.
let snapshot = rng.to_pickle_payload();

// Hand the snapshot to another part of your program (or persist it via your
// own transport; fnp-random does not ship a bytes serializer today, but the
// `GeneratorPicklePayload` struct exposes its fields so you can layer one
// on top, or feed it through a serde derive behind your own feature flag).
let mut rng_resumed = Generator::from_pickle_payload(snapshot)?;

let next_from_original = rng.standard_normal(50);
let next_from_resumed  = rng_resumed.standard_normal(50);
assert_eq!(next_from_original, next_from_resumed);
```

The payload carries the bit-generator state plus the (optional) `SeedSequence` snapshot for spawn-lineage tracking. The bit-generator state includes a schema version, so a future generator change that preserves the underlying algorithm can still rehydrate older payloads.

---

## Migrating from NumPy: A Checklist

A practical checklist for Python users moving an existing NumPy-based codebase to `fnp_python`.

| Step | What to do | Why |
|---|---|---|
| 1 | `import fnp_python as np` (replace `import numpy as np`) | `fnp_python` re-exports the full `numpy.__all__`, so the import rename usually suffices. |
| 2 | Run your existing test suite. | Most tests will pass unchanged because the surface is 1:1. |
| 3 | Inspect failures for `is`-comparisons against numpy types (e.g. `type(x) is numpy.ndarray`). | The arrays returned from re-exported numpy functions ARE numpy ndarrays; those returned from Rust-engine fast paths might be the same. If `is`-comparisons break, switch to `isinstance(x, np.ndarray)`. |
| 4 | Pin a specific seed everywhere that uses `np.random.default_rng()`. | `SeedMaterial::None` is currently deterministic in `fnp-random` (divergence ledger row `franken_numpy-ucc2o`); explicit seeds avoid surprise. |
| 5 | Replace single-RNG parallel patterns with `SeedSequence.spawn(n)`. | If you've been calling `np.random.default_rng()` once per worker without seed spawning, you've been relying on OS entropy for stream independence; switch to explicit spawn so the behavior is reproducible AND parallel-safe. |
| 6 | Audit your `.npy` / `.npz` consumers if you handle untrusted files. | The `fnp-io` parsers have hard bounds (`MAX_HEADER_BYTES = 65,536`, `MAX_ARCHIVE_MEMBERS = 4,096`, `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 GiB`). If your existing pipeline relies on loading files larger than these bounds, document the gap before flipping over. |
| 7 | If you use `np.empty()` with the expectation of uninitialized memory for speed, you now get zero-filled memory. | Safe Rust forbids uninitialized memory; the perf delta is small in practice but measurable for very large pre-allocations. |
| 8 | If you use the C-extension API (`np.PyArray_*` from your own C extension), that surface is not exposed by `fnp_python`. | You'd need to keep using upstream numpy for those calls, or to invoke `fnp_python` via Python rather than via the C API. |
| 9 | Cap any `errstate` or `seterr` usage to RAII scopes when possible. | The Rust-side `errstate()` is RAII-scoped per thread. Global `seterr` works but is process-wide and harder to reason about. |
| 10 | Run a parity-test pass: pin a representative input, call the operation under both `numpy` and `fnp_python`, assert equality / closeness. | This gives you a regression net for the migration even if you don't adopt the upstream conformance harness. |

Most codebases need only steps 1, 2, and 4. Steps 5–10 matter if you depend on specific semantics that the migration could perturb.

---

## Common Pitfalls

Things that look broken but aren't, plus things that look fine but bite you later.

| Pattern | What happens | What to do |
|---|---|---|
| `Generator::from_pcg64_dxsm(seed).standard_normal(5)`, calling it twice and expecting the same result | Second call gives the **next** 5 samples, not the first 5 again. The generator is stateful. | Build the generator inside the test/function; or `to_pickle_payload()` + `from_pickle_payload()` to rewind. |
| `default_rng()` without a seed and expecting fresh randomness across runs | The result is **deterministic**; fixed default seed `0xC0DE_CAFE_F00D_BAAD` (divergence ledger row `franken_numpy-ucc2o`) | Pass an explicit seed; or accept the determinism as a feature. |
| `np.empty((1_000_000,))` expecting uninitialized memory for benchmark realism | Result is **zero-filled** (safe Rust forbids uninit memory) | Benchmark realistically by writing into the buffer before timing reads. |
| `reduce_mean(None, false)` on a float array, then asserting shape `[1]` | Result shape is `[]` (0-D scalar), not `[1]`. `keepdims=true` gives `[1, ...]` of all-ones. | Use `keepdims=true` if you want a shape-preserving reduction. |
| Sorting an array containing NaN and expecting NaN in the middle | NaN sorts to **end** (NumPy convention). | Use `nansort_index` style explicit ordering if you need to filter out NaN first. |
| Comparing two NaN values with `==` and getting `true` | `==` returns `false` for NaN (IEEE 754). | Use `f64::is_nan()` or `equal_nan=True` parameter in helpers like `allclose`. |
| Passing a `Vec<f32>` where the API expects a `&[f64]` | Compile error. The API is strongly typed; there is no auto-coercion. | Use `as_slice()` and convert dtypes explicitly via `cast` / `astype`. |
| Calling `binomial(10, 0.5, 100)` and forgetting `?` | Compile error: returns `Result<Vec<u64>, RandomError>`. | Add `?` or `.unwrap()` (test code) / explicit match (production). |
| Building a view with `as_strided(shape, strides, offset)` that exceeds the backing allocation | Returns `Err(ShapeError::OutOfBoundsView { required_nbytes, available_nbytes })`, not UB. | Use the error to debug your stride math; the byte-span check is the safety net. |
| Calling `f64` arithmetic on integer arrays > 2^53 and expecting exact results | f64 arithmetic loses precision above 2^53; `IntegerSidecar` preserves the original bits through storage round-trips but not through arithmetic. | Cast to a wider integer type at the right moment; see Limitations. |
| Assuming `np.linalg.solve(A, b)` is BLAS-backed | It's pure Rust Householder QR / partial-pivot LU. Competitive for small matrices, slower for very large ones. | For massive matrices, Phase 3 BLAS linkage is on the roadmap; for now, batch via `batch_solve` where the broadcast is natural. |

These cover most "wait, why does this behave that way?" questions in the first week of using FrankenNumPy.

---

## Reading the Evidence Ledger

The `EvidenceLedger` in `fnp-runtime` is the audit trail for every runtime decision. If you embed the dual-mode runtime, you get a structured record of *which* input class hit *which* code path with *what* posterior risk. The actual data model (verified against `crates/fnp-runtime/src/lib.rs`):

```rust
pub struct DecisionEvent {
    pub ts_millis: u128,                       // milliseconds since the Unix epoch
    pub mode: RuntimeMode,                     // Strict | Hardened
    pub class: CompatibilityClass,             // KnownCompatible | KnownIncompatible | Unknown
    pub risk_score: f64,                       // log-likelihood ratio of incompatibility
    pub action: DecisionAction,                // Allow | FullValidate | FailClosed (reason travels via `reason_code` below)
    pub posterior_incompatible: f64,           // probability of incompatibility, 0.0..=1.0
    pub expected_loss_allow: f64,              // loss-model term for the Allow branch
    pub expected_loss_full_validate: f64,      // loss-model term for the FullValidate branch
    pub expected_loss_fail_closed: f64,        // loss-model term for the FailClosed branch
    pub selected_expected_loss: f64,           // the minimum of the three (whichever branch we picked)
    pub evidence_terms: Vec<EvidenceTerm>,     // per-evidence log-likelihood-ratio contributions
    pub fixture_id: String,                    // correlates to the fixture, request, or call site
    pub seed: u64,                             // RNG seed in effect (when relevant)
    pub env_fingerprint: String,               // host/CPU/toolchain fingerprint
    pub artifact_refs: Vec<String>,            // pointers to associated artifacts (sidecars, proofs)
    pub reason_code: String,                   // stable slug, mirrors the per-error reason_code() pattern
    pub note: String,                          // free-form annotation
}

pub struct EvidenceTerm {
    pub name: &'static str,                    // e.g. "input.shape_within_bounds"
    pub log_likelihood_ratio: f64,             // negative = pulls toward compatible; positive = toward incompatible
}
```

`EvidenceLedger` itself is a thin `Vec<DecisionEvent>` with `record(...)`, `events() -> &[DecisionEvent]`, and `last()` methods. **There is no built-in JSON/JSONL serializer.** `DecisionEvent` derives only `Debug + Clone + PartialEq`, no serde. Embedders that need on-disk ledgers add their own writer over the public field accessors (the field set is stable across patches).

When you read an event:

- **`mode`** tells you which runtime mode was active. Strict skips most full-validate paths; Hardened opts into them on elevated risk.
- **`class`** is the input classification. `Unknown` and `KnownIncompatible` always fail closed; `KnownCompatible` is the interesting case.
- **`action`** is what we did. Compare against the [Runtime Mode Matrix](#how-it-works-per-crate-deep-dive) to confirm it matches the policy.
- **`risk_score`** is a log-likelihood ratio. Higher means more evidence of incompatibility.
- **`posterior_incompatible`** is the probability of incompatibility under the loss-model decision rule. Values close to 0 mean "safe accept"; values close to 1 mean "definitely incompatible".
- **`expected_loss_{allow, full_validate, fail_closed}`** show the three branches' loss-model evaluations; `selected_expected_loss` is the one that drove the choice. This makes the decision auditable end-to-end (you can see not just *what* we picked, but how the other branches scored).
- **`evidence_terms`** is the breakdown: each `EvidenceTerm.log_likelihood_ratio` adds (negative) or subtracts (positive) from the log-odds. The naming convention is dotted-string slugs like `input.shape_within_bounds` or `input.has_nan`.
- **`fixture_id` / `seed` / `env_fingerprint` / `artifact_refs` / `reason_code`** are the six fields that the Threat Model section requires every audit entry to carry. `note` is free-form context.

Override events live in a separate `OverrideAuditEvent` stream so that explicit human bypasses of a fail-closed gate are auditable distinctly from automatic decisions. An override record's fields are: `ts_millis`, `mode`, `class`, `requested_deviation_class`, `packet_id`, `requested_by`, `reason_code`, `approved: bool`, `action`, `audit_ref`. The `requested_by` + `audit_ref` pair makes the human responsible for each override traceable to a ticket/PR.

For pipelines that need cryptographic non-repudiation, a serialized ledger snapshot can be fed into the same RaptorQ artifact-durability stack (sidecar + scrub + decode proof) so it survives single-symbol loss.

---

## Security Model

FrankenNumPy's security posture covers more than memory safety.

- **Zero unsafe Rust in the workspace today.** 9 of the 10 implementation crates declare `#![forbid(unsafe_code)]`; the 10th (`fnp-python`) does not declare the lint because PyO3's procedural macros may expand to unsafe code as the cdylib entry point is generated. In practice, the current `fnp-python` source contains zero hand-written `unsafe` blocks (verified by ripgrep). The lint is opt-out, not invoked.
- **Fail-closed by default.** Unknown wire formats, unrecognized dtype descriptors, future metadata-version markers, and metadata schema violations cause explicit errors, not silent fallbacks.
- **Bounded resource consumption.** NPY header caps at 64 KB. NPZ archives cap at 4,096 members and 2 GiB uncompressed. Text I/O caps at 16,777,216 elements. Memmap validation retries cap at 64. These prevent denial of service via crafted inputs.
- **Pickle rejection.** Object dtype arrays that could execute arbitrary code during deserialization require explicit `allow_pickle=true`, matching NumPy's security gate.
- **Adversarial conformance.** The security gate (`run_security_gate`) tests exploit scenarios from a versioned threat matrix mapped to specific parser / IO / shape-validation boundaries.
- **No production code mocks or stubs.** An automated audit ([`audit_numpy_mocks.md`](audit_numpy_mocks.md)) shows zero `TODO` / `FIXME` / `HACK` / `STUB` / `unimplemented!()` / `todo!()` anywhere in the 10 production crates, and zero production `.unwrap()` outside `fnp-conformance` fixture-harness code.

---

## Threat Model

12 threat classes are formally mapped in `security_control_checks_v1.yaml`, each with assigned conformance suites, compatibility gates, and override audit policies:

| Threat class | What could go wrong | Control |
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
| `linalg_policy_unknown_metadata` | Unknown linalg-policy metadata silently bypasses fail-closed enforcement | Linalg adversarial suite |
| `corrupt_durable_artifact` | Bit-rot or tampering in stored conformance artifacts | RaptorQ decode proof hash gate |
| `policy_override_abuse` | Unauthorized bypass of compatibility gates | Runtime policy adversarial + explicit audited override |

Every threat log entry must include: `fixture_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `reason_code`.

---

## Phase2C Extraction Packets

The conformance system is organized around 9 extraction packets, each covering one domain of NumPy behavior:

| Packet | Domain | Key contracts |
|---|---|---|
| FNP-P2C-001 | **Shape/reshape** | Element-count conservation, single `-1` dimension, broadcast compatibility |
| FNP-P2C-002 | **Dtype/promotion** | Promotion matrix determinism, safe-cast policy, dtype lifecycle |
| FNP-P2C-003 | **Strided transfer** | Transfer-loop selection, cast pipeline, overlap handling, where-mask assignment |
| FNP-P2C-004 | **NDIter traversal** | Iterator construction, multi-index seek, C/F tracking, external-loop mode |
| FNP-P2C-005 | **Ufunc dispatch** | Signature parsing, method selection, override precedence, gufunc reduction |
| FNP-P2C-006 | **Stride tricks/broadcast** | `as_strided` views, zero-stride propagation, writeability contracts |
| FNP-P2C-007 | **RNG contracts** | Seed normalization, child-stream derivation, deterministic state, jump-ahead |
| FNP-P2C-008 | **Linalg bridge** | Solver contracts, factorization modes, spectral operations, backend dispatch |
| FNP-P2C-009 | **NPY/NPZ IO** | Magic/version validation, header-length bounds, pickle policy, truncated-data detection |

Each packet produces 8 artifact files: `legacy_anchor_map.md`, `contract_table.md`, `fixture_manifest.json`, `parity_gate.yaml`, `risk_note.md`, `parity_report.json`, `parity_report.raptorq.json`, `parity_report.decode_proof.json`. The packet readiness validator (`validate_phase2c_packet`) checks all 8 files exist and contain required fields before a packet is marked `ready`.

---

## Test Coverage

Live counts as of 2026-05-16 (see [`FEATURE_PARITY.md`](FEATURE_PARITY.md) for the authoritative live inventory):

| Crate | Tests | Focus |
|---|---:|---|
| `fnp-ufunc` | 2,191 | Core array ops, math, sorting, polynomials, reductions, oracle tests, linalg bridge, FFT (hfft/ihfft), masked cov/corrcoef, gufunc validation, parameter parity, einsum, NaN/Inf/signed-zero edge cases |
| `fnp-python` | 2,127 | PyO3 surface parity across all 499 `numpy.__all__` names + 133 dedicated conformance shards covering live callable parity, sorter/side bridging, in-place mutation, generator rejection, dtype preservation, etc. |
| `fnp-conformance` | 344 | Differential parity, metamorphic identities, adversarial fuzzing, witness stability, matmul conformance |
| `fnp-random` | 310 | RNG distributions with statistical conformance, `permuted` (1D/2D/axis/deterministic), seeding, reproducibility, large-n binomial/multinomial |
| `fnp-linalg` | 308 | Decompositions, solvers, norms, batch ops, 16 NumPy oracle tests, extreme-scale regression, non-finite parity |
| `fnp-io` | 303 | NPY/NPZ read/write, text formats, compression, 7 format oracle tests, `genfromtxt_full`, `fromfile_text` / `tofile_text` |
| `fnp-dtype` | 257 | Dtype taxonomy, all 324 promotion pairs explicit, cast policy primitives, NumPy byte-width parsing |
| `fnp-ndarray` | 221 | Shape legality, stride calculus, broadcast contracts, overlap detection, multi-axis negative strides, F-order, `required_view_nbytes` |
| `fnp-iter` | 200 | Transfer-loop selector, NDIter traversal/broadcast/overlap contracts, stateful `Nditer` (`iterindex`/`multi_index`/reset/seek/external-loop), flatiter, ndindex |
| `fnp-runtime` | 131 | Mode split, fail-closed decoding, override-audit gate, risk-aware decision engine, evidence ledger |
| **Total** | **6,392** | Sum across all 10 crates |

### Oracle Test Strategy

The oracle suite verifies bit-exact or behaviorally equivalent parity against NumPy across the major subsystems:

- **RNG oracle:** every distribution produces identical output from the same seed, verified against `numpy.random.Generator(PCG64DXSM(12345))`
- **Ufunc edge-case tests:** NaN propagation, empty arrays, Inf arithmetic, boolean dtype promotion, sort ordering
- **Linalg oracle:** `det`, `inv`, `solve`, `eig`, `svd`, `cholesky`, `qr`, `norm`, `cond`, `slogdet`, `lstsq`, rank
- **I/O format tests:** NPY round-trip, magic bytes, header dict format, 16-byte alignment

---

## Conformance Methodology Deep-Dive

The "verify with a real NumPy oracle" claim deserves to be unpacked. The conformance system has **four orthogonal layers** that catch different classes of bug. They are independent: a bug that slips one layer is almost always caught by another.

### 1. Differential testing

The most direct check: for a fixture input, run the same operation under both NumPy and FrankenNumPy, then compare the outputs.

```
capture_numpy_oracle:   fixture → NumPy            → oracle_outputs.json
run_ufunc_differential: fixture → FrankenNumPy     → fnp_outputs.json
                                comparator         → parity_report.json
```

The comparator is tiered: it first compares shapes, then dtypes, then values. Value comparison uses a per-test relative tolerance (default `1e-12` for f64, `1e-6` for f32) but accepts bit-exact equality when both implementations produce identical floats. The same skeleton powers `run_ufunc_differential`, the linalg/FFT/polynomial/string/masked/datetime/RNG/IO differential binaries, and the per-function shards under `crates/fnp-python/tests/conformance_*.rs`.

**Why this catches bugs:** it surfaces any case where a function returns the wrong number, the wrong shape, or the wrong dtype, even when the function "looks right" on inspection.

**Why it isn't sufficient on its own:** a bug that exactly matches a corresponding bug in NumPy will pass differentially. Hence layers 2–4.

### 2. Metamorphic testing

Verifies algebraic identities that must hold *regardless of input*. These don't depend on an oracle at all; they catch bugs that an oracle test couldn't, because they probe relationships, not values.

Representative identities (from `crates/fnp-conformance/src/metamorphic_*`):

| Identity | What it catches |
|---|---|
| `a + b == b + a` (elementwise) | Asymmetric implementation of commutative ops |
| `a * 1 == a` | Multiplicative-identity bugs in dispatch |
| `sum(a) == sum(sort(a))` | Order-dependent accumulator bugs |
| `sum(a) + sum(b) ≈ sum(concat(a, b))` | Reduction kernel reset/restart bugs |
| `transpose(transpose(a)) == a` | View / stride inversion errors |
| `inv(inv(a)) ≈ a` for well-conditioned a | Numerical regression in matrix inversion |
| `fft(ifft(x)) ≈ x` and `ifft(fft(x)) ≈ x` | Inverse-transform symmetry breakage |
| `cumsum(a)[-1] == sum(a)` | Cumulative reduction vs total reduction drift |
| `sort(a)[::-1] == sort(a, descending)` | Sort-order parameter bugs |
| `concat(split(a, n)) == a` | Round-trip bugs in split/concat |
| `cholesky(a).T @ cholesky(a) ≈ a` for SPD a | Triangular factor orientation errors |
| `det(a) * det(b) ≈ det(a @ b)` | Determinant computation drift |
| `eig(a) values == roots of det(a - λI)` (small a) | Eigenvalue solver convergence regressions |

### 3. Adversarial testing

Hostile inputs designed to provoke crashes, panics, or silent corruption. Two complementary mechanisms:

**Curated adversarial fixtures.** Hand-authored corner cases stored under `crates/fnp-*/tests/fixtures/adversarial/`: NaN-filled arrays, ±Inf inputs, denormals at the transition boundary, very large shapes (close to `usize::MAX`), zero-element axes, single-element axes, broadcast shapes that align to the right margin, malformed `.npy` headers (truncated magic, future version, oversized header dict), corrupt ZIP EOCD records.

**Coverage-guided fuzzing.** 7 fuzz crates, 27 targets, ~200 curated seed corpus files (see [`docs/FUZZING.md`](docs/FUZZING.md)). `cargo-fuzz` / libFuzzer drives:

- `fnp-io`: `.npy`/`.npz`/`fromstring`/`loadtxt` parsers
- `fnp-ndarray`: `broadcast_shape`, `fix_unknown_dim`, `as_strided`, `sliding_window_view`
- `fnp-iter`: `ndindex`, `flatiter_indices`, `nditer_plan`, `transfer_class`
- `fnp-dtype`: `dtype_parse`, `can_cast`, `result_type`, `min_scalar_type`
- `fnp-ufunc`: `parse_gufunc_signature`, `parse_fixed_signature`, `datetime_unit_parse`
- `fnp-random`: seed entropy edge cases (`from_u64_seed`, `seed_sequence`)
- `fnp-linalg`: `cholesky_nxn`, `det_nxn`, `qr_mxn` over arbitrary matrices up to 16–32 dim

**Crash discipline.** A fuzz crash never becomes a "fix and move on" event. It becomes (a) a permanent seed file in the corpus so the case is regression-tested forever, (b) a `parity_debt` row in [`docs/DIVERGENCES.md`](docs/DIVERGENCES.md) if it exposes a NumPy-compatibility gap, or (c) a closed bead with a commit pointer if it is a straight bug fix.

### 4. Witness stability

Hard-coded "expected output" values for every RNG distribution and every algorithmic kernel. These pin the *exact* output sequence so that an unintended algorithmic change is caught immediately rather than silently shifting downstream pipelines.

Example: the `standard_normal(5)` witness for `PCG64DXSM(12345)` is the exact 5-element float vector that NumPy produces from the same seed. When we intentionally change an algorithm (e.g. the 2026-03-15 BTPE binomial port), the witness values are *regenerated* from the new implementation and the diff is reviewed in the same commit. That diff is itself an artifact: the old vs new witness arrays appear in the commit, so any future drift is obvious.

### How the four layers compose

```
                         ┌───── differential (oracle equal)  ──── (1)
                         │
   fixture / input  ─────┼───── metamorphic (identity holds) ──── (2)
                         │
                         ├───── adversarial (no crash/UB)    ──── (3)
                         │
                         └───── witness   (exact output bits) ─── (4)
```

A bug that lives in NumPy too will pass (1) but fail (2). A bug that produces a different-but-plausible value will fail (1). A bug that crashes will fail (3). A bug that silently shifts an RNG stream will fail (4). The orthogonality is the point.

---

## CI Gate Topology

Eight ordered gates run from fast to heavy, defined in `.github/workflows/ci.yml`:

```
G1  fmt + lint           cargo fmt --check && cargo clippy -- -D warnings
G2  unit + property      cargo test --workspace
G3  oracle differential  capture_numpy_oracle → run_ufunc_differential (real numpy required)
G4  adversarial+security run_security_policy_gate.sh
G5  test/logging contract run_test_contract_gate.sh
G6  workflow forensics   run_workflow_scenario_gate.sh
G7  performance budget   run_performance_budget_gate.sh
G8  durability/decode    run_raptorq_gate.sh
```

G3 enforces a real-numpy oracle via `FNP_REQUIRE_REAL_NUMPY_ORACLE=1` and rejects `pure_python_fallback` output.

Run all gates locally:

```bash
scripts/e2e/run_ci_gate_topology.sh
```

For manual `cargo check` workspace runs, disable Cargo incremental artifacts to avoid noisy retrieval races under `target/debug/incremental`:

```bash
CARGO_INCREMENTAL=0 cargo check  --workspace --all-targets
CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets -- -D warnings
```

---

## Conformance Pipeline

The oracle capture pipeline runs real NumPy, captures its output, and compares against our implementation:

```bash
# Recommended: require a real NumPy oracle and let the runner bootstrap a repo-local
# .venv-numpy314 with uv if needed.
FNP_REQUIRE_REAL_NUMPY_ORACLE=1 \
  cargo run -p fnp-conformance --bin capture_numpy_oracle

# Run differential comparison
cargo run -p fnp-conformance --bin run_ufunc_differential
```

To manage the interpreter explicitly:

```bash
uv venv --python 3.14 .venv-numpy314
uv pip install --python .venv-numpy314/bin/python numpy
FNP_ORACLE_PYTHON="$(pwd)/.venv-numpy314/bin/python3" \
  cargo run -p fnp-conformance --bin capture_numpy_oracle
```

| Environment variable | Effect |
|---|---|
| `FNP_ORACLE_PYTHON` | Path to a Python interpreter with NumPy. Explicit non-default paths win over repo-local bootstrap. |
| `FNP_REQUIRE_REAL_NUMPY_ORACLE=1` | Require a real NumPy oracle. If `FNP_ORACLE_PYTHON` is unset (or left at `python3`), `capture_numpy_oracle` bootstraps and reuses `.venv-numpy314` automatically, preferring `uv`, then standard `venv` + `pip`, finally a user-site `pip install numpy` fallback when the worker lacks venv tooling. |

Additional conformance / artifact commands:

```bash
cargo run -p fnp-conformance --bin generate_benchmark_baseline
cargo run -p fnp-conformance --bin run_performance_budget_gate
cargo run -p fnp-conformance --bin generate_raptorq_sidecars
cargo run -p fnp-conformance --bin validate_phase2c_packet -- --packet-id FNP-P2C-001
cargo run -p fnp-conformance --bin run_security_gate
cargo run -p fnp-conformance --bin run_test_contract_gate
cargo run -p fnp-conformance --bin run_workflow_scenario_gate
cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
cargo run -p fnp-conformance --bin run_fnp_python_api_coverage -- --fail-on-missing
```

---

## Performance

FrankenNumPy is profile-driven: every optimization is paired with a baseline, a single targeted lever, and a proof-backed delta artifact.

- **Release profile.** The workspace `Cargo.toml` does not override `[profile.release]`, so Cargo's defaults apply (`opt-level = 3`, `lto = false`, `codegen-units = 16`, `strip = false`). The workspace adds one custom profile, `release-perf`, defined explicitly: `inherits = "release"`, `lto = "thin"`, `codegen-units = 1`, `debug = "line-tables-only"`. That is the right profile for flamegraph profiling. Downstream consumers that want full LTO are free to add `[profile.release] lto = true` in their own top-level Cargo.toml.
- **Contiguous reduction kernel.** Axis reductions on contiguous data avoid per-element index computation. A targeted optimization pass (commit `d9cfe90`, 2026-02-13) reduced axis-reduction latency by ~56% (p50/p95/p99 deltas of ~90% on contiguous workloads). See `artifacts/optimization/` and `artifacts/baselines/` for the proof bundle.
- **Broadcast index mapping.** Output-to-source index mapping uses an incremental odometer instead of full unravel/remap per element.
- **2×2 fast paths.** Linear algebra has specialized 2×2 implementations that bypass general NxN overhead for the most common small-matrix case.
- **Horner's method.** Polynomial evaluation uses Horner form for numerical stability and minimal multiplications. The Stirling-series helpers in `fnp-random` (`logfactorial` for `k > 125` and the binomial BTPE acceptance path) use the same evaluation style.
- **Ziggurat sampling.** Normal and exponential random variates use Ziggurat (same as NumPy), which accepts ~97% of samples on the first try.

The G7 budget gate (`run_performance_budget_gate`) measures p50/p95/p99 latencies for ufunc and reduction sentinel workloads and rejects regressions. The cross-engine benchmark (`run_cross_engine_benchmark`) compares directly against NumPy.

**Current cross-engine picture** (2026-04-10 baseline at `artifacts/baselines/cross_engine_benchmark_v1.json`, 37 workloads — verified 2026-05-17 to be the most recent baseline on disk; later parity / metadata / fuzz work has not regenerated it, so ratios below reflect pre-May code paths):

| Op family | Median ratio (FNP / NumPy) | Verdict |
|---|---|---|
| I/O | 0.14× | **FrankenNumPy wins** |
| Random | 1.00× | Parity |
| Linalg | 1.02× | Parity |
| Reductions | 1.18× | Near-parity |
| Sorting | 1.90× | Acceptable |
| Statistics | 2.79× | Yellow |
| Matmul | 3.40× | Yellow |
| FFT | 10.56× | Mixed (power-of-2 fast, non-power-of-2 slow) |
| Ufunc broadcast | 13.53× | Red, large-scale |
| Ufunc elementwise | 30.76× | Red, large-scale |

The red-band workloads are the targets of the future Phase 3 work (SIMD, BLAS linkage, parallel execution). See Roadmap. For small/medium arrays and for I/O / random / linalg / reductions the picture is already at or near parity.

### Benchmark methodology

The cross-engine benchmark (`run_cross_engine_benchmark.sh`) is pedantic by design, so that ratios published in this README are reproducible:

- **Workload definition.** Every workload (`mul_f64_large`, `axis_reduction_axis0_3d`, `fft_pow2_1024`, ...) is defined in `artifacts/contracts/cross_engine_benchmark_workloads_v1.yaml` with its operation name, input shape, dtype, fixed seed, and warmup/measurement iteration counts.
- **Same inputs both engines.** Inputs are constructed once from the fixed seed and passed to both NumPy and FrankenNumPy. There is no per-engine input randomness.
- **Per-workload statistics.** For each workload we run a warmup pass (≥ 100 iterations on small ops, scaled down for slow ones), then a measurement pass, then record p50 / p95 / p99 latency and total bytes processed. Ratios reported in the table are p50 ratios.
- **Environment fingerprint.** Every baseline records a fingerprint: hostname, CPU model, core count, cache sizes, kernel version, Rust toolchain version, NumPy version, BLAS backend NumPy linked against (`numpy.show_config()`), and the workload YAML hash. Two baselines are only compared when their environment fingerprints match.
- **No NumPy fallback for the FNP side.** When we benchmark a tier-1 wrapper (see Python attribute resolution model), the benchmark forces the fast path: a fast-path-skipped run would be silently measuring numpy, not FrankenNumPy, and would be discarded.
- **Acceptable-degradation gates.** The G7 performance budget gate (`run_performance_budget_gate`) doesn't gate the ratio against NumPy; it gates the ratio against our *previous* baseline. A 5% regression on a sentinel workload fails the gate; new improvements are recorded as a new baseline. The proof bundle (`artifacts/optimization/<commit>.json`) is checked in.

The 2026-04-10 cross-engine baseline currently published is the one ADR-001 quotes when discussing the case for Phase 3. A refresh after each major performance lever lands is part of the optimization governance pattern: baseline → profile → single lever → conformance check → re-baseline.

---

## Artifact Durability (RaptorQ)

Every conformance artifact (fixture bundles, benchmark baselines, migration manifests, reproducibility ledgers, long-lived state snapshots) is protected by erasure-coding sidecars.

**Encoding.** Source data is hashed (SHA-256) and encoded into source + repair symbols using RaptorQ fountain codes. The sidecar records the codec parameters (symbol size, block count, repair overhead) alongside the encoded symbols in base64.

**Scrubbing.** A scrub report decodes all symbols, computes the SHA-256 of the decoded payload, and verifies it matches the source hash. It then drops one symbol and verifies recovery from the remaining symbols still produces the correct hash.

**Decode proof.** An explicit artifact records which symbol was dropped, how many repair symbols were needed to recover, and whether recovery succeeded. This is machine-checkable evidence that the artifact can survive single-symbol loss.

The G8 CI gate (`scripts/e2e/run_raptorq_gate.sh`) enforces that all required bundles have valid sidecars, scrub reports with `status: "ok"`, and decode proofs with `recovery_success: true`. A stress variant (`run_raptorq_stress_gate.sh`) drives recovery under higher loss budgets.

### Why RaptorQ?

A conformance archive is only useful if you can prove, at any point in the future, that the bytes you stored are the bytes you encoded. The usual alternatives have specific failure modes:

| Approach | Problem |
|---|---|
| SHA-256 of the bundle and call it a day | Detects corruption but cannot recover from it. A single flipped bit silently invalidates the entire artifact. |
| Mirror to a second location | Doubles storage cost; doesn't help when both copies share a failure mode (same FS bug, same backup tool, same rot). |
| `par2` / Parchive | Reed–Solomon based; recovery is fixed by the parity-set parameters at encode time. Adding redundancy after the fact requires re-encoding. |
| RAID-style striping | Works at the block layer, not the artifact layer. Doesn't survive an artifact moving between hosts. |
| **RaptorQ (RFC 6330)** | Fountain code: any sufficient subset of source-plus-repair symbols decodes back to the original payload. Repair overhead is configurable per-artifact; partial losses up to the configured tolerance recover transparently. |

RaptorQ encodes the artifact once into `k` source symbols plus `r` repair symbols; any `k + small_delta` symbols suffice to decode. The scrub gate verifies this property in CI: it deletes a symbol, runs decode, and verifies the SHA-256 of the decoded payload matches the pre-encoding hash. The decode proof is the machine-checkable receipt that this recovery actually worked, not just that the encoder ran.

The `asupersync` RaptorQ primitives (`fnp-conformance` uses them for the sidecar pipeline) give us cancellation-safe encoding and structured telemetry; useful when a benchmark batch encodes a thousand workload outputs in parallel.

---

## Divergence Ledger

Behavioral differences vs upstream NumPy that we accept either intentionally or as tracked parity debt live in [`docs/DIVERGENCES.md`](docs/DIVERGENCES.md). The ledger is machine-readable: a diagnostic case can only be marked `intentional_divergence` when it references a row here.

**Current state (2026-05-16): 1 active row.**

| ID | Disposition | Surface | Behavior |
|---|---|---|---|
| `franken_numpy-ucc2o` | `parity_debt` | `fnp-random` `SeedMaterial::None` / no-seed `default_rng()` | Uses a fixed deterministic seed (`0xC0DE_CAFE_F00D_BAAD`); NumPy sources OS entropy. Explicit seeds always match NumPy bit-for-bit. |

The ledger gate is enforced by:

```bash
cargo run -p fnp-conformance --bin run_divergence_ledger -- --fail-on-missing
```

---

## Fuzzing

The workspace ships **7 fuzz crates** with **27 fuzz targets** and **~200 curated seed-corpus files** as of 2026-05-16. Every fuzz crate is excluded from the main workspace (see `Cargo.toml` `[workspace] exclude`) so normal `cargo` commands don't pull in `libfuzzer-sys`. Full inventory in [`docs/FUZZING.md`](docs/FUZZING.md).

| Crate | Path | Targets |
|---|---|---|
| `fnp-dtype` | `crates/fnp-dtype/fuzz` | `fuzz_dtype_parse`, `fuzz_min_scalar_type`, `fuzz_can_cast`, `fuzz_result_type` |
| `fnp-io` | `crates/fnp-io/fuzz` | `fuzz_npy`, `fuzz_npz`, `fuzz_load_auto`, `fuzz_header`, `fuzz_fromstring`, `fuzz_loadtxt`, `fuzz_fromfile` |
| `fnp-iter` | `crates/fnp-iter/fuzz` | `fuzz_ndindex`, `fuzz_flatiter_indices`, `fuzz_nditer_plan`, `fuzz_transfer_class` |
| `fnp-linalg` | `crates/fnp-linalg/fuzz` | `fuzz_cholesky_nxn`, `fuzz_det_nxn`, `fuzz_qr_mxn` |
| `fnp-ndarray` | `crates/fnp-ndarray/fuzz` | `fuzz_broadcast_shape`, `fuzz_fix_unknown_dim`, `fuzz_as_strided`, `fuzz_sliding_window` |
| `fnp-random` | `crates/fnp-random/fuzz` | `fuzz_from_u64_seed`, `fuzz_seed_sequence` |
| `fnp-ufunc` | `crates/fnp-ufunc/fuzz` | `fuzz_parse_gufunc_signature`, `fuzz_datetime_unit_parse`, `fuzz_parse_fixed_signature` |

```bash
cargo install cargo-fuzz
cd crates/fnp-io/fuzz
cargo +nightly fuzz run fuzz_npy -- -max_total_time=300
```

---

## Parity Status

Every feature family is `parity_green`:

| Feature family | Status |
|---|---|
| `numpy.__all__` Python surface (499/499 names) | parity_green (structurally locked) |
| NumPy 2.0+ API (`unique_all/counts/inverse/values`, `cumulative_sum/prod`, `matrix_transpose`, `vecdot`, `permuted`, `trapezoid`, `unstack`) | parity_green |
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

See [`FEATURE_PARITY.md`](FEATURE_PARITY.md) for the live matrix with evidence links.

---

## Comparison with Other Rust Array Libraries

FrankenNumPy is not the first Rust array library. It is the only one that targets NumPy compatibility as the primary success metric. Some quick positioning:

| Library | What it is | When you'd use it instead |
|---|---|---|
| [`ndarray`](https://docs.rs/ndarray) | The de-facto Rust ndarray crate. Generic over element type and dimensionality, with BLAS bindings via `ndarray-linalg`. Idiomatic Rust API designed from scratch. | You're writing fresh Rust code that doesn't need NumPy semantics, you want compile-time dimensional checking via the `Ix` types, and a typed generic API matters more than NumPy parity. |
| [`nalgebra`](https://docs.rs/nalgebra) | Linear-algebra library with statically-sized vectors and matrices. Strong on geometry, graphics, robotics. | Your shapes are statically known and small, and you want the compiler to track them. Not a general N-D array library. |
| [`polars`](https://docs.rs/polars) | DataFrame engine (Arrow-backed), columnar, vectorized. | You're doing dataframe-style analytics on tabular data, not numerical array math. |
| [`burn`](https://docs.rs/burn) / [`candle`](https://docs.rs/candle-core) | Deep-learning tensor crates with autodiff and GPU backends. | You want autograd, GPU, or modern DL primitives. Tensor shape and dtype semantics differ deliberately from NumPy. |
| **FrankenNumPy** | NumPy-faithful array library with `fnp_python` PyO3 surface, dual-mode runtime, evidence ledger, and oracle-verified differential conformance. | You need the NumPy semantic contract: exact promotion table, exact broadcast rules, exact RNG sequences, NumPy-compatible `.npy`/`.npz`. Or you're shipping a Rust app that needs to read/write data produced by NumPy on the other side of the wire. |

The three libraries above are excellent in their target domains; FrankenNumPy is not trying to compete with them on idiom or on GPU. It is the right tool for "I need an answer NumPy would have given," and the wrong tool for almost everything else.

---

## Repository Layout

```
franken_numpy/
├── Cargo.toml                         # Workspace root (10 crates)
├── rust-toolchain.toml                # nightly-2026-02-20 (single source of truth)
├── FEATURE_PARITY.md                  # Live parity matrix + evidence links
├── CHANGELOG.md                       # Capability-area changelog
├── PROPOSED_ARCHITECTURE.md           # Architecture notes
├── audit_numpy_reality.md             # `numpy.__all__` coverage architecture + lock-in
├── audit_numpy_mocks.md               # Mock/stub/unwrap audit (zero production mocks)
├── docs/
│   ├── DIVERGENCES.md                 # Machine-readable divergence ledger
│   ├── FUZZING.md                     # Fuzz crate / target / seed inventory
│   └── adr/
│       └── ADR-001-parity-pivot.md    # Phase 3 (FFI / BLAS / threading) proposal
├── crates/
│   ├── fnp-dtype/                     # Dtype taxonomy, promotion table, cast policy
│   ├── fnp-ndarray/                   # Stride Calculus Engine (SCE)
│   ├── fnp-iter/                      # Transfer semantics, overlap-safe iteration, Nditer
│   ├── fnp-ufunc/                     # 800+ array operations, reductions, einsum, masked arrays
│   ├── fnp-linalg/                    # solve, eig, svd, qr, cholesky, lstsq, batched, complex
│   ├── fnp-random/                    # 5 bit generators, distributions, PCG64DXSM bit-exact parity
│   ├── fnp-io/                        # NPY/NPZ read/write, text I/O, DEFLATE, memmap
│   ├── fnp-python/                    # PyO3 bindings, 100% numpy.__all__ surface
│   ├── fnp-conformance/               # Oracle capture, differential / metamorphic / adversarial / RaptorQ
│   └── fnp-runtime/                   # Strict/hardened mode, evidence ledger, decision engine
├── legacy_numpy_code/numpy/           # Behavioral oracle (upstream NumPy source)
├── artifacts/                         # Contracts, security maps, logs, proofs, RaptorQ sidecars
├── scripts/e2e/                       # CI gate scripts (G1–G8) + cross-engine benchmark
└── .github/workflows/ci.yml           # CI gate topology (mirrors rust-toolchain.toml)
```

---

## Limitations

What doesn't work today.

- **No `pip install frankennumpy` packaging story yet.** The Python surface is reached today by building the `fnp-python` PyO3 extension and putting the renamed cdylib on `PYTHONPATH`. The pyproject.toml + wheel + PyPI publishing flow is future work. *Surface coverage itself is no longer a limitation*: the `fnp_python` module reaches **100% of `numpy.__all__`** (499/499 names), structurally locked.
- **No BLAS/LAPACK backend.** Linear algebra uses pure-Rust implementations (Householder QR, Golub–Kahan SVD, implicit shifted QR for eigenvalues). Competitive with BLAS for small matrices; slower for large ones. Optional BLAS linkage is a Phase 3 work-stream (ADR-001).
- **Complex elementwise arithmetic uses interleaved storage.** Complex64/Complex128 dtypes store real/imaginary parts as interleaved floats. Elementwise `multiply` and `divide` apply true complex arithmetic `(a+bi)(c+di) = (ac−bd)+(ad+bc)i`, but the interleaved representation adds overhead vs native complex types.
- **`multivariate_normal` uses Cholesky.** NumPy defaults to SVD. Switching would pull `fnp-linalg` into `fnp-random`'s dependency graph (currently `fnp-random` has zero external crates.io dependencies; only intra-workspace `fnp-ndarray`).
- **`multivariate_hypergeometric` uses sequential draws.** NumPy uses the `random_mvhg_marginals` algorithm.
- **Single-threaded.** All array operations are single-threaded. The `asupersync` integration is optional and used only for conformance pipeline orchestration, not parallel array computation. Multi-threading is a Phase 3 work-stream.
- **f64 internal representation for `UFuncArray`.** Numeric values are stored as `Vec<f64>` internally for arithmetic. For i64/u64 values > 2^53, `IntegerSidecar` preserves exact integer values through storage round-trips; arithmetic on such values still uses f64 approximation. Native i64/u64 paths are a Phase 3 work-stream.
- **`SeedMaterial::None` is deterministic.** A no-seed `default_rng()` uses a fixed default seed; NumPy uses OS entropy. Explicit-seed forms match NumPy bit-for-bit. See the Divergence Ledger.
- **Large-scale ufunc-elementwise / ufunc-broadcast hotspots.** Without SIMD or BLAS, these workloads sit in the 10–30× range vs NumPy at large array sizes. Small/medium and contiguous workloads are at or near parity.

---

## Roadmap (Phase 3 candidates)

[`docs/adr/ADR-001-parity-pivot.md`](docs/adr/ADR-001-parity-pivot.md) records the proposal to pivot from parity grinding to performance/distribution work now that the `numpy.__all__` surface is complete and structurally locked. The active candidates:

1. **Python packaging.** Build out the `pyproject.toml` + wheel + PyPI flow so `pip install frankennumpy` works on Linux, macOS, and Windows.
2. **BLAS / LAPACK backend.** Feature-gated `blas` linkage in `fnp-linalg`, dispatch large matrices to BLAS, keep pure-Rust for small.
3. **SIMD vectorization.** Safe SIMD via `std::simd` / `portable_simd`. Target: reduce large-elementwise ratio from 30× to <3×.
4. **Multi-threading.** Feature-gated `parallel` (Rayon) for reductions, matmul, sort on arrays above a threshold.
5. **Native i64/u64 arithmetic.** Remove the f64 intermediary for exact integer operations above 2^53.

---

## Anti-Goals

What FrankenNumPy explicitly will *not* try to be:

| Anti-goal | Why |
|---|---|
| **A GPU array library.** | Out of scope. CUDA/ROCm/Metal are entire engineering programs in themselves. Use CuPy, JAX, PyTorch, or candle for GPU. |
| **An autodifferentiation engine.** | Out of scope. Differentiable NumPy lives in JAX and PyTorch. We are about NumPy-semantic correctness, not gradient propagation. |
| **A DataFrame.** | Use polars or pandas. Tabular ops (joins, group-by, columnar dtypes per column) are a different abstraction. |
| **A symbolic computer-algebra system.** | Use SymPy. Symbolic differentiation, expression simplification, and arbitrary-precision rational arithmetic are not in scope. |
| **A drop-in for `scipy`.** | `scipy.linalg`, `scipy.stats`, `scipy.signal`, etc. are vast surfaces with their own design decisions. We mirror NumPy, not SciPy. (Some overlap is unavoidable in `scimath`, polynomial families, statistical distributions, but it is incidental, not the goal.) |
| **A reactive / streaming array library.** | No subscriptions, no observable updates, no "if you change A, recompute B." Arrays are values, not signals. |
| **A backwards-compatibility wrapper for ancient NumPy.** | We track current NumPy (1.x late and 2.x). Pre-1.0 NumPy idioms are not preserved. |
| **A formally-verified library.** | The conformance gates give very strong empirical assurance; formal verification (Coq / Lean / F\*) is a different project. |
| **A `no_std` library.** | We use `std::collections`, `std::sync`, allocator-backed `Vec`, and standard error traits. Embedded users without `std` are not the target audience today. |
| **A zero-allocation library.** | Numerical correctness comes first; we allocate `Vec<T>` freely. Pool-based, arena-based, or stack-allocated arrays are a different design point. |

Saying these things explicitly saves everyone time. If your use case sits on the wrong side of one of these lines, FrankenNumPy is the wrong tool and there is no point waiting for it to grow into one.

---

## Performance Levers Applied So Far

Concrete optimization passes that have landed, paired with what they bought. Each lever follows the optimization-governance pattern in [`PROPOSED_ARCHITECTURE.md`](PROPOSED_ARCHITECTURE.md): baseline → profile → single lever → conformance check → re-baseline → proof artifact.

| Lever | What it does | Where | Effect |
|---|---|---|---|
| **Contiguous-axis reduction kernel** | Replaces per-element unravel/ravel with a stride-aware fast path | `reduce_sum_axis_contiguous` in `fnp-ufunc/src/lib.rs` | Axis reductions on contiguous data ~56% p50 faster; commit `d9cfe90` (2026-02-13). Proof bundle under `artifacts/optimization/` |
| **Neumaier-compensated sum for large arrays** | Switches from naive accumulation to compensated sum above 1,000,000 finite elements | `reduce_sum_values`, `neumaier_sum_values` | Error stays at O(ε) per element instead of O(n·ε); no measurable runtime cost below threshold |
| **Incremental odometer for broadcast index mapping** | Output-to-source mapping increments stride state in a loop instead of recomputing per element | Inside `elementwise_binary` and the broadcast view machinery | Cuts per-element overhead on broadcast ops; particularly noticeable at small-medium array sizes |
| **2×2 linear-algebra fast paths** | Specialized closed-form `solve_2x2`, `det_2x2`, `inv_2x2`, `qr_2x2`, `svd_2x2`, `eigh_2x2`, `cholesky_2x2` | `fnp-linalg/src/lib.rs` | Avoids LU/Householder/QR overhead for the most common small-matrix case; dominant cost is the 2-line algebraic formula |
| **Horner / Clenshaw polynomial evaluation** | Standard nested multiplication for power-series; orthogonal-polynomial recurrence for Chebyshev/Legendre/Hermite/Laguerre | Throughout `fnp-ufunc` polynomial section | Numerical stability + minimal multiplications |
| **Ziggurat sampling for normal / exponential** | First-acceptance rate ~97%; rejected samples fall through to tail or wedge handling | `fnp-random/src/ziggurat.rs` | Matches NumPy's per-sample cost characteristic |
| **Zero-copy view machinery via `Arc<RwLock<Vec<f64>>>`** | Transpose, broadcast, sliding-window all produce views, not copies | `UFuncArrayView` in `fnp-ufunc/src/lib.rs` | Eliminates allocation on shape-only operations |
| **`Arc<[u8]>` for NPY payloads** | NPY load returns `Arc<[u8]>` rather than `Vec<u8>`; downstream consumers share the payload without cloning | `NpyArrayBytes` in `fnp-io` | Zero-copy sharing of loaded payloads (commit `3ed4487`) |
| **8-element chunk processing for boolean mask count** | `count_true_mask` processes booleans in 8-element groups for vectorizable counting | `fnp-iter/src/lib.rs` | Helps the optimizer emit popcnt-style code |
| **Bluestein chirp-Z FFT for non-power-of-2** | Rewrites arbitrary-length DFT as power-of-2 convolution; uses Cooley–Tukey for the convolution | `fft_dit` in `fnp-ufunc/src/lib.rs` | Lets every FFT length go through a single fast path |
| **Constant-time dtype promotion** | All 324 dtype pairs in a `const fn` match expression with no runtime allocation | `pub const fn promote(...)` in `fnp-dtype/src/lib.rs` | Promotion lookup is a couple of branches, not a hashmap |

Optimizations on the list for future work (not landed yet) are the SIMD/BLAS/parallel-execution items in the [Roadmap](#roadmap-phase-3-candidates). Every closed bead with a perf lever has a proof artifact in `artifacts/optimization/`; the same baseline/profile/lever/conformance/re-baseline cycle continues to apply.

---

## What "Clean-Room" Means Here

"Clean-room" appears repeatedly in this README; it deserves an unambiguous definition.

**What we do:**
- Read the published references behind every numerical algorithm (papers cited below).
- Read NumPy's C source code for *behavior* (what inputs produce what outputs, edge cases, error paths, dtype promotion rules, RNG state schemas), but treat the C as a behavioral specification rather than as code to translate.
- Write new Rust against that specification, structured idiomatically for Rust (enums, `Result`, `const fn`, no unsafe).
- Verify the result against NumPy's actual output via the differential oracle. The conformance gate is the proof of behavioral equivalence; the source is not the artifact.

**What this implies:**
- The Rust code is original. There is no line-by-line translation of NumPy's C macros, no replicated header layouts, no copied comments.
- License compatibility is straightforward: FrankenNumPy is MIT-with-rider (see `LICENSE`), independent of NumPy's BSD-3-Clause. The behavioral surface is interoperable; the codebases are not.
- Algorithmic correctness comes from the *mathematics* and the *oracle*, not from the original implementation. When NumPy's C has a bug, our differential test surfaces it as a divergence to be triaged on its own merits; we do not silently inherit the bug.
- A reader of `crates/fnp-*/src/lib.rs` should be able to understand each kernel from first principles without needing to consult NumPy's C. The reader of the conformance harness gets the additional guarantee that the kernel matches NumPy's observable behavior.

The `legacy_numpy_code/numpy/` directory is checked into the repo as a behavioral oracle for the conformance system; it is the upstream NumPy source for differential-testing reference only.

---

## F-order vs C-order in Practice

NumPy supports two memory orderings, and FrankenNumPy preserves the contract:

- **C-order** (row-major, `order='C'`): the rightmost dimension is contiguous. Stride for the last axis is `item_size`; each axis to the left multiplies by the next axis's size. This is the default.
- **F-order** (column-major, `order='F'`, "Fortran order"): the leftmost dimension is contiguous. Stride for the first axis is `item_size`.

The choice matters in three concrete ways:

1. **Iteration speed.** A reduction along the contiguous axis is dramatically faster than a reduction across a strided axis. For a C-order array, `sum(axis=-1)` is the fast path; for an F-order array, `sum(axis=0)` is.
2. **Interop with FORTRAN-style code.** SciPy's LAPACK bindings, OpenBLAS, and most numerical-linear-algebra libraries expect F-order matrices. NPY files default to C-order; `numpy.asfortranarray` (and FrankenNumPy's equivalent) makes a copy in F-order.
3. **`reshape` ambiguity.** `reshape(shape, order='C')` walks elements in row-major order; `reshape(shape, order='F')` walks them column-major. The same shape can yield different element orderings.

SCE's `contiguous_strides(shape, order)` computes the correct stride vector for either order. The `NdLayout` type tracks which order an array currently has. A view created from a C-order array via `transpose()` becomes F-order at zero cost; the transpose is just a stride permutation.

The reduce-axis contiguity rule shows up directly in the performance picture: axis-0 reductions on F-order arrays hit the same contiguous fast-path as axis-(-1) reductions on C-order arrays, because in both cases the inner loop walks unit-stride memory.

---

## Reading the Source Code

If you want to understand how FrankenNumPy works at the source level, here are the natural entry points:

| To understand... | Start at |
|---|---|
| Shape and stride arithmetic | `crates/fnp-ndarray/src/lib.rs` (1,531 lines). Small enough to read in one sitting. |
| Dtype promotion rules | `crates/fnp-dtype/src/lib.rs` around `pub const fn promote(...)`. The whole 324-entry table is in one match expression. |
| The transfer-loop selector | `crates/fnp-iter/src/lib.rs` around `TransferClass`, `TransferSelectorInput`, and `Nditer`. |
| A representative ufunc | `crates/fnp-ufunc/src/lib.rs`: `elementwise_binary` (around line 5920) for the broadcast skeleton, `reduce_sum_values` (around line 25684) for the compensated-summation logic. |
| Eigenvalue and SVD algorithms | `crates/fnp-linalg/src/lib.rs`: `svd_mxn`, `eig_nxn`, `qr_mxn` are stand-alone implementations of the Householder / Golub–Kahan / Francis machinery. |
| RNG bit generators | `crates/fnp-random/src/lib.rs`: search for `Pcg64DxsmRng`, `Mt19937Rng`, `PhiloxRng`, `Sfc64Rng`, each in its own struct. The `Generator` API is in the second half of the file. |
| NPY/NPZ binary format | `crates/fnp-io/src/lib.rs`: `pub fn save` and `pub fn load` are entry points; the header parser and writer sit next to them. |
| The runtime decision engine | `crates/fnp-runtime/src/lib.rs` is the smallest crate (1,672 lines); reading it end-to-end is a complete tour of the strict/hardened mode split and the evidence ledger. |
| How parity is verified | `crates/fnp-conformance/src/lib.rs` plus the 47 binaries under `crates/fnp-conformance/src/bin/`. Start with `capture_numpy_oracle.rs` and `run_ufunc_differential.rs`. |
| The PyO3 surface | `crates/fnp-python/src/lib.rs`: single 68k-line file because PyO3 wants all `#[pymodule]` bindings co-located. The structural lock-in test is at `crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs::fnp_python_covers_full_numpy_all`. |

The largest crate (`fnp-ufunc` at 59k lines) sits in one file because the ufunc dispatch table is centralized; splitting it would scatter the dispatcher and obscure the structure.

**Finding a function fast.** With 1,575 `pub fn` declarations across the workspace's library code (`crates/*/src/**/*.rs`, excluding `src/bin/`), ripgrep is the navigation tool of choice:

```bash
# Find a pub fn by name (across all crates):
rg -n 'pub fn <name>' crates/

# Find the impl block on UFuncArray that defines a method:
rg -nB1 -A1 'impl UFuncArray' crates/fnp-ufunc/

# Find every test exercising a name:
rg -n 'fn .*<name>' crates/*/tests/ crates/*/src/lib.rs
```


---

## Inspirations and Prior Art

FrankenNumPy stands on the shoulders of some specific design lineages worth acknowledging.

| Project | What we learned from it |
|---|---|
| **[NumPy](https://github.com/numpy/numpy)** itself | The behavioral specification we're targeting, and the source of every algorithm correspondence in the [Algorithm References](#algorithm-references-and-citations) section. NumPy is also the oracle in every differential-conformance run: there is no FrankenNumPy without a real NumPy on the build host. |
| **[NEP 19 — RNG Policy](https://numpy.org/neps/nep-0019-rng-policy.html)** (Robert Kern, 2018) | The stream-stability policy that made it acceptable to introduce new bit generators (PCG64, PCG64DXSM, Philox, SFC64) alongside MT19937 without surprising existing users. The `SeedSequence` API NumPy ships alongside the NEP is itself derived from Melissa O'Neill's C++11 `std::seed_seq` (per the copyright header on `numpy/random/bit_generator.pyx`); we re-implement that design directly from the upstream source: hierarchical `spawn(n)`, entropy-pool mixing, schema-versioned state payload. |
| **[NEP 1 — A Simple File Format for NumPy Arrays](https://numpy.org/neps/nep-0001-npy-format.html)** (Robert Kern) | The `.npy` binary format. Bit-exact round-trip is our promise; the spec is the contract. NPY 2.0 extends NPY 1.0 to support headers larger than 65,535 bytes. |
| **[CPython](https://github.com/python/cpython)** | The Python runtime that `fnp_python` is loaded into via PyO3, and the source of the exception base classes (`ValueError`, `TypeError`, `OSError`, ...) that `numpy.exceptions` (and the `fnp_python.exceptions` re-export) ultimately inherit from. The numpy-specific assertion helpers in `numpy.testing` and the numpy-specific exception classes (`AxisError`, `ComplexWarning`, ...) come from numpy itself, not from CPython. |
| **[ndarray](https://github.com/rust-ndarray/ndarray)** (bluss et al.) | The "what does idiomatic Rust N-D arrays look like?" reference. `ndarray` made different design choices (statically-known dimensionality via `Ix`, BLAS via `ndarray-linalg`, generic over element types). We departed from them to target NumPy semantics. |
| **[nalgebra](https://github.com/dimforge/nalgebra)** (Sébastien Crozet et al.) | The reference for "what fully-typed, statically-sized linear algebra looks like in Rust." Our `2×2 fast paths` borrow the spirit but not the typing; we keep dynamic shapes throughout because NumPy's contract is dynamic. |
| **[RaptorQ (RFC 6330)](https://datatracker.ietf.org/doc/html/rfc6330)** | The fountain code that protects every conformance artifact. The IETF spec is the contract; the asupersync implementation is the engine. |
| **[asupersync](https://github.com/Dicklesworthstone/asupersync)** | The optional structured-async runtime used in `fnp-runtime`'s observability pipelines and in `fnp-conformance`'s RaptorQ primitives. Not a runtime dependency of the numerical core. |
| **[frankentui](https://github.com/Dicklesworthstone/frankentui)** | The optional terminal-native dashboards exposed via `fnp-runtime`'s `frankentui` feature. Same project family as FrankenNumPy; same "rewrite an established interface in safe Rust with explicit contracts" pattern. |
| **[beads_rust](https://github.com/Dicklesworthstone/beads_rust)** (`br`) | The dependency-aware issue tracker that drives the multi-agent workflow described in [Multi-Agent Development Process](#multi-agent-development-process). Every closed bead is a contract we've fulfilled. |

The "clean-room reimplementation of an established interface in safe Rust" pattern itself is widely used; FrankenNumPy is one of a family. See also Frankensearch (the Tantivy-on-steroids hybrid search engine), Frankentui (the terminal UI library), and the broader "Frankenstack" of memory-safe replacements for legacy infrastructure.

---

## Algorithm References and Citations

FrankenNumPy is a clean-room port. We re-implemented the algorithms by reading the published references and NumPy's C source, then wrote new Rust against the same mathematical contract. Significant references behind specific kernels:

**RNG and distributions**

- **PCG64 / PCG64DXSM:** M. E. O'Neill, *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation*, 2014. DXSM is the variant NumPy 1.20+ ships as its high-quality default.
- **Mersenne Twister (MT19937):** M. Matsumoto and T. Nishimura, *Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator*, ACM TOMACS 8(1), 1998.
- **Philox:** J. K. Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3*, SC'11.
- **SFC64:** C. Doty-Humphrey, *PractRand*, 2014 (Small Fast Counting generator).
- **SeedSequence:** adapted from M. E. O'Neill's C++11 `std::seed_seq` (2015); shipped in NumPy's `bit_generator.pyx` alongside the broader RNG-policy framework formalized in NEP 19 (R. Kern, 2018).
- **Lemire bounded integers:** D. Lemire, *Fast Random Integer Generation in an Interval*, ACM TOMS 45(1), 2019.
- **Ziggurat sampling:** G. Marsaglia and W. Tsang, *The Ziggurat Method for Generating Random Variables*, J. Stat. Software 5(8), 2000.
- **BTPE binomial:** V. Kachitvichyanukul and B. W. Schmeiser, *Binomial random variate generation*, CACM 31(2), 1988.
- **HRUA hypergeometric:** E. Stadlober, *Sampling from Poisson, binomial and hypergeometric distributions: ratio of uniforms as a simple and fast alternative*, Math. Statist. Sektion 303, 1989.
- **PTRS Poisson:** W. Hörmann, *The Transformed Rejection Method for Generating Poisson Random Variables*, Insurance: Math. Econ. 12, 1993.
- **Marsaglia–Tsang gamma:** G. Marsaglia and W. Tsang, *A Simple Method for Generating Gamma Variables*, ACM TOMS 26(3), 2000.

**Linear algebra**

- **Householder QR:** A. S. Householder, *Unitary triangularization of a nonsymmetric matrix*, J. ACM 5(4), 1958.
- **Golub–Kahan bidiagonalization + implicit shifted QR for SVD:** G. H. Golub and W. Kahan, *Calculating the Singular Values and Pseudo-Inverse of a Matrix*, SIAM JNA 2(2), 1965; J. Demmel and W. Kahan, *Accurate singular values of bidiagonal matrices*, SIAM JSSC 11(5), 1990.
- **Implicit QR for unsymmetric eigenvalues:** J. G. F. Francis, *The QR Transformation*, Comput. J. 4(3), 1961.
- **Symmetric tridiagonal QL/QR:** H. Bowdler, R. S. Martin, C. Reinsch, J. H. Wilkinson, *The QR and QL algorithms for symmetric matrices*, Numer. Math. 11, 1968.
- **Padé approximation for `expm`:** N. J. Higham, *The Scaling and Squaring Method for the Matrix Exponential Revisited*, SIAM JMAA 26(4), 2005.
- **Matrix logarithm via Schur–Padé:** N. J. Higham and L. Lin, *A Schur–Padé Algorithm for Fractional Powers of a Matrix*, SIAM JMAA 32(3), 2011.
- **Polar decomposition:** N. J. Higham, *Computing the Polar Decomposition with Applications*, SIAM JSSC 7(4), 1986.

**FFT**

- **Cooley–Tukey decimation-in-time:** J. W. Cooley and J. W. Tukey, *An Algorithm for the Machine Calculation of Complex Fourier Series*, Math. Comp. 19(90), 1965.
- **Bluestein chirp-Z:** L. I. Bluestein, *A linear filtering approach to the computation of the discrete Fourier transform*, IEEE Trans. AU 18(4), 1970.

**Special functions, signal, statistics**

- **Modified Bessel I0 for Kaiser windowing:** piecewise rational approximations from W. J. Cody, *Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)*, Math. Comp. 28(125), 1974.
- **Histogram bin selection (Sturges, sqrt-choice, auto):** H. A. Sturges, *The choice of a class interval*, J. ASA 21(153), 1926; D. Freedman and P. Diaconis, *On the histogram as a density estimator: L2 theory*, Z. Wahr. Verw. Gebiete 57, 1981.
- **Polynomial evaluation (Horner) and orthogonal-polynomial recurrence (Clenshaw):** W. G. Horner, *A new method of solving numerical equations of all orders, by continuous approximation*, Phil. Trans. R. Soc. 109, 1819; C. W. Clenshaw, *A note on the summation of Chebyshev series*, MTAC 9(51), 1955.

**Format and infrastructure**

- **NPY 1.0 / 2.0 binary format:** R. Kern, *NEP 1: A Simple File Format for NumPy Arrays*, 2007, plus the version-2.0 extension to support long headers.
- **RaptorQ fountain code:** M. Luby, A. Shokrollahi, M. Watson, T. Stockhammer, L. Minder, *RaptorQ Forward Error Correction Scheme for Object Delivery*, IETF RFC 6330, 2011.

Every kernel ported from one of these papers has at least one oracle test in the conformance suite that compares its output against NumPy's implementation of the same algorithm, at one or more curated seeds and input distributions.

---

## Glossary

Project-specific vocabulary used throughout the README, docs, and code comments:

| Term | Definition |
|---|---|
| **SCE** | Stride Calculus Engine. The single Rust subsystem (`fnp-ndarray`) that owns all shape-transformation rules: shape→strides, broadcast legality, reshape with `-1` inference, alias-sensitive view transitions. |
| **Strict mode** | Runtime mode that maximizes observable NumPy compatibility for the full legacy behavior matrix. No behavior-altering repairs. |
| **Hardened mode** | Runtime mode that preserves the API contract while adding safety guards and bounded defensive recovery for malformed inputs and hostile edge cases. |
| **Evidence ledger** | Append-only structured log of every runtime decision (action, class, evidence terms, posterior probability, timestamp, env fingerprint). Lives in `fnp-runtime`. |
| **Override audit event** | A separate, narrowly-scoped record for any explicit human-requested bypass of a fail-closed gate. Always paired with an audit reference. |
| **Parity debt** | A behavioral divergence from NumPy that is **scheduled to be closed**, not an accepted scope reduction. The `parity_debt` rows in `docs/DIVERGENCES.md` are the live tracker. |
| **Intentional divergence** | A behavioral difference from NumPy that we deliberately accept. Currently: **zero** such rows in the divergence ledger. |
| **Oracle** | The reference implementation we compare against; almost always a real NumPy on the build host. A `pure_python_fallback` oracle is rejected by the G3 gate. |
| **Witness** | A hard-coded expected output for a specific RNG seed or kernel input. Witness comparison catches silent algorithmic drift even when the new output is mathematically valid. |
| **Phase2C extraction packet** | One of nine domain-scoped specification bundles (FNP-P2C-001 through FNP-P2C-009). Each packet ships a fixture manifest, contract table, parity gate, risk note, and parity report with RaptorQ sidecar + decode proof. |
| **Contract schema** | The `phase2c-contract-v1` schema that the packet readiness validator (`validate_phase2c_packet`) checks each packet against. A packet that is missing a required field is `not_ready`. |
| **Differential** | A test layer that compares FrankenNumPy output to NumPy output for the same input. |
| **Metamorphic** | A test layer that verifies an identity property (e.g. `sum(a) == sum(sort(a))`) that must hold regardless of input. |
| **Adversarial** | A test layer driven by curated hostile fixtures and coverage-guided fuzzing. |
| **Witness stability** | A test layer that pins exact RNG and kernel outputs as hard-coded constants, so unintended algorithmic changes show up as diffs. |
| **G1–G8** | The eight CI gates, in order: fmt+lint, unit/property, oracle differential, adversarial+security, test contract, workflow forensics, performance budget, durability/decode. |
| **RaptorQ sidecar** | An auxiliary file alongside an artifact bundle containing the fountain-code encoding parameters plus base64-encoded source and repair symbols. |
| **Scrub** | The process of decoding all symbols, computing the payload hash, dropping a symbol, and verifying that recovery from the remaining symbols still hashes to the same value. |
| **Decode proof** | A machine-checkable record asserting that scrub succeeded (`recovery_success: true`). |
| **PCG64DXSM** | The default bit generator. 128-bit state, 128-bit increment, DXSM output function; bit-exact match for NumPy's `numpy.random.PCG64DXSM`. |
| **BTPE / HRUA / PTRS** | Distribution-specific rejection algorithms ported from NumPy's C source. BTPE = Binomial / Triangle / Parallelogram / Exponential (Kachitvichyanukul & Schmeiser 1988); the four regions of the bounding-envelope rejection sampler. HRUA = Hypergeometric Ratio-of-Uniforms with Aliasing (Stadlober 1989). PTRS = Poisson Transformed Rejection (Hörmann 1993). |
| **Lemire's method** | The bounded-integer rejection algorithm used by NumPy for `random_bounded_uint64`. Two-tier dispatch: 32-bit ranges use buffered `next_uint32`, larger ranges use 128-bit multiplication. |
| **Bead / `br`** | An issue in the project's `beads_rust` tracker (`br ready`, `br close`, etc.). Used for dependency-aware work selection. As of 2026-05-16, 1,337 beads are closed. |
| **`bv`** | Graph-aware triage on top of the bead database (PageRank, betweenness, critical-path). |
| **MCP Agent Mail** | The advisory file-reservation and inter-agent messaging system used during multi-agent development on this repo. Not a runtime dependency of FrankenNumPy itself. |

---

## Project Timeline at a Glance

Major milestones in the order they landed. For the full per-commit history, see [`CHANGELOG.md`](CHANGELOG.md); for the live bead trail, see `.beads/issues.jsonl`.

| Date | Milestone |
|---|---|
| **2026-02-13** | Project bootstrap: 9-crate workspace, Stride Calculus Engine, dual-mode runtime, 9 Phase2C extraction packets, conformance harness. First performance lever (contiguous reduction kernel) lands the same day. |
| 2026-02-14 → 2026-02-18 | CI gate topology (G1–G8) wired up; security gate, workflow scenario gate, RaptorQ durability gate all online. Differential conformance against real NumPy enforced on every CI run. |
| 2026-02-15 → 2026-02-21 | First major operator wave: `fnp-ufunc` grows from stub to 35 binary ops + 22 unary ops + reductions; `fnp-linalg` adds SVD, QR, eig, LU; `fnp-random` ships PCG64DXSM + MT19937 + SeedSequence; `fnp-io` ships NPY 1.0/2.0 + NPZ + DEFLATE. |
| 2026-02-26 | Structured dtype support; complex dtype I/O. Golub–Kahan SVD lands. |
| 2026-03-02 → 2026-03-12 | `ArrayStorage` bridge eliminates `Vec<u8>` reinterpretation; F16 support via `half` crate; complex linalg; safe memmap. Copy-on-write views with fine-grained memory overlap detection. |
| 2026-03-15 → 2026-03-21 | NumPy-compatible BTPE binomial, HRUA hypergeometric, PTRS Poisson, Lemire bounded integers; NaN-propagation correctness sweep across all reductions, sorts, percentiles, ptp. Eigenvalue sort order fixed to ascending. Full `lstsq` 4-tuple. Philox bit generator. |
| 2026-04-10 | First cross-engine benchmark baseline (37 workloads, ADR-001). Pivot proposal drafted but the parity track keeps going. |
| 2026-04-22 | First `numpy.__all__` reality-check audit: 43.3% (216/499) reachable. Plan to close the gap is formed. |
| 2026-04-24 | `numpy.random` parity epic closes: 41/56 MUST distributions at 73.2% compliance. |
| 2026-04-24 | `numpy.polynomial` parity epic closes: 63/66 MUST at 95.5%. |
| 2026-05-09 | `numpy.__all__` reaches ~96% reachable through the parity wave. |
| **2026-05-13** | **100% of `numpy.__all__` reachable (499/499)**, structurally locked by `fnp_python_covers_full_numpy_all`. The headline milestone. |
| 2026-05-14 | Mock/stub audit confirms zero TODO/FIXME/HACK/STUB/unimplemented across all 10 implementation crates. |
| 2026-05-16 | Documentation precision wave: 92 beads closed on this single date across README, AGENTS.md, FEATURE_PARITY.md, audit docs, divergence ledger, fuzzing docs. |
| 2026-05-16 | Divergence ledger active rows: **1** (the `SeedMaterial::None` deterministic seed). Workspace fully green: 6,392 tests, all 10 crates passing, zero unsafe blocks. |

In quantitative terms: **~650 commits in May 2026 alone**, **~1,880 commits since 2026-04-01**, **1,337 closed beads** as of 2026-05-16, all under a single git author working with a multi-agent swarm. (These are live numbers; the repository keeps moving.)

---

## FAQ

**Is this a drop-in replacement for NumPy?**
Surface-wise yes: `fnp-python` exposes **100% of `numpy.__all__`** (499/499 names), structurally locked by `fnp_python_covers_full_numpy_all`. Distribution-wise not yet: there is no pip wheel today. See the Limitations section for the full packaging gap discussion.

**How do you verify parity with NumPy?**
Oracle tests. We run the same operations with the same inputs in both NumPy and FrankenNumPy and compare outputs to floating-point tolerance. For RNG, the comparison is bit-exact. The G3 CI gate enforces `FNP_REQUIRE_REAL_NUMPY_ORACLE=1` and rejects pure-Python fallback oracle output.

**Why Rust nightly?**
Rust Edition 2024. The toolchain is pinned to `nightly-2026-02-20` for reproducibility; see `rust-toolchain.toml` and the `RUST_TOOLCHAIN` env var in `.github/workflows/ci.yml`, which is the single source of truth for the CI gates.

**Why zero unsafe code?**
Memory safety is a core value. 9 of the 10 implementation crates declare `#![forbid(unsafe_code)]`. `fnp-python` is the one that doesn't, because PyO3 procedural macros may expand into unsafe as part of generating the cdylib entry point. In practice the current `fnp-python` source has zero hand-written unsafe blocks, so the workspace is unsafe-free today across all 10 crates. The lint is opt-out for `fnp-python`, not invoked.

**How fast is it?**
Profile-driven. I/O, random, linalg, reductions, and sorting are at or near parity with NumPy. FFT is mixed (power-of-two fast, non-power-of-two slower). Large-scale elementwise/broadcast ufuncs are the main hotspot, with 10–30× ratios; the Phase 3 SIMD/BLAS work-streams target these. Small/medium array workloads are competitive across the board.

**Can I use just the RNG crate?**
Yes. `fnp-random` pulls **zero external crates.io dependencies** (only depends on `fnp-ndarray` within the workspace) and produces bit-exact NumPy-compatible random sequences from a given seed.

**What's the difference between `fnp_python` and just calling numpy?**
`fnp_python` is the parity oracle surface. Hot operations execute on the Rust engine for native speed; everything else falls back to numpy verbatim so behavior is identical (including version-gated and deprecation paths). You get one drop-in module, with Rust under the hood where it matters.

**Is anything intentionally divergent from NumPy?**
[`docs/DIVERGENCES.md`](docs/DIVERGENCES.md) is the machine-readable ledger. As of 2026-05-16 there is exactly one active row, and it is parity debt (`fnp-random` no-seed default), not an intentional design divergence. The ledger gate is enforced in CI.

**Are there any stubs, TODOs, or mock code in production?**
No. The [`audit_numpy_mocks.md`](audit_numpy_mocks.md) automated audit shows zero `TODO` / `FIXME` / `HACK` / `STUB` / `unimplemented!()` / `todo!()` across the 10 production crates, and zero production `.unwrap()` outside `fnp-conformance` fixture-harness code.

**How do I find what changed recently?**
[`CHANGELOG.md`](CHANGELOG.md) is organized by capability area, with representative commit links and bead IDs. [`FEATURE_PARITY.md`](FEATURE_PARITY.md) is the live parity matrix. [`audit_numpy_reality.md`](audit_numpy_reality.md) documents how the 100% surface coverage is maintained.

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## License

MIT with an OpenAI/Anthropic rider. See [`LICENSE`](LICENSE).
