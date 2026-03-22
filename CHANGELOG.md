# Changelog

All notable changes to FrankenNumPy are documented in this file.

FrankenNumPy is a memory-safe, clean-room Rust reimplementation of NumPy. The
workspace version is `0.1.0` (pre-release). There are no tagged releases or
GitHub Releases yet; every entry below maps to a date range on the `main` branch.
Representative commits link to
`https://github.com/Dicklesworthstone/franken_numpy/commit/<hash>`.

---

## [Unreleased] — development head

All work to date lives on `main` with no formal release tags. The sections below
are organized chronologically by the date ranges in which capabilities landed.

---

### 2026-03-21 — Philox RNG, full lstsq, Arc-backed I/O payloads

**RNG**
- Implement Philox counter-based RNG (CBRNG) with full state serialization and
  deserialization.
  [ee17681](https://github.com/Dicklesworthstone/franken_numpy/commit/ee1768141ce11f8a23228e4a351efef539a04503),
  [509055b](https://github.com/Dicklesworthstone/franken_numpy/commit/509055b1ed331590db3c21ab86ca7344a6dd42ce)
- Add RNG state serialization (`get_state` / `set_state`) and harden `.npy`
  header parsing against allocation bombs.
  [9894042](https://github.com/Dicklesworthstone/franken_numpy/commit/989404292d75fc5c103eb0f1ae08b99488c84b94)
- Revert ziggurat sampler to match original NumPy golden values; conformance
  golden values updated accordingly.
  [29ae5cd](https://github.com/Dicklesworthstone/franken_numpy/commit/29ae5cd117f6743a6a233dda21c954a40b2276c3)
- Fix binomial BTPE algorithm edge cases and simplify ziggurat samplers.
  [e341c26](https://github.com/Dicklesworthstone/franken_numpy/commit/e341c263725935ad46cb25cd4bd953ec281778e4)
- Fix SVD convergence, reduction shapes, and RNG state width.
  [c89e90c](https://github.com/Dicklesworthstone/franken_numpy/commit/c89e90c2db4693d798c23abf635537c744539ff7)

**Linalg**
- Expand `lstsq` to return the full NumPy 4-tuple `(x, residuals, rank, singular_values)`.
  [f9d51d0](https://github.com/Dicklesworthstone/franken_numpy/commit/f9d51d052fedc2cb5f7cbd5320da6c072eda549e)
- Add reduced SVD support via `svd_full(full_matrices)` parameter.
  [ad079d8](https://github.com/Dicklesworthstone/franken_numpy/commit/ad079d86e3d0a94c824b91de0644086b24c0da7e)
- Fix eigenvalue sort order to ascending to match NumPy convention.
  [5c57a67](https://github.com/Dicklesworthstone/franken_numpy/commit/5c57a6779122ab621786857e886ffd27b4d713b8)

**I/O**
- Switch `NpyArrayBytes` payload to `Arc<[u8]>` for zero-copy sharing.
  [3ed4487](https://github.com/Dicklesworthstone/franken_numpy/commit/3ed44873d78346240f7baa6baf763a51402f0829)

**Ufunc**
- Implement Python-compatible whitespace `split`/`rsplit` with `maxsplit`.
  [85789d2](https://github.com/Dicklesworthstone/franken_numpy/commit/85789d27a5a23e5998db0eeeee49a7b5c6771d85)
- Add `masked_sort_order` helper and expand array function coverage.
  [5d166d6](https://github.com/Dicklesworthstone/franken_numpy/commit/5d166d6325b4ae82931baaa0f5032a62ed5f6d16)

---

### 2026-03-20 — Overlap hardening, LU-based inverse, dtype cast fixes

**Ufunc**
- Reject out-of-bounds `as_strided` views instead of zero-filling.
  [e771acb](https://github.com/Dicklesworthstone/franken_numpy/commit/e771acb1b63295b3776c65b0ade2e9d3f56e536e)
- Harden memory overlap detection, broadcasting, and eliminate `unwrap` panics.
  [3cff47a](https://github.com/Dicklesworthstone/franken_numpy/commit/3cff47ae9bb2d88422755a7642df0a3d6cb031de)

**Linalg**
- Rewrite `inv_nxn` to use `lu_solve_multi` and optimize `cholesky_solve_multi`.
  [062be4e](https://github.com/Dicklesworthstone/franken_numpy/commit/062be4eed99ab1af2b4133e9954620d2c34f0bde)

**Dtype**
- Expand safe cast rules for 64-bit integers and complex-to-string conversions.
  [65e0832](https://github.com/Dicklesworthstone/franken_numpy/commit/65e0832419b8e81ac21f0d734d0e24baf8a5f55c)

---

### 2026-03-19 — multi_dot chain ordering, NPY parser overhaul, +57 tests

**Ufunc**
- Implement optimal `multi_dot` chain multiplication via dynamic programming.
  [38cb001](https://github.com/Dicklesworthstone/franken_numpy/commit/38cb00133d0b9d01399ad54d95173f81b6962205)

**I/O**
- Major overhaul of NPY header parsing with stricter validation.
  [9a5b5b9](https://github.com/Dicklesworthstone/franken_numpy/commit/9a5b5b9cb253d9dd304170768949975396180ba6)
- Expand CI gate topology and refactor NPY header parsing.
  [3289a6f](https://github.com/Dicklesworthstone/franken_numpy/commit/3289a6fcf79f0086f9d1ba4f58cb58176b4b6af4)

**Cross-crate**
- Correct `finfo` precision, `remainder` semantics, and stride classification.
  [52c4efb](https://github.com/Dicklesworthstone/franken_numpy/commit/52c4efbb6c327ae64a280b7f5d8b9ed55954b330)
- Expand ufunc, linalg, runtime, and iterator implementations (+57 tests).
  [752082f](https://github.com/Dicklesworthstone/franken_numpy/commit/752082fc994da1bbbbd29c99b24c000f6e111351)

---

### 2026-03-18 — Overlap detection, readonly tracking, IntegerSidecar, 2,016 tests

**Ndarray / Iter**
- Add fine-grained memory overlap detection, readonly view tracking, and
  reduction edge-case handling.
  [4326f30](https://github.com/Dicklesworthstone/franken_numpy/commit/4326f30eecd5920e53866d1347f014b446a4864c)
- Prevent signed-to-unsigned `same_kind` cast and fix negative stride
  classification.
  [0c43819](https://github.com/Dicklesworthstone/franken_numpy/commit/0c43819a7d4b0b72db69d0c96f2264ef4429ffb5)

**Dtype**
- Add `IntegerSidecar` for lossless integer storage, complex arithmetic
  improvements, and transfer-loop selector.
  [36b77e8](https://github.com/Dicklesworthstone/franken_numpy/commit/36b77e8edebd1898166d900ae83a5cd5730930be)

**Conformance**
- Add datetime differential test cases.
  [3c7c84c](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7c84cfc4abf601385a2baba55db5b0046593a8)
- Add P2C003 transfer invariant tests.
  [96e61de](https://github.com/Dicklesworthstone/franken_numpy/commit/96e61de3c5efad49e5e76d51704873ee5451925f)

---

### 2026-03-16 — NaN propagation sweep, oracle parity tests, BTPE Poisson rewrite

**NaN correctness** (systematic fix sweep across reductions and sorting)
- `reduce_min`, `reduce_max`: propagate NaN instead of skipping.
  [4891c64](https://github.com/Dicklesworthstone/franken_numpy/commit/4891c640a288efda6bc73acc03301d9a095cff76)
- `ptp` (peak-to-peak): propagate NaN for both `UFuncArray` and `MaskedArray`.
  [2fec037](https://github.com/Dicklesworthstone/franken_numpy/commit/2fec037f30db8b9743d0caca13993fafc8b97be6)
- `cummin`, `cummax`, `mode`, `heapsort`: propagate NaN through accumulators.
  [2a82df1](https://github.com/Dicklesworthstone/franken_numpy/commit/2a82df1f0e31ade8f7773da00934f10028dd9ac0)
- `median`, `percentile`, `quantile`: NaN early-returns before sorting.
  [6f3a4ae](https://github.com/Dicklesworthstone/franken_numpy/commit/6f3a4ae4a3f50b850aa9bc5313a3ab42b0b49b42)

**Oracle parity tests**
- Add NumPy oracle parity tests for `det`, `inv`, `solve`, `norm`, `eig`,
  `svd`, `cholesky`, `qr`.
  [7b51f50](https://github.com/Dicklesworthstone/franken_numpy/commit/7b51f50b616427b8d5a04b365e3c7b8e316cb6cc)
- Add `.npy` format oracle tests for roundtrip, magic bytes, and header parsing.
  [710c3b0](https://github.com/Dicklesworthstone/franken_numpy/commit/710c3b0229938b7c5f5379a180846dfa75e7d0a9)
- Add end-to-end workflow oracle tests for z-score, sort-cumsum, and boolean
  filter pipelines.
  [a2da789](https://github.com/Dicklesworthstone/franken_numpy/commit/a2da789d5470c361fd6b67ddb2c9b40b36377e0f)

**Random**
- Rewrite Poisson sampling with PTRS algorithm; add `random_loggam`.
  [b7af286](https://github.com/Dicklesworthstone/franken_numpy/commit/b7af2866cbcfe186590472058f8371bc5bc3652b)

---

### 2026-03-15 — BTPE binomial, hypergeometric/zipf/multinomial, RNG alignment

**Random**
- Implement NumPy-compatible BTPE binomial sampling algorithm.
  [6e6eb29](https://github.com/Dicklesworthstone/franken_numpy/commit/6e6eb29a8c12d7d7de92a098f3080f745d028176)
- Add LOGFACT lookup table and implement NumPy-compatible hypergeometric, zipf,
  and multinomial distributions.
  [e6e45aa](https://github.com/Dicklesworthstone/franken_numpy/commit/e6e45aaa783883ef9d361e304dddec4bd50269fe)
- Align bounded-integer and distribution algorithms with NumPy (Lemire method,
  buffered uint32 splitting).
  [7512f6c](https://github.com/Dicklesworthstone/franken_numpy/commit/7512f6c33fec5df0274ab5340a68f9fe47a88b14)
- Reject `choice()` from empty array when `size > 0`.
  [292448b](https://github.com/Dicklesworthstone/franken_numpy/commit/292448b964a377720eff804b3110fca1f4980887)
- Add bounds check and `debug_assert` to `logfactorial` for negative k.
  [8b916f8](https://github.com/Dicklesworthstone/franken_numpy/commit/8b916f8673d9552875560dde1db7782ca885a2dd)

---

### 2026-03-12 — Complex linalg storage, safe memmap, precision guards

**Ufunc**
- Add complex interleaved storage support for linalg operations.
  [c6cd0b2](https://github.com/Dicklesworthstone/franken_numpy/commit/c6cd0b2be0929e1fe8766b10e45a29dcb57ec8ae)
- Add precision guards for `Vec<f64>` storage bridge conversions to prevent
  silent truncation.
  [ab013e2](https://github.com/Dicklesworthstone/franken_numpy/commit/ab013e243c867b1067bd34cb23d1868cdcc4e1bc)

**Cross-crate**
- Add safe memmap support, complex linalg, RNG witness tests, and polynomial
  conformance tests.
  [cd96dfb](https://github.com/Dicklesworthstone/franken_numpy/commit/cd96dfbd6b931bc93a8aa05be2090204b51765bd)
- Fix structured dtype handling, NaN risk sanitization, and lossy fallback logic.
  [73846b9](https://github.com/Dicklesworthstone/franken_numpy/commit/73846b919870aa5cdd72e9ea72e1ff7f037685b3)

---

### 2026-03-10 — Float16 dtype, complex I/O, float-error state

**Dtype**
- Add `F16` (float16) dtype support via the `half` crate.
  [fd9fbd1](https://github.com/Dicklesworthstone/franken_numpy/commit/fd9fbd1d0816916a0b591d3ede6e2a049e98a2b2)

**I/O**
- Add complex number dtype support for NumPy-compatible binary I/O.
  [41cab7b](https://github.com/Dicklesworthstone/franken_numpy/commit/41cab7b3738a75ddfd884080a98f1bb23041756f)

**Runtime**
- Add float-error state scaffolding (`seterr`, `geterr`, `errstate`).
  [8f065df](https://github.com/Dicklesworthstone/franken_numpy/commit/8f065dfa9874ca64985471206f72a0daaaddd6b7)

---

### 2026-03-02 .. 2026-03-03 — Copy-on-write views, differential test suites, ArrayStorage bridge

**Ufunc**
- Add copy-on-write views with fine-grained memory overlap detection and
  `broadcast_to_view`.
  [becd0d9](https://github.com/Dicklesworthstone/franken_numpy/commit/becd0d9e0df9c881e545524d3b458ab444137c89)
- Add `Arc`-backed shared array views with strided addressing.
  [30815d5](https://github.com/Dicklesworthstone/franken_numpy/commit/30815d5608adc47bddc7690fbbd56cc97dcb97db)
- Route all array construction through typed `ArrayStorage` bridge (eliminates
  `Vec<u8>` reinterpretation).
  [f988b2a](https://github.com/Dicklesworthstone/franken_numpy/commit/f988b2a5c3b08d41af4ffadb047c03ac9dca6d31)
- Integer true-division now promotes to `f64`.
  [d734c97](https://github.com/Dicklesworthstone/franken_numpy/commit/d734c97f2015713c9c2604c51c2e6b1173ad82c2)

**Conformance — differential test suite expansion**
- Add FFT differential test suite with golden-value fixtures.
  [50655be](https://github.com/Dicklesworthstone/franken_numpy/commit/50655be961a81f470e7ed945add67d6ddc01f95d)
- Add polynomial differential test suite.
  [db60f94](https://github.com/Dicklesworthstone/franken_numpy/commit/db60f94f904c57d4d1862c0a29117fb332930a6a)
- Add string differential test suite and expand polynomial fixtures.
  [27aea42](https://github.com/Dicklesworthstone/franken_numpy/commit/27aea4265e69ddbd9b879a16578c541e5554f34b)
- Add masked array differential test suite.
  [7b15398](https://github.com/Dicklesworthstone/franken_numpy/commit/7b15398bad4d0aafc74950123e48ad283f78cdc8)
- Add datetime/timedelta64 differential test suite.
  [8baf998](https://github.com/Dicklesworthstone/franken_numpy/commit/8baf998ca39d01c186b81ca23336cd8eb5771710)
- Add RNG statistical distribution test suite with Kolmogorov-Smirnov tests.
  [ef13518](https://github.com/Dicklesworthstone/franken_numpy/commit/ef1351812ccd9d1cf5d5634fa61185eb827c8040)
- Add 8 linalg differential test cases for edge-case coverage and batched
  linalg differential tests.
  [c827489](https://github.com/Dicklesworthstone/franken_numpy/commit/c827489f04e2b19121b198a882446a1a0d89039f),
  [ed7e878](https://github.com/Dicklesworthstone/franken_numpy/commit/ed7e878caa86bb2622a856a5cd04820acec8e271)

**Cross-crate**
- Add `ArrayStorage` bridge APIs, enforce real NumPy oracle in CI, normalize
  formatting.
  [bc93d31](https://github.com/Dicklesworthstone/franken_numpy/commit/bc93d3155a707f749ddd611dc51afb5ab01c8580)

---

### 2026-02-26 — Structured dtypes, PCG64/MT19937 RNGs, Golub-Kahan SVD, polynomial families

**Dtype**
- Add structured dtype support.

**Random**
- Add PCG64DXSM and MT19937 (Mersenne Twister) bit generators with full
  `SeedSequence` seeding.

**Linalg**
- Add Golub-Kahan SVD algorithm, batched linalg operations.

**Ufunc**
- Add Hermite (physicist), Hermite (probabilist), and Laguerre polynomial
  families.
  [91ffa0e](https://github.com/Dicklesworthstone/franken_numpy/commit/91ffa0e4c95bcbf2cd4a46ddb09c01a491b4e1cb)
- Add polymorphic `ArrayStorage` with sort kind parameter, advanced gradient,
  higher-degree roots, and N-D trapezoid.
  [fd4e477](https://github.com/Dicklesworthstone/franken_numpy/commit/fd4e477f680ea3b67c7ffe38afb002f15059f197)
- Expand `searchsorted`, `isin`, `count_nonzero` APIs; massive ufunc and string
  coverage expansion.
  [5e6f6fa](https://github.com/Dicklesworthstone/franken_numpy/commit/5e6f6fa6d6372099c49a3c415e7b4e156e18be55)

**I/O**
- Add high-level I/O convenience functions; extend linalg norms and `funm`; new
  RNG distributions.
  [6f67d0a](https://github.com/Dicklesworthstone/franken_numpy/commit/6f67d0a464cc6eb14af0b86a3b5e3974f5be91f5)

**Cross-crate**
- Add complex arithmetic, structured dtypes, PCG64/MT19937 RNGs, Golub-Kahan
  SVD, batched linalg, ufunc refinements.
  [e6a73d6](https://github.com/Dicklesworthstone/franken_numpy/commit/e6a73d6940f67a98789ba033cf8f540fcabc7316)

---

### 2026-02-25 — Major ufunc rewrite, DEFLATE NPZ, big-endian I/O

**Ufunc**
- Major ufunc rewrite with expanded dtype dispatch, type coercion, and
  broadcasting paths.
  [b319933](https://github.com/Dicklesworthstone/franken_numpy/commit/b31993339c444697a31eeb22bd2967d427b50bc7)
- Expand ufunc module substantially and enhance linalg operations.
  [0566a48](https://github.com/Dicklesworthstone/franken_numpy/commit/0566a48568df454bb2d8d63b948df5c68f318ab3)
- Expand ufunc dispatch, linalg decompositions, I/O formats, and fix stride
  conformance.
  [9f57a60](https://github.com/Dicklesworthstone/franken_numpy/commit/9f57a6063ad31d842ae16c8a2030e2548e530caa)

**I/O**
- Add DEFLATE compression support for NPZ read/write.
  [8ce05d0](https://github.com/Dicklesworthstone/franken_numpy/commit/8ce05d07991f461e2d48e8664c63c1399213ba10)
- Add big-endian dtype support for all numeric types.
  [7cd4159](https://github.com/Dicklesworthstone/franken_numpy/commit/7cd4159e6e628e9c6511f2e1a3ee6f2e9afbcba0)

---

### 2026-02-21 — Rectangular SVD/QR, complex/string/datetime dtypes, Schur decomposition, NPZ support

**Linalg**
- Add rectangular SVD and QR decomposition for non-square matrices.
  [2f9c88d](https://github.com/Dicklesworthstone/franken_numpy/commit/2f9c88de8522f3080c9b41392667b1ee22bab1e3)
- Add Schur decomposition, cross product, Kronecker product, `multi_dot`, and
  Bessel functions.
  [55e5957](https://github.com/Dicklesworthstone/franken_numpy/commit/55e5957add3f14700e4c0b4a12179c17b862b0a0)
- Add eigenvectors, and LU factorization with multi-RHS solve and triangular
  solve.
  [ee46123](https://github.com/Dicklesworthstone/franken_numpy/commit/ee461234f69f0b48b9724930dfeeaa0602934fa8),
  [41a95cf](https://github.com/Dicklesworthstone/franken_numpy/commit/41a95cf57eba29fd422f21a98766a9fe1ae8a3e3)

**Dtype**
- Add complex, string, and datetime dtype variants.
  [3c7ee4a](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7ee4a63d144f8f3a0aa2341b05ecc9e12c0f6a)

**Ndarray**
- Add negative-stride support for reverse slicing.
  [2f9c88d](https://github.com/Dicklesworthstone/franken_numpy/commit/2f9c88de8522f3080c9b41392667b1ee22bab1e3)

**Random**
- Add noncentral distributions.
  [3c7ee4a](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7ee4a63d144f8f3a0aa2341b05ecc9e12c0f6a)

**Ufunc**
- Add multi-dimensional FFT (`fft2`, `ifft2`, `fftn`, `ifftn`).
  [ee46123](https://github.com/Dicklesworthstone/franken_numpy/commit/ee461234f69f0b48b9724930dfeeaa0602934fa8)
- Add NPZ archive support.
  [ee46123](https://github.com/Dicklesworthstone/franken_numpy/commit/ee461234f69f0b48b9724930dfeeaa0602934fa8)

---

### 2026-02-20 — Massive ufunc expansion (76+ operations), eig_nxn, SeedSequence, comprehensive edge-case fixes

**Ufunc — operation expansion** (largest single-day expansion)
- Add 22 unary ops, 5 binary ops, 4 reductions, and fix sum dtype promotion.
  [0c78354](https://github.com/Dicklesworthstone/franken_numpy/commit/0c783544ae59a5a13f1fb346c8f16e02dc8ceb86)
- Add 8 more unary ops: `cbrt`, `expm1`, `log1p`, `degrees`, `radians`,
  `rint`, `trunc`, `spacing`.
  [24cbc71](https://github.com/Dicklesworthstone/franken_numpy/commit/24cbc71778aa6b77f8281419169528a24f442646)
- Add 29 new ufunc operations (17 binary, 12 unary) plus `cumsum`, `cumprod`,
  and `clip`.
  [2d5b6a7](https://github.com/Dicklesworthstone/franken_numpy/commit/2d5b6a7e28d09522b23bd7866945ad5612c91483)
- Implement `var`, `std`, `argmin`, `argmax` reductions.
  [82cf6fc](https://github.com/Dicklesworthstone/franken_numpy/commit/82cf6fc76733b3c7c4dda95fcc4624327d04bf00)
- Expand reduction operations with cumulative and axis support.
  [31fd494](https://github.com/Dicklesworthstone/franken_numpy/commit/31fd49454e367253903c545c0f34a097daee5d72)

**Linalg**
- Add `eig_nxn` for general (non-symmetric) eigenvalue decomposition via
  Hessenberg + implicit shifted QR.
  [13950b9](https://github.com/Dicklesworthstone/franken_numpy/commit/13950b99f00b7314f612cdf68368ba7e7017ed95)

**Cross-crate**
- Comprehensive edge-case and correctness fixes across dtype, ndarray, random,
  I/O, and ufunc for NumPy behavioral parity.
  [bba14a3](https://github.com/Dicklesworthstone/franken_numpy/commit/bba14a3cbf6cab1a2b985f93b2ccd81483e4789c)
- Fix space-delimited file parsing, replace epsilon-based singularity checks
  with exact zero, fix cholesky division guard.
  [18d1e73](https://github.com/Dicklesworthstone/franken_numpy/commit/18d1e7384f2444e8e97a8d35ae3d88d721141ada)

**Random**
- Add RNG policy metadata validation, `BitGenerator.spawn`, and structured
  replay logging.
  [c0df45e](https://github.com/Dicklesworthstone/franken_numpy/commit/c0df45e38c4f00a6d7ebf1d6f65d644a5c63fe1a)
- Upgrade workspace dependencies and pin nightly toolchain to
  `nightly-2026-02-20`.
  [b0175c6](https://github.com/Dicklesworthstone/franken_numpy/commit/b0175c60b6c4d722a4297faeeb41865c62351535)

---

### 2026-02-19 — NPY binary I/O, SVD/QR decomposition, SeedSequence, BitGenerator facade

**I/O**
- Implement NPY binary format read/write with version (1.0/2.0) and pickle
  policy support.
  [a9fbabb](https://github.com/Dicklesworthstone/franken_numpy/commit/a9fbabbdab5046cf6a679d4c796647d57db23cb7)

**Linalg**
- Add SVD and QR decomposition.
  [bbad90f](https://github.com/Dicklesworthstone/franken_numpy/commit/bbad90ff34e5ed85017e956957e42446f18eed8c)
- Add 2x2 matrix decomposition fast paths and expand ufunc dispatch.
  [be2e02b](https://github.com/Dicklesworthstone/franken_numpy/commit/be2e02bf7ec1483551a08f10e7c4059609a167fd)

**Random**
- Add `BitGenerator` facade, state serialization, and `SeedSequence`
  snapshot/restore.
  [98f5bea](https://github.com/Dicklesworthstone/franken_numpy/commit/98f5bea1cf64e2a6a25affa2e6202c09ee37d4e6)
- Add `SeedSequence`, `SeedMaterial`, and `default_rng` constructor API.
  [cba0c3e](https://github.com/Dicklesworthstone/franken_numpy/commit/cba0c3e1a929a1d70e0488b07674b8ab536c7407)

**Conformance**
- Add high-rank adversarial ufunc broadcasting fixtures.
  [024d5f0](https://github.com/Dicklesworthstone/franken_numpy/commit/024d5f074650dd1adbf9352b8f01e99f5ddc3d0d)
- Add signed-axis reduction parity for ufunc packet.
  [8801c99](https://github.com/Dicklesworthstone/franken_numpy/commit/8801c99da7dfcb25c9199caf7731a074ff85dc5d)

---

### 2026-02-18 — Conformance corpus expansion, CI gate hardening, MIT+rider licensing

**Conformance**
- Expand conformance corpus across all 9 P2C packets with differential,
  metamorphic, and adversarial fixture suites for dtype (P2C-002), ufunc
  (P2C-005), RNG (P2C-007), linalg (P2C-008), and I/O (P2C-009).
  [837319b](https://github.com/Dicklesworthstone/franken_numpy/commit/837319bebbe71fbc60fb73663e130693d1760e5c),
  [96ab03b](https://github.com/Dicklesworthstone/franken_numpy/commit/96ab03bd3525728c69f8404c24272e0c139f189e),
  [720f108](https://github.com/Dicklesworthstone/franken_numpy/commit/720f1087ecdfed09a47debd05c333494e83b80a4),
  [faead61](https://github.com/Dicklesworthstone/franken_numpy/commit/faead6158532ca5fd73ed0cb9665410db9d567b6),
  [b5fe99f](https://github.com/Dicklesworthstone/franken_numpy/commit/b5fe99f13035761091761a658c388c64708a0bdb)
- Harden CI gate topology contract for mandatory G7/G8 checks.
  [893781c](https://github.com/Dicklesworthstone/franken_numpy/commit/893781c795d6bb1c431d8a0cfbfe42d85272644f)
- Require non-fallback oracle source for G3 gate in CI.
  [d42637b](https://github.com/Dicklesworthstone/franken_numpy/commit/d42637bc993c46f386600dccfd55b3652e563eea)
- Fix double-write race in packet005 evidence pack.
  [358c377](https://github.com/Dicklesworthstone/franken_numpy/commit/358c377dcab18399f95461f8a7358b085710dbda)

**Licensing**
- Adopt MIT license with OpenAI/Anthropic rider across all workspace crates.
  [adb03a5](https://github.com/Dicklesworthstone/franken_numpy/commit/adb03a54741afe4a6891301d25b9111b6fa83840)

---

### 2026-02-17 — Phase2c evidence packs, CI gate topology, hot-path optimization

**Conformance**
- Add CI gate topology and contract infrastructure with 8-gate system
  (G1-G8: build, unit, differential, metamorphic, adversarial, witness,
  security, RaptorQ).
  [921b7a4](https://github.com/Dicklesworthstone/franken_numpy/commit/921b7a4f9ad90dc5d98e1223db6a83deaa561ac3)
- Generate phase2c evidence packs for all 9 P2C packets with e2e replay
  scripts.
  [7fc60b4](https://github.com/Dicklesworthstone/franken_numpy/commit/7fc60b4fe5a550b44fc618b9b7c7cf56d708308e),
  [f68f25e](https://github.com/Dicklesworthstone/franken_numpy/commit/f68f25e81cbc5e87cd467e161e0275aa069e26ac)
- Expand differential, metamorphic, and adversarial test coverage across iter,
  I/O, linalg, RNG, and ufunc.
  [27840ee](https://github.com/Dicklesworthstone/franken_numpy/commit/27840ee4c75ba80fcd915dfec63e11a9480fd922)

**Performance**
- Optimize hot paths across fnp-io, fnp-iter, fnp-linalg, fnp-ndarray, and
  fnp-ufunc.
  [b605554](https://github.com/Dicklesworthstone/franken_numpy/commit/b605554023cefc9336fdf739d2bd3f5c3f65bc9c)

---

### 2026-02-16 — Ufunc dispatch build-out, RNG/linalg/IO contract validation, security gate

**Ufunc**
- Expand universal function subsystem with full dispatch table, type coercion
  matrix, and broadcasting engine.
  [5eb4246](https://github.com/Dicklesworthstone/franken_numpy/commit/5eb424659ddfa4caeab47c5c7430eca151153d5c)

**Random**
- Expand PRNG subsystem and refine ufunc dispatch.
  [6f8fdc8](https://github.com/Dicklesworthstone/franken_numpy/commit/6f8fdc809f822e92f4f567885ad08e6b182632d6)

**Conformance**
- Add linalg test contract framework with adversarial, differential, and
  metamorphic fixtures.
  [cb5ac2b](https://github.com/Dicklesworthstone/franken_numpy/commit/cb5ac2b128392e33691d9300211b0728f59b3676),
  [141792b](https://github.com/Dicklesworthstone/franken_numpy/commit/141792b3a0f177286ce3bfb1b2029384a6629e22)
- Add RNG differential, metamorphic, and adversarial fixture validation.
  [dc5c8c9](https://github.com/Dicklesworthstone/franken_numpy/commit/dc5c8c9ce5d52ee5765a012a301789bd633b3bfb)
- Add shape/stride differential testing and expand security/contract gates.
  [dd3a865](https://github.com/Dicklesworthstone/franken_numpy/commit/dd3a8652212267faa6e1c0c297aa32806b328c07)

**I/O**
- Implement P2C-009 I/O contract validation for NPY/NPZ formats.
  [c117674](https://github.com/Dicklesworthstone/franken_numpy/commit/c1176744a004236c9fc8388fe5019589ec286a38)

**Linalg**
- Implement P2C-008 linalg contract validation (shape, solver, decomposition,
  policy checks).
  [7a1d947](https://github.com/Dicklesworthstone/franken_numpy/commit/7a1d947729edd1aece3c64d8ca7fd05e7645949d)

---

### 2026-02-15 — Stride-tricks API, NDIter subsystem, RaptorQ durability gate, conformance infrastructure

**Ndarray**
- Add stride-tricks API: `as_strided`, `broadcast_to`, `sliding_window_view`
  with full conformance coverage and reliability gate.
  [d6241a9](https://github.com/Dicklesworthstone/franken_numpy/commit/d6241a9a45b21dc9f571917e35dfb3aaebbdf85e)
- Add shape/stride conformance cases and `coverage_ratio` improvements.
  [a67c25b](https://github.com/Dicklesworthstone/franken_numpy/commit/a67c25b2b91217792caceedd5ee458cd3559b070)

**Iter**
- Implement transfer selector, overlap detection, `flatiter`, and `nditer`
  subsystem with conformance tests.
  [e162403](https://github.com/Dicklesworthstone/franken_numpy/commit/e1624033d2024dd505f693d86925fe3ad715fc46)

**Conformance**
- Add RaptorQ artifact verification suite and gate binary (G8).
  [2582ac5](https://github.com/Dicklesworthstone/franken_numpy/commit/2582ac554f1c4d6b06aa46a97110a390846ceb4b)
- Add benchmarking telemetry and environment fingerprinting.
  [2f4b16f](https://github.com/Dicklesworthstone/franken_numpy/commit/2f4b16f389987d6022a66d4e196cde54d9cb93f7)
- Add reliability retry/flake-budget support to RaptorQ gate.
  [1a9a9c3](https://github.com/Dicklesworthstone/franken_numpy/commit/1a9a9c326848a4c2190d298bfca74fd4ca5c11ed)
- Expand workflow scenario gate and security gate with comprehensive
  orchestration.
  [82dc7d1](https://github.com/Dicklesworthstone/franken_numpy/commit/82dc7d1233415472f633d54b9ae3d823f349b7a6),
  [6996c2f](https://github.com/Dicklesworthstone/franken_numpy/commit/6996c2f444eb0eb357f75284f7a54ec9c0828e73)

**Dtype**
- Expand dtype promotion and parity verification infrastructure.
  [1f902e5](https://github.com/Dicklesworthstone/franken_numpy/commit/1f902e5b581f58deb05d2c8e2051fa6a5c95b17c)

---

### 2026-02-14 — Phase2c extraction ledger, test contracts, adversarial fixtures

**Conformance**
- Add differential conformance harness, test contracts, and full-parity
  doctrine.
  [3f2b977](https://github.com/Dicklesworthstone/franken_numpy/commit/3f2b977a3b7fc83cc5c4e8ca3c5086a3569af872)
- Add test contract system, adversarial fixtures, and essence extraction ledger.
  [278d62e](https://github.com/Dicklesworthstone/franken_numpy/commit/278d62e6fe3d37e3ff1a79eeee9c05bf654b50fe)
- Add user workflow scenario corpus with golden journey gates.
  [f3803b7](https://github.com/Dicklesworthstone/franken_numpy/commit/f3803b73bea003d6f28521ba82a9ef1df3c290ae)

**Docs**
- Register P2C-001 through P2C-009 phase2c artifacts in extraction ledger.
  [8ddf799](https://github.com/Dicklesworthstone/franken_numpy/commit/8ddf799d0026ca33940f1306a75668b5f3502929)

---

### 2026-02-13 — Initial commit: 9-crate workspace, SCE, Bayesian runtime, security gate

**Project bootstrap.** Initialize FrankenNumPy as a 9-crate Rust workspace with
clean-room NumPy port, conformance infrastructure, and durability artifacts.
[527bd9d](https://github.com/Dicklesworthstone/franken_numpy/commit/527bd9d11123e149377bb6edd5e9bc1a5ab3b70f)

**Core architecture established:**
- `fnp-dtype` — 18 dtype variants with NumPy-exact promotion table and 5 cast
  policies.
- `fnp-ndarray` — Stride Calculus Engine (SCE) with shape-to-strides, broadcast
  legality, reshape with `-1` inference, and view safety.
- `fnp-iter` — Transfer-loop selector state machine (stub).
- `fnp-ufunc` — Universal function dispatch framework (stub).
- `fnp-linalg` — Linear algebra decompositions (stub).
- `fnp-random` — PRNG subsystem (stub).
- `fnp-io` — NPY/NPZ binary I/O (stub).
- `fnp-runtime` — Dual-mode runtime (strict/hardened) with Bayesian decision
  engine.
- `fnp-conformance` — Differential conformance harness with test contracts.

**Infrastructure:**
- Phase2c contract schema lock and Bayesian evidence ledger.
  [264afa4](https://github.com/Dicklesworthstone/franken_numpy/commit/264afa4e5fcd82befd4fc501ea08e864b8fffd19)
- Security gate suite.
  [952a123](https://github.com/Dicklesworthstone/franken_numpy/commit/952a1232fbe6b6546aed375277d2a1309e7c4552)
- Replace per-element unravel/ravel axis reduction with contiguous kernel.
  [d9cfe90](https://github.com/Dicklesworthstone/franken_numpy/commit/d9cfe90b001463d852d8b07284d9965851ab26bb)

---

## Workspace Crate Summary

| Crate | Purpose | Tests |
|---|---|---|
| `fnp-dtype` | 18 dtype variants, promotion table, cast policies | 116 |
| `fnp-ndarray` | Stride Calculus Engine (SCE), shape transforms | 54 |
| `fnp-iter` | Transfer selector, overlap, flatiter, nditer | 79 |
| `fnp-ufunc` | Array ops, math, sort, polynomials, FFT, einsum (~30k lines) | 1,274 |
| `fnp-linalg` | 93 public functions: decompositions, solvers, batch ops | 199 |
| `fnp-random` | PCG64DXSM, MT19937, 40 oracle-verified distributions | 182 |
| `fnp-io` | NPY/NPZ binary, text I/O, DEFLATE compression | 143 |
| `fnp-conformance` | Differential, metamorphic, adversarial, witness suites | 122 |
| `fnp-runtime` | Strict/hardened modes, Bayesian decision engine | 52 |
| **Total** | | **2,221** |

## Project Metadata

- **Repository:** <https://github.com/Dicklesworthstone/franken_numpy>
- **Created:** 2026-02-13
- **License:** MIT (with OpenAI/Anthropic rider)
- **Rust edition:** 2024 (nightly, pinned to `nightly-2026-02-20`)
- **Workspace version:** `0.1.0`
- **Releases:** None (pre-release development)
- **Primary authors:** Dicklesworthstone (192 commits), Jeff Emanuel (18 commits)
