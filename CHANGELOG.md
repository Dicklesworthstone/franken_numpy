# Changelog

All notable changes to FrankenNumPy are documented in this file.

FrankenNumPy is a memory-safe, clean-room Rust reimplementation of NumPy. The
workspace version is `0.1.0` (pre-release). There are no tagged releases or
GitHub Releases yet, although the May 2026 "Workspace metadata + crates.io
publish-readiness" wave (see section below) completed the metadata side of
publishing — what remains for a tag is the explicit decision to publish.
Every entry below maps to a date range on the `main` branch. Representative
commits link to
`https://github.com/Dicklesworthstone/franken_numpy/commit/<hash>`.

The sections below are organized by **capability area** rather than diff order,
so that readers can quickly find what changed in the subsystem they care about.

---

## [Unreleased] — development head (2026-02-13 through 2026-05-16)

**Headline milestone (2026-05-13):** `fnp-python` now covers **100% of
`numpy.__all__`** (499/499 names), structurally locked by the
`fnp_python_covers_full_numpy_all` conformance test in
`crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs` so
any regression fails CI. See `audit_numpy_reality.md` for the 43.3% →
100% progression.

`main` branch, no formal release tags yet. After 2026-03-21 the entries
below summarize work by capability area rather than by date range,
since many commits ship across multiple areas in a single session.

### NumPy surface parity (April–May 2026)

Closed the long-standing gap between the README's "drop-in NumPy
compatibility" claim and the actual reachable surface of `fnp_python`.
Coverage progression:

| Date       | Covered / 499 | Share  |
|------------|--------------:|-------:|
| 2026-04-22 |          216  |  43.3% |
| 2026-05-09 |         ~480  |  ~96%  |
| 2026-05-13 |          499  | **100.0%** |

The 100% baseline is structurally enforced by the conformance test
`fnp_python_covers_full_numpy_all` (in
`crates/fnp-python/tests/conformance_remaining_top_level_attrs.rs`),
which iterates `numpy.__all__` at run time and fails CI if any name
regresses. See [`audit_numpy_reality.md`](audit_numpy_reality.md) for
the architectural choices that drove the close-out.

Representative parity-wave bead IDs (each shipped its own focused
conformance file under `crates/fnp-python/tests/conformance_*.rs`):

  - `wcoth` — `savetxt` + `tofile`
  - `86iws` — `recfunctions_merge_arrays` single-field flattening fix
  - `qe1i5` — `count_nonzero` scalar return type
  - `ydixp` — `unique_values` numpy 2.x set semantics
  - `vek3z` / `ghsx4` / `bntjh` — Array API: `unstack`, `permute_dims`, `vecdot`
  - `tmg0c` — `fromregex`, `min_scalar_type`, `get_printoptions`, `mintypecode`
  - `4t0ql` — `numpy.strings` submodule
  - `xdxvn` — `numpy.char` / `numpy.rec` / `numpy.emath` / `numpy.matrixlib`
  - `r1xmi` — `numpy.mgrid` / `numpy.ogrid`
  - `cp8xw` — `s_` / `index_exp` / `newaxis` / `False_` / `True_` /
            `errstate` / `seterr` family / `show_config` / `show_runtime` / `info`
  - `t3eb4` — `iterable`, `ndim`, `size`, `packbits`, `unpackbits`,
            `fromfunction`, `pow` (ufunc alias), `typecodes`, `typename`,
            `sctypeDict`, `ScalarType`, `typing` submodule, `ctypeslib`
            submodule, `test`, `getbufsize`, `nested_iters`, `from_dlpack`
  - `dm9bn` — `numpy.core` and `numpy.f2py` (final 2) and the lock-in test
  - `0xpje` — `fnp_python.__doc__` + `fnp_python.__numpy_version__`

### Diagnostic parity wave (May 2026)

Epic `33vtd` ("Diagnostic parity and upstream drift wave") shipped the
diagnostic oracle schema, dtype/shape diagnostic gates, IO parser
diagnostic gates, an fnp-python warnings/exceptions shard, a
cross-version NumPy drift matrix, the divergence ledger doc and checker,
the structured-dtype + recfunctions oracle corpus expansion, and an
agent-readable validation recipe selector. See `docs/DIVERGENCES.md`
and `crates/fnp-conformance/src/divergence_ledger.rs`.

### Fuzz infrastructure expansion (May 2026)

The fuzz harness grew from 4 crates to **7** and from mostly-unseeded
targets to **~200 curated edge-case seeds across 27 fuzz targets** (see
[`docs/FUZZING.md`](docs/FUZZING.md) for the live inventory). All fuzz
crates live under `crates/fnp-*/fuzz` and are excluded from the main
workspace so normal `cargo` invocations don't pull in `libfuzzer-sys`.
Representative seeding beads:

  - `62oir` — new `fnp-random/fuzz` (Pcg64Rng/Pcg64DxsmRng seed entropy,
    SeedSequence::new) and `fnp-linalg/fuzz` (cholesky_nxn, det_nxn,
    qr_mxn for arbitrary f64 matrices up to 16-32 dim).
  - `aaq0g` — 32 curated seeds for the new fuzz crates: u64-seed edge
    cases (0/1/u64::MAX/alt-bits), SeedSequence entropy variants,
    SPD Hilbert matrix, singular matrix, NaN/±Inf, subnormal,
    rectangular m×n QR shapes.
  - `s46p2` — 45 seeds for existing byte-driven fnp-dtype/fnp-io
    targets (fuzz_dtype_parse, fuzz_min_scalar_type, fuzz_fromstring,
    fuzz_loadtxt) covering dtype strings (i4/<f8/V16/U10/datetime64
    units), f64 specials, text-parsing edge cases.
  - `8fftx` — 19 Arbitrary-format seeds for fnp-dtype Arbitrary-derive
    targets (fuzz_can_cast, fuzz_result_type) covering same-kind
    upcasts, narrowing, complex drop, datetime/timedelta aliasing.
  - `cv45i` — 31 seeds for fnp-ufunc string parsers
    (fuzz_parse_gufunc_signature, fuzz_datetime_unit_parse) covering
    matmul/dot/outer/scalar-vector + every canonical datetime64 unit.
  - `i8ipt` — 15 seeds for fnp-ndarray fuzz_broadcast_shape and
    fuzz_fix_unknown_dim covering identity/scalar/3D/zero-dim cases
    and -1-placeholder reshape edge cases.
  - `y3dhc` — 46 NPY/NPZ binary seeds for fnp-io binary parsers
    (fuzz_npy, fuzz_npz, fuzz_load_auto, fuzz_header) generated via
    numpy.save / numpy.savez / numpy.savez_compressed plus
    handcrafted malformed inputs (truncated magic, future version,
    corrupt ZIP EOCD).
  - `diqz3` — 11 seeds for fnp-iter/fuzz_ndindex covering empty/1D-8D
    shape boundaries and the 100k-cap edge.
  - `m5y8s` — 15 seeds for fnp-ufunc/fuzz_parse_fixed_signature
    covering signature shapes (unary/binary/multi-output, all dtype
    letters) and adversarial cases (no arrow, double arrow, unicode).

The seed corpora are committed under each `fuzz/corpus/<target>/seed_*`
per the existing .gitignore exemption.

### Workspace metadata + crates.io publish-readiness (May 2026)

13-bead consolidation wave that prepared the workspace for a future
crates.io publish:

  - `j65bv` — 9 of 10 crates now inherit `version` / `edition` /
    `license-file` from `[workspace.package]` (the tenth, fnp-python,
    already did).
  - `dqoz0` — every crate has a per-crate `description` field.
  - `7k1h3` — `[workspace.package]` gained `repository`, `homepage`,
    `readme`, `keywords`, `categories`.
  - `gto3b` — each crate wires the 5 new fields via `.workspace = true`.
  - `dcjb7` + `voj6z` — criterion versions reduced from 3 (0.5/0.6/0.8.2)
    to 1 (0.6), plus the `criterion::black_box` → `std::hint::black_box`
    deprecation cleanup. -144 lines from Cargo.lock as the criterion-plot
    chain dropped out.
  - `0ovft` / `fowyr` / `ndzpa` — `serde_json`, `serde`, and `criterion`
    promoted to `[workspace.dependencies]`; each crate references via
    `.workspace = true`.
  - `ekzhv` — Cargo.toml profile comment moved back to its rightful home
    after the workspace.dependencies block was inserted.
  - `ihevd` — `rust-toolchain.toml` pinned to `nightly-2026-02-20` to
    match the CI workflow exactly.
  - `62b26` — `release-perf` profile drops redundant `opt-level=3` and
    `strip=false` (already implied by `inherits=release`).
  - `ypb8t` — `ci.yml` consolidates 8 hard-coded `nightly-2026-02-20`
    toolchain values to a single `env.RUST_TOOLCHAIN` variable.
  - `lmhu7` — `AGENTS.md` gains a "Current state (2026-05-16)" section
    pointing at audit_numpy_reality / FEATURE_PARITY / audit_numpy_mocks.
  - `nvtyj` / `8mcj6` — `docs/FUZZING.md` onboarding doc and
    headline-count refresh (7 fuzz crates / 27 targets / ~200 curated
    seed files); linked from README.

### Diagnostic parity and divergence ledger (April–May 2026)

Epic `33vtd` shipped a structured diagnostic-oracle pipeline so that
every warning, exception, and printed message we emit can be compared
against the live NumPy reference. Notable pieces:

  - **Diagnostic oracle schema + harness** under
    `crates/fnp-conformance/src/diagnostic_oracle.rs` and the
    `run_diagnostic_oracle` binary, with shards for dtype/shape, IO
    parsers, and fnp-python warnings/exceptions.
  - **Cross-version drift matrix** that captures the same fixture set
    against multiple NumPy versions to detect upstream behavior shifts
    (`run_oracle_drift_matrix`).
  - **Divergence ledger** at [`docs/DIVERGENCES.md`](docs/DIVERGENCES.md)
    plus a machine-readable checker
    (`cargo run -p fnp-conformance --bin run_divergence_ledger --
    --fail-on-missing`). A diagnostic case can only be marked
    `intentional_divergence` if it references a row in the ledger;
    otherwise the gate fails closed. Active rows: **1** (as of
    2026-05-16) — `franken_numpy-ucc2o` records that `fnp-random`'s
    `SeedMaterial::None` / no-seed `default_rng()` uses a fixed
    deterministic seed (`DEFAULT_RNG_SEED = 0xC0DE_CAFE_F00D_BAAD`)
    rather than OS entropy, pending a decision on adding `getrandom`
    as the crate's first external dependency.
  - **Structured-dtype + recfunctions oracle corpus** expanded so
    structured-dtype edge cases (record arrays, mixed-endian fields,
    inner shapes) now get per-warning, per-exception, per-field
    parity coverage.
  - **Validation recipe selector** (`run_validation_recipe_selector`)
    routes a fixture/case through the correct gate composition,
    making agent-driven diagnostic checks one command away.

### Documentation correctness polish wave (May 2026)

A documentation polish push spanning 2026-05-16 to 2026-05-17:
92 beads closed on 2026-05-16 (the verified spike that the README's
Project Timeline calls out) plus 18 follow-up beads on 2026-05-17,
for ~110 beads total. Touched README.md, AGENTS.md,
FEATURE_PARITY.md, audit_numpy_reality.md, audit_numpy_mocks.md,
docs/DIVERGENCES.md, docs/FUZZING.md, PROPOSED_ARCHITECTURE.md and
this CHANGELOG. (Counts reproducible from `.beads/issues.jsonl`:
`grep -oE '"closed_at":"2026-05-1[67]T[^"]*"' .beads/issues.jsonl
| wc -l` returns 110.) The wave was aimed at removing claims that
had drifted past the implementation, and at adding precision where
prose was vague. Representative entries:

  - README hero subtitle now states **concrete numbers** (6,392 tests,
    100% of `numpy.__all__`, zero unsafe blocks) instead of "extensive
    conformance coverage" (`h0n92`); MIT license badge clarified to
    MIT+Rider (`obyab`); Rust pin badge now names `nightly-2026-02-20`
    explicitly (`75nl6`).
  - **All 5 production bit generators** enumerated in the README
    `Random` row (`2gah9`); the "None (random)" RNG claim corrected to
    reflect the fixed deterministic default seed (`81e5e`); the
    "zero-dep" claim refined to "no external crates.io deps, intra-
    workspace fnp-ndarray only" (`vtt6c`).
  - **All 18 FFT entry points** (including hfft/ihfft/rfft2/rfftn/
    irfft2/irfftn) enumerated in both the README API Surface row
    (`s4t0g`) and FEATURE_PARITY (`ng0nc`); convolve modes
    full/same/valid documented (`g2bra`); correlate2d/convolve2d
    symmetry made explicit (`biwgf`).
  - README **Threat Model** restored the missing
    `linalg_policy_unknown_metadata` row (12 claimed vs 11 listed →
    12/12) (`7e9ol`); README API Surface now catalogs **all 12 PyO3
    classes** registered by `fnp-python` (`5dfb6`); README Repository
    Layout and AGENTS.md Repository Layout both restored the missing
    `fnp-python` crate row (`0zk2h`, `be5p9`, `rcno9`).
  - README Installation gained a "For Python access" subsection with
    cdylib build hint tightened for Linux / macOS / Windows file
    extensions (`bndjp`, `mnbz7`).
  - AGENTS.md gained a "Current state (2026-05-16)" section that
    surfaces the live divergence count and links the audit documents
    (`lmhu7`, `yf0yl`), plus a testing cost hint so contributors know
    `fnp-ufunc`/`fnp-python` dominate workspace test time (`saivr`).
  - `audit_numpy_reality.md` pinned its vision quote to the exact
    README anchor (`0yllc`) and surfaced the CI lock-in mechanism in
    its headline tagline (`5hvay`); `audit_numpy_mocks.md` clarified
    that the 115-unwrap row is fixture-only, not production crates
    (`8iikg`), added the `println!`/`eprintln!`/`dbg!` matrix row
    (`jr0hg`), and named all 10 crates explicitly (`mf2v3`).
  - [`docs/adr/ADR-001-parity-pivot.md`](docs/adr/ADR-001-parity-pivot.md)
    captures the 2026-04-10 proposal to pivot from parity grinding to
    Phase 3 (FFI / BLAS / threading); a 2026-05-16 status update
    notes that the parity track the ADR considered pivoting *away
    from* has since reached 100% `numpy.__all__` coverage, so the
    empirical case has materially changed.

### Pre-2026-03-21 details preserved below

The dated capability sections below are unchanged; they cover the
2026-02-13 through 2026-03-21 window in detail.

---

### Array Operations (`fnp-ufunc`)

The ufunc crate is the largest subsystem (~60,000 lines, 2,191 tests). It grew
from a stub on day one to 35 binary operations, 42 unary operations, and 22+
reductions.

#### Operation Expansion

- **76+ ufunc operations in a single day (2026-02-20).** Add 22 unary ops, 5
  binary ops, 4 reductions, and fix `sum` dtype promotion; then 8 more unary ops
  (`cbrt`, `expm1`, `log1p`, `degrees`, `radians`, `rint`, `trunc`, `spacing`);
  then 29 more (17 binary, 12 unary) plus `cumsum`, `cumprod`, and `clip`.
  [0c78354](https://github.com/Dicklesworthstone/franken_numpy/commit/0c783544ae59a5a13f1fb346c8f16e02dc8ceb86),
  [24cbc71](https://github.com/Dicklesworthstone/franken_numpy/commit/24cbc71778aa6b77f8281419169528a24f442646),
  [2d5b6a7](https://github.com/Dicklesworthstone/franken_numpy/commit/2d5b6a7e28d09522b23bd7866945ad5612c91483)
- Implement `var`, `std`, `argmin`, `argmax` reductions.
  [82cf6fc](https://github.com/Dicklesworthstone/franken_numpy/commit/82cf6fc76733b3c7c4dda95fcc4624327d04bf00)
- Expand reduction operations with cumulative and axis support.
  [31fd494](https://github.com/Dicklesworthstone/franken_numpy/commit/31fd49454e367253903c545c0f34a097daee5d72)
- Expand ufunc dispatch, type coercion, and broadcasting (2026-02-16).
  [5eb4246](https://github.com/Dicklesworthstone/franken_numpy/commit/5eb424659ddfa4caeab47c5c7430eca151153d5c)
- Major ufunc rewrite with expanded dtype dispatch paths (2026-02-25).
  [b319933](https://github.com/Dicklesworthstone/franken_numpy/commit/b31993339c444697a31eeb22bd2967d427b50bc7),
  [0566a48](https://github.com/Dicklesworthstone/franken_numpy/commit/0566a48568df454bb2d8d63b948df5c68f318ab3),
  [9f57a60](https://github.com/Dicklesworthstone/franken_numpy/commit/9f57a6063ad31d842ae16c8a2030e2548e530caa)
- Expand `searchsorted`, `isin`, `count_nonzero` APIs; massive string coverage
  expansion.
  [5e6f6fa](https://github.com/Dicklesworthstone/franken_numpy/commit/5e6f6fa6d6372099c49a3c415e7b4e156e18be55)
- Add `masked_sort_order` helper and additional array functions (2026-03-21).
  [5d166d6](https://github.com/Dicklesworthstone/franken_numpy/commit/5d166d6325b4ae82931baaa0f5032a62ed5f6d16),
  [3ed4487](https://github.com/Dicklesworthstone/franken_numpy/commit/3ed44873d78346240f7baa6baf763a51402f0829)
- Expand ufunc, linalg, runtime, and iterator implementations (+57 tests).
  [752082f](https://github.com/Dicklesworthstone/franken_numpy/commit/752082fc994da1bbbbd29c99b24c000f6e111351)

#### NaN Propagation (systematic correctness sweep, 2026-03-16)

- `reduce_min`, `reduce_max`: propagate NaN instead of skipping.
  [4891c64](https://github.com/Dicklesworthstone/franken_numpy/commit/4891c640a288efda6bc73acc03301d9a095cff76)
- `ptp` (peak-to-peak): propagate NaN for both `UFuncArray` and `MaskedArray`.
  [2fec037](https://github.com/Dicklesworthstone/franken_numpy/commit/2fec037f30db8b9743d0caca13993fafc8b97be6)
- `cummin`, `cummax`, `mode`, `heapsort`: propagate NaN through accumulators.
  [2a82df1](https://github.com/Dicklesworthstone/franken_numpy/commit/2a82df1f0e31ade8f7773da00934f10028dd9ac0)
- `median`, `percentile`, `quantile`: NaN early-returns before sorting.
  [6f3a4ae](https://github.com/Dicklesworthstone/franken_numpy/commit/6f3a4ae4a3f50b850aa9bc5313a3ab42b0b49b42)

#### Memory Views and Overlap Detection

- Add copy-on-write views with fine-grained memory overlap detection and
  `broadcast_to_view` (2026-03-03).
  [becd0d9](https://github.com/Dicklesworthstone/franken_numpy/commit/becd0d9e0df9c881e545524d3b458ab444137c89)
- Add `Arc`-backed shared array views with strided addressing.
  [30815d5](https://github.com/Dicklesworthstone/franken_numpy/commit/30815d5608adc47bddc7690fbbd56cc97dcb97db)
- Harden memory overlap detection, broadcasting, and eliminate `unwrap` panics
  (2026-03-20).
  [3cff47a](https://github.com/Dicklesworthstone/franken_numpy/commit/3cff47ae9bb2d88422755a7642df0a3d6cb031de)
- Reject out-of-bounds `as_strided` views instead of zero-filling.
  [e771acb](https://github.com/Dicklesworthstone/franken_numpy/commit/e771acb1b63295b3776c65b0ade2e9d3f56e536e)
- Add overlap detection, readonly view tracking, and reduction edge-case
  handling (2026-03-18).
  [4326f30](https://github.com/Dicklesworthstone/franken_numpy/commit/4326f30eecd5920e53866d1347f014b446a4864c)

#### ArrayStorage Bridge

- Route all array construction through typed `ArrayStorage` bridge, eliminating
  `Vec<u8>` reinterpretation (2026-03-02).
  [f988b2a](https://github.com/Dicklesworthstone/franken_numpy/commit/f988b2a5c3b08d41af4ffadb047c03ac9dca6d31),
  [6829729](https://github.com/Dicklesworthstone/franken_numpy/commit/6829729024474af709fe9c081a41051c1babd21e)
- Add `ArrayStorage` bridge APIs and enforce real NumPy oracle in CI.
  [bc93d31](https://github.com/Dicklesworthstone/franken_numpy/commit/bc93d3155a707f749ddd611dc51afb5ab01c8580)
- Add polymorphic `ArrayStorage` with sort kind parameter, advanced gradient,
  higher-degree roots, and N-D trapezoid.
  [fd4e477](https://github.com/Dicklesworthstone/franken_numpy/commit/fd4e477f680ea3b67c7ffe38afb002f15059f197)
- Add precision guards for `Vec<f64>` storage bridge conversions to prevent
  silent truncation.
  [ab013e2](https://github.com/Dicklesworthstone/franken_numpy/commit/ab013e243c867b1067bd34cb23d1868cdcc4e1bc)
- Add complex interleaved storage support for linalg operations.
  [c6cd0b2](https://github.com/Dicklesworthstone/franken_numpy/commit/c6cd0b2be0929e1fe8766b10e45a29dcb57ec8ae)

#### Polynomial Families

- Add Hermite (physicist and probabilist) and Laguerre polynomial operations.
  [91ffa0e](https://github.com/Dicklesworthstone/franken_numpy/commit/91ffa0e4c95bcbf2cd4a46ddb09c01a491b4e1cb)

#### FFT

- Add multi-dimensional FFT (`fft2`, `ifft2`, `fftn`, `ifftn`) and 1-D
  transforms (`fft`, `ifft`, `rfft`, `irfft`).
  [ee46123](https://github.com/Dicklesworthstone/franken_numpy/commit/ee461234f69f0b48b9724930dfeeaa0602934fa8)
- Implement optimal `multi_dot` chain multiplication via dynamic programming
  and add FFT metamorphic tests (2026-03-19).
  [38cb001](https://github.com/Dicklesworthstone/franken_numpy/commit/38cb00133d0b9d01399ad54d95173f81b6962205)

#### Least Squares and Type Promotion

- Expand `lstsq` to return the full NumPy 4-tuple
  `(x, residuals, rank, singular_values)` (2026-03-21).
  [f9d51d0](https://github.com/Dicklesworthstone/franken_numpy/commit/f9d51d052fedc2cb5f7cbd5320da6c072eda549e)
- Integer true-division now promotes to `f64`.
  [d734c97](https://github.com/Dicklesworthstone/franken_numpy/commit/d734c97f2015713c9c2604c51c2e6b1173ad82c2)

#### String Operations

- Implement Python-compatible whitespace `split`/`rsplit` with `maxsplit`
  (2026-03-21).
  [85789d2](https://github.com/Dicklesworthstone/franken_numpy/commit/85789d27a5a23e5998db0eeeee49a7b5c6771d85)

---

### Linear Algebra (`fnp-linalg`)

93 public functions across 2x2 fast paths, general NxN algorithms, spectral
methods, batch operations, and complex number support.

#### Core Decompositions

- Add SVD and QR decomposition (2026-02-19).
  [bbad90f](https://github.com/Dicklesworthstone/franken_numpy/commit/bbad90ff34e5ed85017e956957e42446f18eed8c)
- Add 2x2 matrix decomposition fast paths (`solve_2x2`, `det_2x2`, `inv_2x2`,
  `qr_2x2`, `svd_2x2`, `eigh_2x2`, `cholesky_2x2`).
  [be2e02b](https://github.com/Dicklesworthstone/franken_numpy/commit/be2e02bf7ec1483551a08f10e7c4059609a167fd)
- Add `eig_nxn` for general (non-symmetric) eigenvalue decomposition via
  Hessenberg + implicit shifted QR (2026-02-20).
  [13950b9](https://github.com/Dicklesworthstone/franken_numpy/commit/13950b99f00b7314f612cdf68368ba7e7017ed95)
- Add LU factorization, multi-RHS solve, and triangular solve (2026-02-21).
  [41a95cf](https://github.com/Dicklesworthstone/franken_numpy/commit/41a95cf57eba29fd422f21a98766a9fe1ae8a3e3)
- Add Schur decomposition, cross product, Kronecker product, `multi_dot`, and
  Bessel functions.
  [55e5957](https://github.com/Dicklesworthstone/franken_numpy/commit/55e5957add3f14700e4c0b4a12179c17b862b0a0)
- Add rectangular SVD and QR decomposition for non-square matrices.
  [2f9c88d](https://github.com/Dicklesworthstone/franken_numpy/commit/2f9c88de8522f3080c9b41392667b1ee22bab1e3)
- Add eigenvectors and multi-dimensional FFT support.
  [ee46123](https://github.com/Dicklesworthstone/franken_numpy/commit/ee461234f69f0b48b9724930dfeeaa0602934fa8)
- Add Golub-Kahan SVD algorithm and batched linalg operations (2026-02-26).
  [e6a73d6](https://github.com/Dicklesworthstone/franken_numpy/commit/e6a73d6940f67a98789ba033cf8f540fcabc7316)
- Add reduced SVD support via `svd_full(full_matrices)` parameter (2026-03-21).
  [ad079d8](https://github.com/Dicklesworthstone/franken_numpy/commit/ad079d86e3d0a94c824b91de0644086b24c0da7e)

#### Correctness Fixes

- Fix eigenvalue sort order to ascending to match NumPy convention (2026-03-21).
  [5c57a67](https://github.com/Dicklesworthstone/franken_numpy/commit/5c57a6779122ab621786857e886ffd27b4d713b8),
  [0ef8b20](https://github.com/Dicklesworthstone/franken_numpy/commit/0ef8b201914609f9d7a8a63a9312dbdb2fdf39c7)
- Fix SVD convergence, reduction shapes, and RNG state width.
  [c89e90c](https://github.com/Dicklesworthstone/franken_numpy/commit/c89e90c2db4693d798c23abf635537c744539ff7)
- Rewrite `inv_nxn` to use `lu_solve_multi` and optimize `cholesky_solve_multi`
  (2026-03-20).
  [062be4e](https://github.com/Dicklesworthstone/franken_numpy/commit/062be4eed99ab1af2b4133e9954620d2c34f0bde)
- Fix space-delimited file parsing, replace epsilon-based singularity checks
  with exact zero, fix Cholesky division guard (2026-02-20).
  [18d1e73](https://github.com/Dicklesworthstone/franken_numpy/commit/18d1e7384f2444e8e97a8d35ae3d88d721141ada)

#### Complex Number Support

- Add safe memmap, complex linalg, RNG witness tests, and polynomial
  conformance tests (2026-03-12).
  [cd96dfb](https://github.com/Dicklesworthstone/franken_numpy/commit/cd96dfbd6b931bc93a8aa05be2090204b51765bd)

#### Oracle Parity Tests

- Add NumPy oracle parity tests for `det`, `inv`, `solve`, `norm`, `eig`,
  `svd`, `cholesky`, `qr` (2026-03-16).
  [7b51f50](https://github.com/Dicklesworthstone/franken_numpy/commit/7b51f50b616427b8d5a04b365e3c7b8e316cb6cc)

---

### Random Number Generation (`fnp-random`)

Bit-exact parity with NumPy across 40+ oracle-verified distributions,
**five production bit generators** (PCG64, PCG64DXSM, MT19937, Philox,
SFC64) plus an internal `DeterministicRng` for tests, and the full
`SeedSequence` hierarchy.

#### Bit Generators

- Add `SeedSequence`, `SeedMaterial`, and `default_rng` constructor API
  (2026-02-19).
  [cba0c3e](https://github.com/Dicklesworthstone/franken_numpy/commit/cba0c3e1a929a1d70e0488b07674b8ab536c7407)
- Add `BitGenerator` facade, state serialization, and `SeedSequence`
  snapshot/restore.
  [98f5bea](https://github.com/Dicklesworthstone/franken_numpy/commit/98f5bea1cf64e2a6a25affa2e6202c09ee37d4e6)
- Add RNG policy metadata validation, `BitGenerator.spawn`, and structured
  replay logging (2026-02-20).
  [c0df45e](https://github.com/Dicklesworthstone/franken_numpy/commit/c0df45e38c4f00a6d7ebf1d6f65d644a5c63fe1a)
- Add PCG64DXSM and MT19937 (Mersenne Twister) bit generators with full
  `SeedSequence` seeding (2026-02-26).
  [e6a73d6](https://github.com/Dicklesworthstone/franken_numpy/commit/e6a73d6940f67a98789ba033cf8f540fcabc7316)
- Implement Philox counter-based RNG (CBRNG) with full state serialization
  and deserialization (2026-03-21).
  [ee17681](https://github.com/Dicklesworthstone/franken_numpy/commit/ee1768141ce11f8a23228e4a351efef539a04503),
  [509055b](https://github.com/Dicklesworthstone/franken_numpy/commit/509055b1ed331590db3c21ab86ca7344a6dd42ce)
- Add SFC64 small-fast-counting bit generator with NumPy-parity output
  and `SeedSequence` seeding, completing the production bit-generator
  set to 5 (PCG64, PCG64DXSM, MT19937, Philox, SFC64). Landed in the
  same commit as the Philox addition above (2026-03-21). The set is
  enumerated under `pub enum BitGeneratorKind` in `crates/fnp-random/src/lib.rs`.
  Pickle state-restore for Philox/SFC64 hardened to require all state
  entries present (2026-03-25).
  [ee17681](https://github.com/Dicklesworthstone/franken_numpy/commit/ee1768141ce11f8a23228e4a351efef539a04503),
  [2d358b2](https://github.com/Dicklesworthstone/franken_numpy/commit/2d358b2de785c7a1e1f90f61bfe47157f6e16cbf)
- Add RNG state serialization (`get_state` / `set_state`) and harden `.npy`
  header parsing against allocation bombs.
  [9894042](https://github.com/Dicklesworthstone/franken_numpy/commit/989404292d75fc5c103eb0f1ae08b99488c84b94)

#### Distribution Algorithms

- Implement NumPy-compatible BTPE binomial sampling algorithm (2026-03-15).
  [6e6eb29](https://github.com/Dicklesworthstone/franken_numpy/commit/6e6eb29a8c12d7d7de92a098f3080f745d028176)
- Add LOGFACT lookup table and implement NumPy-compatible hypergeometric, zipf,
  and multinomial distributions.
  [e6e45aa](https://github.com/Dicklesworthstone/franken_numpy/commit/e6e45aaa783883ef9d361e304dddec4bd50269fe)
- Rewrite Poisson sampling with PTRS algorithm; add `random_loggam` (2026-03-16).
  [b7af286](https://github.com/Dicklesworthstone/franken_numpy/commit/b7af2866cbcfe186590472058f8371bc5bc3652b)
- Add noncentral distributions (2026-02-21).
  [3c7ee4a](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7ee4a63d144f8f3a0aa2341b05ecc9e12c0f6a)
- Expand PRNG subsystem and refine ufunc dispatch (2026-02-16).
  [6f8fdc8](https://github.com/Dicklesworthstone/franken_numpy/commit/6f8fdc809f822e92f4f567885ad08e6b182632d6)

#### Bounded Integer and Sampling

- Align bounded-integer and distribution algorithms with NumPy (Lemire method,
  buffered uint32 splitting) (2026-03-15).
  [7512f6c](https://github.com/Dicklesworthstone/franken_numpy/commit/7512f6c33fec5df0274ab5340a68f9fe47a88b14)
- Reject `choice()` from empty array when `size > 0`.
  [292448b](https://github.com/Dicklesworthstone/franken_numpy/commit/292448b964a377720eff804b3110fca1f4980887)
- Add bounds check and `debug_assert` to `logfactorial` for negative k.
  [8b916f8](https://github.com/Dicklesworthstone/franken_numpy/commit/8b916f8673d9552875560dde1db7782ca885a2dd)

#### RNG Correctness Fixes

- Fix binomial BTPE algorithm edge cases and simplify ziggurat samplers
  (2026-03-21).
  [e341c26](https://github.com/Dicklesworthstone/franken_numpy/commit/e341c263725935ad46cb25cd4bd953ec281778e4)
- Revert ziggurat sampler to match original NumPy golden values; conformance
  golden values updated.
  [29ae5cd](https://github.com/Dicklesworthstone/franken_numpy/commit/29ae5cd117f6743a6a233dda21c954a40b2276c3)
- Simplify RNG state parsing with let-chains.
  [a765087](https://github.com/Dicklesworthstone/franken_numpy/commit/a765087df8b09bc1fa9cb7265926b14f79e706a2)

---

### Dtype System (`fnp-dtype`)

18 dtype variants covering the full NumPy type hierarchy with deterministic
promotion and 5 cast policies.

#### Type Support

- Add complex, string, and datetime dtype variants (2026-02-21).
  [3c7ee4a](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7ee4a63d144f8f3a0aa2341b05ecc9e12c0f6a)
- Add structured dtype support (2026-02-26).
  [e6a73d6](https://github.com/Dicklesworthstone/franken_numpy/commit/e6a73d6940f67a98789ba033cf8f540fcabc7316)
- Add `F16` (float16) dtype support via the `half` crate (2026-03-10).
  [fd9fbd1](https://github.com/Dicklesworthstone/franken_numpy/commit/fd9fbd1d0816916a0b591d3ede6e2a049e98a2b2)
- Add `IntegerSidecar` for lossless integer storage, complex arithmetic
  improvements, and transfer-loop selector (2026-03-18).
  [36b77e8](https://github.com/Dicklesworthstone/franken_numpy/commit/36b77e8edebd1898166d900ae83a5cd5730930be)

#### Promotion and Casting

- Expand dtype promotion and parity verification infrastructure (2026-02-15).
  [1f902e5](https://github.com/Dicklesworthstone/franken_numpy/commit/1f902e5b581f58deb05d2c8e2051fa6a5c95b17c)
- Expand safe cast rules for 64-bit integers and complex-to-string conversions
  (2026-03-20).
  [65e0832](https://github.com/Dicklesworthstone/franken_numpy/commit/65e0832419b8e81ac21f0d734d0e24baf8a5f55c)
- Prevent signed-to-unsigned `same_kind` cast and fix negative stride
  classification.
  [0c43819](https://github.com/Dicklesworthstone/franken_numpy/commit/0c43819a7d4b0b72db69d0c96f2264ef4429ffb5)
- Fix structured dtype handling, NaN risk sanitization, and lossy fallback logic
  (2026-03-12).
  [73846b9](https://github.com/Dicklesworthstone/franken_numpy/commit/73846b919870aa5cdd72e9ea72e1ff7f037685b3)
- Correct `finfo` precision and `remainder` semantics (2026-03-19).
  [52c4efb](https://github.com/Dicklesworthstone/franken_numpy/commit/52c4efbb6c327ae64a280b7f5d8b9ed55954b330)

---

### Stride Calculus Engine and N-D Array (`fnp-ndarray`)

The SCE is the correctness backbone of all shape transformations: broadcasting,
reshape, transpose, and view aliasing.

#### Stride-Tricks API

- Add `as_strided`, `broadcast_to`, `sliding_window_view` with full conformance
  coverage and reliability gate (2026-02-15).
  [d6241a9](https://github.com/Dicklesworthstone/franken_numpy/commit/d6241a9a45b21dc9f571917e35dfb3aaebbdf85e)
- Add shape/stride conformance cases and `coverage_ratio` improvements.
  [a67c25b](https://github.com/Dicklesworthstone/franken_numpy/commit/a67c25b2b91217792caceedd5ee458cd3559b070)

#### Negative Strides

- Add negative-stride support for reverse slicing (2026-02-21).
  [2f9c88d](https://github.com/Dicklesworthstone/franken_numpy/commit/2f9c88de8522f3080c9b41392667b1ee22bab1e3)

#### Layout Methods

- Add ndarray layout methods (2026-02-21).
  [3c7ee4a](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7ee4a63d144f8f3a0aa2341b05ecc9e12c0f6a)

#### Performance

- Replace per-element unravel/ravel axis reduction with contiguous kernel
  (2026-02-13).
  [d9cfe90](https://github.com/Dicklesworthstone/franken_numpy/commit/d9cfe90b001463d852d8b07284d9965851ab26bb)

---

### Iterator Subsystem (`fnp-iter`)

Transfer-loop selection, overlap detection, and NumPy-compatible iteration
primitives.

- Implement transfer selector, overlap detection, `flatiter`, and `nditer`
  subsystem with conformance tests (2026-02-15).
  [e162403](https://github.com/Dicklesworthstone/franken_numpy/commit/e1624033d2024dd505f693d86925fe3ad715fc46)

---

### I/O Subsystem (`fnp-io`)

Complete NPY/NPZ binary format support plus text I/O with security-hardened
parsing.

#### NPY Format

- Implement NPY binary format read/write with version (1.0/2.0) and pickle
  policy support (2026-02-19).
  [a9fbabb](https://github.com/Dicklesworthstone/franken_numpy/commit/a9fbabbdab5046cf6a679d4c796647d57db23cb7)
- Major overhaul of NPY header parsing with stricter validation (2026-03-19).
  [9a5b5b9](https://github.com/Dicklesworthstone/franken_numpy/commit/9a5b5b9cb253d9dd304170768949975396180ba6),
  [3289a6f](https://github.com/Dicklesworthstone/franken_numpy/commit/3289a6fcf79f0086f9d1ba4f58cb58176b4b6af4)
- Switch `NpyArrayBytes` payload to `Arc<[u8]>` for zero-copy sharing
  (2026-03-21).
  [3ed4487](https://github.com/Dicklesworthstone/franken_numpy/commit/3ed44873d78346240f7baa6baf763a51402f0829)
- Fix header validation (2026-03-21).
  [c89e90c](https://github.com/Dicklesworthstone/franken_numpy/commit/c89e90c2db4693d798c23abf635537c744539ff7)

#### NPZ Format

- Add NPZ archive support (2026-02-21).
  [ee46123](https://github.com/Dicklesworthstone/franken_numpy/commit/ee461234f69f0b48b9724930dfeeaa0602934fa8)
- Add DEFLATE compression support for NPZ read/write (2026-02-25).
  [8ce05d0](https://github.com/Dicklesworthstone/franken_numpy/commit/8ce05d07991f461e2d48e8664c63c1399213ba10)

#### Endianness and Complex I/O

- Add big-endian dtype support for all numeric types (2026-02-25).
  [7cd4159](https://github.com/Dicklesworthstone/franken_numpy/commit/7cd4159e6e628e9c6511f2e1a3ee6f2e9afbcba0)
- Add complex number dtype support for NumPy-compatible binary I/O (2026-03-10).
  [41cab7b](https://github.com/Dicklesworthstone/franken_numpy/commit/41cab7b3738a75ddfd884080a98f1bb23041756f)

#### Convenience Functions

- Add high-level I/O convenience functions (2026-02-26).
  [6f67d0a](https://github.com/Dicklesworthstone/franken_numpy/commit/6f67d0a464cc6eb14af0b86a3b5e3974f5be91f5)

#### Oracle Tests

- Add NumPy `.npy` format oracle tests for roundtrip, magic bytes, and header
  parsing (2026-03-16).
  [710c3b0](https://github.com/Dicklesworthstone/franken_numpy/commit/710c3b0229938b7c5f5379a180846dfa75e7d0a9)

---

### Dual-Mode Runtime (`fnp-runtime`)

Strict mode (max NumPy compatibility) and Hardened mode (safety guards) with
Bayesian decision engine.

- Add float-error state scaffolding (`seterr`, `geterr`, `errstate`)
  (2026-03-10).
  [8f065df](https://github.com/Dicklesworthstone/franken_numpy/commit/8f065dfa9874ca64985471206f72a0daaaddd6b7)
- Implement phase2c contract schema lock and Bayesian evidence ledger
  enrichment (2026-02-13).
  [264afa4](https://github.com/Dicklesworthstone/franken_numpy/commit/264afa4e5fcd82befd4fc501ea08e864b8fffd19)

---

### Conformance Infrastructure (`fnp-conformance`)

Four-layer quality system: differential harness, metamorphic testing,
adversarial fuzzing, and witness stability. Organized into an 8-gate CI
topology (G1-G8).

#### Differential Test Suites

- Add differential conformance harness, test contracts, and full-parity
  doctrine (2026-02-14).
  [3f2b977](https://github.com/Dicklesworthstone/franken_numpy/commit/3f2b977a3b7fc83cc5c4e8ca3c5086a3569af872)
- Add test contract system, adversarial fixtures, and essence extraction ledger.
  [278d62e](https://github.com/Dicklesworthstone/franken_numpy/commit/278d62e6fe3d37e3ff1a79eeee9c05bf654b50fe)
- Add FFT differential test suite with golden-value fixtures (2026-03-02).
  [50655be](https://github.com/Dicklesworthstone/franken_numpy/commit/50655be961a81f470e7ed945add67d6ddc01f95d)
- Add polynomial differential test suite.
  [db60f94](https://github.com/Dicklesworthstone/franken_numpy/commit/db60f94f904c57d4d1862c0a29117fb332930a6a)
- Add string differential test suite and expand polynomial fixtures (2026-03-03).
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
- Add datetime differential test cases and P2C003 transfer invariant tests
  (2026-03-18).
  [3c7c84c](https://github.com/Dicklesworthstone/franken_numpy/commit/3c7c84cfc4abf601385a2baba55db5b0046593a8),
  [96e61de](https://github.com/Dicklesworthstone/franken_numpy/commit/96e61de3c5efad49e5e76d51704873ee5451925f)

#### CI Gate Topology

- Add CI gate topology and contract infrastructure with 8-gate system (G1-G8:
  build, unit, differential, metamorphic, adversarial, witness, security,
  RaptorQ) (2026-02-17).
  [921b7a4](https://github.com/Dicklesworthstone/franken_numpy/commit/921b7a4f9ad90dc5d98e1223db6a83deaa561ac3)
- Harden CI gate topology contract for mandatory G7/G8 checks (2026-02-18).
  [893781c](https://github.com/Dicklesworthstone/franken_numpy/commit/893781c795d6bb1c431d8a0cfbfe42d85272644f)
- Require non-fallback oracle source for G3 gate in CI.
  [d42637b](https://github.com/Dicklesworthstone/franken_numpy/commit/d42637bc993c46f386600dccfd55b3652e563eea)
- Expand CI gate topology and refactor NPY header parsing (2026-03-19).
  [3289a6f](https://github.com/Dicklesworthstone/franken_numpy/commit/3289a6fcf79f0086f9d1ba4f58cb58176b4b6af4)

#### Conformance Corpus (P2C Packets)

- Expand conformance corpus across all 9 P2C packets with differential,
  metamorphic, and adversarial fixture suites for dtype (P2C-002), ufunc
  (P2C-005), RNG (P2C-007), linalg (P2C-008), and I/O (P2C-009) (2026-02-18).
  [837319b](https://github.com/Dicklesworthstone/franken_numpy/commit/837319bebbe71fbc60fb73663e130693d1760e5c),
  [96ab03b](https://github.com/Dicklesworthstone/franken_numpy/commit/96ab03bd3525728c69f8404c24272e0c139f189e),
  [720f108](https://github.com/Dicklesworthstone/franken_numpy/commit/720f1087ecdfed09a47debd05c333494e83b80a4),
  [faead61](https://github.com/Dicklesworthstone/franken_numpy/commit/faead6158532ca5fd73ed0cb9665410db9d567b6),
  [b5fe99f](https://github.com/Dicklesworthstone/franken_numpy/commit/b5fe99f13035761091761a658c388c64708a0bdb)
- Generate phase2c evidence packs for all 9 P2C packets with e2e replay
  scripts (2026-02-17).
  [7fc60b4](https://github.com/Dicklesworthstone/franken_numpy/commit/7fc60b4fe5a550b44fc618b9b7c7cf56d708308e),
  [f68f25e](https://github.com/Dicklesworthstone/franken_numpy/commit/f68f25e81cbc5e87cd467e161e0275aa069e26ac)
- Implement P2C-009 I/O contract validation for NPY/NPZ formats (2026-02-16).
  [c117674](https://github.com/Dicklesworthstone/franken_numpy/commit/c1176744a004236c9fc8388fe5019589ec286a38)
- Implement P2C-008 linalg contract validation (shape, solver, decomposition,
  policy checks).
  [7a1d947](https://github.com/Dicklesworthstone/franken_numpy/commit/7a1d947729edd1aece3c64d8ca7fd05e7645949d)
- Add linalg test contract framework with adversarial, differential, and
  metamorphic fixtures.
  [cb5ac2b](https://github.com/Dicklesworthstone/franken_numpy/commit/cb5ac2b128392e33691d9300211b0728f59b3676),
  [141792b](https://github.com/Dicklesworthstone/franken_numpy/commit/141792b3a0f177286ce3bfb1b2029384a6629e22)
- Add RNG differential, metamorphic, and adversarial fixture validation.
  [dc5c8c9](https://github.com/Dicklesworthstone/franken_numpy/commit/dc5c8c9ce5d52ee5765a012a301789bd633b3bfb)
- Add shape/stride differential testing and expand security/contract gates.
  [dd3a865](https://github.com/Dicklesworthstone/franken_numpy/commit/dd3a8652212267faa6e1c0c297aa32806b328c07)
- Fix double-write race in packet005 evidence pack.
  [358c377](https://github.com/Dicklesworthstone/franken_numpy/commit/358c377dcab18399f95461f8a7358b085710dbda)

#### RaptorQ Durability Gate

- Add RaptorQ artifact verification suite and gate binary (G8) (2026-02-15).
  [2582ac5](https://github.com/Dicklesworthstone/franken_numpy/commit/2582ac554f1c4d6b06aa46a97110a390846ceb4b)
- Add reliability retry/flake-budget support to RaptorQ gate.
  [1a9a9c3](https://github.com/Dicklesworthstone/franken_numpy/commit/1a9a9c326848a4c2190d298bfca74fd4ca5c11ed)

#### Workflow and Security Gates

- Add user workflow scenario corpus with golden journey gates (2026-02-14).
  [f3803b7](https://github.com/Dicklesworthstone/franken_numpy/commit/f3803b73bea003d6f28521ba82a9ef1df3c290ae)
- Expand workflow scenario gate and security gate with comprehensive
  orchestration (2026-02-15).
  [82dc7d1](https://github.com/Dicklesworthstone/franken_numpy/commit/82dc7d1233415472f633d54b9ae3d823f349b7a6),
  [6996c2f](https://github.com/Dicklesworthstone/franken_numpy/commit/6996c2f444eb0eb357f75284f7a54ec9c0828e73)
- Add security gate suite (2026-02-13).
  [952a123](https://github.com/Dicklesworthstone/franken_numpy/commit/952a1232fbe6b6546aed375277d2a1309e7c4552)

#### Benchmarking

- Add benchmarking telemetry and environment fingerprinting (2026-02-15).
  [2f4b16f](https://github.com/Dicklesworthstone/franken_numpy/commit/2f4b16f389987d6022a66d4e196cde54d9cb93f7)

#### Oracle Workflow Tests

- Add end-to-end workflow oracle tests for z-score, sort-cumsum, and boolean
  filter pipelines (2026-03-17).
  [a2da789](https://github.com/Dicklesworthstone/franken_numpy/commit/a2da789d5470c361fd6b67ddb2c9b40b36377e0f)

---

### Cross-Crate and Correctness

Changes that span multiple crates or address broad correctness improvements.

- Comprehensive edge-case and correctness fixes across dtype, ndarray, random,
  I/O, and ufunc for NumPy behavioral parity (2026-02-20).
  [bba14a3](https://github.com/Dicklesworthstone/franken_numpy/commit/bba14a3cbf6cab1a2b985f93b2ccd81483e4789c)
- Expand core crates with dtype casting, linalg ops, random distributions, IO
  improvements, and ufunc coverage.
  [44941db](https://github.com/Dicklesworthstone/franken_numpy/commit/44941db0b9cf0cae35d9b08e62f4ef7c919bd65d)
- Optimize hot paths across fnp-io, fnp-iter, fnp-linalg, fnp-ndarray, and
  fnp-ufunc (2026-02-17).
  [b605554](https://github.com/Dicklesworthstone/franken_numpy/commit/b605554023cefc9336fdf739d2bd3f5c3f65bc9c)

---

### Licensing

- Adopt MIT license with OpenAI/Anthropic rider across all workspace crates
  (2026-02-18).
  [adb03a5](https://github.com/Dicklesworthstone/franken_numpy/commit/adb03a54741afe4a6891301d25b9111b6fa83840),
  [68b171f](https://github.com/Dicklesworthstone/franken_numpy/commit/68b171fdee6b9c7addee37553f9febbba92c2103)

---

### Dependencies

- Upgrade workspace dependencies and pin nightly toolchain to
  `nightly-2026-02-20` (2026-02-20).
  [b0175c6](https://github.com/Dicklesworthstone/franken_numpy/commit/b0175c60b6c4d722a4297faeeb41865c62351535)
- Bump `asupersync` from v0.1.1 to v0.2.0 in conformance and runtime crates
  (2026-02-15).
  [4931f3c](https://github.com/Dicklesworthstone/franken_numpy/commit/4931f3c9ba3bab0ab85f7189fcdc09c978dfc135)
- Bump `ftui` from 0.1.1 to 0.2.0.
  [486992f](https://github.com/Dicklesworthstone/franken_numpy/commit/486992f455b5e1b6ac11a58b56ccfc4363464243)

---

### Project Bootstrap (2026-02-13)

Initialize FrankenNumPy as a 9-crate Rust workspace with clean-room NumPy port,
conformance infrastructure, and durability artifacts.
[527bd9d](https://github.com/Dicklesworthstone/franken_numpy/commit/527bd9d11123e149377bb6edd5e9bc1a5ab3b70f)

**Core architecture established:**
- `fnp-dtype` -- 18 dtype variants with NumPy-exact promotion table and 5 cast
  policies.
- `fnp-ndarray` -- Stride Calculus Engine (SCE) with shape-to-strides, broadcast
  legality, reshape with `-1` inference, and view safety.
- `fnp-iter` -- Transfer-loop selector state machine (stub).
- `fnp-ufunc` -- Universal function dispatch framework (stub).
- `fnp-linalg` -- Linear algebra decompositions (stub).
- `fnp-random` -- PRNG subsystem (stub).
- `fnp-io` -- NPY/NPZ binary I/O (stub).
- `fnp-runtime` -- Dual-mode runtime (strict/hardened) with Bayesian decision
  engine.
- `fnp-conformance` -- Differential conformance harness with test contracts.

---
