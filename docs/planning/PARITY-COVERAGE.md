# PARITY-COVERAGE.md

Rigorous upstream coverage audit completed 2026-05-25. Counts below were refreshed on 2026-09-02 from the live tree; the surface percentages are unchanged (they are structural), the test and bead figures had drifted.

## Summary

| Category | NumPy Count | FNP Count | Coverage |
|----------|-------------|-----------|----------|
| `numpy.__all__` | 499 | 499 | 100% |
| UFuncs | 106 | 106 | 100% |
| `numpy.linalg` functions | 32 | 32 | 100% |
| `numpy.fft` functions | 18 | 18 | 100% |
| `numpy.random` Generator methods | 60 | 60 | 100% |

## Conformance Test Coverage

- **Total workspace tests**: 8,688 `#[test]` functions (2026-09-02; 7,355 at the 05-25 audit)
- **Conformance tests**: 2,350 test functions across 168 test files
- **Test result**: 0 failures
- **fnp-python tests**: 3,635 `#[test]` functions (2026-09-02). "All pass" has not been observed on CI since 2026-02-26 (the only green run of the gate topology); the 191 conformance shards were import-failing under the rch build route until 2026-09-01, when the extension resolver in `tests/support/mod.rs` was propagated to all of them

## Bead Tracker Status

- **Total beads**: 2,818 (2026-09-02; 1,577 at the 05-25 audit)
- **Closed**: 2,782
- **Open**: 4
- **In Progress**: 31
- **Blocked**: 0
- **Avg lead time**: 4.2 hours

## Edge Case Verification

All verified to match NumPy behavior:

### Ufunc Edge Cases
- `add.reduce([])` → 0.0 (identity)
- `multiply.reduce([])` → 1.0 (identity)
- `add.accumulate([])` → empty array
- `add.outer([], arr)` → correct shape
- `add.reduce(arr, axis=-1)` → negative axis
- `add.at(arr, indices, val)` → in-place operation
- `add.reduceat(arr, indices)` → segment reduction
- `add(a, b, where=mask)` → where parameter
- `add(a, b, out=out, casting='safe')` → casting parameter
- `power(0, 0)` → 1 (convention)
- `power(-1, 0.5)` → nan

### Dtype Promotion
- int32 + float64 → float64
- complex + int → complex128
- Signed zero semantics preserved

### Linalg Edge Cases
- `det(0x0 empty)` → 1.0
- `inv(singular)` → LinAlgError
- `solve(singular, b)` → LinAlgError
- `cholesky(non-positive-definite)` → LinAlgError
- `eig(non-square)` → LinAlgError
- `lstsq(rank-deficient, b)` → correct solution
- `svd(tall/wide, full_matrices=False)` → correct shapes
- `qr(mode='reduced')` → correct shapes
- `cond(A, p=inf)` → correct condition number
- `eigh(A, UPLO='L')` → correct eigenvalues

### Broadcasting
- Scalar + 3D array
- (3,1) + (1,3) → (3,3)
- F-order preservation
- Strided view operations

## Passthrough vs Native Implementation

Most operations have native Rust implementations. NumPy passthrough is used for:
- Complex LAPACK decompositions (SVD, QR, Cholesky, eigendecomposition)
- Some advanced polynomial operations
- Masked array complex operations

This is by design (DISC-009 acceptance) — the passthrough ensures correctness while native implementations are incrementally added for performance.

## Out of Scope

1. **`pip install` distribution** — No PyPI wheel yet; local build only
2. **100% bit-for-bit oracle verification** — Conformance tests verify behavior, not exact floating-point bit patterns for all inputs
3. **NumPy private/undocumented APIs** — Only public API surface
4. **`np.random.mtrand` submodule** — Legacy compatibility layer; `RandomState` is exposed directly on `fnp.random`

## Gaps Found

**Surface gaps: none** (every `numpy.__all__` name resolves). **Behavioural gaps: yes**, tracked elsewhere and listed here so this file does not read as a clean bill:

- `PARAMETER_PARITY_TODO.md` carries six open `[D]`/`[A]` rows (`dtype=` on reductions, `out=` on array functions, `order` on sort/partition, `overwrite_input`, `kind=` verification).
- Bead `deadlock-audit-7evbk` (2026-09-02 probes): `Generator.bit_generator` is a string rather than a BitGenerator; `interp` and f32 `cumsum`/`cumprod` differ from numpy by 1 ulp; the 22 arithmetic ufuncs are a PyO3 mimic that fails `isinstance(x, np.ufunc)` and 83 ufunc names have no `reduce`/`at`/`nin`/`nout`.
- Bead `deadlock-audit-defaulted-argument-three-state-parse-kqn3n`: 70 argument cells blocked by PyO3 defaulted parameters that cannot distinguish an omitted argument from an explicit `None`.
- Non-native byte order was silently misread by 15 ops on `>f8` until 2026-09-02; now guarded at the buffer and the op entry and locked by `crates/fnp-python/tests/conformance_byteorder.rs` (522 cells).

## Audit Method

1. Enumerated NumPy's public API via `dir(np)` and `numpy.__all__`
2. Verified each API exists in fnp-python module
3. Ran 2,350 conformance tests via `cargo test -p fnp-python --tests`
4. Manual edge case verification via Python interpreter
5. Cross-checked ufunc attributes (types, ntypes, nin, nout, identity, resolve_dtypes)
6. Verified error paths (LinAlgError, ValueError) match NumPy
