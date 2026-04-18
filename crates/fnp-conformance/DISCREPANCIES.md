# Known Conformance Divergences

> Intentional deviations from NumPy behavior, documented per testing-conformance-harnesses skill.

## Status Legend

- **ACCEPTED** - Deviation is intentional and will not be changed
- **INVESTIGATING** - Under review, may be fixed
- **WILL-FIX** - Known issue to be addressed

---

## DISC-001: empty() / empty_like() zero-initialize

- **Reference:** NumPy returns uninitialized memory (may contain garbage)
- **Our impl:** Returns zero-filled array (delegates to `zeros()`)
- **Impact:** FrankenNumPy arrays are predictably zero; NumPy arrays have undefined content
- **Resolution:** ACCEPTED
- **Reason:** `#![forbid(unsafe_code)]` precludes uninitialized memory in safe Rust. Zero-fill is the only safe option and matches NumPy's observed behavior for freshly allocated arrays in practice.
- **Tests affected:** None - behavior is compatible (zero is valid "uninitialized" content)
- **Review date:** 2026-03-15

---

## DISC-002: Unicode width tables (Unicode 15.1 vs 13.0)

- **Reference:** NumPy uses system-dependent Unicode width tables
- **Our impl:** Uses `unicode-width` crate (Unicode 15.1)
- **Impact:** Some CJK character widths may differ in string width calculations
- **Resolution:** ACCEPTED
- **Reason:** Unicode 15.1 tables are more correct and up-to-date
- **Tests affected:** string_differential_cases.json CJK width tests (if any)
- **Review date:** 2026-03-15

---

## DISC-003: Error message format differences

- **Reference:** NumPy error messages have specific wording
- **Our impl:** Rust error messages use different phrasing
- **Impact:** Error string content differs (error semantics identical)
- **Resolution:** ACCEPTED
- **Reason:** We test error categories and types, not exact message strings
- **Tests affected:** All adversarial test suites check error class, not message text
- **Review date:** 2026-03-15

---

## DISC-004: multivariate_normal uses Cholesky (not SVD)

- **Reference:** NumPy defaults to SVD decomposition for covariance
- **Our impl:** Uses Cholesky decomposition
- **Impact:** May fail on some ill-conditioned covariance matrices that SVD handles
- **Resolution:** ACCEPTED
- **Reason:** Adding SVD would require `fnp-linalg` as a dependency of `fnp-random` (currently zero-dependency). Cholesky is faster and sufficient for well-conditioned cases.
- **Tests affected:** rng_differential_cases.json multivariate_normal tests use well-conditioned matrices
- **Review date:** 2026-03-15

---

## DISC-005: multivariate_hypergeometric algorithm

- **Reference:** NumPy uses `random_mvhg_marginals` algorithm
- **Our impl:** Uses sequential draws
- **Impact:** Different internal RNG consumption pattern (outputs still correct)
- **Resolution:** ACCEPTED
- **Reason:** Sequential draws are simpler and statistically correct
- **Tests affected:** rng_statistical_cases.json verifies distribution properties, not RNG sequence
- **Review date:** 2026-03-15

---

## DISC-006: Complex storage format

- **Reference:** NumPy uses native complex128 type
- **Our impl:** Uses interleaved (f64, f64) pairs with trailing dimension of 2
- **Impact:** Storage layout differs; arithmetic results are identical
- **Resolution:** ACCEPTED
- **Reason:** Safe Rust representation without unsafe transmutes
- **Tests affected:** Complex arithmetic tests verify values, not memory layout
- **Review date:** 2026-03-15

---

## DISC-007: f64 internal representation for arithmetic

- **Reference:** NumPy preserves exact integer values for i64/u64
- **Our impl:** `UFuncArray` uses `Vec<f64>` internally; `IntegerSidecar` preserves i64/u64 > 2^53 through storage round-trips
- **Impact:** Arithmetic on large integers (> 2^53) uses f64 approximation
- **Resolution:** ACCEPTED
- **Reason:** Unified f64 storage simplifies broadcasting and type promotion. Sidecar preserves integrity for storage, not arithmetic.
- **Tests affected:** Large integer arithmetic tests document this behavior
- **Review date:** 2026-04-01

---

## Adding New Divergences

When documenting a new divergence:

1. Assign sequential ID: DISC-NNN
2. Document both behaviors (reference vs ours)
3. State resolution: ACCEPTED, INVESTIGATING, or WILL-FIX
4. Explain WHY the divergence exists
5. List affected test cases
6. Include review date
7. For ACCEPTED divergences, use XFAIL in tests (not SKIP)

## References

- [COVERAGE.md](COVERAGE.md) - Coverage matrix and gaps
- [FEATURE_PARITY.md](../../FEATURE_PARITY.md) - Intentional design decisions section
