# Reality-Check Audit: README Claims vs Implementation

**Date:** 2026-05-03  
**Auditor:** cc_1

## Summary

Audited README.md, AGENTS.md, and FEATURE_PARITY.md against actual codebase state. Found 4 documentation gaps requiring correction. Core implementation claims are accurate.

---

## Gaps Found

### GAP-001: Crate Count Mismatch

**Claim (README lines 32, 96, 344, 920, 1012):** "9 crates"

**Reality:** 10 crates exist:
1. fnp-conformance
2. fnp-dtype
3. fnp-io
4. fnp-iter
5. fnp-linalg
6. fnp-ndarray
7. fnp-python (added after original documentation)
8. fnp-random
9. fnp-runtime
10. fnp-ufunc

**Severity:** Low (documentation lag)

---

### GAP-002: Architecture Diagram Shows Non-Existent fnp-fft Crate

**Claim (README line 151):** Diagram shows `│ fnp-fft  │` as separate crate

**Reality:** No fnp-fft crate exists. FFT functions are implemented in fnp-ufunc:
- `fft()`, `ifft()`, `rfft()`, `fft2()`, `ifft2()`, `fftn()`, `ifftn()`, `rfftn()`, `rfft2()`, `fftfreq()`

**Severity:** Medium (misleading architecture representation)

---

### GAP-003: RaptorQ Gate Script Path Incorrect

**Claim (README line 511):** References `run_raptorq_gate.sh`

**Reality:** Actual path is `scripts/e2e/run_raptorq_gate.sh`

**Severity:** Low (minor path discrepancy)

---

### GAP-004: Line Count Significantly Understated *(refreshed 2026-05-13)*

**Original claim audited (README lines 344, 1012):** "92,000+ lines of Rust"

**Reality at audit (2026-05-03):** 254,570 lines across all .rs files

**Refresh (2026-05-13):** README now reads "300,000+ lines of Rust" (commit
9a04782); live count is 304,276 .rs lines across the workspace. Claim is
now accurate (slightly understated, which is the desired direction).

**Severity:** Resolved — README is now accurate.

---

## Verified Claims (Accurate)

| Claim | Verification |
|-------|--------------|
| `#![forbid(unsafe_code)]` on all crates | Confirmed: 10/10 crates have forbid(unsafe_code) |
| "Over 1,000 public functions" | Confirmed: 1,481 pub fn at audit (2026-05-03); 1,575 pub fn at refresh (2026-05-13). README updated to "1,500+ public functions" in commit 9a04782. |
| PCG64DXSM RNG implementation | Confirmed: `Pcg64DxsmRng` struct in fnp-random |
| Stride Calculus Engine | Confirmed: `broadcast_shape`, `broadcast_shapes` in fnp-ndarray |
| Dual-mode runtime (strict/hardened) | Confirmed: enum variants in fnp-runtime |
| Evidence ledger | Confirmed: `EvidenceLedger`, `DecisionEvent` in fnp-runtime |
| einsum implementation | Confirmed: `einsum()`, `einsum_path()`, `einsum_optimized()` in fnp-ufunc |
| gufunc support | Confirmed: `GufuncSignature` struct in fnp-ufunc |
| MaskedArray support | Confirmed: `MaskedArray` struct with full method suite in fnp-ufunc |
| datetime64/timedelta64 | Confirmed: DType variants in fnp-dtype |
| linalg decompositions | Confirmed: 106 pub fn in fnp-linalg (svd, eig, qr, cholesky, etc.) |
| CI workflow | Confirmed: .github/workflows/ci.yml exists |

---

## Test Count Verification

| Source | Claimed (2026-05-03) | Actual (2026-05-03) | Actual (2026-05-13 refresh) |
|--------|---------------------:|--------------------:|----------------------------:|
| FEATURE_PARITY.md fnp-ufunc | 1,797 | — | — |
| FEATURE_PARITY.md fnp-linalg | 235 | — | — |
| FEATURE_PARITY.md fnp-io | 176 | — | — |
| Workspace total `#[test]` runs | — | 5,047 | **6,392** (+27%) |
| Files with tests | — | 134 | unchanged scan basis |

The +1,344 test growth between the two audit dates tracks the parity wave
(see `audit_numpy_reality.md` for the 43% → 100% `numpy.__all__` close-out)
and the `33vtd` diagnostic-parity wave.

---

## Beads Filed

- `[reality-check] README: update crate count from 9 to 10`
- `[reality-check] README: remove fnp-fft from architecture diagram or note FFT is in fnp-ufunc`
- `[reality-check] README: fix run_raptorq_gate.sh path`
