//! Quarantined CBLAS/LAPACK capability boundary (`cblas-lapack` feature, ADR-001 Stream 2).
//!
//! Governance invariants (`deadlock-audit-043z0`):
//! 1. **Off by default:** `crates/fnp-linalg/Cargo.toml` sets `default = []`, so the default
//!    10-crate numeric core links zero external C symbols and remains 100% `#![forbid(unsafe_code)]`.
//! 2. **Never a benchmark arm:** Per `/data/projects/AGENTS.md` ("Dependency smuggling"), this
//!    feature is an opt-in capability surface only and must never be enabled in benchmark arms.
//! 3. **Safe-Rust reference execution:** All capability entrypoints validate CBLAS/LAPACK buffer,
//!    leading-dimension (`lda`/`ldb`/`ldc`), and shape contracts before routing to `fnp-linalg`'s
//!    verified safe-Rust kernels.

#[cfg(feature = "cblas-lapack")]
use crate::{
    LinAlgError, SvdFullResult, cholesky_nxn, eigh_nxn, mat_mul_rect, qr_mxn, solve_nxn,
    solve_nxn_multi, svd_mxn_full,
};

/// Returns `true` when `fnp-linalg` was compiled with `--features cblas-lapack`.
#[must_use]
pub const fn is_cblas_lapack_feature_enabled() -> bool {
    cfg!(feature = "cblas-lapack")
}

/// Returns the list of external C BLAS/LAPACK symbols linked by `fnp-linalg`.
///
/// Even when the `cblas-lapack` capability feature is enabled, `fnp-linalg` remains
/// `#![forbid(unsafe_code)]` and links zero external C symbols unless an external C
/// provider is explicitly linked out-of-tree.
#[must_use]
pub const fn linked_external_c_blas_lapack_symbols() -> &'static [&'static str] {
    &[]
}

/// Canonical external C BLAS/LAPACK shared-library prefixes forbidden in the default build.
pub const FORBIDDEN_EXTERNAL_BLAS_LAPACK_LIBS: [&str; 6] = [
    "libopenblas",
    "libblas",
    "liblapack",
    "libcblas",
    "libmkl",
    "libblis",
];

/// Descriptor for an opt-in CBLAS/LAPACK capability entrypoint.
#[cfg(feature = "cblas-lapack")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CblasLapackCapabilityDescriptor {
    pub symbol: &'static str,
    pub family: &'static str,
    pub backing_kernel: &'static str,
}

/// Returns the capability table exposed when `--features cblas-lapack` is enabled.
#[cfg(feature = "cblas-lapack")]
#[must_use]
pub const fn capability_snapshot() -> [CblasLapackCapabilityDescriptor; 6] {
    [
        CblasLapackCapabilityDescriptor {
            symbol: "cblas_dgemm",
            family: "blas_level3",
            backing_kernel: "fnp_linalg::mat_mul_rect",
        },
        CblasLapackCapabilityDescriptor {
            symbol: "lapack_dgesv",
            family: "lapack_solve",
            backing_kernel: "fnp_linalg::solve_nxn / solve_nxn_multi",
        },
        CblasLapackCapabilityDescriptor {
            symbol: "lapack_dpotrf",
            family: "lapack_cholesky",
            backing_kernel: "fnp_linalg::cholesky_nxn",
        },
        CblasLapackCapabilityDescriptor {
            symbol: "lapack_dgeqrf",
            family: "lapack_qr",
            backing_kernel: "fnp_linalg::qr_mxn",
        },
        CblasLapackCapabilityDescriptor {
            symbol: "lapack_dgesdd",
            family: "lapack_svd",
            backing_kernel: "fnp_linalg::svd_mxn_full",
        },
        CblasLapackCapabilityDescriptor {
            symbol: "lapack_dsyev",
            family: "lapack_eigh",
            backing_kernel: "fnp_linalg::eigh_nxn",
        },
    ]
}

/// CBLAS matrix memory layout selector (`CblasRowMajor` vs `CblasColMajor`).
#[cfg(feature = "cblas-lapack")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasOrder {
    RowMajor,
    ColMajor,
}

/// CBLAS matrix transpose selector (`CblasNoTrans`, `CblasTrans`, `CblasConjTrans`).
#[cfg(feature = "cblas-lapack")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasTranspose {
    NoTrans,
    Trans,
    ConjTrans,
}

/// LAPACK triangular storage selector (`'U'` vs `'L'`).
#[cfg(feature = "cblas-lapack")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LapackUpLo {
    Upper,
    Lower,
}

#[cfg(feature = "cblas-lapack")]
fn extract_packed_row_major(
    order: CblasOrder,
    trans: CblasTranspose,
    op_rows: usize,
    op_cols: usize,
    buf: &[f64],
    ld: usize,
    label: &'static str,
) -> Result<Vec<f64>, LinAlgError> {
    let is_trans = matches!(trans, CblasTranspose::Trans | CblasTranspose::ConjTrans);
    let (phys_rows, phys_cols) = if is_trans {
        (op_cols, op_rows)
    } else {
        (op_rows, op_cols)
    };
    let min_ld = match order {
        CblasOrder::RowMajor => phys_cols.max(1),
        CblasOrder::ColMajor => phys_rows.max(1),
    };
    if ld < min_ld {
        return Err(LinAlgError::BackendBridgeInvalid(label));
    }
    if op_rows == 0 || op_cols == 0 {
        return Ok(Vec::new());
    }
    let required_len = match order {
        CblasOrder::RowMajor => (phys_rows - 1)
            .checked_mul(ld)
            .and_then(|v| v.checked_add(phys_cols))
            .ok_or(LinAlgError::ShapeContractViolation(
                "cblas_dgemm: stride overflow",
            ))?,
        CblasOrder::ColMajor => (phys_cols - 1)
            .checked_mul(ld)
            .and_then(|v| v.checked_add(phys_rows))
            .ok_or(LinAlgError::ShapeContractViolation(
                "cblas_dgemm: stride overflow",
            ))?,
    };
    if buf.len() < required_len {
        return Err(LinAlgError::ShapeContractViolation(
            "cblas_dgemm: input buffer shorter than leading-dimension extent",
        ));
    }
    let mut out = vec![0.0; op_rows * op_cols];
    for r in 0..op_rows {
        for c in 0..op_cols {
            let (pr, pc) = if is_trans { (c, r) } else { (r, c) };
            let idx = match order {
                CblasOrder::RowMajor => pr * ld + pc,
                CblasOrder::ColMajor => pc * ld + pr,
            };
            out[r * op_cols + c] = buf[idx];
        }
    }
    Ok(out)
}

/// CBLAS-compatible `dgemm`: computes `C := alpha * op(A) * op(B) + beta * C`.
///
/// Validates leading dimensions (`lda`, `ldb`, `ldc`) and slice bounds before
/// executing the product through `fnp-linalg`'s safe-Rust `mat_mul_rect` kernel.
#[cfg(feature = "cblas-lapack")]
#[allow(clippy::too_many_arguments)]
pub fn cblas_dgemm(
    order: CblasOrder,
    trans_a: CblasTranspose,
    trans_b: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) -> Result<(), LinAlgError> {
    let min_ldc = match order {
        CblasOrder::RowMajor => n.max(1),
        CblasOrder::ColMajor => m.max(1),
    };
    if ldc < min_ldc {
        return Err(LinAlgError::BackendBridgeInvalid(
            "cblas_dgemm: invalid ldc for matrix dimensions",
        ));
    }
    let a_packed = extract_packed_row_major(
        order,
        trans_a,
        m,
        k,
        a,
        lda,
        "cblas_dgemm: invalid lda for matrix dimensions",
    )?;
    let b_packed = extract_packed_row_major(
        order,
        trans_b,
        k,
        n,
        b,
        ldb,
        "cblas_dgemm: invalid ldb for matrix dimensions",
    )?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    let required_c = match order {
        CblasOrder::RowMajor => (m - 1)
            .checked_mul(ldc)
            .and_then(|v| v.checked_add(n))
            .ok_or(LinAlgError::ShapeContractViolation(
                "cblas_dgemm: output stride overflow",
            ))?,
        CblasOrder::ColMajor => (n - 1)
            .checked_mul(ldc)
            .and_then(|v| v.checked_add(m))
            .ok_or(LinAlgError::ShapeContractViolation(
                "cblas_dgemm: output stride overflow",
            ))?,
    };
    if c.len() < required_c {
        return Err(LinAlgError::ShapeContractViolation(
            "cblas_dgemm: output buffer shorter than leading-dimension extent",
        ));
    }
    let ab = if k == 0 || alpha == 0.0 {
        vec![0.0; m * n]
    } else {
        mat_mul_rect(&a_packed, &b_packed, m, k, n)
    };
    for r in 0..m {
        for col in 0..n {
            let dst_idx = match order {
                CblasOrder::RowMajor => r * ldc + col,
                CblasOrder::ColMajor => col * ldc + r,
            };
            let prev = if beta == 0.0 { 0.0 } else { beta * c[dst_idx] };
            c[dst_idx] = alpha * ab[r * n + col] + prev;
        }
    }
    Ok(())
}

/// LAPACK-compatible `dgesv`: solves `A * X = B` for an `n x n` matrix `A` and `n x nrhs` right-hand side `B`.
#[cfg(feature = "cblas-lapack")]
pub fn lapack_dgesv(n: usize, nrhs: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, LinAlgError> {
    if a.len() != n.saturating_mul(n) || b.len() != n.saturating_mul(nrhs) {
        return Err(LinAlgError::ShapeContractViolation(
            "lapack_dgesv: matrix A must be n*n and B must be n*nrhs",
        ));
    }
    if nrhs == 1 {
        solve_nxn(a, b, n)
    } else {
        solve_nxn_multi(a, b, n, nrhs)
    }
}

/// LAPACK-compatible `dpotrf`: computes the Cholesky factorization of an `n x n` SPD matrix `A`.
///
/// Returns the lower triangular factor `L` (`A = L * L^T`) when `uplo == LapackUpLo::Lower`,
/// or the upper triangular factor `U = L^T` (`A = U^T * U`) when `uplo == LapackUpLo::Upper`.
#[cfg(feature = "cblas-lapack")]
pub fn lapack_dpotrf(uplo: LapackUpLo, n: usize, a: &[f64]) -> Result<Vec<f64>, LinAlgError> {
    let lower = cholesky_nxn(a, n)?;
    match uplo {
        LapackUpLo::Lower => Ok(lower),
        LapackUpLo::Upper => {
            let mut upper = vec![0.0; n * n];
            for r in 0..n {
                for c in r..n {
                    upper[r * n + c] = lower[c * n + r];
                }
            }
            Ok(upper)
        }
    }
}

/// LAPACK-compatible `dgeqrf` + `dorgqr`: computes the reduced QR factorization `(Q, R)` of an `m x n` matrix `A`.
#[cfg(feature = "cblas-lapack")]
pub fn lapack_dgeqrf(m: usize, n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    qr_mxn(a, m, n)
}

/// LAPACK-compatible `dgesdd`: computes the full singular value decomposition `(U, S, Vt)` of an `m x n` matrix `A`.
#[cfg(feature = "cblas-lapack")]
pub fn lapack_dgesdd(m: usize, n: usize, a: &[f64]) -> Result<SvdFullResult, LinAlgError> {
    svd_mxn_full(a, m, n)
}

/// LAPACK-compatible `dsyev`: computes ascending eigenvalues and column eigenvectors `(w, V)` of a symmetric `n x n` matrix `A`.
#[cfg(feature = "cblas-lapack")]
pub fn lapack_dsyev(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    eigh_nxn(a, n)
}

#[cfg(test)]
mod tests {
    use super::{
        FORBIDDEN_EXTERNAL_BLAS_LAPACK_LIBS, is_cblas_lapack_feature_enabled,
        linked_external_c_blas_lapack_symbols,
    };

    #[test]
    fn external_c_blas_lapack_symbol_table_is_empty() {
        assert!(
            linked_external_c_blas_lapack_symbols().is_empty(),
            "fnp-linalg must not link external C BLAS/LAPACK symbols in safe-Rust mode"
        );
        #[cfg(not(feature = "cblas-lapack"))]
        assert!(
            !is_cblas_lapack_feature_enabled(),
            "cblas-lapack feature must be off by default"
        );
        #[cfg(feature = "cblas-lapack")]
        assert!(
            is_cblas_lapack_feature_enabled(),
            "cblas-lapack feature flag should report enabled when compiled with --features cblas-lapack"
        );

        // Inspect our own process /proc/self/maps on Linux to prove zero external C BLAS/LAPACK
        // shared libraries (libopenblas, libblas, liblapack, libcblas, libmkl, libblis) are loaded.
        if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
            let lower = maps.to_ascii_lowercase();
            for forbidden in FORBIDDEN_EXTERNAL_BLAS_LAPACK_LIBS {
                assert!(
                    !lower.contains(forbidden),
                    "process loaded forbidden external C BLAS/LAPACK library {forbidden}"
                );
            }
        }
    }

    #[cfg(feature = "cblas-lapack")]
    #[test]
    fn cblas_lapack_capability_parity_and_negative_controls() {
        use super::{
            CblasOrder, CblasTranspose, LapackUpLo, capability_snapshot, cblas_dgemm,
            lapack_dgeqrf, lapack_dgesdd, lapack_dgesv, lapack_dpotrf, lapack_dsyev,
        };
        use crate::{
            LinAlgError, cholesky_nxn, eigh_nxn, mat_mul_rect, qr_mxn, solve_nxn, solve_nxn_multi,
            svd_mxn_full,
        };

        let snap = capability_snapshot();
        assert_eq!(snap.len(), 6);
        assert_eq!(snap[0].symbol, "cblas_dgemm");
        assert_eq!(snap[5].symbol, "lapack_dsyev");

        // 1. cblas_dgemm parity (RowMajor, ColMajor, Transpose, alpha/beta, and padded ldc)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2
        let expected_ab = mat_mul_rect(&a, &b, 2, 3, 2);
        let mut c = vec![10.0, 20.0, 30.0, 40.0];
        cblas_dgemm(
            CblasOrder::RowMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            3,
            2.0,
            &a,
            3,
            &b,
            2,
            0.5,
            &mut c,
            2,
        )
        .unwrap();
        for i in 0..4 {
            let want = 2.0 * expected_ab[i] + 0.5 * ((i + 1) as f64 * 10.0);
            assert!((c[i] - want).abs() <= 1e-12);
        }

        // Negative control: invalid lda < k in RowMajor NoTrans must fail closed.
        let err = cblas_dgemm(
            CblasOrder::RowMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            2,
            2,
            3,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        )
        .unwrap_err();
        assert_eq!(err.reason_code(), "linalg_backend_bridge_invalid");

        // 2. lapack_dgesv parity (single-RHS and multi-RHS + singular negative control)
        let a_sys = vec![3.0, 1.0, 1.0, 2.0];
        let b_1 = vec![9.0, 8.0];
        assert_eq!(
            lapack_dgesv(2, 1, &a_sys, &b_1).unwrap(),
            solve_nxn(&a_sys, &b_1, 2).unwrap()
        );
        let b_2 = vec![9.0, 5.0, 8.0, 5.0];
        assert_eq!(
            lapack_dgesv(2, 2, &a_sys, &b_2).unwrap(),
            solve_nxn_multi(&a_sys, &b_2, 2, 2).unwrap()
        );
        assert_eq!(
            lapack_dgesv(2, 1, &[1.0, 2.0, 2.0, 4.0], &b_1).unwrap_err(),
            LinAlgError::SolverSingularity
        );

        // 3. lapack_dpotrf parity (Lower and Upper + non-SPD negative control)
        let spd = vec![4.0, 2.0, 2.0, 5.0];
        let l = lapack_dpotrf(LapackUpLo::Lower, 2, &spd).unwrap();
        let u = lapack_dpotrf(LapackUpLo::Upper, 2, &spd).unwrap();
        assert_eq!(l, cholesky_nxn(&spd, 2).unwrap());
        assert!((l[2] - u[1]).abs() <= 1e-12);
        assert!(lapack_dpotrf(LapackUpLo::Lower, 2, &[1.0, 2.0, 2.0, 1.0]).is_err());

        // 4. lapack_dgeqrf, lapack_dgesdd, lapack_dsyev parity against native kernels
        let rect = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            lapack_dgeqrf(3, 2, &rect).unwrap(),
            qr_mxn(&rect, 3, 2).unwrap()
        );
        assert_eq!(
            lapack_dgesdd(3, 2, &rect).unwrap(),
            svd_mxn_full(&rect, 3, 2).unwrap()
        );
        assert_eq!(lapack_dsyev(2, &spd).unwrap(), eigh_nxn(&spd, 2).unwrap());
    }
}
