//! linalg-domain criterion benches. The TSQR least-squares A/B harness:
//! `fnp.lstsq` vs `numpy.linalg.lstsq` (LAPACK gelsd) on full-rank tall-skinny
//! float64 systems, both arms live in the same invocation.
//!
//! HISTORY: the copy-based native TSQR wiring was REJECTED on 2026-07-24 for
//! `franken_numpy-ixs5y.i546h` (1.02x / 1.006x / 0.87x on a clean fleet — the
//! kernel's flop advantage was swamped by per-call O(mn) Python-surface costs).
//! That row's retry predicate named BOTH remedies, and this measures the build
//! where both have landed: `try_native_lstsq_tsqr` extracts A and b through a
//! zero-copy `PyBuffer` instead of `extract_numeric_array`'s asarray+cast copy,
//! and `tsqr_qtb` accumulates ‖(Qᵀb)_dropped‖² up the reduction tree instead of
//! recomputing ‖b−Ax‖² in a second pass over A.
//!
//! Split as its own per-domain bench binary so a re-measure compiles without the
//! whole monolith (bead deadlock-audit-x7nnf).

#[path = "common/mod.rs"]
mod common;

use common::ensure_numpy_available;
use criterion::Criterion;
use fnp_python::fnp_python;
use pyo3::Python;
use pyo3::types::{PyAnyMethods, PyModule, PyTuple};
use std::hint::black_box;

/// `fnp.lstsq` against live `numpy.linalg.lstsq` on full-rank tall-skinny f64.
///
/// NumPy runs LAPACK `gelsd`, a divide-and-conquer SVD of the whole m×n matrix.
/// The candidate reduces A to a tiny n×n R in one streaming pass over the rows
/// (one Rayon leaf per worker, then a deterministic pairwise fold), back
/// substitutes for x, and takes rank and singular values from an n×n SVD of R
/// since σ(A) = σ(R). Nothing here calls BLAS.
///
/// Both arms receive the identical read-only ndarrays and the identical thread
/// budget: pinning OpenBLAS to one thread while the candidate keeps a full Rayon
/// pool would be an unmatched-config win, not a win.
fn bench_lstsq_tsqr_tall_skinny(_c: &mut Criterion) {
    const REQUIRED_BUILD_PROFILE: &str = "release-perf";
    const CONTRACT_ROUNDS: usize = 21;
    const CONTRACT_MIN_OF: usize = 1;
    const THREAD_ACTIVITY_REPETITIONS: usize = 3;

    assert_eq!(
        std::env::var("FNP_BENCH_PROFILE").as_deref(),
        Ok(REQUIRED_BUILD_PROFILE),
        "ship-grade TSQR lstsq evidence requires FNP_BENCH_PROFILE=release-perf"
    );
    let build_worker =
        std::env::var("FNP_BUILD_WORKER").expect("FNP_BUILD_WORKER records the build origin");
    assert!(
        !build_worker.trim().is_empty(),
        "FNP_BUILD_WORKER must be set"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .expect("RAYON_NUM_THREADS must be explicitly pinned before lstsq timing");
    // Matched config: the incumbent's gelsd is a threaded LAPACK routine, so it
    // gets the same worker count the candidate's Rayon pool gets. Starving one
    // arm's threading layer is trap 2 (unmatched config), and it would bias the
    // ratio in our favour.
    for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        assert_eq!(
            std::env::var(variable).as_deref(),
            Ok(threads.as_str()),
            "{variable} must equal RAYON_NUM_THREADS so both arms get the same thread budget"
        );
    }
    let threads: usize = threads.parse().expect("thread count is numeric");
    assert_eq!(
        rayon::current_num_threads(),
        threads,
        "Rayon pool width does not match the pinned lstsq configuration"
    );

    Python::initialize();
    Python::attach(|py| {
        ensure_numpy_available(py).expect("numpy available");
        let module = PyModule::new(py, "fnp_python_lstsq").expect("lstsq bench module");
        fnp_python(&module).expect("initialize fnp_python lstsq module");
        let numpy = py.import("numpy").expect("numpy incumbent");

        // The candidate builds its four small output arrays through numpy.array
        // (n, 1, and n elements). That is NumPy code inside the candidate arm, so
        // it is disclosed rather than implied absent; the incumbent allocates its
        // own outputs too, so the disclosure is conservative.
        common::report_incumbent_topology_with_shared_component(
            "fnp.lstsq",
            "numpy.linalg.lstsq",
            "numpy.array_small_output_construction",
        );
        println!("NUMPY_BUILD_CONFIG_BEGIN workload=lstsq_tsqr_tall_skinny");
        numpy
            .getattr("show_config")
            .expect("numpy.show_config")
            .call0()
            .expect("report NumPy build configuration");
        println!("NUMPY_BUILD_CONFIG_END workload=lstsq_tsqr_tall_skinny");
        println!(
            "BLAS_RELEVANCE workload=lstsq_tsqr_tall_skinny \
             numpy_lstsq_uses_blas=true candidate_uses_blas=false \
             blas_threads_pinned={threads} \
             reason=incumbent_is_lapack_gelsd_candidate_is_safe_rust_householder"
        );

        let numpy_lstsq = numpy
            .getattr("linalg")
            .expect("numpy.linalg")
            .getattr("lstsq")
            .expect("numpy lstsq");
        let fnp_lstsq = module.getattr("lstsq").expect("fnp lstsq");
        assert!(
            !fnp_lstsq.is(&numpy_lstsq),
            "dispatch trap: fnp.lstsq resolved to the NumPy callable"
        );
        common::report_numpy_incumbent_identity(py, "linalg.lstsq", &numpy_lstsq);

        let allclose = numpy.getattr("allclose").expect("np.allclose");
        let rng = numpy
            .getattr("random")
            .expect("numpy.random")
            .call_method1("default_rng", (12345_u64,))
            .expect("seeded generator");

        // The 4-tuple arms are allclose, not bit-identical (TSQR and gelsd apply
        // different orthogonal sequences), so the in-contract drift digest is
        // taken on a quantised solution. x components here are ~1e-3 and the arms
        // agree to ~1e-17 absolute, so a 1e-9 quantum has eight orders of margin.
        let checksum_of = |result: &pyo3::Bound<'_, pyo3::PyAny>| -> u64 {
            let tuple = result.cast::<PyTuple>().expect("lstsq returns a 4-tuple");
            let solution = tuple.get_item(0).expect("solution");
            let bytes = solution
                .call_method1("round", (9_i32,))
                .expect("quantise solution")
                .call_method0("tobytes")
                .expect("solution tobytes")
                .extract::<Vec<u8>>()
                .expect("solution bytes");
            let rank = tuple
                .get_item(2)
                .expect("rank")
                .extract::<i64>()
                .expect("rank value");
            let mut state = 0xcbf2_9ce4_8422_2325_u64 ^ (rank as u64);
            for byte in bytes {
                state = (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            state
        };

        for (rows, cols, label) in [
            (1_000_000_usize, 8_usize, "1000000x8"),
            (1_000_000, 16, "1000000x16"),
            (2_000_000, 8, "2000000x8"),
        ] {
            let a = rng
                .call_method1("standard_normal", ((rows, cols),))
                .expect("operand a");
            let b = rng
                .call_method1("standard_normal", (rows,))
                .expect("operand b");

            let run_incumbent = || {
                numpy_lstsq
                    .call1((black_box(&a), black_box(&b)))
                    .expect("numpy lstsq arm")
            };
            let run_candidate = || {
                fnp_lstsq
                    .call1((black_box(&a), black_box(&b)))
                    .expect("fnp lstsq arm")
            };

            let ours = run_candidate();
            let theirs = run_incumbent();
            let ours_tuple = ours.cast::<PyTuple>().expect("fnp 4-tuple");
            let theirs_tuple = theirs.cast::<PyTuple>().expect("numpy 4-tuple");
            assert_eq!(ours_tuple.len().expect("len"), 4);
            for (index, field) in ["solution", "residuals", "rank", "singular_values"]
                .into_iter()
                .enumerate()
            {
                let mine = ours_tuple.get_item(index).expect("candidate field");
                let theirs_field = theirs_tuple.get_item(index).expect("incumbent field");
                assert!(
                    allclose
                        .call1((&mine, &theirs_field))
                        .expect("allclose")
                        .extract::<bool>()
                        .expect("bool"),
                    "lstsq {label} {field} diverged from NumPy"
                );
                assert_eq!(
                    mine.getattr("dtype")
                        .expect("candidate dtype")
                        .str()
                        .expect("dtype str")
                        .to_string(),
                    theirs_field
                        .getattr("dtype")
                        .expect("incumbent dtype")
                        .str()
                        .expect("dtype str")
                        .to_string(),
                    "lstsq {label} {field} dtype diverged from NumPy"
                );
            }

            // ROUTE ENGAGEMENT PROOF. If the gate had declined, fnp.lstsq would
            // return NumPy's own object and the two arms would be byte-identical,
            // measuring ~1.0 by construction — the "green route proves nothing"
            // trap. TSQR and gelsd apply different orthogonal sequences, so
            // byte-INEQUALITY on the solution is what shows the native reduction
            // actually produced this answer.
            let ours_bytes = ours_tuple
                .get_item(0)
                .expect("candidate solution")
                .call_method0("tobytes")
                .expect("candidate tobytes")
                .extract::<Vec<u8>>()
                .expect("candidate bytes");
            let theirs_bytes = theirs_tuple
                .get_item(0)
                .expect("incumbent solution")
                .call_method0("tobytes")
                .expect("incumbent tobytes")
                .extract::<Vec<u8>>()
                .expect("incumbent bytes");
            assert_ne!(
                ours_bytes, theirs_bytes,
                "lstsq {label}: candidate returned NumPy's exact bytes, so the \
                 native TSQR route did NOT engage and this row would measure a \
                 passthrough against itself"
            );

            let row = format!("python_lstsq_tsqr_{label}_vs_numpy");
            println!(
                "PARITY row={row} allclose_4tuple=passed exact_dtype=passed \
                 byte_identical=false rows={rows} cols={cols} \
                 input_bytes={} checksum={:016x}",
                rows * cols * 8,
                checksum_of(&theirs)
            );
            println!(
                "ROUTE_PRECONDITIONS row={row} dtype=float64 exact_ndarray=true \
                 c_contiguous=true a_ndim=2 b_ndim=1 rows={rows} cols={cols} \
                 tall_skinny=true full_rank=true rcond=default_none \
                 pinned_threads={threads} host_avx2={} host_avx512f={} \
                 candidate_route=try_native_lstsq_tsqr",
                std::arch::is_x86_feature_detected!("avx2"),
                std::arch::is_x86_feature_detected!("avx512f"),
            );
            println!(
                "COUNTED_MECHANISM row={row} class=streaming_qr_vs_full_svd \
                 incumbent_algorithm=lapack_gelsd_divide_and_conquer_svd_of_m_by_n \
                 candidate_algorithm=parallel_householder_tsqr_to_n_by_n_r_then_back_substitution \
                 candidate_input_copies=0_zerocopy_pybuffer \
                 candidate_full_passes_over_a=1 incumbent_full_passes_over_a=at_least_1 \
                 candidate_residual=accumulated_in_tree_no_second_pass shared_input=true"
            );

            common::report_observed_thread_activity(
                &row,
                "numpy",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_incumbent()));
                },
            );
            common::report_observed_thread_activity(
                &row,
                "fnp",
                THREAD_ACTIVITY_REPETITIONS,
                || {
                    black_box(checksum_of(&run_candidate()));
                },
            );

            let mut observe_incumbent = || {
                let started = std::time::Instant::now();
                let result = run_incumbent();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let mut observe_candidate = || {
                let started = std::time::Instant::now();
                let result = run_candidate();
                let elapsed = started.elapsed();
                common::ContractObservation {
                    elapsed,
                    checksum: checksum_of(&result),
                }
            };
            let (effect, incumbent_null, candidate_null) =
                common::run_dual_null_median_ci_contract_with_sampling(
                    &row,
                    &mut observe_incumbent,
                    &mut observe_candidate,
                    CONTRACT_ROUNDS,
                    CONTRACT_MIN_OF,
                );
            let verdict =
                common::dual_null_contract_verdict(effect, incumbent_null, candidate_null);
            println!(
                "LSTSQ_TSQR_RESULT row={row} rows={rows} cols={cols} threads={threads} \
                 verdict={verdict} incumbent_median_ms={:.6} candidate_median_ms={:.6} \
                 ratio_median={:.6} ratio_ci95=[{:.6},{:.6}] \
                 incumbent_null_ratio={:.6} incumbent_null_ci95=[{:.6},{:.6}] \
                 candidate_null_ratio={:.6} candidate_null_ci95=[{:.6},{:.6}] \
                 corrected_dual_null_gate=true incumbent=numpy_live_same_invocation",
                effect.arm_a_median_ns / 1_000_000.0,
                effect.arm_b_median_ns / 1_000_000.0,
                effect.ratio_median,
                effect.ratio_ci_low,
                effect.ratio_ci_high,
                incumbent_null.ratio_median,
                incumbent_null.ratio_ci_low,
                incumbent_null.ratio_ci_high,
                candidate_null.ratio_median,
                candidate_null.ratio_ci_low,
                candidate_null.ratio_ci_high,
            );
            let decision = if verdict == "DECIDABLE_WIN" {
                "choose_fnp"
            } else {
                "choose_numpy"
            };
            println!(
                "CHOOSER_STATEMENT workload=lstsq_tsqr_{label} decision={decision} \
                 verdict={verdict} incumbent=numpy_live_same_invocation \
                 measured_scope={rows}x{cols}_c_contiguous_float64_full_rank_1d_rhs_at_{threads}_matched_threads \
                 outside_scope=run_same_contract_before_choosing"
            );
        }
    });
}

fn main() {
    common::gated_main(&[("bench_lstsq_tsqr_tall_skinny", bench_lstsq_tsqr_tall_skinny)]);
}
