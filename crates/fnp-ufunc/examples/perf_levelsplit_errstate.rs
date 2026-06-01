//! Profiling-only level-split harness (measurement, not a correctness test).
//!
//! Attributes the `ufunc_add_broadcast` hotspot from the perf campaign
//! (`tests/artifacts/perf/2026-06-01_perf_baseline/`). The elementwise binary
//! kernels split on `geterr().is_all_ignore()`: when the thread-local float
//! error state is *not* all-ignore (the NumPy-compatible default is
//! `divide/over/invalid = warn`), the hot loop calls `note_binary_float_errors`
//! per element, which blocks auto-vectorization. This harness measures the same
//! add under the default state vs. an all-ignore state to quantify that cost.
//!
//! Run: `cargo run --release --example perf_levelsplit_errstate -p fnp-ufunc`
//! It prints median wall-clock per op and the default/all-ignore ratio. It does
//! not assert and is excluded from the correctness suites.

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::{FloatErrorMode, FloatErrorState, add, seterr_state};

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn time_op<F: Fn()>(iters: usize, warmup: usize, f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e3); // ms
    }
    median(samples)
}

fn main() {
    let dim = 1024usize;
    let lhs = UFuncArrayLike::matrix(dim);
    let rhs = UFuncArrayLike::row(dim);

    let all_ignore = FloatErrorState {
        divide: FloatErrorMode::Ignore,
        over: FloatErrorMode::Ignore,
        under: FloatErrorMode::Ignore,
        invalid: FloatErrorMode::Ignore,
    };

    // Default (NumPy-compatible warn) state — restore it after toggling.
    let default_state = fnp_ufunc::geterr();

    let iters = 60;
    let warmup = 10;

    // Broadcast add [1024,1024] (+) [1024] under default (warn) errstate.
    let _ = seterr_state(default_state);
    let broadcast_default = time_op(iters, warmup, || {
        std::hint::black_box(add(&lhs.0, &rhs.0).unwrap());
    });

    // Same add under all-ignore (hits the vectorizable fast path).
    let _ = seterr_state(all_ignore);
    let broadcast_ignore = time_op(iters, warmup, || {
        std::hint::black_box(add(&lhs.0, &rhs.0).unwrap());
    });

    // Equal-shape 1M add to show the split is systemic, not broadcast-specific.
    let rhs_full = UFuncArrayLike::matrix(dim);
    let _ = seterr_state(default_state);
    let equal_default = time_op(iters, warmup, || {
        std::hint::black_box(add(&lhs.0, &rhs_full.0).unwrap());
    });
    let _ = seterr_state(all_ignore);
    let equal_ignore = time_op(iters, warmup, || {
        std::hint::black_box(add(&lhs.0, &rhs_full.0).unwrap());
    });

    // reduce_sum control (single code path, no errstate split) for scale.
    let _ = seterr_state(default_state);
    let reduce_default = time_op(iters, warmup, || {
        std::hint::black_box(lhs.0.reduce_sum(Some(1), false).unwrap());
    });

    // --- Stage isolation: construction vs raw arithmetic ---
    // Cost of building one result array (the path every elementwise op runs on
    // its output: from_storage_with_dtype -> cast_to -> to_f64_vec -> new).
    let proto: Vec<f64> = (0..dim * dim).map(|i| (i % 257) as f64 * 0.5 + 1.0).collect();
    let construct_only = time_op(iters, warmup, || {
        std::hint::black_box(fnp_ufunc::UFuncArray::from_vec(proto.clone()));
    });
    // Cost of a hand-written f64 add into a fresh Vec (no library construction):
    // this is what the arithmetic alone "should" cost (memory-bound).
    let lhs_raw = proto.clone();
    let rhs_raw: Vec<f64> = (0..dim * dim).map(|i| (i % 29) as f64).collect();
    let raw_add = time_op(iters, warmup, || {
        let mut out = vec![0.0f64; lhs_raw.len()];
        for i in 0..out.len() {
            out[i] = lhs_raw[i] + rhs_raw[i];
        }
        std::hint::black_box(out[0]);
    });
    // Cost of just allocating+zeroing the 8MB output buffer.
    let alloc_only = time_op(iters, warmup, || {
        std::hint::black_box(vec![0.0f64; lhs_raw.len()]);
    });

    let _ = seterr_state(default_state);

    println!("# perf level-split: errstate cost on elementwise add (1024x1024)");
    println!("op,errstate,median_ms");
    println!("add_broadcast_1024x1024_by_1024,default_warn,{broadcast_default:.4}");
    println!("add_broadcast_1024x1024_by_1024,all_ignore,{broadcast_ignore:.4}");
    println!("add_equal_1024x1024,default_warn,{equal_default:.4}");
    println!("add_equal_1024x1024,all_ignore,{equal_ignore:.4}");
    println!("reduce_sum_axis1_1024x1024,default_warn,{reduce_default:.4}");
    println!("construct_from_vec_1m,n/a,{construct_only:.4}");
    println!("raw_handwritten_add_1m,n/a,{raw_add:.4}");
    println!("alloc_zero_8mb,n/a,{alloc_only:.4}");
    println!();
    println!(
        "broadcast default/all-ignore ratio = {:.2}x",
        broadcast_default / broadcast_ignore
    );
    println!(
        "equal-shape default/all-ignore ratio = {:.2}x",
        equal_default / equal_ignore
    );
}

/// Tiny wrapper so the array construction stays readable above.
struct UFuncArrayLike(fnp_ufunc::UFuncArray);

impl UFuncArrayLike {
    fn matrix(dim: usize) -> Self {
        let values: Vec<f64> = (0..dim * dim).map(|i| (i % 257) as f64 * 0.5 + 1.0).collect();
        Self(fnp_ufunc::UFuncArray::new(vec![dim, dim], values, DType::F64).unwrap())
    }
    fn row(dim: usize) -> Self {
        let values: Vec<f64> = (0..dim).map(|i| (i % 29) as f64).collect();
        Self(fnp_ufunc::UFuncArray::new(vec![dim], values, DType::F64).unwrap())
    }
}
