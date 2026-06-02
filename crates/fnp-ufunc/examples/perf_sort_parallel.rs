//! Profiling-only golden + timing harness for the parallel last-axis sort lever
//! (bead franken_numpy-2ffgx).
//!
//! Sorts a large 2-D f64 array along axis=-1 via the public API (now parallel
//! across lanes) and compares against a serial per-row `sort_unstable_by(total_cmp)`
//! reference. Asserts the result is bit-for-bit identical (the isomorphism proof),
//! prints an FNV-1a checksum, and reports the speedup.
//!
//! Run: `cargo run --release --example perf_sort_parallel -p fnp-ufunc -- 2048`

use std::time::Instant;

use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;

fn fnv1a(values: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn time_median_ms<T, F: FnMut() -> T>(iters: usize, mut run: F) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(run());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

// Serial reference: each contiguous row sorted with the same total order the
// quicksort kind uses for finite, non-integer data.
fn serial_row_sort(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = data.to_vec();
    for r in 0..rows {
        out[r * cols..(r + 1) * cols].sort_unstable_by(f64::total_cmp);
    }
    out
}

fn main() {
    let dim = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048usize);
    let (rows, cols) = (dim, dim);
    // Non-integer, finite values (avoid the exact-integer counting-sort fast path).
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| (((i as u64).wrapping_mul(2654435761) % 1_000_003) as f64) / 7.0 + 0.5)
        .collect();

    let arr = UFuncArray::new(vec![rows, cols], data.clone(), DType::F64).unwrap();
    let parallel = arr.sort(Some(-1), Some("quicksort")).expect("sort");
    let serial = serial_row_sort(&data, rows, cols);
    assert_eq!(parallel.values(), serial.as_slice(), "parallel sort != serial");
    let checksum = fnv1a(parallel.values());

    for _ in 0..2 {
        std::hint::black_box(arr.sort(Some(-1), Some("quicksort")).unwrap());
    }
    let par_ms = time_median_ms(7, || arr.sort(Some(-1), Some("quicksort")).unwrap());
    let ser_ms = time_median_ms(7, || serial_row_sort(&data, rows, cols));
    let speedup = ser_ms / par_ms;
    println!(
        "sort_{rows}x{cols} fnv1a=0x{checksum:016x} serial_ms={ser_ms:.4} parallel_ms={par_ms:.4} speedup={speedup:.2}x"
    );
}
