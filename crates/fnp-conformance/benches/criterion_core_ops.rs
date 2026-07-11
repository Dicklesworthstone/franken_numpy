#![forbid(unsafe_code)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_io::{IOSupportedDType, load, save};
use fnp_ufunc::{BinaryOp, UFuncArray};
use std::hint::black_box;
use std::time::Duration;

fn build_matrix_values(dim: usize, step: usize, modulo: usize) -> Vec<f64> {
    (0..(dim * dim))
        .map(|i| f64::from(((i * step) % modulo) as u32))
        .collect()
}

fn bench_core_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_ops");

    let add_dim = 1024usize;
    let add_elements = add_dim * add_dim;
    let add_lhs = UFuncArray::new(
        vec![add_dim, add_dim],
        build_matrix_values(add_dim, 7, 257),
        DType::F64,
    )
    .expect("broadcast lhs setup must succeed");
    let add_rhs = UFuncArray::new(
        vec![add_dim],
        (0..add_dim).map(|i| f64::from((i % 29) as u32)).collect(),
        DType::F64,
    )
    .expect("broadcast rhs setup must succeed");
    group.bench_with_input(
        BenchmarkId::new("ufunc_add_broadcast", "1024x1024_by_1024"),
        &add_elements,
        |b, _| {
            b.iter(|| {
                let out = add_lhs
                    .elementwise_binary(&add_rhs, BinaryOp::Add)
                    .expect("broadcast add must succeed");
                black_box(out.values()[0]);
            });
        },
    );

    let reduce_dim = 1024usize;
    let reduce_in = UFuncArray::new(
        vec![reduce_dim, reduce_dim],
        build_matrix_values(reduce_dim, 17, 509),
        DType::F64,
    )
    .expect("reduction setup must succeed");
    group.bench_function("reduce_sum_axis1_1024x1024", |b| {
        b.iter(|| {
            let out = reduce_in
                .reduce_sum(Some(1), false)
                .expect("axis reduction must succeed");
            black_box(out.values()[0]);
        });
    });

    group.bench_function("reduce_prod_axis1_1024x1024", |b| {
        b.iter(|| {
            let out = reduce_in
                .reduce_prod(Some(1), false)
                .expect("axis product reduction must succeed");
            black_box(out.values()[0]);
        });
    });

    let matmul_dim = 256usize;
    let matmul_lhs = UFuncArray::new(
        vec![matmul_dim, matmul_dim],
        build_matrix_values(matmul_dim, 13, 997),
        DType::F64,
    )
    .expect("matmul lhs setup must succeed");
    let matmul_rhs = UFuncArray::new(
        vec![matmul_dim, matmul_dim],
        build_matrix_values(matmul_dim, 19, 991),
        DType::F64,
    )
    .expect("matmul rhs setup must succeed");
    group.bench_function("matmul_256x256_by_256x256", |b| {
        b.iter(|| {
            let out = matmul_lhs.matmul(&matmul_rhs).expect("matmul must succeed");
            black_box(out.values()[0]);
        });
    });

    let sort_len = 1_000_000usize;
    let sort_in = UFuncArray::new(
        vec![sort_len],
        (0..sort_len)
            .map(|i| f64::from(((i * 48_271) % sort_len) as u32))
            .collect(),
        DType::F64,
    )
    .expect("sort setup must succeed");
    group.bench_function("sort_quicksort_1m", |b| {
        b.iter(|| {
            let out = sort_in
                .sort(None, Some("quicksort"))
                .expect("sort must succeed");
            black_box(out.values()[0]);
        });
    });

    let fft_len = 65_536usize;
    let fft_in = UFuncArray::new(
        vec![fft_len],
        (0..fft_len)
            .map(|i| {
                let t = i as f64 / fft_len as f64;
                (std::f64::consts::TAU * 5.0 * t).sin()
                    + 0.5 * (std::f64::consts::TAU * 13.0 * t).cos()
            })
            .collect(),
        DType::F64,
    )
    .expect("fft setup must succeed");
    group.bench_function("fft_65536", |b| {
        b.iter(|| {
            let out = fft_in.fft(None).expect("fft must succeed");
            black_box(out.values()[0]);
        });
    });

    let astype_dim = 1024usize;
    let astype_elements = astype_dim * astype_dim;
    let astype_in = UFuncArray::new(
        vec![astype_dim, astype_dim],
        build_matrix_values(astype_dim, 23, 10_003),
        DType::F64,
    )
    .expect("astype setup must succeed");
    group.bench_with_input(
        BenchmarkId::new("astype_f64_to_i32", "1024x1024"),
        &astype_elements,
        |b, _| {
            b.iter(|| {
                let out = astype_in.astype(DType::I32);
                black_box(out.values()[0]);
            });
        },
    );

    group.bench_function("reshape_1024x1024_to_2048x512", |b| {
        b.iter(|| {
            let out = astype_in
                .reshape(&[2048, 512])
                .expect("reshape must succeed");
            black_box(out.shape()[0]);
        });
    });

    let io_dim = 512usize;
    let io_values: Vec<f64> = (0..(io_dim * io_dim))
        .map(|i| f64::from(((i * 29) % 65_537) as u32) / 11.0)
        .collect();
    group.bench_function("io_npy_save_load_512x512_f64", |b| {
        b.iter(|| {
            let payload = save(&[io_dim, io_dim], &io_values, IOSupportedDType::F64).expect("save");
            let (shape, values, dtype) = load(&payload).expect("load");
            black_box(shape);
            black_box(values[0]);
            black_box(dtype);
        });
    });

    group.finish();
}

// One-binary ABBA/BAAB median gate for the reduce_fold last-axis row-band
// lever, per the 2026-07-11 no-ship retry predicate: serial arm inline (exact
// per-row left fold), candidate arm = the public reduce_prod path, finite
// near-one input, plus a serial/serial A/A null. Run with RAYON_NUM_THREADS
// pinned via the runner wrapper and RCH_WORKER pinned to the no-ship worker.
fn bench_reduce_prod_row_band_median_gate(c: &mut Criterion) {
    use std::cell::{Cell, RefCell};
    use std::time::Instant;

    let mut group = c.benchmark_group("core_reduce_prod_median_gate");
    let rows = 1024usize;
    let cols = 1024usize;
    let values: Vec<f64> = (0..rows * cols)
        .map(|i| 1.0 + (((i * 131) % 4093) as f64 - 2046.0) * 1.0e-9)
        .collect();
    let array = UFuncArray::new(vec![rows, cols], values.clone(), DType::F64)
        .expect("median-gate input must build");
    let serial_arm = || {
        let mut out = Vec::with_capacity(rows);
        for row in values.chunks_exact(cols) {
            out.push(row[1..].iter().fold(row[0], |acc, &v| acc * v));
        }
        out
    };
    let candidate_arm = || {
        array
            .reduce_prod(Some(1), false)
            .expect("candidate reduce_prod must succeed")
    };
    // Bit parity before timing: the row-band candidate must equal the serial
    // per-row fold exactly.
    let serial_ref = serial_arm();
    let candidate_ref = candidate_arm();
    assert_eq!(candidate_ref.values().len(), serial_ref.len());
    for (candidate, serial) in candidate_ref.values().iter().zip(serial_ref.iter()) {
        assert_eq!(
            candidate.to_bits(),
            serial.to_bits(),
            "row-band product must be bit-identical to the serial fold"
        );
    }

    let report = |label: &str, a: &RefCell<Vec<f64>>, b: &RefCell<Vec<f64>>| {
        let median = |samples: &mut Vec<f64>| -> f64 {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let mut a_ns = a.borrow().clone();
        let mut b_ns = b.borrow().clone();
        if a_ns.is_empty() || b_ns.is_empty() {
            return;
        }
        let ratios: Vec<f64> = a_ns.iter().zip(b_ns.iter()).map(|(x, y)| x / y).collect();
        let mut sorted = ratios.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "CORE_PROD_GATE row={label} samples={} a_median_us={:.3} b_median_us={:.3} \
             ratio_median={:.4} ratio_p10={:.4} ratio_p90={:.4}",
            sorted.len(),
            median(&mut a_ns) / 1e3,
            median(&mut b_ns) / 1e3,
            sorted[sorted.len() / 2],
            sorted[sorted.len() / 10],
            sorted[sorted.len() - 1 - sorted.len() / 10],
        );
    };

    let serial_ns = RefCell::new(Vec::new());
    let candidate_ns = RefCell::new(Vec::new());
    let order = Cell::new(0u64);
    group.bench_function("prod_1024x1024_serial_vs_rowband_paired", |bench| {
        bench.iter_custom(|iterations| {
            let mut serial_total = Duration::ZERO;
            let mut candidate_total = Duration::ZERO;
            for _ in 0..iterations {
                let serial_first = order.get() & 1 == 0;
                order.set(order.get().wrapping_add(1));
                if serial_first {
                    let start = Instant::now();
                    black_box(serial_arm());
                    serial_total += start.elapsed();
                    let start = Instant::now();
                    black_box(candidate_arm());
                    candidate_total += start.elapsed();
                    let start = Instant::now();
                    black_box(candidate_arm());
                    candidate_total += start.elapsed();
                    let start = Instant::now();
                    black_box(serial_arm());
                    serial_total += start.elapsed();
                } else {
                    let start = Instant::now();
                    black_box(candidate_arm());
                    candidate_total += start.elapsed();
                    let start = Instant::now();
                    black_box(serial_arm());
                    serial_total += start.elapsed();
                    let start = Instant::now();
                    black_box(serial_arm());
                    serial_total += start.elapsed();
                    let start = Instant::now();
                    black_box(candidate_arm());
                    candidate_total += start.elapsed();
                }
            }
            serial_ns
                .borrow_mut()
                .push(serial_total.as_secs_f64() * 1e9 / (2.0 * iterations as f64));
            candidate_ns
                .borrow_mut()
                .push(candidate_total.as_secs_f64() * 1e9 / (2.0 * iterations as f64));
            serial_total + candidate_total
        });
    });
    report("serial_over_rowband", &serial_ns, &candidate_ns);

    let null_a = RefCell::new(Vec::new());
    let null_b = RefCell::new(Vec::new());
    let null_order = Cell::new(0u64);
    group.bench_function("prod_1024x1024_serial_null_aa", |bench| {
        bench.iter_custom(|iterations| {
            let mut a_total = Duration::ZERO;
            let mut b_total = Duration::ZERO;
            for _ in 0..iterations {
                let b_first = null_order.get() & 1 == 1;
                null_order.set(null_order.get().wrapping_add(1));
                if b_first {
                    let start = Instant::now();
                    black_box(serial_arm());
                    b_total += start.elapsed();
                    let start = Instant::now();
                    black_box(serial_arm());
                    a_total += start.elapsed();
                } else {
                    let start = Instant::now();
                    black_box(serial_arm());
                    a_total += start.elapsed();
                    let start = Instant::now();
                    black_box(serial_arm());
                    b_total += start.elapsed();
                }
            }
            null_a
                .borrow_mut()
                .push(a_total.as_secs_f64() * 1e9 / iterations as f64);
            null_b
                .borrow_mut()
                .push(b_total.as_secs_f64() * 1e9 / iterations as f64);
            a_total + b_total
        });
    });
    report("serial_null_aa", &null_a, &null_b);

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(8))
        .warm_up_time(Duration::from_secs(2))
        .sample_size(12)
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_core_ops, bench_reduce_prod_row_band_median_gate
}
criterion_main!(benches);
