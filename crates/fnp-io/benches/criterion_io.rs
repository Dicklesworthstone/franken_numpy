//! Criterion benchmarks for fnp-io.
//!
//! Measures performance baselines for I/O operations:
//! - write_npy_bytes: serialize array to .npy format
//! - read_npy_bytes: deserialize array from .npy format
//! - write_npz_bytes: serialize multiple arrays to .npz archive
//! - read_npz_bytes: deserialize .npz archive
//!
//! These operations are critical for data persistence workflows.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_io::{
    IOSupportedDType, NpyHeader, read_npy_bytes, read_npz_bytes, write_npy_bytes, write_npz_bytes,
};
use std::hint::black_box;

fn generate_f64_data(n: usize) -> Vec<u8> {
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    bytemuck::cast_slice(&data).to_vec()
}

fn make_npy_header(shape: &[usize]) -> NpyHeader {
    NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: shape.to_vec(),
    }
}

fn bench_write_npy(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_npy_bytes");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| {
                let result = write_npy_bytes(black_box(&header), black_box(&data), false);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_read_npy(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_npy_bytes");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);
        let npy_bytes = write_npy_bytes(&header, &data, false).expect("write");

        group.throughput(Throughput::Bytes(npy_bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("elements", n),
            &npy_bytes,
            |bench, payload| {
                bench.iter(|| {
                    let result = read_npy_bytes(black_box(payload), false);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_write_npz(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_npz_bytes");

    for num_arrays in [1, 5, 10, 20] {
        let n = 10_000;
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
            .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
            .collect();

        let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
            .iter()
            .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
            .collect();

        let total_bytes = (num_arrays * n * 8) as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("num_arrays", num_arrays),
            &entry_refs,
            |bench, refs| {
                bench.iter(|| {
                    let result = write_npz_bytes(black_box(refs));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_read_npz(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_npz_bytes");

    for num_arrays in [1, 5, 10, 20] {
        let n = 10_000;
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
            .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
            .collect();

        let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
            .iter()
            .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
            .collect();

        let npz_bytes = write_npz_bytes(&entry_refs).expect("write npz");

        group.throughput(Throughput::Bytes(npz_bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("num_arrays", num_arrays),
            &npz_bytes,
            |bench, payload| {
                bench.iter(|| {
                    let result = read_npz_bytes(black_box(payload), false);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_npy_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("npy_roundtrip");

    for n in [10_000, 100_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| {
                let written = write_npy_bytes(black_box(&header), black_box(&data), false).unwrap();
                let read = read_npy_bytes(black_box(&written), false).unwrap();
                black_box(read)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_write_npy,
    bench_read_npy,
    bench_write_npz,
    bench_read_npz,
    bench_npy_roundtrip,
);

criterion_main!(benches);
