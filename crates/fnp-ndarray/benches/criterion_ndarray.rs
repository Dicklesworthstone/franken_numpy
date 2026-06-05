//! Criterion benchmarks for fnp-ndarray.
//!
//! Measures performance baselines for array layout operations:
//! - can_broadcast: check if two shapes can broadcast
//! - broadcast_shape: compute result shape from two inputs
//! - broadcast_shapes: compute result shape from N inputs
//! - element_count: count total elements in shape
//! - contiguous_strides: compute C-order strides
//! - NdLayout operations: as_strided, broadcast_to, is_contiguous

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_ndarray::{
    MemoryOrder, NdLayout, broadcast_shape, broadcast_shapes, broadcast_strides, can_broadcast,
    contiguous_strides, element_count,
};
use std::hint::black_box;

// ─────────────────────────────────────────────────────────────────────────────
// can_broadcast benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_can_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("can_broadcast");

    let cases = [
        ("scalar_to_1d", &[][..], &[100][..]),
        ("1d_to_1d_same", &[100][..], &[100][..]),
        ("1d_to_1d_one", &[1][..], &[100][..]),
        ("2d_to_2d", &[10, 20][..], &[10, 20][..]),
        ("2d_broadcast", &[1, 20][..], &[10, 1][..]),
        ("3d_broadcast", &[1, 5, 1][..], &[4, 1, 6][..]),
        ("4d_broadcast", &[1, 2, 1, 4][..], &[3, 1, 5, 1][..]),
        ("high_dim", &[1, 2, 3, 4, 5][..], &[5, 4, 3, 2, 1][..]),
    ];

    for (name, lhs, rhs) in cases {
        group.bench_with_input(
            BenchmarkId::new("shapes", name),
            &(lhs, rhs),
            |b, (l, r)| {
                b.iter(|| can_broadcast(black_box(l), black_box(r)));
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shape benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_broadcast_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_shape");

    let cases = [
        ("scalar_to_1d", &[][..], &[100][..]),
        ("1d_same", &[100][..], &[100][..]),
        ("2d_broadcast", &[1, 20][..], &[10, 1][..]),
        ("3d_broadcast", &[1, 5, 1][..], &[4, 1, 6][..]),
        ("4d_broadcast", &[1, 2, 1, 4][..], &[3, 1, 5, 1][..]),
    ];

    for (name, lhs, rhs) in cases {
        group.bench_with_input(
            BenchmarkId::new("shapes", name),
            &(lhs, rhs),
            |b, (l, r)| {
                b.iter(|| broadcast_shape(black_box(l), black_box(r)));
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_shapes (multi-input) benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_broadcast_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_shapes");

    // 2 shapes
    let shapes_2: Vec<&[usize]> = vec![&[10, 1], &[1, 20]];
    group.bench_function("2_shapes", |b| {
        b.iter(|| broadcast_shapes(black_box(&shapes_2)));
    });

    // 4 shapes
    let shapes_4: Vec<&[usize]> = vec![&[1, 10], &[5, 1], &[1, 1], &[5, 10]];
    group.bench_function("4_shapes", |b| {
        b.iter(|| broadcast_shapes(black_box(&shapes_4)));
    });

    // 8 shapes
    let shapes_8: Vec<&[usize]> = vec![
        &[1, 1, 10],
        &[1, 5, 1],
        &[3, 1, 1],
        &[1, 1, 10],
        &[3, 5, 1],
        &[1, 5, 10],
        &[3, 1, 10],
        &[1, 1, 1],
    ];
    group.bench_function("8_shapes", |b| {
        b.iter(|| broadcast_shapes(black_box(&shapes_8)));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// element_count benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_element_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_count");

    let shapes = [
        ("1d_small", vec![100]),
        ("1d_large", vec![1_000_000]),
        ("2d", vec![1000, 1000]),
        ("3d", vec![100, 100, 100]),
        ("4d", vec![10, 20, 30, 40]),
        ("high_dim", vec![2, 3, 4, 5, 6, 7]),
    ];

    for (name, shape) in shapes {
        group.bench_with_input(BenchmarkId::new("shape", name), &shape, |b, s| {
            b.iter(|| element_count(black_box(s)));
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// contiguous_strides benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_contiguous_strides(c: &mut Criterion) {
    let mut group = c.benchmark_group("contiguous_strides");

    let shapes = [
        ("1d", vec![100]),
        ("2d", vec![100, 100]),
        ("3d", vec![10, 20, 30]),
        ("4d", vec![5, 10, 15, 20]),
        ("6d", vec![2, 3, 4, 5, 6, 7]),
    ];

    for (name, shape) in shapes {
        group.bench_with_input(BenchmarkId::new("shape", name), &shape, |b, s| {
            b.iter(|| contiguous_strides(black_box(s), 8, MemoryOrder::C));
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_strides benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_broadcast_strides(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_strides");

    let cases: Vec<(&str, Vec<usize>, Vec<usize>)> = vec![
        ("2d_expand_rows", vec![1, 10], vec![100, 10]),
        ("2d_expand_cols", vec![10, 1], vec![10, 100]),
        ("3d_expand", vec![1, 5, 1], vec![4, 5, 6]),
        ("4d_expand", vec![1, 2, 1, 4], vec![3, 2, 5, 4]),
    ];

    for (name, from_shape, to_shape) in cases {
        let src_strides = contiguous_strides(&from_shape, 8, MemoryOrder::C).unwrap();
        group.bench_with_input(
            BenchmarkId::new("expand", name),
            &(from_shape.clone(), src_strides.clone(), to_shape.clone()),
            |b, (from, strides, to)| {
                b.iter(|| broadcast_strides(black_box(from), black_box(strides), black_box(to)));
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// NdLayout operations benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_ndlayout_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("NdLayout_contiguous");

    let shapes = [
        ("1d", vec![1000]),
        ("2d", vec![100, 100]),
        ("3d", vec![10, 20, 30]),
        ("4d", vec![5, 10, 15, 20]),
    ];

    for (name, shape) in shapes {
        group.bench_with_input(BenchmarkId::new("create", name), &shape, |b, s| {
            b.iter(|| NdLayout::contiguous(black_box(s.clone()), 8, MemoryOrder::C));
        });
    }

    group.finish();
}

fn bench_ndlayout_broadcast_to(c: &mut Criterion) {
    let mut group = c.benchmark_group("NdLayout_broadcast_to");

    let cases = [
        ("expand_1d", vec![1], vec![1000]),
        ("expand_2d_rows", vec![1, 100], vec![1000, 100]),
        ("expand_3d", vec![1, 10, 1], vec![5, 10, 20]),
    ];

    for (name, from_shape, to_shape) in cases {
        let layout = NdLayout::contiguous(from_shape.clone(), 8, MemoryOrder::C).unwrap();
        group.bench_with_input(
            BenchmarkId::new("broadcast", name),
            &(layout, to_shape),
            |b, (l, ts)| {
                b.iter(|| l.broadcast_to(black_box(ts.clone())));
            },
        );
    }

    group.finish();
}

fn bench_ndlayout_is_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("NdLayout_is_contiguous");

    // Contiguous layout
    let contiguous = NdLayout::contiguous(vec![100, 100], 8, MemoryOrder::C).unwrap();
    group.bench_function("contiguous_2d", |b| {
        b.iter(|| black_box(&contiguous).is_contiguous());
    });

    // Non-contiguous (broadcast)
    let layout_1x100 = NdLayout::contiguous(vec![1, 100], 8, MemoryOrder::C).unwrap();
    let broadcast = layout_1x100.broadcast_to(vec![100, 100]).unwrap();
    group.bench_function("broadcast_2d", |b| {
        b.iter(|| black_box(&broadcast).is_contiguous());
    });

    // High dimensional contiguous
    let high_dim = NdLayout::contiguous(vec![2, 3, 4, 5, 6], 8, MemoryOrder::C).unwrap();
    group.bench_function("contiguous_5d", |b| {
        b.iter(|| black_box(&high_dim).is_contiguous());
    });

    group.finish();
}

fn bench_ndlayout_has_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("NdLayout_has_overlap");

    // No overlap (contiguous)
    let contiguous = NdLayout::contiguous(vec![100, 100], 8, MemoryOrder::C).unwrap();
    group.bench_function("no_overlap_contiguous", |b| {
        b.iter(|| black_box(&contiguous).has_internal_overlap());
    });

    // Potential overlap (broadcast from [1,100] to [100,100])
    let layout_1x100 = NdLayout::contiguous(vec![1, 100], 8, MemoryOrder::C).unwrap();
    let broadcast = layout_1x100.broadcast_to(vec![100, 100]).unwrap();
    group.bench_function("overlap_broadcast", |b| {
        b.iter(|| black_box(&broadcast).has_internal_overlap());
    });

    group.finish();
}

fn bench_ndlayout_nbytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("NdLayout_nbytes");

    let layouts = [
        (
            "small_1d",
            NdLayout::contiguous(vec![100], 8, MemoryOrder::C).unwrap(),
        ),
        (
            "medium_2d",
            NdLayout::contiguous(vec![100, 100], 8, MemoryOrder::C).unwrap(),
        ),
        (
            "large_3d",
            NdLayout::contiguous(vec![100, 100, 100], 8, MemoryOrder::C).unwrap(),
        ),
    ];

    for (name, layout) in layouts {
        group.bench_with_input(BenchmarkId::new("compute", name), &layout, |b, l| {
            b.iter(|| l.nbytes());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_can_broadcast,
    bench_broadcast_shape,
    bench_broadcast_shapes,
    bench_element_count,
    bench_contiguous_strides,
    bench_broadcast_strides,
    bench_ndlayout_contiguous,
    bench_ndlayout_broadcast_to,
    bench_ndlayout_is_contiguous,
    bench_ndlayout_has_overlap,
    bench_ndlayout_nbytes,
);

criterion_main!(benches);
