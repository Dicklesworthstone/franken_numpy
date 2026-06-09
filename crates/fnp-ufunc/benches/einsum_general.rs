//! einsum general-path A/B: cases that fall through the 2-operand GEMM and
//! outer-product fast paths into the general radix-decode contraction loop.
//! Single-operand reductions with a large contracted subspace are the clearest
//! demonstration of the per-element decode cost.

use criterion::{Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn arr(shape: &[usize]) -> UFuncArray {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 0.5).collect();
    UFuncArray::new(shape.to_vec(), data, DType::F64).unwrap()
}

fn bench_einsum_general(c: &mut Criterion) {
    // (subscripts, operand shapes)
    let cases: &[(&str, Vec<Vec<usize>>)] = &[
        ("ij->ji", vec![vec![2048, 2048]]),
        ("ii->i", vec![vec![4096, 4096]]),
        ("ijk->i", vec![vec![256, 256, 256]]),
        ("ijk->ik", vec![vec![256, 256, 256]]),
        ("ijkl->il", vec![vec![64, 64, 64, 64]]),
        ("ij,ij->i", vec![vec![1024, 1024], vec![1024, 1024]]),
        (
            "ij,jk,kl->il",
            vec![vec![192, 192], vec![192, 192], vec![192, 192]],
        ),
    ];
    for (subs, shapes) in cases {
        let ops: Vec<UFuncArray> = shapes.iter().map(|s| arr(s)).collect();
        let refs: Vec<&UFuncArray> = ops.iter().collect();
        // sanity: ensure it runs
        let _ = UFuncArray::einsum(subs, &refs).unwrap();
        let mut group = c.benchmark_group(format!("einsum_{}", subs.replace(['-', '>', ','], "_")));
        group.sample_size(20);
        group.bench_function("eval", |b| {
            b.iter(|| UFuncArray::einsum(black_box(subs), black_box(&refs)).unwrap())
        });
        group.finish();
    }
}

criterion_group!(benches, bench_einsum_general);
criterion_main!(benches);
