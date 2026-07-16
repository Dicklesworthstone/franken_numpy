//! 2-D transpose A/B: old per-element coordinate-decomposition gather vs the
//! shipped cache-tiled parallel data move (`transpose_2d_par`).
//!
//! "new" calls the real `UFuncArray::transpose` (exercises the production fast
//! path). "old" replicates the previous behaviour: a flat scan that decomposes
//! every output index into a multi-index with `ndim` integer divisions and then
//! gathers strided from the source. Both produce bit-identical output.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use rayon::prelude::*;
use std::hint::black_box;

fn focused_identity_only() -> bool {
    std::env::args().any(|arg| arg.starts_with("transpose_identity"))
}

fn focused_suffix_only() -> bool {
    std::env::args().any(|arg| arg.starts_with("transpose_suffix"))
}

/// Previous kernel: per-element index decomposition + strided gather (2-D).
fn old_transpose(values: &[f64], r: usize, c: usize) -> Vec<f64> {
    // new shape is [c, r]; new C-strides are [r, 1]; old C-strides are [c, 1].
    let total = r * c;
    let mut out = vec![0.0f64; total];
    let new_strides = [r, 1usize];
    let old_strides = [c, 1usize];
    let perm = [1usize, 0usize];
    for (flat_new, slot) in out.iter_mut().enumerate() {
        let mut remainder = flat_new;
        let mut flat_old = 0usize;
        for (new_axis, &ns) in new_strides.iter().enumerate() {
            let idx = remainder / ns;
            remainder %= ns;
            flat_old += idx * old_strides[perm[new_axis]];
        }
        *slot = values[flat_old];
    }
    out
}

fn bench_transpose(c: &mut Criterion) {
    if focused_identity_only() || focused_suffix_only() {
        return;
    }
    for &(r, cc) in &[(4096usize, 4096usize), (2048, 2048), (1024, 4096)] {
        let n = r * cc;
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let arr = UFuncArray::new(vec![r, cc], data.clone(), DType::F64).unwrap();
        // Correctness: production matches the old per-element gather bit-for-bit.
        let got = arr.transpose(None).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_transpose(&data, r, cc)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let mut group = c.benchmark_group(format!("transpose_{r}x{cc}"));
        group.bench_with_input(BenchmarkId::new("old_gather", n), &n, |b, _| {
            b.iter(|| black_box(old_transpose(black_box(&data), r, cc)))
        });
        group.bench_with_input(BenchmarkId::new("tiled_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.transpose(black_box(None)).unwrap()))
        });
        group.finish();
    }
}

/// Previous kernel for the batched (N-D) last-two-swap: per-element index
/// decomposition + strided gather over the whole flat array.
fn old_batched_t(values: &[f64], dims: &[usize]) -> Vec<f64> {
    let ndim = dims.len();
    let r = dims[ndim - 2];
    let c = dims[ndim - 1];
    let batch: usize = dims[..ndim - 2].iter().product();
    let plane = r * c;
    let mut out = vec![0.0f64; values.len()];
    // Mirror the generic transpose: out index (b, jo, io) <- in (b, io, jo).
    for b in 0..batch {
        let base = b * plane;
        for io in 0..r {
            let src_row = base + io * c;
            for jo in 0..c {
                out[base + jo * r + io] = values[src_row + jo];
            }
        }
    }
    out
}

fn bench_transpose_batched(c: &mut Criterion) {
    if focused_identity_only() || focused_suffix_only() {
        return;
    }
    // Batched matrix transpose (swapaxes(-1,-2) on a stack): the dominant N-D case.
    for dims in [
        vec![64usize, 512, 512],
        vec![256, 256, 256],
        vec![1024, 128, 128],
    ] {
        let n: usize = dims.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 - 3.0).collect();
        let arr = UFuncArray::new(dims.clone(), data.clone(), DType::F64).unwrap();
        let got = arr.matrix_transpose().unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_batched_t(&data, &dims)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("transpose_batched_{dims:?}"));
        group.bench_with_input(BenchmarkId::new("old_gather", n), &n, |b, _| {
            b.iter(|| black_box(old_batched_t(black_box(&data), &dims)))
        });
        group.bench_with_input(BenchmarkId::new("tiled_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.matrix_transpose().unwrap()))
        });
        group.finish();
    }
}

/// Previous general kernel: per-element coordinate decomposition (ndim divisions)
/// + strided gather, serial — the path arbitrary permutations used to take.
fn old_general(values: &[f64], dims: &[usize], perm: &[usize]) -> Vec<f64> {
    let ndim = dims.len();
    let new_shape: Vec<usize> = perm.iter().map(|&a| dims[a]).collect();
    let mut old_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        old_strides[d] = old_strides[d + 1] * dims[d + 1];
    }
    let mut new_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        new_strides[d] = new_strides[d + 1] * new_shape[d + 1];
    }
    let total: usize = dims.iter().product();
    let mut out = vec![0.0f64; total];
    for (flat_new, slot) in out.iter_mut().enumerate() {
        let mut rem = flat_new;
        let mut flat_old = 0usize;
        for (na, &ns) in new_strides.iter().enumerate() {
            let i = rem / ns;
            rem %= ns;
            flat_old += i * old_strides[perm[na]];
        }
        *slot = values[flat_old];
    }
    out
}

fn former_identity_transpose(values: &[f64], dims: &[usize]) -> (Vec<usize>, Vec<f64>) {
    let ndim = dims.len();
    let perm: Vec<usize> = (0..ndim).collect();
    let mut seen = vec![false; ndim];
    for &axis in &perm {
        seen[axis] = true;
    }
    assert!(seen.into_iter().all(|present| present));

    let new_shape: Vec<usize> = perm.iter().map(|&axis| dims[axis]).collect();
    let old_strides = c_strides(dims);
    let new_strides = c_strides(&new_shape);
    let src_step: Vec<usize> = (0..ndim).map(|axis| old_strides[perm[axis]]).collect();
    let mut new_values = vec![0.0; values.len()];
    const TRANSPOSE_CHUNK: usize = 1 << 14;

    for (chunk_index, chunk) in new_values.chunks_mut(TRANSPOSE_CHUNK).enumerate() {
        let first_output = chunk_index * TRANSPOSE_CHUNK;
        let mut coordinates = vec![0usize; ndim];
        let mut remainder = first_output;
        for axis in 0..ndim {
            coordinates[axis] = remainder / new_strides[axis];
            remainder %= new_strides[axis];
        }
        let mut source_offset: usize = (0..ndim)
            .map(|axis| coordinates[axis] * src_step[axis])
            .sum();
        for slot in chunk {
            *slot = values[source_offset];
            let mut axis = ndim;
            while axis > 0 {
                axis -= 1;
                coordinates[axis] += 1;
                source_offset += src_step[axis];
                if coordinates[axis] < new_shape[axis] {
                    break;
                }
                coordinates[axis] = 0;
                source_offset -= new_shape[axis] * src_step[axis];
            }
        }
    }

    (new_shape, new_values)
}

fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    strides
}

fn bench_transpose_identity(c: &mut Criterion) {
    if focused_suffix_only() {
        return;
    }
    let dims = vec![64usize, 32, 8];
    let identity: Vec<usize> = (0..dims.len()).collect();
    let count: usize = dims.iter().product();
    assert_eq!(count, 1 << 14);
    let data: Vec<f64> = (0..count)
        .map(|index| f64::from_bits(0x3ff0_0000_0000_0000 ^ index as u64))
        .collect();
    let array = UFuncArray::new(dims.clone(), data.clone(), DType::F64)
        .expect("identity-transpose benchmark input");
    let (former_shape, former_values) = former_identity_transpose(&data, &dims);
    let public = array
        .transpose(Some(&identity))
        .expect("public identity transpose");
    assert_eq!(public.shape(), former_shape);
    assert_eq!(
        public
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("transpose_identity");
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function("former_generic_odometer", |bench| {
        bench.iter(|| {
            black_box(former_identity_transpose(
                black_box(&data),
                black_box(&dims),
            ))
        })
    });
    group.bench_function("public_identity_path", |bench| {
        bench.iter(|| {
            black_box(
                array
                    .transpose(black_box(Some(&identity)))
                    .expect("public identity transpose"),
            )
        })
    });
    group.finish();
}

/// Chunked parallel odometer gather — a frozen copy of the production general
/// path (values, no sidecar) so the suffix-identity A/B isolates the
/// block-copy lever inside one binary.
fn former_suffix_odometer(values: &[f64], dims: &[usize], perm: &[usize]) -> (Vec<usize>, Vec<f64>) {
    let ndim = dims.len();
    let new_shape: Vec<usize> = perm.iter().map(|&a| dims[a]).collect();
    let old_strides = c_strides(dims);
    let new_strides = c_strides(&new_shape);
    let src_step: Vec<usize> = (0..ndim).map(|d| old_strides[perm[d]]).collect();
    let total: usize = dims.iter().product();
    const TRANSPOSE_CHUNK: usize = 1 << 14;
    const TRANSPOSE_PAR_MIN: usize = 1 << 15;
    let parallel = total >= TRANSPOSE_PAR_MIN && rayon::current_num_threads() >= 2;
    let mut new_values = vec![0.0f64; total];
    let fill_values = |(ci, chunk): (usize, &mut [f64])| {
        if chunk.is_empty() {
            return;
        }
        let f0 = ci * TRANSPOSE_CHUNK;
        let mut coord = vec![0usize; ndim];
        let mut rem = f0;
        for d in 0..ndim {
            coord[d] = rem / new_strides[d];
            rem %= new_strides[d];
        }
        let mut off: usize = (0..ndim).map(|d| coord[d] * src_step[d]).sum();
        for slot in chunk.iter_mut() {
            *slot = values[off];
            let mut d = ndim;
            while d > 0 {
                d -= 1;
                coord[d] += 1;
                off += src_step[d];
                if coord[d] < new_shape[d] {
                    break;
                }
                coord[d] = 0;
                off -= new_shape[d] * src_step[d];
            }
        }
    };
    if parallel {
        new_values
            .par_chunks_mut(TRANSPOSE_CHUNK)
            .enumerate()
            .for_each(fill_values);
    } else {
        new_values
            .chunks_mut(TRANSPOSE_CHUNK)
            .enumerate()
            .for_each(fill_values);
    }
    (new_shape, new_values)
}

fn bench_transpose_suffix(c: &mut Criterion) {
    if focused_identity_only() {
        return;
    }
    // Leading-axes permutations that keep the last axis in place (swapaxes(0,1)
    // on a stack and its 4-D sibling): the last-two-swap fast path does NOT
    // cover these, so they ride the generic gather. One cache-resident shape
    // and one DRAM-resident shape.
    let cases: &[(Vec<usize>, Vec<usize>)] = &[
        (vec![64, 64, 64], vec![1, 0, 2]),
        (vec![256, 256, 256], vec![1, 0, 2]),
        (vec![32, 32, 32, 16], vec![2, 0, 1, 3]),
    ];
    for (dims, perm) in cases {
        let n: usize = dims.iter().product();
        let data: Vec<f64> = (0..n)
            .map(|i| f64::from_bits(0x3ff0_0000_0000_0000 ^ (i as u64).wrapping_mul(0x9e37_79b9)))
            .collect();
        let arr = UFuncArray::new(dims.clone(), data.clone(), DType::F64).unwrap();
        let pu: Vec<usize> = perm.clone();
        let (former_shape, former_values) = former_suffix_odometer(&data, dims, perm);
        let got = arr.transpose(Some(&pu)).unwrap();
        assert_eq!(got.shape(), former_shape);
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            former_values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("transpose_suffix_{dims:?}_{perm:?}"));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("former_parallel_odometer", n),
            &n,
            |b, _| b.iter(|| black_box(former_suffix_odometer(black_box(&data), dims, perm))),
        );
        group.bench_with_input(BenchmarkId::new("public_path", n), &n, |b, _| {
            b.iter(|| black_box(arr.transpose(black_box(Some(&pu))).unwrap()))
        });
        group.finish();
    }
}

fn bench_transpose_general(c: &mut Criterion) {
    if focused_identity_only() || focused_suffix_only() {
        return;
    }
    // Arbitrary permutations (rotations / non-adjacent moveaxis), the case the
    // last-two-swap fast path does NOT cover.
    let cases: &[(Vec<usize>, Vec<usize>)] = &[
        (vec![256, 256, 256], vec![2, 0, 1]),
        (vec![256, 256, 256], vec![1, 2, 0]),
        (vec![64, 64, 64, 64], vec![3, 1, 0, 2]),
    ];
    for (dims, perm) in cases {
        let n: usize = dims.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.125 - 2.0).collect();
        let arr = UFuncArray::new(dims.clone(), data.clone(), DType::F64).unwrap();
        let pu: Vec<usize> = perm.clone();
        let got = arr.transpose(Some(&pu)).unwrap();
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_general(&data, dims, perm)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut group = c.benchmark_group(format!("transpose_general_{dims:?}_{perm:?}"));
        group.bench_with_input(BenchmarkId::new("old_gather", n), &n, |b, _| {
            b.iter(|| black_box(old_general(black_box(&data), dims, perm)))
        });
        group.bench_with_input(BenchmarkId::new("odometer_par", n), &n, |b, _| {
            b.iter(|| black_box(arr.transpose(Some(&pu)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_transpose_identity,
    bench_transpose,
    bench_transpose_batched,
    bench_transpose_general,
    bench_transpose_suffix
);
criterion_main!(benches);
