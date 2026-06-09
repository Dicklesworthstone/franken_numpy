use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fnp_dtype::ArrayStorage;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

fn mk_i64(n: usize, seed: u64) -> (UFuncArray, Vec<i64>) {
    let mut s = seed;
    let v: Vec<i64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2_000_000) as i64
        })
        .collect();
    (
        UFuncArray::from_storage(vec![n], ArrayStorage::I64(v.clone())).unwrap(),
        v,
    )
}

// Serial reference replicating the old intersect1d i64 path (serial sort).
fn old_intersect(va: &[i64], vb: &[i64]) -> Vec<i64> {
    let mut a = va.to_vec();
    let mut b = vb.to_vec();
    a.sort_unstable();
    a.dedup();
    b.sort_unstable();
    b.dedup();
    let (mut i, mut j) = (0, 0);
    let mut r = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                r.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    r
}

fn bench(c: &mut Criterion) {
    for &n in &[1usize << 19, 1 << 21] {
        let (a, va) = mk_i64(n, 0x1234);
        let (b, vb) = mk_i64(n, 0x9876);
        let mut g = c.benchmark_group(format!("intersect1d_i64_n{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("old_serial", n), &n, |bb, _| {
            bb.iter(|| black_box(old_intersect(black_box(&va), black_box(&vb))))
        });
        g.bench_with_input(BenchmarkId::new("par_sort", n), &n, |bb, _| {
            bb.iter(|| black_box(a.intersect1d(black_box(&b))))
        });
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
