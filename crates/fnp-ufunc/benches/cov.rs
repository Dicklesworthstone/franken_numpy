use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;
use std::hint::black_box;

// np.cov(m): each row a variable, each column an observation. The slow regime
// vs numpy is few-variables / many-observations (nvars small, nobs large): the
// per-output-cell dot product runs a single ascending-k accumulator whose FMA
// latency chain is fully exposed, while numpy's BLAS dgemm keeps many cells in
// flight. Shapes below span that regime plus a large-nvars control.
fn make(nvars: usize, nobs: usize) -> UFuncArray {
    let mut v = vec![0.0f64; nvars * nobs];
    for (i, slot) in v.iter_mut().enumerate() {
        // deterministic, non-trivial values
        *slot = ((i * 1103515245 + 12345) % 1000) as f64 * 0.001 - 0.5;
    }
    UFuncArray::new(vec![nvars, nobs], v, DType::F64).unwrap()
}

fn bench(c: &mut Criterion) {
    let shapes = [(50usize, 5000usize), (96, 4000), (50, 1000), (200, 1000)];
    for &(nvars, nobs) in &shapes {
        let m = make(nvars, nobs);
        let _ = m.cov().unwrap();
        let mut g = c.benchmark_group(format!("cov_{nvars}x{nobs}"));
        g.sample_size(30);
        g.bench_with_input(
            BenchmarkId::new("cov", format!("{nvars}x{nobs}")),
            &m,
            |b, m| b.iter(|| black_box(black_box(m).cov().unwrap())),
        );
        g.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
