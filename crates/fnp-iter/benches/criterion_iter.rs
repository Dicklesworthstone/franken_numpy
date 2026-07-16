//! Focused performance benchmarks for iterator planning and chunk emission.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fnp_iter::{Nditer, NditerOptions, NditerOrder, NditerPlan, NditerStep};
use std::hint::black_box;
use std::time::Duration;

fn former_f_order_external_chunk(plan: NditerPlan) -> NditerStep {
    let iterindex = 0usize;
    let end = iterindex
        .checked_add(plan.inner_loop_len())
        .expect("former chunk end must fit");
    let linear_indices = (iterindex..end)
        .map(|index| {
            let multi_index = plan
                .linear_index_to_multi_index(index)
                .expect("former multi-index conversion must succeed");
            multi_index
                .iter()
                .enumerate()
                .try_fold(0usize, |linear, (axis, &coordinate)| {
                    linear
                        .checked_mul(plan.shape()[axis])
                        .and_then(|value| value.checked_add(coordinate))
                })
                .expect("former operand index must fit")
        })
        .collect();

    NditerStep {
        iterindex,
        multi_index: plan
            .linear_index_to_multi_index(iterindex)
            .expect("former chunk start must resolve"),
        linear_indices,
    }
}

fn public_f_order_external_chunk(plan: NditerPlan) -> NditerStep {
    Nditer::from_plan(plan)
        .next()
        .expect("F-order external-loop chunk must exist")
}

fn bench_nditer_f_external_chunk(c: &mut Criterion) {
    let plan = NditerPlan::new(
        vec![65_536, 2, 2, 2],
        8,
        NditerOptions {
            order: NditerOrder::F,
            external_loop: true,
        },
    )
    .expect("F-order external-loop plan must build");
    assert_eq!(
        public_f_order_external_chunk(plan.clone()),
        former_f_order_external_chunk(plan.clone())
    );

    let mut group = c.benchmark_group("nditer_f_external_chunk");
    group.throughput(Throughput::Elements(plan.inner_loop_len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_index_round_trip", |bench| {
        bench.iter(|| black_box(former_f_order_external_chunk(black_box(plan.clone()))))
    });
    group.bench_function("public_chunk_path", |bench| {
        bench.iter(|| black_box(public_f_order_external_chunk(black_box(plan.clone()))))
    });
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default().configure_from_args()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_nditer_f_external_chunk
}
criterion_main!(benches);
