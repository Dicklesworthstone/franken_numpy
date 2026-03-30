use fnp_ndarray::{MemoryOrder, NdLayout};

#[test]
fn test_strided_flaw() {
    let layout = NdLayout::contiguous(vec![5], 8, MemoryOrder::C).unwrap();
    // Base layout: shape=[5], strides=[8]. Span is 0 to 32. Total 40 bytes.
    // Let's create a view with shape=[5], strides=[-8].
    // Span is -32 to 0. Total 40 bytes.
    let view = layout.as_strided(vec![5], vec![-8]);
    println!("{:?}", view);
    assert!(view.is_ok(), "View should be ok according to length check");
}
