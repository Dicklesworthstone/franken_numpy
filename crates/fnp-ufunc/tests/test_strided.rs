use fnp_dtype::DType;
use fnp_ufunc::{UFuncArray, UFuncError};

#[test]
fn test_strided_backwards() -> Result<(), UFuncError> {
    // Create [5, 6, 7, 8, 9], then slice to get a view starting at offset 4
    // so that stride -1 can traverse backwards: offsets 4, 3, 2 → values 9, 8, 7
    let a = UFuncArray::arange(0.0, 10.0, 1.0, DType::F64)?;
    let reversed = a.slice_axis(0, None, None, -1)?;
    let view = reversed.shared_view()?;
    // The reversed view sees [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] via negative stride
    let sub = view.slice_axis(0, None, Some(3), 1)?;
    // sub = [9, 8, 7] — 3 elements
    assert_eq!(sub.shape(), &[3]);
    assert_eq!(sub.item(&[0])?, 9.0);
    assert_eq!(sub.item(&[1])?, 8.0);
    assert_eq!(sub.item(&[2])?, 7.0);
    Ok(())
}
