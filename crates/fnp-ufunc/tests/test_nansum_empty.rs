use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;

#[test]
fn test_nansum_empty() {
    let arr = UFuncArray::zeros(vec![0, 5], DType::F64).unwrap();
    let res = arr.nansum(Some(0), false).unwrap();
    assert_eq!(res.shape(), &[5]);
    assert_eq!(res.values(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
}
