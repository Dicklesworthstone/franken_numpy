use crate::{DType, min_scalar_type};

#[test]
fn min_scalar_type_negative_zero_and_boundary_floats() {
    assert_eq!(min_scalar_type(-0.0), DType::F16);
    assert_eq!(min_scalar_type(0.0), DType::F16);
    assert_eq!(min_scalar_type(64_999.0), DType::F16);
    assert_eq!(min_scalar_type(-64_999.0), DType::F16);
    assert_eq!(min_scalar_type(65_000.0), DType::F32);
    assert_eq!(min_scalar_type(-65_000.0), DType::F32);
}
