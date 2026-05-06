//! Golden tests for fnp-dtype core operations.
//!
//! Tests DType parsing, name serialization, item sizes, type promotion,
//! and casting rules against known-good outputs.
//!
//! Finding: fnp-dtype (1,700+ LOC) had ZERO golden tests despite containing
//! the foundational type system for the entire FrankenNumPy stack.

use fnp_dtype::{DType, can_cast, common_type, min_scalar_type, result_type};

// ─────────────────────────────────────────────────────────────────────────────
// DType::parse golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_parse_bool() {
    assert_eq!(DType::parse("bool"), Some(DType::Bool));
}

#[test]
fn golden_parse_int8_variants() {
    assert_eq!(DType::parse("i1"), Some(DType::I8));
    assert_eq!(DType::parse("int8"), Some(DType::I8));
}

#[test]
fn golden_parse_int16_variants() {
    assert_eq!(DType::parse("i2"), Some(DType::I16));
    assert_eq!(DType::parse("i16"), Some(DType::I16));
    assert_eq!(DType::parse("int16"), Some(DType::I16));
}

#[test]
fn golden_parse_int32_variants() {
    assert_eq!(DType::parse("i4"), Some(DType::I32));
    assert_eq!(DType::parse("i32"), Some(DType::I32));
    assert_eq!(DType::parse("int32"), Some(DType::I32));
}

#[test]
fn golden_parse_int64_variants() {
    assert_eq!(DType::parse("i8"), Some(DType::I64));
    assert_eq!(DType::parse("i64"), Some(DType::I64));
    assert_eq!(DType::parse("int64"), Some(DType::I64));
}

#[test]
fn golden_parse_uint_variants() {
    assert_eq!(DType::parse("u1"), Some(DType::U8));
    assert_eq!(DType::parse("uint8"), Some(DType::U8));
    assert_eq!(DType::parse("u2"), Some(DType::U16));
    assert_eq!(DType::parse("u4"), Some(DType::U32));
    assert_eq!(DType::parse("u8"), Some(DType::U64));
}

#[test]
fn golden_parse_float_variants() {
    assert_eq!(DType::parse("f2"), Some(DType::F16));
    assert_eq!(DType::parse("f16"), Some(DType::F16));
    assert_eq!(DType::parse("float16"), Some(DType::F16));
    assert_eq!(DType::parse("f4"), Some(DType::F32));
    assert_eq!(DType::parse("f32"), Some(DType::F32));
    assert_eq!(DType::parse("float32"), Some(DType::F32));
    assert_eq!(DType::parse("f8"), Some(DType::F64));
    assert_eq!(DType::parse("f64"), Some(DType::F64));
    assert_eq!(DType::parse("float64"), Some(DType::F64));
}

#[test]
fn golden_parse_complex() {
    assert_eq!(DType::parse("c8"), Some(DType::Complex64));
    assert_eq!(DType::parse("complex64"), Some(DType::Complex64));
    assert_eq!(DType::parse("c16"), Some(DType::Complex128));
    assert_eq!(DType::parse("complex128"), Some(DType::Complex128));
}

#[test]
fn golden_parse_datetime() {
    assert_eq!(DType::parse("datetime64"), Some(DType::DateTime64));
    assert_eq!(DType::parse("timedelta64"), Some(DType::TimeDelta64));
}

#[test]
fn golden_parse_invalid() {
    assert_eq!(DType::parse("invalid"), None);
    assert_eq!(DType::parse(""), None);
    assert_eq!(DType::parse("INT64"), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// DType::name golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_name_integers() {
    assert_eq!(DType::Bool.name(), "bool");
    assert_eq!(DType::I8.name(), "int8");
    assert_eq!(DType::I16.name(), "i16");
    assert_eq!(DType::I32.name(), "i32");
    assert_eq!(DType::I64.name(), "i64");
    assert_eq!(DType::U8.name(), "uint8");
    assert_eq!(DType::U16.name(), "u16");
    assert_eq!(DType::U32.name(), "u32");
    assert_eq!(DType::U64.name(), "u64");
}

#[test]
fn golden_name_floats() {
    assert_eq!(DType::F16.name(), "f16");
    assert_eq!(DType::F32.name(), "f32");
    assert_eq!(DType::F64.name(), "f64");
}

#[test]
fn golden_name_complex() {
    assert_eq!(DType::Complex64.name(), "complex64");
    assert_eq!(DType::Complex128.name(), "complex128");
}

#[test]
fn golden_name_temporal() {
    assert_eq!(DType::DateTime64.name(), "datetime64");
    assert_eq!(DType::TimeDelta64.name(), "timedelta64");
}

// ─────────────────────────────────────────────────────────────────────────────
// DType::item_size golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_item_size_1byte() {
    assert_eq!(DType::Bool.item_size(), 1);
    assert_eq!(DType::I8.item_size(), 1);
    assert_eq!(DType::U8.item_size(), 1);
}

#[test]
fn golden_item_size_2byte() {
    assert_eq!(DType::I16.item_size(), 2);
    assert_eq!(DType::U16.item_size(), 2);
    assert_eq!(DType::F16.item_size(), 2);
}

#[test]
fn golden_item_size_4byte() {
    assert_eq!(DType::I32.item_size(), 4);
    assert_eq!(DType::U32.item_size(), 4);
    assert_eq!(DType::F32.item_size(), 4);
}

#[test]
fn golden_item_size_8byte() {
    assert_eq!(DType::I64.item_size(), 8);
    assert_eq!(DType::U64.item_size(), 8);
    assert_eq!(DType::F64.item_size(), 8);
    assert_eq!(DType::Complex64.item_size(), 8);
    assert_eq!(DType::DateTime64.item_size(), 8);
    assert_eq!(DType::TimeDelta64.item_size(), 8);
}

#[test]
fn golden_item_size_16byte() {
    assert_eq!(DType::Complex128.item_size(), 16);
}

// ─────────────────────────────────────────────────────────────────────────────
// result_type golden tests (type promotion)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_result_type_same() {
    assert_eq!(result_type(&[DType::F64, DType::F64]), DType::F64);
    assert_eq!(result_type(&[DType::I32, DType::I32]), DType::I32);
}

#[test]
fn golden_result_type_int_promotion() {
    assert_eq!(result_type(&[DType::I8, DType::I16]), DType::I16);
    assert_eq!(result_type(&[DType::I16, DType::I32]), DType::I32);
    assert_eq!(result_type(&[DType::I32, DType::I64]), DType::I64);
}

#[test]
fn golden_result_type_int_to_float() {
    assert_eq!(result_type(&[DType::I32, DType::F32]), DType::F64);
    assert_eq!(result_type(&[DType::I64, DType::F32]), DType::F64);
}

#[test]
fn golden_result_type_float_promotion() {
    assert_eq!(result_type(&[DType::F16, DType::F32]), DType::F32);
    assert_eq!(result_type(&[DType::F32, DType::F64]), DType::F64);
}

#[test]
fn golden_result_type_complex() {
    assert_eq!(
        result_type(&[DType::F32, DType::Complex64]),
        DType::Complex64
    );
    assert_eq!(
        result_type(&[DType::F64, DType::Complex64]),
        DType::Complex128
    );
    assert_eq!(
        result_type(&[DType::Complex64, DType::Complex128]),
        DType::Complex128
    );
}

#[test]
fn golden_result_type_bool_promotes() {
    assert_eq!(result_type(&[DType::Bool, DType::I32]), DType::I32);
    assert_eq!(result_type(&[DType::Bool, DType::F64]), DType::F64);
}

// ─────────────────────────────────────────────────────────────────────────────
// can_cast golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_can_cast_same_type() {
    assert!(can_cast(DType::F64, DType::F64, "no"));
    assert!(can_cast(DType::I32, DType::I32, "no"));
}

#[test]
fn golden_can_cast_safe_int_widening() {
    assert!(can_cast(DType::I8, DType::I16, "safe"));
    assert!(can_cast(DType::I16, DType::I32, "safe"));
    assert!(can_cast(DType::I32, DType::I64, "safe"));
}

#[test]
fn golden_can_cast_safe_float_widening() {
    assert!(can_cast(DType::F16, DType::F32, "safe"));
    assert!(can_cast(DType::F32, DType::F64, "safe"));
}

#[test]
fn golden_can_cast_int_to_float_safe() {
    // I64 -> F64 is safe per can_cast_lossless
    assert!(can_cast(DType::I64, DType::F64, "safe"));
    assert!(can_cast(DType::I32, DType::F64, "safe"));
    assert!(can_cast(DType::I64, DType::F64, "unsafe"));
}

#[test]
fn golden_can_cast_same_kind() {
    assert!(can_cast(DType::I64, DType::I32, "same_kind"));
    assert!(can_cast(DType::F64, DType::F32, "same_kind"));
}

#[test]
fn golden_can_cast_no_rejects_different() {
    assert!(!can_cast(DType::I64, DType::F64, "no"));
    assert!(!can_cast(DType::F32, DType::I32, "no"));
}

// ─────────────────────────────────────────────────────────────────────────────
// min_scalar_type golden tests
// NumPy's float scalar path uses magnitude thresholds, always returns float.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_min_scalar_type_small_magnitude() {
    // Values in (-65000, 65000) map to F16
    assert_eq!(min_scalar_type(0.0), DType::F16);
    assert_eq!(min_scalar_type(127.0), DType::F16);
    assert_eq!(min_scalar_type(255.0), DType::F16);
    assert_eq!(min_scalar_type(-129.0), DType::F16);
    assert_eq!(min_scalar_type(0.5), DType::F16);
}

#[test]
fn golden_min_scalar_type_medium_magnitude() {
    // Values >= 65000 but < 3.4e38 map to F32
    assert_eq!(min_scalar_type(65_000.0), DType::F32);
    assert_eq!(min_scalar_type(-65_000.0), DType::F32);
    assert_eq!(min_scalar_type(1.0e10), DType::F32);
}

#[test]
fn golden_min_scalar_type_large_magnitude() {
    // Values >= 3.4e38 map to F64
    assert_eq!(min_scalar_type(1.0e39), DType::F64);
}

#[test]
fn golden_min_scalar_type_special() {
    // NaN and Inf use F16 (within threshold check returns early)
    assert_eq!(min_scalar_type(f64::NAN), DType::F16);
    assert_eq!(min_scalar_type(f64::INFINITY), DType::F16);
}

// ─────────────────────────────────────────────────────────────────────────────
// common_type golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_common_type_same() {
    assert_eq!(common_type(&[DType::F64, DType::F64]), DType::F64);
}

#[test]
fn golden_common_type_mixed_float() {
    assert_eq!(common_type(&[DType::F32, DType::F64]), DType::F64);
}

#[test]
fn golden_common_type_int_and_float() {
    assert_eq!(common_type(&[DType::I32, DType::F32]), DType::F64);
}

#[test]
fn golden_common_type_complex() {
    assert_eq!(
        common_type(&[DType::F64, DType::Complex128]),
        DType::Complex128
    );
}
