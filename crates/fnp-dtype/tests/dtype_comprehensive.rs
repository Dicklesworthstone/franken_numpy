//! Comprehensive tests for fnp-dtype.
//!
//! This test suite covers:
//! - DType::parse() for all supported format strings
//! - DType::item_size() correctness
//! - Type predicates (is_integer, is_float, is_complex, is_numeric)
//! - Type promotion rules via promote()
//! - Sum reduction promotion via promote_for_sum_reduction()
//!
//! Motivation: fnp-dtype is a foundational crate with 0 test files despite
//! 3186 lines of source code. This addresses a critical test coverage gap.

use fnp_dtype::{DType, promote, promote_for_sum_reduction};

// ─────────────────────────────────────────────────────────────────────────────
// DType::parse() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_bool() {
    assert_eq!(DType::parse("bool"), Some(DType::Bool));
    assert_eq!(DType::parse("bool_"), Some(DType::Bool));
}

#[test]
fn parse_signed_integers() {
    assert_eq!(DType::parse("int8"), Some(DType::I8));
    assert_eq!(DType::parse("i1"), Some(DType::I8));
    assert_eq!(DType::parse("int16"), Some(DType::I16));
    assert_eq!(DType::parse("i2"), Some(DType::I16));
    assert_eq!(DType::parse("i16"), Some(DType::I16));
    assert_eq!(DType::parse("int32"), Some(DType::I32));
    assert_eq!(DType::parse("i4"), Some(DType::I32));
    assert_eq!(DType::parse("i32"), Some(DType::I32));
    assert_eq!(DType::parse("int64"), Some(DType::I64));
    assert_eq!(DType::parse("i8"), Some(DType::I64));
    assert_eq!(DType::parse("i64"), Some(DType::I64));
    assert_eq!(DType::parse("int"), Some(DType::I64));
    assert_eq!(DType::parse("int_"), Some(DType::I64));
    assert_eq!(DType::parse("long"), Some(DType::I64));
    assert_eq!(DType::parse("longlong"), Some(DType::I64));
}

#[test]
fn parse_unsigned_integers() {
    assert_eq!(DType::parse("uint8"), Some(DType::U8));
    assert_eq!(DType::parse("u1"), Some(DType::U8));
    assert_eq!(DType::parse("uint16"), Some(DType::U16));
    assert_eq!(DType::parse("u2"), Some(DType::U16));
    assert_eq!(DType::parse("u16"), Some(DType::U16));
    assert_eq!(DType::parse("uint32"), Some(DType::U32));
    assert_eq!(DType::parse("u4"), Some(DType::U32));
    assert_eq!(DType::parse("u32"), Some(DType::U32));
    assert_eq!(DType::parse("uint64"), Some(DType::U64));
    assert_eq!(DType::parse("u8"), Some(DType::U64));
    assert_eq!(DType::parse("u64"), Some(DType::U64));
    assert_eq!(DType::parse("uint"), Some(DType::U64));
}

#[test]
fn parse_floats() {
    assert_eq!(DType::parse("float16"), Some(DType::F16));
    assert_eq!(DType::parse("f2"), Some(DType::F16));
    assert_eq!(DType::parse("f16"), Some(DType::F16));
    assert_eq!(DType::parse("half"), Some(DType::F16));
    assert_eq!(DType::parse("float32"), Some(DType::F32));
    assert_eq!(DType::parse("f4"), Some(DType::F32));
    assert_eq!(DType::parse("f32"), Some(DType::F32));
    assert_eq!(DType::parse("single"), Some(DType::F32));
    assert_eq!(DType::parse("float64"), Some(DType::F64));
    assert_eq!(DType::parse("f8"), Some(DType::F64));
    assert_eq!(DType::parse("f64"), Some(DType::F64));
    assert_eq!(DType::parse("float"), Some(DType::F64));
    assert_eq!(DType::parse("double"), Some(DType::F64));
}

#[test]
fn parse_complex() {
    assert_eq!(DType::parse("complex64"), Some(DType::Complex64));
    assert_eq!(DType::parse("c8"), Some(DType::Complex64));
    assert_eq!(DType::parse("csingle"), Some(DType::Complex64));
    assert_eq!(DType::parse("complex128"), Some(DType::Complex128));
    assert_eq!(DType::parse("c16"), Some(DType::Complex128));
    assert_eq!(DType::parse("complex"), Some(DType::Complex128));
    assert_eq!(DType::parse("cdouble"), Some(DType::Complex128));
}

#[test]
fn parse_string_types() {
    assert_eq!(DType::parse("str"), Some(DType::Str));
    assert_eq!(DType::parse("str_"), Some(DType::Str));
    assert_eq!(DType::parse("unicode"), Some(DType::Str));
    assert_eq!(DType::parse("bytes"), Some(DType::Str));
    assert_eq!(DType::parse("bytes_"), Some(DType::Str));
    assert_eq!(DType::parse("U"), Some(DType::Str));
    assert_eq!(DType::parse("U10"), Some(DType::Str));
    assert_eq!(DType::parse("U256"), Some(DType::Str));
    assert_eq!(DType::parse("S10"), Some(DType::Str));
}

#[test]
fn parse_void_structured() {
    assert_eq!(DType::parse("void"), Some(DType::Structured));
    assert_eq!(DType::parse("V"), Some(DType::Structured));
    assert_eq!(DType::parse("|V10"), Some(DType::Structured));
    assert_eq!(DType::parse("<V16"), Some(DType::Structured));
    assert_eq!(DType::parse(">V8"), Some(DType::Structured));
    assert_eq!(DType::parse("void80"), Some(DType::Structured));
}

#[test]
fn parse_datetime_timedelta() {
    assert_eq!(DType::parse("datetime64"), Some(DType::DateTime64));
    assert_eq!(DType::parse("M8"), Some(DType::DateTime64));
    assert_eq!(DType::parse("datetime64[ns]"), Some(DType::DateTime64));
    assert_eq!(DType::parse("datetime64[us]"), Some(DType::DateTime64));
    assert_eq!(DType::parse("timedelta64"), Some(DType::TimeDelta64));
    assert_eq!(DType::parse("m8"), Some(DType::TimeDelta64));
    assert_eq!(DType::parse("timedelta64[s]"), Some(DType::TimeDelta64));
}

#[test]
fn parse_invalid_returns_none() {
    assert_eq!(DType::parse("invalid"), None);
    assert_eq!(DType::parse("float256"), None);
    assert_eq!(DType::parse(""), None);
    assert_eq!(DType::parse("i99"), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// DType::item_size() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn item_size_1_byte() {
    assert_eq!(DType::Bool.item_size(), 1);
    assert_eq!(DType::I8.item_size(), 1);
    assert_eq!(DType::U8.item_size(), 1);
    assert_eq!(DType::Str.item_size(), 1);
    assert_eq!(DType::Structured.item_size(), 1);
}

#[test]
fn item_size_2_bytes() {
    assert_eq!(DType::I16.item_size(), 2);
    assert_eq!(DType::U16.item_size(), 2);
    assert_eq!(DType::F16.item_size(), 2);
}

#[test]
fn item_size_4_bytes() {
    assert_eq!(DType::I32.item_size(), 4);
    assert_eq!(DType::U32.item_size(), 4);
    assert_eq!(DType::F32.item_size(), 4);
}

#[test]
fn item_size_8_bytes() {
    assert_eq!(DType::I64.item_size(), 8);
    assert_eq!(DType::U64.item_size(), 8);
    assert_eq!(DType::F64.item_size(), 8);
    assert_eq!(DType::Complex64.item_size(), 8);
    assert_eq!(DType::DateTime64.item_size(), 8);
    assert_eq!(DType::TimeDelta64.item_size(), 8);
}

#[test]
fn item_size_16_bytes() {
    assert_eq!(DType::Complex128.item_size(), 16);
}

// ─────────────────────────────────────────────────────────────────────────────
// Type predicate tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn is_integer_true() {
    for dt in [
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        assert!(dt.is_integer(), "{:?} should be integer", dt);
    }
}

#[test]
fn is_integer_false() {
    for dt in [
        DType::Bool,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
        DType::Str,
        DType::DateTime64,
        DType::TimeDelta64,
        DType::Structured,
    ] {
        assert!(!dt.is_integer(), "{:?} should not be integer", dt);
    }
}

#[test]
fn is_float_true() {
    for dt in [
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        assert!(dt.is_float(), "{:?} should be float", dt);
    }
}

#[test]
fn is_float_false() {
    for dt in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::Str,
        DType::DateTime64,
        DType::TimeDelta64,
        DType::Structured,
    ] {
        assert!(!dt.is_float(), "{:?} should not be float", dt);
    }
}

#[test]
fn is_complex_true() {
    assert!(DType::Complex64.is_complex());
    assert!(DType::Complex128.is_complex());
}

#[test]
fn is_complex_false() {
    for dt in [
        DType::Bool,
        DType::I8,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Str,
    ] {
        assert!(!dt.is_complex(), "{:?} should not be complex", dt);
    }
}

#[test]
fn is_numeric_true() {
    for dt in [
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        assert!(dt.is_numeric(), "{:?} should be numeric", dt);
    }
}

#[test]
fn is_numeric_false() {
    for dt in [
        DType::Bool,
        DType::Str,
        DType::DateTime64,
        DType::TimeDelta64,
        DType::Structured,
    ] {
        assert!(!dt.is_numeric(), "{:?} should not be numeric", dt);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type promotion tests — promote()
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn promote_bool_is_identity() {
    for dt in [
        DType::Bool,
        DType::I8,
        DType::I64,
        DType::U8,
        DType::U64,
        DType::F32,
        DType::F64,
        DType::Complex128,
    ] {
        assert_eq!(promote(DType::Bool, dt), dt);
        assert_eq!(promote(dt, DType::Bool), dt);
    }
}

#[test]
fn promote_signed_signed_picks_wider() {
    assert_eq!(promote(DType::I8, DType::I8), DType::I8);
    assert_eq!(promote(DType::I8, DType::I16), DType::I16);
    assert_eq!(promote(DType::I16, DType::I32), DType::I32);
    assert_eq!(promote(DType::I32, DType::I64), DType::I64);
    assert_eq!(promote(DType::I8, DType::I64), DType::I64);
}

#[test]
fn promote_unsigned_unsigned_picks_wider() {
    assert_eq!(promote(DType::U8, DType::U8), DType::U8);
    assert_eq!(promote(DType::U8, DType::U16), DType::U16);
    assert_eq!(promote(DType::U16, DType::U32), DType::U32);
    assert_eq!(promote(DType::U32, DType::U64), DType::U64);
}

#[test]
fn promote_signed_unsigned_cross() {
    assert_eq!(promote(DType::U8, DType::I8), DType::I16);
    assert_eq!(promote(DType::U16, DType::I16), DType::I32);
    assert_eq!(promote(DType::U32, DType::I32), DType::I64);
    assert_eq!(promote(DType::U64, DType::I64), DType::F64);
}

#[test]
fn promote_float_float_picks_wider() {
    assert_eq!(promote(DType::F16, DType::F16), DType::F16);
    assert_eq!(promote(DType::F16, DType::F32), DType::F32);
    assert_eq!(promote(DType::F32, DType::F64), DType::F64);
    assert_eq!(promote(DType::F16, DType::F64), DType::F64);
}

#[test]
fn promote_f16_with_integers() {
    assert_eq!(promote(DType::F16, DType::I8), DType::F16);
    assert_eq!(promote(DType::F16, DType::U8), DType::F16);
    assert_eq!(promote(DType::F16, DType::I16), DType::F32);
    assert_eq!(promote(DType::F16, DType::I32), DType::F64);
    assert_eq!(promote(DType::F16, DType::I64), DType::F64);
}

#[test]
fn promote_f32_with_integers() {
    assert_eq!(promote(DType::F32, DType::I8), DType::F32);
    assert_eq!(promote(DType::F32, DType::I16), DType::F32);
    assert_eq!(promote(DType::F32, DType::I32), DType::F64);
    assert_eq!(promote(DType::F32, DType::I64), DType::F64);
}

#[test]
fn promote_f64_with_integers() {
    assert_eq!(promote(DType::F64, DType::I8), DType::F64);
    assert_eq!(promote(DType::F64, DType::I64), DType::F64);
    assert_eq!(promote(DType::F64, DType::U64), DType::F64);
}

#[test]
fn promote_complex_with_floats() {
    assert_eq!(promote(DType::Complex64, DType::F32), DType::Complex64);
    assert_eq!(promote(DType::Complex64, DType::F64), DType::Complex128);
    assert_eq!(promote(DType::Complex128, DType::F64), DType::Complex128);
}

#[test]
fn promote_complex_with_integers() {
    assert_eq!(promote(DType::Complex64, DType::I8), DType::Complex64);
    assert_eq!(promote(DType::Complex64, DType::I16), DType::Complex64);
    assert_eq!(promote(DType::Complex64, DType::I32), DType::Complex128);
    assert_eq!(promote(DType::Complex128, DType::I64), DType::Complex128);
}

#[test]
fn promote_complex_complex() {
    assert_eq!(
        promote(DType::Complex64, DType::Complex64),
        DType::Complex64
    );
    assert_eq!(
        promote(DType::Complex64, DType::Complex128),
        DType::Complex128
    );
    assert_eq!(
        promote(DType::Complex128, DType::Complex128),
        DType::Complex128
    );
}

#[test]
fn promote_is_commutative() {
    let types = [
        DType::I8,
        DType::I32,
        DType::U8,
        DType::U32,
        DType::F32,
        DType::F64,
        DType::Complex64,
    ];
    for &a in &types {
        for &b in &types {
            assert_eq!(
                promote(a, b),
                promote(b, a),
                "promote({:?}, {:?}) != promote({:?}, {:?})",
                a,
                b,
                b,
                a
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sum reduction promotion tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sum_reduction_widens_small_signed() {
    assert_eq!(promote_for_sum_reduction(DType::Bool), DType::I64);
    assert_eq!(promote_for_sum_reduction(DType::I8), DType::I64);
    assert_eq!(promote_for_sum_reduction(DType::I16), DType::I64);
    assert_eq!(promote_for_sum_reduction(DType::I32), DType::I64);
}

#[test]
fn sum_reduction_widens_small_unsigned() {
    assert_eq!(promote_for_sum_reduction(DType::U8), DType::U64);
    assert_eq!(promote_for_sum_reduction(DType::U16), DType::U64);
    assert_eq!(promote_for_sum_reduction(DType::U32), DType::U64);
}

#[test]
fn sum_reduction_preserves_64bit() {
    assert_eq!(promote_for_sum_reduction(DType::I64), DType::I64);
    assert_eq!(promote_for_sum_reduction(DType::U64), DType::U64);
}

#[test]
fn sum_reduction_preserves_floats() {
    assert_eq!(promote_for_sum_reduction(DType::F16), DType::F16);
    assert_eq!(promote_for_sum_reduction(DType::F32), DType::F32);
    assert_eq!(promote_for_sum_reduction(DType::F64), DType::F64);
}

#[test]
fn sum_reduction_preserves_complex() {
    assert_eq!(
        promote_for_sum_reduction(DType::Complex64),
        DType::Complex64
    );
    assert_eq!(
        promote_for_sum_reduction(DType::Complex128),
        DType::Complex128
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DType::name() tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn name_returns_expected_strings() {
    assert_eq!(DType::Bool.name(), "bool");
    assert_eq!(DType::I8.name(), "int8");
    assert_eq!(DType::I64.name(), "i64");
    assert_eq!(DType::U8.name(), "uint8");
    assert_eq!(DType::F32.name(), "f32");
    assert_eq!(DType::F64.name(), "f64");
    assert_eq!(DType::Complex64.name(), "complex64");
    assert_eq!(DType::Complex128.name(), "complex128");
    assert_eq!(DType::Str.name(), "str");
    assert_eq!(DType::DateTime64.name(), "datetime64");
    assert_eq!(DType::TimeDelta64.name(), "timedelta64");
    assert_eq!(DType::Structured.name(), "void");
}
