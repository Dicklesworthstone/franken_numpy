#![forbid(unsafe_code)]

/// Canonical NumPy-like dtypes for FrankenNumPy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Complex64,
    Complex128,
    Str,
    DateTime64,
    TimeDelta64,
}

impl DType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
            Self::Str => "str",
            Self::DateTime64 => "datetime64",
            Self::TimeDelta64 => "timedelta64",
        }
    }

    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64
            | Self::U64
            | Self::F64
            | Self::Complex64
            | Self::DateTime64
            | Self::TimeDelta64 => 8,
            Self::Complex128 => 16,
            Self::Str => 0, // variable-length
        }
    }

    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "i8" | "int8" => Some(Self::I8),
            "i16" | "int16" => Some(Self::I16),
            "i32" | "int32" => Some(Self::I32),
            "i64" | "int64" => Some(Self::I64),
            "u8" | "uint8" => Some(Self::U8),
            "u16" | "uint16" => Some(Self::U16),
            "u32" | "uint32" => Some(Self::U32),
            "u64" | "uint64" => Some(Self::U64),
            "f32" | "float32" => Some(Self::F32),
            "f64" | "float64" => Some(Self::F64),
            "c8" | "complex64" => Some(Self::Complex64),
            "c16" | "complex128" => Some(Self::Complex128),
            "str" | "U" => Some(Self::Str),
            "datetime64" | "M8" => Some(Self::DateTime64),
            "timedelta64" | "m8" => Some(Self::TimeDelta64),
            _ => {
                // Handle string dtype with length: U10, U32, etc.
                if name.starts_with('U') && name[1..].parse::<usize>().is_ok() {
                    return Some(Self::Str);
                }
                // Handle bytes dtype: S10, S32, etc.
                if name.starts_with('S') && name[1..].parse::<usize>().is_ok() {
                    return Some(Self::Str);
                }
                // Handle datetime with unit: datetime64[ns], datetime64[us], etc.
                if name.starts_with("datetime64") {
                    return Some(Self::DateTime64);
                }
                if name.starts_with("timedelta64") {
                    return Some(Self::TimeDelta64);
                }
                None
            }
        }
    }

    /// Returns `true` if this is a signed or unsigned integer type (not Bool).
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
        )
    }

    /// Returns `true` if this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Returns `true` if this is a complex floating-point type.
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex64 | Self::Complex128)
    }

    /// Returns `true` if this is a numeric type (integer, float, or complex).
    #[must_use]
    pub const fn is_numeric(self) -> bool {
        self.is_integer() || self.is_float() || self.is_complex()
    }
}

/// Deterministic promotion table following NumPy semantics.
///
/// Rules:
/// - `promote(Bool, X) = X` for any X (Bool is the weakest type)
/// - Same kind: pick the wider type
/// - Signed + unsigned: smallest signed that fits both ranges, or F64 for U64+signed
/// - Any integer + any float: F64 (because F32 cannot represent all I32/I64 values)
/// - Float + Float: wider float
#[must_use]
pub const fn promote(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    // Bool is identity element for promotion
    match (lhs, rhs) {
        (Bool, x) | (x, Bool) => x,

        // Signed-signed: pick wider
        (I8, I8) => I8,
        (I8, I16) | (I16, I8) | (I16, I16) => I16,
        (I8, I32) | (I16, I32) | (I32, I8) | (I32, I16) | (I32, I32) => I32,
        (I8, I64) | (I16, I64) | (I32, I64) | (I64, I8) | (I64, I16) | (I64, I32) | (I64, I64) => {
            I64
        }

        // Unsigned-unsigned: pick wider
        (U8, U8) => U8,
        (U8, U16) | (U16, U8) | (U16, U16) => U16,
        (U8, U32) | (U16, U32) | (U32, U8) | (U32, U16) | (U32, U32) => U32,
        (U8, U64) | (U16, U64) | (U32, U64) | (U64, U8) | (U64, U16) | (U64, U32) | (U64, U64) => {
            U64
        }

        // Signed-unsigned cross: smallest signed type that fits both ranges
        // U8 (0..255) + I8 (-128..127) -> I16 (covers both)
        (U8, I8) | (I8, U8) => I16,
        // U8 + I16/I32/I64 -> I16/I32/I64 (signed is already big enough)
        (U8, I16) | (I16, U8) => I16,
        (U8, I32) | (I32, U8) => I32,
        (U8, I64) | (I64, U8) => I64,
        // U16 (0..65535) + I8/I16 -> I32
        (U16, I8) | (I8, U16) | (U16, I16) | (I16, U16) => I32,
        (U16, I32) | (I32, U16) => I32,
        (U16, I64) | (I64, U16) => I64,
        // U32 (0..4B) + I8/I16/I32 -> I64
        (U32, I8) | (I8, U32) | (U32, I16) | (I16, U32) | (U32, I32) | (I32, U32) => I64,
        (U32, I64) | (I64, U32) => I64,
        // U64 + any signed -> F64 (no integer type can hold both U64 max and negative values)
        (U64, I8)
        | (I8, U64)
        | (U64, I16)
        | (I16, U64)
        | (U64, I32)
        | (I32, U64)
        | (U64, I64)
        | (I64, U64) => F64,

        // Small integers + F32 -> F32 (F32 has 24-bit mantissa, enough for 8/16-bit ints)
        (F32, I8 | I16 | U8 | U16) | (I8 | I16 | U8 | U16, F32) => F32,
        // Larger integers + F32 -> F64 (F32 can't exactly represent all I32/I64/U32/U64 values)
        (F32, I32 | I64 | U32 | U64) | (I32 | I64 | U32 | U64, F32) => F64,
        // Float-float: pick wider
        (F32, F32) => F32,

        // Complex promotion: complex absorbs float and integer
        (Complex64, Complex64) => Complex64,
        (Complex128, Complex128) => Complex128,
        (Complex64, Complex128) | (Complex128, Complex64) => Complex128,
        // Float + Complex: promote to matching complex
        (F32, Complex64) | (Complex64, F32) => Complex64,
        (F64, Complex64) | (Complex64, F64) | (F32, Complex128) | (Complex128, F32) => Complex128,
        (F64, Complex128) | (Complex128, F64) => Complex128,
        // Small integers + Complex64 -> Complex64 (mirrors F32 mantissa rule)
        // Note: Bool is already handled by the (Bool, x) | (x, Bool) => x rule above
        (Complex64, I8 | I16 | U8 | U16) | (I8 | I16 | U8 | U16, Complex64) => Complex64,
        // Larger integers + Complex64 -> Complex128 (mirrors F32->F64 rule)
        (Complex64, I32 | I64 | U32 | U64) | (I32 | I64 | U32 | U64, Complex64) => Complex128,
        // Any integer + Complex128 -> Complex128
        (Complex128, _) | (_, Complex128) => Complex128,

        // DateTime and TimeDelta: don't promote with numeric types
        (DateTime64, DateTime64) => DateTime64,
        (TimeDelta64, TimeDelta64) => TimeDelta64,
        (DateTime64, TimeDelta64) | (TimeDelta64, DateTime64) => DateTime64,

        // Str: stays Str
        (Str, Str) => Str,

        // Any remaining combinations with non-numeric types: F64 as fallback
        _ => F64,
    }
}

/// NumPy-compatible dtype promotion for sum (and similar accumulating)
/// reductions. Small integer and boolean inputs are widened to prevent
/// overflow, matching `numpy.add.reduce` behaviour.
#[must_use]
pub const fn promote_for_sum_reduction(dt: DType) -> DType {
    match dt {
        DType::Bool | DType::I8 | DType::I16 | DType::I32 => DType::I64,
        DType::U8 | DType::U16 | DType::U32 => DType::U64,
        DType::U64 => DType::U64,
        DType::I64 => DType::I64,
        DType::F32 => DType::F32,
        DType::F64 => DType::F64,
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        DType::Str | DType::DateTime64 | DType::TimeDelta64 => dt,
    }
}

/// NumPy-compatible dtype promotion for mean reductions.
/// Integer and boolean inputs are widened to float64, matching
/// `numpy.mean` behaviour.
#[must_use]
pub const fn promote_for_mean_reduction(dt: DType) -> DType {
    #[allow(clippy::match_same_arms)]
    match dt {
        DType::Bool
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64 => DType::F64,
        DType::F32 => DType::F32,
        DType::F64 => DType::F64,
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        DType::Str | DType::DateTime64 | DType::TimeDelta64 => dt,
    }
}

/// Returns `true` if `src` can be cast to `dst` without information loss.
/// Follows NumPy's safe-cast rules.
#[must_use]
pub const fn can_cast_lossless(src: DType, dst: DType) -> bool {
    use DType::*;

    matches!(
        (src, dst),
        // Bool can cast to anything numeric
        (Bool, Bool | I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F32 | F64 | Complex64 | Complex128)
        // Signed integers: can cast to wider signed or to F64
        | (I8, I8 | I16 | I32 | I64 | F32 | F64 | Complex64 | Complex128)
        | (I16, I16 | I32 | I64 | F32 | F64 | Complex64 | Complex128)
        | (I32, I32 | I64 | F64 | Complex128)
        | (I64, I64 | F64 | Complex128)
        // Unsigned integers: can cast to wider unsigned, wider signed, or F64
        | (U8, U8 | U16 | U32 | U64 | I16 | I32 | I64 | F32 | F64 | Complex64 | Complex128)
        | (U16, U16 | U32 | U64 | I32 | I64 | F32 | F64 | Complex64 | Complex128)
        | (U32, U32 | U64 | I64 | F64 | Complex128)
        | (U64, U64 | F64 | Complex128)
        // Floats: can cast to wider float or matching complex
        | (F32, F32 | F64 | Complex64 | Complex128)
        | (F64, F64 | Complex128)
        // Complex: can cast to wider complex
        | (Complex64, Complex64 | Complex128)
        | (Complex128, Complex128)
        // Non-numeric: only self-cast
        | (Str, Str)
        | (DateTime64, DateTime64)
        | (TimeDelta64, TimeDelta64)
    )
}

/// Determines the result dtype from promotion across a list of dtypes
/// (np.result_type).
#[must_use]
pub fn result_type(dtypes: &[DType]) -> DType {
    dtypes.iter().copied().fold(DType::Bool, promote)
}

/// Check whether a cast is allowed under the given casting rule
/// (np.can_cast).
///
/// Rules:
/// - `"no"` / `"equiv"`: types must be identical
/// - `"safe"`: lossless cast only
/// - `"same_kind"`: within kind or to a higher kind (bool < int < float)
/// - `"unsafe"`: any cast allowed
#[must_use]
pub fn can_cast(from: DType, to: DType, casting: &str) -> bool {
    match casting {
        "no" | "equiv" => from == to,
        "safe" => can_cast_lossless(from, to),
        "same_kind" => {
            if from == to {
                return true;
            }
            if from == DType::Bool {
                return true;
            }
            if from.is_integer() && to.is_integer() {
                return true;
            }
            if from.is_float() && to.is_float() {
                return true;
            }
            if from.is_complex() && to.is_complex() {
                return true;
            }
            if from.is_float() && to.is_complex() {
                return true;
            }
            if from.is_integer() && to.is_complex() {
                return true;
            }
            from.is_integer() && to.is_float()
        }
        "unsafe" => true,
        _ => false,
    }
}

/// Minimum scalar dtype that can hold a given f64 value
/// (np.min_scalar_type).
#[must_use]
pub fn min_scalar_type(value: f64) -> DType {
    if value.is_nan() || value.is_infinite() {
        return DType::F64;
    }
    if value != value.floor() {
        #[allow(clippy::cast_possible_truncation)]
        let round_trip = (value as f32) as f64;
        if round_trip == value {
            return DType::F32;
        }
        return DType::F64;
    }
    if value == 0.0 || value == 1.0 {
        return DType::Bool;
    }
    if value >= 0.0 {
        if value <= f64::from(u8::MAX) {
            return DType::U8;
        }
        if value <= f64::from(u16::MAX) {
            return DType::U16;
        }
        if value <= f64::from(u32::MAX) {
            return DType::U32;
        }
        // u64::MAX cannot be exactly represented in f64, but 2^64 is 18446744073709551616.0
        if value < 18446744073709551616.0 {
            return DType::U64;
        }
        return DType::F64;
    }
    if value >= f64::from(i8::MIN) {
        return DType::I8;
    }
    if value >= f64::from(i16::MIN) {
        return DType::I16;
    }
    if value >= f64::from(i32::MIN) {
        return DType::I32;
    }
    // i64::MIN is exactly representable in f64 (-9223372036854775808.0)
    if value >= -9223372036854775808.0 {
        return DType::I64;
    }
    DType::F64
}

/// Integer type information (np.iinfo).
///
/// Returns `(min, max, bits)` for the given integer dtype.
/// Returns `None` if the dtype is not an integer type.
#[must_use]
pub const fn iinfo(dtype: DType) -> Option<(i128, i128, u32)> {
    match dtype {
        DType::Bool => Some((0, 1, 1)),
        DType::I8 => Some((i8::MIN as i128, i8::MAX as i128, 8)),
        DType::I16 => Some((i16::MIN as i128, i16::MAX as i128, 16)),
        DType::I32 => Some((i32::MIN as i128, i32::MAX as i128, 32)),
        DType::I64 => Some((i64::MIN as i128, i64::MAX as i128, 64)),
        DType::U8 => Some((0, u8::MAX as i128, 8)),
        DType::U16 => Some((0, u16::MAX as i128, 16)),
        DType::U32 => Some((0, u32::MAX as i128, 32)),
        DType::U64 => Some((0, u64::MAX as i128, 64)),
        _ => None,
    }
}

/// Float type information (np.finfo).
///
/// Returns `(bits, eps, min_positive, max, min_exp, max_exp)` for the given
/// float dtype. Returns `None` if the dtype is not a float type.
///
/// - `bits`: total number of bits
/// - `eps`: machine epsilon (smallest representable positive number s.t. 1.0 + eps != 1.0)
/// - `min_positive`: smallest positive normalized value (tiny)
/// - `max`: largest finite value
/// - `min_exp`: minimum exponent (base 2)
/// - `max_exp`: maximum exponent (base 2)
#[must_use]
pub const fn finfo(dtype: DType) -> Option<(u32, f64, f64, f64, i32, i32)> {
    match dtype {
        DType::F32 => Some((
            32,
            f32::EPSILON as f64,
            f32::MIN_POSITIVE as f64,
            f32::MAX as f64,
            f32::MIN_EXP,
            f32::MAX_EXP,
        )),
        DType::F64 => Some((
            64,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN_EXP,
            f64::MAX_EXP,
        )),
        DType::Complex64 => Some((
            64,
            f32::EPSILON as f64,
            f32::MIN_POSITIVE as f64,
            f32::MAX as f64,
            f32::MIN_EXP,
            f32::MAX_EXP,
        )),
        DType::Complex128 => Some((
            128,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN_EXP,
            f64::MAX_EXP,
        )),
        _ => None,
    }
}

/// Find common float type among dtypes (np.common_type).
/// Integer types are promoted to F64; float types use normal promotion.
#[must_use]
pub fn common_type(dtypes: &[DType]) -> DType {
    let mut result = DType::F32;
    for &dt in dtypes {
        let as_float = if dt.is_float() || dt.is_complex() {
            dt
        } else {
            DType::F64
        };
        result = promote(result, as_float);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{
        DType, can_cast, can_cast_lossless, common_type, finfo, iinfo, min_scalar_type, promote,
        promote_for_mean_reduction, promote_for_sum_reduction, result_type,
    };

    fn all_numeric_dtypes() -> [DType; 13] {
        [
            DType::Bool,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::F32,
            DType::F64,
            DType::Complex64,
            DType::Complex128,
        ]
    }

    #[test]
    fn promotion_is_commutative() {
        let dtypes = all_numeric_dtypes();

        for &lhs in &dtypes {
            for &rhs in &dtypes {
                assert_eq!(promote(lhs, rhs), promote(rhs, lhs), "{lhs:?}/{rhs:?}");
            }
        }
    }

    #[test]
    fn promotion_is_transitive_over_scoped_matrix() {
        let dtypes = all_numeric_dtypes();

        for &src in &dtypes {
            for &mid in &dtypes {
                for &dst in &dtypes {
                    if promote(src, mid) == mid && promote(mid, dst) == dst {
                        assert_eq!(
                            promote(src, dst),
                            dst,
                            "non-transitive promotion path: {src:?} -> {mid:?} -> {dst:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn promotion_expectations_hold() {
        assert_eq!(promote(DType::Bool, DType::I32), DType::I32);
        assert_eq!(promote(DType::I32, DType::I64), DType::I64);
        assert_eq!(promote(DType::I32, DType::F32), DType::F64);
        assert_eq!(promote(DType::F32, DType::F64), DType::F64);
    }

    #[test]
    fn promotion_signed_unsigned_cross() {
        assert_eq!(promote(DType::U8, DType::I8), DType::I16);
        assert_eq!(promote(DType::U16, DType::I16), DType::I32);
        assert_eq!(promote(DType::U32, DType::I32), DType::I64);
        assert_eq!(promote(DType::U64, DType::I64), DType::F64);
    }

    #[test]
    fn promotion_unsigned_same_kind() {
        assert_eq!(promote(DType::U8, DType::U16), DType::U16);
        assert_eq!(promote(DType::U16, DType::U32), DType::U32);
        assert_eq!(promote(DType::U32, DType::U64), DType::U64);
    }

    #[test]
    fn promotion_int_float_rules() {
        // Small ints + F32 -> F32 (F32 mantissa can hold 8/16-bit values)
        assert_eq!(promote(DType::I8, DType::F32), DType::F32);
        assert_eq!(promote(DType::I16, DType::F32), DType::F32);
        assert_eq!(promote(DType::U8, DType::F32), DType::F32);
        assert_eq!(promote(DType::U16, DType::F32), DType::F32);
        // Larger ints + F32 -> F64 (F32 can't represent all 32/64-bit int values)
        assert_eq!(promote(DType::I32, DType::F32), DType::F64);
        assert_eq!(promote(DType::I64, DType::F32), DType::F64);
        assert_eq!(promote(DType::U32, DType::F32), DType::F64);
        assert_eq!(promote(DType::U64, DType::F32), DType::F64);
        // All ints + F64 -> F64
        for &int_dt in &[
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
        ] {
            assert_eq!(promote(int_dt, DType::F64), DType::F64, "{int_dt:?}+F64");
        }
    }

    #[test]
    fn cast_matrix_smoke() {
        assert!(can_cast_lossless(DType::Bool, DType::F64));
        assert!(can_cast_lossless(DType::I32, DType::I64));
        assert!(!can_cast_lossless(DType::I64, DType::I32));
        assert!(!can_cast_lossless(DType::F64, DType::F32));
    }

    #[test]
    fn cast_unsigned_to_signed() {
        assert!(can_cast_lossless(DType::U8, DType::I16));
        assert!(can_cast_lossless(DType::U16, DType::I32));
        assert!(can_cast_lossless(DType::U32, DType::I64));
        assert!(!can_cast_lossless(DType::U8, DType::I8));
        assert!(!can_cast_lossless(DType::U64, DType::I64));
    }

    #[test]
    fn parse_roundtrip_for_known_dtypes() {
        for dtype in all_numeric_dtypes() {
            let parsed = DType::parse(dtype.name()).expect("known dtype should parse");
            assert_eq!(parsed, dtype);
        }
        // complex128 now parses successfully (added with complex type support)
        assert_eq!(DType::parse("complex128"), Some(DType::Complex128));
    }

    #[test]
    fn parse_numpy_style_names() {
        assert_eq!(DType::parse("int8"), Some(DType::I8));
        assert_eq!(DType::parse("int16"), Some(DType::I16));
        assert_eq!(DType::parse("uint8"), Some(DType::U8));
        assert_eq!(DType::parse("uint64"), Some(DType::U64));
        assert_eq!(DType::parse("float32"), Some(DType::F32));
        assert_eq!(DType::parse("float64"), Some(DType::F64));
    }

    #[test]
    fn item_size_matches_expectations() {
        assert_eq!(DType::Bool.item_size(), 1);
        assert_eq!(DType::I8.item_size(), 1);
        assert_eq!(DType::U8.item_size(), 1);
        assert_eq!(DType::I16.item_size(), 2);
        assert_eq!(DType::U16.item_size(), 2);
        assert_eq!(DType::I32.item_size(), 4);
        assert_eq!(DType::U32.item_size(), 4);
        assert_eq!(DType::F32.item_size(), 4);
        assert_eq!(DType::I64.item_size(), 8);
        assert_eq!(DType::U64.item_size(), 8);
        assert_eq!(DType::F64.item_size(), 8);
    }

    #[test]
    fn promotion_is_idempotent() {
        for dtype in all_numeric_dtypes() {
            assert_eq!(promote(dtype, dtype), dtype);
        }
    }

    #[test]
    fn lossless_cast_is_reflexive() {
        for dtype in all_numeric_dtypes() {
            assert!(
                can_cast_lossless(dtype, dtype),
                "{dtype:?} must cast to itself"
            );
        }
    }

    #[test]
    fn lossless_cast_is_transitive_over_scoped_matrix() {
        let dtypes = all_numeric_dtypes();
        for &src in &dtypes {
            for &mid in &dtypes {
                for &dst in &dtypes {
                    if can_cast_lossless(src, mid) && can_cast_lossless(mid, dst) {
                        assert!(
                            can_cast_lossless(src, dst),
                            "non-transitive cast path: {src:?} -> {mid:?} -> {dst:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lossless_cast_implies_promotion_to_destination() {
        let dtypes = all_numeric_dtypes();
        for &src in &dtypes {
            for &dst in &dtypes {
                if can_cast_lossless(src, dst) {
                    assert_eq!(
                        promote(src, dst),
                        dst,
                        "promotion should pick cast-safe destination for {src:?}->{dst:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn sum_reduction_promotion_matches_numpy() {
        assert_eq!(promote_for_sum_reduction(DType::Bool), DType::I64);
        assert_eq!(promote_for_sum_reduction(DType::I8), DType::I64);
        assert_eq!(promote_for_sum_reduction(DType::I16), DType::I64);
        assert_eq!(promote_for_sum_reduction(DType::I32), DType::I64);
        assert_eq!(promote_for_sum_reduction(DType::I64), DType::I64);
        assert_eq!(promote_for_sum_reduction(DType::U8), DType::U64);
        assert_eq!(promote_for_sum_reduction(DType::U16), DType::U64);
        assert_eq!(promote_for_sum_reduction(DType::U32), DType::U64);
        assert_eq!(promote_for_sum_reduction(DType::U64), DType::U64);
        assert_eq!(promote_for_sum_reduction(DType::F32), DType::F32);
        assert_eq!(promote_for_sum_reduction(DType::F64), DType::F64);
    }

    #[test]
    fn mean_reduction_promotion_matches_numpy() {
        assert_eq!(promote_for_mean_reduction(DType::Bool), DType::F64);
        assert_eq!(promote_for_mean_reduction(DType::I32), DType::F64);
        assert_eq!(promote_for_mean_reduction(DType::I64), DType::F64);
        assert_eq!(promote_for_mean_reduction(DType::U8), DType::F64);
        assert_eq!(promote_for_mean_reduction(DType::U64), DType::F64);
        assert_eq!(promote_for_mean_reduction(DType::F32), DType::F32);
        assert_eq!(promote_for_mean_reduction(DType::F64), DType::F64);
    }

    #[test]
    fn is_integer_and_is_float_are_disjoint() {
        for dtype in all_numeric_dtypes() {
            assert!(
                !(dtype.is_integer() && dtype.is_float()),
                "{dtype:?} should not be both integer and float"
            );
        }
    }

    #[test]
    fn result_type_folds_promotion() {
        assert_eq!(result_type(&[DType::I32, DType::F32]), DType::F64);
        assert_eq!(
            result_type(&[DType::I8, DType::I16, DType::I32]),
            DType::I32
        );
        assert_eq!(result_type(&[DType::Bool]), DType::Bool);
        assert_eq!(result_type(&[DType::U8, DType::I8]), DType::I16);
    }

    #[test]
    fn can_cast_rules() {
        // no/equiv: must be identical
        assert!(can_cast(DType::I32, DType::I32, "no"));
        assert!(!can_cast(DType::I32, DType::I64, "no"));
        // safe: lossless only
        assert!(can_cast(DType::I32, DType::I64, "safe"));
        assert!(!can_cast(DType::I64, DType::I32, "safe"));
        // same_kind: within kind or int -> float
        assert!(can_cast(DType::I32, DType::F32, "same_kind"));
        assert!(!can_cast(DType::F32, DType::I32, "same_kind"));
        assert!(can_cast(DType::I64, DType::I8, "same_kind"));
        assert!(can_cast(DType::Bool, DType::F64, "same_kind"));
        // unsafe: anything goes
        assert!(can_cast(DType::F64, DType::Bool, "unsafe"));
    }

    #[test]
    fn min_scalar_type_picks_smallest() {
        assert_eq!(min_scalar_type(0.0), DType::Bool);
        assert_eq!(min_scalar_type(1.0), DType::Bool);
        assert_eq!(min_scalar_type(255.0), DType::U8);
        assert_eq!(min_scalar_type(256.0), DType::U16);
        assert_eq!(min_scalar_type(-1.0), DType::I8);
        assert_eq!(min_scalar_type(-129.0), DType::I16);
        assert_eq!(min_scalar_type(0.5), DType::F32);
        assert_eq!(min_scalar_type(f64::NAN), DType::F64);
    }

    #[test]
    fn common_type_promotes_to_float() {
        assert_eq!(common_type(&[DType::F32, DType::F32]), DType::F32);
        assert_eq!(common_type(&[DType::F32, DType::F64]), DType::F64);
        assert_eq!(common_type(&[DType::I32, DType::F32]), DType::F64);
        assert_eq!(common_type(&[DType::I32, DType::I64]), DType::F64);
    }

    // ── Complex and non-numeric dtype tests ──

    #[test]
    fn parse_complex_dtypes() {
        assert_eq!(DType::parse("complex64"), Some(DType::Complex64));
        assert_eq!(DType::parse("complex128"), Some(DType::Complex128));
        assert_eq!(DType::parse("c8"), Some(DType::Complex64));
        assert_eq!(DType::parse("c16"), Some(DType::Complex128));
    }

    #[test]
    fn parse_string_datetime_dtypes() {
        assert_eq!(DType::parse("str"), Some(DType::Str));
        assert_eq!(DType::parse("U"), Some(DType::Str));
        assert_eq!(DType::parse("U10"), Some(DType::Str));
        assert_eq!(DType::parse("S32"), Some(DType::Str));
        assert_eq!(DType::parse("datetime64"), Some(DType::DateTime64));
        assert_eq!(DType::parse("M8"), Some(DType::DateTime64));
        assert_eq!(DType::parse("datetime64[ns]"), Some(DType::DateTime64));
        assert_eq!(DType::parse("timedelta64"), Some(DType::TimeDelta64));
        assert_eq!(DType::parse("m8"), Some(DType::TimeDelta64));
    }

    #[test]
    fn complex_promotion() {
        assert_eq!(
            promote(DType::Complex64, DType::Complex64),
            DType::Complex64
        );
        assert_eq!(
            promote(DType::Complex64, DType::Complex128),
            DType::Complex128
        );
        assert_eq!(promote(DType::F32, DType::Complex64), DType::Complex64);
        assert_eq!(promote(DType::F64, DType::Complex64), DType::Complex128);
        assert_eq!(promote(DType::I32, DType::Complex64), DType::Complex128);
    }

    #[test]
    fn complex_is_complex() {
        assert!(DType::Complex64.is_complex());
        assert!(DType::Complex128.is_complex());
        assert!(!DType::F64.is_complex());
    }

    #[test]
    fn is_numeric_classification() {
        // Bool is not integer/float/complex in our classification
        assert!(!DType::Bool.is_numeric());
        assert!(!DType::Str.is_numeric());
        assert!(!DType::DateTime64.is_numeric());
        assert!(!DType::TimeDelta64.is_numeric());
        assert!(DType::I32.is_numeric());
        assert!(DType::F64.is_numeric());
        assert!(DType::Complex128.is_numeric());
    }

    #[test]
    fn complex_item_sizes() {
        assert_eq!(DType::Complex64.item_size(), 8); // 2 * f32
        assert_eq!(DType::Complex128.item_size(), 16); // 2 * f64
    }

    #[test]
    fn can_cast_complex_safe() {
        assert!(can_cast_lossless(DType::F32, DType::Complex64));
        assert!(can_cast_lossless(DType::F64, DType::Complex128));
        assert!(can_cast_lossless(DType::Complex64, DType::Complex128));
        assert!(!can_cast_lossless(DType::Complex128, DType::F64));
        assert!(!can_cast_lossless(DType::Complex64, DType::F32));
    }

    // ── iinfo tests ──

    #[test]
    fn iinfo_integer_types() {
        let (min, max, bits) = iinfo(DType::I8).unwrap();
        assert_eq!(min, -128);
        assert_eq!(max, 127);
        assert_eq!(bits, 8);

        let (min, max, bits) = iinfo(DType::U8).unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 255);
        assert_eq!(bits, 8);

        let (min, max, bits) = iinfo(DType::I32).unwrap();
        assert_eq!(min, i32::MIN as i128);
        assert_eq!(max, i32::MAX as i128);
        assert_eq!(bits, 32);

        let (min, max, bits) = iinfo(DType::U64).unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, u64::MAX as i128);
        assert_eq!(bits, 64);
    }

    #[test]
    fn iinfo_non_integer_returns_none() {
        assert!(iinfo(DType::F32).is_none());
        assert!(iinfo(DType::F64).is_none());
        assert!(iinfo(DType::Str).is_none());
        assert!(iinfo(DType::Complex128).is_none());
    }

    #[test]
    fn iinfo_bool() {
        let (min, max, bits) = iinfo(DType::Bool).unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 1);
        assert_eq!(bits, 1);
    }

    // ── finfo tests ──

    #[test]
    fn finfo_float_types() {
        let (bits, eps, tiny, max, min_exp, max_exp) = finfo(DType::F32).unwrap();
        assert_eq!(bits, 32);
        assert!((eps - f64::from(f32::EPSILON)).abs() < 1e-15);
        assert!(tiny > 0.0);
        assert!(max > 1e30);
        assert!(min_exp < 0);
        assert!(max_exp > 0);

        let (bits, eps, tiny, max, min_exp, max_exp) = finfo(DType::F64).unwrap();
        assert_eq!(bits, 64);
        assert!((eps - f64::EPSILON).abs() < 1e-30);
        assert!(tiny > 0.0);
        assert!(max > 1e300);
        assert!(min_exp < 0);
        assert!(max_exp > 0);
    }

    #[test]
    fn finfo_complex_types() {
        // Complex64 has F32 components
        let (bits, eps, _, _, _, _) = finfo(DType::Complex64).unwrap();
        assert_eq!(bits, 64);
        assert!((eps - f64::from(f32::EPSILON)).abs() < 1e-15);

        // Complex128 has F64 components
        let (bits, eps, _, _, _, _) = finfo(DType::Complex128).unwrap();
        assert_eq!(bits, 128);
        assert!((eps - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn finfo_non_float_returns_none() {
        assert!(finfo(DType::I32).is_none());
        assert!(finfo(DType::Bool).is_none());
        assert!(finfo(DType::Str).is_none());
    }
}
