#![forbid(unsafe_code)]

pub use half::f16;

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
    F16,
    F32,
    F64,
    Complex64,
    Complex128,
    Str,
    DateTime64,
    TimeDelta64,
    /// Marker for structured/record dtypes. Field descriptors stored externally.
    Structured,
}

impl DType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8 => "int8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "uint8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
            Self::Str => "str",
            Self::DateTime64 => "datetime64",
            Self::TimeDelta64 => "timedelta64",
            Self::Structured => "void",
        }
    }

    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 | Self::Str | Self::Structured => 1,
            Self::I16 | Self::U16 | Self::F16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64
            | Self::U64
            | Self::F64
            | Self::Complex64
            | Self::DateTime64
            | Self::TimeDelta64 => 8,
            Self::Complex128 => 16,
        }
    }

    #[must_use]
    /// Parse a dtype name string into a `DType` variant.
    ///
    /// Accepts NumPy-compatible descriptor strings where the digit after `i`/`u`/`f`
    /// denotes **byte size**, not bit width: `"i8"` = 8-byte int = `I64`, not `I8`.
    /// Use `"int8"` / `"uint8"` for the 8-bit integer types.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "i1" | "int8" => Some(Self::I8),
            "i2" | "i16" | "int16" => Some(Self::I16),
            "i4" | "i32" | "int32" => Some(Self::I32),
            "i8" | "i64" | "int64" => Some(Self::I64),
            "u1" | "uint8" => Some(Self::U8),
            "u2" | "u16" | "uint16" => Some(Self::U16),
            "u4" | "u32" | "uint32" => Some(Self::U32),
            "u8" | "u64" | "uint64" => Some(Self::U64),
            "f2" | "f16" | "float16" | "half" => Some(Self::F16),
            "f4" | "f32" | "float32" => Some(Self::F32),
            "f8" | "f64" | "float64" => Some(Self::F64),
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

    /// Returns `true` if this is a floating-point type (including complex).
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::F32 | Self::F64 | Self::Complex64 | Self::Complex128
        )
    }

    /// Returns `true` if this is a complex floating-point type.
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex64 | Self::Complex128)
    }

    /// Returns `true` if this is a numeric type (integer, float, or complex).
    #[must_use]
    pub const fn is_numeric(self) -> bool {
        self.is_integer() || self.is_float()
    }
}

/// Deterministic promotion table following NumPy semantics.
///
/// Rules:
/// - `promote(Bool, X) = X` for any X (Bool is the weakest type)
/// - Same kind: pick the wider type
/// - Signed + unsigned: smallest signed that fits both ranges, or F64 for U64+signed
/// - Float16 mirrors NumPy promotion rules: only 8-bit integers stay in
///   float16; 16-bit integers widen to float32 and 32/64-bit integers widen
///   to float64.
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

        // Small integers + F16 -> F16 (NumPy preserves float16 for 8-bit ints)
        (F16, I8 | U8) | (I8 | U8, F16) => F16,
        // 16-bit integers widen float16 to float32
        (F16, I16 | U16) | (I16 | U16, F16) => F32,
        // 32/64-bit integers widen float16 to float64
        (F16, I32 | I64 | U32 | U64) | (I32 | I64 | U32 | U64, F16) => F64,
        // Float-float: pick wider
        (F16, F16) => F16,
        (F16, F32) | (F32, F16) | (F32, F32) => F32,
        (F16, F64) | (F64, F16) | (F32, F64) | (F64, F32) | (F64, F64) => F64,
        // Small integers + F32 -> F32 (F32 has 24-bit mantissa, enough for 8/16-bit ints)
        (F32, I8 | I16 | U8 | U16) | (I8 | I16 | U8 | U16, F32) => F32,
        // Larger integers + F32 -> F64 (F32 can't exactly represent all I32/I64/U32/U64 values)
        (F32, I32 | I64 | U32 | U64) | (I32 | I64 | U32 | U64, F32) => F64,
        // Any integer + F64 -> F64 (F64 already wider than any integer except U64 > 2^53)
        (F64, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64)
        | (I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64, F64) => F64,

        // Complex promotion: complex absorbs float and integer
        (Complex64, Complex64) => Complex64,
        (Complex128, Complex128) => Complex128,
        (Complex64, Complex128) | (Complex128, Complex64) => Complex128,
        // Float + Complex: promote to matching complex
        (F16, Complex64) | (Complex64, F16) => Complex64,
        (F32, Complex64) | (Complex64, F32) => Complex64,
        (F64, Complex64)
        | (Complex64, F64)
        | (F16, Complex128)
        | (Complex128, F16)
        | (F32, Complex128)
        | (Complex128, F32) => Complex128,
        (F64, Complex128) | (Complex128, F64) => Complex128,
        // Small integers + Complex64 -> Complex64 (mirrors F32 mantissa rule)
        // Note: Bool is already handled by the (Bool, x) | (x, Bool) => x rule above
        (Complex64, I8 | I16 | U8 | U16) | (I8 | I16 | U8 | U16, Complex64) => Complex64,
        // Larger integers + Complex64 -> Complex128 (mirrors F32->F64 rule)
        (Complex64, I32 | I64 | U32 | U64) | (I32 | I64 | U32 | U64, Complex64) => Complex128,
        // Any integer + Complex128 -> Complex128
        (Complex128, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64)
        | (I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64, Complex128) => Complex128,

        // DateTime and TimeDelta: same-family promotion only
        (DateTime64, DateTime64) => DateTime64,
        (TimeDelta64, TimeDelta64) => TimeDelta64,
        (DateTime64, TimeDelta64) | (TimeDelta64, DateTime64) => DateTime64,
        // DateTime64/TimeDelta64 + numeric: incompatible in NumPy (TypeError).
        // Fail-closed to F64 since promote() returns DType, not Result.
        (
            DateTime64,
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F16 | F32 | F64 | Complex64 | Complex128,
        )
        | (
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F16 | F32 | F64 | Complex64 | Complex128,
            DateTime64,
        ) => F64,
        (
            TimeDelta64,
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F16 | F32 | F64 | Complex64 | Complex128,
        )
        | (
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F16 | F32 | F64 | Complex64 | Complex128,
            TimeDelta64,
        ) => F64,

        // Str: always wins against numeric and temporal types
        (Str, x) | (x, Str) => match x {
            Structured => F64, // Structured is incompatible
            _ => Str,
        },

        // Structured: incompatible with everything else (even itself, if different fields)
        (Structured, _) | (_, Structured) => F64,
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
        DType::F16 => DType::F16,
        DType::F32 => DType::F32,
        DType::F64 => DType::F64,
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        DType::Str | DType::DateTime64 | DType::TimeDelta64 | DType::Structured => dt,
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
        DType::F16 => DType::F16,
        DType::F32 => DType::F32,
        DType::F64 => DType::F64,
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        DType::Str | DType::DateTime64 | DType::TimeDelta64 | DType::Structured => dt,
    }
}

/// Returns `true` if `src` can be cast to `dst` without information loss.
/// Follows NumPy's safe-cast rules.
#[must_use]
pub const fn can_cast_lossless(src: DType, dst: DType) -> bool {
    if matches!(
        (src, dst),
        (DType::Bool, DType::Bool)
            | (DType::I8, DType::I8)
            | (DType::I16, DType::I16)
            | (DType::I32, DType::I32)
            | (DType::I64, DType::I64)
            | (DType::U8, DType::U8)
            | (DType::U16, DType::U16)
            | (DType::U32, DType::U32)
            | (DType::U64, DType::U64)
            | (DType::F16, DType::F16)
            | (DType::F32, DType::F32)
            | (DType::F64, DType::F64)
            | (DType::Complex64, DType::Complex64)
            | (DType::Complex128, DType::Complex128)
            | (DType::Str, DType::Str)
            | (DType::DateTime64, DType::DateTime64)
            | (DType::TimeDelta64, DType::TimeDelta64)
    ) {
        return true;
    }

    match src {
        DType::Structured | DType::Str | DType::DateTime64 | DType::TimeDelta64 => false,
        DType::Bool => matches!(
            dst,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::Complex64
                | DType::Complex128
                | DType::Str
                | DType::TimeDelta64
        ),
        DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64 => safe_integer_cast(src, dst),
        DType::F16 | DType::F32 | DType::F64 => safe_float_cast(src, dst),
        DType::Complex64 | DType::Complex128 => safe_complex_cast(src, dst),
    }
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
        "same_kind" => can_cast_same_kind(from, to),
        "unsafe" => true,
        _ => false,
    }
}

const fn integer_bits(dtype: DType) -> Option<u32> {
    match dtype {
        DType::I8 | DType::U8 => Some(8),
        DType::I16 | DType::U16 => Some(16),
        DType::I32 | DType::U32 => Some(32),
        DType::I64 | DType::U64 => Some(64),
        _ => None,
    }
}

const fn is_signed_integer_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64)
}

const fn is_unsigned_integer_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::U8 | DType::U16 | DType::U32 | DType::U64)
}

const fn safe_integer_cast(src: DType, dst: DType) -> bool {
    let Some(src_bits) = integer_bits(src) else {
        return false;
    };
    match dst {
        DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64 => {
            let Some(dst_bits) = integer_bits(dst) else {
                return false;
            };
            if (is_signed_integer_dtype(src) && is_signed_integer_dtype(dst))
                || (is_unsigned_integer_dtype(src) && is_unsigned_integer_dtype(dst))
            {
                dst_bits >= src_bits
            } else if is_unsigned_integer_dtype(src) && is_signed_integer_dtype(dst) {
                dst_bits > src_bits
            } else {
                false
            }
        }
        DType::F16 | DType::F32 | DType::F64 | DType::Complex64 | DType::Complex128 => {
            let Some(dst_bits) = (match dst {
                DType::F16 => Some(8),
                DType::F32 | DType::Complex64 => Some(16),
                // NumPy treats float64/complex128 as safely holding any 64-bit integer dtype.
                DType::F64 | DType::Complex128 => Some(64),
                _ => None,
            }) else {
                return false;
            };
            src_bits <= dst_bits
        }
        DType::Str => true,
        DType::TimeDelta64 => !matches!(src, DType::U64),
        _ => false,
    }
}

const fn safe_float_cast(src: DType, dst: DType) -> bool {
    match src {
        DType::F16 => matches!(
            dst,
            DType::F32 | DType::F64 | DType::Complex64 | DType::Complex128 | DType::Str
        ),
        DType::F32 => matches!(
            dst,
            DType::F64 | DType::Complex64 | DType::Complex128 | DType::Str
        ),
        DType::F64 => matches!(dst, DType::Complex128 | DType::Str),
        _ => false,
    }
}

const fn safe_complex_cast(src: DType, dst: DType) -> bool {
    match src {
        DType::Complex64 => matches!(dst, DType::Complex128 | DType::Str),
        DType::Complex128 => matches!(dst, DType::Str),
        _ => false,
    }
}

fn can_cast_same_kind(from: DType, to: DType) -> bool {
    if from == to {
        return true;
    }
    if from == DType::Structured || to == DType::Structured {
        return false;
    }
    // Str is a sink for numeric kinds
    if to == DType::Str {
        return from.is_numeric() || from == DType::Bool;
    }
    // Temporal types are isolated kinds, BUT TimeDelta64 allows integer->m8
    if to == DType::DateTime64 {
        return from == DType::DateTime64;
    }
    if to == DType::TimeDelta64 {
        return from == DType::TimeDelta64 || from.is_integer() || from == DType::Bool;
    }
    if from == DType::DateTime64 || from == DType::TimeDelta64 {
        return false;
    }

    // Kind Hierarchy: Bool < Integer < Float < Complex
    if from == DType::Bool {
        return to.is_integer() || to.is_float(); // is_float includes complex
    }
    if from.is_integer() {
        if is_signed_integer_dtype(from) {
            // Signed int can cast to other signed ints, floats, or complex
            return is_signed_integer_dtype(to) || to.is_float();
        }
        // Unsigned int can cast to any int, float, or complex
        return to.is_integer() || to.is_float();
    }
    if from.is_float() {
        // Float can cast to complex, but NOT back to standard integer kinds
        if from.is_complex() {
            return to.is_complex();
        }
        return to.is_float(); // includes complex
    }
    false
}

/// Minimum scalar dtype for a floating scalar value
/// (np.min_scalar_type on Python/NumPy float inputs).
#[must_use]
pub fn min_scalar_type(value: f64) -> DType {
    // NumPy's floating-scalar path never demotes to integer/bool kinds.
    // It uses conservative magnitude thresholds rather than exact
    // representability checks.
    if !value.is_finite() || (value > -65_000.0 && value < 65_000.0) {
        return DType::F16;
    }
    if value > -3.4e38 && value < 3.4e38 {
        return DType::F32;
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
        DType::F16 => Some((
            16,
            0.000_976_562_5,
            0.000_061_035_156_25,
            65_504.0,
            f16::MIN_EXP,
            f16::MAX_EXP,
        )),
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
            32,
            f32::EPSILON as f64,
            f32::MIN_POSITIVE as f64,
            f32::MAX as f64,
            f32::MIN_EXP,
            f32::MAX_EXP,
        )),
        DType::Complex128 => Some((
            64,
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
    let mut iter = dtypes.iter().copied();
    let Some(first) = iter.next() else {
        return DType::F64;
    };
    let mut result = if first.is_float() || first.is_complex() {
        first
    } else {
        DType::F64
    };
    for dt in iter {
        let as_float = if dt.is_float() || dt.is_complex() {
            dt
        } else {
            DType::F64
        };
        result = promote(result, as_float);
    }
    result
}

/// Descriptor for a single field in a structured dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredField {
    /// Field name.
    pub name: std::string::String,
    /// Field dtype.
    pub dtype: DType,
    /// Byte offset within the record.
    pub offset: usize,
}

/// Storage for structured/record arrays.
/// Each record contains named, typed fields at fixed offsets.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredStorage {
    /// Field descriptors (name, dtype, byte offset).
    pub fields: Vec<StructuredField>,
    /// Number of records.
    pub num_records: usize,
    /// Per-field typed data, parallel with `fields`.
    pub columns: Vec<ArrayStorage>,
}

impl StructuredStorage {
    /// Create a structured storage with no fields and no records.
    #[must_use]
    pub fn empty() -> Self {
        Self::empty_with_len(0)
    }

    /// Create a structured storage with no fields and a fixed logical record count.
    #[must_use]
    pub fn empty_with_len(num_records: usize) -> Self {
        Self {
            fields: Vec::new(),
            num_records,
            columns: Vec::new(),
        }
    }

    /// Create a structured storage from field descriptors and column data.
    /// All columns must have the same length.
    pub fn new(
        fields: Vec<StructuredField>,
        columns: Vec<ArrayStorage>,
    ) -> Result<Self, StorageError> {
        if fields.len() != columns.len() {
            return Err(StorageError::StructuredFieldMismatch {
                expected: fields.len(),
                got: columns.len(),
            });
        }
        let num_records = if columns.is_empty() {
            0
        } else {
            let n = columns[0].len();
            for (field, col) in fields.iter().zip(columns.iter()) {
                if col.len() != n {
                    return Err(StorageError::StructuredFieldMismatch {
                        expected: n,
                        got: col.len(),
                    });
                }
                if col.dtype() != field.dtype {
                    return Err(StorageError::UnsupportedCast {
                        from: col.dtype(),
                        to: field.dtype,
                    });
                }
            }
            n
        };
        Ok(Self {
            fields,
            num_records,
            columns,
        })
    }

    /// Total byte width of one logical record, respecting explicit field offsets.
    #[must_use]
    pub fn record_size(&self) -> usize {
        self.fields
            .iter()
            .map(|field| field.offset.saturating_add(field.dtype.item_size()))
            .max()
            .unwrap_or(0)
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_records
    }

    /// Whether the storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_records == 0
    }

    /// Number of fields.
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// Get field names.
    #[must_use]
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get field index by name.
    #[must_use]
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// Access a column (field) by name, returning the storage for that field.
    #[must_use]
    pub fn get_field(&self, name: &str) -> Option<&ArrayStorage> {
        self.field_index(name).map(|idx| &self.columns[idx])
    }

    /// Access a column (field) by index.
    #[must_use]
    pub fn get_field_by_index(&self, index: usize) -> Option<&ArrayStorage> {
        self.columns.get(index)
    }

    /// Get the dtype string representation (NumPy format).
    #[must_use]
    pub fn dtype_str(&self) -> std::string::String {
        let field_strs: Vec<std::string::String> = self
            .fields
            .iter()
            .map(|f| format!("('{}', '{}')", f.name, f.dtype.name()))
            .collect();
        format!("[{}]", field_strs.join(", "))
    }
}

/// Polymorphic typed storage backend for array data.
///
/// Replaces the `Vec<f64>`-only storage pattern. Each variant holds a
/// homogeneous typed buffer matching a NumPy dtype. This enables correct
/// integer fidelity (i64 > 2^53), proper f32 identity, native complex
/// storage, and memory-efficient booleans.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayStorage {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Complex64(Vec<(f32, f32)>),
    Complex128(Vec<(f64, f64)>),
    String(Vec<std::string::String>),
    Structured(StructuredStorage),
}

/// Convert f64 to i128 with NumPy-compatible semantics: truncate toward zero,
/// then allow integer wrapping via `as` casts. NaN and infinity map to 0,
/// matching `numpy`'s behavior for integer casts of non-finite floats.
fn f64_to_wrapping_i128(val: f64) -> i128 {
    if val.is_nan() || val.is_infinite() {
        return 0;
    }
    let truncated = val.trunc();
    // i128 range: roughly ±1.7e38, covering all practical integer results.
    // Values outside i128 range saturate, which is acceptable since they
    // exceed any integer dtype's range and will wrap anyway.
    truncated as i128
}

/// Error type for storage operations.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageError {
    IndexOutOfBounds { index: usize, len: usize },
    UnsupportedCast { from: DType, to: DType },
    StructuredFieldMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for storage of length {len}")
            }
            Self::UnsupportedCast { from, to } => {
                write!(f, "cannot cast from {} to {}", from.name(), to.name())
            }
            Self::StructuredFieldMismatch { expected, got } => {
                write!(
                    f,
                    "structured storage field count mismatch: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl ArrayStorage {
    /// Number of elements in the storage.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Complex64(v) => v.len(),
            Self::Complex128(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Structured(s) => s.len(),
        }
    }

    /// Whether the storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The dtype this storage represents.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::Complex64(_) => DType::Complex64,
            Self::Complex128(_) => DType::Complex128,
            Self::String(_) => DType::Str,
            Self::Structured(_) => DType::Structured,
        }
    }

    /// Read element at `index` as f64 (lossy for integer types > 2^53).
    /// Complex types return the real part. Bool maps 0.0/1.0.
    /// String returns NaN.
    pub fn get_f64(&self, index: usize) -> Result<f64, StorageError> {
        let n = self.len();
        if index >= n {
            return Err(StorageError::IndexOutOfBounds { index, len: n });
        }
        Ok(match self {
            Self::Bool(v) => {
                if v[index] {
                    1.0
                } else {
                    0.0
                }
            }
            Self::I8(v) => f64::from(v[index]),
            Self::I16(v) => f64::from(v[index]),
            Self::I32(v) => f64::from(v[index]),
            Self::I64(v) => v[index] as f64,
            Self::U8(v) => f64::from(v[index]),
            Self::U16(v) => f64::from(v[index]),
            Self::U32(v) => f64::from(v[index]),
            Self::U64(v) => v[index] as f64,
            Self::F16(v) => f64::from(v[index]),
            Self::F32(v) => f64::from(v[index]),
            Self::F64(v) => v[index],
            Self::Complex64(v) => f64::from(v[index].0),
            Self::Complex128(v) => v[index].0,
            Self::String(_) | Self::Structured(_) => f64::NAN,
        })
    }

    /// Write element at `index` from an f64 value (lossy cast to target type).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn set_f64(&mut self, index: usize, val: f64) -> Result<(), StorageError> {
        let n = self.len();
        if index >= n {
            return Err(StorageError::IndexOutOfBounds { index, len: n });
        }
        // NumPy-compatible wrapping: float→int casts truncate toward zero then
        // wrap via two's-complement, matching C semantics. Rust's `f64 as u8`
        // saturates, so we route through i128 (integer→integer `as` wraps).
        // NaN and infinity map to 0 (matching NumPy's integer cast behavior).
        match self {
            Self::Bool(v) => v[index] = val != 0.0,
            Self::I8(v) => v[index] = f64_to_wrapping_i128(val) as i8,
            Self::I16(v) => v[index] = f64_to_wrapping_i128(val) as i16,
            Self::I32(v) => v[index] = f64_to_wrapping_i128(val) as i32,
            Self::I64(v) => v[index] = f64_to_wrapping_i128(val) as i64,
            Self::U8(v) => v[index] = f64_to_wrapping_i128(val) as u8,
            Self::U16(v) => v[index] = f64_to_wrapping_i128(val) as u16,
            Self::U32(v) => v[index] = f64_to_wrapping_i128(val) as u32,
            Self::U64(v) => v[index] = f64_to_wrapping_i128(val) as u64,
            Self::F16(v) => v[index] = f16::from_f64(val),
            Self::F32(v) => v[index] = val as f32,
            Self::F64(v) => v[index] = val,
            Self::Complex64(v) => v[index] = (val as f32, 0.0),
            Self::Complex128(v) => v[index] = (val, 0.0),
            Self::String(v) => v[index] = val.to_string(),
            Self::Structured(_) => {
                return Err(StorageError::UnsupportedCast {
                    from: DType::F64,
                    to: DType::Structured,
                });
            }
        }
        Ok(())
    }

    /// Create a new storage of the given dtype with `n` zero-valued elements.
    #[must_use]
    pub fn zeros(dtype: DType, n: usize) -> Self {
        match dtype {
            DType::Bool => Self::Bool(vec![false; n]),
            DType::I8 => Self::I8(vec![0; n]),
            DType::I16 => Self::I16(vec![0; n]),
            DType::I32 => Self::I32(vec![0; n]),
            DType::I64 => Self::I64(vec![0; n]),
            DType::U8 => Self::U8(vec![0; n]),
            DType::U16 => Self::U16(vec![0; n]),
            DType::U32 => Self::U32(vec![0; n]),
            DType::U64 => Self::U64(vec![0; n]),
            DType::F16 => Self::F16(vec![f16::from_f32(0.0); n]),
            DType::F32 => Self::F32(vec![0.0; n]),
            DType::F64 => Self::F64(vec![0.0; n]),
            DType::Complex64 => Self::Complex64(vec![(0.0, 0.0); n]),
            DType::Complex128 => Self::Complex128(vec![(0.0, 0.0); n]),
            DType::Str => Self::String(vec![std::string::String::new(); n]),
            DType::DateTime64 | DType::TimeDelta64 => Self::I64(vec![0; n]),
            DType::Structured => Self::Structured(StructuredStorage::empty_with_len(n)),
        }
    }

    /// Read element at `index` as i128 (lossless for all integer types).
    /// Returns 0 for non-numeric types. Floats are truncated.
    pub fn get_i128(&self, index: usize) -> Result<i128, StorageError> {
        let n = self.len();
        if index >= n {
            return Err(StorageError::IndexOutOfBounds { index, len: n });
        }
        Ok(match self {
            Self::Bool(v) => {
                if v[index] {
                    1
                } else {
                    0
                }
            }
            Self::I8(v) => i128::from(v[index]),
            Self::I16(v) => i128::from(v[index]),
            Self::I32(v) => i128::from(v[index]),
            Self::I64(v) => i128::from(v[index]),
            Self::U8(v) => i128::from(v[index]),
            Self::U16(v) => i128::from(v[index]),
            Self::U32(v) => i128::from(v[index]),
            Self::U64(v) => i128::from(v[index]),
            Self::F16(v) => f64::from(v[index]) as i128,
            Self::F32(v) => v[index] as i128,
            Self::F64(v) => v[index] as i128,
            Self::Complex64(v) => v[index].0 as i128,
            Self::Complex128(v) => v[index].0 as i128,
            Self::String(_) | Self::Structured(_) => 0,
        })
    }

    /// Write element at `index` from an i128 value (lossy cast to target type).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn set_i128(&mut self, index: usize, val: i128) -> Result<(), StorageError> {
        let n = self.len();
        if index >= n {
            return Err(StorageError::IndexOutOfBounds { index, len: n });
        }
        match self {
            Self::Bool(v) => v[index] = val != 0,
            Self::I8(v) => v[index] = val as i8,
            Self::I16(v) => v[index] = val as i16,
            Self::I32(v) => v[index] = val as i32,
            Self::I64(v) => v[index] = val as i64,
            Self::U8(v) => v[index] = val as u8,
            Self::U16(v) => v[index] = val as u16,
            Self::U32(v) => v[index] = val as u32,
            Self::U64(v) => v[index] = val as u64,
            Self::F16(v) => v[index] = f16::from_f64(val as f64),
            Self::F32(v) => v[index] = val as f32,
            Self::F64(v) => v[index] = val as f64,
            Self::Complex64(v) => v[index] = (val as f32, 0.0),
            Self::Complex128(v) => v[index] = (val as f64, 0.0),
            Self::String(v) => v[index] = val.to_string(),
            Self::Structured(_) => {
                return Err(StorageError::UnsupportedCast {
                    from: DType::I64, // using I64 as proxy for integer kind
                    to: DType::Structured,
                });
            }
        }
        Ok(())
    }

    /// Cast this storage to a different dtype.
    /// Returns a new storage with elements converted to the target type.
    pub fn cast_to(&self, target: DType) -> Result<Self, StorageError> {
        if self.dtype() == target {
            return Ok(self.clone());
        }
        if matches!(self, Self::Structured(_)) {
            return Err(StorageError::UnsupportedCast {
                from: DType::Structured,
                to: target,
            });
        }
        if target == DType::Structured {
            return Err(StorageError::UnsupportedCast {
                from: self.dtype(),
                to: DType::Structured,
            });
        }
        let n = self.len();
        // String target: convert all elements to string representation
        if target == DType::Str {
            let strings: Vec<std::string::String> = match self {
                Self::Bool(v) => v
                    .iter()
                    .map(|&b| if b { "True".into() } else { "False".into() })
                    .collect(),
                Self::I8(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::I16(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::I32(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::I64(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::U8(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::U16(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::U32(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::U64(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::F16(v) => v.iter().map(|x| f32::from(*x).to_string()).collect(),
                Self::F32(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::F64(v) => v.iter().map(|x| x.to_string()).collect(),
                Self::Complex64(v) => v.iter().map(|&(re, im)| format!("({re}+{im}j)")).collect(),
                Self::Complex128(v) => v.iter().map(|&(re, im)| format!("({re}+{im}j)")).collect(),
                Self::String(v) => v.clone(),
                Self::Structured(_) => vec!["NaN".to_string(); n],
            };
            return Ok(Self::String(strings));
        }
        // String source: cannot cast to numeric
        if matches!(self, Self::String(_)) {
            return Err(StorageError::UnsupportedCast {
                from: DType::Str,
                to: target,
            });
        }

        let mut result = Self::zeros(target, n);
        let src_is_int = self.dtype().is_integer() || self.dtype() == DType::Bool;
        let dst_is_int = target.is_integer() || target == DType::Bool;

        if src_is_int && dst_is_int {
            // Integer-to-integer: use i128 for lossless transfer
            for i in 0..n {
                let val = self.get_i128(i)?;
                result.set_i128(i, val)?;
            }
        } else {
            // Numeric-to-numeric: go through f64 intermediary
            for i in 0..n {
                let val = self.get_f64(i)?;
                result.set_f64(i, val)?;
            }
        }
        Ok(result)
    }

    /// Extract the underlying data as a `Vec<f64>` (lossy for non-f64 types).
    #[must_use]
    pub fn to_f64_vec(&self) -> Vec<f64> {
        let n = self.len();
        (0..n)
            .map(|i| self.get_f64(i).unwrap_or(f64::NAN))
            .collect()
    }

    /// Create F64 storage from a Vec<f64>.
    #[must_use]
    pub fn from_f64_vec(data: Vec<f64>) -> Self {
        Self::F64(data)
    }

    /// Create F16 storage from a Vec<f16>.
    #[must_use]
    pub fn from_f16_vec(data: Vec<f16>) -> Self {
        Self::F16(data)
    }

    /// Create Complex128 storage from a Vec of (real, imag) pairs.
    #[must_use]
    pub fn from_complex128_vec(data: Vec<(f64, f64)>) -> Self {
        Self::Complex128(data)
    }

    /// Create Complex64 storage from a Vec of (real, imag) pairs.
    #[must_use]
    pub fn from_complex64_vec(data: Vec<(f32, f32)>) -> Self {
        Self::Complex64(data)
    }

    /// Extract data as Vec of (f64, f64) complex pairs.
    /// Non-complex types get zero imaginary part.
    #[must_use]
    pub fn to_complex128_vec(&self) -> Vec<(f64, f64)> {
        match self {
            Self::Complex128(v) => v.clone(),
            Self::Complex64(v) => v
                .iter()
                .map(|&(r, i)| (f64::from(r), f64::from(i)))
                .collect(),
            _ => {
                let n = self.len();
                (0..n)
                    .map(|i| (self.get_f64(i).unwrap_or(f64::NAN), 0.0))
                    .collect()
            }
        }
    }

    /// Read element at `index` as a complex128 (f64, f64) pair.
    /// Non-complex types return (real_value, 0.0).
    pub fn get_complex128(&self, index: usize) -> Result<(f64, f64), StorageError> {
        let n = self.len();
        if index >= n {
            return Err(StorageError::IndexOutOfBounds { index, len: n });
        }
        Ok(match self {
            Self::Complex128(v) => v[index],
            Self::Complex64(v) => (f64::from(v[index].0), f64::from(v[index].1)),
            _ => (self.get_f64(index)?, 0.0),
        })
    }

    /// Write element at `index` as a complex128 (re, im) pair.
    /// For non-complex storage, only the real part is written.
    pub fn set_complex128(&mut self, index: usize, re: f64, im: f64) -> Result<(), StorageError> {
        let n = self.len();
        if index >= n {
            return Err(StorageError::IndexOutOfBounds { index, len: n });
        }
        match self {
            Self::Complex128(v) => v[index] = (re, im),
            Self::Complex64(v) => v[index] = (re as f32, im as f32),
            _ => self.set_f64(index, re)?,
        }
        Ok(())
    }

    /// Whether this storage holds complex data.
    #[must_use]
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::Complex64(_) | Self::Complex128(_))
    }

    // ── Complex arithmetic (element-wise, returns Complex128) ──

    /// Element-wise complex addition.
    pub fn complex_add(&self, other: &Self) -> Result<Self, StorageError> {
        let a = self.to_complex128_vec();
        let b = other.to_complex128_vec();
        if a.len() != b.len() {
            return Err(StorageError::UnsupportedCast {
                from: self.dtype(),
                to: other.dtype(),
            });
        }
        Ok(Self::Complex128(
            a.iter()
                .zip(b.iter())
                .map(|(&(ar, ai), &(br, bi))| (ar + br, ai + bi))
                .collect(),
        ))
    }

    /// Element-wise complex subtraction.
    pub fn complex_sub(&self, other: &Self) -> Result<Self, StorageError> {
        let a = self.to_complex128_vec();
        let b = other.to_complex128_vec();
        if a.len() != b.len() {
            return Err(StorageError::UnsupportedCast {
                from: self.dtype(),
                to: other.dtype(),
            });
        }
        Ok(Self::Complex128(
            a.iter()
                .zip(b.iter())
                .map(|(&(ar, ai), &(br, bi))| (ar - br, ai - bi))
                .collect(),
        ))
    }

    /// Element-wise complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    pub fn complex_mul(&self, other: &Self) -> Result<Self, StorageError> {
        let a = self.to_complex128_vec();
        let b = other.to_complex128_vec();
        if a.len() != b.len() {
            return Err(StorageError::UnsupportedCast {
                from: self.dtype(),
                to: other.dtype(),
            });
        }
        Ok(Self::Complex128(
            a.iter()
                .zip(b.iter())
                .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
                .collect(),
        ))
    }

    /// Element-wise complex division: (a+bi)/(c+di).
    pub fn complex_div(&self, other: &Self) -> Result<Self, StorageError> {
        let a = self.to_complex128_vec();
        let b = other.to_complex128_vec();
        if a.len() != b.len() {
            return Err(StorageError::UnsupportedCast {
                from: self.dtype(),
                to: other.dtype(),
            });
        }
        Ok(Self::Complex128(
            a.iter()
                .zip(b.iter())
                .map(|(&(ar, ai), &(br, bi))| {
                    let denom = br * br + bi * bi;
                    if denom == 0.0 {
                        (f64::NAN, f64::NAN)
                    } else {
                        ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
                    }
                })
                .collect(),
        ))
    }

    // ── Complex unary operations ──

    /// Element-wise complex conjugate: (a+bi) → (a-bi).
    #[must_use]
    pub fn complex_conjugate(&self) -> Self {
        match self {
            Self::Complex128(v) => Self::Complex128(v.iter().map(|&(r, i)| (r, -i)).collect()),
            Self::Complex64(v) => Self::Complex64(v.iter().map(|&(r, i)| (r, -i)).collect()),
            // For real types, conjugate is identity
            _ => self.clone(),
        }
    }

    /// Element-wise complex absolute value (magnitude): |a+bi| = sqrt(a²+b²).
    /// Returns F64 storage.
    #[must_use]
    pub fn complex_abs(&self) -> Self {
        match self {
            Self::Complex128(v) => {
                Self::F64(v.iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect())
            }
            Self::Complex64(v) => Self::F64(
                v.iter()
                    .map(|&(r, i)| {
                        let r64 = f64::from(r);
                        let i64 = f64::from(i);
                        (r64 * r64 + i64 * i64).sqrt()
                    })
                    .collect(),
            ),
            // For real types, abs is just f64 abs
            _ => {
                let vals = self.to_f64_vec();
                Self::F64(vals.into_iter().map(f64::abs).collect())
            }
        }
    }

    /// Element-wise complex angle (argument): atan2(imag, real).
    /// Returns F64 storage.
    #[must_use]
    pub fn complex_angle(&self) -> Self {
        match self {
            Self::Complex128(v) => Self::F64(v.iter().map(|&(r, i)| i.atan2(r)).collect()),
            Self::Complex64(v) => Self::F64(
                v.iter()
                    .map(|&(r, i)| f64::from(i).atan2(f64::from(r)))
                    .collect(),
            ),
            // For real types, angle is 0.0 for non-negative, pi for negative
            _ => {
                let vals = self.to_f64_vec();
                Self::F64(
                    vals.into_iter()
                        .map(|v| if v < 0.0 { std::f64::consts::PI } else { 0.0 })
                        .collect(),
                )
            }
        }
    }

    /// Extract real part. Returns F64 storage.
    #[must_use]
    pub fn complex_real(&self) -> Self {
        match self {
            Self::Complex128(v) => Self::F64(v.iter().map(|&(r, _)| r).collect()),
            Self::Complex64(v) => Self::F64(v.iter().map(|&(r, _)| f64::from(r)).collect()),
            // For real types, real part is the value itself
            _ => Self::F64(self.to_f64_vec()),
        }
    }

    /// Extract imaginary part. Returns F64 storage.
    #[must_use]
    pub fn complex_imag(&self) -> Self {
        match self {
            Self::Complex128(v) => Self::F64(v.iter().map(|&(_, i)| i).collect()),
            Self::Complex64(v) => Self::F64(v.iter().map(|&(_, i)| f64::from(i)).collect()),
            // For real types, imaginary part is zero
            _ => Self::F64(vec![0.0; self.len()]),
        }
    }

    /// Element-wise complex exponential: exp(a+bi) = exp(a) * (cos(b) + i*sin(b)).
    #[must_use]
    pub fn complex_exp(&self) -> Self {
        let pairs = self.to_complex128_vec();
        Self::Complex128(
            pairs
                .iter()
                .map(|&(r, i)| {
                    let ea = r.exp();
                    (ea * i.cos(), ea * i.sin())
                })
                .collect(),
        )
    }

    /// Element-wise complex natural logarithm: log(a+bi) = log|z| + i*arg(z).
    #[must_use]
    pub fn complex_log(&self) -> Self {
        let pairs = self.to_complex128_vec();
        Self::Complex128(
            pairs
                .iter()
                .map(|&(r, i)| {
                    let mag = (r * r + i * i).sqrt();
                    let ang = i.atan2(r);
                    (mag.ln(), ang)
                })
                .collect(),
        )
    }

    /// Element-wise complex square root.
    #[must_use]
    pub fn complex_sqrt(&self) -> Self {
        let pairs = self.to_complex128_vec();
        Self::Complex128(
            pairs
                .iter()
                .map(|&(r, i)| {
                    let mag = (r * r + i * i).sqrt();
                    let re = f64::midpoint(mag, r).sqrt();
                    let im = f64::midpoint(mag, -r).sqrt();
                    (re, if i >= 0.0 { im } else { -im })
                })
                .collect(),
        )
    }

    /// Element-wise complex power: z^w = exp(w * log(z)).
    #[must_use]
    pub fn complex_pow(&self, exponent: &Self) -> Self {
        let bases = self.to_complex128_vec();
        let exps = exponent.to_complex128_vec();
        let n = bases.len().min(exps.len());
        Self::Complex128(
            (0..n)
                .map(|idx| {
                    let (zr, zi) = bases[idx];
                    let (wr, wi) = exps[idx];
                    // z^w = exp(w * log(z))
                    let mag = (zr * zr + zi * zi).sqrt();
                    let ang = zi.atan2(zr);
                    let log_r = mag.ln();
                    let log_i = ang;
                    // w * log(z)
                    let prod_r = wr * log_r - wi * log_i;
                    let prod_i = wr * log_i + wi * log_r;
                    // exp(prod)
                    let ea = prod_r.exp();
                    (ea * prod_i.cos(), ea * prod_i.sin())
                })
                .collect(),
        )
    }

    /// Element-wise complex sin: sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b).
    #[must_use]
    pub fn complex_sin(&self) -> Self {
        let pairs = self.to_complex128_vec();
        Self::Complex128(
            pairs
                .iter()
                .map(|&(r, i)| (r.sin() * i.cosh(), r.cos() * i.sinh()))
                .collect(),
        )
    }

    /// Element-wise complex cos: cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b).
    #[must_use]
    pub fn complex_cos(&self) -> Self {
        let pairs = self.to_complex128_vec();
        Self::Complex128(
            pairs
                .iter()
                .map(|&(r, i)| (r.cos() * i.cosh(), -(r.sin() * i.sinh())))
                .collect(),
        )
    }

    /// Element-wise complex sum reduction.
    #[must_use]
    pub fn complex_sum(&self) -> (f64, f64) {
        let pairs = self.to_complex128_vec();
        pairs
            .iter()
            .fold((0.0, 0.0), |(sr, si), &(r, i)| (sr + r, si + i))
    }

    /// Element-wise complex product reduction.
    #[must_use]
    pub fn complex_prod(&self) -> (f64, f64) {
        let pairs = self.to_complex128_vec();
        pairs.iter().fold((1.0, 0.0), |(pr, pi), &(r, i)| {
            (pr * r - pi * i, pr * i + pi * r)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArrayStorage, DType, StorageError, StructuredField, StructuredStorage, can_cast,
        can_cast_lossless, common_type, f16, finfo, iinfo, min_scalar_type, promote,
        promote_for_mean_reduction, promote_for_sum_reduction, result_type,
    };

    fn all_numeric_dtypes() -> [DType; 14] {
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
            DType::F16,
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
        assert_eq!(promote(DType::I8, DType::F16), DType::F16);
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
        // Float16 follows NumPy's narrower integer thresholds
        assert_eq!(promote(DType::I8, DType::F16), DType::F16);
        assert_eq!(promote(DType::U8, DType::F16), DType::F16);
        assert_eq!(promote(DType::I16, DType::F16), DType::F32);
        assert_eq!(promote(DType::U16, DType::F16), DType::F32);
        assert_eq!(promote(DType::I32, DType::F16), DType::F64);
        assert_eq!(promote(DType::U64, DType::F16), DType::F64);
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
        assert!(can_cast_lossless(DType::I8, DType::F16));
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
        assert_eq!(DType::parse("float16"), Some(DType::F16));
        assert_eq!(DType::parse("float32"), Some(DType::F32));
        assert_eq!(DType::parse("float64"), Some(DType::F64));
    }

    #[test]
    fn parse_numpy_byte_width_shorthand() {
        // NumPy convention: iN/uN/fN where N is byte count, not bit count
        assert_eq!(DType::parse("i1"), Some(DType::I8));
        assert_eq!(DType::parse("i2"), Some(DType::I16));
        assert_eq!(DType::parse("i4"), Some(DType::I32));
        assert_eq!(DType::parse("i8"), Some(DType::I64)); // 8 bytes = int64
        assert_eq!(DType::parse("u1"), Some(DType::U8));
        assert_eq!(DType::parse("u2"), Some(DType::U16));
        assert_eq!(DType::parse("u4"), Some(DType::U32));
        assert_eq!(DType::parse("u8"), Some(DType::U64)); // 8 bytes = uint64
        assert_eq!(DType::parse("f2"), Some(DType::F16));
        assert_eq!(DType::parse("f4"), Some(DType::F32));
        assert_eq!(DType::parse("f8"), Some(DType::F64));
    }

    #[test]
    fn item_size_matches_expectations() {
        assert_eq!(DType::Bool.item_size(), 1);
        assert_eq!(DType::I8.item_size(), 1);
        assert_eq!(DType::U8.item_size(), 1);
        assert_eq!(DType::I16.item_size(), 2);
        assert_eq!(DType::U16.item_size(), 2);
        assert_eq!(DType::F16.item_size(), 2);
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
    fn lossless_cast_is_reflexive_for_non_numeric_dtypes() {
        for dtype in [DType::Str, DType::DateTime64, DType::TimeDelta64] {
            assert!(
                can_cast_lossless(dtype, dtype),
                "{dtype:?} must cast to itself"
            );
        }
    }

    #[test]
    fn lossless_cast_rejects_schema_erased_structured_dtype() {
        assert!(!can_cast_lossless(DType::Structured, DType::Structured));
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
        assert_eq!(promote_for_sum_reduction(DType::F16), DType::F16);
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
        assert_eq!(promote_for_mean_reduction(DType::F16), DType::F16);
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
        assert_eq!(result_type(&[DType::F16, DType::U8]), DType::F16);
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
        assert!(can_cast(DType::F16, DType::F32, "safe"));
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
    fn can_cast_safe_matches_numpy_category_boundaries() {
        assert!(can_cast(DType::Bool, DType::TimeDelta64, "safe"));
        assert!(can_cast(DType::I8, DType::F16, "safe"));
        assert!(can_cast(DType::I16, DType::F32, "safe"));
        assert!(can_cast(DType::I32, DType::F64, "safe"));
        assert!(can_cast(DType::I64, DType::F64, "safe"));
        assert!(can_cast(DType::U32, DType::I64, "safe"));
        assert!(can_cast(DType::U64, DType::F64, "safe"));
        assert!(can_cast(DType::F64, DType::Complex128, "safe"));
        assert!(can_cast(DType::I64, DType::Complex128, "safe"));
        assert!(can_cast(DType::U64, DType::Complex128, "safe"));
        assert!(can_cast(DType::F32, DType::Str, "safe"));

        assert!(!can_cast(DType::I32, DType::F32, "safe"));
        assert!(!can_cast(DType::U64, DType::I64, "safe"));
        assert!(!can_cast(DType::U64, DType::TimeDelta64, "safe"));
        assert!(!can_cast(DType::F64, DType::Complex64, "safe"));
        assert!(can_cast(DType::Complex64, DType::Str, "safe"));
        assert!(!can_cast(DType::DateTime64, DType::TimeDelta64, "safe"));
    }

    #[test]
    fn can_cast_same_kind_matches_numpy_category_boundaries() {
        assert!(can_cast(DType::Bool, DType::Complex128, "same_kind"));
        assert!(can_cast(DType::I64, DType::I8, "same_kind"));
        assert!(can_cast(DType::U64, DType::I8, "same_kind"));
        assert!(!can_cast(DType::I32, DType::U8, "same_kind"));
        assert!(can_cast(DType::I32, DType::Complex64, "same_kind"));
        assert!(can_cast(DType::F64, DType::F16, "same_kind"));
        assert!(can_cast(DType::F64, DType::Str, "same_kind"));
        assert!(can_cast(DType::Complex128, DType::Str, "same_kind"));
        assert!(can_cast(DType::U64, DType::TimeDelta64, "same_kind"));

        assert!(!can_cast(DType::Bool, DType::DateTime64, "same_kind"));
        assert!(!can_cast(DType::F16, DType::U8, "same_kind"));
        assert!(!can_cast(DType::Complex64, DType::F64, "same_kind"));
        assert!(!can_cast(DType::TimeDelta64, DType::Str, "same_kind"));
        assert!(!can_cast(
            DType::DateTime64,
            DType::TimeDelta64,
            "same_kind"
        ));
    }

    #[test]
    fn min_scalar_type_follows_numpy_float_scalar_thresholds() {
        assert_eq!(min_scalar_type(0.0), DType::F16);
        assert_eq!(min_scalar_type(1.0), DType::F16);
        assert_eq!(min_scalar_type(255.0), DType::F16);
        assert_eq!(min_scalar_type(-129.0), DType::F16);
        assert_eq!(min_scalar_type(0.5), DType::F16);
        assert_eq!(min_scalar_type(f64::NAN), DType::F16);
        assert_eq!(min_scalar_type(f64::INFINITY), DType::F16);
        assert_eq!(min_scalar_type(65_000.0), DType::F32);
        assert_eq!(min_scalar_type(-65_000.0), DType::F32);
        assert_eq!(min_scalar_type(1.0e10), DType::F32);
        assert_eq!(min_scalar_type(1.0e39), DType::F64);
    }

    #[test]
    fn common_type_promotes_to_float() {
        assert_eq!(common_type(&[DType::F16]), DType::F16);
        assert_eq!(common_type(&[DType::F16, DType::F32]), DType::F32);
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
        let (bits, eps, tiny, max, min_exp, max_exp) = finfo(DType::F16).unwrap();
        assert_eq!(bits, 16);
        assert!((eps - f64::from(f16::EPSILON)).abs() < 1e-15);
        assert_eq!(tiny, f64::from(f16::MIN_POSITIVE));
        assert_eq!(max, f64::from(f16::MAX));
        assert_eq!(min_exp, f16::MIN_EXP);
        assert_eq!(max_exp, f16::MAX_EXP);

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
        assert_eq!(bits, 32);
        assert!((eps - f64::from(f32::EPSILON)).abs() < 1e-15);

        // Complex128 has F64 components
        let (bits, eps, _, _, _, _) = finfo(DType::Complex128).unwrap();
        assert_eq!(bits, 64);
        assert!((eps - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn finfo_non_float_returns_none() {
        assert!(finfo(DType::I32).is_none());
        assert!(finfo(DType::Bool).is_none());
        assert!(finfo(DType::Str).is_none());
    }

    // ── ArrayStorage tests ──

    #[test]
    fn storage_dtype_roundtrip() {
        let dtypes = [
            DType::Bool,
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
            DType::Str,
        ];
        for dt in dtypes {
            let storage = ArrayStorage::zeros(dt, 5);
            assert_eq!(storage.dtype(), dt, "dtype mismatch for {}", dt.name());
            assert_eq!(storage.len(), 5);
            assert!(!storage.is_empty());
        }
    }

    #[test]
    fn storage_zeros_are_zero() {
        let storage = ArrayStorage::zeros(DType::F64, 3);
        for i in 0..3 {
            assert_eq!(storage.get_f64(i).unwrap(), 0.0);
        }
    }

    #[test]
    fn storage_f64_get_set() {
        let mut storage = ArrayStorage::F64(vec![1.5, 2.5, 3.5]);
        assert_eq!(storage.get_f64(0).unwrap(), 1.5);
        assert_eq!(storage.get_f64(2).unwrap(), 3.5);
        storage.set_f64(1, 99.0).unwrap();
        assert_eq!(storage.get_f64(1).unwrap(), 99.0);
    }

    #[test]
    fn storage_i64_roundtrip() {
        let mut storage = ArrayStorage::I64(vec![100, -200, 300]);
        assert_eq!(storage.get_f64(0).unwrap(), 100.0);
        assert_eq!(storage.get_f64(1).unwrap(), -200.0);
        storage.set_f64(2, 42.0).unwrap();
        assert_eq!(storage.get_f64(2).unwrap(), 42.0);
    }

    #[test]
    fn storage_bool_roundtrip() {
        let mut storage = ArrayStorage::Bool(vec![true, false, true]);
        assert_eq!(storage.get_f64(0).unwrap(), 1.0);
        assert_eq!(storage.get_f64(1).unwrap(), 0.0);
        storage.set_f64(1, 5.0).unwrap(); // nonzero -> true
        assert_eq!(storage.get_f64(1).unwrap(), 1.0);
        storage.set_f64(0, 0.0).unwrap(); // zero -> false
        assert_eq!(storage.get_f64(0).unwrap(), 0.0);
    }

    #[test]
    fn storage_complex128_roundtrip() {
        let storage = ArrayStorage::Complex128(vec![(3.0, 4.0), (1.0, -2.0)]);
        // get_f64 returns real part
        assert_eq!(storage.get_f64(0).unwrap(), 3.0);
        assert_eq!(storage.get_f64(1).unwrap(), 1.0);
    }

    #[test]
    fn storage_u8_roundtrip() {
        let mut storage = ArrayStorage::U8(vec![0, 128, 255]);
        assert_eq!(storage.get_f64(0).unwrap(), 0.0);
        assert_eq!(storage.get_f64(1).unwrap(), 128.0);
        assert_eq!(storage.get_f64(2).unwrap(), 255.0);
        storage.set_f64(0, 42.0).unwrap();
        assert_eq!(storage.get_f64(0).unwrap(), 42.0);
    }

    #[test]
    fn storage_f32_roundtrip() {
        let storage = ArrayStorage::F32(vec![1.25, 2.75]);
        assert!((storage.get_f64(0).unwrap() - 1.25).abs() < 1e-6);
        assert!((storage.get_f64(1).unwrap() - 2.75).abs() < 1e-6);
    }

    #[test]
    fn storage_f16_roundtrip() {
        let mut storage = ArrayStorage::F16(vec![f16::from_f32(0.1), f16::from_f32(1.5)]);
        assert!((storage.get_f64(0).unwrap() - 0.099_975_585_937_5).abs() < 1e-15);
        assert_eq!(storage.get_f64(1).unwrap(), 1.5);
        storage.set_f64(1, 2.25).unwrap();
        assert_eq!(storage.get_f64(1).unwrap(), 2.25);
    }

    #[test]
    fn storage_out_of_bounds() {
        let storage = ArrayStorage::F64(vec![1.0, 2.0]);
        let err = storage.get_f64(5).unwrap_err();
        assert_eq!(err, StorageError::IndexOutOfBounds { index: 5, len: 2 });
    }

    #[test]
    fn storage_cast_f64_to_i32() {
        let storage = ArrayStorage::F64(vec![1.0, 2.5, -3.7]);
        let cast = storage.cast_to(DType::I32).unwrap();
        assert_eq!(cast.dtype(), DType::I32);
        assert_eq!(cast.get_f64(0).unwrap(), 1.0);
        assert_eq!(cast.get_f64(1).unwrap(), 2.0); // truncated
        assert_eq!(cast.get_f64(2).unwrap(), -3.0); // truncated
    }

    #[test]
    fn storage_cast_f64_to_f16_quantizes_like_numpy() {
        let storage = ArrayStorage::F64(vec![0.1, 1.5, 65_504.0]);
        let cast = storage.cast_to(DType::F16).unwrap();
        assert_eq!(cast.dtype(), DType::F16);
        assert!((cast.get_f64(0).unwrap() - 0.099_975_585_937_5).abs() < 1e-15);
        assert_eq!(cast.get_f64(1).unwrap(), 1.5);
        assert_eq!(cast.get_f64(2).unwrap(), 65_504.0);
    }

    #[test]
    fn storage_cast_i64_to_f64() {
        let storage = ArrayStorage::I64(vec![10, 20, 30]);
        let cast = storage.cast_to(DType::F64).unwrap();
        assert_eq!(cast.dtype(), DType::F64);
        assert_eq!(cast.get_f64(0).unwrap(), 10.0);
        assert_eq!(cast.get_f64(2).unwrap(), 30.0);
    }

    #[test]
    fn storage_cast_same_dtype_is_clone() {
        let storage = ArrayStorage::F64(vec![1.0, 2.0, 3.0]);
        let cast = storage.cast_to(DType::F64).unwrap();
        assert_eq!(storage, cast);
    }

    #[test]
    fn storage_cast_to_string() {
        let storage = ArrayStorage::Bool(vec![true, false]);
        let cast = storage.cast_to(DType::Str).unwrap();
        assert_eq!(cast.dtype(), DType::Str);
        assert!(
            matches!(&cast, ArrayStorage::String(_)),
            "expected String storage"
        );
        if let ArrayStorage::String(v) = &cast {
            assert_eq!(v[0], "True");
            assert_eq!(v[1], "False");
        }
    }

    #[test]
    fn storage_cast_string_to_numeric_fails() {
        let storage = ArrayStorage::String(vec!["hello".into()]);
        let err = storage.cast_to(DType::F64).unwrap_err();
        assert_eq!(
            err,
            StorageError::UnsupportedCast {
                from: DType::Str,
                to: DType::F64
            }
        );
    }

    #[test]
    fn storage_to_f64_vec() {
        let storage = ArrayStorage::I32(vec![10, 20, 30]);
        assert_eq!(storage.to_f64_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn storage_from_f64_vec() {
        let storage = ArrayStorage::from_f64_vec(vec![1.0, 2.0]);
        assert_eq!(storage.dtype(), DType::F64);
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn storage_empty() {
        let storage = ArrayStorage::zeros(DType::F64, 0);
        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);
    }

    // ── Complex storage tests ──

    #[test]
    fn storage_from_complex128_vec() {
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        assert_eq!(s.dtype(), DType::Complex128);
        assert_eq!(s.len(), 2);
        assert!(s.is_complex());
    }

    #[test]
    fn storage_from_complex64_vec() {
        let s = ArrayStorage::from_complex64_vec(vec![(1.0_f32, 2.0), (3.0, 4.0)]);
        assert_eq!(s.dtype(), DType::Complex64);
        assert!(s.is_complex());
    }

    #[test]
    fn storage_real_is_not_complex() {
        let s = ArrayStorage::from_f64_vec(vec![1.0, 2.0]);
        assert!(!s.is_complex());
    }

    #[test]
    fn storage_get_set_complex128() {
        let mut s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        assert_eq!(s.get_complex128(0).unwrap(), (1.0, 2.0));
        assert_eq!(s.get_complex128(1).unwrap(), (3.0, 4.0));
        s.set_complex128(0, 5.0, 6.0).unwrap();
        assert_eq!(s.get_complex128(0).unwrap(), (5.0, 6.0));
    }

    #[test]
    fn storage_get_complex128_from_real() {
        let s = ArrayStorage::from_f64_vec(vec![7.0, 8.0]);
        assert_eq!(s.get_complex128(0).unwrap(), (7.0, 0.0));
        assert_eq!(s.get_complex128(1).unwrap(), (8.0, 0.0));
    }

    #[test]
    fn storage_get_complex128_out_of_bounds() {
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0)]);
        assert!(s.get_complex128(1).is_err());
    }

    #[test]
    fn storage_to_complex128_vec() {
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        assert_eq!(s.to_complex128_vec(), vec![(1.0, 2.0), (3.0, 4.0)]);
    }

    #[test]
    fn storage_to_complex128_vec_from_real() {
        let s = ArrayStorage::from_f64_vec(vec![5.0, 6.0]);
        assert_eq!(s.to_complex128_vec(), vec![(5.0, 0.0), (6.0, 0.0)]);
    }

    #[test]
    fn storage_complex_add() {
        let a = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        let b = ArrayStorage::from_complex128_vec(vec![(5.0, 6.0), (7.0, 8.0)]);
        let c = a.complex_add(&b).unwrap();
        assert_eq!(c.to_complex128_vec(), vec![(6.0, 8.0), (10.0, 12.0)]);
    }

    #[test]
    fn storage_complex_sub() {
        let a = ArrayStorage::from_complex128_vec(vec![(5.0, 6.0), (7.0, 8.0)]);
        let b = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        let c = a.complex_sub(&b).unwrap();
        assert_eq!(c.to_complex128_vec(), vec![(4.0, 4.0), (4.0, 4.0)]);
    }

    #[test]
    fn storage_complex_mul() {
        // (1+2i)*(3+4i) = (1*3-2*4) + (1*4+2*3)i = -5 + 10i
        let a = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0)]);
        let b = ArrayStorage::from_complex128_vec(vec![(3.0, 4.0)]);
        let c = a.complex_mul(&b).unwrap();
        let v = c.to_complex128_vec();
        assert!((v[0].0 - (-5.0)).abs() < 1e-10);
        assert!((v[0].1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_div() {
        // (1+2i)/(3+4i) = (1*3+2*4)/(9+16) + (2*3-1*4)/(9+16)i = 11/25 + 2/25 i
        let a = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0)]);
        let b = ArrayStorage::from_complex128_vec(vec![(3.0, 4.0)]);
        let c = a.complex_div(&b).unwrap();
        let v = c.to_complex128_vec();
        assert!((v[0].0 - 11.0 / 25.0).abs() < 1e-10);
        assert!((v[0].1 - 2.0 / 25.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_div_by_zero() {
        let a = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0)]);
        let b = ArrayStorage::from_complex128_vec(vec![(0.0, 0.0)]);
        let c = a.complex_div(&b).unwrap();
        let v = c.to_complex128_vec();
        assert!(v[0].0.is_nan());
        assert!(v[0].1.is_nan());
    }

    #[test]
    fn storage_complex_conjugate() {
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, -4.0)]);
        let c = s.complex_conjugate();
        assert_eq!(c.to_complex128_vec(), vec![(1.0, -2.0), (3.0, 4.0)]);
    }

    #[test]
    fn storage_complex_conjugate_real_is_identity() {
        let s = ArrayStorage::from_f64_vec(vec![5.0, 6.0]);
        let c = s.complex_conjugate();
        assert_eq!(c.to_f64_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn storage_complex_abs() {
        // |3+4i| = 5
        let s = ArrayStorage::from_complex128_vec(vec![(3.0, 4.0), (0.0, 1.0)]);
        let a = s.complex_abs();
        let v = a.to_f64_vec();
        assert!((v[0] - 5.0).abs() < 1e-10);
        assert!((v[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_abs_real() {
        let s = ArrayStorage::from_f64_vec(vec![-3.0, 4.0]);
        let a = s.complex_abs();
        let v = a.to_f64_vec();
        assert!((v[0] - 3.0).abs() < 1e-10);
        assert!((v[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_angle() {
        // angle(1+1i) = pi/4
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 1.0), (0.0, 1.0), (-1.0, 0.0)]);
        let a = s.complex_angle();
        let v = a.to_f64_vec();
        assert!((v[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((v[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((v[2] - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_angle_real() {
        let s = ArrayStorage::from_f64_vec(vec![1.0, -1.0]);
        let a = s.complex_angle();
        let v = a.to_f64_vec();
        assert!((v[0]).abs() < 1e-10); // positive → 0
        assert!((v[1] - std::f64::consts::PI).abs() < 1e-10); // negative → pi
    }

    #[test]
    fn storage_complex_real_imag() {
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        let re = s.complex_real();
        let im = s.complex_imag();
        assert_eq!(re.to_f64_vec(), vec![1.0, 3.0]);
        assert_eq!(im.to_f64_vec(), vec![2.0, 4.0]);
    }

    #[test]
    fn storage_complex_real_imag_from_real() {
        let s = ArrayStorage::from_f64_vec(vec![5.0, 6.0]);
        let re = s.complex_real();
        let im = s.complex_imag();
        assert_eq!(re.to_f64_vec(), vec![5.0, 6.0]);
        assert_eq!(im.to_f64_vec(), vec![0.0, 0.0]);
    }

    #[test]
    fn storage_complex_exp() {
        // exp(0+0i) = 1+0i
        let s = ArrayStorage::from_complex128_vec(vec![(0.0, 0.0)]);
        let e = s.complex_exp();
        let v = e.to_complex128_vec();
        assert!((v[0].0 - 1.0).abs() < 1e-10);
        assert!(v[0].1.abs() < 1e-10);
    }

    #[test]
    fn storage_complex_exp_pure_imaginary() {
        // exp(i*pi) = -1 + 0i (Euler's formula)
        let s = ArrayStorage::from_complex128_vec(vec![(0.0, std::f64::consts::PI)]);
        let e = s.complex_exp();
        let v = e.to_complex128_vec();
        assert!((v[0].0 - (-1.0)).abs() < 1e-10);
        assert!(v[0].1.abs() < 1e-10);
    }

    #[test]
    fn storage_complex_log() {
        // log(1+0i) = 0+0i
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 0.0)]);
        let l = s.complex_log();
        let v = l.to_complex128_vec();
        assert!(v[0].0.abs() < 1e-10);
        assert!(v[0].1.abs() < 1e-10);
    }

    #[test]
    fn storage_complex_log_negative_real() {
        // log(-1) = 0 + pi*i
        let s = ArrayStorage::from_complex128_vec(vec![(-1.0, 0.0)]);
        let l = s.complex_log();
        let v = l.to_complex128_vec();
        assert!(v[0].0.abs() < 1e-10);
        assert!((v[0].1 - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_sqrt() {
        // sqrt(4+0i) = 2+0i
        let s = ArrayStorage::from_complex128_vec(vec![(4.0, 0.0)]);
        let r = s.complex_sqrt();
        let v = r.to_complex128_vec();
        assert!((v[0].0 - 2.0).abs() < 1e-10);
        assert!(v[0].1.abs() < 1e-10);
    }

    #[test]
    fn storage_complex_sqrt_negative() {
        // sqrt(-1+0i) = 0+1i
        let s = ArrayStorage::from_complex128_vec(vec![(-1.0, 0.0)]);
        let r = s.complex_sqrt();
        let v = r.to_complex128_vec();
        assert!(v[0].0.abs() < 1e-10);
        assert!((v[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_pow() {
        // (1+1i)^2 = 0+2i
        let base = ArrayStorage::from_complex128_vec(vec![(1.0, 1.0)]);
        let exp = ArrayStorage::from_complex128_vec(vec![(2.0, 0.0)]);
        let r = base.complex_pow(&exp);
        let v = r.to_complex128_vec();
        assert!(v[0].0.abs() < 1e-10);
        assert!((v[0].1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_sin() {
        // sin(0+0i) = 0+0i
        let s = ArrayStorage::from_complex128_vec(vec![(0.0, 0.0)]);
        let r = s.complex_sin();
        let v = r.to_complex128_vec();
        assert!(v[0].0.abs() < 1e-10);
        assert!(v[0].1.abs() < 1e-10);
    }

    #[test]
    fn storage_complex_cos() {
        // cos(0+0i) = 1+0i
        let s = ArrayStorage::from_complex128_vec(vec![(0.0, 0.0)]);
        let r = s.complex_cos();
        let v = r.to_complex128_vec();
        assert!((v[0].0 - 1.0).abs() < 1e-10);
        assert!(v[0].1.abs() < 1e-10);
    }

    #[test]
    fn storage_complex_sum() {
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        let (sr, si) = s.complex_sum();
        assert!((sr - 4.0).abs() < 1e-10);
        assert!((si - 6.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_prod() {
        // (1+2i)*(3+4i) = -5+10i
        let s = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        let (pr, pi) = s.complex_prod();
        assert!((pr - (-5.0)).abs() < 1e-10);
        assert!((pi - 10.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex_add_length_mismatch() {
        let a = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0)]);
        let b = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0), (3.0, 4.0)]);
        assert!(a.complex_add(&b).is_err());
    }

    #[test]
    fn storage_complex_mixed_types() {
        // Add complex + real
        let a = ArrayStorage::from_complex128_vec(vec![(1.0, 2.0)]);
        let b = ArrayStorage::from_f64_vec(vec![3.0]);
        let c = a.complex_add(&b).unwrap();
        let v = c.to_complex128_vec();
        assert!((v[0].0 - 4.0).abs() < 1e-10);
        assert!((v[0].1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn storage_complex64_to_complex128_conversion() {
        let s = ArrayStorage::from_complex64_vec(vec![(1.5_f32, 2.5)]);
        let v = s.to_complex128_vec();
        assert!((v[0].0 - 1.5).abs() < 1e-6);
        assert!((v[0].1 - 2.5).abs() < 1e-6);
    }

    #[test]
    fn storage_complex64_conjugate() {
        let s = ArrayStorage::from_complex64_vec(vec![(1.0_f32, 2.0)]);
        let c = s.complex_conjugate();
        assert_eq!(c.dtype(), DType::Complex64);
        let v = c.to_complex128_vec();
        assert!((v[0].0 - 1.0).abs() < 1e-6);
        assert!((v[0].1 - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn storage_complex_sin_pure_imaginary() {
        // sin(i*y) = i*sinh(y); for y=1: sin(i) = i*sinh(1)
        let s = ArrayStorage::from_complex128_vec(vec![(0.0, 1.0)]);
        let r = s.complex_sin();
        let v = r.to_complex128_vec();
        assert!(v[0].0.abs() < 1e-10); // real part should be 0
        assert!((v[0].1 - 1.0_f64.sinh()).abs() < 1e-10); // imag part = sinh(1)
    }

    #[test]
    fn storage_complex_exp_log_roundtrip() {
        let s = ArrayStorage::from_complex128_vec(vec![(2.0, 3.0)]);
        let e = s.complex_exp();
        let l = e.complex_log();
        let v = l.to_complex128_vec();
        assert!((v[0].0 - 2.0).abs() < 1e-10);
        assert!((v[0].1 - 3.0).abs() < 1e-10);
    }

    // ── Structured storage tests ──

    #[test]
    fn structured_storage_empty() {
        let s = StructuredStorage::empty();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.num_fields(), 0);
        assert!(s.field_names().is_empty());
    }

    #[test]
    fn structured_storage_new_basic() {
        let fields = vec![
            StructuredField {
                name: "x".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "y".into(),
                dtype: DType::I32,
                offset: 8,
            },
        ];
        let columns = vec![
            ArrayStorage::F64(vec![1.0, 2.0, 3.0]),
            ArrayStorage::I32(vec![10, 20, 30]),
        ];
        let s = StructuredStorage::new(fields, columns).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.num_fields(), 2);
        assert_eq!(s.field_names(), vec!["x", "y"]);
    }

    #[test]
    fn structured_storage_field_access() {
        let fields = vec![
            StructuredField {
                name: "name".into(),
                dtype: DType::Str,
                offset: 0,
            },
            StructuredField {
                name: "value".into(),
                dtype: DType::F64,
                offset: 0,
            },
        ];
        let columns = vec![
            ArrayStorage::String(vec!["alice".into(), "bob".into()]),
            ArrayStorage::F64(vec![42.0, 99.0]),
        ];
        let s = StructuredStorage::new(fields, columns).unwrap();
        let name_col = s.get_field("name").unwrap();
        assert_eq!(name_col.dtype(), DType::Str);
        assert_eq!(name_col.len(), 2);
        let val_col = s.get_field("value").unwrap();
        assert_eq!(val_col.get_f64(0).unwrap(), 42.0);
        assert_eq!(val_col.get_f64(1).unwrap(), 99.0);
    }

    #[test]
    fn structured_storage_field_index() {
        let fields = vec![
            StructuredField {
                name: "a".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "b".into(),
                dtype: DType::I64,
                offset: 8,
            },
        ];
        let columns = vec![ArrayStorage::F64(vec![1.0]), ArrayStorage::I64(vec![2])];
        let s = StructuredStorage::new(fields, columns).unwrap();
        assert_eq!(s.field_index("a"), Some(0));
        assert_eq!(s.field_index("b"), Some(1));
        assert_eq!(s.field_index("c"), None);
    }

    #[test]
    fn structured_storage_field_by_index() {
        let fields = vec![StructuredField {
            name: "x".into(),
            dtype: DType::F64,
            offset: 0,
        }];
        let columns = vec![ArrayStorage::F64(vec![7.0, 8.0])];
        let s = StructuredStorage::new(fields, columns).unwrap();
        let col = s.get_field_by_index(0).unwrap();
        assert_eq!(col.get_f64(0).unwrap(), 7.0);
        assert!(s.get_field_by_index(1).is_none());
    }

    #[test]
    fn structured_storage_mismatched_column_lengths() {
        let fields = vec![
            StructuredField {
                name: "a".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "b".into(),
                dtype: DType::F64,
                offset: 8,
            },
        ];
        let columns = vec![
            ArrayStorage::F64(vec![1.0, 2.0]),
            ArrayStorage::F64(vec![3.0]), // wrong length
        ];
        assert!(StructuredStorage::new(fields, columns).is_err());
    }

    #[test]
    fn structured_storage_mismatched_field_count() {
        let fields = vec![StructuredField {
            name: "a".into(),
            dtype: DType::F64,
            offset: 0,
        }];
        let columns = vec![ArrayStorage::F64(vec![1.0]), ArrayStorage::F64(vec![2.0])];
        assert!(StructuredStorage::new(fields, columns).is_err());
    }

    #[test]
    fn structured_storage_dtype_mismatch() {
        let fields = vec![
            StructuredField {
                name: "x".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "y".into(),
                dtype: DType::I64,
                offset: 8,
            },
        ];
        let columns = vec![
            ArrayStorage::F64(vec![1.0]),
            ArrayStorage::F64(vec![2.0]), // expected I64
        ];
        assert!(StructuredStorage::new(fields, columns).is_err());
    }

    #[test]
    fn structured_storage_first_column_dtype_mismatch() {
        let fields = vec![StructuredField {
            name: "x".into(),
            dtype: DType::I64,
            offset: 0,
        }];
        let columns = vec![ArrayStorage::F64(vec![1.0])];
        assert_eq!(
            StructuredStorage::new(fields, columns).unwrap_err(),
            StorageError::UnsupportedCast {
                from: DType::F64,
                to: DType::I64,
            }
        );
    }

    #[test]
    fn structured_storage_record_size() {
        let fields = vec![
            StructuredField {
                name: "x".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "y".into(),
                dtype: DType::I32,
                offset: 8,
            },
            StructuredField {
                name: "z".into(),
                dtype: DType::Bool,
                offset: 12,
            },
        ];
        let columns = vec![
            ArrayStorage::F64(vec![1.0]),
            ArrayStorage::I32(vec![2]),
            ArrayStorage::Bool(vec![true]),
        ];
        let s = StructuredStorage::new(fields, columns).unwrap();
        assert_eq!(s.record_size(), 8 + 4 + 1); // f64 + i32 + bool = 13
    }

    #[test]
    fn structured_storage_record_size_respects_offsets() {
        let fields = vec![
            StructuredField {
                name: "x".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "y".into(),
                dtype: DType::I32,
                offset: 16,
            },
        ];
        let columns = vec![ArrayStorage::F64(vec![1.0]), ArrayStorage::I32(vec![2])];
        let s = StructuredStorage::new(fields, columns).unwrap();
        assert_eq!(s.record_size(), 20);
    }

    #[test]
    fn structured_storage_dtype_str() {
        let fields = vec![
            StructuredField {
                name: "x".into(),
                dtype: DType::F64,
                offset: 0,
            },
            StructuredField {
                name: "y".into(),
                dtype: DType::I32,
                offset: 8,
            },
        ];
        let columns = vec![ArrayStorage::F64(vec![1.0]), ArrayStorage::I32(vec![2])];
        let s = StructuredStorage::new(fields, columns).unwrap();
        assert_eq!(s.dtype_str(), "[('x', 'f64'), ('y', 'i32')]");
    }

    #[test]
    fn structured_storage_as_array_storage() {
        let fields = vec![StructuredField {
            name: "v".into(),
            dtype: DType::F64,
            offset: 0,
        }];
        let columns = vec![ArrayStorage::F64(vec![1.0, 2.0])];
        let ss = StructuredStorage::new(fields, columns).unwrap();
        let storage = ArrayStorage::Structured(ss);
        assert_eq!(storage.dtype(), DType::Structured);
        assert_eq!(storage.len(), 2);
        // get_f64 returns NaN for structured
        assert!(storage.get_f64(0).unwrap().is_nan());
    }

    #[test]
    fn structured_dtype_name() {
        assert_eq!(DType::Structured.name(), "void");
    }

    #[test]
    fn structured_dtype_item_size() {
        assert_eq!(DType::Structured.item_size(), 1);
    }

    #[test]
    fn structured_storage_zeros() {
        let s = ArrayStorage::zeros(DType::Structured, 0);
        assert_eq!(s.dtype(), DType::Structured);
        assert!(s.is_empty());
    }

    #[test]
    fn structured_storage_zeros_preserve_requested_len() {
        let s = ArrayStorage::zeros(DType::Structured, 3);
        assert_eq!(s.dtype(), DType::Structured);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert!(
            matches!(&s, ArrayStorage::Structured(_)),
            "expected structured storage"
        );
        if let ArrayStorage::Structured(storage) = s {
            assert_eq!(storage.len(), 3);
            assert_eq!(storage.num_fields(), 0);
        }
    }

    #[test]
    fn storage_cast_numeric_to_structured_fails() {
        let storage = ArrayStorage::F64(vec![1.0]);
        assert_eq!(
            storage.cast_to(DType::Structured).unwrap_err(),
            StorageError::UnsupportedCast {
                from: DType::F64,
                to: DType::Structured,
            }
        );
    }

    #[test]
    fn storage_cast_structured_to_numeric_fails() {
        let fields = vec![StructuredField {
            name: "x".into(),
            dtype: DType::F64,
            offset: 0,
        }];
        let columns = vec![ArrayStorage::F64(vec![1.0])];
        let storage = ArrayStorage::Structured(StructuredStorage::new(fields, columns).unwrap());
        assert_eq!(
            storage.cast_to(DType::F64).unwrap_err(),
            StorageError::UnsupportedCast {
                from: DType::Structured,
                to: DType::F64,
            }
        );
    }

    #[test]
    fn structured_storage_mixed_types() {
        let fields = vec![
            StructuredField {
                name: "id".into(),
                dtype: DType::U32,
                offset: 0,
            },
            StructuredField {
                name: "score".into(),
                dtype: DType::F64,
                offset: 4,
            },
            StructuredField {
                name: "label".into(),
                dtype: DType::Str,
                offset: 12,
            },
        ];
        let columns = vec![
            ArrayStorage::U32(vec![1, 2, 3]),
            ArrayStorage::F64(vec![0.5, 0.8, 0.3]),
            ArrayStorage::String(vec!["cat".into(), "dog".into(), "fish".into()]),
        ];
        let s = StructuredStorage::new(fields, columns).unwrap();
        assert_eq!(s.len(), 3);
        let ids = s.get_field("id").unwrap();
        assert_eq!(ids.get_f64(2).unwrap(), 3.0);
        let labels = s.get_field("label").unwrap();
        assert_eq!(labels.dtype(), DType::Str);
    }

    // ── F64 + integer explicit promotion tests ────────

    #[test]
    fn promote_f64_f64_and_f32_f64_cross() {
        // These three pairs were previously implicit (caught by _ => F64 fallback).
        assert_eq!(promote(DType::F64, DType::F64), DType::F64);
        assert_eq!(promote(DType::F32, DType::F64), DType::F64);
        assert_eq!(promote(DType::F64, DType::F32), DType::F64);
    }

    #[test]
    fn promote_f64_with_all_integers() {
        let ints = [
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
        ];
        for &i in &ints {
            assert_eq!(promote(DType::F64, i), DType::F64, "F64 + {i:?}");
            assert_eq!(promote(i, DType::F64), DType::F64, "{i:?} + F64");
        }
    }

    // ── DateTime64/TimeDelta64 + numeric fail-closed tests ────────

    #[test]
    fn promote_datetime64_with_numerics_is_f64() {
        // NumPy raises TypeError for these; we fail-closed to F64.
        let numerics = [
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
        ];
        for &n in &numerics {
            assert_eq!(
                promote(DType::DateTime64, n),
                DType::F64,
                "DateTime64 + {n:?}"
            );
            assert_eq!(
                promote(n, DType::DateTime64),
                DType::F64,
                "{n:?} + DateTime64"
            );
        }
    }

    #[test]
    fn promote_timedelta64_with_numerics_is_f64() {
        // NumPy raises TypeError for these; we fail-closed to F64.
        let numerics = [
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
        ];
        for &n in &numerics {
            assert_eq!(
                promote(DType::TimeDelta64, n),
                DType::F64,
                "TimeDelta64 + {n:?}"
            );
            assert_eq!(
                promote(n, DType::TimeDelta64),
                DType::F64,
                "{n:?} + TimeDelta64"
            );
        }
    }

    #[test]
    fn promote_datetime_timedelta_same_family() {
        assert_eq!(
            promote(DType::DateTime64, DType::DateTime64),
            DType::DateTime64
        );
        assert_eq!(
            promote(DType::TimeDelta64, DType::TimeDelta64),
            DType::TimeDelta64
        );
        assert_eq!(
            promote(DType::DateTime64, DType::TimeDelta64),
            DType::DateTime64
        );
        assert_eq!(
            promote(DType::TimeDelta64, DType::DateTime64),
            DType::DateTime64
        );
    }

    #[test]
    fn promote_structured_with_anything_is_f64() {
        let dtypes = [
            DType::I8,
            DType::F64,
            DType::Complex128,
            DType::DateTime64,
            DType::TimeDelta64,
            DType::Structured,
        ];
        for &d in &dtypes {
            assert_eq!(
                promote(DType::Structured, d),
                DType::F64,
                "Structured + {d:?}"
            );
            assert_eq!(
                promote(d, DType::Structured),
                DType::F64,
                "{d:?} + Structured"
            );
        }
    }

    #[test]
    fn promote_all_324_pairs_are_explicit() {
        // Exhaustively test all DType pairs to ensure no pair panics
        // and the catch-all is effectively unreachable.
        let all = [
            DType::Bool,
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
            DType::Str,
            DType::DateTime64,
            DType::TimeDelta64,
            DType::Structured,
        ];
        for &lhs in &all {
            for &rhs in &all {
                let _ = promote(lhs, rhs); // should not panic
            }
        }
    }
}
