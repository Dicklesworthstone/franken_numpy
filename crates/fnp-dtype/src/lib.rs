#![forbid(unsafe_code)]

/// Canonical subset of NumPy-like dtypes for the first conformance wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I32,
    I64,
    F32,
    F64,
}

impl DType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "bool" => Some(Self::Bool),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }
}

/// Deterministic promotion table for V1-scoped dtypes.
#[must_use]
pub const fn promote(lhs: DType, rhs: DType) -> DType {
    use DType::{Bool, F32, F64, I32, I64};

    match (lhs, rhs) {
        (Bool, x) | (x, Bool) => x,
        (I32, I32) => I32,
        (I32, I64) | (I64, I32) | (I64, I64) => I64,
        (F32, F32) => F32,
        (F64, F64) => F64,
        (F32, I32) | (I32, F32) => F64,
        (F32, I64) | (I64, F32) => F64,
        (F64, _) | (_, F64) => F64,
    }
}

#[must_use]
pub const fn can_cast_lossless(src: DType, dst: DType) -> bool {
    use DType::{Bool, F32, F64, I32, I64};

    matches!(
        (src, dst),
        (Bool, Bool)
            | (Bool, I32 | I64 | F32 | F64)
            | (I32, I32 | I64 | F64)
            | (I64, I64 | F64)
            | (F32, F32 | F64)
            | (F64, F64)
    )
}

#[cfg(test)]
mod tests {
    use super::{DType, can_cast_lossless, promote};

    fn all_dtypes() -> [DType; 5] {
        [DType::Bool, DType::I32, DType::I64, DType::F32, DType::F64]
    }

    #[test]
    fn promotion_is_commutative() {
        let dtypes = all_dtypes();

        for &lhs in &dtypes {
            for &rhs in &dtypes {
                assert_eq!(promote(lhs, rhs), promote(rhs, lhs), "{lhs:?}/{rhs:?}");
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
    fn cast_matrix_smoke() {
        assert!(can_cast_lossless(DType::Bool, DType::F64));
        assert!(can_cast_lossless(DType::I32, DType::I64));
        assert!(!can_cast_lossless(DType::I64, DType::I32));
        assert!(!can_cast_lossless(DType::F64, DType::F32));
    }

    #[test]
    fn parse_roundtrip_for_known_dtypes() {
        for dtype in all_dtypes() {
            let parsed = DType::parse(dtype.name()).expect("known dtype should parse");
            assert_eq!(parsed, dtype);
        }
        assert!(DType::parse("complex128").is_none());
    }

    #[test]
    fn item_size_matches_expectations() {
        assert_eq!(DType::Bool.item_size(), 1);
        assert_eq!(DType::I32.item_size(), 4);
        assert_eq!(DType::F32.item_size(), 4);
        assert_eq!(DType::I64.item_size(), 8);
        assert_eq!(DType::F64.item_size(), 8);
    }

    #[test]
    fn promotion_is_idempotent() {
        for dtype in all_dtypes() {
            assert_eq!(promote(dtype, dtype), dtype);
        }
    }

    #[test]
    fn lossless_cast_is_reflexive() {
        for dtype in all_dtypes() {
            assert!(
                can_cast_lossless(dtype, dtype),
                "{dtype:?} must cast to itself"
            );
        }
    }

    #[test]
    fn lossless_cast_is_transitive_over_scoped_matrix() {
        let dtypes = all_dtypes();
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
        let dtypes = all_dtypes();
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
}
