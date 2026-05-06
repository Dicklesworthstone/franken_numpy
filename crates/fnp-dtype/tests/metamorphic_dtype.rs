//! Metamorphic tests for fnp-dtype type operations.
//!
//! Tests invariants and mathematical properties that must hold regardless of
//! specific input values.

use fnp_dtype::{DType, can_cast, common_type, min_scalar_type, result_type};

const ALL_DTYPES: &[DType] = &[
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
];

// ─────────────────────────────────────────────────────────────────────────────
// result_type metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_result_type_permutation_invariance() {
    for &a in ALL_DTYPES {
        for &b in ALL_DTYPES {
            let ab = result_type(&[a, b]);
            let ba = result_type(&[b, a]);
            assert_eq!(
                ab, ba,
                "result_type([{a:?}, {b:?}]) != result_type([{b:?}, {a:?}])"
            );
        }
    }
}

#[test]
fn mr_result_type_idempotence() {
    for &dtype in ALL_DTYPES {
        let result = result_type(&[dtype, dtype]);
        assert_eq!(
            result, dtype,
            "result_type([{dtype:?}, {dtype:?}]) should be {dtype:?}, got {result:?}"
        );
    }
}

#[test]
fn mr_result_type_left_associativity() {
    // result_type uses left-fold: ((init op a) op b) op c
    // Verify left-associative grouping is consistent
    for &a in &[DType::I32, DType::F32, DType::U8] {
        for &b in &[DType::I64, DType::F64, DType::Bool] {
            for &c in &[DType::I16, DType::U16, DType::F16] {
                let abc = result_type(&[a, b, c]);
                let ab_c = result_type(&[result_type(&[a, b]), c]);
                // Left-associativity should hold: ((a op b) op c) == fold([a,b,c])
                assert_eq!(
                    abc, ab_c,
                    "result_type is not left-associative: [{a:?},{b:?},{c:?}] gave {abc:?} but ((a,b),c) gave {ab_c:?}"
                );
            }
        }
    }
}

#[test]
fn mr_result_type_single_element() {
    for &dtype in ALL_DTYPES {
        let result = result_type(&[dtype]);
        assert_eq!(
            result, dtype,
            "result_type([{dtype:?}]) should be {dtype:?}, got {result:?}"
        );
    }
}

#[test]
fn mr_result_type_triple_permutation() {
    for &a in &[DType::I32, DType::F32, DType::U8] {
        for &b in &[DType::I64, DType::F64, DType::Bool] {
            for &c in &[DType::I16, DType::U16, DType::F16] {
                let abc = result_type(&[a, b, c]);
                let acb = result_type(&[a, c, b]);
                let bac = result_type(&[b, a, c]);
                let bca = result_type(&[b, c, a]);
                let cab = result_type(&[c, a, b]);
                let cba = result_type(&[c, b, a]);
                assert_eq!(abc, acb);
                assert_eq!(abc, bac);
                assert_eq!(abc, bca);
                assert_eq!(abc, cab);
                assert_eq!(abc, cba);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// can_cast metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_can_cast_reflexivity() {
    for &dtype in ALL_DTYPES {
        for casting in ["no", "equiv", "safe", "same_kind", "unsafe"] {
            assert!(
                can_cast(dtype, dtype, casting),
                "can_cast({dtype:?}, {dtype:?}, {casting:?}) should be true"
            );
        }
    }
}

#[test]
fn mr_can_cast_unsafe_is_universal() {
    for &from in ALL_DTYPES {
        for &to in ALL_DTYPES {
            assert!(
                can_cast(from, to, "unsafe"),
                "can_cast({from:?}, {to:?}, \"unsafe\") should always be true"
            );
        }
    }
}

#[test]
fn mr_can_cast_safe_implies_same_kind() {
    for &from in ALL_DTYPES {
        for &to in ALL_DTYPES {
            if can_cast(from, to, "safe") {
                assert!(
                    can_cast(from, to, "same_kind"),
                    "if can_cast({from:?}, {to:?}, \"safe\") then same_kind should also be true"
                );
            }
        }
    }
}

#[test]
fn mr_can_cast_same_kind_implies_unsafe() {
    for &from in ALL_DTYPES {
        for &to in ALL_DTYPES {
            if can_cast(from, to, "same_kind") {
                assert!(
                    can_cast(from, to, "unsafe"),
                    "if can_cast({from:?}, {to:?}, \"same_kind\") then unsafe should also be true"
                );
            }
        }
    }
}

#[test]
fn mr_can_cast_equiv_implies_safe() {
    for &from in ALL_DTYPES {
        for &to in ALL_DTYPES {
            if can_cast(from, to, "equiv") {
                assert!(
                    can_cast(from, to, "safe"),
                    "if can_cast({from:?}, {to:?}, \"equiv\") then safe should also be true"
                );
            }
        }
    }
}

#[test]
fn mr_can_cast_no_equals_equiv() {
    for &from in ALL_DTYPES {
        for &to in ALL_DTYPES {
            assert_eq!(
                can_cast(from, to, "no"),
                can_cast(from, to, "equiv"),
                "can_cast({from:?}, {to:?}, \"no\") should equal can_cast(..., \"equiv\")"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// common_type metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_common_type_permutation_invariance() {
    for &a in ALL_DTYPES {
        for &b in ALL_DTYPES {
            let ab = common_type(&[a, b]);
            let ba = common_type(&[b, a]);
            assert_eq!(
                ab, ba,
                "common_type([{a:?}, {b:?}]) != common_type([{b:?}, {a:?}])"
            );
        }
    }
}

#[test]
fn mr_common_type_idempotence_floats() {
    // common_type promotes integers to F64, so only check float types for idempotence
    for &dtype in &[
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let result = common_type(&[dtype, dtype]);
        assert_eq!(
            result, dtype,
            "common_type([{dtype:?}, {dtype:?}]) should be {dtype:?}, got {result:?}"
        );
    }
}

#[test]
fn mr_common_type_integers_become_f64() {
    // common_type promotes integers to F64 per numpy semantics
    for &dtype in &[
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let result = common_type(&[dtype]);
        assert_eq!(
            result,
            DType::F64,
            "common_type([{dtype:?}]) should be F64 (integer promotion), got {result:?}"
        );
    }
}

#[test]
fn mr_common_type_single_float() {
    for &dtype in &[
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
        DType::Complex128,
    ] {
        let result = common_type(&[dtype]);
        assert_eq!(
            result, dtype,
            "common_type([{dtype:?}]) should be {dtype:?}, got {result:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// min_scalar_type metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_min_scalar_type_zero() {
    let result = min_scalar_type(0.0);
    assert!(
        matches!(result, DType::F16 | DType::F32 | DType::F64),
        "min_scalar_type(0.0) should be a float type, got {result:?}"
    );
}

#[test]
fn mr_min_scalar_type_negation_invariance() {
    for val in [1.0, 100.0, 1e10, 1e-10, 0.5] {
        let pos = min_scalar_type(val);
        let neg = min_scalar_type(-val);
        assert_eq!(
            pos, neg,
            "min_scalar_type({val}) should equal min_scalar_type({}) but got {pos:?} vs {neg:?}",
            -val
        );
    }
}

#[test]
fn mr_min_scalar_type_returns_float() {
    for val in [0.0, 1.0, -1.0, 1e30, 1e-30, f64::MAX, f64::MIN_POSITIVE] {
        let result = min_scalar_type(val);
        assert!(
            matches!(result, DType::F16 | DType::F32 | DType::F64),
            "min_scalar_type({val}) should return a float type, got {result:?}"
        );
    }
}

#[test]
fn mr_min_scalar_type_special_values() {
    let inf = min_scalar_type(f64::INFINITY);
    let neg_inf = min_scalar_type(f64::NEG_INFINITY);
    let nan = min_scalar_type(f64::NAN);

    assert!(
        matches!(inf, DType::F16 | DType::F32 | DType::F64),
        "min_scalar_type(INF) should be float, got {inf:?}"
    );
    assert!(
        matches!(neg_inf, DType::F16 | DType::F32 | DType::F64),
        "min_scalar_type(-INF) should be float, got {neg_inf:?}"
    );
    assert!(
        matches!(nan, DType::F16 | DType::F32 | DType::F64),
        "min_scalar_type(NAN) should be float, got {nan:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-function metamorphic relations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mr_result_type_consistency_with_can_cast_safe() {
    for &a in ALL_DTYPES {
        for &b in ALL_DTYPES {
            let result = result_type(&[a, b]);
            assert!(
                can_cast(a, result, "safe") || can_cast(a, result, "same_kind"),
                "result_type([{a:?}, {b:?}]) = {result:?} but cannot cast {a:?} to {result:?}"
            );
            assert!(
                can_cast(b, result, "safe") || can_cast(b, result, "same_kind"),
                "result_type([{a:?}, {b:?}]) = {result:?} but cannot cast {b:?} to {result:?}"
            );
        }
    }
}
