//! Regression tests from fuzzing and edge case probing.

use fnp_ufunc::{DateTimeUnit, parse_fixed_signature_string, parse_gufunc_signature};

// ─────────────────────────────────────────────────────────────────────────────
// DateTimeUnit::parse edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn datetime_parse_valid_units() {
    assert!(DateTimeUnit::parse("w").is_ok());
    assert!(DateTimeUnit::parse("d").is_ok());
    assert!(DateTimeUnit::parse("h").is_ok());
    assert!(DateTimeUnit::parse("m").is_ok());
    assert!(DateTimeUnit::parse("s").is_ok());
    assert!(DateTimeUnit::parse("ms").is_ok());
    assert!(DateTimeUnit::parse("us").is_ok());
    assert!(DateTimeUnit::parse("ns").is_ok());
}

#[test]
fn datetime_parse_case_insensitive() {
    assert!(DateTimeUnit::parse("W").is_ok());
    assert!(DateTimeUnit::parse("D").is_ok());
    assert!(DateTimeUnit::parse("MS").is_ok());
    assert!(DateTimeUnit::parse("Us").is_ok());
}

#[test]
fn datetime_parse_with_whitespace() {
    assert!(DateTimeUnit::parse("  w  ").is_ok());
    assert!(DateTimeUnit::parse("\tms\n").is_ok());
}

#[test]
fn datetime_parse_invalid() {
    assert!(DateTimeUnit::parse("").is_err());
    assert!(DateTimeUnit::parse("x").is_err());
    assert!(DateTimeUnit::parse("weeks").is_err());
    assert!(DateTimeUnit::parse("日").is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_gufunc_signature edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gufunc_signature_none_both() {
    let result = parse_gufunc_signature(None, None);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn gufunc_signature_simple() {
    let result = parse_gufunc_signature(Some("(n),(n)->()"), None);
    assert!(result.is_ok());
    let sig = result.unwrap().unwrap();
    assert_eq!(sig.canonical(), "(n),(n)->()");
}

#[test]
fn gufunc_signature_empty_string_rejected() {
    let result = parse_gufunc_signature(Some(""), None);
    assert!(
        result.is_err(),
        "empty string should be rejected as invalid signature"
    );
}

#[test]
fn gufunc_signature_whitespace_only_rejected() {
    let result = parse_gufunc_signature(Some("   "), None);
    assert!(
        result.is_err(),
        "whitespace-only should be rejected as invalid signature"
    );
}

#[test]
fn gufunc_signature_missing_arrow() {
    let result = parse_gufunc_signature(Some("(n),(n)()"), None);
    assert!(result.is_err());
}

#[test]
fn gufunc_signature_multiple_arrows() {
    let result = parse_gufunc_signature(Some("(n)->()->(m)"), None);
    assert!(result.is_err());
}

#[test]
fn gufunc_signature_unicode() {
    let result = parse_gufunc_signature(Some("(α),(β)->(γ)"), None);
    assert!(result.is_err());
}

#[test]
fn gufunc_signature_deep_nesting() {
    let result = parse_gufunc_signature(Some("((((n))))->()"), None);
    assert!(result.is_err());
}

#[test]
fn gufunc_signature_long_input() {
    let long_sig = format!("({})->()", "a,".repeat(100));
    let result = parse_gufunc_signature(Some(&long_sig), None);
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn gufunc_signature_conflict_same_value() {
    let result = parse_gufunc_signature(Some("(n)->()"), Some("(n)->()"));
    assert!(
        result.is_err(),
        "providing both sig and signature should error even if same"
    );
}

#[test]
fn gufunc_signature_conflict_different_values() {
    let result = parse_gufunc_signature(Some("(n)->()"), Some("(m)->()"));
    assert!(
        result.is_err(),
        "providing both sig and signature should error"
    );
}

#[test]
fn gufunc_signature_via_signature_param() {
    let result = parse_gufunc_signature(None, Some("(n),(n)->()"));
    assert!(result.is_ok());
    let sig = result.unwrap().unwrap();
    assert_eq!(sig.canonical(), "(n),(n)->()");
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_fixed_signature_string edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fixed_signature_simple() {
    let result = parse_fixed_signature_string("dd->d", 2, 1);
    assert!(result.is_ok());
}

#[test]
fn fixed_signature_empty() {
    let result = parse_fixed_signature_string("", 0, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_whitespace_only() {
    let result = parse_fixed_signature_string("   ", 0, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_no_arrow() {
    let result = parse_fixed_signature_string("ddd", 2, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_multiple_arrows() {
    let result = parse_fixed_signature_string("d->d->d", 1, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_zero_nout() {
    let result = parse_fixed_signature_string("d->", 1, 0);
    assert!(result.is_err() || result.is_ok());
}

#[test]
fn fixed_signature_mismatched_nin() {
    let result = parse_fixed_signature_string("dd->d", 3, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_mismatched_nout() {
    let result = parse_fixed_signature_string("d->dd", 1, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_unicode_chars() {
    let result = parse_fixed_signature_string("日->日", 1, 1);
    assert!(result.is_err());
}

#[test]
fn fixed_signature_long_types() {
    let long_sig = format!("{}->d", "d".repeat(100));
    let result = parse_fixed_signature_string(&long_sig, 100, 1);
    assert!(result.is_ok() || result.is_err());
}
