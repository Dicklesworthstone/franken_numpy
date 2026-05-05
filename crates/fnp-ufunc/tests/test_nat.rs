//! Tests for NaT (Not a Time) parsing and handling in datetime64/timedelta64.
//!
//! Finding: This test file had ZERO assertions — only a println. Fixed to verify:
//! - NaT string parsing for datetime64
//! - NaT string parsing for timedelta64
//! - isnat() correctly identifies NaT values
//! - datetime_as_string formats NaT correctly
//! - NaT values are distinct from valid dates

use fnp_ufunc::{UFuncArray, datetime_as_string, isnat};

#[test]
fn nat_parsing_datetime64_succeeds() {
    let result = UFuncArray::from_datetime_strings(vec![1], vec!["NaT".to_string()], None);
    assert!(
        result.is_ok(),
        "NaT should parse successfully for datetime64"
    );
    let arr = result.unwrap();
    assert_eq!(arr.shape(), &[1]);

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    assert_eq!(
        isnat_result.values()[0],
        1.0,
        "NaT should be detected by isnat()"
    );
}

#[test]
fn nat_parsing_timedelta64_succeeds() {
    let result = UFuncArray::from_timedelta_strings(vec![1], vec!["NaT".to_string()], None);
    assert!(
        result.is_ok(),
        "NaT should parse successfully for timedelta64"
    );
    let arr = result.unwrap();
    assert_eq!(arr.shape(), &[1]);

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    assert_eq!(
        isnat_result.values()[0],
        1.0,
        "NaT should be detected by isnat()"
    );
}

#[test]
fn isnat_identifies_nat_values() {
    let arr = UFuncArray::from_datetime_strings(
        vec![3],
        vec![
            "2024-01-15".to_string(),
            "NaT".to_string(),
            "2024-06-30".to_string(),
        ],
        Some("D"),
    )
    .expect("datetime parsing should succeed");

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    assert_eq!(isnat_result.shape(), &[3]);

    let values = isnat_result.values();
    assert_eq!(values[0], 0.0, "valid date should not be NaT");
    assert_eq!(values[1], 1.0, "NaT value should be detected as NaT");
    assert_eq!(values[2], 0.0, "valid date should not be NaT");
}

#[test]
fn datetime_as_string_formats_nat_correctly() {
    let arr = UFuncArray::from_datetime_strings(
        vec![2],
        vec!["2024-03-14".to_string(), "NaT".to_string()],
        Some("D"),
    )
    .expect("datetime parsing should succeed");

    let strings = datetime_as_string(&arr, None).expect("formatting should succeed");
    let values = strings.values();
    assert_eq!(values.len(), 2);
    assert!(
        values[0].starts_with("2024-03-14"),
        "valid date should format as ISO string"
    );
    assert_eq!(values[1], "NaT", "NaT should format as 'NaT' string");
}

#[test]
fn nat_mixed_with_valid_dates_preserves_all() {
    let input_values = vec![
        "NaT".to_string(),
        "2020-01-01".to_string(),
        "NaT".to_string(),
        "2020-12-31".to_string(),
        "NaT".to_string(),
    ];
    let arr = UFuncArray::from_datetime_strings(vec![5], input_values, Some("D"))
        .expect("mixed NaT and valid dates should parse");

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    let is_nat = isnat_result.values();
    assert_eq!(is_nat, &[1.0, 0.0, 1.0, 0.0, 1.0]);

    let strings = datetime_as_string(&arr, None).expect("formatting should succeed");
    let str_values = strings.values();
    assert_eq!(str_values[0], "NaT");
    assert!(str_values[1].starts_with("2020-01-01"));
    assert_eq!(str_values[2], "NaT");
    assert!(str_values[3].starts_with("2020-12-31"));
    assert_eq!(str_values[4], "NaT");
}

#[test]
fn nat_timedelta_mixed_with_valid() {
    let values = vec![
        "NaT".to_string(),
        "1D".to_string(),
        "NaT".to_string(),
        "30D".to_string(),
    ];
    let arr = UFuncArray::from_timedelta_strings(vec![4], values, Some("D"))
        .expect("mixed NaT and valid timedeltas should parse");

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    assert_eq!(isnat_result.values(), &[1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn valid_date_is_not_nat() {
    let arr = UFuncArray::from_datetime_strings(vec![1], vec!["2024-07-04".to_string()], Some("D"))
        .expect("valid date should parse");

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    assert_eq!(
        isnat_result.values()[0],
        0.0,
        "valid date should NOT be NaT"
    );
}

#[test]
fn valid_timedelta_is_not_nat() {
    let arr = UFuncArray::from_timedelta_strings(vec![1], vec!["42D".to_string()], Some("D"))
        .expect("valid timedelta should parse");

    let isnat_result = isnat(&arr).expect("isnat should succeed");
    assert_eq!(
        isnat_result.values()[0],
        0.0,
        "valid timedelta should NOT be NaT"
    );
}
