//! Golden tests for fnp-io text I/O functions.
//!
//! Tests loadtxt, genfromtxt, fromstring, fromfile_text, tofile_text
//! against known-good inputs and outputs.
//!
//! Golden artifacts embedded as string literals to avoid external file
//! dependencies. Each test verifies exact output matches expected values.

use fnp_io::{
    IOSupportedDType, fromfile_text, fromstring, genfromtxt, loadtxt, loadtxt_unpack,
    loadtxt_usecols, tofile_text,
};

const EPSILON: f64 = 1e-14;

fn approx_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() < EPSILON
}

fn assert_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: got {}, expected {}",
        context,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e),
            "{}: value mismatch at index {}: got {}, expected {}",
            context,
            i,
            a,
            e
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// loadtxt golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_loadtxt_simple_integers() {
    let input = "1 2 3\n4 5 6\n7 8 9\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_close(&result.values, &expected, "loadtxt simple integers");
    assert_eq!(result.nrows, 3);
    assert_eq!(result.ncols, 3);
}

#[test]
fn golden_loadtxt_floats() {
    let input = "1.5 2.5 3.5\n4.5 5.5 6.5\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
    assert_close(&result.values, &expected, "loadtxt floats");
}

#[test]
fn golden_loadtxt_scientific_notation() {
    let input = "1e-3 2e+4 3.14e2\n-1.5e-2 0 1e10\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![1e-3, 2e4, 3.14e2, -1.5e-2, 0.0, 1e10];
    assert_close(&result.values, &expected, "loadtxt scientific notation");
}

#[test]
fn golden_loadtxt_comma_delimiter() {
    let input = "1,2,3\n4,5,6\n";
    let result = loadtxt(input, ',', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "loadtxt comma delimiter");
}

#[test]
fn golden_loadtxt_tab_delimiter() {
    let input = "1\t2\t3\n4\t5\t6\n";
    let result = loadtxt(input, '\t', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "loadtxt tab delimiter");
}

#[test]
fn golden_loadtxt_skip_header() {
    let input = "# header line\n# another header\n1 2 3\n4 5 6\n";
    let result = loadtxt(input, ' ', '#', 2, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "loadtxt skip header");
}

#[test]
fn golden_loadtxt_max_rows() {
    let input = "1 2\n3 4\n5 6\n7 8\n";
    let result = loadtxt(input, ' ', '#', 0, 2).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0];
    assert_close(&result.values, &expected, "loadtxt max rows");
    assert_eq!(result.nrows, 2);
}

#[test]
fn golden_loadtxt_comment_lines() {
    let input = "1 2 3\n# comment\n4 5 6\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "loadtxt comment lines");
}

#[test]
fn golden_loadtxt_negative_values() {
    let input = "-1 -2 -3\n-4.5 -5.5 -6.5\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![-1.0, -2.0, -3.0, -4.5, -5.5, -6.5];
    assert_close(&result.values, &expected, "loadtxt negative values");
}

#[test]
fn golden_loadtxt_single_column() {
    let input = "1\n2\n3\n4\n5\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_close(&result.values, &expected, "loadtxt single column");
    assert_eq!(result.ncols, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// loadtxt_usecols golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_loadtxt_usecols_select_first() {
    let input = "1 2 3\n4 5 6\n7 8 9\n";
    let result = loadtxt_usecols(input, ' ', '#', 0, usize::MAX, Some(&[0])).unwrap();
    let expected = vec![1.0, 4.0, 7.0];
    assert_close(&result.values, &expected, "loadtxt_usecols first column");
}

#[test]
fn golden_loadtxt_usecols_select_last() {
    let input = "1 2 3\n4 5 6\n7 8 9\n";
    let result = loadtxt_usecols(input, ' ', '#', 0, usize::MAX, Some(&[2])).unwrap();
    let expected = vec![3.0, 6.0, 9.0];
    assert_close(&result.values, &expected, "loadtxt_usecols last column");
}

#[test]
fn golden_loadtxt_usecols_multiple() {
    let input = "1 2 3 4\n5 6 7 8\n";
    let result = loadtxt_usecols(input, ' ', '#', 0, usize::MAX, Some(&[0, 2])).unwrap();
    let expected = vec![1.0, 3.0, 5.0, 7.0];
    assert_close(
        &result.values,
        &expected,
        "loadtxt_usecols multiple columns",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// loadtxt_unpack golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_loadtxt_unpack_transpose() {
    let input = "1 2 3\n4 5 6\n";
    let result = loadtxt_unpack(input, ' ', '#', 0, usize::MAX, None, true).unwrap();
    // unpack transposes: nrows becomes ncols
    assert_eq!(result.nrows, 3);
    assert_eq!(result.ncols, 2);
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_close(&result.values, &expected, "loadtxt_unpack transpose");
}

// ─────────────────────────────────────────────────────────────────────────────
// genfromtxt golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_genfromtxt_basic() {
    let input = "1 2 3\n4 5 6\n";
    let result = genfromtxt(input, ' ', '#', 0, f64::NAN).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "genfromtxt basic");
}

#[test]
fn golden_genfromtxt_with_missing_nan() {
    let input = "1,2,3\n4,,6\n";
    let result = genfromtxt(input, ',', '#', 0, f64::NAN).unwrap();
    assert_eq!(result.nrows, 2, "should have 2 rows");
    assert_eq!(result.ncols, 3, "should have 3 columns");
    assert!(
        approx_eq(result.values[0], 1.0),
        "first element should be 1.0"
    );
    assert!(result.values[4].is_nan(), "missing element should be NaN");
    assert!(
        approx_eq(result.values[5], 6.0),
        "last element should be 6.0"
    );
}

#[test]
fn golden_genfromtxt_skip_header() {
    let input = "col1 col2 col3\n1 2 3\n4 5 6\n";
    let result = genfromtxt(input, ' ', '#', 1, f64::NAN).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "genfromtxt skip header");
}

// ─────────────────────────────────────────────────────────────────────────────
// fromstring golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_fromstring_space_sep() {
    let input = b"1 2 3 4 5";
    let result = fromstring(input, IOSupportedDType::F64, " ").unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_close(&result, &expected, "fromstring space sep");
}

#[test]
fn golden_fromstring_comma_sep() {
    let input = b"1.5,2.5,3.5";
    let result = fromstring(input, IOSupportedDType::F64, ",").unwrap();
    let expected = vec![1.5, 2.5, 3.5];
    assert_close(&result, &expected, "fromstring comma sep");
}

#[test]
fn golden_fromstring_scientific() {
    let input = b"1e-10 2e+5 3.14e0";
    let result = fromstring(input, IOSupportedDType::F64, " ").unwrap();
    let expected = vec![1e-10, 2e5, 314.0_f64 / 100.0];
    assert_close(&result, &expected, "fromstring scientific");
}

// ─────────────────────────────────────────────────────────────────────────────
// fromfile_text / tofile_text golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_fromfile_text_roundtrip() {
    let original = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let text = tofile_text(&original, " ");
    let recovered = fromfile_text(&text, " ", None).unwrap();
    assert_close(&recovered, &original, "fromfile_text roundtrip");
}

#[test]
fn golden_tofile_text_format() {
    let values = vec![1.0, 2.0, 3.0];
    let text = tofile_text(&values, ",");
    assert!(text.contains(","), "should use comma separator");
    let parts: Vec<&str> = text.split(',').collect();
    assert_eq!(parts.len(), 3, "should have 3 values");
}

#[test]
fn golden_fromfile_text_with_count() {
    let text = "1 2 3 4 5 6 7 8 9 10";
    let result = fromfile_text(text, " ", Some(5)).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_close(&result, &expected, "fromfile_text with count");
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge case golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_loadtxt_trailing_whitespace() {
    let input = "1 2 3   \n4 5 6  \n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_close(&result.values, &expected, "loadtxt trailing whitespace");
}

#[test]
fn golden_loadtxt_inf_values() {
    let input = "inf -inf 0\n1 inf -inf\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    assert!(
        result.values[0].is_infinite() && result.values[0] > 0.0,
        "should be +inf"
    );
    assert!(
        result.values[1].is_infinite() && result.values[1] < 0.0,
        "should be -inf"
    );
    assert!(
        result.values[4].is_infinite() && result.values[4] > 0.0,
        "should be +inf"
    );
}

#[test]
fn golden_loadtxt_nan_values() {
    let input = "nan 1 2\n3 nan 4\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    assert!(result.values[0].is_nan(), "first element should be NaN");
    assert!(result.values[4].is_nan(), "fifth element should be NaN");
    assert!(
        approx_eq(result.values[1], 1.0),
        "second element should be 1.0"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dimension tracking golden tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn golden_loadtxt_dimensions_tracked() {
    let input = "1 2 3 4\n5 6 7 8\n9 10 11 12\n";
    let result = loadtxt(input, ' ', '#', 0, usize::MAX).unwrap();
    assert_eq!(result.nrows, 3, "should have 3 rows");
    assert_eq!(result.ncols, 4, "should have 4 columns");
    assert_eq!(result.values.len(), 12, "should have 12 values");
}

#[test]
fn golden_genfromtxt_dimensions() {
    let input = "1 2\n3 4\n5 6\n7 8\n";
    let result = genfromtxt(input, ' ', '#', 0, f64::NAN).unwrap();
    assert_eq!(result.nrows, 4, "should have 4 rows");
    assert_eq!(result.ncols, 2, "should have 2 columns");
}
