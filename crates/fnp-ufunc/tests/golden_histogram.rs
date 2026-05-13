//! Golden artifact tests for histogram and binning operations.
//!
//! These tests capture expected output for histogram, bincount, digitize,
//! and searchsorted operations, detecting any regressions in behavior.

use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use fnp_dtype::DType;
use fnp_ufunc::UFuncArray;

#[test]
fn histogram_golden_matches() -> Result<(), Box<dyn Error>> {
    let actual = render_histogram_snapshot()?;
    assert_golden("histogram", &actual)
}

fn render_histogram_snapshot() -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("# fnp-ufunc histogram/binning golden v1\n\n");

    // ────────────────────────────────────────────────────────────────────────
    // Basic histogram
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## histogram basic\n");

    let data = UFuncArray::new(
        vec![10],
        vec![0.5, 1.2, 2.3, 3.1, 4.5, 5.8, 6.2, 7.9, 8.1, 9.5],
        DType::F64,
    )?;
    let (counts, edges) = data.histogram(5)?;
    push_array(&mut out, "hist_5bins_counts", &counts)?;
    push_array(&mut out, "hist_5bins_edges", &edges)?;

    let uniform = UFuncArray::linspace(0.0, 10.0, 21, DType::F64)?;
    let (counts2, edges2) = uniform.histogram(10)?;
    push_array(&mut out, "hist_uniform_counts", &counts2)?;
    push_array(&mut out, "hist_uniform_edges", &edges2)?;

    // ────────────────────────────────────────────────────────────────────────
    // Histogram with explicit edges
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## histogram with edges\n");

    let edges_input = UFuncArray::new(vec![4], vec![0.0, 2.5, 5.0, 10.0], DType::F64)?;
    let (counts_edges, edges_out) = data.histogram_edges(&edges_input)?;
    push_array(&mut out, "hist_custom_edges_counts", &counts_edges)?;
    push_array(&mut out, "hist_custom_edges_edges", &edges_out)?;

    // ────────────────────────────────────────────────────────────────────────
    // Histogram with auto bin strategies
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## histogram auto strategies\n");

    let large_data = UFuncArray::linspace(0.0, 100.0, 101, DType::F64)?;

    let (counts_auto, _edges_auto) = large_data.histogram_auto("auto")?;
    push_scalar(&mut out, "hist_auto_nbins", counts_auto.shape()[0] as f64)?;

    let (counts_sturges, _edges_sturges) = large_data.histogram_auto("sturges")?;
    push_scalar(
        &mut out,
        "hist_sturges_nbins",
        counts_sturges.shape()[0] as f64,
    )?;

    let (counts_sqrt, _edges_sqrt) = large_data.histogram_auto("sqrt")?;
    push_scalar(&mut out, "hist_sqrt_nbins", counts_sqrt.shape()[0] as f64)?;

    // ────────────────────────────────────────────────────────────────────────
    // Bincount
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## bincount\n");

    let indices = UFuncArray::new(
        vec![8],
        vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0],
        DType::I64,
    )?;
    let bc = indices.bincount()?;
    push_array(&mut out, "bincount_basic", &bc)?;

    let weights = UFuncArray::new(
        vec![8],
        vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.1, 0.2],
        DType::F64,
    )?;
    let bc_weighted = indices.bincount_with(Some(&weights), 0)?;
    push_array(&mut out, "bincount_weighted", &bc_weighted)?;

    let bc_minlength = indices.bincount_with(Some(&weights), 10)?;
    push_array(&mut out, "bincount_minlength_10", &bc_minlength)?;

    // ────────────────────────────────────────────────────────────────────────
    // Digitize
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## digitize\n");

    let values = UFuncArray::new(vec![7], vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], DType::F64)?;
    let bins = UFuncArray::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0], DType::F64)?;
    let dig = values.digitize(&bins)?;
    push_array(&mut out, "digitize_left", &dig)?;

    let dig_right = values.digitize_right(&bins, true)?;
    push_array(&mut out, "digitize_right", &dig_right)?;

    // Edge cases
    let edge_values = UFuncArray::new(vec![4], vec![-1.0, 1.0, 5.0, 10.0], DType::F64)?;
    let dig_edges = edge_values.digitize(&bins)?;
    push_array(&mut out, "digitize_edge_cases", &dig_edges)?;

    // ────────────────────────────────────────────────────────────────────────
    // Searchsorted
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## searchsorted\n");

    let sorted_arr = UFuncArray::new(vec![6], vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0], DType::F64)?;
    let search_vals = UFuncArray::new(vec![5], vec![0.0, 2.0, 5.0, 8.0, 12.0], DType::F64)?;

    let ss_left = sorted_arr.searchsorted(&search_vals, Some("left"), None)?;
    push_array(&mut out, "searchsorted_left", &ss_left)?;

    let ss_right = sorted_arr.searchsorted(&search_vals, Some("right"), None)?;
    push_array(&mut out, "searchsorted_right", &ss_right)?;

    // Exact matches
    let exact_vals = UFuncArray::new(vec![3], vec![1.0, 5.0, 11.0], DType::F64)?;
    let ss_exact_left = sorted_arr.searchsorted(&exact_vals, Some("left"), None)?;
    let ss_exact_right = sorted_arr.searchsorted(&exact_vals, Some("right"), None)?;
    push_array(&mut out, "searchsorted_exact_left", &ss_exact_left)?;
    push_array(&mut out, "searchsorted_exact_right", &ss_exact_right)?;

    // ────────────────────────────────────────────────────────────────────────
    // Histogram with range
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## histogram with range\n");

    let (counts_range, edges_range) = data.histogram_full(5, Some((2.0, 8.0)), None, false)?;
    push_array(&mut out, "hist_range_2_8_counts", &counts_range)?;
    push_array(&mut out, "hist_range_2_8_edges", &edges_range)?;

    // ────────────────────────────────────────────────────────────────────────
    // Edge cases
    // ────────────────────────────────────────────────────────────────────────
    out.push_str("## edge cases\n");

    // Single element
    let single = UFuncArray::new(vec![1], vec![5.0], DType::F64)?;
    let (counts_single, edges_single) = single.histogram(3)?;
    push_array(&mut out, "hist_single_element_counts", &counts_single)?;
    push_array(&mut out, "hist_single_element_edges", &edges_single)?;

    // All same value
    let same = UFuncArray::full(vec![10], 3.0, DType::F64)?;
    let (counts_same, edges_same) = same.histogram(5)?;
    push_array(&mut out, "hist_same_value_counts", &counts_same)?;
    push_array(&mut out, "hist_same_value_edges", &edges_same)?;

    // Empty bincount result for sparse indices
    let sparse = UFuncArray::new(vec![3], vec![0.0, 5.0, 10.0], DType::I64)?;
    let bc_sparse = sparse.bincount()?;
    push_array(&mut out, "bincount_sparse", &bc_sparse)?;

    if out.ends_with("\n\n") {
        out.pop();
    }
    Ok(out)
}

fn push_array(out: &mut String, label: &str, array: &UFuncArray) -> Result<(), Box<dyn Error>> {
    let storage = array.to_storage()?;
    writeln!(out, "{label}")?;
    writeln!(out, "  shape: {}", format_shape(array.shape()))?;
    writeln!(out, "  dtype: {:?}", storage.dtype())?;
    writeln!(out, "  values: {}", format_values(array.values()))?;
    out.push('\n');
    Ok(())
}

fn push_scalar(out: &mut String, label: &str, value: f64) -> Result<(), Box<dyn Error>> {
    writeln!(out, "{label}")?;
    writeln!(out, "  value: {}", format_float(value))?;
    out.push('\n');
    Ok(())
}

fn format_shape(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_values(values: &[f64]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| format_float(*v))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn format_float(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e10 {
        format!("{:.1}", v)
    } else {
        format!("{:.6}", v)
    }
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.golden"))
}

fn assert_golden(name: &str, actual: &str) -> Result<(), Box<dyn Error>> {
    let path = golden_path(name);

    if std::env::var("UPDATE_GOLDEN").is_ok() {
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(&path, actual)?;
        eprintln!("Updated golden file: {}", path.display());
        return Ok(());
    }

    let expected = fs::read_to_string(&path).map_err(|e| {
        format!(
            "Golden file not found: {}. Run with UPDATE_GOLDEN=1 to create it. Error: {}",
            path.display(),
            e
        )
    })?;

    if actual != expected {
        let diff = diff_strings(&expected, actual);
        return Err(format!(
            "Golden mismatch for '{name}':\n{diff}\n\nRun with UPDATE_GOLDEN=1 to update."
        )
        .into());
    }

    Ok(())
}

fn diff_strings(expected: &str, actual: &str) -> String {
    let mut diff = String::new();
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();

    for (i, (e, a)) in expected_lines.iter().zip(actual_lines.iter()).enumerate() {
        if e != a {
            writeln!(diff, "Line {}: expected '{}', got '{}'", i + 1, e, a).ok();
        }
    }

    if expected_lines.len() != actual_lines.len() {
        writeln!(
            diff,
            "Line count: expected {}, got {}",
            expected_lines.len(),
            actual_lines.len()
        )
        .ok();
    }

    diff
}
