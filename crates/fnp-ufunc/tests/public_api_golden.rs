use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use fnp_dtype::DType;
use fnp_ufunc::{AxisSlice, BinaryOp, UFuncArray, UnaryOp, unique_all, where_nonzero};

#[test]
fn public_api_output_matches_golden() -> Result<(), Box<dyn Error>> {
    let actual = render_public_api_snapshot()?;
    assert_golden("public_api", &actual)
}

fn render_public_api_snapshot() -> Result<String, Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("# fnp-ufunc public API golden v1\n\n");

    out.push_str("## ndarray creation\n");
    push_array(
        &mut out,
        "zeros_2x3_f64",
        &UFuncArray::zeros(vec![2, 3], DType::F64)?,
    )?;
    push_array(
        &mut out,
        "ones_3_i64",
        &UFuncArray::ones(vec![3], DType::I64)?,
    )?;
    push_array(
        &mut out,
        "full_2x2_f64",
        &UFuncArray::full(vec![2, 2], -3.5, DType::F64)?,
    )?;
    push_array(
        &mut out,
        "arange_even_f64",
        &UFuncArray::arange(0.0, 6.0, 2.0, DType::F64)?,
    )?;
    push_array(
        &mut out,
        "linspace_quarters_f64",
        &UFuncArray::linspace(0.0, 1.0, 5, DType::F64)?,
    )?;

    out.push_str("## broadcasting\n");
    let column = UFuncArray::new(vec![2, 1], vec![1.0, 2.0], DType::F64)?;
    let row = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64)?;
    push_array(
        &mut out,
        "column_plus_row",
        &column.elementwise_binary(&row, BinaryOp::Add)?,
    )?;
    push_array(
        &mut out,
        "broadcast_1x3_to_2x3",
        &UFuncArray::new(vec![1, 3], vec![4.0, 5.0, 6.0], DType::F64)?.broadcast_to(&[2, 3])?,
    )?;

    out.push_str("## ufunc results\n");
    let lhs = UFuncArray::new(vec![4], vec![1.0, -2.0, 3.5, -4.5], DType::F64)?;
    let rhs = UFuncArray::new(vec![4], vec![2.0, 4.0, -2.0, -3.0], DType::F64)?;
    push_array(
        &mut out,
        "binary_multiply",
        &lhs.elementwise_binary(&rhs, BinaryOp::Mul)?,
    )?;
    push_array(
        &mut out,
        "binary_divide",
        &lhs.elementwise_binary(&rhs, BinaryOp::Div)?,
    )?;
    push_array(
        &mut out,
        "unary_negative",
        &lhs.elementwise_unary(UnaryOp::Negative),
    )?;
    push_array(
        &mut out,
        "unary_sqrt",
        &UFuncArray::new(vec![4], vec![0.0, 1.0, 4.0, 9.0], DType::F64)?
            .elementwise_unary(UnaryOp::Sqrt),
    )?;
    push_array(
        &mut out,
        "logical_not_nan_truthiness",
        &UFuncArray::new(vec![3], vec![0.0, 1.0, f64::NAN], DType::F64)?
            .elementwise_unary(UnaryOp::LogicalNot),
    )?;

    out.push_str("## reductions\n");
    let matrix = UFuncArray::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64)?;
    push_array(&mut out, "sum_all", &matrix.reduce_sum(None, false)?)?;
    push_array(&mut out, "sum_axis0", &matrix.reduce_sum(Some(0), false)?)?;
    push_array(
        &mut out,
        "sum_axis1_keepdims",
        &matrix.reduce_sum(Some(1), true)?,
    )?;
    push_array(&mut out, "mean_axis0", &matrix.reduce_mean(Some(0), false)?)?;
    push_array(&mut out, "min_axis1", &matrix.reduce_min(Some(1), false)?)?;
    push_array(&mut out, "max_all", &matrix.reduce_max(None, false)?)?;

    out.push_str("## slicing and indexing\n");
    let grid = UFuncArray::new(vec![3, 4], (0..12).map(f64::from).collect(), DType::F64)?;
    push_scalar(&mut out, "item_2_1", grid.item(&[2, 1])?)?;

    let axis_view = grid.slice_axis_view(1, Some(1), Some(4), 2)?;
    push_array(
        &mut out,
        "slice_axis_cols_1_to_4_step_2",
        &UFuncArray::from_shared_view(&axis_view)?,
    )?;

    let strided_view = grid.slice_view(&[
        AxisSlice::new(0, Some(2), None, -1),
        AxisSlice::new(1, Some(0), Some(4), 2),
    ])?;
    push_array(
        &mut out,
        "slice_rows_reverse_cols_even",
        &UFuncArray::from_shared_view(&strided_view)?,
    )?;

    out.push_str("## shape transforms and joins\n");
    let small = UFuncArray::new(vec![2, 3], (1..=6).map(f64::from).collect(), DType::F64)?;
    push_array(&mut out, "reshape_2x3_to_3x2", &small.reshape(&[3, 2])?)?;
    push_array(&mut out, "transpose_default_2x3", &small.transpose(None)?)?;

    let join_a = UFuncArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0], DType::F64)?;
    let join_b = UFuncArray::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0], DType::F64)?;
    push_array(
        &mut out,
        "concatenate_axis0",
        &UFuncArray::concatenate(&[&join_a, &join_b], 0)?,
    )?;

    let stack_x = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64)?;
    let stack_y = UFuncArray::new(vec![3], vec![10.0, 20.0, 30.0], DType::F64)?;
    push_array(
        &mut out,
        "stack_axis1",
        &UFuncArray::stack(&[&stack_x, &stack_y], 1)?,
    )?;

    out.push_str("## selection and ordering\n");
    let unsorted = UFuncArray::new(vec![2, 3], vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], DType::F64)?;
    push_array(&mut out, "clip_1_5_to_4_5", &unsorted.clip(1.5, 4.5))?;
    push_array(
        &mut out,
        "sort_axis1_quicksort",
        &unsorted.sort(Some(1), Some("quicksort"))?,
    )?;
    push_array(
        &mut out,
        "argsort_axis1_stable",
        &unsorted.argsort(Some(1), Some("stable"))?,
    )?;

    let condition = UFuncArray::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0], DType::Bool)?;
    let select_x = UFuncArray::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0], DType::F64)?;
    let select_y = UFuncArray::full(vec![2, 2], -1.0, DType::F64)?;
    push_array(
        &mut out,
        "where_select_2x2",
        &UFuncArray::where_select(&condition, &select_x, &select_y)?,
    )?;

    out.push_str("## indexing and set helpers\n");
    push_array(
        &mut out,
        "take_axis1_indices_2_0",
        &small.take(&[2, 0], Some(1))?,
    )?;

    let nonzero_input =
        UFuncArray::new(vec![2, 3], vec![0.0, 2.0, 0.0, 3.0, 0.0, 4.0], DType::F64)?;
    let nonzero = where_nonzero(&nonzero_input)?;
    for (axis, indices) in nonzero.iter().enumerate() {
        push_array(&mut out, &format!("where_nonzero_axis{axis}"), indices)?;
    }

    let unique_input = UFuncArray::new(vec![5], vec![3.0, 1.0, 3.0, 2.0, 1.0], DType::F64)?;
    let (unique_values, unique_indices, unique_inverse, unique_counts) = unique_all(&unique_input);
    push_array(&mut out, "unique_all_values", &unique_values)?;
    push_array(&mut out, "unique_all_indices", &unique_indices)?;
    push_array(&mut out, "unique_all_inverse", &unique_inverse)?;
    push_array(&mut out, "unique_all_counts", &unique_counts)?;

    out.push_str("## linear algebra and contractions\n");
    let mat_a = UFuncArray::new(vec![2, 3], (1..=6).map(f64::from).collect(), DType::F64)?;
    let mat_b = UFuncArray::new(vec![3, 2], (7..=12).map(f64::from).collect(), DType::F64)?;
    push_array(&mut out, "matmul_2x3_by_3x2", &mat_a.matmul(&mat_b)?)?;

    let dot_lhs = UFuncArray::new(vec![3], vec![1.0, 2.0, 3.0], DType::F64)?;
    let dot_rhs = UFuncArray::new(vec![3], vec![4.0, 5.0, 6.0], DType::F64)?;
    push_array(&mut out, "dot_1d_inner", &dot_lhs.dot(&dot_rhs)?)?;

    let weights = UFuncArray::new(vec![3], vec![1.0, 0.5, -1.0], DType::F64)?;
    push_array(
        &mut out,
        "einsum_weighted_row_sum",
        &UFuncArray::einsum("ij,j->i", &[&mat_a, &weights])?,
    )?;

    out.push_str("## cumulative and comparison summaries\n");
    push_array(&mut out, "cumsum_axis1", &small.cumsum(Some(1))?)?;
    push_array(
        &mut out,
        "percentile_75_axis0",
        &small.percentile(75.0, Some(0))?,
    )?;

    let close_lhs = UFuncArray::new(vec![3], vec![1.0, 1.000000001, 2.0], DType::F64)?;
    let close_rhs = UFuncArray::new(vec![3], vec![1.0, 1.0, 2.1], DType::F64)?;
    push_array(
        &mut out,
        "isclose_mixed",
        &close_lhs.isclose(&close_rhs, 1e-6, 1e-12)?,
    )?;
    push_bool(
        &mut out,
        "allclose_first_two",
        close_lhs
            .take(&[0, 1], None)?
            .allclose(&close_rhs.take(&[0, 1], None)?, 1e-6, 1e-12)?,
    )?;

    out.push_str("## statistical analysis\n");
    let diff_input = UFuncArray::new(vec![5], vec![1.0, 3.0, 6.0, 10.0, 15.0], DType::F64)?;
    push_array(&mut out, "diff_n1", &diff_input.diff(1, None)?)?;
    push_array(&mut out, "diff_n2", &diff_input.diff(2, None)?)?;

    let ptp_input = UFuncArray::new(vec![2, 3], vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], DType::F64)?;
    push_array(&mut out, "ptp_all", &ptp_input.ptp(None)?)?;
    push_array(&mut out, "ptp_axis1", &ptp_input.ptp(Some(1))?)?;

    let cov_input = UFuncArray::new(
        vec![3, 4],
        vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 0.5, 1.0, 1.5, 2.0],
        DType::F64,
    )?;
    push_array(&mut out, "cov_3x4", &cov_input.cov()?)?;
    push_array(&mut out, "corrcoef_3x4", &cov_input.corrcoef()?)?;

    out.push_str("## nan-aware reductions\n");
    let nan_input = UFuncArray::new(vec![5], vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0], DType::F64)?;
    push_array(&mut out, "nansum_all", &nan_input.nansum(None, false)?)?;
    push_array(&mut out, "nanmean_all", &nan_input.nanmean(None, false)?)?;

    let nan_2d = UFuncArray::new(
        vec![2, 3],
        vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN],
        DType::F64,
    )?;
    push_array(&mut out, "nansum_axis1", &nan_2d.nansum(Some(1), false)?)?;
    push_array(&mut out, "nanmean_axis0", &nan_2d.nanmean(Some(0), false)?)?;

    out.push_str("## histogram and binning\n");
    let hist_input = UFuncArray::new(
        vec![10],
        vec![0.5, 1.2, 2.3, 2.7, 3.1, 3.5, 4.2, 4.8, 5.1, 5.9],
        DType::F64,
    )?;
    let (counts, edges) = hist_input.histogram(5)?;
    push_array(&mut out, "histogram_5bins_counts", &counts)?;
    push_array(&mut out, "histogram_5bins_edges", &edges)?;

    let bin_edges = UFuncArray::new(vec![4], vec![0.0, 2.0, 4.0, 6.0], DType::F64)?;
    let (edge_counts, _) = hist_input.histogram_edges(&bin_edges)?;
    push_array(&mut out, "histogram_custom_edges_counts", &edge_counts)?;

    let bincount_input = UFuncArray::new(
        vec![8],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 2.0],
        DType::I64,
    )?;
    push_array(&mut out, "bincount_simple", &bincount_input.bincount()?)?;

    let digitize_x = UFuncArray::new(vec![5], vec![0.5, 1.5, 2.5, 3.5, 4.5], DType::F64)?;
    let digitize_bins = UFuncArray::new(vec![4], vec![1.0, 2.0, 3.0, 4.0], DType::F64)?;
    push_array(
        &mut out,
        "digitize_default",
        &digitize_x.digitize(&digitize_bins)?,
    )?;
    push_array(
        &mut out,
        "digitize_right",
        &digitize_x.digitize_right(&digitize_bins, true)?,
    )?;

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

fn push_bool(out: &mut String, label: &str, value: bool) -> Result<(), Box<dyn Error>> {
    writeln!(out, "{label}")?;
    writeln!(out, "  value: {value}")?;
    out.push('\n');
    Ok(())
}

fn format_shape(shape: &[usize]) -> String {
    let mut formatted = String::from("[");
    for (index, dim) in shape.iter().enumerate() {
        if index > 0 {
            formatted.push_str(", ");
        }
        formatted.push_str(&dim.to_string());
    }
    formatted.push(']');
    formatted
}

fn format_values(values: &[f64]) -> String {
    let mut formatted = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            formatted.push_str(", ");
        }
        formatted.push_str(&format_float(*value));
    }
    formatted.push(']');
    formatted
}

fn format_float(value: f64) -> String {
    if value.is_nan() {
        return String::from("NaN");
    }
    if value == f64::INFINITY {
        return String::from("inf");
    }
    if value == f64::NEG_INFINITY {
        return String::from("-inf");
    }
    if value == 0.0 {
        if value.is_sign_negative() {
            return String::from("-0.0");
        }
        return String::from("0.0");
    }

    let mut text = format!("{value:.12}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    text
}

fn assert_golden(name: &str, actual: &str) -> Result<(), Box<dyn Error>> {
    let path = golden_path(name);
    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, actual)?;
        return Ok(());
    }

    let expected = fs::read_to_string(&path).map_err(|err| {
        format!(
            "missing golden {}: {err}; run UPDATE_GOLDENS=1 cargo test -p fnp-ufunc public_api_output_matches_golden",
            path.display()
        )
    })?;

    if expected == actual {
        return Ok(());
    }

    let actual_path = path.with_extension("actual");
    fs::write(&actual_path, actual)?;
    Err(format!(
        "golden mismatch for {name}; compare {} with {}",
        path.display(),
        actual_path.display()
    )
    .into())
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(format!("{name}.golden"))
}
