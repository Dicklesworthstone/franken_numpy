//! Criterion benchmarks for fnp-io.
//!
//! Measures performance baselines for I/O operations:
//! - write_npy_bytes: serialize array to .npy format
//! - read_npy_bytes: deserialize array from .npy format
//! - write_npz_bytes: serialize multiple arrays to .npz archive
//! - read_npz_bytes: deserialize .npz archive
//!
//! These operations are critical for data persistence workflows.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fnp_io::{
    IOSupportedDType, NpyHeader, fromfile, fromfile_text, load, read_npy_bytes, read_npz_bytes,
    read_npz_bytes_linear_overlap_control, write_npy_bytes, write_npz_bytes,
};
use std::hint::black_box;
use std::time::Duration;

fn generate_f64_data(n: usize) -> Vec<u8> {
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    bytemuck::cast_slice(&data).to_vec()
}

fn make_npy_header(shape: &[usize]) -> NpyHeader {
    NpyHeader {
        descr: IOSupportedDType::F64,
        fortran_order: false,
        shape: shape.to_vec(),
    }
}

fn native_u64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U64
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U64Be
    }
}

fn non_native_u64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U64Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U64
    }
}

fn native_i64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I64
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I64Be
    }
}

fn non_native_i64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I64Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I64
    }
}

fn non_native_f64_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::F64Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::F64
    }
}

fn native_f32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::F32
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::F32Be
    }
}

fn non_native_f32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::F32Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::F32
    }
}

fn native_i32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I32
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I32Be
    }
}

fn non_native_i32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I32Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I32
    }
}

fn native_u32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U32
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U32Be
    }
}

fn non_native_u32_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U32Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U32
    }
}

fn native_i16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I16
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I16Be
    }
}

fn non_native_i16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::I16Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::I16
    }
}

fn native_u16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U16
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U16Be
    }
}

fn non_native_u16_dtype() -> IOSupportedDType {
    #[cfg(target_endian = "little")]
    {
        IOSupportedDType::U16Be
    }
    #[cfg(target_endian = "big")]
    {
        IOSupportedDType::U16
    }
}

fn fromfile_u64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            u64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f64
        })
        .collect()
}

fn fromfile_non_native_u64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            u64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
            .swap_bytes() as f64
        })
        .collect()
}

fn fromfile_i64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f64
        })
        .collect()
}

fn fromfile_non_native_i64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
            .swap_bytes() as f64
        })
        .collect()
}

fn fromfile_non_native_f64_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<f64>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from_bits(
                u64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
                .swap_bytes(),
            )
        })
        .collect()
}

fn fromfile_f32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<f32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn fromfile_non_native_f32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<f32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(f32::from_bits(
                u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).swap_bytes(),
            ))
        })
        .collect()
}

fn fromfile_i32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn fromfile_non_native_i32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).swap_bytes())
        })
        .collect()
}

fn fromfile_u32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        })
        .collect()
}

fn fromfile_non_native_u32_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u32>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).swap_bytes())
        })
        .collect()
}

fn fromfile_i16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i16::from_ne_bytes([chunk[0], chunk[1]]))
        })
        .collect()
}

fn fromfile_non_native_i16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<i16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(i16::from_ne_bytes([chunk[0], chunk[1]]).swap_bytes())
        })
        .collect()
}

fn fromfile_u16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u16::from_ne_bytes([chunk[0], chunk[1]]))
        })
        .collect()
}

fn fromfile_non_native_u16_former(data: &[u8], count: Option<usize>) -> Vec<f64> {
    let item_size = core::mem::size_of::<u16>();
    let max_elems = data.len() / item_size;
    let n = count.map_or(max_elems, |requested| requested.min(max_elems));
    (0..n)
        .map(|index| {
            let offset = index * item_size;
            let chunk = &data[offset..offset + item_size];
            f64::from(u16::from_ne_bytes([chunk[0], chunk[1]]).swap_bytes())
        })
        .collect()
}

#[inline(never)]
fn fromfile_text_bounded_prefix_former(text: &str, count: usize) -> Vec<f64> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    fields
        .into_iter()
        .take(count)
        .map(|field| field.trim().parse::<f64>().unwrap())
        .collect()
}

fn bench_fromfile_text_bounded_prefix(c: &mut Criterion) {
    const TOKEN_COUNT: usize = 131_071;
    const PREFIX_COUNT: usize = 32;

    let text = "1.25 ".repeat(TOKEN_COUNT);
    let former = fromfile_text_bounded_prefix_former(&text, PREFIX_COUNT);
    let current = fromfile_text(&text, " ", Some(PREFIX_COUNT)).unwrap();
    assert!(
        current
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );
    assert_eq!(current.len(), PREFIX_COUNT);

    let mut group = c.benchmark_group("fromfile_text_bounded_prefix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(PREFIX_COUNT as u64));
    group.bench_function("former_eager_collect", |bench| {
        bench.iter(|| {
            black_box(fromfile_text_bounded_prefix_former(
                black_box(&text),
                PREFIX_COUNT,
            ))
        })
    });
    group.bench_function("public_bounded_count", |bench| {
        bench.iter(|| black_box(fromfile_text(black_box(&text), " ", Some(PREFIX_COUNT)).unwrap()))
    });
    group.finish();
}

/// Faithful replica of the CURRENT general path for a pure-literal separator
/// with a bounded count: eager whole-input `split(sep)` collect, then the
/// same trim/empty-field/parse loop with the count break. The candidate must
/// reproduce it bit-for-bit while only streaming the tokenization.
#[inline(never)]
fn fromfile_text_literal_bounded_former(text: &str, sep: &str, count: usize) -> Vec<f64> {
    let fields: Vec<&str> = text.split(sep).collect();
    let mut values = Vec::new();
    let mut iter = fields.into_iter().peekable();
    while let Some(field) = iter.next() {
        if values.len() >= count {
            break;
        }
        let field = field.trim();
        if field.is_empty() {
            if iter.peek().is_none() {
                continue;
            }
            panic!("unexpected empty field in benchmark input");
        }
        values.push(field.parse::<f64>().unwrap());
    }
    values
}

fn bench_fromfile_text_literal_bounded_prefix(c: &mut Criterion) {
    const TOKEN_COUNT: usize = 131_071;
    const PREFIX_COUNT: usize = 32;

    let text = vec!["1.25"; TOKEN_COUNT].join(",");
    let former = fromfile_text_literal_bounded_former(&text, ",", PREFIX_COUNT);
    let current = fromfile_text(&text, ",", Some(PREFIX_COUNT)).unwrap();
    assert_eq!(current.len(), PREFIX_COUNT);
    assert_eq!(former.len(), PREFIX_COUNT);
    assert!(
        current
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_text_literal_bounded_prefix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(PREFIX_COUNT as u64));
    group.bench_function("former_eager_collect", |bench| {
        bench.iter(|| {
            black_box(fromfile_text_literal_bounded_former(
                black_box(&text),
                black_box(","),
                PREFIX_COUNT,
            ))
        })
    });
    group.bench_function("public_bounded_count", |bench| {
        bench
            .iter(|| black_box(fromfile_text(black_box(&text), ",", Some(PREFIX_COUNT)).unwrap()))
    });
    group.finish();
}

/// Faithful replica of the CURRENT space-wildcard path with a bounded count:
/// eager whole-input scan collecting every field, then the same
/// trim/empty-field/parse loop with the count break. Frozen copy of the
/// production scanner (tokens, wildcard matcher, emission order) so the lazy
/// candidate must reproduce it bit-for-bit.
#[derive(Clone, Copy)]
enum FormerSepToken {
    SpaceWildcard,
    Literal(char),
}

fn former_match_space_wildcard_sep(
    text: &str,
    start: usize,
    tokens: &[FormerSepToken],
) -> Option<usize> {
    let mut offset = 0usize;
    let mut iter = text[start..].chars().peekable();
    for token in tokens {
        match token {
            FormerSepToken::SpaceWildcard => {
                while let Some(&ch) = iter.peek() {
                    if ch.is_whitespace() {
                        iter.next();
                        offset += ch.len_utf8();
                    } else {
                        break;
                    }
                }
            }
            FormerSepToken::Literal(expected) => match iter.next() {
                Some(ch) if ch == *expected => {
                    offset += ch.len_utf8();
                }
                _ => return None,
            },
        }
    }
    Some(start + offset)
}

#[inline(never)]
fn fromfile_text_wildcard_bounded_former(text: &str, sep: &str, count: usize) -> Vec<f64> {
    let tokens: Vec<FormerSepToken> = sep
        .chars()
        .map(|c| {
            if c.is_whitespace() {
                FormerSepToken::SpaceWildcard
            } else {
                FormerSepToken::Literal(c)
            }
        })
        .collect();
    let mut parts = Vec::new();
    let mut field_start = 0usize;
    let mut idx = 0usize;
    while idx <= text.len() {
        if let Some(end) = former_match_space_wildcard_sep(text, idx, &tokens) {
            parts.push(&text[field_start..idx]);
            field_start = end;
            idx = end;
            continue;
        }
        if idx == text.len() {
            break;
        }
        let ch = text[idx..].chars().next().expect("valid utf-8");
        idx += ch.len_utf8();
    }
    parts.push(&text[field_start..]);

    let mut values = Vec::new();
    let mut iter = parts.into_iter().peekable();
    while let Some(field) = iter.next() {
        if values.len() >= count {
            break;
        }
        let field = field.trim();
        if field.is_empty() {
            if iter.peek().is_none() {
                continue;
            }
            panic!("unexpected empty field in benchmark input");
        }
        values.push(field.parse::<f64>().unwrap());
    }
    values
}

fn bench_fromfile_text_wildcard_bounded_prefix(c: &mut Criterion) {
    const TOKEN_COUNT: usize = 131_071;
    const PREFIX_COUNT: usize = 32;

    let text = vec!["1.25"; TOKEN_COUNT].join(", ");
    let former = fromfile_text_wildcard_bounded_former(&text, ", ", PREFIX_COUNT);
    let current = fromfile_text(&text, ", ", Some(PREFIX_COUNT)).unwrap();
    assert_eq!(current.len(), PREFIX_COUNT);
    assert_eq!(former.len(), PREFIX_COUNT);
    assert!(
        current
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_text_wildcard_bounded_prefix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements(PREFIX_COUNT as u64));
    group.bench_function("former_eager_scan", |bench| {
        bench.iter(|| {
            black_box(fromfile_text_wildcard_bounded_former(
                black_box(&text),
                black_box(", "),
                PREFIX_COUNT,
            ))
        })
    });
    group.bench_function("public_bounded_count", |bench| {
        bench
            .iter(|| black_box(fromfile_text(black_box(&text), ", ", Some(PREFIX_COUNT)).unwrap()))
    });
    group.finish();
}

/// Faithful replica of the FORMER `loadtxt_usecols` unquoted f64 path: the
/// row loop (comment strip, trims, ragged checks) with the column plan -
/// `BTreeMap<column, output positions>` plus its inner Vecs - rebuilt for
/// every accepted row, exactly as production did before the hoist.
#[inline(never)]
fn loadtxt_usecols_former(
    text: &str,
    delimiter: char,
    comments: char,
    cols: &[usize],
) -> (Vec<f64>, usize, usize) {
    use std::collections::BTreeMap;
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        // Former per-row plan build.
        let mut positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut max_col = 0usize;
        for (pos, &col) in cols.iter().enumerate() {
            positions.entry(col).or_default().push(pos);
            if col > max_col {
                max_col = col;
            }
        }
        let mut selected = vec![0.0; cols.len()];
        let mut col_idx = 0usize;
        if delimiter == ' ' {
            for token in trimmed.split_whitespace() {
                if col_idx > max_col {
                    break;
                }
                if let Some(pos_list) = positions.get(&col_idx) {
                    let value = token.parse::<f64>().unwrap();
                    for &pos in pos_list {
                        selected[pos] = value;
                    }
                }
                col_idx += 1;
            }
        } else {
            for token in trimmed.split(delimiter) {
                if col_idx > max_col {
                    break;
                }
                if let Some(pos_list) = positions.get(&col_idx) {
                    let value = token.trim().parse::<f64>().unwrap();
                    for &pos in pos_list {
                        selected[pos] = value;
                    }
                }
                col_idx += 1;
            }
        }
        assert!(col_idx > max_col, "usecols index out of bounds");
        match ncols {
            None => ncols = Some(selected.len()),
            Some(expected) => assert_eq!(selected.len(), expected),
        }
        values.extend(selected);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

fn bench_loadtxt_usecols_plan(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    // Duplicate and out-of-order selections deliberately present (x6teb shape).
    const USECOLS: [usize; 4] = [13, 1, 7, 13];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(' ');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) =
        loadtxt_usecols_former(&text, ' ', '#', &USECOLS);
    let current = fnp_io::loadtxt_usecols(&text, ' ', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert_eq!(current.values.len(), former_values.len());
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // x6teb retry protocol: 20 samples and a 2 s window on a warm pinned
    // worker, with the significance floor predeclared in the bead/ledger.
    let mut group = c.benchmark_group("loadtxt_usecols_plan");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));
    group.bench_function("former_per_row_planner", |bench| {
        bench.iter(|| {
            black_box(loadtxt_usecols_former(
                black_box(&text),
                ' ',
                '#',
                black_box(&USECOLS),
            ))
        })
    });
    group.bench_function("hoisted_plan_candidate", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols(
                    black_box(&text),
                    ' ',
                    '#',
                    0,
                    usize::MAX,
                    black_box(Some(&USECOLS)),
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

/// Faithful replica of the CURRENT `genfromtxt` comma path: a fresh
/// `Vec<f64>` collected per accepted row, then copied into the output -
/// exactly production's per-row allocation shape.
#[inline(never)]
fn genfromtxt_former(
    text: &str,
    delimiter: char,
    comments: char,
    filling_values: f64,
) -> (Vec<f64>, usize, usize) {
    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = match line.find(comments) {
            Some(pos) => &line[..pos],
            None => line,
        }
        .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        let row_vals: Vec<f64> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>().unwrap_or(filling_values))
            .collect();
        let current_ncols = row_vals.len();
        match ncols {
            None => ncols = Some(current_ncols),
            Some(expected) => assert_eq!(current_ncols, expected),
        }
        values.extend(row_vals);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

fn bench_genfromtxt_row_scratch(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            // Mix parseable floats with unparseable tokens to exercise
            // filling_values.
            if (row + col) % 37 == 0 {
                text.push_str("n/a");
            } else {
                text.push_str(&format!("{}.{}", row % 977, col));
            }
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) = genfromtxt_former(&text, ',', '#', -9.5);
    let current = fnp_io::genfromtxt(&text, ',', '#', 0, -9.5).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, warm pinned worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("genfromtxt_row_scratch");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("former_per_row_vec", |bench| {
        bench.iter(|| black_box(genfromtxt_former(black_box(&text), ',', '#', -9.5)))
    });
    group.bench_function("candidate_scratch_reuse", |bench| {
        bench.iter(|| black_box(fnp_io::genfromtxt(black_box(&text), ',', '#', 0, -9.5).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_u64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u64::MAX,
            2 => 1,
            3 => (1_u64 << 53) - 1,
            4 => 1_u64 << 53,
            5 => (1_u64 << 53) + 1,
            6 => 1_u64 << 63,
            _ => (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_u64_former(bytes, None);
    let candidate = fromfile(bytes, native_u64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u64>()];
    let misaligned_offset = (0..core::mem::align_of::<u64>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u64>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_u64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_u64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_native_u64_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_u64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_u64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_u64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u64::MAX,
            2 => 1,
            3 => (1_u64 << 53) - 1,
            4 => 1_u64 << 53,
            5 => (1_u64 << 53) + 1,
            6 => 1_u64 << 63,
            _ => (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let stored: Vec<u64> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_u64_former(bytes, None);
    let candidate = fromfile(bytes, non_native_u64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u64>()];
    let misaligned_offset = (0..core::mem::align_of::<u64>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u64>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_u64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_u64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_u64_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_u64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_u64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_i64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i64::MIN,
            1 => i64::MAX,
            2 => -1,
            3 => 0,
            4 => (1_i64 << 53) - 1,
            5 => 1_i64 << 53,
            6 => (1_i64 << 53) + 1,
            _ => (index as i64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_i64_former(bytes, None);
    let candidate = fromfile(bytes, native_i64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i64>()];
    let misaligned_offset = (0..core::mem::align_of::<i64>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<i64>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_i64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_i64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_native_i64_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_i64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_i64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_i64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i64::MIN,
            1 => i64::MAX,
            2 => -1,
            3 => 0,
            4 => (1_i64 << 53) - 1,
            5 => 1_i64 << 53,
            6 => (1_i64 << 53) + 1,
            _ => (index as i64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let stored: Vec<i64> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_i64_former(bytes, None);
    let candidate = fromfile(bytes, non_native_i64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i64>()];
    let misaligned_offset = (0..core::mem::align_of::<i64>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<i64>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_i64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_i64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_i64_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_i64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_i64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_f64(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let bits: Vec<u64> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0.0_f64.to_bits(),
            1 => (-0.0_f64).to_bits(),
            2 => f64::INFINITY.to_bits(),
            3 => f64::NEG_INFINITY.to_bits(),
            4 => 1,
            5 => 0x7fef_ffff_ffff_ffff,
            6 => 0x7ff8_0000_0000_0042,
            _ => (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        })
        .collect();
    let stored: Vec<u64> = bits.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_f64_former(bytes, None);
    let candidate = fromfile(bytes, non_native_f64_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u64>()];
    let misaligned_offset = (0..core::mem::align_of::<u64>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u64>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_f64_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_f64_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_f64_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_f64_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_f64_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_i16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i16).wrapping_mul(257).wrapping_sub(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_i16_former(bytes, None);
    let candidate = fromfile(bytes, native_i16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i16>()];
    let misaligned_offset = (0..core::mem::align_of::<i16>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<i16>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_i16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_i16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_i16_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_i16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_i16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_i16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i16::MIN,
            1 => i16::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i16).wrapping_mul(257).wrapping_sub(17),
        })
        .collect();
    let stored: Vec<i16> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_i16_former(bytes, None);
    let candidate = fromfile(bytes, non_native_i16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i16>()];
    let misaligned_offset = (0..core::mem::align_of::<i16>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<i16>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_i16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_i16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_i16_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_i16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_i16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_u16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u16::MAX,
            2 => 1,
            3 => 0x8000,
            _ => (index as u16).wrapping_mul(257).wrapping_add(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_u16_former(bytes, None);
    let candidate = fromfile(bytes, native_u16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u16>()];
    let misaligned_offset = (0..core::mem::align_of::<u16>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u16>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_u16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_u16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_u16_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_u16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_u16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_u16(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u16> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u16::MAX,
            2 => 1,
            3 => 0x8000,
            _ => (index as u16).wrapping_mul(257).wrapping_add(17),
        })
        .collect();
    let stored: Vec<u16> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_u16_former(bytes, None);
    let candidate = fromfile(bytes, non_native_u16_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u16>()];
    let misaligned_offset = (0..core::mem::align_of::<u16>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u16>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_u16_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_u16_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_u16_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_u16_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_u16_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_u32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u32::MAX,
            2 => 1,
            3 => 0x8000_0000,
            _ => (index as u32).wrapping_mul(2_654_435_761).wrapping_add(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_u32_former(bytes, None);
    let candidate = fromfile(bytes, native_u32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u32>()];
    let misaligned_offset = (0..core::mem::align_of::<u32>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u32>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_u32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_u32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_u32_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_u32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_u32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_u32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<u32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0,
            1 => u32::MAX,
            2 => 1,
            3 => 0x8000_0000,
            _ => (index as u32).wrapping_mul(2_654_435_761).wrapping_add(17),
        })
        .collect();
    let stored: Vec<u32> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_u32_former(bytes, None);
    let candidate = fromfile(bytes, non_native_u32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u32>()];
    let misaligned_offset = (0..core::mem::align_of::<u32>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u32>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_u32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_u32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_u32_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_u32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_u32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_i32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i32).wrapping_mul(65_537).wrapping_sub(17),
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_i32_former(bytes, None);
    let candidate = fromfile(bytes, native_i32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i32>()];
    let misaligned_offset = (0..core::mem::align_of::<i32>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<i32>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_i32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, native_i32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_native_i32_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_i32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_i32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_i32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<i32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => -1,
            3 => 0,
            _ => (index as i32).wrapping_mul(65_537).wrapping_sub(17),
        })
        .collect();
    let stored: Vec<i32> = values.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_i32_former(bytes, None);
    let candidate = fromfile(bytes, non_native_i32_dtype(), None).unwrap();
    assert_eq!(candidate, former);

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<i32>()];
    let misaligned_offset = (0..core::mem::align_of::<i32>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<i32>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_i32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_i32_dtype(), Some(257)).unwrap();
    assert_eq!(candidate_misaligned, former_misaligned);

    let mut group = c.benchmark_group("fromfile_non_native_i32_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_i32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_i32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_native_f32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let values: Vec<f32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => -0.0,
            1 => f32::from_bits(0x7fc0_0042),
            2 => f32::INFINITY,
            3 => f32::NEG_INFINITY,
            _ => index as f32 * 0.25 - 17.0,
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&values);
    let former = fromfile_f32_former(bytes, None);
    let candidate = fromfile(bytes, native_f32_dtype(), None).unwrap();
    assert_eq!(
        candidate
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut misaligned = Vec::with_capacity(bytes.len() + 1);
    misaligned.push(0);
    misaligned.extend_from_slice(bytes);
    let former_misaligned = fromfile_f32_former(&misaligned[1..], Some(257));
    let candidate_misaligned = fromfile(&misaligned[1..], native_f32_dtype(), Some(257)).unwrap();
    assert_eq!(
        candidate_misaligned
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        former_misaligned
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut group = c.benchmark_group("fromfile_native_f32_typed_slice");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_f32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), native_f32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn bench_fromfile_non_native_f32(c: &mut Criterion) {
    const ELEMENTS: usize = 262_144;
    let bits: Vec<u32> = (0..ELEMENTS)
        .map(|index| match index % 8 {
            0 => 0.0_f32.to_bits(),
            1 => (-0.0_f32).to_bits(),
            2 => f32::INFINITY.to_bits(),
            3 => f32::NEG_INFINITY.to_bits(),
            4 => 1,
            5 => 0x7f7f_ffff,
            6 => 0x7fc0_0042,
            _ => (index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223),
        })
        .collect();
    let stored: Vec<u32> = bits.iter().map(|&value| value.swap_bytes()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&stored);
    let former = fromfile_non_native_f32_former(bytes, None);
    let candidate = fromfile(bytes, non_native_f32_dtype(), None).unwrap();
    assert!(
        candidate
            .iter()
            .zip(&former)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut padded = vec![0u8; bytes.len() + core::mem::align_of::<u32>()];
    let misaligned_offset = (0..core::mem::align_of::<u32>())
        .find(|&offset| (padded.as_ptr() as usize + offset) % core::mem::align_of::<u32>() != 0)
        .unwrap();
    padded[misaligned_offset..misaligned_offset + bytes.len()].copy_from_slice(bytes);
    let misaligned = &padded[misaligned_offset..misaligned_offset + bytes.len()];
    let former_misaligned = fromfile_non_native_f32_former(misaligned, Some(257));
    let candidate_misaligned = fromfile(misaligned, non_native_f32_dtype(), Some(257)).unwrap();
    assert!(
        candidate_misaligned
            .iter()
            .zip(&former_misaligned)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("fromfile_non_native_f32_typed_byteswap");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("former_element_decode", |bench| {
        bench.iter(|| black_box(fromfile_non_native_f32_former(black_box(bytes), None)))
    });
    group.bench_function("typed_slice_byteswap_candidate", |bench| {
        bench.iter(|| black_box(fromfile(black_box(bytes), non_native_f32_dtype(), None).unwrap()))
    });
    group.finish();
}

fn load_npy_owned_body_former(
    data: &[u8],
) -> (Vec<usize>, Vec<f64>, IOSupportedDType) {
    let npy = read_npy_bytes(data, false).expect("read NPY");
    let dtype = npy.header.descr;
    let shape = npy.header.shape;
    let values = fromfile(npy.payload.as_ref(), dtype, None).expect("decode NPY body");
    (shape, values, dtype)
}

fn assert_loaded_f64_bits_eq(
    former: &(Vec<usize>, Vec<f64>, IOSupportedDType),
    current: &(Vec<usize>, Vec<f64>, IOSupportedDType),
) {
    assert_eq!(current.0, former.0);
    assert_eq!(current.2, former.2);
    assert_eq!(current.1.len(), former.1.len());
    for (current, former) in current.1.iter().zip(&former.1) {
        assert_eq!(current.to_bits(), former.to_bits());
    }
}

fn bench_load_npy_borrowed_body(c: &mut Criterion) {
    const ELEMENTS: usize = 1_000_000;

    let data = generate_f64_data(ELEMENTS);
    let header = make_npy_header(&[ELEMENTS]);
    let npy_bytes = write_npy_bytes(&header, &data, false).expect("write NPY");
    let former = load_npy_owned_body_former(&npy_bytes);
    let current = load(&npy_bytes).expect("load NPY");
    assert_loaded_f64_bits_eq(&former, &current);

    let mut group = c.benchmark_group("load_npy_borrowed_body");
    group.throughput(Throughput::Bytes(npy_bytes.len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_owned_body_copy", |bench| {
        bench.iter(|| black_box(load_npy_owned_body_former(black_box(&npy_bytes))))
    });
    group.bench_function("public_load", |bench| {
        bench.iter(|| black_box(load(black_box(&npy_bytes)).expect("load NPY")))
    });
    group.finish();
}

fn bench_write_npy(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_npy_bytes");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| {
                let result = write_npy_bytes(black_box(&header), black_box(&data), false);
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_read_npy(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_npy_bytes");

    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);
        let npy_bytes = write_npy_bytes(&header, &data, false).expect("write");

        group.throughput(Throughput::Bytes(npy_bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("elements", n),
            &npy_bytes,
            |bench, payload| {
                bench.iter(|| {
                    let result = read_npy_bytes(black_box(payload), false);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_write_npz(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_npz_bytes");

    for num_arrays in [1, 5, 10, 20] {
        let n = 10_000;
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
            .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
            .collect();

        let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
            .iter()
            .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
            .collect();

        let total_bytes = (num_arrays * n * 8) as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("num_arrays", num_arrays),
            &entry_refs,
            |bench, refs| {
                bench.iter(|| {
                    let result = write_npz_bytes(black_box(refs));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_read_npz(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_npz_bytes");

    for num_arrays in [1, 5, 10, 20] {
        let n = 10_000;
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
            .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
            .collect();

        let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
            .iter()
            .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
            .collect();

        let npz_bytes = write_npz_bytes(&entry_refs).expect("write npz");

        group.throughput(Throughput::Bytes(npz_bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("num_arrays", num_arrays),
            &npz_bytes,
            |bench, payload| {
                bench.iter(|| {
                    let result = read_npz_bytes(black_box(payload), false);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_read_npz_overlap_tracking(c: &mut Criterion) {
    // Maximum legal member count, but only one f64 per member: this isolates
    // ZIP metadata validation rather than payload bandwidth.
    let num_arrays = 4_096usize;
    let data = generate_f64_data(1);
    let header = make_npy_header(&[1]);
    let entries: Vec<(String, NpyHeader, Vec<u8>)> = (0..num_arrays)
        .map(|i| (format!("arr_{i}"), header.clone(), data.clone()))
        .collect();
    let entry_refs: Vec<(&str, &NpyHeader, &[u8])> = entries
        .iter()
        .map(|(name, h, d)| (name.as_str(), h, d.as_slice()))
        .collect();
    let npz_bytes = write_npz_bytes(&entry_refs).expect("write metadata-heavy npz");

    let linear = read_npz_bytes_linear_overlap_control(&npz_bytes, false)
        .expect("linear overlap control");
    let ordered = read_npz_bytes(&npz_bytes, false).expect("ordered overlap candidate");
    assert_eq!(ordered, linear);

    let mut group = c.benchmark_group("read_npz_overlap_tracking");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("linear_control_4096", |bench| {
        bench.iter(|| {
            black_box(
                read_npz_bytes_linear_overlap_control(black_box(&npz_bytes), false).unwrap(),
            )
        })
    });
    group.bench_function("ordered_candidate_4096", |bench| {
        bench.iter(|| black_box(read_npz_bytes(black_box(&npz_bytes), false).unwrap()))
    });
    group.finish();
}

fn bench_npy_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("npy_roundtrip");

    for n in [10_000, 100_000] {
        let data = generate_f64_data(n);
        let header = make_npy_header(&[n]);

        group.throughput(Throughput::Bytes((n * 8) as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| {
                let written = write_npy_bytes(black_box(&header), black_box(&data), false).unwrap();
                let read = read_npy_bytes(black_box(&written), false).unwrap();
                black_box(read)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fromfile_text_bounded_prefix,
    bench_fromfile_text_literal_bounded_prefix,
    bench_fromfile_text_wildcard_bounded_prefix,
    bench_loadtxt_usecols_plan,
    bench_genfromtxt_row_scratch,
    bench_fromfile_native_u64,
    bench_fromfile_non_native_u64,
    bench_fromfile_native_i64,
    bench_fromfile_non_native_i64,
    bench_fromfile_non_native_f64,
    bench_fromfile_native_i16,
    bench_fromfile_non_native_i16,
    bench_fromfile_native_u16,
    bench_fromfile_non_native_u16,
    bench_fromfile_native_u32,
    bench_fromfile_non_native_u32,
    bench_fromfile_native_i32,
    bench_fromfile_non_native_i32,
    bench_fromfile_native_f32,
    bench_fromfile_non_native_f32,
    bench_load_npy_borrowed_body,
    bench_write_npy,
    bench_read_npy,
    bench_write_npz,
    bench_read_npz,
    bench_read_npz_overlap_tracking,
    bench_npy_roundtrip,
);

criterion_main!(benches);
