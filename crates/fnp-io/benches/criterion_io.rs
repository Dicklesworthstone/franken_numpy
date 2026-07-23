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
    IOSupportedDType, NpyHeader, StructuredIODescriptor, StructuredIOField, StructuredNpyData,
    fromfile, fromfile_structured, fromfile_text, load, read_npy_bytes, read_npz_bytes,
    read_npz_bytes_linear_overlap_control, write_npy_bytes, write_npz_bytes,
};
use std::cell::{Cell, RefCell};
use std::hint::black_box;
use std::time::{Duration, Instant};

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
        bench.iter(|| black_box(fromfile_text(black_box(&text), ",", Some(PREFIX_COUNT)).unwrap()))
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
        bench.iter(|| black_box(fromfile_text(black_box(&text), ", ", Some(PREFIX_COUNT)).unwrap()))
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

/// Faithful replica of the CURRENT unselected `loadtxt` path: a fresh
/// `Vec<f64>` collected per accepted row (with parse short-circuit), then the
/// caller's ncols-before-budget checks, then a copy into the output - exactly
/// production's per-row allocation shape.
#[inline(never)]
fn loadtxt_plain_former(text: &str, delimiter: char, comments: char) -> (Vec<f64>, usize, usize) {
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
        let row_vals: Vec<f64> = if delimiter == ' ' {
            trimmed
                .split_whitespace()
                .map(|s| s.parse::<f64>().unwrap())
                .collect()
        } else {
            trimmed
                .split(delimiter)
                .map(|s| s.trim().parse::<f64>().unwrap())
                .collect()
        };
        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) => assert_eq!(row_vals.len(), expected),
        }
        values.extend(row_vals);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

/// Faithful replica of the CURRENT unselected `genfromtxt_full` path: the
/// eager `all_lines` collect (kept by the lever, shared cost in both arms)
/// plus a fresh `Vec<f64>` per row copied into the output.
#[inline(never)]
fn genfromtxt_full_plain_former(
    text: &str,
    delimiter: char,
    comments: char,
    filling_values: f64,
) -> (Vec<f64>, usize, usize) {
    let all_lines: Vec<&str> = text
        .lines()
        .filter_map(|line| {
            let trimmed = match line.find(comments) {
                Some(pos) => &line[..pos],
                None => line,
            }
            .trim();
            if trimmed.is_empty() || trimmed.starts_with(comments) {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect();

    let mut values = Vec::new();
    let mut ncols: Option<usize> = None;
    let mut nrows = 0usize;
    for &trimmed in all_lines.iter() {
        let row_vals: Vec<f64> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>().unwrap_or(filling_values))
            .collect();
        match ncols {
            None => ncols = Some(row_vals.len()),
            Some(expected) => assert_eq!(row_vals.len(), expected),
        }
        values.extend(row_vals);
        nrows += 1;
    }
    (values, nrows, ncols.unwrap_or(0))
}

/// Faithful replica of the CURRENT `tofile_text`: every integral-valued
/// element routed through `write!` fmt machinery as an i64, floats through
/// `write!("{v}")`, no capacity hint - exactly production's shape.
#[inline(never)]
fn tofile_text_former(values: &[f64], sep: &str) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (idx, v) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(sep);
        }
        if v.fract() == 0.0
            && v.is_finite()
            && v.abs() < 1e15
            && !(*v == 0.0 && v.is_sign_negative())
        {
            let _ = write!(&mut out, "{}", *v as i64);
        } else {
            let _ = write!(&mut out, "{v}");
        }
    }
    out
}

fn bench_tofile_text_integral(c: &mut Criterion) {
    const ELEMENTS: usize = 131_072;
    // Integral-heavy with a sprinkle of true floats and specials, mirroring
    // typical integer-valued exports.
    let values: Vec<f64> = (0..ELEMENTS)
        .map(|i| match i % 19 {
            17 => (i as f64) * 0.25 + 0.5,
            18 => -(i as f64) * 1.75,
            _ => ((i as i64 * 7919) % 2_000_003 - 1_000_001) as f64,
        })
        .collect();

    let former = tofile_text_former(&values, ",");
    let current = fnp_io::tofile_text(&values, ",");
    assert_eq!(current, former);

    // Variance protocol: 20 samples, 2 s window; floor predeclared in the
    // bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("tofile_text_integral");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(ELEMENTS as u64));
    group.bench_function("former_fmt_machinery", |bench| {
        bench.iter(|| black_box(tofile_text_former(black_box(&values), black_box(","))))
    });
    group.bench_function("candidate_manual_int", |bench| {
        bench.iter(|| black_box(fnp_io::tofile_text(black_box(&values), black_box(","))))
    });
    group.finish();
}

fn bench_genfromtxt_full_plain_rows(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            if (row + col) % 37 == 0 {
                text.push_str("n/a");
            } else {
                text.push_str(&format!("{}.{}", row % 977, col));
            }
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) =
        genfromtxt_full_plain_former(&text, ',', '#', -9.5);
    let config = fnp_io::GenFromTxtConfig {
        delimiter: ',',
        filling_values: -9.5,
        ..Default::default()
    };
    let current = fnp_io::genfromtxt_full(&text, &config).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, quiet worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("genfromtxt_full_plain_rows");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("former_per_row_vec", |bench| {
        bench.iter(|| {
            black_box(genfromtxt_full_plain_former(
                black_box(&text),
                ',',
                '#',
                -9.5,
            ))
        })
    });
    group.bench_function("candidate_direct_extend", |bench| {
        bench.iter(|| black_box(fnp_io::genfromtxt_full(black_box(&text), &config).unwrap()))
    });
    group.finish();
}

fn bench_loadtxt_plain_rows(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) = loadtxt_plain_former(&text, ',', '#');
    let current = fnp_io::loadtxt_usecols(&text, ',', '#', 0, usize::MAX, None).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, quiet worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("loadtxt_plain_rows");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("former_per_row_vec", |bench| {
        bench.iter(|| black_box(loadtxt_plain_former(black_box(&text), ',', '#')))
    });
    group.bench_function("candidate_direct_extend", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols(black_box(&text), ',', '#', 0, usize::MAX, None).unwrap(),
            )
        })
    });
    group.finish();
}

/// Faithful replica of the CURRENT (post-.342) usecols path: the column plan
/// hoisted once per call, but each row still scattering into a fresh
/// `selected` Vec (`vec![0.0; n_out]`) that is then copied into the output.
/// The scatter-into candidate must reproduce it bit-for-bit while removing
/// only the per-row Vec and copy.
#[inline(never)]
fn loadtxt_usecols_hoisted_former(
    text: &str,
    delimiter: char,
    comments: char,
    cols: &[usize],
) -> (Vec<f64>, usize, usize) {
    use std::collections::BTreeMap;
    let mut positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut max_col = 0usize;
    for (pos, &col) in cols.iter().enumerate() {
        positions.entry(col).or_default().push(pos);
        if col > max_col {
            max_col = col;
        }
    }
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
        let mut selected = vec![0.0; cols.len()];
        let mut col_idx = 0usize;
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

fn bench_loadtxt_usecols_scatter(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    const USECOLS: [usize; 4] = [13, 1, 7, 13];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let (former_values, former_rows, former_cols) =
        loadtxt_usecols_hoisted_former(&text, ',', '#', &USECOLS);
    let current = fnp_io::loadtxt_usecols(&text, ',', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    assert_eq!(current.nrows, former_rows);
    assert_eq!(current.ncols, former_cols);
    assert!(
        current
            .values
            .iter()
            .zip(&former_values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    // Variance protocol: 20 samples, 2 s window, quiet worker; floor
    // predeclared in the bead (disjoint AND >= 1.05x).
    let mut group = c.benchmark_group("loadtxt_usecols_scatter");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));
    group.bench_function("former_selected_vec", |bench| {
        bench.iter(|| {
            black_box(loadtxt_usecols_hoisted_former(
                black_box(&text),
                ',',
                '#',
                black_box(&USECOLS),
            ))
        })
    });
    group.bench_function("candidate_scatter_into", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols(
                    black_box(&text),
                    ',',
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

/// Former signed-usecols valid-input path, retained as the A/B comparator for
/// the exact nonnegative corpus measured by this benchmark.
#[inline(never)]
fn loadtxt_signed_nonnegative_former(text: &str, usecols: &[isize]) -> fnp_io::TextArrayData {
    let mut values = Vec::new();
    let mut ncols = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = line
            .split_once('#')
            .map_or(line, |(prefix, _)| prefix)
            .trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split(',').collect::<Vec<_>>();
        let mut row_values = Vec::with_capacity(usecols.len());
        for &column in usecols {
            let index = usize::try_from(column).expect("nonnegative former usecol");
            assert!(index < fields.len(), "former usecol in bounds");
            row_values.push(fields[index].trim().parse::<f64>().unwrap());
        }
        match ncols {
            None => ncols = Some(row_values.len()),
            Some(expected) => assert_eq!(row_values.len(), expected),
        }
        values.extend(row_values);
        nrows += 1;
    }
    fnp_io::TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    }
}

#[inline(never)]
fn loadtxt_signed_nonnegative_staged(
    text: &str,
    delimiter: char,
    comments: char,
    usecols: &[isize],
) -> fnp_io::TextArrayData {
    let unsigned = usecols
        .iter()
        .map(|&col| usize::try_from(col).expect("nonnegative staged usecol"))
        .collect::<Vec<_>>();
    fnp_io::loadtxt_usecols(text, delimiter, comments, 0, usize::MAX, Some(&unsigned)).unwrap()
}

fn time_loadtxt_signed(text: &str, usecols: &[isize], staged: bool) -> Duration {
    const REPETITIONS: u32 = 8;
    let start = Instant::now();
    for _ in 0..REPETITIONS {
        let output = if staged {
            fnp_io::loadtxt_usecols_signed(text, ',', '#', 0, usize::MAX, Some(usecols)).unwrap()
        } else {
            loadtxt_signed_nonnegative_former(text, usecols)
        };
        drop(black_box(output));
    }
    start.elapsed() / REPETITIONS
}

fn report_loadtxt_signed_pair(
    row: &str,
    lhs_samples: &RefCell<Vec<f64>>,
    rhs_samples: &RefCell<Vec<f64>>,
) {
    if lhs_samples.borrow().len() < 2 || rhs_samples.borrow().len() < 2 {
        return;
    }

    fn tail_stats(samples: &RefCell<Vec<f64>>) -> (usize, f64, f64) {
        let samples = samples.borrow();
        let count = samples.len().min(10);
        assert!(count >= 2, "paired loadtxt bench retained too few samples");
        let tail = &samples[samples.len() - count..];
        let mean = tail.iter().sum::<f64>() / count as f64;
        let variance = tail
            .iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .sum::<f64>()
            / (count - 1) as f64;
        (count, mean, variance.sqrt() * 100.0 / mean)
    }

    let (lhs_n, lhs_ns, lhs_cv) = tail_stats(lhs_samples);
    let (rhs_n, rhs_ns, rhs_cv) = tail_stats(rhs_samples);
    assert_eq!(lhs_n, rhs_n);
    println!(
        "LOADTXT_SIGNED_PAIR row={row} samples={lhs_n} lhs_mean_ms={:.6} \
         lhs_cv_pct={lhs_cv:.3} rhs_mean_ms={:.6} rhs_cv_pct={rhs_cv:.3} \
         lhs_over_rhs={:.4}",
        lhs_ns / 1_000_000.0,
        rhs_ns / 1_000_000.0,
        lhs_ns / rhs_ns,
    );
}

fn bench_loadtxt_signed_nonnegative_staging(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 16;
    const USECOLS: [isize; 4] = [13, 1, 7, 13];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let current =
        fnp_io::loadtxt_usecols_signed(&text, ',', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    let staged = loadtxt_signed_nonnegative_staged(&text, ',', '#', &USECOLS);
    assert_eq!(current.nrows, staged.nrows);
    assert_eq!(current.ncols, staged.ncols);
    assert_eq!(current.values.len(), staged.values.len());
    assert!(
        current
            .values
            .iter()
            .zip(&staged.values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("loadtxt_signed_nonnegative_staging");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements((ROWS * USECOLS.len()) as u64));

    let base_samples = RefCell::new(Vec::new());
    let staged_samples = RefCell::new(Vec::new());
    let order = Cell::new(0usize);
    group.bench_function("former_vs_candidate_abba", |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                let staged_outer = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let (base_total, staged_total) = if staged_outer {
                    let b1 = time_loadtxt_signed(&text, &USECOLS, true);
                    let a1 = time_loadtxt_signed(&text, &USECOLS, false);
                    let a2 = time_loadtxt_signed(&text, &USECOLS, false);
                    let b2 = time_loadtxt_signed(&text, &USECOLS, true);
                    (a1 + a2, b1 + b2)
                } else {
                    let a1 = time_loadtxt_signed(&text, &USECOLS, false);
                    let b1 = time_loadtxt_signed(&text, &USECOLS, true);
                    let b2 = time_loadtxt_signed(&text, &USECOLS, true);
                    let a2 = time_loadtxt_signed(&text, &USECOLS, false);
                    (a1 + a2, b1 + b2)
                };
                base_samples
                    .borrow_mut()
                    .push(base_total.as_secs_f64() * 0.5e9);
                staged_samples
                    .borrow_mut()
                    .push(staged_total.as_secs_f64() * 0.5e9);
                combined += base_total + staged_total;
            }
            combined
        });
    });
    report_loadtxt_signed_pair(
        "effect_former_over_candidate",
        &base_samples,
        &staged_samples,
    );

    let null_a = RefCell::new(Vec::new());
    let null_b = RefCell::new(Vec::new());
    let null_order = Cell::new(0usize);
    group.bench_function("candidate_aa_null_abba", |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                let b_outer = null_order.get() & 1 == 1;
                null_order.set(null_order.get().wrapping_add(1));
                let (a_total, b_total) = if b_outer {
                    let b1 = time_loadtxt_signed(&text, &USECOLS, true);
                    let a1 = time_loadtxt_signed(&text, &USECOLS, true);
                    let a2 = time_loadtxt_signed(&text, &USECOLS, true);
                    let b2 = time_loadtxt_signed(&text, &USECOLS, true);
                    (a1 + a2, b1 + b2)
                } else {
                    let a1 = time_loadtxt_signed(&text, &USECOLS, true);
                    let b1 = time_loadtxt_signed(&text, &USECOLS, true);
                    let b2 = time_loadtxt_signed(&text, &USECOLS, true);
                    let a2 = time_loadtxt_signed(&text, &USECOLS, true);
                    (a1 + a2, b1 + b2)
                };
                null_a.borrow_mut().push(a_total.as_secs_f64() * 0.5e9);
                null_b.borrow_mut().push(b_total.as_secs_f64() * 0.5e9);
                combined += a_total + b_total;
            }
            combined
        });
    });
    report_loadtxt_signed_pair("null_candidate_aa", &null_a, &null_b);
    group.finish();
}

#[inline(never)]
fn loadtxt_signed_tail_candidate(
    text: &str,
    delimiter: char,
    comments: char,
    usecols: &[isize],
) -> fnp_io::TextArrayData {
    let offsets = usecols
        .iter()
        .map(|&column| {
            assert!(column < 0, "tail candidate requires negative usecols");
            usize::try_from(column.checked_neg().expect("representable tail offset"))
                .expect("positive tail offset")
        })
        .collect::<Vec<_>>();
    let max_tail = offsets.iter().copied().max().expect("nonempty usecols");

    let mut values = Vec::new();
    let mut ncols = None;
    let mut nrows = 0usize;
    for line in text.lines() {
        let trimmed = line
            .split_once(comments)
            .map_or(line, |(prefix, _)| prefix)
            .trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }

        let mut tail = vec![None; max_tail];
        let mut width = 0usize;
        if delimiter == ' ' {
            for field in trimmed.split_whitespace() {
                tail[width % max_tail] = Some(field);
                width += 1;
            }
        } else {
            for field in trimmed.split(delimiter) {
                tail[width % max_tail] = Some(field);
                width += 1;
            }
        }

        let mut row_values = Vec::with_capacity(offsets.len());
        for &offset in &offsets {
            assert!(offset <= width, "tail candidate usecol in bounds");
            let field = tail[(width - offset) % max_tail].expect("retained tail field");
            row_values.push(field.trim().parse::<f64>().unwrap());
        }
        match ncols {
            None => ncols = Some(row_values.len()),
            Some(expected) => assert_eq!(row_values.len(), expected),
        }
        values.extend(row_values);
        nrows += 1;
    }

    fnp_io::TextArrayData {
        values,
        nrows,
        ncols: ncols.unwrap_or(0),
    }
}

fn time_loadtxt_signed_tail(text: &str, usecols: &[isize], candidate: bool) -> Duration {
    const REPETITIONS: u32 = 8;
    let start = Instant::now();
    for _ in 0..REPETITIONS {
        let output = if candidate {
            loadtxt_signed_tail_candidate(text, ',', '#', usecols)
        } else {
            fnp_io::loadtxt_usecols_signed(text, ',', '#', 0, usize::MAX, Some(usecols)).unwrap()
        };
        drop(black_box(output));
    }
    start.elapsed() / REPETITIONS
}

fn bench_loadtxt_signed_tail_staging(c: &mut Criterion) {
    const ROWS: usize = 8_192;
    const COLS: usize = 64;
    const USECOLS: [isize; 4] = [-1, -8, -32, -1];

    let mut text = String::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            if col > 0 {
                text.push(',');
            }
            text.push_str(&format!("{}.{}", row % 977, col));
        }
        text.push('\n');
    }

    let output =
        fnp_io::loadtxt_usecols_signed(&text, ',', '#', 0, usize::MAX, Some(&USECOLS)).unwrap();
    assert_eq!(output.nrows, ROWS);
    assert_eq!(output.ncols, USECOLS.len());
    let candidate = loadtxt_signed_tail_candidate(&text, ',', '#', &USECOLS);
    assert_eq!(output.nrows, candidate.nrows);
    assert_eq!(output.ncols, candidate.ncols);
    assert_eq!(output.values.len(), candidate.values.len());
    assert!(
        output
            .values
            .iter()
            .zip(&candidate.values)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    );

    let mut group = c.benchmark_group("loadtxt_signed_tail_staging");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.throughput(Throughput::Elements((ROWS * COLS) as u64));
    group.bench_function("current_width_relative", |bench| {
        bench.iter(|| {
            black_box(
                fnp_io::loadtxt_usecols_signed(
                    black_box(&text),
                    ',',
                    '#',
                    0,
                    usize::MAX,
                    black_box(Some(&USECOLS)),
                )
                .unwrap(),
            )
        })
    });

    let former_samples = RefCell::new(Vec::new());
    let candidate_samples = RefCell::new(Vec::new());
    let order = Cell::new(0usize);
    group.bench_function("former_vs_tail_ring_abba", |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                let candidate_outer = order.get() & 1 == 1;
                order.set(order.get().wrapping_add(1));
                let (former_total, candidate_total) = if candidate_outer {
                    let b1 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let a1 = time_loadtxt_signed_tail(&text, &USECOLS, false);
                    let a2 = time_loadtxt_signed_tail(&text, &USECOLS, false);
                    let b2 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    (a1 + a2, b1 + b2)
                } else {
                    let a1 = time_loadtxt_signed_tail(&text, &USECOLS, false);
                    let b1 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let b2 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let a2 = time_loadtxt_signed_tail(&text, &USECOLS, false);
                    (a1 + a2, b1 + b2)
                };
                former_samples
                    .borrow_mut()
                    .push(former_total.as_secs_f64() * 0.5e9);
                candidate_samples
                    .borrow_mut()
                    .push(candidate_total.as_secs_f64() * 0.5e9);
                combined += former_total + candidate_total;
            }
            combined
        });
    });
    report_loadtxt_signed_pair(
        "effect_former_over_tail_ring",
        &former_samples,
        &candidate_samples,
    );

    let null_a = RefCell::new(Vec::new());
    let null_b = RefCell::new(Vec::new());
    let null_order = Cell::new(0usize);
    group.bench_function("tail_ring_aa_null_abba", |bench| {
        bench.iter_custom(|iterations| {
            let mut combined = Duration::ZERO;
            for _ in 0..iterations {
                let b_outer = null_order.get() & 1 == 1;
                null_order.set(null_order.get().wrapping_add(1));
                let (a_total, b_total) = if b_outer {
                    let b1 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let a1 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let a2 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let b2 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    (a1 + a2, b1 + b2)
                } else {
                    let a1 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let b1 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let b2 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    let a2 = time_loadtxt_signed_tail(&text, &USECOLS, true);
                    (a1 + a2, b1 + b2)
                };
                null_a.borrow_mut().push(a_total.as_secs_f64() * 0.5e9);
                null_b.borrow_mut().push(b_total.as_secs_f64() * 0.5e9);
                combined += a_total + b_total;
            }
            combined
        });
    });
    report_loadtxt_signed_pair("null_tail_ring_aa", &null_a, &null_b);
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u64>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u64>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i64>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i64>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u64>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i16>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i16>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u16>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u16>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u32>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u32>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i32>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<i32>())
        })
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
        .find(|&offset| {
            !(padded.as_ptr() as usize + offset).is_multiple_of(core::mem::align_of::<u32>())
        })
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

fn load_npy_owned_body_former(data: &[u8]) -> (Vec<usize>, Vec<f64>, IOSupportedDType) {
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
    let owned_npy = read_npy_bytes(&npy_bytes, false).expect("read NPY profile fixture");
    let former = load_npy_owned_body_former(&npy_bytes);
    let current = load(&npy_bytes).expect("load NPY");
    assert_loaded_f64_bits_eq(&former, &current);

    let mut group = c.benchmark_group("load_npy_borrowed_body");
    group.throughput(Throughput::Bytes(npy_bytes.len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("profile_parse_and_owned_body_copy", |bench| {
        bench.iter(|| black_box(read_npy_bytes(black_box(&npy_bytes), false).expect("read NPY")))
    });
    group.bench_function("profile_decode_owned_body", |bench| {
        bench.iter(|| {
            black_box(
                fromfile(
                    black_box(owned_npy.payload.as_ref()),
                    owned_npy.header.descr,
                    None,
                )
                .expect("decode NPY body"),
            )
        })
    });
    group.bench_function("former_owned_body_copy", |bench| {
        bench.iter(|| black_box(load_npy_owned_body_former(black_box(&npy_bytes))))
    });
    group.bench_function("public_load", |bench| {
        bench.iter(|| black_box(load(black_box(&npy_bytes)).expect("load NPY")))
    });
    group.finish();
}

#[inline(never)]
fn fromfile_structured_single_field_former(
    data: &[u8],
    descriptor: &StructuredIODescriptor,
    count: Option<usize>,
) -> StructuredNpyData {
    let record_size = descriptor.record_size().expect("valid descriptor");
    let max_records = data.len() / record_size;
    let n = count.map_or(max_records, |requested| requested.min(max_records));
    let offsets = descriptor.field_offsets().expect("field offsets");
    let mut columns: Vec<Vec<u8>> = descriptor
        .fields
        .iter()
        .map(|field| {
            let size = field.dtype.item_size().expect("sized dtype");
            Vec::with_capacity(n * size)
        })
        .collect();

    for record_idx in 0..n {
        let record_start = record_idx * record_size;
        for (field_idx, field) in descriptor.fields.iter().enumerate() {
            let field_size = field.dtype.item_size().expect("sized dtype");
            let field_start = record_start + offsets[field_idx];
            let field_end = field_start + field_size;
            columns[field_idx].extend_from_slice(&data[field_start..field_end]);
        }
    }

    StructuredNpyData {
        shape: vec![n],
        fortran_order: false,
        descriptor: descriptor.clone(),
        columns,
    }
}

fn assert_structured_data_eq(lhs: &StructuredNpyData, rhs: &StructuredNpyData) {
    assert_eq!(lhs.shape, rhs.shape);
    assert_eq!(lhs.fortran_order, rhs.fortran_order);
    assert_eq!(lhs.descriptor, rhs.descriptor);
    assert_eq!(lhs.columns, rhs.columns);
}

fn bench_fromfile_structured_single_field(c: &mut Criterion) {
    const RECORDS: usize = 1_048_576;

    let descriptor = StructuredIODescriptor {
        fields: vec![StructuredIOField {
            name: "value".to_string(),
            dtype: IOSupportedDType::F64,
        }],
    };
    let values: Vec<u64> = (0..RECORDS)
        .map(|index| (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let data = bytemuck::cast_slice::<u64, u8>(&values);
    let current = fromfile_structured(data, &descriptor, None).expect("structured read");
    let former = fromfile_structured_single_field_former(data, &descriptor, None);
    assert_structured_data_eq(&current, &former);

    let mut group = c.benchmark_group("fromfile_structured_single_field");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));
    group.bench_function("former_exact_record_loop", |bench| {
        bench.iter(|| {
            black_box(fromfile_structured_single_field_former(
                black_box(data),
                black_box(&descriptor),
                None,
            ))
        })
    });
    group.bench_function("public_single_prefix_bulk_copy", |bench| {
        bench.iter(|| {
            black_box(
                fromfile_structured(black_box(data), black_box(&descriptor), None)
                    .expect("structured read"),
            )
        })
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

    let linear =
        read_npz_bytes_linear_overlap_control(&npz_bytes, false).expect("linear overlap control");
    let ordered = read_npz_bytes(&npz_bytes, false).expect("ordered overlap candidate");
    assert_eq!(ordered, linear);

    let mut group = c.benchmark_group("read_npz_overlap_tracking");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("linear_control_4096", |bench| {
        bench.iter(|| {
            black_box(read_npz_bytes_linear_overlap_control(black_box(&npz_bytes), false).unwrap())
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
    bench_loadtxt_usecols_scatter,
    bench_loadtxt_signed_nonnegative_staging,
    bench_loadtxt_signed_tail_staging,
    bench_loadtxt_plain_rows,
    bench_genfromtxt_full_plain_rows,
    bench_tofile_text_integral,
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
    bench_fromfile_structured_single_field,
    bench_write_npy,
    bench_read_npy,
    bench_write_npz,
    bench_read_npz,
    bench_read_npz_overlap_tracking,
    bench_npy_roundtrip,
);

criterion_main!(benches);
