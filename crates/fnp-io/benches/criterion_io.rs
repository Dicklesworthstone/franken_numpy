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
    IOSupportedDType, NpyHeader, fromfile, read_npy_bytes, read_npz_bytes,
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
    bench_fromfile_native_u64,
    bench_fromfile_non_native_u64,
    bench_fromfile_native_i64,
    bench_fromfile_non_native_i64,
    bench_fromfile_non_native_f64,
    bench_fromfile_native_i16,
    bench_fromfile_native_u16,
    bench_fromfile_native_u32,
    bench_fromfile_non_native_u32,
    bench_fromfile_native_i32,
    bench_fromfile_native_f32,
    bench_fromfile_non_native_f32,
    bench_write_npy,
    bench_read_npy,
    bench_write_npz,
    bench_read_npz,
    bench_read_npz_overlap_tracking,
    bench_npy_roundtrip,
);

criterion_main!(benches);
