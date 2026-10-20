//! Small fixed-width value-sort kernels.
//!
//! These are deliberately dtype-specific.  A value sort of `i64` has no
//! NaN/signed-zero ordering surface, and equal values have identical bytes, so
//! Rust's unstable integer order is byte-exact for every NumPy `kind`.

use std::simd::cmp::SimdOrd;
use std::simd::{Mask, Select, Simd, simd_swizzle};

const LANES: usize = 4;
type V = Simd<i64, LANES>;
/// The mask type must be named explicitly: `Mask::from_array` alone leaves the
/// element type open, and the inherent `select` cannot then be resolved.
type M = Mask<i64, LANES>;

/// Largest length the branchless network handles; above it `sort_unstable` is
/// the kernel.  256 is the size of the campaign's measured cell and is the
/// point at which a bitonic network's `O(n log^2 n)` comparator count is still
/// paying for its branchlessness.
const BITONIC_MAX: usize = 256;

/// Smallest slice length routed to the `i64` bitonic network (`sort_i64` uses it
/// for `16..=32`, and `sort_i64_network_forced` uses it for `16..=256`).
const BITONIC_MIN: usize = 16;

const MAX_VECS: usize = BITONIC_MAX / LANES;

/// Fused intra-register ascending bitonic finish (`j = 2` then `j = 1`) for a
/// 4-lane `i64` vector whose halves are already bitonic. Both blend masks are
/// compile-time constants (`asc` is always true in a symmetric butterfly
/// network).
#[inline(always)]
fn finish_intra_i64(v: V) -> V {
    let p2 = simd_swizzle!(v, [2, 3, 0, 1]);
    let v2 = M::from_array([true, true, false, false]).select(v.simd_min(p2), v.simd_max(p2));
    let p1 = simd_swizzle!(v2, [1, 0, 3, 2]);
    M::from_array([true, false, true, false]).select(v2.simd_min(p1), v2.simd_max(p1))
}

/// Sort a single 4-lane `i64` vector in ascending order (`k = 2` then `k = 4`
/// symmetric butterfly).
#[inline(always)]
fn sort_1_vec_i64(v: V) -> V {
    let p1 = simd_swizzle!(v, [1, 0, 3, 2]);
    let v1 = M::from_array([true, false, true, false]).select(v.simd_min(p1), v.simd_max(p1));
    let p2 = simd_swizzle!(v1, [3, 2, 1, 0]);
    let v2 = M::from_array([true, true, false, false]).select(v1.simd_min(p2), v1.simd_max(p2));
    let p3 = simd_swizzle!(v2, [1, 0, 3, 2]);
    M::from_array([true, false, true, false]).select(v2.simd_min(p3), v2.simd_max(p3))
}

#[inline(always)]
fn cas_pair_i64(a: &mut V, b: &mut V) {
    let mn = a.simd_min(*b);
    let mx = a.simd_max(*b);
    *a = mn;
    *b = mx;
}

#[inline(always)]
fn butterfly_pair_i64(a: &mut V, b: &mut V) {
    let b_rev = simd_swizzle!(*b, [3, 2, 1, 0]);
    let mn = a.simd_min(b_rev);
    let mx_rev = a.simd_max(b_rev);
    *a = mn;
    *b = simd_swizzle!(mx_rev, [3, 2, 1, 0]);
}

/// Complete a bitonic merge of 8 vectors (32 `i64`s) entirely in YMM registers
/// Finish merging 8 vectors (32 `i64`s) that are already in registers `v[0..8]`
/// from `j_vecs = 2` (`j = 8`) down to `j = 1`.
#[inline(always)]
fn finish_merge_8_vecs_from_j2(v: &mut [V; 8]) {
    let [
        mut v0,
        mut v1,
        mut v2,
        mut v3,
        mut v4,
        mut v5,
        mut v6,
        mut v7,
    ] = *v;
    // j_vecs = 2 (j = 8)
    cas_pair_i64(&mut v0, &mut v2);
    cas_pair_i64(&mut v1, &mut v3);
    cas_pair_i64(&mut v4, &mut v6);
    cas_pair_i64(&mut v5, &mut v7);
    // j_vecs = 1 (j = 4)
    cas_pair_i64(&mut v0, &mut v1);
    cas_pair_i64(&mut v2, &mut v3);
    cas_pair_i64(&mut v4, &mut v5);
    cas_pair_i64(&mut v6, &mut v7);
    // j = 2, 1 intra-register
    *v = [
        finish_intra_i64(v0),
        finish_intra_i64(v1),
        finish_intra_i64(v2),
        finish_intra_i64(v3),
        finish_intra_i64(v4),
        finish_intra_i64(v5),
        finish_intra_i64(v6),
        finish_intra_i64(v7),
    ];
}

/// Merge 8 vectors (32 `i64`s) in registers from `j_vecs = 4` (`j = 16`) down
/// to `j = 1`.
#[inline(always)]
fn merge_8_vecs_i64(block: &mut [V]) {
    let [
        mut v0,
        mut v1,
        mut v2,
        mut v3,
        mut v4,
        mut v5,
        mut v6,
        mut v7,
    ] = block[..8]
    else {
        return;
    };
    cas_pair_i64(&mut v0, &mut v4);
    cas_pair_i64(&mut v1, &mut v5);
    cas_pair_i64(&mut v2, &mut v6);
    cas_pair_i64(&mut v3, &mut v7);
    let mut regs = [v0, v1, v2, v3, v4, v5, v6, v7];
    finish_merge_8_vecs_from_j2(&mut regs);
    block[..8].copy_from_slice(&regs);
}

/// Sort 8 vectors (32 `i64`s) completely in registers (`k = 2, 4, 8, 16, 32`).
#[inline(always)]
fn sort_8_vecs_i64(block: &mut [V]) {
    let [
        mut v0,
        mut v1,
        mut v2,
        mut v3,
        mut v4,
        mut v5,
        mut v6,
        mut v7,
    ] = block[..8]
    else {
        return;
    };
    // k = 4 (each 4-lane vector sorted ascending)
    v0 = sort_1_vec_i64(v0);
    v1 = sort_1_vec_i64(v1);
    v2 = sort_1_vec_i64(v2);
    v3 = sort_1_vec_i64(v3);
    v4 = sort_1_vec_i64(v4);
    v5 = sort_1_vec_i64(v5);
    v6 = sort_1_vec_i64(v6);
    v7 = sort_1_vec_i64(v7);

    // k = 8 (pairs of vectors)
    butterfly_pair_i64(&mut v0, &mut v1);
    butterfly_pair_i64(&mut v2, &mut v3);
    butterfly_pair_i64(&mut v4, &mut v5);
    butterfly_pair_i64(&mut v6, &mut v7);
    v0 = finish_intra_i64(v0);
    v1 = finish_intra_i64(v1);
    v2 = finish_intra_i64(v2);
    v3 = finish_intra_i64(v3);
    v4 = finish_intra_i64(v4);
    v5 = finish_intra_i64(v5);
    v6 = finish_intra_i64(v6);
    v7 = finish_intra_i64(v7);

    // k = 16 (quads of vectors)
    butterfly_pair_i64(&mut v0, &mut v3);
    butterfly_pair_i64(&mut v1, &mut v2);
    butterfly_pair_i64(&mut v4, &mut v7);
    butterfly_pair_i64(&mut v5, &mut v6);
    cas_pair_i64(&mut v0, &mut v1);
    cas_pair_i64(&mut v2, &mut v3);
    cas_pair_i64(&mut v4, &mut v5);
    cas_pair_i64(&mut v6, &mut v7);
    v0 = finish_intra_i64(v0);
    v1 = finish_intra_i64(v1);
    v2 = finish_intra_i64(v2);
    v3 = finish_intra_i64(v3);
    v4 = finish_intra_i64(v4);
    v5 = finish_intra_i64(v5);
    v6 = finish_intra_i64(v6);
    v7 = finish_intra_i64(v7);

    // k = 32 (all 8 vectors)
    butterfly_pair_i64(&mut v0, &mut v7);
    butterfly_pair_i64(&mut v1, &mut v6);
    butterfly_pair_i64(&mut v2, &mut v5);
    butterfly_pair_i64(&mut v3, &mut v4);
    let mut regs = [v0, v1, v2, v3, v4, v5, v6, v7];
    finish_merge_8_vecs_from_j2(&mut regs);
    block[..8].copy_from_slice(&regs);
}

/// Sort 4 vectors (16 `i64`s) completely in registers (`k = 2, 4, 8, 16`).
#[inline(always)]
fn sort_4_vecs_i64(block: &mut [V]) {
    let [mut v0, mut v1, mut v2, mut v3] = block[..4] else {
        return;
    };
    v0 = sort_1_vec_i64(v0);
    v1 = sort_1_vec_i64(v1);
    v2 = sort_1_vec_i64(v2);
    v3 = sort_1_vec_i64(v3);

    butterfly_pair_i64(&mut v0, &mut v1);
    butterfly_pair_i64(&mut v2, &mut v3);
    v0 = finish_intra_i64(v0);
    v1 = finish_intra_i64(v1);
    v2 = finish_intra_i64(v2);
    v3 = finish_intra_i64(v3);

    butterfly_pair_i64(&mut v0, &mut v3);
    butterfly_pair_i64(&mut v1, &mut v2);
    cas_pair_i64(&mut v0, &mut v1);
    cas_pair_i64(&mut v2, &mut v3);
    block[0] = finish_intra_i64(v0);
    block[1] = finish_intra_i64(v1);
    block[2] = finish_intra_i64(v2);
    block[3] = finish_intra_i64(v3);
}

/// Register-blocked symmetric bitonic sort of `buf` in ascending order.
/// `buf.len()` must be a power of two (`4, 8, 16, 32, 64` vectors for
/// `n = 16, 32, 64, 128, 256`).
fn bitonic_sort_pow2(buf: &mut [V]) {
    let nv = buf.len();
    if nv == 4 {
        sort_4_vecs_i64(buf);
        return;
    }
    for block in buf.as_chunks_mut::<8>().0 {
        sort_8_vecs_i64(block);
    }

    let mut k_vecs = 16;
    while k_vecs <= nv {
        let half = k_vecs / 2;
        for base in (0..nv).step_by(k_vecs) {
            for off in 0..half {
                let v = base + off;
                let p = base + k_vecs - 1 - off;
                let mut a = buf[v];
                let mut b = buf[p];
                butterfly_pair_i64(&mut a, &mut b);
                buf[v] = a;
                buf[p] = b;
            }
            let mut j_vecs = half / 2;
            while j_vecs >= 8 {
                for sub in (0..k_vecs).step_by(j_vecs * 2) {
                    for off in 0..j_vecs {
                        let v = base + sub + off;
                        let p = v + j_vecs;
                        let mut a = buf[v];
                        let mut b = buf[p];
                        cas_pair_i64(&mut a, &mut b);
                        buf[v] = a;
                        buf[p] = b;
                    }
                }
                j_vecs /= 2;
            }
            for sub in (0..k_vecs).step_by(8) {
                merge_8_vecs_i64(&mut buf[base + sub..base + sub + 8]);
            }
        }
        k_vecs *= 2;
    }
}

macro_rules! define_simd8_sort_32bit {
    (
        $elem:ty,
        $max_val:expr,
        $vec_ty:ident,
        $mask_ty:ident,
        $finish_intra:ident,
        $sort_1:ident,
        $cas_pair:ident,
        $butterfly_pair:ident,
        $finish_merge_8:ident,
        $merge_8:ident,
        $sort_2:ident,
        $sort_4:ident,
        $sort_8:ident,
        $sort_pow2:ident,
        $sort_network:ident
    ) => {
        type $vec_ty = Simd<$elem, 8>;
        type $mask_ty = Mask<i32, 8>;

        #[inline(always)]
        fn $finish_intra(v: $vec_ty) -> $vec_ty {
            let p4 = simd_swizzle!(v, [4, 5, 6, 7, 0, 1, 2, 3]);
            let v4 = $mask_ty::from_array([true, true, true, true, false, false, false, false])
                .select(v.simd_min(p4), v.simd_max(p4));
            let p2 = simd_swizzle!(v4, [2, 3, 0, 1, 6, 7, 4, 5]);
            let v2 = $mask_ty::from_array([true, true, false, false, true, true, false, false])
                .select(v4.simd_min(p2), v4.simd_max(p2));
            let p1 = simd_swizzle!(v2, [1, 0, 3, 2, 5, 4, 7, 6]);
            $mask_ty::from_array([true, false, true, false, true, false, true, false])
                .select(v2.simd_min(p1), v2.simd_max(p1))
        }

        #[inline(always)]
        fn $sort_1(v: $vec_ty) -> $vec_ty {
            // k = 2
            let p1 = simd_swizzle!(v, [1, 0, 3, 2, 5, 4, 7, 6]);
            let v1 = $mask_ty::from_array([true, false, true, false, true, false, true, false])
                .select(v.simd_min(p1), v.simd_max(p1));
            // k = 4
            let p2 = simd_swizzle!(v1, [3, 2, 1, 0, 7, 6, 5, 4]);
            let v2 = $mask_ty::from_array([true, true, false, false, true, true, false, false])
                .select(v1.simd_min(p2), v1.simd_max(p2));
            let p3 = simd_swizzle!(v2, [1, 0, 3, 2, 5, 4, 7, 6]);
            let v3 = $mask_ty::from_array([true, false, true, false, true, false, true, false])
                .select(v2.simd_min(p3), v2.simd_max(p3));
            // k = 8
            let p4 = simd_swizzle!(v3, [7, 6, 5, 4, 3, 2, 1, 0]);
            let v4 = $mask_ty::from_array([true, true, true, true, false, false, false, false])
                .select(v3.simd_min(p4), v3.simd_max(p4));
            let p5 = simd_swizzle!(v4, [2, 3, 0, 1, 6, 7, 4, 5]);
            let v5 = $mask_ty::from_array([true, true, false, false, true, true, false, false])
                .select(v4.simd_min(p5), v4.simd_max(p5));
            let p6 = simd_swizzle!(v5, [1, 0, 3, 2, 5, 4, 7, 6]);
            $mask_ty::from_array([true, false, true, false, true, false, true, false])
                .select(v5.simd_min(p6), v5.simd_max(p6))
        }

        #[inline(always)]
        fn $cas_pair(a: &mut $vec_ty, b: &mut $vec_ty) {
            let mn = a.simd_min(*b);
            let mx = a.simd_max(*b);
            *a = mn;
            *b = mx;
        }

        #[inline(always)]
        fn $butterfly_pair(a: &mut $vec_ty, b: &mut $vec_ty) {
            let b_rev = simd_swizzle!(*b, [7, 6, 5, 4, 3, 2, 1, 0]);
            let mn = a.simd_min(b_rev);
            let mx_rev = a.simd_max(b_rev);
            *a = mn;
            *b = simd_swizzle!(mx_rev, [7, 6, 5, 4, 3, 2, 1, 0]);
        }

        #[inline(always)]
        fn $finish_merge_8(v: &mut [$vec_ty; 8]) {
            let [
                mut v0,
                mut v1,
                mut v2,
                mut v3,
                mut v4,
                mut v5,
                mut v6,
                mut v7,
            ] = *v;
            $cas_pair(&mut v0, &mut v2);
            $cas_pair(&mut v1, &mut v3);
            $cas_pair(&mut v4, &mut v6);
            $cas_pair(&mut v5, &mut v7);
            $cas_pair(&mut v0, &mut v1);
            $cas_pair(&mut v2, &mut v3);
            $cas_pair(&mut v4, &mut v5);
            $cas_pair(&mut v6, &mut v7);
            *v = [
                $finish_intra(v0),
                $finish_intra(v1),
                $finish_intra(v2),
                $finish_intra(v3),
                $finish_intra(v4),
                $finish_intra(v5),
                $finish_intra(v6),
                $finish_intra(v7),
            ];
        }

        #[inline(always)]
        fn $merge_8(block: &mut [$vec_ty]) {
            let [
                mut v0,
                mut v1,
                mut v2,
                mut v3,
                mut v4,
                mut v5,
                mut v6,
                mut v7,
            ] = block[..8]
            else {
                return;
            };
            $cas_pair(&mut v0, &mut v4);
            $cas_pair(&mut v1, &mut v5);
            $cas_pair(&mut v2, &mut v6);
            $cas_pair(&mut v3, &mut v7);
            let mut regs = [v0, v1, v2, v3, v4, v5, v6, v7];
            $finish_merge_8(&mut regs);
            block[..8].copy_from_slice(&regs);
        }

        #[inline(always)]
        fn $sort_2(block: &mut [$vec_ty]) {
            let [mut v0, mut v1] = block[..2] else {
                return;
            };
            v0 = $sort_1(v0);
            v1 = $sort_1(v1);
            $butterfly_pair(&mut v0, &mut v1);
            block[0] = $finish_intra(v0);
            block[1] = $finish_intra(v1);
        }

        #[inline(always)]
        fn $sort_4(block: &mut [$vec_ty]) {
            let [mut v0, mut v1, mut v2, mut v3] = block[..4] else {
                return;
            };
            v0 = $sort_1(v0);
            v1 = $sort_1(v1);
            v2 = $sort_1(v2);
            v3 = $sort_1(v3);
            $butterfly_pair(&mut v0, &mut v1);
            $butterfly_pair(&mut v2, &mut v3);
            v0 = $finish_intra(v0);
            v1 = $finish_intra(v1);
            v2 = $finish_intra(v2);
            v3 = $finish_intra(v3);
            $butterfly_pair(&mut v0, &mut v3);
            $butterfly_pair(&mut v1, &mut v2);
            $cas_pair(&mut v0, &mut v1);
            $cas_pair(&mut v2, &mut v3);
            block[0] = $finish_intra(v0);
            block[1] = $finish_intra(v1);
            block[2] = $finish_intra(v2);
            block[3] = $finish_intra(v3);
        }

        #[inline(always)]
        fn $sort_8(block: &mut [$vec_ty]) {
            let [
                mut v0,
                mut v1,
                mut v2,
                mut v3,
                mut v4,
                mut v5,
                mut v6,
                mut v7,
            ] = block[..8]
            else {
                return;
            };
            v0 = $sort_1(v0);
            v1 = $sort_1(v1);
            v2 = $sort_1(v2);
            v3 = $sort_1(v3);
            v4 = $sort_1(v4);
            v5 = $sort_1(v5);
            v6 = $sort_1(v6);
            v7 = $sort_1(v7);

            $butterfly_pair(&mut v0, &mut v1);
            $butterfly_pair(&mut v2, &mut v3);
            $butterfly_pair(&mut v4, &mut v5);
            $butterfly_pair(&mut v6, &mut v7);
            v0 = $finish_intra(v0);
            v1 = $finish_intra(v1);
            v2 = $finish_intra(v2);
            v3 = $finish_intra(v3);
            v4 = $finish_intra(v4);
            v5 = $finish_intra(v5);
            v6 = $finish_intra(v6);
            v7 = $finish_intra(v7);

            $butterfly_pair(&mut v0, &mut v3);
            $butterfly_pair(&mut v1, &mut v2);
            $butterfly_pair(&mut v4, &mut v7);
            $butterfly_pair(&mut v5, &mut v6);
            $cas_pair(&mut v0, &mut v1);
            $cas_pair(&mut v2, &mut v3);
            $cas_pair(&mut v4, &mut v5);
            $cas_pair(&mut v6, &mut v7);
            v0 = $finish_intra(v0);
            v1 = $finish_intra(v1);
            v2 = $finish_intra(v2);
            v3 = $finish_intra(v3);
            v4 = $finish_intra(v4);
            v5 = $finish_intra(v5);
            v6 = $finish_intra(v6);
            v7 = $finish_intra(v7);

            $butterfly_pair(&mut v0, &mut v7);
            $butterfly_pair(&mut v1, &mut v6);
            $butterfly_pair(&mut v2, &mut v5);
            $butterfly_pair(&mut v3, &mut v4);
            let mut regs = [v0, v1, v2, v3, v4, v5, v6, v7];
            $finish_merge_8(&mut regs);
            block[..8].copy_from_slice(&regs);
        }

        fn $sort_pow2(buf: &mut [$vec_ty]) {
            let nv = buf.len();
            match nv {
                1 => {
                    buf[0] = $sort_1(buf[0]);
                    return;
                }
                2 => {
                    $sort_2(buf);
                    return;
                }
                4 => {
                    $sort_4(buf);
                    return;
                }
                _ => {}
            }
            for block in buf.as_chunks_mut::<8>().0 {
                $sort_8(block);
            }
            let mut k_vecs = 16;
            while k_vecs <= nv {
                let half = k_vecs / 2;
                for base in (0..nv).step_by(k_vecs) {
                    for off in 0..half {
                        let v = base + off;
                        let p = base + k_vecs - 1 - off;
                        let mut a = buf[v];
                        let mut b = buf[p];
                        $butterfly_pair(&mut a, &mut b);
                        buf[v] = a;
                        buf[p] = b;
                    }
                    let mut j_vecs = half / 2;
                    while j_vecs >= 8 {
                        for sub in (0..k_vecs).step_by(j_vecs * 2) {
                            for off in 0..j_vecs {
                                let v = base + sub + off;
                                let p = v + j_vecs;
                                let mut a = buf[v];
                                let mut b = buf[p];
                                $cas_pair(&mut a, &mut b);
                                buf[v] = a;
                                buf[p] = b;
                            }
                        }
                        j_vecs /= 2;
                    }
                    for sub in (0..k_vecs).step_by(8) {
                        $merge_8(&mut buf[base + sub..base + sub + 8]);
                    }
                }
                k_vecs *= 2;
            }
        }

        fn $sort_network(values: &mut [$elem]) {
            let n = values.len().next_power_of_two().max(8);
            let nv = n / 8;
            let mut buf = [$vec_ty::splat($max_val); 32];
            let (chunks, rem) = values.as_chunks::<8>();
            for (slot, chunk) in buf.iter_mut().zip(chunks) {
                *slot = $vec_ty::from_array(*chunk);
            }
            if !rem.is_empty() {
                let mut tail = [$max_val; 8];
                tail[..rem.len()].copy_from_slice(rem);
                buf[chunks.len()] = $vec_ty::from_array(tail);
            }
            $sort_pow2(&mut buf[..nv]);
            let full = chunks.len();
            let (out_chunks, out_rem) = values.as_chunks_mut::<8>();
            for (chunk, v) in out_chunks.iter_mut().zip(buf[..full].iter()) {
                *chunk = v.to_array();
            }
            if !out_rem.is_empty() {
                let tail = buf[full].to_array();
                out_rem.copy_from_slice(&tail[..out_rem.len()]);
            }
        }
    };
}

define_simd8_sort_32bit!(
    i32,
    i32::MAX,
    V32I,
    M32I,
    finish_intra_i32,
    sort_1_vec_i32,
    cas_pair_i32,
    butterfly_pair_i32,
    finish_merge_8_vecs_i32,
    merge_8_vecs_i32,
    sort_2_vecs_i32,
    sort_4_vecs_i32,
    sort_8_vecs_i32,
    bitonic_sort_pow2_i32,
    sort_i32_network
);

define_simd8_sort_32bit!(
    u32,
    u32::MAX,
    V32U,
    M32U,
    finish_intra_u32,
    sort_1_vec_u32,
    cas_pair_u32,
    butterfly_pair_u32,
    finish_merge_8_vecs_u32,
    merge_8_vecs_u32,
    sort_2_vecs_u32,
    sort_4_vecs_u32,
    sort_8_vecs_u32,
    bitonic_sort_pow2_u32,
    sort_u32_network
);

macro_rules! define_simd16_sort_16bit {
    (
        $elem:ty,
        $max_val:expr,
        $vec_ty:ident,
        $mask_ty:ident,
        $finish_intra:ident,
        $sort_1:ident,
        $cas_pair:ident,
        $butterfly_pair:ident,
        $finish_merge_8:ident,
        $merge_8:ident,
        $sort_2:ident,
        $sort_4:ident,
        $sort_8:ident,
        $sort_pow2:ident,
        $sort_network:ident
    ) => {
        type $vec_ty = Simd<$elem, 16>;
        type $mask_ty = Mask<i16, 16>;

        #[inline(always)]
        fn $finish_intra(v: $vec_ty) -> $vec_ty {
            let p8 = simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]);
            let v8 = $mask_ty::from_array([
                true, true, true, true, true, true, true, true, false, false, false, false, false,
                false, false, false,
            ])
            .select(v.simd_min(p8), v.simd_max(p8));
            let p4 = simd_swizzle!(v8, [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]);
            let v4 = $mask_ty::from_array([
                true, true, true, true, false, false, false, false, true, true, true, true, false,
                false, false, false,
            ])
            .select(v8.simd_min(p4), v8.simd_max(p4));
            let p2 = simd_swizzle!(v4, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]);
            let v2 = $mask_ty::from_array([
                true, true, false, false, true, true, false, false, true, true, false, false, true,
                true, false, false,
            ])
            .select(v4.simd_min(p2), v4.simd_max(p2));
            let p1 = simd_swizzle!(v2, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
            $mask_ty::from_array([
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false,
            ])
            .select(v2.simd_min(p1), v2.simd_max(p1))
        }

        #[inline(always)]
        fn $sort_1(v: $vec_ty) -> $vec_ty {
            // k = 2
            let p1 = simd_swizzle!(v, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
            let v1 = $mask_ty::from_array([
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false,
            ])
            .select(v.simd_min(p1), v.simd_max(p1));
            // k = 4
            let p2 = simd_swizzle!(v1, [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]);
            let v2 = $mask_ty::from_array([
                true, true, false, false, true, true, false, false, true, true, false, false, true,
                true, false, false,
            ])
            .select(v1.simd_min(p2), v1.simd_max(p2));
            let p3 = simd_swizzle!(v2, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
            let v3 = $mask_ty::from_array([
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false,
            ])
            .select(v2.simd_min(p3), v2.simd_max(p3));
            // k = 8
            let p4 = simd_swizzle!(v3, [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]);
            let v4 = $mask_ty::from_array([
                true, true, true, true, false, false, false, false, true, true, true, true, false,
                false, false, false,
            ])
            .select(v3.simd_min(p4), v3.simd_max(p4));
            let p5 = simd_swizzle!(v4, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]);
            let v5 = $mask_ty::from_array([
                true, true, false, false, true, true, false, false, true, true, false, false, true,
                true, false, false,
            ])
            .select(v4.simd_min(p5), v4.simd_max(p5));
            let p6 = simd_swizzle!(v5, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
            let v6 = $mask_ty::from_array([
                true, false, true, false, true, false, true, false, true, false, true, false, true,
                false, true, false,
            ])
            .select(v5.simd_min(p6), v5.simd_max(p6));
            // k = 16
            let p7 = simd_swizzle!(v6, [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
            $finish_intra(
                $mask_ty::from_array([
                    true, true, true, true, true, true, true, true, false, false, false, false,
                    false, false, false, false,
                ])
                .select(v6.simd_min(p7), v6.simd_max(p7)),
            )
        }

        #[inline(always)]
        fn $cas_pair(a: &mut $vec_ty, b: &mut $vec_ty) {
            let mn = a.simd_min(*b);
            let mx = a.simd_max(*b);
            *a = mn;
            *b = mx;
        }

        #[inline(always)]
        fn $butterfly_pair(a: &mut $vec_ty, b: &mut $vec_ty) {
            let b_rev = simd_swizzle!(*b, [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
            let mn = a.simd_min(b_rev);
            let mx_rev = a.simd_max(b_rev);
            *a = mn;
            *b = simd_swizzle!(
                mx_rev,
                [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            );
        }

        #[inline(always)]
        fn $finish_merge_8(v: &mut [$vec_ty; 8]) {
            let [
                mut v0,
                mut v1,
                mut v2,
                mut v3,
                mut v4,
                mut v5,
                mut v6,
                mut v7,
            ] = *v;
            $cas_pair(&mut v0, &mut v2);
            $cas_pair(&mut v1, &mut v3);
            $cas_pair(&mut v4, &mut v6);
            $cas_pair(&mut v5, &mut v7);
            $cas_pair(&mut v0, &mut v1);
            $cas_pair(&mut v2, &mut v3);
            $cas_pair(&mut v4, &mut v5);
            $cas_pair(&mut v6, &mut v7);
            *v = [
                $finish_intra(v0),
                $finish_intra(v1),
                $finish_intra(v2),
                $finish_intra(v3),
                $finish_intra(v4),
                $finish_intra(v5),
                $finish_intra(v6),
                $finish_intra(v7),
            ];
        }

        #[inline(always)]
        fn $merge_8(block: &mut [$vec_ty]) {
            let [
                mut v0,
                mut v1,
                mut v2,
                mut v3,
                mut v4,
                mut v5,
                mut v6,
                mut v7,
            ] = block[..8]
            else {
                return;
            };
            $cas_pair(&mut v0, &mut v4);
            $cas_pair(&mut v1, &mut v5);
            $cas_pair(&mut v2, &mut v6);
            $cas_pair(&mut v3, &mut v7);
            let mut regs = [v0, v1, v2, v3, v4, v5, v6, v7];
            $finish_merge_8(&mut regs);
            block[..8].copy_from_slice(&regs);
        }

        #[inline(always)]
        fn $sort_2(block: &mut [$vec_ty]) {
            let [mut v0, mut v1] = block[..2] else {
                return;
            };
            v0 = $sort_1(v0);
            v1 = $sort_1(v1);
            $butterfly_pair(&mut v0, &mut v1);
            block[0] = $finish_intra(v0);
            block[1] = $finish_intra(v1);
        }

        #[inline(always)]
        fn $sort_4(block: &mut [$vec_ty]) {
            let [mut v0, mut v1, mut v2, mut v3] = block[..4] else {
                return;
            };
            v0 = $sort_1(v0);
            v1 = $sort_1(v1);
            v2 = $sort_1(v2);
            v3 = $sort_1(v3);
            $butterfly_pair(&mut v0, &mut v1);
            $butterfly_pair(&mut v2, &mut v3);
            v0 = $finish_intra(v0);
            v1 = $finish_intra(v1);
            v2 = $finish_intra(v2);
            v3 = $finish_intra(v3);
            $butterfly_pair(&mut v0, &mut v3);
            $butterfly_pair(&mut v1, &mut v2);
            $cas_pair(&mut v0, &mut v1);
            $cas_pair(&mut v2, &mut v3);
            block[0] = $finish_intra(v0);
            block[1] = $finish_intra(v1);
            block[2] = $finish_intra(v2);
            block[3] = $finish_intra(v3);
        }

        #[inline(always)]
        fn $sort_8(block: &mut [$vec_ty]) {
            let [
                mut v0,
                mut v1,
                mut v2,
                mut v3,
                mut v4,
                mut v5,
                mut v6,
                mut v7,
            ] = block[..8]
            else {
                return;
            };
            v0 = $sort_1(v0);
            v1 = $sort_1(v1);
            v2 = $sort_1(v2);
            v3 = $sort_1(v3);
            v4 = $sort_1(v4);
            v5 = $sort_1(v5);
            v6 = $sort_1(v6);
            v7 = $sort_1(v7);

            $butterfly_pair(&mut v0, &mut v1);
            $butterfly_pair(&mut v2, &mut v3);
            $butterfly_pair(&mut v4, &mut v5);
            $butterfly_pair(&mut v6, &mut v7);
            v0 = $finish_intra(v0);
            v1 = $finish_intra(v1);
            v2 = $finish_intra(v2);
            v3 = $finish_intra(v3);
            v4 = $finish_intra(v4);
            v5 = $finish_intra(v5);
            v6 = $finish_intra(v6);
            v7 = $finish_intra(v7);

            $butterfly_pair(&mut v0, &mut v3);
            $butterfly_pair(&mut v1, &mut v2);
            $butterfly_pair(&mut v4, &mut v7);
            $butterfly_pair(&mut v5, &mut v6);
            $cas_pair(&mut v0, &mut v1);
            $cas_pair(&mut v2, &mut v3);
            $cas_pair(&mut v4, &mut v5);
            $cas_pair(&mut v6, &mut v7);
            v0 = $finish_intra(v0);
            v1 = $finish_intra(v1);
            v2 = $finish_intra(v2);
            v3 = $finish_intra(v3);
            v4 = $finish_intra(v4);
            v5 = $finish_intra(v5);
            v6 = $finish_intra(v6);
            v7 = $finish_intra(v7);

            $butterfly_pair(&mut v0, &mut v7);
            $butterfly_pair(&mut v1, &mut v6);
            $butterfly_pair(&mut v2, &mut v5);
            $butterfly_pair(&mut v3, &mut v4);
            let mut regs = [v0, v1, v2, v3, v4, v5, v6, v7];
            $finish_merge_8(&mut regs);
            block[..8].copy_from_slice(&regs);
        }

        fn $sort_pow2(buf: &mut [$vec_ty]) {
            let nv = buf.len();
            match nv {
                1 => {
                    buf[0] = $sort_1(buf[0]);
                    return;
                }
                2 => {
                    $sort_2(buf);
                    return;
                }
                4 => {
                    $sort_4(buf);
                    return;
                }
                8 => {
                    $sort_8(buf);
                    return;
                }
                _ => {}
            }
            for block in buf.as_chunks_mut::<8>().0 {
                $sort_8(block);
            }
            for off in 0..8 {
                let p = 15 - off;
                let mut a = buf[off];
                let mut b = buf[p];
                $butterfly_pair(&mut a, &mut b);
                buf[off] = a;
                buf[p] = b;
            }
            $merge_8(&mut buf[0..8]);
            $merge_8(&mut buf[8..16]);
        }

        fn $sort_network(values: &mut [$elem]) {
            let n = values.len().next_power_of_two().max(16);
            let nv = n / 16;
            let mut buf = [$vec_ty::splat($max_val); 16];
            let (chunks, rem) = values.as_chunks::<16>();
            for (slot, chunk) in buf.iter_mut().zip(chunks) {
                *slot = $vec_ty::from_array(*chunk);
            }
            if !rem.is_empty() {
                let mut tail = [$max_val; 16];
                tail[..rem.len()].copy_from_slice(rem);
                buf[chunks.len()] = $vec_ty::from_array(tail);
            }
            $sort_pow2(&mut buf[..nv]);
            let full = chunks.len();
            let (out_chunks, out_rem) = values.as_chunks_mut::<16>();
            for (chunk, v) in out_chunks.iter_mut().zip(buf[..full].iter()) {
                *chunk = v.to_array();
            }
            if !out_rem.is_empty() {
                let tail = buf[full].to_array();
                out_rem.copy_from_slice(&tail[..out_rem.len()]);
            }
        }
    };
}

define_simd16_sort_16bit!(
    i16,
    i16::MAX,
    V16I,
    M16I,
    finish_intra_i16,
    sort_1_vec_i16,
    cas_pair_i16,
    butterfly_pair_i16,
    finish_merge_8_vecs_i16,
    merge_8_vecs_i16,
    sort_2_vecs_i16,
    sort_4_vecs_i16,
    sort_8_vecs_i16,
    bitonic_sort_pow2_i16,
    sort_i16_network
);

define_simd16_sort_16bit!(
    u16,
    u16::MAX,
    V16U,
    M16U,
    finish_intra_u16,
    sort_1_vec_u16,
    cas_pair_u16,
    butterfly_pair_u16,
    finish_merge_8_vecs_u16,
    merge_8_vecs_u16,
    sort_2_vecs_u16,
    sort_4_vecs_u16,
    sort_8_vecs_u16,
    bitonic_sort_pow2_u16,
    sort_u16_network
);

/// Sort `values` (length `BITONIC_MIN..=BITONIC_MAX`) with the branchless
/// network.  Shorter or longer slices are not handled here.
///
/// The slice is padded up to a power of two with `i64::MAX`.  Padding with the
/// maximum keeps every real element ahead of every pad element, so truncating
/// the result back to `values.len()` yields exactly the sorted real multiset —
/// including when the real data itself contains `i64::MAX`, because equal
/// values are indistinguishable in a value sort.
fn sort_i64_network(values: &mut [i64]) {
    let n = values.len().next_power_of_two();
    let nv = n / LANES;
    let mut buf = [V::splat(i64::MAX); MAX_VECS];
    let (chunks, rem) = values.as_chunks::<LANES>();
    for (slot, chunk) in buf.iter_mut().zip(chunks) {
        *slot = V::from_array(*chunk);
    }
    if !rem.is_empty() {
        let mut tail = [i64::MAX; LANES];
        tail[..rem.len()].copy_from_slice(rem);
        buf[chunks.len()] = V::from_array(tail);
    }
    bitonic_sort_pow2(&mut buf[..nv]);
    let full = chunks.len();
    let (out_chunks, out_rem) = values.as_chunks_mut::<LANES>();
    for (chunk, v) in out_chunks.iter_mut().zip(buf[..full].iter()) {
        *chunk = v.to_array();
    }
    if !out_rem.is_empty() {
        let tail = buf[full].to_array();
        out_rem.copy_from_slice(&tail[..out_rem.len()]);
    }
}

/// Sort a short `int64` value slice in ascending NumPy order.
///
/// `slice::sort_unstable` uses Rust's pattern-defeating quicksort with its
/// small-slice sorting network/insertion base cases.  It avoids Rayon setup on
/// the tiny flat Python route while retaining the proven large-array kernel.
/// THE BRANCHLESS NETWORK BELOW IS A MEASURED REJECT AND IS NOT ON THIS PATH.
/// `examples/h2h_sort_i64_kernel.rs`, worker `vmi1227854`, both kernels in one
/// process on one corpus: it loses at EVERY covered length — n=64 0.894x, n=96
/// 0.573x, n=128 0.910x, n=192 0.615x, n=256 **0.795x** (1678.1 ns against
/// `sort_unstable`'s 1333.4 ns).  The ledger's retry predicate for this cell
/// requires beating `sort_unstable` at n=256 in exactly that harness BEFORE any
/// route-level attempt, so the route-level attempt must not be made.  The code
/// is retained, reachable only from `sort_i64_network_forced`, so the reject
/// stays reproducible rather than becoming a claim someone has to re-derive.
#[inline]
pub fn sort_i64(values: &mut [i64]) {
    if (BITONIC_MIN..=32).contains(&values.len()) {
        sort_i64_network(values);
    } else {
        values.sort_unstable();
    }
}

#[inline]
pub fn sort_i32(values: &mut [i32]) {
    if (8..=BITONIC_MAX).contains(&values.len()) {
        sort_i32_network(values);
    } else {
        values.sort_unstable();
    }
}

#[inline]
pub fn sort_u64(values: &mut [u64]) {
    values.sort_unstable();
}

#[inline]
pub fn sort_u32(values: &mut [u32]) {
    if (8..=BITONIC_MAX).contains(&values.len()) {
        sort_u32_network(values);
    } else {
        values.sort_unstable();
    }
}

#[inline]
pub fn sort_i16(values: &mut [i16]) {
    if (8..=BITONIC_MAX).contains(&values.len()) {
        sort_i16_network(values);
    } else {
        values.sort_unstable();
    }
}

#[inline]
pub fn sort_u16(values: &mut [u16]) {
    if (8..=BITONIC_MAX).contains(&values.len()) {
        sort_u16_network(values);
    } else {
        values.sort_unstable();
    }
}

#[inline]
pub fn sort_i8(values: &mut [i8]) {
    if values.len() >= 64 {
        let mut counts = [0u16; 256];
        for &v in values.iter() {
            counts[(v as u8 ^ 0x80) as usize] += 1;
        }
        let mut off = 0usize;
        for (b, &c) in counts.iter().enumerate() {
            if c != 0 {
                let c = c as usize;
                values[off..off + c].fill((b as u8 ^ 0x80) as i8);
                off += c;
            }
        }
    } else {
        values.sort_unstable();
    }
}

#[inline]
pub fn sort_u8(values: &mut [u8]) {
    if values.len() >= 64 {
        let mut counts = [0u16; 256];
        let mut or_mask = 0u8;
        for &v in values.iter() {
            counts[v as usize] += 1;
            or_mask |= v;
        }
        if or_mask <= 1 {
            let zeros = counts[0] as usize;
            values[..zeros].fill(0);
            values[zeros..].fill(1);
        } else {
            let mut off = 0usize;
            for (b, &c) in counts.iter().enumerate() {
                if c != 0 {
                    let c = c as usize;
                    values[off..off + c].fill(b as u8);
                    off += c;
                }
            }
        }
    } else {
        values.sort_unstable();
    }
}

/// The comparison-sort arm, named so the standalone A/B harness can hold both
/// kernels as values of the same function type and interleave them.
pub fn sort_i64_unstable_forced(values: &mut [i64]) {
    values.sort_unstable();
}

/// Sort with the branchless network regardless of the env switch, for the
/// standalone A/B harness and the differential tests.  Falls back for lengths
/// the network does not cover.
pub fn sort_i64_network_forced(values: &mut [i64]) {
    if (BITONIC_MIN..=BITONIC_MAX).contains(&values.len()) {
        sort_i64_network(values);
    } else {
        values.sort_unstable();
    }
}

/// The 32-bit comparison-sort arm for the standalone A/B harness.
pub fn sort_i32_unstable_forced(values: &mut [i32]) {
    values.sort_unstable();
}

/// Sort `i32` with the 8-lane `std::simd` bitonic network for lengths `8..=256`.
pub fn sort_i32_network_forced(values: &mut [i32]) {
    if (8..=BITONIC_MAX).contains(&values.len()) {
        sort_i32_network(values);
    } else {
        values.sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BITONIC_MAX, BITONIC_MIN, sort_i8, sort_i16, sort_i32, sort_i64, sort_i64_network_forced,
        sort_u8, sort_u16, sort_u32, sort_u64,
    };

    #[test]
    fn i64_small_sort_orders_extrema_and_duplicates() {
        let mut values = [i64::MAX, 0, -7, i64::MIN, -7, 1, 0, -1];
        sort_i64(&mut values);
        assert_eq!(values, [i64::MIN, -7, -7, -1, 0, 0, 1, i64::MAX]);
    }

    #[test]
    fn i64_small_sort_planted_negative_is_not_left_unsorted() {
        let mut values = [2_i64, -1, 1];
        sort_i64(&mut values);
        assert_ne!(values, [2, -1, 1], "the route must perform the sort");
        assert_eq!(values, [-1, 1, 2]);
    }

    #[test]
    fn i32_small_sort_orders_extrema_and_duplicates() {
        let mut values = [i32::MAX, 0, -7, i32::MIN, -7, 1, 0, -1];
        sort_i32(&mut values);
        assert_eq!(values, [i32::MIN, -7, -7, -1, 0, 0, 1, i32::MAX]);
    }

    #[test]
    fn u64_small_sort_orders_extrema_and_duplicates() {
        let mut values = [u64::MAX, 0, 7, 100, 7, 1, 0, 42];
        sort_u64(&mut values);
        assert_eq!(values, [0, 0, 1, 7, 7, 42, 100, u64::MAX]);
    }

    #[test]
    fn u32_small_sort_orders_extrema_and_duplicates() {
        let mut values = [u32::MAX, 0, 7, 100, 7, 1, 0, 42];
        sort_u32(&mut values);
        assert_eq!(values, [0, 0, 1, 7, 7, 42, 100, u32::MAX]);
    }

    #[test]
    fn i16_small_sort_orders_extrema_and_duplicates() {
        let mut values = [i16::MAX, 0, -7, i16::MIN, -7, 1, 0, -1];
        sort_i16(&mut values);
        assert_eq!(values, [i16::MIN, -7, -7, -1, 0, 0, 1, i16::MAX]);
    }

    #[test]
    fn u16_small_sort_orders_extrema_and_duplicates() {
        let mut values = [u16::MAX, 0, 7, 100, 7, 1, 0, 42];
        sort_u16(&mut values);
        assert_eq!(values, [0, 0, 1, 7, 7, 42, 100, u16::MAX]);
    }

    #[test]
    fn i8_small_sort_orders_extrema_and_duplicates() {
        let mut values = [i8::MAX, 0, -7, i8::MIN, -7, 1, 0, -1];
        sort_i8(&mut values);
        assert_eq!(values, [i8::MIN, -7, -7, -1, 0, 0, 1, i8::MAX]);
    }

    #[test]
    fn u8_small_sort_orders_extrema_and_duplicates() {
        let mut values = [u8::MAX, 0, 7, 100, 7, 1, 0, 42];
        sort_u8(&mut values);
        assert_eq!(values, [0, 0, 1, 7, 7, 42, 100, u8::MAX]);
    }

    /// A bitonic network is a FIXED comparator sequence, so a length that is
    /// not a power of two, a run of equal keys, or a value equal to the pad
    /// sentinel are the three ways it silently returns the wrong multiset.
    /// Every covered length is checked against `sort_unstable` on eight input
    /// shapes, which is the corpus the ledger's AVX2 attempt used.
    #[test]
    fn i64_network_matches_sort_unstable_over_every_covered_length() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as i64
        };
        for len in BITONIC_MIN..=BITONIC_MAX {
            for shape in 0..8 {
                let src: Vec<i64> = (0..len)
                    .map(|i| match shape {
                        0 => next(),
                        1 => next() % 4,
                        2 => 7,
                        3 => i as i64,
                        4 => (len - i) as i64,
                        5 => i64::MAX,
                        6 => i64::MIN,
                        _ => {
                            if i % 2 == 0 {
                                i64::MIN
                            } else {
                                i64::MAX
                            }
                        }
                    })
                    .collect();
                let mut want = src.clone();
                want.sort_unstable();
                let mut got = src.clone();
                sort_i64_network_forced(&mut got);
                assert_eq!(got, want, "len={len} shape={shape}");
            }
        }
    }

    /// Lengths outside the window must still sort — the dispatch is part of the
    /// kernel's contract, not an implementation detail.
    #[test]
    fn i64_sort_dispatch_covers_lengths_around_the_network_window() {
        for len in [0usize, 1, 2, 3, 63, 64, 255, 256, 257, 1000] {
            let src: Vec<i64> = (0..len).map(|i| ((len - i) as i64) * 3 - 5).collect();
            let mut want = src.clone();
            want.sort_unstable();
            let mut got = src.clone();
            sort_i64(&mut got);
            assert_eq!(got, want, "len={len}");
        }
    }

    #[test]
    fn i32_u32_i8_u8_match_sort_unstable_over_every_covered_length() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in 0..=BITONIC_MAX + 1 {
            for shape in 0..8 {
                let src_i32: Vec<i32> = (0..len)
                    .map(|i| match shape {
                        0 => next() as i32,
                        1 => (next() as i32) % 4,
                        2 => 7,
                        3 => i as i32,
                        4 => (len - i) as i32,
                        5 => i32::MAX,
                        6 => i32::MIN,
                        _ => {
                            if i % 2 == 0 {
                                i32::MIN
                            } else {
                                i32::MAX
                            }
                        }
                    })
                    .collect();
                let mut want_i32 = src_i32.clone();
                want_i32.sort_unstable();
                let mut got_i32 = src_i32;
                sort_i32(&mut got_i32);
                assert_eq!(got_i32, want_i32, "i32 len={len} shape={shape}");

                let src_u32: Vec<u32> = (0..len)
                    .map(|i| match shape {
                        0 => next() as u32,
                        1 => (next() as u32) % 4,
                        2 => 7,
                        3 => i as u32,
                        4 => (len - i) as u32,
                        5 => u32::MAX,
                        6 => 0,
                        _ => {
                            if i % 2 == 0 {
                                0
                            } else {
                                u32::MAX
                            }
                        }
                    })
                    .collect();
                let mut want_u32 = src_u32.clone();
                want_u32.sort_unstable();
                let mut got_u32 = src_u32;
                sort_u32(&mut got_u32);
                assert_eq!(got_u32, want_u32, "u32 len={len} shape={shape}");

                let src_i16: Vec<i16> = (0..len)
                    .map(|i| match shape {
                        0 => next() as i16,
                        1 => (next() as i16) % 4,
                        2 => -7,
                        3 => i as i16,
                        4 => (len - i) as i16,
                        5 => i16::MAX,
                        6 => i16::MIN,
                        _ => {
                            if i % 2 == 0 {
                                i16::MIN
                            } else {
                                i16::MAX
                            }
                        }
                    })
                    .collect();
                let mut want_i16 = src_i16.clone();
                want_i16.sort_unstable();
                let mut got_i16 = src_i16;
                sort_i16(&mut got_i16);
                assert_eq!(got_i16, want_i16, "i16 len={len} shape={shape}");

                let src_u16: Vec<u16> = (0..len)
                    .map(|i| match shape {
                        0 => next() as u16,
                        1 => (next() as u16) % 4,
                        2 => 7,
                        3 => i as u16,
                        4 => (len - i) as u16,
                        5 => u16::MAX,
                        6 => 0,
                        _ => {
                            if i % 2 == 0 {
                                0
                            } else {
                                u16::MAX
                            }
                        }
                    })
                    .collect();
                let mut want_u16 = src_u16.clone();
                want_u16.sort_unstable();
                let mut got_u16 = src_u16;
                sort_u16(&mut got_u16);
                assert_eq!(got_u16, want_u16, "u16 len={len} shape={shape}");

                let src_i8: Vec<i8> = (0..len)
                    .map(|i| match shape {
                        0 => next() as i8,
                        1 => (next() as i8) % 2,
                        2 => -7,
                        3 => i as i8,
                        4 => (len - i) as i8,
                        5 => i8::MAX,
                        6 => i8::MIN,
                        _ => {
                            if i % 2 == 0 {
                                i8::MIN
                            } else {
                                i8::MAX
                            }
                        }
                    })
                    .collect();
                let mut want_i8 = src_i8.clone();
                want_i8.sort_unstable();
                let mut got_i8 = src_i8;
                sort_i8(&mut got_i8);
                assert_eq!(got_i8, want_i8, "i8 len={len} shape={shape}");

                let src_u8: Vec<u8> = (0..len)
                    .map(|i| match shape {
                        0 => next() as u8,
                        1 => (next() as u8) & 1,
                        2 => 0,
                        3 => 1,
                        4 => (len - i) as u8,
                        5 => u8::MAX,
                        6 => 0,
                        _ => {
                            if i % 2 == 0 {
                                0
                            } else {
                                u8::MAX
                            }
                        }
                    })
                    .collect();
                let mut want_u8 = src_u8.clone();
                want_u8.sort_unstable();
                let mut got_u8 = src_u8;
                sort_u8(&mut got_u8);
                assert_eq!(got_u8, want_u8, "u8 len={len} shape={shape}");
            }
        }
    }
}
