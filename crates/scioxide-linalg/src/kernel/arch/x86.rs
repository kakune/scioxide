//! x86/x86-64 AVX2 and AVX-512 leaves: packed f32/f64 GEMM tiles, dots,
//! finite maxima, bounded affine updates and GKO line updates.

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd,
    _mm256_mul_ps, _mm256_set1_pd, _mm256_set1_ps, _mm256_setzero_pd, _mm256_setzero_ps,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm512_fmadd_pd, _mm512_fmadd_ps, _mm512_loadu_pd,
    _mm512_loadu_ps, _mm512_mul_pd, _mm512_mul_ps, _mm512_set1_pd, _mm512_set1_ps,
    _mm512_setzero_pd, _mm512_setzero_ps, _mm512_storeu_pd, _mm512_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd,
    _mm256_mul_ps, _mm256_set1_pd, _mm256_set1_ps, _mm256_setzero_pd, _mm256_setzero_ps,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm512_fmadd_pd, _mm512_fmadd_ps, _mm512_loadu_pd,
    _mm512_loadu_ps, _mm512_mul_pd, _mm512_mul_ps, _mm512_set1_pd, _mm512_set1_ps,
    _mm512_setzero_pd, _mm512_setzero_ps, _mm512_storeu_pd, _mm512_storeu_ps,
};

use scioxide_core::NumericScalar;
use scioxide_core::backend::{KernelBackend, KernelDispatch, ScalarKind};

use super::super::microkernel::Microkernel;

pub(super) fn select_microkernel<S: NumericScalar>(
    dispatch: KernelDispatch,
) -> Option<Microkernel> {
    match (S::kind(), dispatch.backend()) {
        (ScalarKind::F32, KernelBackend::Avx2Fma) => {
            Some(Microkernel::new_f32(avx2_f32, 4, 16, dispatch))
        }
        (ScalarKind::F32, KernelBackend::Avx512) => {
            Some(Microkernel::new_f32(avx512_f32, 4, 32, dispatch))
        }
        (ScalarKind::F64, KernelBackend::Avx2Fma) => {
            Some(Microkernel::new_f64(avx2_f64, 6, 8, dispatch))
        }
        (ScalarKind::F64, KernelBackend::Avx512) => {
            Some(Microkernel::new_f64(avx512_f64, 4, 16, dispatch))
        }
        _ => None,
    }
}

fn avx2_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row_stride: usize,
    depth: usize,
    beta: f32,
    dispatch: KernelDispatch,
) -> bool {
    if dispatch.backend() != KernelBackend::Avx2Fma
        || !super::super::microkernel::panel_fits(
            4,
            16,
            a.len(),
            b.len(),
            c.len(),
            row_stride,
            depth,
        )
    {
        return false;
    }
    // SAFETY: the verified token proves Avx2Fma; panel_fits checks every packed
    // input and output span with checked arithmetic. Unaligned loads accept scalar
    // alignment, slices are initialized, and exclusive C cannot alias either input.
    unsafe { avx2_f32_leaf(a, b, c, row_stride, depth, beta) }
    true
}

fn avx2_f64(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    row_stride: usize,
    depth: usize,
    beta: f64,
    dispatch: KernelDispatch,
) -> bool {
    if dispatch.backend() != KernelBackend::Avx2Fma
        || !super::super::microkernel::panel_fits(
            6,
            8,
            a.len(),
            b.len(),
            c.len(),
            row_stride,
            depth,
        )
    {
        return false;
    }
    // SAFETY: the verified token proves Avx2Fma; panel_fits checks every packed
    // input and output span with checked arithmetic. Unaligned loads accept scalar
    // alignment, slices are initialized, and exclusive C cannot alias either input.
    unsafe { avx2_f64_leaf(a, b, c, row_stride, depth, beta) }
    true
}

fn avx512_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row_stride: usize,
    depth: usize,
    beta: f32,
    dispatch: KernelDispatch,
) -> bool {
    if dispatch.backend() != KernelBackend::Avx512
        || !super::super::microkernel::panel_fits(
            4,
            32,
            a.len(),
            b.len(),
            c.len(),
            row_stride,
            depth,
        )
    {
        return false;
    }
    // SAFETY: the verified token proves Avx512; panel_fits checks every packed
    // input and output span with checked arithmetic. Unaligned loads accept scalar
    // alignment, slices are initialized, and exclusive C cannot alias either input.
    unsafe { avx512_f32_leaf(a, b, c, row_stride, depth, beta) }
    true
}

fn avx512_f64(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    row_stride: usize,
    depth: usize,
    beta: f64,
    dispatch: KernelDispatch,
) -> bool {
    if dispatch.backend() != KernelBackend::Avx512
        || !super::super::microkernel::panel_fits(
            4,
            16,
            a.len(),
            b.len(),
            c.len(),
            row_stride,
            depth,
        )
    {
        return false;
    }
    // SAFETY: the verified token proves Avx512; panel_fits checks every packed
    // input and output span with checked arithmetic. Unaligned loads accept scalar
    // alignment, slices are initialized, and exclusive C cannot alias either input.
    unsafe { avx512_f64_leaf(a, b, c, row_stride, depth, beta) }
    true
}

/// Apply a packed 4x16 f32 AVX2/FMA microkernel.
///
/// # Safety
///
/// - Alignment: all loads/stores are unaligned.
/// - Length: A has `depth*4`, B has `depth*16`, and C exposes four rows of 16.
/// - Aliasing: packed inputs do not overlap the exclusive output.
/// - Initialization: packed inputs and any C lanes read for nonzero beta are initialized.
/// - Target features: AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_f32_leaf(a: &[f32], b: &[f32], c: &mut [f32], rs: usize, depth: usize, beta: f32) {
    debug_assert!(a.len() >= depth * 4 && b.len() >= depth * 16);
    debug_assert!(c.len() >= 3 * rs + 16);
    let mut acc = [[_mm256_setzero_ps(); 2]; 4];
    let mut row = 0;
    while row < 4 {
        if beta != 0.0 {
            let scale = _mm256_set1_ps(beta);
            let mut group = 0;
            while group < 2 {
                // SAFETY: the C span contract covers every complete unaligned vector.
                // Unaligned vector accesses need only scalar alignment. Packed inputs and
                // C are initialized; shared A/B do not alias exclusive C. Host-verified
                // dispatch proves the leaf's target features.
                let value = unsafe { _mm256_loadu_ps(c.as_ptr().add(row * rs + group * 8)) };
                acc[row][group] = if beta == 1.0 {
                    value
                } else {
                    _mm256_mul_ps(value, scale)
                };
                group += 1;
            }
        }
        row += 1;
    }
    let mut inner = 0;
    while inner < depth {
        // SAFETY: the packed B contract covers two complete vectors per depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b0 = unsafe { _mm256_loadu_ps(b.as_ptr().add(inner * 16)) };
        // SAFETY: the packed B contract covers the second vector at this depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b1 = unsafe { _mm256_loadu_ps(b.as_ptr().add(inner * 16 + 8)) };
        row = 0;
        while row < 4 {
            let av = _mm256_set1_ps(a[inner * 4 + row]);
            acc[row][0] = _mm256_fmadd_ps(av, b0, acc[row][0]);
            acc[row][1] = _mm256_fmadd_ps(av, b1, acc[row][1]);
            row += 1;
        }
        inner += 1;
    }
    row = 0;
    while row < 4 {
        // SAFETY: the C span contract covers both complete output vectors. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        unsafe {
            _mm256_storeu_ps(c.as_mut_ptr().add(row * rs), acc[row][0]);
            _mm256_storeu_ps(c.as_mut_ptr().add(row * rs + 8), acc[row][1]);
        }
        row += 1;
    }
}

/// Apply a packed 6x8 f64 AVX2/FMA microkernel.
///
/// # Safety
///
/// - Alignment: all loads/stores are unaligned.
/// - Length: A has `depth*6`, B has `depth*8`, and C exposes six rows of 8.
/// - Aliasing: packed inputs do not overlap the exclusive output.
/// - Initialization: packed inputs and any C lanes read for nonzero beta are initialized.
/// - Target features: AVX2 and FMA are available.
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_f64_leaf(a: &[f64], b: &[f64], c: &mut [f64], rs: usize, depth: usize, beta: f64) {
    debug_assert!(a.len() >= depth * 6 && b.len() >= depth * 8);
    debug_assert!(c.len() >= 5 * rs + 8);
    // A 6x8 tile keeps twelve accumulators, two B vectors and one broadcast A
    // value live, i.e. 15 of the 16 ymm registers. The accumulators are named
    // individually (not an array) so they stay in registers without spills.
    let mut c00 = _mm256_setzero_pd();
    let mut c01 = _mm256_setzero_pd();
    let mut c10 = _mm256_setzero_pd();
    let mut c11 = _mm256_setzero_pd();
    let mut c20 = _mm256_setzero_pd();
    let mut c21 = _mm256_setzero_pd();
    let mut c30 = _mm256_setzero_pd();
    let mut c31 = _mm256_setzero_pd();
    let mut c40 = _mm256_setzero_pd();
    let mut c41 = _mm256_setzero_pd();
    let mut c50 = _mm256_setzero_pd();
    let mut c51 = _mm256_setzero_pd();
    if beta != 0.0 {
        // SAFETY: the C span contract covers all twelve complete unaligned vectors.
        // Unaligned vector accesses need only scalar alignment. Packed inputs and C are
        // initialized; shared A/B do not alias exclusive C. Host-verified dispatch proves
        // the leaf's target features.
        unsafe {
            c00 = _mm256_loadu_pd(c.as_ptr());
            c01 = _mm256_loadu_pd(c.as_ptr().add(4));
            c10 = _mm256_loadu_pd(c.as_ptr().add(rs));
            c11 = _mm256_loadu_pd(c.as_ptr().add(rs + 4));
            c20 = _mm256_loadu_pd(c.as_ptr().add(2 * rs));
            c21 = _mm256_loadu_pd(c.as_ptr().add(2 * rs + 4));
            c30 = _mm256_loadu_pd(c.as_ptr().add(3 * rs));
            c31 = _mm256_loadu_pd(c.as_ptr().add(3 * rs + 4));
            c40 = _mm256_loadu_pd(c.as_ptr().add(4 * rs));
            c41 = _mm256_loadu_pd(c.as_ptr().add(4 * rs + 4));
            c50 = _mm256_loadu_pd(c.as_ptr().add(5 * rs));
            c51 = _mm256_loadu_pd(c.as_ptr().add(5 * rs + 4));
        }
        if beta != 1.0 {
            let scale = _mm256_set1_pd(beta);
            c00 = _mm256_mul_pd(c00, scale);
            c01 = _mm256_mul_pd(c01, scale);
            c10 = _mm256_mul_pd(c10, scale);
            c11 = _mm256_mul_pd(c11, scale);
            c20 = _mm256_mul_pd(c20, scale);
            c21 = _mm256_mul_pd(c21, scale);
            c30 = _mm256_mul_pd(c30, scale);
            c31 = _mm256_mul_pd(c31, scale);
            c40 = _mm256_mul_pd(c40, scale);
            c41 = _mm256_mul_pd(c41, scale);
            c50 = _mm256_mul_pd(c50, scale);
            c51 = _mm256_mul_pd(c51, scale);
        }
    }
    let mut inner = 0;
    while inner < depth {
        // SAFETY: the packed B contract covers two complete vectors per depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b0 = unsafe { _mm256_loadu_pd(b.as_ptr().add(inner * 8)) };
        // SAFETY: the packed B contract covers the second vector at this depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b1 = unsafe { _mm256_loadu_pd(b.as_ptr().add(inner * 8 + 4)) };
        // Packed A holds six scalars per depth; the bounds-checked slice
        // lets the constant indices below compile without further checks.
        let row = &a[inner * 6..][..6];
        let a0 = _mm256_set1_pd(row[0]);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        let a1 = _mm256_set1_pd(row[1]);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        let a2 = _mm256_set1_pd(row[2]);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        let a3 = _mm256_set1_pd(row[3]);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        let a4 = _mm256_set1_pd(row[4]);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        let a5 = _mm256_set1_pd(row[5]);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        inner += 1;
    }
    // SAFETY: the C span contract covers all twelve complete output vectors. Unaligned
    // vector accesses need only scalar alignment. Packed inputs and C are initialized;
    // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's target
    // features.
    unsafe {
        _mm256_storeu_pd(c.as_mut_ptr(), c00);
        _mm256_storeu_pd(c.as_mut_ptr().add(4), c01);
        _mm256_storeu_pd(c.as_mut_ptr().add(rs), c10);
        _mm256_storeu_pd(c.as_mut_ptr().add(rs + 4), c11);
        _mm256_storeu_pd(c.as_mut_ptr().add(2 * rs), c20);
        _mm256_storeu_pd(c.as_mut_ptr().add(2 * rs + 4), c21);
        _mm256_storeu_pd(c.as_mut_ptr().add(3 * rs), c30);
        _mm256_storeu_pd(c.as_mut_ptr().add(3 * rs + 4), c31);
        _mm256_storeu_pd(c.as_mut_ptr().add(4 * rs), c40);
        _mm256_storeu_pd(c.as_mut_ptr().add(4 * rs + 4), c41);
        _mm256_storeu_pd(c.as_mut_ptr().add(5 * rs), c50);
        _mm256_storeu_pd(c.as_mut_ptr().add(5 * rs + 4), c51);
    }
}

/// Apply a packed 4x32 f32 AVX-512 microkernel.
///
/// # Safety
///
/// - Alignment: all loads/stores are unaligned.
/// - Length: A has `depth*4`, B has `depth*32`, and C exposes four rows of 32.
/// - Aliasing: packed inputs do not overlap the exclusive output.
/// - Initialization: packed inputs and any C lanes read for nonzero beta are initialized.
/// - Target features: AVX-512F, AVX-512DQ, and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn avx512_f32_leaf(a: &[f32], b: &[f32], c: &mut [f32], rs: usize, depth: usize, beta: f32) {
    debug_assert!(a.len() >= depth * 4 && b.len() >= depth * 32);
    debug_assert!(c.len() >= 3 * rs + 32);
    let mut acc = [[_mm512_setzero_ps(); 2]; 4];
    let mut row = 0;
    while row < 4 {
        if beta != 0.0 {
            let scale = _mm512_set1_ps(beta);
            let mut group = 0;
            while group < 2 {
                // SAFETY: the C span contract covers every complete unaligned vector.
                // Unaligned vector accesses need only scalar alignment. Packed inputs and
                // C are initialized; shared A/B do not alias exclusive C. Host-verified
                // dispatch proves the leaf's target features.
                let value = unsafe { _mm512_loadu_ps(c.as_ptr().add(row * rs + group * 16)) };
                acc[row][group] = if beta == 1.0 {
                    value
                } else {
                    _mm512_mul_ps(value, scale)
                };
                group += 1;
            }
        }
        row += 1;
    }
    let mut inner = 0;
    while inner < depth {
        // SAFETY: the packed B contract covers two complete vectors per depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b0 = unsafe { _mm512_loadu_ps(b.as_ptr().add(inner * 32)) };
        // SAFETY: the packed B contract covers the second vector at this depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b1 = unsafe { _mm512_loadu_ps(b.as_ptr().add(inner * 32 + 16)) };
        row = 0;
        while row < 4 {
            let av = _mm512_set1_ps(a[inner * 4 + row]);
            acc[row][0] = _mm512_fmadd_ps(av, b0, acc[row][0]);
            acc[row][1] = _mm512_fmadd_ps(av, b1, acc[row][1]);
            row += 1;
        }
        inner += 1;
    }
    row = 0;
    while row < 4 {
        // SAFETY: the C span contract covers both complete output vectors. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        unsafe {
            _mm512_storeu_ps(c.as_mut_ptr().add(row * rs), acc[row][0]);
            _mm512_storeu_ps(c.as_mut_ptr().add(row * rs + 16), acc[row][1]);
        }
        row += 1;
    }
}

/// Apply a packed 4x16 f64 AVX-512 microkernel.
///
/// # Safety
///
/// - Alignment: all loads/stores are unaligned.
/// - Length: A has `depth*4`, B has `depth*16`, and C exposes four rows of 16.
/// - Aliasing: packed inputs do not overlap the exclusive output.
/// - Initialization: packed inputs and any C lanes read for nonzero beta are initialized.
/// - Target features: AVX-512F, AVX-512DQ, and AVX-512VL are available.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn avx512_f64_leaf(a: &[f64], b: &[f64], c: &mut [f64], rs: usize, depth: usize, beta: f64) {
    debug_assert!(a.len() >= depth * 4 && b.len() >= depth * 16);
    debug_assert!(c.len() >= 3 * rs + 16);
    let mut acc = [[_mm512_setzero_pd(); 2]; 4];
    let mut row = 0;
    while row < 4 {
        if beta != 0.0 {
            let scale = _mm512_set1_pd(beta);
            let mut group = 0;
            while group < 2 {
                // SAFETY: the C span contract covers every complete unaligned vector.
                // Unaligned vector accesses need only scalar alignment. Packed inputs and
                // C are initialized; shared A/B do not alias exclusive C. Host-verified
                // dispatch proves the leaf's target features.
                let value = unsafe { _mm512_loadu_pd(c.as_ptr().add(row * rs + group * 8)) };
                acc[row][group] = if beta == 1.0 {
                    value
                } else {
                    _mm512_mul_pd(value, scale)
                };
                group += 1;
            }
        }
        row += 1;
    }
    let mut inner = 0;
    while inner < depth {
        // SAFETY: the packed B contract covers two complete vectors per depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b0 = unsafe { _mm512_loadu_pd(b.as_ptr().add(inner * 16)) };
        // SAFETY: the packed B contract covers the second vector at this depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b1 = unsafe { _mm512_loadu_pd(b.as_ptr().add(inner * 16 + 8)) };
        row = 0;
        while row < 4 {
            let av = _mm512_set1_pd(a[inner * 4 + row]);
            acc[row][0] = _mm512_fmadd_pd(av, b0, acc[row][0]);
            acc[row][1] = _mm512_fmadd_pd(av, b1, acc[row][1]);
            row += 1;
        }
        inner += 1;
    }
    row = 0;
    while row < 4 {
        // SAFETY: the C span contract covers both complete output vectors. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        unsafe {
            _mm512_storeu_pd(c.as_mut_ptr().add(row * rs), acc[row][0]);
            _mm512_storeu_pd(c.as_mut_ptr().add(row * rs + 8), acc[row][1]);
        }
        row += 1;
    }
}

pub(super) fn dot<S: NumericScalar>(
    left: &[S],
    right: &[S],
    dispatch: KernelDispatch,
) -> Result<Option<S>, scioxide_core::SciOxideError> {
    if left.len() != right.len() {
        return Err(scioxide_core::SciOxideError::dimension_mismatch(
            [left.len()],
            [right.len()],
        ));
    }

    use scioxide_core::backend::{ScalarSlice, ScalarValue};
    match (S::slice(left), S::slice(right), dispatch.backend()) {
        (ScalarSlice::F32(left), ScalarSlice::F32(right), KernelBackend::Avx2Fma) => {
            // SAFETY: the verified dispatch proves avx2,fma; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            Ok(Some(S::from_value(ScalarValue::F32(unsafe {
                dot_f32_256(left, right)
            }))?))
        }
        (ScalarSlice::F64(left), ScalarSlice::F64(right), KernelBackend::Avx2Fma) => {
            // SAFETY: the verified dispatch proves avx2,fma; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            Ok(Some(S::from_value(ScalarValue::F64(unsafe {
                dot_f64_256(left, right)
            }))?))
        }
        (ScalarSlice::F32(left), ScalarSlice::F32(right), KernelBackend::Avx512) => {
            // SAFETY: the verified dispatch proves avx512f,avx512dq,avx512vl; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            Ok(Some(S::from_value(ScalarValue::F32(unsafe {
                dot_f32_512(left, right)
            }))?))
        }
        (ScalarSlice::F64(left), ScalarSlice::F64(right), KernelBackend::Avx512) => {
            // SAFETY: the verified dispatch proves avx512f,avx512dq,avx512vl; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            Ok(Some(S::from_value(ScalarValue::F64(unsafe {
                dot_f64_512(left, right)
            }))?))
        }
        _ => Ok(None),
    }
}

/// Contiguous dot product with fixed lane order.
/// # Safety
/// The caller proves avx2,fma and equal initialized slice lengths.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_256(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 4];
    let mut offset = 0;
    while offset + 32 <= left.len() {
        for (lane, sum) in sums.iter_mut().enumerate() {
            // SAFETY: the block bound covers all four vectors in both slices.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx2,fma.
            let (a, b) = unsafe {
                (
                    _mm256_loadu_ps(left.as_ptr().add(offset + lane * 8)),
                    _mm256_loadu_ps(right.as_ptr().add(offset + lane * 8)),
                )
            };
            *sum = _mm256_fmadd_ps(a, b, *sum);
        }
        offset += 32;
    }
    let mut sum = _mm256_add_ps(
        _mm256_add_ps(sums[0], sums[1]),
        _mm256_add_ps(sums[2], sums[3]),
    );
    while offset + 8 <= left.len() {
        // SAFETY: a complete unaligned vector lies in each equal-length slice.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx2,fma.
        let (a, b) = unsafe {
            (
                _mm256_loadu_ps(left.as_ptr().add(offset)),
                _mm256_loadu_ps(right.as_ptr().add(offset)),
            )
        };
        sum = _mm256_fmadd_ps(a, b, sum);
        offset += 8;
    }
    let mut lanes = [0.0; 8];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx2,fma.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().fold(0.0, |sum, value| sum + value);
    for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
        result = scioxide_core::RealScalar::mul_add(a, b, result);
    }
    result
}

/// Contiguous dot product with fixed lane order.
/// # Safety
/// The caller proves avx2,fma and equal initialized slice lengths.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f64_256(left: &[f64], right: &[f64]) -> f64 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let mut sums = [_mm256_setzero_pd(); 4];
    let mut offset = 0;
    while offset + 16 <= left.len() {
        for (lane, sum) in sums.iter_mut().enumerate() {
            // SAFETY: the block bound covers all four vectors in both slices.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx2,fma.
            let (a, b) = unsafe {
                (
                    _mm256_loadu_pd(left.as_ptr().add(offset + lane * 4)),
                    _mm256_loadu_pd(right.as_ptr().add(offset + lane * 4)),
                )
            };
            *sum = _mm256_fmadd_pd(a, b, *sum);
        }
        offset += 16;
    }
    let mut sum = _mm256_add_pd(
        _mm256_add_pd(sums[0], sums[1]),
        _mm256_add_pd(sums[2], sums[3]),
    );
    while offset + 4 <= left.len() {
        // SAFETY: a complete unaligned vector lies in each equal-length slice.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx2,fma.
        let (a, b) = unsafe {
            (
                _mm256_loadu_pd(left.as_ptr().add(offset)),
                _mm256_loadu_pd(right.as_ptr().add(offset)),
            )
        };
        sum = _mm256_fmadd_pd(a, b, sum);
        offset += 4;
    }
    let mut lanes = [0.0; 4];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx2,fma.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().fold(0.0, |sum, value| sum + value);
    for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
        result = scioxide_core::RealScalar::mul_add(a, b, result);
    }
    result
}

/// Contiguous dot product with fixed lane order.
/// # Safety
/// The caller proves avx512f,avx512dq,avx512vl and equal initialized slice lengths.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn dot_f32_512(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let mut sums = [_mm512_setzero_ps(); 4];
    let mut offset = 0;
    while offset + 64 <= left.len() {
        for (lane, sum) in sums.iter_mut().enumerate() {
            // SAFETY: the block bound covers all four vectors in both slices.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
            let (a, b) = unsafe {
                (
                    _mm512_loadu_ps(left.as_ptr().add(offset + lane * 16)),
                    _mm512_loadu_ps(right.as_ptr().add(offset + lane * 16)),
                )
            };
            *sum = _mm512_fmadd_ps(a, b, *sum);
        }
        offset += 64;
    }
    let mut sum = _mm512_add_ps(
        _mm512_add_ps(sums[0], sums[1]),
        _mm512_add_ps(sums[2], sums[3]),
    );
    while offset + 16 <= left.len() {
        // SAFETY: a complete unaligned vector lies in each equal-length slice.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
        let (a, b) = unsafe {
            (
                _mm512_loadu_ps(left.as_ptr().add(offset)),
                _mm512_loadu_ps(right.as_ptr().add(offset)),
            )
        };
        sum = _mm512_fmadd_ps(a, b, sum);
        offset += 16;
    }
    let mut lanes = [0.0; 16];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().fold(0.0, |sum, value| sum + value);
    for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
        result = scioxide_core::RealScalar::mul_add(a, b, result);
    }
    result
}

/// Contiguous dot product with fixed lane order.
/// # Safety
/// The caller proves avx512f,avx512dq,avx512vl and equal initialized slice lengths.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn dot_f64_512(left: &[f64], right: &[f64]) -> f64 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let mut sums = [_mm512_setzero_pd(); 4];
    let mut offset = 0;
    while offset + 32 <= left.len() {
        for (lane, sum) in sums.iter_mut().enumerate() {
            // SAFETY: the block bound covers all four vectors in both slices.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
            let (a, b) = unsafe {
                (
                    _mm512_loadu_pd(left.as_ptr().add(offset + lane * 8)),
                    _mm512_loadu_pd(right.as_ptr().add(offset + lane * 8)),
                )
            };
            *sum = _mm512_fmadd_pd(a, b, *sum);
        }
        offset += 32;
    }
    let mut sum = _mm512_add_pd(
        _mm512_add_pd(sums[0], sums[1]),
        _mm512_add_pd(sums[2], sums[3]),
    );
    while offset + 8 <= left.len() {
        // SAFETY: a complete unaligned vector lies in each equal-length slice.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
        let (a, b) = unsafe {
            (
                _mm512_loadu_pd(left.as_ptr().add(offset)),
                _mm512_loadu_pd(right.as_ptr().add(offset)),
            )
        };
        sum = _mm512_fmadd_pd(a, b, sum);
        offset += 8;
    }
    let mut lanes = [0.0; 8];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
    unsafe { _mm512_storeu_pd(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().fold(0.0, |sum, value| sum + value);
    for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
        result = scioxide_core::RealScalar::mul_add(a, b, result);
    }
    result
}

pub(super) fn maximum<S: NumericScalar>(
    values: &[S],
    dispatch: KernelDispatch,
) -> Result<Option<S::Real>, scioxide_core::SciOxideError> {
    use scioxide_core::backend::{ScalarSlice, ScalarValue};
    match (S::slice(values), dispatch.backend()) {
        (ScalarSlice::F32(values), KernelBackend::Avx2Fma) => {
            // SAFETY: verified dispatch proves avx2,fma; leaves bound every load.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            let maximum = unsafe { maximum_f32_256(values) }?;
            Ok(Some(
                S::from_value(ScalarValue::F32(maximum))?.real_component(),
            ))
        }
        (ScalarSlice::F64(values), KernelBackend::Avx2Fma) => {
            // SAFETY: verified dispatch proves avx2,fma; leaves bound every load.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            let maximum = unsafe { maximum_f64_256(values) }?;
            Ok(Some(
                S::from_value(ScalarValue::F64(maximum))?.real_component(),
            ))
        }
        (ScalarSlice::F32(values), KernelBackend::Avx512) => {
            // SAFETY: verified dispatch proves avx512f,avx512dq,avx512vl; leaves bound every load.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            let maximum = unsafe { maximum_f32_512(values) }?;
            Ok(Some(
                S::from_value(ScalarValue::F32(maximum))?.real_component(),
            ))
        }
        (ScalarSlice::F64(values), KernelBackend::Avx512) => {
            // SAFETY: verified dispatch proves avx512f,avx512dq,avx512vl; leaves bound every load.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            let maximum = unsafe { maximum_f64_512(values) }?;
            Ok(Some(
                S::from_value(ScalarValue::F64(maximum))?.real_component(),
            ))
        }
        _ => Ok(None),
    }
}

pub(super) fn affine<S: NumericScalar>(
    alpha: S,
    left: &[S],
    beta: S,
    right: &[S],
    output: &mut [S],
    dispatch: KernelDispatch,
) -> bool {
    use scioxide_core::backend::{ScalarSlice, ScalarSliceMut, ScalarValue};
    // This safe boundary also defends direct calls within the architecture module.
    if left.len() != right.len() || output.len() != left.len() {
        return false;
    }
    match (
        alpha.value(),
        S::slice(left),
        beta.value(),
        S::slice(right),
        S::slice_mut(output),
        dispatch.backend(),
    ) {
        (
            ScalarValue::F32(alpha),
            ScalarSlice::F32(left),
            ScalarValue::F32(beta),
            ScalarSlice::F32(right),
            ScalarSliceMut::F32(output),
            KernelBackend::Avx2Fma,
        ) => {
            // SAFETY: verified dispatch proves avx2,fma; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            unsafe { affine_f32_256(alpha, left, beta, right, output) };
            true
        }
        (
            ScalarValue::F64(alpha),
            ScalarSlice::F64(left),
            ScalarValue::F64(beta),
            ScalarSlice::F64(right),
            ScalarSliceMut::F64(output),
            KernelBackend::Avx2Fma,
        ) => {
            // SAFETY: verified dispatch proves avx2,fma; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            unsafe { affine_f64_256(alpha, left, beta, right, output) };
            true
        }
        (
            ScalarValue::F32(alpha),
            ScalarSlice::F32(left),
            ScalarValue::F32(beta),
            ScalarSlice::F32(right),
            ScalarSliceMut::F32(output),
            KernelBackend::Avx512,
        ) => {
            // SAFETY: verified dispatch proves avx512f,avx512dq,avx512vl; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            unsafe { affine_f32_512(alpha, left, beta, right, output) };
            true
        }
        (
            ScalarValue::F64(alpha),
            ScalarSlice::F64(left),
            ScalarValue::F64(beta),
            ScalarSlice::F64(right),
            ScalarSliceMut::F64(output),
            KernelBackend::Avx512,
        ) => {
            // SAFETY: verified dispatch proves avx512f,avx512dq,avx512vl; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            unsafe { affine_f64_512(alpha, left, beta, right, output) };
            true
        }
        _ => false,
    }
}

/// Validate and bound contiguous components without scalar checks per lane.
/// # Safety
/// The caller proves avx2,fma; loads stay within the initialized input slice.
#[target_feature(enable = "avx2,fma")]
unsafe fn maximum_f32_256(values: &[f32]) -> Result<f32, scioxide_core::SciOxideError> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let sign = _mm256_set1_ps(-0.0);
    let limit = _mm256_set1_ps(f32::MAX);
    let mut maximum = _mm256_setzero_ps();
    let mut valid = _mm256_cmp_ps::<_CMP_LE_OQ>(maximum, limit);
    let mut offset = 0;
    let mut maxima = [_mm256_setzero_ps(); 4];
    let mut valids = [valid; 4];
    while offset + 32 <= values.len() {
        for (lane, (maximum, valid)) in maxima.iter_mut().zip(&mut valids).enumerate() {
            // SAFETY: the block bound covers all four complete vectors.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx2,fma.
            let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset + lane * 8)) };
            let magnitude = _mm256_andnot_ps(sign, value);
            *maximum = _mm256_max_ps(*maximum, magnitude);
            *valid = _mm256_and_ps(*valid, _mm256_cmp_ps::<_CMP_LE_OQ>(magnitude, limit));
        }
        offset += 32;
    }
    maximum = _mm256_max_ps(
        _mm256_max_ps(maxima[0], maxima[1]),
        _mm256_max_ps(maxima[2], maxima[3]),
    );
    valid = _mm256_and_ps(
        _mm256_and_ps(valids[0], valids[1]),
        _mm256_and_ps(valids[2], valids[3]),
    );
    while offset + 8 <= values.len() {
        // SAFETY: the loop bounds one complete unaligned vector inside values.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx2,fma.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let magnitude = _mm256_andnot_ps(sign, value);
        maximum = _mm256_max_ps(maximum, magnitude);
        valid = _mm256_and_ps(valid, _mm256_cmp_ps::<_CMP_LE_OQ>(magnitude, limit));
        offset += 8;
    }
    let mut lanes = [0.0_f32; 8];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx2,fma.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), maximum) };
    let valid = _mm256_movemask_ps(valid) == 255;
    if !valid {
        return Err(scioxide_core::SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "dense input contains a non-finite component",
        ));
    }
    let mut maximum = lanes
        .into_iter()
        .fold(0.0_f32, scioxide_core::RealScalar::max);
    for &value in &values[offset..] {
        crate::checked::validate_input(value, "dense input component")?;
        maximum = scioxide_core::RealScalar::max(maximum, scioxide_core::NumericScalar::abs(value));
    }
    Ok(maximum)
}

/// Contiguous bounded affine update.
/// # Safety
/// The caller proves avx2,fma and equal initialized slice lengths; output is exclusive.
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_f32_256(alpha: f32, left: &[f32], beta: f32, right: &[f32], output: &mut [f32]) {
    let a = _mm256_set1_ps(alpha);
    let b = _mm256_set1_ps(beta);
    let mut offset = 0;
    while offset + 8 <= left.len() {
        // SAFETY: equal lengths and the loop bound cover both complete loads and the store.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx2,fma.
        unsafe {
            let x = _mm256_loadu_ps(left.as_ptr().add(offset));
            let y = _mm256_loadu_ps(right.as_ptr().add(offset));
            let result = _mm256_fmadd_ps(a, x, _mm256_mul_ps(b, y));
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
        }
        offset += 8;
    }
    for ((&x, &y), out) in left[offset..]
        .iter()
        .zip(&right[offset..])
        .zip(&mut output[offset..])
    {
        *out = scioxide_core::RealScalar::mul_add(alpha, x, beta * y);
    }
}

/// Validate and bound contiguous components without scalar checks per lane.
/// # Safety
/// The caller proves avx2,fma; loads stay within the initialized input slice.
#[target_feature(enable = "avx2,fma")]
unsafe fn maximum_f64_256(values: &[f64]) -> Result<f64, scioxide_core::SciOxideError> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let sign = _mm256_set1_pd(-0.0);
    let limit = _mm256_set1_pd(f64::MAX);
    let mut maximum = _mm256_setzero_pd();
    let mut valid = _mm256_cmp_pd::<_CMP_LE_OQ>(maximum, limit);
    let mut offset = 0;
    let mut maxima = [_mm256_setzero_pd(); 4];
    let mut valids = [valid; 4];
    while offset + 16 <= values.len() {
        for (lane, (maximum, valid)) in maxima.iter_mut().zip(&mut valids).enumerate() {
            // SAFETY: the block bound covers all four complete vectors.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx2,fma.
            let value = unsafe { _mm256_loadu_pd(values.as_ptr().add(offset + lane * 4)) };
            let magnitude = _mm256_andnot_pd(sign, value);
            *maximum = _mm256_max_pd(*maximum, magnitude);
            *valid = _mm256_and_pd(*valid, _mm256_cmp_pd::<_CMP_LE_OQ>(magnitude, limit));
        }
        offset += 16;
    }
    maximum = _mm256_max_pd(
        _mm256_max_pd(maxima[0], maxima[1]),
        _mm256_max_pd(maxima[2], maxima[3]),
    );
    valid = _mm256_and_pd(
        _mm256_and_pd(valids[0], valids[1]),
        _mm256_and_pd(valids[2], valids[3]),
    );
    while offset + 4 <= values.len() {
        // SAFETY: the loop bounds one complete unaligned vector inside values.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx2,fma.
        let value = unsafe { _mm256_loadu_pd(values.as_ptr().add(offset)) };
        let magnitude = _mm256_andnot_pd(sign, value);
        maximum = _mm256_max_pd(maximum, magnitude);
        valid = _mm256_and_pd(valid, _mm256_cmp_pd::<_CMP_LE_OQ>(magnitude, limit));
        offset += 4;
    }
    let mut lanes = [0.0_f64; 4];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx2,fma.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), maximum) };
    let valid = _mm256_movemask_pd(valid) == 15;
    if !valid {
        return Err(scioxide_core::SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "dense input contains a non-finite component",
        ));
    }
    let mut maximum = lanes
        .into_iter()
        .fold(0.0_f64, scioxide_core::RealScalar::max);
    for &value in &values[offset..] {
        crate::checked::validate_input(value, "dense input component")?;
        maximum = scioxide_core::RealScalar::max(maximum, scioxide_core::NumericScalar::abs(value));
    }
    Ok(maximum)
}

/// Contiguous bounded affine update.
/// # Safety
/// The caller proves avx2,fma and equal initialized slice lengths; output is exclusive.
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_f64_256(alpha: f64, left: &[f64], beta: f64, right: &[f64], output: &mut [f64]) {
    let a = _mm256_set1_pd(alpha);
    let b = _mm256_set1_pd(beta);
    let mut offset = 0;
    while offset + 4 <= left.len() {
        // SAFETY: equal lengths and the loop bound cover both complete loads and the store.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx2,fma.
        unsafe {
            let x = _mm256_loadu_pd(left.as_ptr().add(offset));
            let y = _mm256_loadu_pd(right.as_ptr().add(offset));
            let result = _mm256_fmadd_pd(a, x, _mm256_mul_pd(b, y));
            _mm256_storeu_pd(output.as_mut_ptr().add(offset), result);
        }
        offset += 4;
    }
    for ((&x, &y), out) in left[offset..]
        .iter()
        .zip(&right[offset..])
        .zip(&mut output[offset..])
    {
        *out = scioxide_core::RealScalar::mul_add(alpha, x, beta * y);
    }
}

/// Validate and bound contiguous components without scalar checks per lane.
/// # Safety
/// The caller proves avx512f,avx512dq,avx512vl; loads stay within the initialized input slice.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn maximum_f32_512(values: &[f32]) -> Result<f32, scioxide_core::SciOxideError> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let sign = _mm512_set1_ps(-0.0);
    let limit = _mm512_set1_ps(f32::MAX);
    let mut valid = u16::MAX;
    let mut offset = 0;
    let mut maxima = [_mm512_setzero_ps(); 4];
    let mut valids = [valid; 4];
    while offset + 64 <= values.len() {
        for (lane, (maximum, valid)) in maxima.iter_mut().zip(&mut valids).enumerate() {
            // SAFETY: the block bound covers all four complete vectors.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
            let value = unsafe { _mm512_loadu_ps(values.as_ptr().add(offset + lane * 16)) };
            let magnitude = _mm512_andnot_ps(sign, value);
            *maximum = _mm512_max_ps(*maximum, magnitude);
            *valid &= _mm512_cmp_ps_mask::<_CMP_LE_OQ>(magnitude, limit);
        }
        offset += 64;
    }
    let mut maximum = _mm512_max_ps(
        _mm512_max_ps(maxima[0], maxima[1]),
        _mm512_max_ps(maxima[2], maxima[3]),
    );
    valid = valids
        .into_iter()
        .fold(u16::MAX, |left, right| left & right);
    while offset + 16 <= values.len() {
        // SAFETY: the loop bounds one complete unaligned vector inside values.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
        let value = unsafe { _mm512_loadu_ps(values.as_ptr().add(offset)) };
        let magnitude = _mm512_andnot_ps(sign, value);
        maximum = _mm512_max_ps(maximum, magnitude);
        valid &= _mm512_cmp_ps_mask::<_CMP_LE_OQ>(magnitude, limit);
        offset += 16;
    }
    let mut lanes = [0.0_f32; 16];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), maximum) };
    let valid = valid == 65535;
    if !valid {
        return Err(scioxide_core::SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "dense input contains a non-finite component",
        ));
    }
    let mut maximum = lanes
        .into_iter()
        .fold(0.0_f32, scioxide_core::RealScalar::max);
    for &value in &values[offset..] {
        crate::checked::validate_input(value, "dense input component")?;
        maximum = scioxide_core::RealScalar::max(maximum, scioxide_core::NumericScalar::abs(value));
    }
    Ok(maximum)
}

/// Contiguous bounded affine update.
/// # Safety
/// The caller proves avx512f,avx512dq,avx512vl and equal initialized slice lengths; output is exclusive.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn affine_f32_512(alpha: f32, left: &[f32], beta: f32, right: &[f32], output: &mut [f32]) {
    let a = _mm512_set1_ps(alpha);
    let b = _mm512_set1_ps(beta);
    let mut offset = 0;
    while offset + 16 <= left.len() {
        // SAFETY: equal lengths and the loop bound cover both complete loads and the store.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
        unsafe {
            let x = _mm512_loadu_ps(left.as_ptr().add(offset));
            let y = _mm512_loadu_ps(right.as_ptr().add(offset));
            let result = _mm512_fmadd_ps(a, x, _mm512_mul_ps(b, y));
            _mm512_storeu_ps(output.as_mut_ptr().add(offset), result);
        }
        offset += 16;
    }
    for ((&x, &y), out) in left[offset..]
        .iter()
        .zip(&right[offset..])
        .zip(&mut output[offset..])
    {
        *out = scioxide_core::RealScalar::mul_add(alpha, x, beta * y);
    }
}

/// Validate and bound contiguous components without scalar checks per lane.
/// # Safety
/// The caller proves avx512f,avx512dq,avx512vl; loads stay within the initialized input slice.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn maximum_f64_512(values: &[f64]) -> Result<f64, scioxide_core::SciOxideError> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let sign = _mm512_set1_pd(-0.0);
    let limit = _mm512_set1_pd(f64::MAX);
    let mut valid = u16::MAX;
    let mut offset = 0;
    let mut maxima = [_mm512_setzero_pd(); 4];
    let mut valids = [valid; 4];
    while offset + 32 <= values.len() {
        for (lane, (maximum, valid)) in maxima.iter_mut().zip(&mut valids).enumerate() {
            // SAFETY: the block bound covers all four complete vectors.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
            let value = unsafe { _mm512_loadu_pd(values.as_ptr().add(offset + lane * 8)) };
            let magnitude = _mm512_andnot_pd(sign, value);
            *maximum = _mm512_max_pd(*maximum, magnitude);
            *valid &= u16::from(_mm512_cmp_pd_mask::<_CMP_LE_OQ>(magnitude, limit));
        }
        offset += 32;
    }
    let mut maximum = _mm512_max_pd(
        _mm512_max_pd(maxima[0], maxima[1]),
        _mm512_max_pd(maxima[2], maxima[3]),
    );
    valid = valids
        .into_iter()
        .fold(u16::MAX, |left, right| left & right);
    while offset + 8 <= values.len() {
        // SAFETY: the loop bounds one complete unaligned vector inside values.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
        let value = unsafe { _mm512_loadu_pd(values.as_ptr().add(offset)) };
        let magnitude = _mm512_andnot_pd(sign, value);
        maximum = _mm512_max_pd(maximum, magnitude);
        valid &= u16::from(_mm512_cmp_pd_mask::<_CMP_LE_OQ>(magnitude, limit));
        offset += 8;
    }
    let mut lanes = [0.0_f64; 8];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
    unsafe { _mm512_storeu_pd(lanes.as_mut_ptr(), maximum) };
    let valid = valid & 255 == 255;
    if !valid {
        return Err(scioxide_core::SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "dense input contains a non-finite component",
        ));
    }
    let mut maximum = lanes
        .into_iter()
        .fold(0.0_f64, scioxide_core::RealScalar::max);
    for &value in &values[offset..] {
        crate::checked::validate_input(value, "dense input component")?;
        maximum = scioxide_core::RealScalar::max(maximum, scioxide_core::NumericScalar::abs(value));
    }
    Ok(maximum)
}

/// Contiguous bounded affine update.
/// # Safety
/// The caller proves avx512f,avx512dq,avx512vl and equal initialized slice lengths; output is exclusive.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn affine_f64_512(alpha: f64, left: &[f64], beta: f64, right: &[f64], output: &mut [f64]) {
    let a = _mm512_set1_pd(alpha);
    let b = _mm512_set1_pd(beta);
    let mut offset = 0;
    while offset + 8 <= left.len() {
        // SAFETY: equal lengths and the loop bound cover both complete loads and the store.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies avx512f,avx512dq,avx512vl.
        unsafe {
            let x = _mm512_loadu_pd(left.as_ptr().add(offset));
            let y = _mm512_loadu_pd(right.as_ptr().add(offset));
            let result = _mm512_fmadd_pd(a, x, _mm512_mul_pd(b, y));
            _mm512_storeu_pd(output.as_mut_ptr().add(offset), result);
        }
        offset += 8;
    }
    for ((&x, &y), out) in left[offset..]
        .iter()
        .zip(&right[offset..])
        .zip(&mut output[offset..])
    {
        *out = scioxide_core::RealScalar::mul_add(alpha, x, beta * y);
    }
}

pub(super) fn gko_subtract_product<T: NumericScalar>(
    target: &mut [T],
    start: usize,
    source: &[T],
    source_start: usize,
    count: usize,
    factor: scioxide_core::Complex<T::Real>,
    dispatch: KernelDispatch,
) -> Option<bool> {
    if !T::IS_REAL {
        return None;
    }
    match dispatch.backend() {
        KernelBackend::Avx2Fma => {
            // SAFETY: the host-verified token proves avx2,fma; the inlined
            // safe kernel checks slice bounds and accesses only borrowed storage.
            Some(unsafe {
                gko_subtract_product_256(target, start, source, source_start, count, factor)
            })
        }
        KernelBackend::Avx512 => {
            // SAFETY: the host-verified token proves avx512f,avx512dq,avx512vl; the inlined
            // safe kernel checks slice bounds and accesses only borrowed storage.
            Some(unsafe {
                gko_subtract_product_512(target, start, source, source_start, count, factor)
            })
        }
        _ => None,
    }
}

/// Compile the shared safe loop for the verified ISA, preserving its arithmetic.
/// # Safety
/// The caller must prove avx2,fma executable on the current host.
#[target_feature(enable = "avx2,fma")]
unsafe fn gko_subtract_product_256<T: NumericScalar>(
    target: &mut [T],
    start: usize,
    source: &[T],
    source_start: usize,
    count: usize,
    factor: scioxide_core::Complex<T::Real>,
) -> bool {
    super::super::gko::subtract_product_scalar(target, start, source, source_start, count, factor)
}

/// Compile the shared safe loop for the verified ISA, preserving its arithmetic.
/// # Safety
/// The caller must prove avx512f,avx512dq,avx512vl executable on the current host.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn gko_subtract_product_512<T: NumericScalar>(
    target: &mut [T],
    start: usize,
    source: &[T],
    source_start: usize,
    count: usize,
    factor: scioxide_core::Complex<T::Real>,
) -> bool {
    super::super::gko::subtract_product_scalar(target, start, source, source_start, count, factor)
}

pub(super) fn gko_reconstruct_line<T: NumericScalar>(
    sources: (&[T], &[T], &[T]),
    weights: (
        scioxide_core::Complex<T::Real>,
        scioxide_core::Complex<T::Real>,
    ),
    node: scioxide_core::Complex<T::Real>,
    range: core::ops::Range<usize>,
    output: &mut [T],
    offset: usize,
    dispatch: KernelDispatch,
) -> Option<bool> {
    if !T::IS_REAL {
        return None;
    }
    match dispatch.backend() {
        KernelBackend::Avx2Fma => {
            // SAFETY: the host-verified token proves avx2,fma; the inlined
            // safe kernel checks slice bounds and accesses only borrowed storage.
            Some(unsafe { gko_reconstruct_line_256(sources, weights, node, range, output, offset) })
        }
        KernelBackend::Avx512 => {
            // SAFETY: the host-verified token proves avx512f,avx512dq,avx512vl; the inlined
            // safe kernel checks slice bounds and accesses only borrowed storage.
            Some(unsafe { gko_reconstruct_line_512(sources, weights, node, range, output, offset) })
        }
        _ => None,
    }
}

/// Compile the shared safe loop for the verified ISA, preserving its arithmetic.
/// # Safety
/// The caller must prove avx2,fma executable on the current host.
#[target_feature(enable = "avx2,fma")]
unsafe fn gko_reconstruct_line_256<T: NumericScalar>(
    sources: (&[T], &[T], &[T]),
    weights: (
        scioxide_core::Complex<T::Real>,
        scioxide_core::Complex<T::Real>,
    ),
    node: scioxide_core::Complex<T::Real>,
    range: core::ops::Range<usize>,
    output: &mut [T],
    offset: usize,
) -> bool {
    super::super::gko::reconstruct_line_scalar(sources, weights, node, range, output, offset)
}

/// Compile the shared safe loop for the verified ISA, preserving its arithmetic.
/// # Safety
/// The caller must prove avx512f,avx512dq,avx512vl executable on the current host.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn gko_reconstruct_line_512<T: NumericScalar>(
    sources: (&[T], &[T], &[T]),
    weights: (
        scioxide_core::Complex<T::Real>,
        scioxide_core::Complex<T::Real>,
    ),
    node: scioxide_core::Complex<T::Real>,
    range: core::ops::Range<usize>,
    output: &mut [T],
    offset: usize,
) -> bool {
    super::super::gko::reconstruct_line_scalar(sources, weights, node, range, output, offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn safe_packed_leaves_reject_invalid_spans_and_dispatch() {
        let scalar = KernelDispatch::detect(&scioxide_core::PortableProcessor);
        {
            let a = [1.0_f32; 4];
            let b = [1.0_f32; 16];
            let mut c = [7.0_f32; 64];
            assert!(!avx2_f32(&a, &b, &mut c, 16, 1, 0.0, scalar));
            if let Ok(dispatch) = KernelDispatch::for_backend(
                KernelBackend::Avx2Fma,
                crate::kernel::tests::test_processor(),
            ) {
                assert!(!avx2_f32(&a[..3], &b, &mut c, 16, 1, 0.0, dispatch));
                assert!(!avx2_f32(&a, &b[..15], &mut c, 16, 1, 0.0, dispatch));
                assert!(!avx2_f32(&a, &b, &mut c[..63], 16, 1, 0.0, dispatch));
                assert!(!avx2_f32(&a, &b, &mut c, 15, 1, 0.0, dispatch));
                assert!(!avx2_f32(&a, &b, &mut c, 16, usize::MAX, 0.0, dispatch));
                assert!(!avx2_f32(&a, &b, &mut c, usize::MAX, 1, 0.0, dispatch));
            }
            assert!(c.iter().all(|&value| value == 7.0));
        }
        {
            let a = [1.0_f64; 6];
            let b = [1.0_f64; 8];
            let mut c = [7.0_f64; 48];
            assert!(!avx2_f64(&a, &b, &mut c, 8, 1, 0.0, scalar));
            if let Ok(dispatch) = KernelDispatch::for_backend(
                KernelBackend::Avx2Fma,
                crate::kernel::tests::test_processor(),
            ) {
                assert!(!avx2_f64(&a[..5], &b, &mut c, 8, 1, 0.0, dispatch));
                assert!(!avx2_f64(&a, &b[..7], &mut c, 8, 1, 0.0, dispatch));
                assert!(!avx2_f64(&a, &b, &mut c[..47], 8, 1, 0.0, dispatch));
                assert!(!avx2_f64(&a, &b, &mut c, 7, 1, 0.0, dispatch));
                assert!(!avx2_f64(&a, &b, &mut c, 8, usize::MAX, 0.0, dispatch));
                assert!(!avx2_f64(&a, &b, &mut c, usize::MAX, 1, 0.0, dispatch));
            }
            assert!(c.iter().all(|&value| value == 7.0));
        }
        {
            let a = [1.0_f32; 4];
            let b = [1.0_f32; 32];
            let mut c = [7.0_f32; 128];
            assert!(!avx512_f32(&a, &b, &mut c, 32, 1, 0.0, scalar));
            if let Ok(dispatch) = KernelDispatch::for_backend(
                KernelBackend::Avx512,
                crate::kernel::tests::test_processor(),
            ) {
                assert!(!avx512_f32(&a[..3], &b, &mut c, 32, 1, 0.0, dispatch));
                assert!(!avx512_f32(&a, &b[..31], &mut c, 32, 1, 0.0, dispatch));
                assert!(!avx512_f32(&a, &b, &mut c[..127], 32, 1, 0.0, dispatch));
                assert!(!avx512_f32(&a, &b, &mut c, 31, 1, 0.0, dispatch));
                assert!(!avx512_f32(&a, &b, &mut c, 32, usize::MAX, 0.0, dispatch));
                assert!(!avx512_f32(&a, &b, &mut c, usize::MAX, 1, 0.0, dispatch));
            }
            assert!(c.iter().all(|&value| value == 7.0));
        }
        {
            let a = [1.0_f64; 4];
            let b = [1.0_f64; 16];
            let mut c = [7.0_f64; 64];
            assert!(!avx512_f64(&a, &b, &mut c, 16, 1, 0.0, scalar));
            if let Ok(dispatch) = KernelDispatch::for_backend(
                KernelBackend::Avx512,
                crate::kernel::tests::test_processor(),
            ) {
                assert!(!avx512_f64(&a[..3], &b, &mut c, 16, 1, 0.0, dispatch));
                assert!(!avx512_f64(&a, &b[..15], &mut c, 16, 1, 0.0, dispatch));
                assert!(!avx512_f64(&a, &b, &mut c[..63], 16, 1, 0.0, dispatch));
                assert!(!avx512_f64(&a, &b, &mut c, 15, 1, 0.0, dispatch));
                assert!(!avx512_f64(&a, &b, &mut c, 16, usize::MAX, 0.0, dispatch));
                assert!(!avx512_f64(&a, &b, &mut c, usize::MAX, 1, 0.0, dispatch));
            }
            assert!(c.iter().all(|&value| value == 7.0));
        }
    }
}
