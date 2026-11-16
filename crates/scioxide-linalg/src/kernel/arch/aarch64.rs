//! AArch64 NEON leaves: packed f32/f64 GEMM tiles, dots, finite maxima and
//! bounded affine updates.

use core::arch::aarch64::{
    float32x4_t, float64x2_t, vdupq_n_f32, vdupq_n_f64, vfmaq_f32, vfmaq_f64, vld1q_f32, vld1q_f64,
    vmulq_f32, vmulq_f64, vst1q_f32, vst1q_f64,
};

use scioxide_core::NumericScalar;
use scioxide_core::backend::{KernelBackend, KernelDispatch, ScalarKind};

use super::super::microkernel::Microkernel;

pub(super) fn select_microkernel<S: NumericScalar>(
    dispatch: KernelDispatch,
) -> Option<Microkernel> {
    if dispatch.backend() != KernelBackend::Neon {
        return None;
    }
    match S::kind() {
        ScalarKind::F32 => Some(Microkernel::new_f32(neon_f32, 4, 8, dispatch)),
        ScalarKind::F64 => Some(Microkernel::new_f64(neon_f64, 4, 4, dispatch)),
        _ => None,
    }
}

fn neon_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row_stride: usize,
    depth: usize,
    beta: f32,
    dispatch: KernelDispatch,
) -> bool {
    if dispatch.backend() != KernelBackend::Neon
        || !super::super::microkernel::panel_fits(
            4,
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
    // SAFETY: the verified token proves Neon; panel_fits checks every packed
    // input and output span with checked arithmetic. Unaligned loads accept scalar
    // alignment, slices are initialized, and exclusive C cannot alias either input.
    unsafe { neon_f32_leaf(a, b, c, row_stride, depth, beta) }
    true
}

fn neon_f64(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    row_stride: usize,
    depth: usize,
    beta: f64,
    dispatch: KernelDispatch,
) -> bool {
    if dispatch.backend() != KernelBackend::Neon
        || !super::super::microkernel::panel_fits(
            4,
            4,
            a.len(),
            b.len(),
            c.len(),
            row_stride,
            depth,
        )
    {
        return false;
    }
    // SAFETY: the verified token proves Neon; panel_fits checks every packed
    // input and output span with checked arithmetic. Unaligned loads accept scalar
    // alignment, slices are initialized, and exclusive C cannot alias either input.
    unsafe { neon_f64_leaf(a, b, c, row_stride, depth, beta) }
    true
}

/// Apply a packed 4x8 f32 NEON microkernel.
///
/// # Safety
///
/// - Alignment: NEON loads/stores accept valid f32 slice alignment.
/// - Length: A has `depth*4`, B has `depth*8`, and C exposes four rows of 8.
/// - Aliasing: packed inputs do not overlap the exclusive output.
/// - Initialization: packed inputs and any C lanes read for nonzero beta are initialized.
/// - Target features: NEON is available.
#[target_feature(enable = "neon")]
unsafe fn neon_f32_leaf(a: &[f32], b: &[f32], c: &mut [f32], rs: usize, depth: usize, beta: f32) {
    debug_assert!(a.len() >= depth * 4 && b.len() >= depth * 8);
    debug_assert!(c.len() >= 3 * rs + 8);
    let zero: float32x4_t = vdupq_n_f32(0.0);
    let mut acc = [[zero; 2]; 4];
    let mut row = 0;
    while row < 4 {
        if beta != 0.0 {
            let scale = vdupq_n_f32(beta);
            let mut group = 0;
            while group < 2 {
                // SAFETY: the C span contract covers every complete vector. Unaligned
                // vector accesses need only scalar alignment. Packed inputs and C are
                // initialized; shared A/B do not alias exclusive C. Host-verified dispatch
                // proves the leaf's target features.
                let value = unsafe { vld1q_f32(c.as_ptr().add(row * rs + group * 4)) };
                acc[row][group] = if beta == 1.0 {
                    value
                } else {
                    vmulq_f32(value, scale)
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
        let b0 = unsafe { vld1q_f32(b.as_ptr().add(inner * 8)) };
        // SAFETY: the packed B contract covers the second vector at this depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b1 = unsafe { vld1q_f32(b.as_ptr().add(inner * 8 + 4)) };
        row = 0;
        while row < 4 {
            let av = vdupq_n_f32(a[inner * 4 + row]);
            acc[row][0] = vfmaq_f32(acc[row][0], av, b0);
            acc[row][1] = vfmaq_f32(acc[row][1], av, b1);
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
            vst1q_f32(c.as_mut_ptr().add(row * rs), acc[row][0]);
            vst1q_f32(c.as_mut_ptr().add(row * rs + 4), acc[row][1]);
        }
        row += 1;
    }
}

/// Apply a packed 4x4 f64 NEON microkernel.
///
/// # Safety
///
/// - Alignment: NEON loads/stores accept valid f64 slice alignment.
/// - Length: A has `depth*4`, B has `depth*4`, and C exposes four rows of 4.
/// - Aliasing: packed inputs do not overlap the exclusive output.
/// - Initialization: packed inputs and any C lanes read for nonzero beta are initialized.
/// - Target features: NEON is available.
#[target_feature(enable = "neon")]
unsafe fn neon_f64_leaf(a: &[f64], b: &[f64], c: &mut [f64], rs: usize, depth: usize, beta: f64) {
    debug_assert!(a.len() >= depth * 4 && b.len() >= depth * 4);
    debug_assert!(c.len() >= 3 * rs + 4);
    let zero: float64x2_t = vdupq_n_f64(0.0);
    let mut acc = [[zero; 2]; 4];
    let mut row = 0;
    while row < 4 {
        if beta != 0.0 {
            let scale = vdupq_n_f64(beta);
            let mut group = 0;
            while group < 2 {
                // SAFETY: the C span contract covers every complete vector. Unaligned
                // vector accesses need only scalar alignment. Packed inputs and C are
                // initialized; shared A/B do not alias exclusive C. Host-verified dispatch
                // proves the leaf's target features.
                let value = unsafe { vld1q_f64(c.as_ptr().add(row * rs + group * 2)) };
                acc[row][group] = if beta == 1.0 {
                    value
                } else {
                    vmulq_f64(value, scale)
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
        let b0 = unsafe { vld1q_f64(b.as_ptr().add(inner * 4)) };
        // SAFETY: the packed B contract covers the second vector at this depth. Unaligned
        // vector accesses need only scalar alignment. Packed inputs and C are initialized;
        // shared A/B do not alias exclusive C. Host-verified dispatch proves the leaf's
        // target features.
        let b1 = unsafe { vld1q_f64(b.as_ptr().add(inner * 4 + 2)) };
        row = 0;
        while row < 4 {
            let av = vdupq_n_f64(a[inner * 4 + row]);
            acc[row][0] = vfmaq_f64(acc[row][0], av, b0);
            acc[row][1] = vfmaq_f64(acc[row][1], av, b1);
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
            vst1q_f64(c.as_mut_ptr().add(row * rs), acc[row][0]);
            vst1q_f64(c.as_mut_ptr().add(row * rs + 2), acc[row][1]);
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
    if dispatch.backend() != KernelBackend::Neon {
        return Ok(None);
    }
    match (S::slice(left), S::slice(right)) {
        (ScalarSlice::F32(left), ScalarSlice::F32(right)) => {
            // SAFETY: the verified dispatch proves NEON; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            Ok(Some(S::from_value(ScalarValue::F32(unsafe {
                dot_f32(left, right)
            }))?))
        }
        (ScalarSlice::F64(left), ScalarSlice::F64(right)) => {
            // SAFETY: the verified dispatch proves NEON; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            Ok(Some(S::from_value(ScalarValue::F64(unsafe {
                dot_f64(left, right)
            }))?))
        }
        _ => Ok(None),
    }
}

/// Contiguous dot with a fixed NEON reduction order.
/// # Safety
/// The caller proves NEON and equal initialized slice lengths.
#[target_feature(enable = "neon")]
unsafe fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let mut sums = [vdupq_n_f32(0.0); 4];
    let mut offset = 0;
    while offset + 16 <= left.len() {
        for (lane, sum) in sums.iter_mut().enumerate() {
            // SAFETY: the block bound covers all four vectors in both slices.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies neon.
            let (a, b) = unsafe {
                (
                    vld1q_f32(left.as_ptr().add(offset + lane * 4)),
                    vld1q_f32(right.as_ptr().add(offset + lane * 4)),
                )
            };
            *sum = vfmaq_f32(*sum, a, b);
        }
        offset += 16;
    }
    let mut sum = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
    while offset + 4 <= left.len() {
        // SAFETY: both slices contain the complete initialized vector.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies neon.
        let (a, b) = unsafe {
            (
                vld1q_f32(left.as_ptr().add(offset)),
                vld1q_f32(right.as_ptr().add(offset)),
            )
        };
        sum = vfmaq_f32(sum, a, b);
        offset += 4;
    }
    let mut lanes = [0.0; 4];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies neon.
    unsafe { vst1q_f32(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.into_iter().fold(0.0, |sum, value| sum + value);
    for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
        result = scioxide_core::RealScalar::mul_add(a, b, result);
    }
    result
}

/// Contiguous dot with a fixed NEON reduction order.
/// # Safety
/// The caller proves NEON and equal initialized slice lengths.
#[target_feature(enable = "neon")]
unsafe fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    use core::arch::aarch64::*;
    let mut sums = [vdupq_n_f64(0.0); 4];
    let mut offset = 0;
    while offset + 8 <= left.len() {
        for (lane, sum) in sums.iter_mut().enumerate() {
            // SAFETY: the block bound covers all four vectors in both slices.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            // The enclosing leaf's target-feature contract supplies neon.
            let (a, b) = unsafe {
                (
                    vld1q_f64(left.as_ptr().add(offset + lane * 2)),
                    vld1q_f64(right.as_ptr().add(offset + lane * 2)),
                )
            };
            *sum = vfmaq_f64(*sum, a, b);
        }
        offset += 8;
    }
    let mut sum = vaddq_f64(vaddq_f64(sums[0], sums[1]), vaddq_f64(sums[2], sums[3]));
    while offset + 2 <= left.len() {
        // SAFETY: both slices contain the complete initialized vector.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies neon.
        let (a, b) = unsafe {
            (
                vld1q_f64(left.as_ptr().add(offset)),
                vld1q_f64(right.as_ptr().add(offset)),
            )
        };
        sum = vfmaq_f64(sum, a, b);
        offset += 2;
    }
    let mut lanes = [0.0; 2];
    // SAFETY: the initialized destination holds one complete vector.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies neon.
    unsafe { vst1q_f64(lanes.as_mut_ptr(), sum) };
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
        (ScalarSlice::F32(values), KernelBackend::Neon) => {
            // SAFETY: verified dispatch proves neon; leaves bound every load.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            let maximum = unsafe { maximum_f32_128(values) }?;
            Ok(Some(
                S::from_value(ScalarValue::F32(maximum))?.real_component(),
            ))
        }
        (ScalarSlice::F64(values), KernelBackend::Neon) => {
            // SAFETY: verified dispatch proves neon; leaves bound every load.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            let maximum = unsafe { maximum_f64_128(values) }?;
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
            KernelBackend::Neon,
        ) => {
            // SAFETY: verified dispatch proves neon; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            unsafe { affine_f32_128(alpha, left, beta, right, output) };
            true
        }
        (
            ScalarValue::F64(alpha),
            ScalarSlice::F64(left),
            ScalarValue::F64(beta),
            ScalarSlice::F64(right),
            ScalarSliceMut::F64(output),
            KernelBackend::Neon,
        ) => {
            // SAFETY: verified dispatch proves neon; the length check above proves equal slice lengths.
            // References and local arrays provide initialized, element-aligned storage;
            // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
            // stores use exclusive mutable borrows.
            unsafe { affine_f64_128(alpha, left, beta, right, output) };
            true
        }
        _ => false,
    }
}

/// Validate and bound contiguous components without scalar checks per lane.
/// # Safety
/// The caller proves neon; loads stay within the initialized input slice.
#[target_feature(enable = "neon")]
unsafe fn maximum_f32_128(values: &[f32]) -> Result<f32, scioxide_core::SciOxideError> {
    use core::arch::aarch64::*;
    let limit = vdupq_n_f32(f32::MAX);
    let mut maximum = vdupq_n_f32(0.0);
    let mut valid = vdupq_n_u32(u32::MAX);
    let mut offset = 0;
    while offset + 4 <= values.len() {
        // SAFETY: the loop bounds one complete vector inside the input slice.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies neon.
        let value = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
        let magnitude = vabsq_f32(value);
        maximum = vmaxq_f32(maximum, magnitude);
        valid = vandq_u32(valid, vcleq_f32(magnitude, limit));
        offset += 4;
    }
    let mut lanes = [0.0_f32; 4];
    let mut masks = [0_u32; 4];
    // SAFETY: both initialized arrays hold complete destination vectors.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies neon.
    unsafe {
        vst1q_f32(lanes.as_mut_ptr(), maximum);
        vst1q_u32(masks.as_mut_ptr(), valid);
    };
    let valid = masks.into_iter().all(|mask| mask == u32::MAX);
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
/// The caller proves neon and equal initialized slice lengths; output is exclusive.
#[target_feature(enable = "neon")]
unsafe fn affine_f32_128(alpha: f32, left: &[f32], beta: f32, right: &[f32], output: &mut [f32]) {
    let a = vdupq_n_f32(alpha);
    let b = vdupq_n_f32(beta);
    let mut offset = 0;
    while offset + 4 <= left.len() {
        // SAFETY: equal lengths and the loop bound cover both complete loads and the store.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies neon.
        unsafe {
            let x = vld1q_f32(left.as_ptr().add(offset));
            let y = vld1q_f32(right.as_ptr().add(offset));
            let result = vfmaq_f32(vmulq_f32(b, y), a, x);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
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
/// The caller proves neon; loads stay within the initialized input slice.
#[target_feature(enable = "neon")]
unsafe fn maximum_f64_128(values: &[f64]) -> Result<f64, scioxide_core::SciOxideError> {
    use core::arch::aarch64::*;
    let limit = vdupq_n_f64(f64::MAX);
    let mut maximum = vdupq_n_f64(0.0);
    let mut valid = vdupq_n_u64(u64::MAX);
    let mut offset = 0;
    while offset + 2 <= values.len() {
        // SAFETY: the loop bounds one complete vector inside the input slice.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies neon.
        let value = unsafe { vld1q_f64(values.as_ptr().add(offset)) };
        let magnitude = vabsq_f64(value);
        maximum = vmaxq_f64(maximum, magnitude);
        valid = vandq_u64(valid, vcleq_f64(magnitude, limit));
        offset += 2;
    }
    let mut lanes = [0.0_f64; 2];
    let mut masks = [0_u64; 2];
    // SAFETY: both initialized arrays hold complete destination vectors.
    // References and local arrays provide initialized, element-aligned storage;
    // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
    // stores use exclusive mutable borrows.
    // The enclosing leaf's target-feature contract supplies neon.
    unsafe {
        vst1q_f64(lanes.as_mut_ptr(), maximum);
        vst1q_u64(masks.as_mut_ptr(), valid);
    };
    let valid = masks.into_iter().all(|mask| mask == u64::MAX);
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
/// The caller proves neon and equal initialized slice lengths; output is exclusive.
#[target_feature(enable = "neon")]
unsafe fn affine_f64_128(alpha: f64, left: &[f64], beta: f64, right: &[f64], output: &mut [f64]) {
    let a = vdupq_n_f64(alpha);
    let b = vdupq_n_f64(beta);
    let mut offset = 0;
    while offset + 2 <= left.len() {
        // SAFETY: equal lengths and the loop bound cover both complete loads and the store.
        // References and local arrays provide initialized, element-aligned storage;
        // these unaligned intrinsics need no SIMD alignment. Shared reads may alias;
        // stores use exclusive mutable borrows.
        // The enclosing leaf's target-feature contract supplies neon.
        unsafe {
            let x = vld1q_f64(left.as_ptr().add(offset));
            let y = vld1q_f64(right.as_ptr().add(offset));
            let result = vfmaq_f64(vmulq_f64(b, y), a, x);
            vst1q_f64(output.as_mut_ptr().add(offset), result);
        }
        offset += 2;
    }
    for ((&x, &y), out) in left[offset..]
        .iter()
        .zip(&right[offset..])
        .zip(&mut output[offset..])
    {
        *out = scioxide_core::RealScalar::mul_add(alpha, x, beta * y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn safe_packed_leaves_reject_invalid_spans_and_dispatch() {
        let scalar = KernelDispatch::detect(&scioxide_core::PortableProcessor);
        {
            let a = [1.0_f32; 4];
            let b = [1.0_f32; 8];
            let mut c = [7.0_f32; 32];
            assert!(!neon_f32(&a, &b, &mut c, 8, 1, 0.0, scalar));
            if let Ok(dispatch) = KernelDispatch::for_backend(
                KernelBackend::Neon,
                crate::kernel::tests::test_processor(),
            ) {
                assert!(!neon_f32(&a[..3], &b, &mut c, 8, 1, 0.0, dispatch));
                assert!(!neon_f32(&a, &b[..7], &mut c, 8, 1, 0.0, dispatch));
                assert!(!neon_f32(&a, &b, &mut c[..31], 8, 1, 0.0, dispatch));
                assert!(!neon_f32(&a, &b, &mut c, 7, 1, 0.0, dispatch));
                assert!(!neon_f32(&a, &b, &mut c, 8, usize::MAX, 0.0, dispatch));
                assert!(!neon_f32(&a, &b, &mut c, usize::MAX, 1, 0.0, dispatch));
            }
            assert!(c.iter().all(|&value| value == 7.0));
        }
        {
            let a = [1.0_f64; 4];
            let b = [1.0_f64; 4];
            let mut c = [7.0_f64; 16];
            assert!(!neon_f64(&a, &b, &mut c, 4, 1, 0.0, scalar));
            if let Ok(dispatch) = KernelDispatch::for_backend(
                KernelBackend::Neon,
                crate::kernel::tests::test_processor(),
            ) {
                assert!(!neon_f64(&a[..3], &b, &mut c, 4, 1, 0.0, dispatch));
                assert!(!neon_f64(&a, &b[..3], &mut c, 4, 1, 0.0, dispatch));
                assert!(!neon_f64(&a, &b, &mut c[..15], 4, 1, 0.0, dispatch));
                assert!(!neon_f64(&a, &b, &mut c, 3, 1, 0.0, dispatch));
                assert!(!neon_f64(&a, &b, &mut c, 4, usize::MAX, 0.0, dispatch));
                assert!(!neon_f64(&a, &b, &mut c, usize::MAX, 1, 0.0, dispatch));
            }
            assert!(c.iter().all(|&value| value == 7.0));
        }
    }
}
