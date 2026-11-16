//! AArch64 NEON complex FFT leaves, shared by both scalar widths.
//!
//! Register codelets are shared with AVX2 through [`CodeletLane`].

use super::codelets::{CodeletLane, codelet};

use super::super::{
    FinalStageKernel, GatherKernel, PermutationKernel, StageKernel, StageKernels, scalar,
};
use core::arch::aarch64::*;
use scioxide_core::Complex;

/// A register of complex pairs. Every operation requires NEON; memory
/// operations additionally require LANES complete values beginning at index.
trait NeonLane: CodeletLane {
    const LANES: usize;
    unsafe fn broadcast(value: Complex<Self>) -> Self::Vector;
    unsafe fn load(values: &[Complex<Self>], index: usize) -> Self::Vector;
    unsafe fn store(values: &mut [Complex<Self>], index: usize, value: Self::Vector);
    unsafe fn real(value: Self::Vector) -> Self::Vector;
    unsafe fn imaginary(value: Self::Vector) -> Self::Vector;
    unsafe fn swap(value: Self::Vector) -> Self::Vector;
}

#[inline(always)]
unsafe fn rotate<L: NeonLane, const INVERSE: bool>(value: L::Vector) -> L::Vector {
    let signs = if INVERSE {
        Complex::new(-L::ONE, L::ONE)
    } else {
        Complex::new(L::ONE, -L::ONE)
    };
    // SAFETY: the caller proves NEON; all operands are initialized registers.
    unsafe { L::vmul(L::swap(value), L::broadcast(signs)) }
}

impl CodeletLane for f32 {
    type Vector = float32x4_t;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn splat(value: f32) -> float32x4_t {
        vdupq_n_f32(value)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vadd(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vaddq_f32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vsub(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vsubq_f32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vmul(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vmulq_f32(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn rotate<const INVERSE: bool>(value: float32x4_t) -> float32x4_t {
        // SAFETY: the caller proves NEON and initialized complex registers.
        unsafe { rotate::<Self, INVERSE>(value) }
    }
}

impl NeonLane for f32 {
    const LANES: usize = 2;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn broadcast(value: Complex<f32>) -> float32x4_t {
        // SAFETY: the caller proves NEON and the local array contains complete complex lanes.
        unsafe { Self::load(&[value; 2], 0) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(values: &[Complex<f32>], index: usize) -> float32x4_t {
        // SAFETY: the caller proves NEON and 2 in-bounds complex values.
        unsafe { vld1q_f32(values.as_ptr().add(index).cast::<f32>()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn real(value: float32x4_t) -> float32x4_t {
        vtrn1q_f32(value, value)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn imaginary(value: float32x4_t) -> float32x4_t {
        vtrn2q_f32(value, value)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn swap(value: float32x4_t) -> float32x4_t {
        vrev64q_f32(value)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(values: &mut [Complex<f32>], index: usize, value: float32x4_t) {
        // SAFETY: the caller proves NEON and 2 writable complex values.
        unsafe { vst1q_f32(values.as_mut_ptr().add(index).cast::<f32>(), value) }
    }
}

impl CodeletLane for f64 {
    type Vector = float64x2_t;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn splat(value: f64) -> float64x2_t {
        vdupq_n_f64(value)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vadd(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vaddq_f64(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vsub(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vsubq_f64(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vmul(a: float64x2_t, b: float64x2_t) -> float64x2_t {
        vmulq_f64(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn rotate<const INVERSE: bool>(value: float64x2_t) -> float64x2_t {
        // SAFETY: the caller proves NEON and initialized complex registers.
        unsafe { rotate::<Self, INVERSE>(value) }
    }
}

impl NeonLane for f64 {
    const LANES: usize = 1;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn broadcast(value: Complex<f64>) -> float64x2_t {
        // SAFETY: the caller proves NEON and the local array contains complete complex lanes.
        unsafe { Self::load(&[value; 1], 0) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load(values: &[Complex<f64>], index: usize) -> float64x2_t {
        // SAFETY: the caller proves NEON and 1 in-bounds complex values.
        unsafe { vld1q_f64(values.as_ptr().add(index).cast::<f64>()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn real(value: float64x2_t) -> float64x2_t {
        vdupq_n_f64(vgetq_lane_f64::<0>(value))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn imaginary(value: float64x2_t) -> float64x2_t {
        vdupq_n_f64(vgetq_lane_f64::<1>(value))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn swap(value: float64x2_t) -> float64x2_t {
        vextq_f64::<1>(value, value)
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store(values: &mut [Complex<f64>], index: usize, value: float64x2_t) {
        // SAFETY: the caller proves NEON and 1 writable complex values.
        unsafe { vst1q_f64(values.as_mut_ptr().add(index).cast::<f64>(), value) }
    }
}

// Gather and final-stage leaves reuse portable code; only the in-place
// stages use NEON registers. Plans retain their unfused stage sequence.
pub(crate) fn kernels_f32() -> StageKernels<f32> {
    kernels()
}
pub(crate) fn kernels_f64() -> StageKernels<f64> {
    kernels()
}
fn kernels<L: NeonLane>() -> StageKernels<L> {
    StageKernels {
        forward: StageKernel::new(neon_stage::<L, false>),
        inverse: StageKernel::new(neon_stage::<L, true>),
        fused_gather_forward: None,
        fused_gather_inverse: None,
        fused_stage_forward: None,
        fused_stage_inverse: None,
        final_forward: FinalStageKernel::new(scalar::final_forward),
        final_inverse: FinalStageKernel::new(scalar::final_inverse),
        permute: PermutationKernel::new(scalar::permute::<L, false>, scalar::permute::<L, true>),
        gather_forward: GatherKernel::new(
            scalar::gather::<L, false, false>,
            scalar::gather::<L, false, true>,
        ),
        gather_inverse: GatherKernel::new(
            scalar::gather::<L, true, false>,
            scalar::gather::<L, true, true>,
        ),
    }
}

/// # Safety
/// NEON must be available. The supported radix, positive width, full
/// radix*width data blocks, (radix-1)*width twiddles and odd-radix roots must
/// match the immutable stage selected at preparation.
#[target_feature(enable = "neon")]
unsafe fn neon_stage<L: NeonLane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    roots: &[Complex<L>],
) {
    // SAFETY: the caller proves the common stage contract; each selected
    // constant radix retains the same geometry, tables and NEON requirement.
    unsafe {
        match radix {
            2 => neon_radix::<L, 2, INVERSE>(data, width, twiddles, roots),
            3 => neon_radix::<L, 3, INVERSE>(data, width, twiddles, roots),
            4 => neon_radix::<L, 4, INVERSE>(data, width, twiddles, roots),
            5 => neon_radix::<L, 5, INVERSE>(data, width, twiddles, roots),
            7 => neon_radix::<L, 7, INVERSE>(data, width, twiddles, roots),
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_radix<L: NeonLane, const RADIX: usize, const INVERSE: bool>(
    data: &mut [Complex<L>],
    width: usize,
    twiddles: &[Complex<L>],
    roots: &[Complex<L>],
) {
    // SAFETY: the caller proves whole RADIX*width blocks and full twiddle
    // branches. The loop covers complete registers and checked scalar tails.
    unsafe {
        radix_loop!(
            L,
            INVERSE,
            data,
            RADIX,
            width,
            twiddles,
            roots,
            L::LANES,
            L::splat(L::ZERO),
            L::load,
            L::store,
            complex_mul::<L, INVERSE>,
            |values| codelet::<L, RADIX, INVERSE>(&mut values, roots)
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn complex_mul<L: NeonLane, const CONJUGATE_RIGHT: bool>(
    left: L::Vector,
    right: L::Vector,
) -> L::Vector {
    // SAFETY: the caller proves NEON; all operations use initialized registers.
    unsafe {
        let imaginary = L::imaginary(right);
        let signs = if CONJUGATE_RIGHT {
            Complex::new(L::ONE, -L::ONE)
        } else {
            Complex::new(-L::ONE, L::ONE)
        };
        L::vadd(
            L::vmul(L::real(right), left),
            L::vmul(L::vmul(imaginary, L::broadcast(signs)), L::swap(left)),
        )
    }
}
