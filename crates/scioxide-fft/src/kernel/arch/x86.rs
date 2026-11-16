//! x86/x86-64 typed complex FFT leaves.
//!
//! One AVX2/FMA leaf set serves both precisions through [`Avx2Lane`]: a
//! register packs four f32 or two f64 complex values, and every specialised
//! leaf is written once in terms of that lane count. Register codelets are
//! shared with NEON through [`CodeletLane`]. There are no AVX-512
//! leaves: AVX2's fused and specialised leaves outperform a wider generic
//! stage, so FFT preparation rejects a required AVX-512 backend.

#[cfg(target_arch = "x86")]
use core::arch::x86 as simd;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as simd;

use simd::{
    __m256, __m256d, _mm256_add_pd, _mm256_add_ps, _mm256_castpd_ps, _mm256_castps_pd,
    _mm256_fmaddsub_pd, _mm256_fmaddsub_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_movedup_pd,
    _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_permute_pd,
    _mm256_permute_ps, _mm256_permute2f128_pd, _mm256_set1_pd, _mm256_set1_ps, _mm256_setr_pd,
    _mm256_setr_ps, _mm256_setzero_pd, _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps,
    _mm256_sub_pd, _mm256_sub_ps, _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm256_xor_pd,
    _mm256_xor_ps,
};

use scioxide_core::{AlgebraicScalar, Complex, RealScalar};

use super::codelets::{CodeletLane, codelet};

use super::super::{
    FinalStageKernel, FusedGatherKernel, FusedStageKernel, GatherKernel, PermutationKernel,
    StageKernel, StageKernels, scalar,
};

/// The f32 AVX2/FMA leaf set; callers hold a host-verified `Avx2Fma` token.
pub(crate) fn avx2_kernels_f32(radix: usize, width: usize) -> StageKernels<f32> {
    avx2_kernels(radix, width)
}

/// The f64 AVX2/FMA leaf set; callers hold a host-verified `Avx2Fma` token.
pub(crate) fn avx2_kernels_f64(radix: usize, width: usize) -> StageKernels<f64> {
    avx2_kernels(radix, width)
}

/// One AVX2 register of `LANES` consecutive complex values in one precision.
///
/// Every method requires AVX2 and FMA. `load` and `store` additionally need
/// `LANES` in-bounds values from `index`, and `gather` needs `LANES` in-range
/// permutation entries whose sources are in range.
trait Avx2Lane: CodeletLane {
    const LANES: usize;

    unsafe fn zero() -> Self::Vector;
    unsafe fn broadcast(value: Complex<Self>) -> Self::Vector;
    unsafe fn load(values: &[Complex<Self>], index: usize) -> Self::Vector;
    unsafe fn store(values: &mut [Complex<Self>], index: usize, packed: Self::Vector);
    /// Lane `j` holds `input[permutation[start + j * stride]]`.
    unsafe fn gather<const BOUNDED: bool>(
        input: &[Complex<Self>],
        permutation: &[usize],
        start: usize,
        stride: usize,
    ) -> Self::Vector;
    /// Repeat each complex value's real component in its imaginary lane.
    unsafe fn real(value: Self::Vector) -> Self::Vector;
    unsafe fn complex_mul<const CONJUGATE_RIGHT: bool>(
        left: Self::Vector,
        right: Self::Vector,
    ) -> Self::Vector;
    /// Memory order to branch order: `rows` hold `LANES` contiguous groups of
    /// four, and lane `j` of result `k` is element `k` of group `j`.
    unsafe fn to_branches(rows: [Self::Vector; 4]) -> [Self::Vector; 4];
    /// Inverse of [`Avx2Lane::to_branches`].
    unsafe fn from_branches(branches: [Self::Vector; 4]) -> [Self::Vector; 4];
}

impl CodeletLane for f64 {
    type Vector = __m256d;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn splat(value: f64) -> __m256d {
        _mm256_set1_pd(value)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vadd(left: __m256d, right: __m256d) -> __m256d {
        _mm256_add_pd(left, right)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vsub(left: __m256d, right: __m256d) -> __m256d {
        _mm256_sub_pd(left, right)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vmul(left: __m256d, right: __m256d) -> __m256d {
        _mm256_mul_pd(left, right)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rotate<const INVERSE: bool>(value: __m256d) -> __m256d {
        // Swap re/im and flip one sign bit with XOR against -0.0, which is
        // exact and cheaper than a multiply.
        let swapped = _mm256_permute_pd(value, 0b0101);
        if INVERSE {
            _mm256_xor_pd(swapped, _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0))
        } else {
            _mm256_xor_pd(swapped, _mm256_setr_pd(0.0, -0.0, 0.0, -0.0))
        }
    }
}

impl Avx2Lane for f64 {
    const LANES: usize = 2;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn zero() -> __m256d {
        _mm256_setzero_pd()
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn broadcast(value: Complex<f64>) -> __m256d {
        _mm256_setr_pd(value.re, value.im, value.re, value.im)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load(values: &[Complex<f64>], index: usize) -> __m256d {
        // SAFETY: caller guarantees two complete in-bounds complex values.
        unsafe { _mm256_loadu_pd(values.as_ptr().add(index).cast::<f64>()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store(values: &mut [Complex<f64>], index: usize, packed: __m256d) {
        // SAFETY: caller guarantees two complete in-bounds complex values.
        unsafe { _mm256_storeu_pd(values.as_mut_ptr().add(index).cast::<f64>(), packed) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gather<const BOUNDED: bool>(
        input: &[Complex<f64>],
        permutation: &[usize],
        start: usize,
        stride: usize,
    ) -> __m256d {
        // SAFETY: caller guarantees in-range permutation entries and sources.
        let (first, second) = unsafe {
            (
                *input.get_unchecked(*permutation.get_unchecked(start)),
                *input.get_unchecked(*permutation.get_unchecked(start + stride)),
            )
        };
        {
            let value = _mm256_setr_pd(first.re, first.im, second.re, second.im);
            if BOUNDED {
                let abs = simd::_mm256_andnot_pd(simd::_mm256_set1_pd(-0.0), value);
                let valid =
                    simd::_mm256_cmp_pd::<{ simd::_CMP_LE_OQ }>(abs, simd::_mm256_set1_pd(1.0));
                assert_eq!(
                    simd::_mm256_movemask_pd(valid),
                    15,
                    "bounded FFT input must have finite components with magnitude at most one"
                );
            }
            value
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn real(value: __m256d) -> __m256d {
        _mm256_movedup_pd(value)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn complex_mul<const CONJUGATE_RIGHT: bool>(left: __m256d, right: __m256d) -> __m256d {
        // (a + ib)(c + id): fmaddsub(c, [a, b], d * [b, a]) subtracts in the
        // even (real) lanes and adds in the odd (imaginary) lanes, giving
        // [ac - bd, bc + ad] with one fused multiply-add per lane.
        let right = if CONJUGATE_RIGHT {
            _mm256_mul_pd(right, _mm256_setr_pd(1.0, -1.0, 1.0, -1.0))
        } else {
            right
        };
        let real = _mm256_movedup_pd(right);
        let imaginary = _mm256_permute_pd(right, 0b1111);
        let swapped = _mm256_permute_pd(left, 0b0101);
        _mm256_fmaddsub_pd(real, left, _mm256_mul_pd(imaginary, swapped))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn to_branches([low01, low23, high01, high23]: [__m256d; 4]) -> [__m256d; 4] {
        // Group A fills the low 128-bit lane and group B the high one.
        [
            _mm256_permute2f128_pd::<0x20>(low01, high01),
            _mm256_permute2f128_pd::<0x31>(low01, high01),
            _mm256_permute2f128_pd::<0x20>(low23, high23),
            _mm256_permute2f128_pd::<0x31>(low23, high23),
        ]
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn from_branches([branch0, branch1, branch2, branch3]: [__m256d; 4]) -> [__m256d; 4] {
        [
            _mm256_permute2f128_pd::<0x20>(branch0, branch1),
            _mm256_permute2f128_pd::<0x20>(branch2, branch3),
            _mm256_permute2f128_pd::<0x31>(branch0, branch1),
            _mm256_permute2f128_pd::<0x31>(branch2, branch3),
        ]
    }
}

impl CodeletLane for f32 {
    type Vector = __m256;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn splat(value: f32) -> __m256 {
        _mm256_set1_ps(value)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vadd(left: __m256, right: __m256) -> __m256 {
        _mm256_add_ps(left, right)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vsub(left: __m256, right: __m256) -> __m256 {
        _mm256_sub_ps(left, right)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vmul(left: __m256, right: __m256) -> __m256 {
        _mm256_mul_ps(left, right)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rotate<const INVERSE: bool>(value: __m256) -> __m256 {
        let swapped = _mm256_permute_ps(value, 0b1011_0001);
        if INVERSE {
            _mm256_xor_ps(
                swapped,
                _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0),
            )
        } else {
            _mm256_xor_ps(
                swapped,
                _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0),
            )
        }
    }
}

impl Avx2Lane for f32 {
    const LANES: usize = 4;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn zero() -> __m256 {
        _mm256_setzero_ps()
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn broadcast(value: Complex<f32>) -> __m256 {
        _mm256_setr_ps(
            value.re, value.im, value.re, value.im, value.re, value.im, value.re, value.im,
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load(values: &[Complex<f32>], index: usize) -> __m256 {
        // SAFETY: caller guarantees four complete in-bounds complex values.
        unsafe { _mm256_loadu_ps(values.as_ptr().add(index).cast::<f32>()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store(values: &mut [Complex<f32>], index: usize, packed: __m256) {
        // SAFETY: caller guarantees four complete in-bounds complex values.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(index).cast::<f32>(), packed) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gather<const BOUNDED: bool>(
        input: &[Complex<f32>],
        permutation: &[usize],
        start: usize,
        stride: usize,
    ) -> __m256 {
        // SAFETY: caller guarantees in-range permutation entries and sources.
        let [first, second, third, fourth] = unsafe {
            [
                *input.get_unchecked(*permutation.get_unchecked(start)),
                *input.get_unchecked(*permutation.get_unchecked(start + stride)),
                *input.get_unchecked(*permutation.get_unchecked(start + 2 * stride)),
                *input.get_unchecked(*permutation.get_unchecked(start + 3 * stride)),
            ]
        };
        {
            let value = _mm256_setr_ps(
                first.re, first.im, second.re, second.im, third.re, third.im, fourth.re, fourth.im,
            );
            if BOUNDED {
                let abs = simd::_mm256_andnot_ps(simd::_mm256_set1_ps(-0.0), value);
                let valid =
                    simd::_mm256_cmp_ps::<{ simd::_CMP_LE_OQ }>(abs, simd::_mm256_set1_ps(1.0));
                assert_eq!(
                    simd::_mm256_movemask_ps(valid),
                    255,
                    "bounded FFT input must have finite components with magnitude at most one"
                );
            }
            value
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn real(value: __m256) -> __m256 {
        _mm256_moveldup_ps(value)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn complex_mul<const CONJUGATE_RIGHT: bool>(left: __m256, right: __m256) -> __m256 {
        // Same fmaddsub form as f64, with re/im broadcast inside each pair.
        let right = if CONJUGATE_RIGHT {
            _mm256_mul_ps(
                right,
                _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
            )
        } else {
            right
        };
        let real = _mm256_moveldup_ps(right);
        let imaginary = _mm256_movehdup_ps(right);
        let swapped = _mm256_permute_ps(left, 0b1011_0001);
        _mm256_fmaddsub_ps(real, left, _mm256_mul_ps(imaginary, swapped))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn to_branches(rows: [__m256; 4]) -> [__m256; 4] {
        // SAFETY: register-only transpose under the same target features.
        unsafe { transpose_f32(rows) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn from_branches(branches: [__m256; 4]) -> [__m256; 4] {
        // SAFETY: register-only transpose under the same target features.
        unsafe { transpose_f32(branches) }
    }
}

/// Transpose a 4x4 matrix of f32 complex values, one row per register. Each
/// complex value is one 64-bit element, so the f64 unpack/permute pattern
/// moves whole values.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn transpose_f32([row0, row1, row2, row3]: [__m256; 4]) -> [__m256; 4] {
    let (row0, row1) = (_mm256_castps_pd(row0), _mm256_castps_pd(row1));
    let (row2, row3) = (_mm256_castps_pd(row2), _mm256_castps_pd(row3));
    let low01 = _mm256_unpacklo_pd(row0, row1);
    let high01 = _mm256_unpackhi_pd(row0, row1);
    let low23 = _mm256_unpacklo_pd(row2, row3);
    let high23 = _mm256_unpackhi_pd(row2, row3);
    [
        _mm256_castpd_ps(_mm256_permute2f128_pd::<0x20>(low01, low23)),
        _mm256_castpd_ps(_mm256_permute2f128_pd::<0x20>(high01, high23)),
        _mm256_castpd_ps(_mm256_permute2f128_pd::<0x31>(low01, low23)),
        _mm256_castpd_ps(_mm256_permute2f128_pd::<0x31>(high01, high23)),
    ]
}

// Radix-4 stages dominate the planned sequences, so they get fused
// gather/stage pairs and constant-width leaves for the power-of-four widths a
// radix-4 run produces (the compiler can then fold every stride), with a
// generic radix-4 leaf for other widths. Other radices use the generic stage
// and the scalar final stage.
fn avx2_kernels<L: Avx2Lane>(radix: usize, width: usize) -> StageKernels<L> {
    StageKernels {
        forward: avx2_stage_kernel::<L, false>(radix, width),
        inverse: avx2_stage_kernel::<L, true>(radix, width),
        fused_gather_forward: (radix == 4).then_some(FusedGatherKernel::new(
            avx2_fused_gather::<L, false, false>,
            avx2_fused_gather::<L, false, true>,
        )),
        fused_gather_inverse: (radix == 4).then_some(FusedGatherKernel::new(
            avx2_fused_gather::<L, true, false>,
            avx2_fused_gather::<L, true, true>,
        )),
        fused_stage_forward: avx2_fused_stage_kernel::<L, false>(radix, width),
        fused_stage_inverse: avx2_fused_stage_kernel::<L, true>(radix, width),
        final_forward: avx2_final_kernel::<L, false>(radix, width),
        final_inverse: avx2_final_kernel::<L, true>(radix, width),
        permute: PermutationKernel::new(avx2_permute::<L, false>, avx2_permute::<L, true>),
        gather_forward: GatherKernel::new(
            avx2_gather::<L, false, false>,
            avx2_gather::<L, false, true>,
        ),
        gather_inverse: GatherKernel::new(
            avx2_gather::<L, true, false>,
            avx2_gather::<L, true, true>,
        ),
    }
}

fn avx2_stage_kernel<L: Avx2Lane, const INVERSE: bool>(
    radix: usize,
    width: usize,
) -> StageKernel<L> {
    if radix != 4 {
        return StageKernel::new(avx2_stage::<L, INVERSE>);
    }
    StageKernel::new(match width {
        1 => avx2_radix4_width_stage::<L, 1, INVERSE>,
        4 => avx2_radix4_width_stage::<L, 4, INVERSE>,
        16 => avx2_radix4_width_stage::<L, 16, INVERSE>,
        64 => avx2_radix4_width_stage::<L, 64, INVERSE>,
        256 => avx2_radix4_width_stage::<L, 256, INVERSE>,
        1_024 => avx2_radix4_width_stage::<L, 1_024, INVERSE>,
        _ => avx2_radix4_stage::<L, INVERSE>,
    })
}

fn avx2_fused_stage_kernel<L: Avx2Lane, const INVERSE: bool>(
    radix: usize,
    width: usize,
) -> Option<FusedStageKernel<L>> {
    if radix != 4 {
        return None;
    }
    let leaf = match width {
        1 => avx2_fused_stage16::<L, INVERSE>,
        16 => avx2_fused_pair::<L, 16, INVERSE>,
        64 => avx2_fused_pair::<L, 64, INVERSE>,
        256 => avx2_fused_pair::<L, 256, INVERSE>,
        _ => return None,
    };
    Some(FusedStageKernel::new(leaf))
}

fn avx2_final_kernel<L: Avx2Lane, const INVERSE: bool>(
    radix: usize,
    width: usize,
) -> FinalStageKernel<L> {
    // Width 1 has no index groups to vectorise.
    if radix != 4 || width == 1 {
        return FinalStageKernel::new(scalar_final::<L, INVERSE>);
    }
    FinalStageKernel::new(match width {
        4 => avx2_final_width::<L, 4, INVERSE>,
        16 => avx2_final_width::<L, 16, INVERSE>,
        64 => avx2_final_width::<L, 64, INVERSE>,
        256 => avx2_final_width::<L, 256, INVERSE>,
        1_024 => avx2_final_width::<L, 1_024, INVERSE>,
        _ => avx2_final::<L, INVERSE>,
    })
}

fn scalar_final<S: RealScalar, const INVERSE: bool>(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
    scale: S,
) {
    if INVERSE {
        scalar::final_inverse(input, output, radix, width, twiddles, roots, scale);
    } else {
        scalar::final_forward(input, output, radix, width, twiddles, roots, scale);
    }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_stage<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    roots: &[Complex<L>],
) {
    // SAFETY: selection with a host-verified KernelDispatch token proves
    // AVX2/FMA; the caller proves whole stage blocks and matching tables.
    unsafe { avx2_stage_leaf::<L, INVERSE>(data, radix, width, twiddles, roots) }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_radix4_width_stage<L: Avx2Lane, const WIDTH: usize, const INVERSE: bool>(
    data: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    _roots: &[Complex<L>],
) {
    debug_assert_eq!(radix, 4);
    debug_assert_eq!(width, WIDTH);
    // SAFETY: planning selected the matching constant-width radix-4 leaf and
    // selection with the host-verified KernelDispatch token proved AVX2/FMA.
    unsafe { avx2_radix4_width_leaf::<L, WIDTH, INVERSE>(data, twiddles) }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_radix4_stage<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    _roots: &[Complex<L>],
) {
    debug_assert_eq!(radix, 4);
    // SAFETY: planning selected this leaf only for a complete radix-4 stage,
    // and selection with the host-verified KernelDispatch token proved AVX2/FMA.
    unsafe { avx2_radix4_leaf::<L, INVERSE>(data, width, twiddles) }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_fused_stage16<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    _first_twiddles: &[Complex<L>],
    second_twiddles: &[Complex<L>],
) {
    // SAFETY: planning selects this leaf only for complete radix-16 blocks
    // after selection with a host-verified AVX2/FMA KernelDispatch token.
    unsafe { avx2_fused_stage16_leaf::<L, INVERSE>(data, second_twiddles) }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_fused_pair<L: Avx2Lane, const WIDTH: usize, const INVERSE: bool>(
    data: &mut [Complex<L>],
    first_twiddles: &[Complex<L>],
    second_twiddles: &[Complex<L>],
) {
    // SAFETY: planning selected matching complete radix-4 stage pairs after
    // selection with a host-verified AVX2/FMA KernelDispatch token.
    unsafe {
        avx2_fused_pair_leaf::<L, WIDTH, INVERSE>(data, first_twiddles, second_twiddles);
    }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_fused_gather<L: Avx2Lane, const INVERSE: bool, const BOUNDED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    permutation: &[usize],
    second_twiddles: &[Complex<L>],
) {
    // SAFETY: planning selects this leaf only for complete radix-16 blocks
    // after selection with a host-verified AVX2/FMA KernelDispatch token.
    unsafe {
        avx2_fused_gather_leaf::<L, INVERSE, BOUNDED>(input, output, permutation, second_twiddles);
    }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_gather<L: Avx2Lane, const INVERSE: bool, const BOUNDED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    permutation: &[usize],
    radix: usize,
    roots: &[Complex<L>],
) {
    if radix == 4 {
        // SAFETY: selection with the host-verified KernelDispatch token proved
        // AVX2/FMA and the slices share one plan length.
        unsafe { avx2_gather_radix4_leaf::<L, INVERSE, BOUNDED>(input, output, permutation) }
    } else {
        scalar::gather::<L, INVERSE, BOUNDED>(input, output, permutation, radix, roots);
    }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_permute<L: Avx2Lane, const BOUNDED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    permutation: &[usize],
) {
    // SAFETY: plan construction proves equal lengths and in-range permutation
    // sources before this preselected AVX2 leaf is stored.
    unsafe { avx2_permute_leaf::<L, BOUNDED>(input, output, permutation) }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_final_width<L: Avx2Lane, const WIDTH: usize, const INVERSE: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    _roots: &[Complex<L>],
    scale: L,
) {
    debug_assert_eq!(radix, 4);
    debug_assert_eq!(width, WIDTH);
    // Unscaled directions (scale exactly 1) select a leaf without the extra
    // multiply; the result is identical either way.
    // SAFETY: selection with a host-verified KernelDispatch token proves
    // AVX2/FMA; planning proves complete, equal-length buffers.
    unsafe {
        if scale == L::ONE {
            avx2_final_width_leaf::<L, WIDTH, INVERSE, false>(input, output, twiddles, scale);
        } else {
            avx2_final_width_leaf::<L, WIDTH, INVERSE, true>(input, output, twiddles, scale);
        }
    }
}

/// # Safety
/// The caller must prove AVX2/FMA and the matching StagePlan geometry,
/// table lengths, permutation bounds and any selected specialization.
unsafe fn avx2_final<L: Avx2Lane, const INVERSE: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    roots: &[Complex<L>],
    scale: L,
) {
    debug_assert_eq!(radix, 4);
    // The vector path has no scalar tail, so widths that are not whole
    // registers go to the portable leaf.
    if !width.is_multiple_of(L::LANES) {
        scalar_final::<L, INVERSE>(input, output, radix, width, twiddles, roots, scale);
        return;
    }
    // SAFETY: selection with the host-verified KernelDispatch token proved
    // AVX2/FMA, and planning proves complete, equal-length final-stage
    // buffers with a radix-4 width of whole registers.
    unsafe {
        if scale == L::ONE {
            avx2_final_leaf::<L, INVERSE, false>(input, output, width, twiddles, scale);
        } else {
            avx2_final_leaf::<L, INVERSE, true>(input, output, width, twiddles, scale);
        }
    }
}

/// Execute one AVX2/FMA stage of radix 2, 3, 5 or 7.
///
/// # Safety
///
/// AVX2 and FMA must be available. Complex slices are initialized, may be
/// unaligned, and do not alias. Plan metadata must match `radix` and `width`.
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_stage_leaf<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<L>],
    roots: &[Complex<L>],
) {
    match radix {
        // SAFETY: the caller established the common stage contract.
        2 => unsafe { avx2_radix_leaf::<L, 2, INVERSE>(data, width, twiddles, roots) },
        // SAFETY: the caller established the common stage contract.
        3 => unsafe { avx2_radix_leaf::<L, 3, INVERSE>(data, width, twiddles, roots) },
        // SAFETY: the caller established the common stage contract.
        5 => unsafe { avx2_radix_leaf::<L, 5, INVERSE>(data, width, twiddles, roots) },
        // SAFETY: the caller established the common stage contract.
        7 => unsafe { avx2_radix_leaf::<L, 7, INVERSE>(data, width, twiddles, roots) },
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_radix_leaf<L: Avx2Lane, const RADIX: usize, const INVERSE: bool>(
    data: &mut [Complex<L>],
    width: usize,
    twiddles: &[Complex<L>],
    roots: &[Complex<L>],
) {
    // SAFETY: the validated stage consists of whole radix*width blocks and
    // complete twiddle branches. Unaligned vector operations cover only
    // complete lanes; the shared loop handles the remainder with scalar code.
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
            L::zero(),
            L::load,
            L::store,
            L::complex_mul::<INVERSE>,
            |values| codelet::<L, RADIX, INVERSE>(&mut values, roots)
        );
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_radix4_width_leaf<L: Avx2Lane, const WIDTH: usize, const INVERSE: bool>(
    data: &mut [Complex<L>],
    twiddles: &[Complex<L>],
) {
    // SAFETY: planning established complete radix-4 blocks of width WIDTH;
    // the generic leaf folds every stride for the constant width.
    unsafe {
        if WIDTH == 1 {
            avx2_contiguous_radix4::<L, INVERSE>(data);
        } else {
            avx2_radix4_leaf::<L, INVERSE>(data, WIDTH, twiddles);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_radix4_leaf<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    width: usize,
    twiddles: &[Complex<L>],
) {
    let lanes = L::LANES;
    let vector_width = width / lanes * lanes;
    let block_size = 4 * width;
    let mut block_start = 0_usize;
    while block_start < data.len() {
        let mut index = 0_usize;
        while index + 2 * lanes <= vector_width {
            // SAFETY: both calls cover independent register-wide butterflies.
            unsafe {
                avx2_radix4_group::<L, INVERSE>(data, block_start, index, width, twiddles);
                avx2_radix4_group::<L, INVERSE>(data, block_start, index + lanes, width, twiddles);
            }
            index += 2 * lanes;
        }
        if index < vector_width {
            // SAFETY: one register-wide butterfly remains in the current block.
            unsafe { avx2_radix4_group::<L, INVERSE>(data, block_start, index, width, twiddles) };
        }
        let block = &mut data[block_start..block_start + block_size];
        for tail in vector_width..width {
            scalar::butterfly::<L, 4, INVERSE>(block, width, tail, twiddles, &[]);
        }
        block_start += block_size;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_radix4_values<L: Avx2Lane, const INVERSE: bool>(
    input: &[Complex<L>],
    block_start: usize,
    index: usize,
    width: usize,
    twiddles: &[Complex<L>],
) -> [L::Vector; 4] {
    // SAFETY: the caller proves a complete radix-4 block, `LANES` adjacent
    // indices per branch, three matching twiddle registers, and AVX2/FMA.
    unsafe {
        let mut values = [L::zero(); 7];
        values[0] = L::load(input, block_start + index);
        let mut branch = 1_usize;
        while branch < 4 {
            values[branch] = L::complex_mul::<INVERSE>(
                L::load(input, block_start + branch * width + index),
                L::load(twiddles, (branch - 1) * width + index),
            );
            branch += 1;
        }
        codelet::<L, 4, INVERSE>(&mut values, &[]);
        [values[0], values[1], values[2], values[3]]
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_radix4_group<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    block_start: usize,
    index: usize,
    width: usize,
    twiddles: &[Complex<L>],
) {
    // SAFETY: the caller establishes a complete radix-4 block and a whole
    // register of indices.
    unsafe {
        let values = avx2_radix4_values::<L, INVERSE>(data, block_start, index, width, twiddles);
        let mut branch = 0_usize;
        while branch < 4 {
            L::store(data, block_start + branch * width + index, values[branch]);
            branch += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_contiguous_radix4<L: Avx2Lane, const INVERSE: bool>(data: &mut [Complex<L>]) {
    // Width 1: each butterfly's four inputs are contiguous, so `LANES`
    // neighbouring blocks are transposed into lanes, run through one codelet,
    // and transposed back. Remaining blocks use the scalar butterfly.
    let lanes = L::LANES;
    let grouped_len = data.len() / (4 * lanes) * (4 * lanes);
    let mut start = 0_usize;
    while start < grouped_len {
        // SAFETY: the loads and stores cover `LANES` complete radix-4 blocks.
        unsafe {
            let [branch0, branch1, branch2, branch3] = L::to_branches([
                L::load(data, start),
                L::load(data, start + lanes),
                L::load(data, start + 2 * lanes),
                L::load(data, start + 3 * lanes),
            ]);
            let zero = L::zero();
            let mut values = [branch0, branch1, branch2, branch3, zero, zero, zero];
            codelet::<L, 4, INVERSE>(&mut values, &[]);
            let rows = L::from_branches([values[0], values[1], values[2], values[3]]);
            let mut row = 0_usize;
            while row < 4 {
                L::store(data, start + row * lanes, rows[row]);
                row += 1;
            }
        }
        start += 4 * lanes;
    }
    for block in data[grouped_len..].chunks_exact_mut(4) {
        scalar::butterfly::<L, 4, INVERSE>(block, 1, 0, &[], &[]);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fused_pair_leaf<L: Avx2Lane, const WIDTH: usize, const INVERSE: bool>(
    data: &mut [Complex<L>],
    first_twiddles: &[Complex<L>],
    second_twiddles: &[Complex<L>],
) {
    debug_assert!(WIDTH.is_multiple_of(L::LANES));
    debug_assert!(data.len().is_multiple_of(16 * WIDTH));
    debug_assert_eq!(first_twiddles.len(), 3 * WIDTH);
    debug_assert_eq!(second_twiddles.len(), 12 * WIDTH);
    // Two consecutive radix-4 stages (widths WIDTH and 4 * WIDTH) over one
    // 16 * WIDTH block, evaluated per register of indices without writing the
    // intermediate stage to memory. The four first-stage sub-blocks are
    // processed as the even pair (0, 2) and then the odd pair (1, 3), so only
    // half of the second-stage inputs are live at once, which limits register
    // pressure; the second stage is then the usual even/odd radix-4
    // combination of those twiddled halves.
    let mut block_start = 0_usize;
    while block_start < data.len() {
        let mut index = 0_usize;
        while index < WIDTH {
            // SAFETY: the current register of indices is in four complete
            // first-stage blocks and their following radix-4 block.
            unsafe {
                let mut branch0 = [L::zero(); 7];
                let mut branch2 = [L::zero(); 7];
                branch0[0] = L::load(data, block_start + index);
                branch2[0] = L::load(data, block_start + 8 * WIDTH + index);
                for branch in 1..4 {
                    let twiddle = L::load(first_twiddles, (branch - 1) * WIDTH + index);
                    branch0[branch] = L::complex_mul::<INVERSE>(
                        L::load(data, block_start + branch * WIDTH + index),
                        twiddle,
                    );
                    branch2[branch] = L::complex_mul::<INVERSE>(
                        L::load(data, block_start + (8 + branch) * WIDTH + index),
                        twiddle,
                    );
                }
                codelet::<L, 4, INVERSE>(&mut branch0, &[]);
                codelet::<L, 4, INVERSE>(&mut branch2, &[]);
                let mut even_sum = [L::zero(); 4];
                let mut even_difference = [L::zero(); 4];
                for branch in 0..4 {
                    let stage_index = branch * WIDTH + index;
                    let right = L::complex_mul::<INVERSE>(
                        branch2[branch],
                        L::load(second_twiddles, 4 * WIDTH + stage_index),
                    );
                    even_sum[branch] = L::vadd(branch0[branch], right);
                    even_difference[branch] = L::vsub(branch0[branch], right);
                }

                let mut branch1 = [L::zero(); 7];
                let mut branch3 = [L::zero(); 7];
                branch1[0] = L::load(data, block_start + 4 * WIDTH + index);
                branch3[0] = L::load(data, block_start + 12 * WIDTH + index);
                for branch in 1..4 {
                    let twiddle = L::load(first_twiddles, (branch - 1) * WIDTH + index);
                    branch1[branch] = L::complex_mul::<INVERSE>(
                        L::load(data, block_start + (4 + branch) * WIDTH + index),
                        twiddle,
                    );
                    branch3[branch] = L::complex_mul::<INVERSE>(
                        L::load(data, block_start + (12 + branch) * WIDTH + index),
                        twiddle,
                    );
                }
                codelet::<L, 4, INVERSE>(&mut branch1, &[]);
                codelet::<L, 4, INVERSE>(&mut branch3, &[]);
                for branch in 0..4 {
                    let stage_index = branch * WIDTH + index;
                    let odd1 = L::complex_mul::<INVERSE>(
                        branch1[branch],
                        L::load(second_twiddles, stage_index),
                    );
                    let odd3 = L::complex_mul::<INVERSE>(
                        branch3[branch],
                        L::load(second_twiddles, 8 * WIDTH + stage_index),
                    );
                    let odd_sum = L::vadd(odd1, odd3);
                    let odd_difference = L::rotate::<INVERSE>(L::vsub(odd1, odd3));
                    L::store(
                        data,
                        block_start + stage_index,
                        L::vadd(even_sum[branch], odd_sum),
                    );
                    L::store(
                        data,
                        block_start + 4 * WIDTH + stage_index,
                        L::vadd(even_difference[branch], odd_difference),
                    );
                    L::store(
                        data,
                        block_start + 8 * WIDTH + stage_index,
                        L::vsub(even_sum[branch], odd_sum),
                    );
                    L::store(
                        data,
                        block_start + 12 * WIDTH + stage_index,
                        L::vsub(even_difference[branch], odd_difference),
                    );
                }
            }
            index += L::LANES;
        }
        block_start += 16 * WIDTH;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fused_stage16_leaf<L: Avx2Lane, const INVERSE: bool>(
    data: &mut [Complex<L>],
    second_twiddles: &[Complex<L>],
) {
    debug_assert!(data.len().is_multiple_of(16));
    debug_assert_eq!(second_twiddles.len(), 12);
    // Width-1 then width-4 radix-4 stages form a 16-point DFT (4 x 4
    // Cooley-Tukey). The first stage's twiddles are all 1, which is why the
    // wrapper drops `first_twiddles`. Each set of `LANES` contiguous groups of
    // four is transposed into branch order for the column DFTs.
    let lanes = L::LANES;
    let mut start = 0_usize;
    while start < data.len() {
        // SAFETY: one complete radix-16 block remains at `start`, and every
        // load of the block precedes its in-place stores.
        unsafe {
            let mut sets = [[L::zero(); 7]; 2];
            for (set, values) in sets.iter_mut().take(4 / lanes).enumerate() {
                let offset = start + 4 * lanes * set;
                let branches = L::to_branches([
                    L::load(data, offset),
                    L::load(data, offset + lanes),
                    L::load(data, offset + 2 * lanes),
                    L::load(data, offset + 3 * lanes),
                ]);
                values[..4].copy_from_slice(&branches);
            }
            avx2_radix16_rows::<L, INVERSE>(sets, second_twiddles, data, start);
        }
        start += 16;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fused_gather_leaf<L: Avx2Lane, const INVERSE: bool, const BOUNDED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    permutation: &[usize],
    second_twiddles: &[Complex<L>],
) {
    debug_assert_eq!(output.len(), permutation.len());
    debug_assert!(output.len().is_multiple_of(16));
    debug_assert_eq!(second_twiddles.len(), 12);
    // Same 16-point structure as `avx2_fused_stage16_leaf`, but the
    // digit-reversal gather loads straight into branch order, so the permuted
    // data never goes through memory before the first butterflies.
    let lanes = L::LANES;
    let mut start = 0_usize;
    while start < output.len() {
        // SAFETY: the sixteen permutation entries of this block are complete,
        // in-range plan sources, and the output block at `start` is complete.
        unsafe {
            let mut sets = [[L::zero(); 7]; 2];
            for (set, values) in sets.iter_mut().take(4 / lanes).enumerate() {
                for (branch, value) in values.iter_mut().take(4).enumerate() {
                    *value = L::gather::<BOUNDED>(
                        input,
                        permutation,
                        start + 4 * lanes * set + branch,
                        4,
                    );
                }
            }
            avx2_radix16_rows::<L, INVERSE>(sets, second_twiddles, output, start);
        }
        start += 16;
    }
}

/// Finish one 16-point block from first-stage inputs in branch order: `sets`
/// holds `4 / LANES` sets, lane `j` of branch `k` in set `s` being element `k`
/// of group `s * LANES + j`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_radix16_rows<L: Avx2Lane, const INVERSE: bool>(
    mut sets: [[L::Vector; 7]; 2],
    second_twiddles: &[Complex<L>],
    output: &mut [Complex<L>],
    start: usize,
) {
    let lanes = L::LANES;
    // Registers per group of four, which is also the number of sets.
    let per_group = 4 / lanes;
    // SAFETY: register arithmetic, three twiddle rows of four values, and
    // stores within the complete output block at `start`.
    unsafe {
        // Column DFTs, then the second-stage twiddles. The twiddle table
        // w^(branch * index) is symmetric, so the row of output `k` supplies
        // the factors for groups `s * LANES..` directly.
        let mut rows = [L::zero(); 8];
        for (set, values) in sets.iter_mut().take(per_group).enumerate() {
            codelet::<L, 4, INVERSE>(values, &[]);
            for (row, value) in values[1..4].iter_mut().enumerate() {
                *value = L::complex_mul::<INVERSE>(
                    *value,
                    L::load(second_twiddles, row * 4 + set * lanes),
                );
            }
            let memory = L::from_branches([values[0], values[1], values[2], values[3]]);
            rows[4 * set..4 * set + 4].copy_from_slice(&memory);
        }
        // `rows` now hold the block in memory order, `LANES` values each; the
        // row DFTs combine the registers of one column across the groups.
        for column in 0..per_group {
            let mut values = [L::zero(); 7];
            for (branch, value) in values.iter_mut().take(4).enumerate() {
                *value = rows[branch * per_group + column];
            }
            codelet::<L, 4, INVERSE>(&mut values, &[]);
            for (branch, value) in values.iter().take(4).enumerate() {
                L::store(output, start + 4 * branch + column * lanes, *value);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_gather_radix4_leaf<L: Avx2Lane, const INVERSE: bool, const BOUNDED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    permutation: &[usize],
) {
    // Gather `LANES` radix-4 groups at once, one per lane, run the width-1
    // radix-4 codelet on all of them, then transpose back into the contiguous
    // groups. Leftover groups use the scalar gather.
    let lanes = L::LANES;
    let grouped_len = output.len() / (4 * lanes) * (4 * lanes);
    let mut start = 0_usize;
    while start < grouped_len {
        // SAFETY: planning constructs one in-range source for every
        // destination, the gathers stay within the current groups, and the
        // whole-register stores cover the same groups of the output slice.
        unsafe {
            let mut values = [L::zero(); 7];
            for (branch, value) in values.iter_mut().take(4).enumerate() {
                *value = L::gather::<BOUNDED>(input, permutation, start + branch, 4);
            }
            codelet::<L, 4, INVERSE>(&mut values, &[]);
            let rows = L::from_branches([values[0], values[1], values[2], values[3]]);
            for (row, value) in rows.into_iter().enumerate() {
                L::store(output, start + row * lanes, value);
            }
        }
        start += 4 * lanes;
    }
    if grouped_len < output.len() {
        scalar::gather::<L, INVERSE, BOUNDED>(
            input,
            &mut output[grouped_len..],
            &permutation[grouped_len..],
            4,
            &[],
        );
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_permute_leaf<L: Avx2Lane, const BOUNDED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    permutation: &[usize],
) {
    // Loads are inherently scattered; the gain is in assembling a full
    // register and issuing one contiguous unaligned store per register.
    let lanes = L::LANES;
    let vectorized = output.len() / lanes * lanes;
    let mut index = 0_usize;
    while index < vectorized {
        // SAFETY: the plan owns a complete, in-range permutation and a whole
        // register of destinations remains.
        unsafe {
            L::store(
                output,
                index,
                L::gather::<BOUNDED>(input, permutation, index, 1),
            )
        };
        index += lanes;
    }
    for (destination, &source) in output[vectorized..]
        .iter_mut()
        .zip(&permutation[vectorized..])
    {
        // SAFETY: plan permutation sources are in range.
        let value = unsafe { *input.get_unchecked(source) };
        *destination = if BOUNDED {
            crate::validation::bounded(value)
        } else {
            value
        };
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_final_width_leaf<
    L: Avx2Lane,
    const WIDTH: usize,
    const INVERSE: bool,
    const SCALED: bool,
>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    twiddles: &[Complex<L>],
    scale: L,
) {
    // SAFETY: the caller's contract is the generic leaf's at constant WIDTH,
    // which lets the compiler fold every stride.
    unsafe { avx2_final_leaf::<L, INVERSE, SCALED>(input, output, WIDTH, twiddles, scale) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_final_leaf<L: Avx2Lane, const INVERSE: bool, const SCALED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    width: usize,
    twiddles: &[Complex<L>],
    scale: L,
) {
    // Groups of sixteen registers expose independent butterflies to the
    // out-of-order core; narrower widths take the remainder loop only.
    const UNROLL: usize = 16;
    let lanes = L::LANES;
    let block_size = 4 * width;
    // SAFETY: register-only broadcast.
    let scale = unsafe { L::splat(scale) };
    let mut block_start = 0_usize;
    while block_start < input.len() {
        let mut index = 0_usize;
        while index + UNROLL * lanes <= width {
            let mut step = 0_usize;
            while step < UNROLL {
                // SAFETY: the group covers independent register-wide
                // butterflies in one complete final-stage block.
                unsafe {
                    avx2_final_group::<L, INVERSE, SCALED>(
                        input,
                        output,
                        block_start,
                        index + step * lanes,
                        width,
                        twiddles,
                        scale,
                    );
                }
                step += 1;
            }
            index += UNROLL * lanes;
        }
        while index < width {
            // SAFETY: a whole register of indices remains in the block.
            unsafe {
                avx2_final_group::<L, INVERSE, SCALED>(
                    input,
                    output,
                    block_start,
                    index,
                    width,
                    twiddles,
                    scale,
                );
            }
            index += lanes;
        }
        block_start += block_size;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_final_group<L: Avx2Lane, const INVERSE: bool, const SCALED: bool>(
    input: &[Complex<L>],
    output: &mut [Complex<L>],
    block_start: usize,
    index: usize,
    width: usize,
    twiddles: &[Complex<L>],
    scale: L::Vector,
) {
    // SAFETY: the stage contract proves four register-wide branch loads,
    // three twiddle loads, and four output stores.
    unsafe {
        let values = avx2_radix4_values::<L, INVERSE>(input, block_start, index, width, twiddles);
        let mut branch = 0_usize;
        while branch < 4 {
            let value = if SCALED {
                L::vmul(values[branch], scale)
            } else {
                values[branch]
            };
            L::store(output, block_start + branch * width + index, value);
            branch += 1;
        }
    }
}

/// AVX2/FMA build of a portable lane-block pass.
///
/// Panics before running the pass if dispatch does not prove AVX2/FMA.
pub(crate) fn avx2_pass<P: super::super::block::BlockPass>(
    dispatch: scioxide_core::backend::KernelDispatch,
    pass: P,
) -> P::Output {
    assert_eq!(
        dispatch.backend(),
        scioxide_core::backend::KernelBackend::Avx2Fma,
        "AVX2 pass requires a verified AVX2/FMA dispatch token"
    );
    // SAFETY: the checked host-verified dispatch proves AVX2 and FMA;
    // the pass body is safe Rust with checked indexing.
    unsafe { avx2_pass_leaf(pass) }
}

/// # Safety
/// AVX2 and FMA must be available on the host.
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_pass_leaf<P: super::super::block::BlockPass>(pass: P) -> P::Output {
    pass.run()
}

// NUFFT rows keep interleaved complex storage in registers. Repeating each
// real weight avoids the deinterleave/reinterleave shuffles of scalar loops.
#[target_feature(enable = "avx2,fma")]
unsafe fn nufft_spread<S: Avx2Lane>(
    output: &mut [Complex<S>],
    correction: &mut [Complex<S>],
    weights: &[Complex<S>],
    value: Complex<S>,
) {
    let end = output.len() / S::LANES * S::LANES;
    // SAFETY: the safe entry proves AVX2/FMA and equal row lengths. Each
    // vector starts below end, so all LANES values are in bounds; correction
    // also contains a complete row.
    unsafe {
        let value_vector = S::broadcast(value);
        for index in (0..end).step_by(S::LANES) {
            let term = S::vmul(value_vector, S::real(S::load(weights, index)));
            let sum = S::load(output, index);
            let adjusted = S::vsub(term, S::load(correction, index));
            let next = S::vadd(sum, adjusted);
            S::store(correction, index, S::vsub(S::vsub(next, sum), adjusted));
            S::store(output, index, next);
        }
    }
    for index in end..output.len() {
        let term = value * weights[index].re;
        let adjusted = term - correction[index];
        let next = output[index] + adjusted;
        correction[index] = (next - output[index]) - adjusted;
        output[index] = next;
    }
}

macro_rules! nufft_spread_entry {
    ($name:ident, $scalar:ty) => {
        pub(crate) fn $name(
            dispatch: scioxide_core::backend::KernelDispatch,
            output: &mut [Complex<$scalar>],
            correction: &mut [Complex<$scalar>],
            weights: &[Complex<$scalar>],
            value: Complex<$scalar>,
        ) {
            assert_eq!(
                dispatch.backend(),
                scioxide_core::backend::KernelBackend::Avx2Fma
            );
            assert_eq!(output.len(), weights.len());
            assert_eq!(output.len(), correction.len());
            // SAFETY: dispatch proves AVX2/FMA and all three rows have equal
            // lengths, proving every vector and scalar tail access.
            unsafe { nufft_spread::<$scalar>(output, correction, weights, value) }
        }
    };
}
nufft_spread_entry!(nufft_spread_f32, f32);
nufft_spread_entry!(nufft_spread_f64, f64);

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn nufft_pairs(
    grid: &[Complex<f64>],
    weights: &[Complex<f64>],
    index: usize,
    outer: __m256d,
) -> __m256d {
    // SAFETY: caller proves four in-bounds values in both rows and AVX2/FMA.
    let (a, b) = unsafe {
        (
            f64::vmul(
                f64::load(grid, index),
                f64::vmul(outer, f64::real(f64::load(weights, index))),
            ),
            f64::vmul(
                f64::load(grid, index + 2),
                f64::vmul(outer, f64::real(f64::load(weights, index + 2))),
            ),
        )
    };
    _mm256_add_pd(
        _mm256_permute2f128_pd(a, b, 0x20),
        _mm256_permute2f128_pd(a, b, 0x31),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn nufft_two_sum(sum: &mut __m256d, correction: &mut __m256d, term: __m256d) {
    let next = _mm256_add_pd(*sum, term);
    let part = _mm256_sub_pd(next, *sum);
    *correction = _mm256_add_pd(
        *correction,
        _mm256_add_pd(
            _mm256_sub_pd(*sum, _mm256_sub_pd(next, part)),
            _mm256_sub_pd(term, part),
        ),
    );
    *sum = next;
}

#[target_feature(enable = "avx2,fma")]
unsafe fn nufft_interpolate3(
    grid: &[Complex<f64>],
    weights: &[Complex<f64>],
    axes: [(usize, usize, usize); 3],
    width: usize,
) -> Complex<f64> {
    let [(sx, nx, wx), (sy, ny, wy), (sz, nz, wz)] = axes;
    let (mut sa, mut sb, mut ca, mut cb) = (
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    );
    let mut pending = None;
    let mut x = sx;
    for a in &weights[..wx] {
        let mut y = sy;
        for b in &weights[width..width + wy] {
            let outer = a.re * b.re;
            let base = (x * ny + y) * nz;
            let head = wz.min(nz - sz);
            for (start, row) in [
                (base + sz, &weights[2 * width..2 * width + head]),
                (base, &weights[2 * width + head..2 * width + wz]),
            ] {
                let paired = row.len() / 2 * 2;
                let mut index = 0;
                let outer_vector = _mm256_set1_pd(outer);
                let values = &grid[start..start + paired];
                // SAFETY: safe entry proves the complete periodic grid,
                // support <= extent and weight lengths. Each vector is
                // guarded by four remaining values within the split row.
                unsafe {
                    while index + 8 <= paired {
                        nufft_two_sum(
                            &mut sa,
                            &mut ca,
                            nufft_pairs(values, row, index, outer_vector),
                        );
                        nufft_two_sum(
                            &mut sb,
                            &mut cb,
                            nufft_pairs(values, row, index + 4, outer_vector),
                        );
                        index += 8;
                    }
                    if index + 4 <= paired {
                        nufft_two_sum(
                            &mut sa,
                            &mut ca,
                            nufft_pairs(values, row, index, outer_vector),
                        );
                        index += 4;
                    }
                }
                if index < paired {
                    let pair = values[index] * (outer * row[index].re)
                        + values[index + 1] * (outer * row[index + 1].re);
                    nufft_two_sum(&mut sb, &mut cb, _mm256_setr_pd(pair.re, pair.im, 0.0, 0.0));
                }
                if paired < row.len() {
                    let term = grid[start + paired] * (outer * row[paired].re);
                    if let Some(previous) = pending.take() {
                        let term: Complex<f64> = previous + term;
                        nufft_two_sum(&mut sa, &mut ca, _mm256_setr_pd(term.re, term.im, 0.0, 0.0));
                    } else {
                        pending = Some(term);
                    }
                }
            }
            y += 1;
            if y == ny {
                y = 0;
            }
        }
        x += 1;
        if x == nx {
            x = 0;
        }
    }
    let mut sums = [Complex::ZERO; 4];
    let mut corrections = sums;
    // SAFETY: each two-complex store fits its four-element local array and
    // this function runs only with the verified AVX2/FMA token.
    unsafe {
        f64::store(&mut sums, 0, sa);
        f64::store(&mut sums, 2, sb);
        f64::store(&mut corrections, 0, ca);
        f64::store(&mut corrections, 2, cb);
    }
    let (mut sum, mut correction) = (Complex::ZERO, Complex::ZERO);
    for term in sums
        .into_iter()
        .zip(corrections)
        .flat_map(|(sum, correction)| [sum, correction])
        .chain(pending)
    {
        crate::numeric::compensated_term(&mut sum.re, &mut correction.re, term.re);
        crate::numeric::compensated_term(&mut sum.im, &mut correction.im, term.im);
    }
    sum + correction
}

pub(crate) fn nufft_interpolate3_f64(
    dispatch: scioxide_core::backend::KernelDispatch,
    grid: &[Complex<f64>],
    weights: &[Complex<f64>],
    axes: [(usize, usize, usize); 3],
    width: usize,
) -> Complex<f64> {
    assert_eq!(
        dispatch.backend(),
        scioxide_core::backend::KernelBackend::Avx2Fma
    );
    assert_eq!(
        axes.iter()
            .try_fold(1_usize, |size, &(_, extent, _)| size.checked_mul(extent)),
        Some(grid.len())
    );
    assert_eq!(width.checked_mul(3), Some(weights.len()));
    assert!(
        axes.iter().all(|&(start, extent, support)| start < extent
            && support <= extent
            && support <= width)
    );
    // SAFETY: verified dispatch and checked geometry prove every periodic
    // row and weight access, including wrapped tails and degenerate axes.
    unsafe { nufft_interpolate3(grid, weights, axes, width) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn nufft_tile<S: Avx2Lane>(
    tile: &mut [Complex<S>],
    weights: &[Complex<S>],
    value: Complex<S>,
    side: usize,
    axes: [(usize, usize); 3],
) {
    let [(x, nx), (y, ny), (z, nz)] = axes;
    let width = weights.len() / 3;
    for (i, wx) in weights[..nx].iter().enumerate() {
        for (j, wy) in weights[width..width + ny].iter().enumerate() {
            let value = value * (wx.re * wy.re);
            let start = ((x + i) * side + y + j) * side + z;
            let row = &mut tile[start..start + nz];
            let weights = &weights[2 * width..2 * width + nz];
            let end = nz / S::LANES * S::LANES;
            // SAFETY: entry proves AVX2/FMA and the complete tile bounds.
            // Each vector fits the validated row and equally long weights.
            unsafe {
                let value_vector = S::broadcast(value);
                for index in (0..end).step_by(S::LANES) {
                    let term = S::vmul(value_vector, S::real(S::load(weights, index)));
                    S::store(row, index, S::vadd(S::load(row, index), term));
                }
            }
            for (out, weight) in row[end..].iter_mut().zip(&weights[end..]) {
                *out += value * weight.re;
            }
        }
    }
}

macro_rules! nufft_tile_entry {
    ($name:ident, $scalar:ty) => {
        pub(crate) fn $name(
            dispatch: scioxide_core::backend::KernelDispatch,
            tile: &mut [Complex<$scalar>],
            weights: &[Complex<$scalar>],
            value: Complex<$scalar>,
            side: usize,
            axes: [(usize, usize); 3],
        ) {
            assert_eq!(
                dispatch.backend(),
                scioxide_core::backend::KernelBackend::Avx2Fma
            );
            assert_eq!(side.checked_pow(3), Some(tile.len()));
            assert!(weights.len().is_multiple_of(3));
            assert!(axes.iter().all(|&(start, support)| start <= side
                && support <= side - start
                && support <= weights.len() / 3));
            // SAFETY: verified dispatch and checked tile geometry prove all
            // row and weight accesses, including each vector's scalar tail.
            unsafe { nufft_tile(tile, weights, value, side, axes) }
        }
    };
}
nufft_tile_entry!(nufft_tile_f32, f32);
nufft_tile_entry!(nufft_tile_f64, f64);

#[cfg(test)]
mod tests {
    use super::avx2_pass;
    use crate::kernel::block::BlockPass;
    use core::cell::Cell;
    use scioxide_core::backend::{KernelBackend, KernelDispatch};

    struct Pass<'a>(&'a Cell<bool>);

    impl BlockPass for Pass<'_> {
        type Output = usize;

        #[inline(always)]
        fn run(self) -> usize {
            self.0.set(true);
            42
        }
    }

    #[test]
    fn avx2_pass_rejects_scalar_dispatch_before_running_the_body() {
        let dispatch =
            KernelDispatch::for_backend(KernelBackend::Scalar, &scioxide_core::PortableProcessor)
                .unwrap();
        let ran = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            avx2_pass(dispatch, Pass(&ran))
        }));
        assert!(result.is_err());
        assert!(!ran.get());
    }

    #[test]
    fn avx2_pass_runs_with_verified_dispatch() {
        let context = scioxide_core::ExecutionContext::sequential_native();
        let Ok(dispatch) = KernelDispatch::for_backend(KernelBackend::Avx2Fma, context.processor())
        else {
            return;
        };
        let ran = Cell::new(false);
        assert_eq!(avx2_pass(dispatch, Pass(&ran)), 42);
        assert!(ran.get());
    }
}
