//! Fused private-vector SIMD leaves. Safe entry points check lengths and use
//! caller-resolved dispatch. Inputs are finite; callers prove update ranges
//! and validate reduction candidates before using them.
use crate::kernel::recurrence::{Moments, Products};
use scioxide_core::backend::KernelDispatch;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use scioxide_core::backend::{KernelBackend, ScalarSlice, ScalarSliceMut, ScalarValue};
use scioxide_core::{NumericScalar, SciOxideError};

/// `compensated` selects the Dot2 leaf of [`crate::kernel::recurrence::compensated_products`].
pub(crate) fn products<S: NumericScalar>(
    left: &[S],
    right: &[S],
    compensated: bool,
    dispatch: KernelDispatch,
) -> Result<Option<Products<S>>, SciOxideError> {
    if left.len() != right.len() {
        return Err(SciOxideError::dimension_mismatch(
            [left.len()],
            [right.len()],
        ));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let values = match (S::slice(left), S::slice(right), dispatch.backend()) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (ScalarSlice::F32(left), ScalarSlice::F32(right), KernelBackend::Avx2Fma) => {
                // SAFETY: verified dispatch supplies the ISA; equal initialized
                // slices were checked above. The leaf bounds every unaligned load.
                unsafe {
                    if compensated {
                        native::compensated_f32_256(left, right)
                    } else {
                        native::products_f32_256(left, right)
                    }
                }
                .map(ScalarValue::F32)
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (ScalarSlice::F64(left), ScalarSlice::F64(right), KernelBackend::Avx2Fma) => {
                // SAFETY: verified dispatch supplies the ISA; equal initialized
                // slices were checked above. The leaf bounds every unaligned load.
                unsafe {
                    if compensated {
                        native::compensated_f64_256(left, right)
                    } else {
                        native::products_f64_256(left, right)
                    }
                }
                .map(ScalarValue::F64)
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (ScalarSlice::F32(left), ScalarSlice::F32(right), KernelBackend::Avx512) => {
                // SAFETY: verified dispatch supplies the ISA; equal initialized
                // slices were checked above. The leaf bounds every unaligned load.
                unsafe {
                    if compensated {
                        native::compensated_f32_512(left, right)
                    } else {
                        native::products_f32_512(left, right)
                    }
                }
                .map(ScalarValue::F32)
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (ScalarSlice::F64(left), ScalarSlice::F64(right), KernelBackend::Avx512) => {
                // SAFETY: verified dispatch supplies the ISA; equal initialized
                // slices were checked above. The leaf bounds every unaligned load.
                unsafe {
                    if compensated {
                        native::compensated_f64_512(left, right)
                    } else {
                        native::products_f64_512(left, right)
                    }
                }
                .map(ScalarValue::F64)
            }

            #[cfg(target_arch = "aarch64")]
            (ScalarSlice::F32(left), ScalarSlice::F32(right), KernelBackend::Neon) => {
                // SAFETY: verified dispatch supplies the ISA; equal initialized
                // slices were checked above. The leaf bounds every unaligned load.
                unsafe {
                    if compensated {
                        native::compensated_f32_neon(left, right)
                    } else {
                        native::products_f32_neon(left, right)
                    }
                }
                .map(ScalarValue::F32)
            }

            #[cfg(target_arch = "aarch64")]
            (ScalarSlice::F64(left), ScalarSlice::F64(right), KernelBackend::Neon) => {
                // SAFETY: verified dispatch supplies the ISA; equal initialized
                // slices were checked above. The leaf bounds every unaligned load.
                unsafe {
                    if compensated {
                        native::compensated_f64_neon(left, right)
                    } else {
                        native::products_f64_neon(left, right)
                    }
                }
                .map(ScalarValue::F64)
            }
            _ => return Ok(None),
        };
        let [value, left_square, right_square, left_max, right_max] = values;
        Ok(Some(Products {
            value: S::from_value(value)?,
            left: Moments {
                maximum: S::from_value(left_max)?.real_component(),
                square: S::from_value(left_square)?.real_component(),
            },
            right: Moments {
                maximum: S::from_value(right_max)?.real_component(),
                square: S::from_value(right_square)?.real_component(),
            },
        }))
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (compensated, dispatch);
        Ok(None)
    }
}

/// None input denotes scaling, with no aliased shared view of the output.
#[expect(
    clippy::type_complexity,
    reason = "optional ISA specialization returns change and range statistics"
)]
pub(crate) fn update<S: NumericScalar>(
    output: &mut [S],
    alpha: S,
    input: Option<&[S]>,
    beta: S,
    dispatch: KernelDispatch,
) -> Result<Option<(bool, Moments<S::Real>)>, SciOxideError> {
    if let Some(input) = input
        && input.len() != output.len()
    {
        return Err(SciOxideError::dimension_mismatch(
            [output.len()],
            [input.len()],
        ));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let scale = input.is_none();
        let result = match (
            alpha.value(),
            S::slice(input.unwrap_or(&[])),
            beta.value(),
            S::slice_mut(output),
            dispatch.backend(),
        ) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (
                ScalarValue::F32(alpha),
                ScalarSlice::F32(input),
                ScalarValue::F32(beta),
                ScalarSliceMut::F32(output),
                KernelBackend::Avx2Fma,
            ) => {
                // SAFETY: verified dispatch supplies the ISA and lengths are checked.
                // Exclusive output cannot alias input. Scaling skips the empty input.
                let (changed, values) = unsafe {
                    if scale {
                        native::update_f32_256::<true>(output, alpha, input, beta)
                    } else {
                        native::update_f32_256::<false>(output, alpha, input, beta)
                    }
                };
                (changed, values.map(ScalarValue::F32))
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (
                ScalarValue::F64(alpha),
                ScalarSlice::F64(input),
                ScalarValue::F64(beta),
                ScalarSliceMut::F64(output),
                KernelBackend::Avx2Fma,
            ) => {
                // SAFETY: verified dispatch supplies the ISA and lengths are checked.
                // Exclusive output cannot alias input. Scaling skips the empty input.
                let (changed, values) = unsafe {
                    if scale {
                        native::update_f64_256::<true>(output, alpha, input, beta)
                    } else {
                        native::update_f64_256::<false>(output, alpha, input, beta)
                    }
                };
                (changed, values.map(ScalarValue::F64))
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (
                ScalarValue::F32(alpha),
                ScalarSlice::F32(input),
                ScalarValue::F32(beta),
                ScalarSliceMut::F32(output),
                KernelBackend::Avx512,
            ) => {
                // SAFETY: verified dispatch supplies the ISA and lengths are checked.
                // Exclusive output cannot alias input. Scaling skips the empty input.
                let (changed, values) = unsafe {
                    if scale {
                        native::update_f32_512::<true>(output, alpha, input, beta)
                    } else {
                        native::update_f32_512::<false>(output, alpha, input, beta)
                    }
                };
                (changed, values.map(ScalarValue::F32))
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            (
                ScalarValue::F64(alpha),
                ScalarSlice::F64(input),
                ScalarValue::F64(beta),
                ScalarSliceMut::F64(output),
                KernelBackend::Avx512,
            ) => {
                // SAFETY: verified dispatch supplies the ISA and lengths are checked.
                // Exclusive output cannot alias input. Scaling skips the empty input.
                let (changed, values) = unsafe {
                    if scale {
                        native::update_f64_512::<true>(output, alpha, input, beta)
                    } else {
                        native::update_f64_512::<false>(output, alpha, input, beta)
                    }
                };
                (changed, values.map(ScalarValue::F64))
            }

            #[cfg(target_arch = "aarch64")]
            (
                ScalarValue::F32(alpha),
                ScalarSlice::F32(input),
                ScalarValue::F32(beta),
                ScalarSliceMut::F32(output),
                KernelBackend::Neon,
            ) => {
                // SAFETY: verified dispatch supplies the ISA and lengths are checked.
                // Exclusive output cannot alias input. Scaling skips the empty input.
                let (changed, values) = unsafe {
                    if scale {
                        native::update_f32_neon::<true>(output, alpha, input, beta)
                    } else {
                        native::update_f32_neon::<false>(output, alpha, input, beta)
                    }
                };
                (changed, values.map(ScalarValue::F32))
            }

            #[cfg(target_arch = "aarch64")]
            (
                ScalarValue::F64(alpha),
                ScalarSlice::F64(input),
                ScalarValue::F64(beta),
                ScalarSliceMut::F64(output),
                KernelBackend::Neon,
            ) => {
                // SAFETY: verified dispatch supplies the ISA and lengths are checked.
                // Exclusive output cannot alias input. Scaling skips the empty input.
                let (changed, values) = unsafe {
                    if scale {
                        native::update_f64_neon::<true>(output, alpha, input, beta)
                    } else {
                        native::update_f64_neon::<false>(output, alpha, input, beta)
                    }
                };
                (changed, values.map(ScalarValue::F64))
            }
            _ => return Ok(None),
        };
        let (changed, [maximum, square]) = result;
        Ok(Some((
            changed,
            Moments {
                maximum: S::from_value(maximum)?.real_component(),
                square: S::from_value(square)?.real_component(),
            },
        )))
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (alpha, beta, dispatch);
        Ok(None)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
mod native {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    // The six ISA/type instantiations share traversal and reduction order.
    macro_rules! leaves {
        ($products:ident, $compensated:ident, $update:ident, $ty:ty, $lanes:expr, $features:literal,
         $splat:ident, $load:ident, $store:ident, $add:ident, $sub:ident,
         $mul:ident, $fma:ident, $max:ident, $neon:expr) => {
            /// # Safety
            /// Caller proves the target features and equal initialized spans.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $products(left: &[$ty], right: &[$ty]) -> [$ty; 5] {
                let zero = $splat(0.);
                let fma = |a, b, c| if $neon { $fma(c, a, b) } else { $fma(a, b, c) };
                let abs = |x| $max(x, $sub(zero, x));
                let mut sums = [[zero; 4]; 3];
                let mut maxima = [zero; 2];
                let mut offset = 0;
                while offset + 4 * $lanes <= left.len() {
                    for lane in 0..4 {
                        // SAFETY: the block bound covers both unaligned loads.
                        // Equal-length initialized shared spans may alias.
                        let (a, b) = unsafe {
                            (
                                $load(left.as_ptr().add(offset + lane * $lanes)),
                                $load(right.as_ptr().add(offset + lane * $lanes)),
                            )
                        };
                        sums[0][lane] = fma(a, b, sums[0][lane]);
                        sums[1][lane] = fma(a, a, sums[1][lane]);
                        sums[2][lane] = fma(b, b, sums[2][lane]);
                        maxima[0] = $max(maxima[0], abs(a));
                        maxima[1] = $max(maxima[1], abs(b));
                    }
                    offset += 4 * $lanes;
                }
                let mut sums = sums.map(|s| $add($add(s[0], s[1]), $add(s[2], s[3])));
                while offset + $lanes <= left.len() {
                    // SAFETY: a complete unaligned vector fits in both spans.
                    let (a, b) = unsafe {
                        (
                            $load(left.as_ptr().add(offset)),
                            $load(right.as_ptr().add(offset)),
                        )
                    };
                    sums[0] = fma(a, b, sums[0]);
                    sums[1] = fma(a, a, sums[1]);
                    sums[2] = fma(b, b, sums[2]);
                    maxima[0] = $max(maxima[0], abs(a));
                    maxima[1] = $max(maxima[1], abs(b));
                    offset += $lanes;
                }
                let mut result = [0.; 5];
                for (i, sum) in sums.into_iter().chain(maxima).enumerate() {
                    let mut lanes = [0.; $lanes];
                    // SAFETY: the local initialized array holds one vector.
                    unsafe { $store(lanes.as_mut_ptr(), sum) };
                    result[i] = if i < 3 {
                        lanes.into_iter().sum()
                    } else {
                        lanes.into_iter().fold(0., <$ty>::max)
                    };
                }
                for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
                    result[0] = scioxide_core::RealScalar::mul_add(a, b, result[0]);
                    result[1] = scioxide_core::RealScalar::mul_add(a, a, result[1]);
                    result[2] = scioxide_core::RealScalar::mul_add(b, b, result[2]);
                    result[3] = result[3].max(a.abs());
                    result[4] = result[4].max(b.abs());
                }
                result
            }

            /// Dot2 of `left * right` with the moments of the products leaf:
            /// two lane groups of TwoProduct and TwoSum, an error-free lane
            /// combination and a scalar Dot2 tail.
            /// # Safety
            /// Caller proves the target features and equal initialized spans.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $compensated(left: &[$ty], right: &[$ty]) -> [$ty; 5] {
                use crate::kernel::recurrence::Dot2;
                let zero = $splat(0.);
                let fma = |a, b, c| if $neon { $fma(c, a, b) } else { $fma(a, b, c) };
                let abs = |x| $max(x, $sub(zero, x));
                let dot2 = |(sum, error), a, b| {
                    let product = $mul(a, b);
                    let next = $add(sum, product);
                    let virtual_product = $sub(next, sum);
                    let rounding = $add(
                        $sub(sum, $sub(next, virtual_product)),
                        $sub(product, virtual_product),
                    );
                    let residual = fma(a, b, $sub(zero, product));
                    (next, $add(error, $add(rounding, residual)))
                };
                let mut dots = [(zero, zero); 2];
                let mut squares = [[zero; 2]; 2];
                let mut maxima = [zero; 2];
                let mut offset = 0;
                while offset + 2 * $lanes <= left.len() {
                    for group in 0..2 {
                        // SAFETY: the block bound covers both unaligned loads.
                        // Equal-length initialized shared spans may alias.
                        let (a, b) = unsafe {
                            (
                                $load(left.as_ptr().add(offset + group * $lanes)),
                                $load(right.as_ptr().add(offset + group * $lanes)),
                            )
                        };
                        dots[group] = dot2(dots[group], a, b);
                        squares[0][group] = fma(a, a, squares[0][group]);
                        squares[1][group] = fma(b, b, squares[1][group]);
                        maxima[0] = $max(maxima[0], abs(a));
                        maxima[1] = $max(maxima[1], abs(b));
                    }
                    offset += 2 * $lanes;
                }
                if offset + $lanes <= left.len() {
                    // SAFETY: a complete unaligned vector fits in both spans.
                    let (a, b) = unsafe {
                        (
                            $load(left.as_ptr().add(offset)),
                            $load(right.as_ptr().add(offset)),
                        )
                    };
                    dots[0] = dot2(dots[0], a, b);
                    squares[0][0] = fma(a, a, squares[0][0]);
                    squares[1][0] = fma(b, b, squares[1][0]);
                    maxima[0] = $max(maxima[0], abs(a));
                    maxima[1] = $max(maxima[1], abs(b));
                    offset += $lanes;
                }
                let store = |vector| {
                    let mut lanes = [0.; $lanes];
                    // SAFETY: the local initialized array holds one vector.
                    unsafe { $store(lanes.as_mut_ptr(), vector) };
                    lanes
                };
                let mut dot = Dot2::ZERO;
                for (sum, error) in dots {
                    for (sum, error) in store(sum).into_iter().zip(store(error)) {
                        dot = dot.add_term(sum, error);
                    }
                }
                let [mut left_square, mut right_square] =
                    squares.map(|s| store($add(s[0], s[1])).into_iter().sum());
                let [mut left_max, mut right_max] =
                    maxima.map(|m| store(m).into_iter().fold(0., <$ty>::max));
                for (&a, &b) in left[offset..].iter().zip(&right[offset..]) {
                    dot = dot.add_product(a, b);
                    left_square = scioxide_core::RealScalar::mul_add(a, a, left_square);
                    right_square = scioxide_core::RealScalar::mul_add(b, b, right_square);
                    left_max = left_max.max(a.abs());
                    right_max = right_max.max(b.abs());
                }
                [dot.total(), left_square, right_square, left_max, right_max]
            }

            /// # Safety
            /// Caller proves the ISA and finite bounded update. Unless SCALE,
            /// input and output have equal length. Exclusive output cannot alias input.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $update<const SCALE: bool>(
                output: &mut [$ty],
                alpha: $ty,
                input: &[$ty],
                beta: $ty,
            ) -> (bool, [$ty; 2]) {
                let zero = $splat(0.);
                let fma = |a, b, c| if $neon { $fma(c, a, b) } else { $fma(a, b, c) };
                let abs = |x| $max(x, $sub(zero, x));
                let av = $splat(alpha);
                let bv = $splat(beta);
                let mut sums = [zero; 4];
                let mut maximum = zero;
                let mut difference = zero;
                let mut offset = 0;
                while offset + 4 * $lanes <= output.len() {
                    for (lane, sum) in sums.iter_mut().enumerate() {
                        let i = offset + lane * $lanes;
                        // SAFETY: the block bound covers loads and exclusive
                        // stores. SCALE skips input; otherwise lengths match.
                        let (old, next) = unsafe {
                            let old = $load(output.as_ptr().add(i));
                            let next = if SCALE {
                                $mul(bv, old)
                            } else {
                                fma(av, $load(input.as_ptr().add(i)), $mul(bv, old))
                            };
                            $store(output.as_mut_ptr().add(i), next);
                            (old, next)
                        };
                        *sum = fma(next, next, *sum);
                        maximum = $max(maximum, abs(next));
                        difference = $max(difference, abs($sub(next, old)));
                    }
                    offset += 4 * $lanes;
                }
                let mut sum = $add($add(sums[0], sums[1]), $add(sums[2], sums[3]));
                while offset + $lanes <= output.len() {
                    // SAFETY: one full vector fits in output and, unless SCALE,
                    // equal-length input. Exclusive stores cannot alias input.
                    let (old, next) = unsafe {
                        let old = $load(output.as_ptr().add(offset));
                        let next = if SCALE {
                            $mul(bv, old)
                        } else {
                            fma(av, $load(input.as_ptr().add(offset)), $mul(bv, old))
                        };
                        $store(output.as_mut_ptr().add(offset), next);
                        (old, next)
                    };
                    sum = fma(next, next, sum);
                    maximum = $max(maximum, abs(next));
                    difference = $max(difference, abs($sub(next, old)));
                    offset += $lanes;
                }
                let mut lanes = [[0.; $lanes]; 3];
                for (out, value) in lanes.iter_mut().zip([maximum, sum, difference]) {
                    // SAFETY: each initialized local array holds one vector.
                    unsafe { $store(out.as_mut_ptr(), value) };
                }
                let mut maximum = lanes[0].into_iter().fold(0., <$ty>::max);
                let mut square = lanes[1].into_iter().sum();
                let mut changed = lanes[2].into_iter().any(|v| v != 0.);
                for (i, out) in output[offset..].iter_mut().enumerate() {
                    let next = if SCALE {
                        beta * *out
                    } else {
                        scioxide_core::RealScalar::mul_add(alpha, input[offset + i], beta * *out)
                    };
                    changed |= next != *out;
                    *out = next;
                    maximum = maximum.max(next.abs());
                    square = scioxide_core::RealScalar::mul_add(next, next, square);
                }
                (changed, [maximum, square])
            }
        };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    leaves!(
        products_f32_256,
        compensated_f32_256,
        update_f32_256,
        f32,
        8,
        "avx2,fma",
        _mm256_set1_ps,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_sub_ps,
        _mm256_mul_ps,
        _mm256_fmadd_ps,
        _mm256_max_ps,
        false
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    leaves!(
        products_f64_256,
        compensated_f64_256,
        update_f64_256,
        f64,
        4,
        "avx2,fma",
        _mm256_set1_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        _mm256_fmadd_pd,
        _mm256_max_pd,
        false
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    leaves!(
        products_f32_512,
        compensated_f32_512,
        update_f32_512,
        f32,
        16,
        "avx512f,avx512dq,avx512vl",
        _mm512_set1_ps,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_sub_ps,
        _mm512_mul_ps,
        _mm512_fmadd_ps,
        _mm512_max_ps,
        false
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    leaves!(
        products_f64_512,
        compensated_f64_512,
        update_f64_512,
        f64,
        8,
        "avx512f,avx512dq,avx512vl",
        _mm512_set1_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_mul_pd,
        _mm512_fmadd_pd,
        _mm512_max_pd,
        false
    );
    #[cfg(target_arch = "aarch64")]
    leaves!(
        products_f32_neon,
        compensated_f32_neon,
        update_f32_neon,
        f32,
        4,
        "neon",
        vdupq_n_f32,
        vld1q_f32,
        vst1q_f32,
        vaddq_f32,
        vsubq_f32,
        vmulq_f32,
        vfmaq_f32,
        vmaxq_f32,
        true
    );
    #[cfg(target_arch = "aarch64")]
    leaves!(
        products_f64_neon,
        compensated_f64_neon,
        update_f64_neon,
        f64,
        2,
        "neon",
        vdupq_n_f64,
        vld1q_f64,
        vst1q_f64,
        vaddq_f64,
        vsubq_f64,
        vmulq_f64,
        vfmaq_f64,
        vmaxq_f64,
        true
    );
}
