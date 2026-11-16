//! Checked, numerically stable reductions for linear algebra kernels.

use scioxide_core::{
    Dimensions, ExecutionContext, NumericScalar, ProductSum, RealScalar, ScaledNorm, SciOxideError,
};

use crate::dense::{FiniteVectorView, VectorView};
use crate::kernel::standard;
use crate::{Accuracy, KernelContext};

pub(crate) fn dotu_with_context<'a, S: NumericScalar>(
    left: VectorView<'_, S>,
    right: VectorView<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<S, SciOxideError> {
    dot_with_context(left, right, false, context)
}

pub(crate) fn dot_with_context<'a, S: NumericScalar>(
    left: VectorView<'_, S>,
    right: VectorView<'_, S>,
    conjugate_left: bool,
    context: impl Into<KernelContext<'a>>,
) -> Result<S, SciOxideError> {
    validate_dot_dimensions(left, right)?;
    let context: crate::KernelContext<'_> = context.into();
    let left = left.try_finite_dispatched(context.dispatch())?;
    let right = right.try_finite_dispatched(context.dispatch())?;
    context.execution().check()?;
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas()
        && let Some(result) = crate::blas::dispatch::dot(
            left.as_view(),
            right.as_view(),
            conjugate_left,
            dispatch,
            context.accuracy(),
            crate::kernel::provider_policy_workers(context),
        )?
    {
        return Ok(result);
    }
    match context.accuracy() {
        Accuracy::Exact => canonical_dot(left, right, conjugate_left),
        Accuracy::Standard => standard_dot(left, right, conjugate_left, context.dispatch()),
    }
}

fn validate_dot_dimensions<S: NumericScalar>(
    left: VectorView<'_, S>,
    right: VectorView<'_, S>,
) -> Result<(), SciOxideError> {
    if left.len() != right.len() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(left.len()),
            Dimensions::one(right.len()),
        ));
    }
    Ok(())
}

pub(crate) fn standard_dot<S: NumericScalar>(
    left: FiniteVectorView<'_, S>,
    right: FiniteVectorView<'_, S>,
    conjugate_left: bool,
    dispatch: scioxide_core::backend::KernelDispatch,
) -> Result<S, SciOxideError> {
    let (a, b) = standard::product_scales(left.maximum, right.maximum, left.len())?;
    if a == 0
        && b == 0
        && let (Some(left), Some(right)) = (
            left.as_view().as_contiguous_slice(),
            right.as_view().as_contiguous_slice(),
        )
        && let Some(sum) = crate::kernel::arch::dot(left, right, dispatch)?
    {
        return standard::restore(sum, 0);
    }
    standard::unit_then_normalized(a, b, |a, b| {
        standard::dot(left.iter().zip(right.iter()).map(|(x, y)| {
            (
                standard::scale(if conjugate_left { x.conj() } else { x }, -a),
                standard::scale(y, -b),
            )
        }))
    })
}

pub(crate) fn canonical_dot<S: NumericScalar>(
    left: FiniteVectorView<'_, S>,
    right: FiniteVectorView<'_, S>,
    conjugate_left: bool,
) -> Result<S, SciOxideError> {
    let mut sum = ProductSum::new();
    left.iter()
        .zip(right.iter())
        .try_for_each(|(left, right)| {
            sum.add_product(if conjugate_left { left.conj() } else { left }, right)
        })?;
    sum.finish()
}

pub(crate) fn l2_norm_with_context<'a, S: NumericScalar>(
    values: VectorView<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<S::Real, SciOxideError> {
    let context = context.into();
    let values = values.try_finite_dispatched(context.dispatch())?;
    context.execution().check()?;
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas()
        && let Some(result) = crate::blas::dispatch::norm2(
            values.as_view(),
            dispatch,
            context.accuracy(),
            crate::kernel::provider_policy_workers(context),
        )?
    {
        return Ok(result);
    }
    match context.accuracy() {
        Accuracy::Exact => canonical_norm_state(values)?.finish(),
        Accuracy::Standard => {
            // Ordinary real squares admit a blocked dot. Extreme inputs retain
            // Blue's three buckets; input validation and bounds are reused.
            if S::IS_REAL
                && standard::product_scales(values.maximum, values.maximum, values.len())? == (0, 0)
            {
                let sum = if let Some(slice) = values.as_view().as_contiguous_slice()
                    && let Some(sum) = crate::kernel::arch::dot(slice, slice, context.dispatch())?
                {
                    sum
                } else {
                    standard::dot(values.iter().map(|value| (value, value)))
                };
                return crate::checked::finite_candidate(sum.real_component().sqrt());
            }
            standard::norm_finite(values)
        }
    }
}

pub(crate) fn scaled_norm_state_with_context<S: NumericScalar>(
    values: VectorView<'_, S>,
    context: ExecutionContext<'_>,
) -> Result<ScaledNorm<S::Real>, SciOxideError> {
    let values = values.try_finite()?;
    context.check()?;
    canonical_norm_state(values)
}

fn canonical_norm_state<S: NumericScalar>(
    values: FiniteVectorView<'_, S>,
) -> Result<ScaledNorm<S::Real>, SciOxideError> {
    // `|a + ib|^2 = a^2 + b^2`, so a complex entry contributes its real and
    // imaginary magnitudes as two separate terms. This avoids forming `|z|`
    // (a hypot) per element and keeps the scaled recurrence purely real.
    values
        .iter()
        .try_fold(ScaledNorm::new(), |mut state, value| {
            state.add_magnitude(value.real_component().abs())?;
            state.add_magnitude(value.imaginary_component().abs())?;
            Ok(state)
        })
}

/// Neumaier reduction of `offset + sign * sum(left * right)` on finite views.
/// Native products retry range hazards in a binary frame. Its exponent stays
/// separate until a solve applies its filter or diagonal divisor; no extended
/// arithmetic occurs in the element loop.
pub(crate) fn compensated_dot<S: NumericScalar>(
    left: FiniteVectorView<'_, S>,
    right: FiniteVectorView<'_, S>,
    offset: S,
    sign: S,
) -> Result<(S, i32), SciOxideError> {
    use scioxide_core::{AlgebraicScalar, RealScalar};
    fn add<R: RealScalar>(sum: &mut R, correction: &mut R, value: R) {
        let next = *sum + value;
        *correction += if sum.abs() >= value.abs() {
            (*sum - next) + value
        } else {
            (value - next) + *sum
        };
        *sum = next;
    }
    debug_assert_eq!(left.len(), right.len());
    if left.maximum == S::Real::ZERO || right.maximum == S::Real::ZERO {
        let exponent = standard::exponent(standard::magnitude(offset))?;
        return Ok((standard::scale(offset, -exponent), exponent));
    }
    // Independent native lanes shorten the dependency chain. Merge their
    // Neumaier parts in lane order; replay range hazards in the binary frame.
    debug_assert!(sign == S::ONE || sign == -S::ONE);
    let negative = sign == -S::ONE;
    let native = || {
        let mut real = [S::Real::ZERO; 4];
        let mut imaginary = real;
        let mut rc = real;
        let mut ic = real;
        real[0] = offset.real_component();
        imaginary[0] = offset.imaginary_component();
        let mut products = left.iter().zip(right.iter());
        'products: loop {
            for (((re, rc), im), ic) in real
                .iter_mut()
                .zip(&mut rc)
                .zip(&mut imaginary)
                .zip(&mut ic)
            {
                let Some((a, b)) = products.next() else {
                    break 'products;
                };
                let signed = if negative { -a } else { a };
                let product = signed * b;
                if scioxide_core::backend::standard::product_needs_retry(signed, b, product) {
                    return None;
                }
                add(re, rc, signed.real_component() * b.real_component());
                if !S::IS_REAL {
                    add(
                        re,
                        rc,
                        -signed.imaginary_component() * b.imaginary_component(),
                    );
                    add(im, ic, signed.real_component() * b.imaginary_component());
                    add(im, ic, signed.imaginary_component() * b.real_component());
                }
            }
        }
        let (mut re, mut im, mut re_correction, mut im_correction) =
            (S::Real::ZERO, S::Real::ZERO, S::Real::ZERO, S::Real::ZERO);
        for (((real, rc), imaginary), ic) in real.into_iter().zip(rc).zip(imaginary).zip(ic) {
            add(&mut re, &mut re_correction, real);
            add(&mut re, &mut re_correction, rc);
            if !S::IS_REAL {
                add(&mut im, &mut im_correction, imaginary);
                add(&mut im, &mut im_correction, ic);
            }
        }
        let result = S::from_components(re + re_correction, im + im_correction);
        result.has_finite_components().then_some(result)
    };
    if let Some(result) = native() {
        // Downstream filters and triangular solves consume a bounded
        // significand before restoring this exponent, even for a huge sum.
        let exponent = standard::exponent(standard::magnitude(result))?;
        return Ok((standard::scale(result, -exponent), exponent));
    }
    let ae = standard::exponent(left.maximum)?;
    let be = standard::exponent(right.maximum)?;
    let e = if offset == S::ZERO {
        ae + be
    } else {
        (ae + be).max(standard::exponent(standard::magnitude(offset))?)
    };
    let initial = standard::scale(offset, -e);
    let (mut real, mut imaginary) = (initial.real_component(), initial.imaginary_component());
    let (mut rc, mut ic) = (S::Real::ZERO, S::Real::ZERO);
    for (a, b) in left.iter().zip(right.iter()) {
        let a = sign * standard::scale(a, -ae);
        let b = standard::scale(b, -be);
        let shift = ae + be - e;
        add(
            &mut real,
            &mut rc,
            (a.real_component() * b.real_component()).scalbn(shift),
        );
        if !S::IS_REAL {
            add(
                &mut real,
                &mut rc,
                (-a.imaginary_component() * b.imaginary_component()).scalbn(shift),
            );
            add(
                &mut imaginary,
                &mut ic,
                (a.real_component() * b.imaginary_component()).scalbn(shift),
            );
            add(
                &mut imaginary,
                &mut ic,
                (a.imaginary_component() * b.real_component()).scalbn(shift),
            );
        }
    }
    Ok((S::from_components(real + rc, imaginary + ic), e))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use scioxide_core::{AlgebraicScalar, Complex};

    fn compensated_native_and_binary_replay<S: NumericScalar>() {
        let real = S::from_real;
        let one = S::Real::ONE;
        let large = one / S::Real::EPSILON;
        let left = [S::ONE; 3];
        let right = [real(large), S::ONE, -real(large)];
        let finite = |values| FiniteVectorView::from_workspace(VectorView::from_slice(values));
        let (sum, exponent) =
            compensated_dot(finite(&left), finite(&right), S::ZERO, S::ONE).unwrap();
        assert_eq!(standard::restore(sum, exponent).unwrap(), S::ONE);

        // An overflowing product cancels to a representable result.
        let left = [real(S::Real::MAX); 2];
        let right = [real(one + one), -S::ONE];
        let (sum, exponent) =
            compensated_dot(finite(&left), finite(&right), S::ZERO, S::ONE).unwrap();
        let result = standard::restore(sum, exponent).unwrap();
        assert!(standard::magnitude(sum) <= one);
        assert_eq!(result, real(S::Real::MAX));

        // Keep products below the destination range in a binary frame until
        // the solve applies its singular-value filter or diagonal divisor.
        let tiny = S::Real::MIN_POSITIVE;
        let left = [real(tiny)];
        let (sum, exponent) =
            compensated_dot(finite(&left), finite(&left), S::ZERO, S::ONE).unwrap();
        let expected = scioxide_core::Scaled::from_value(tiny)
            .unwrap()
            .mul(scioxide_core::Scaled::from_value(tiny).unwrap())
            .unwrap();
        let ratio = scioxide_core::Scaled::from_value(sum.real_component())
            .unwrap()
            .shifted(i64::from(exponent))
            .unwrap()
            .div(expected)
            .unwrap()
            .round()
            .unwrap();
        assert!((ratio - one).abs() <= S::Real::EPSILON);

        let (sum, exponent) = compensated_dot(
            finite(&[S::ONE; 2]),
            finite(&[real(large), -S::ONE]),
            real(large),
            -S::ONE,
        )
        .unwrap();
        assert_eq!(standard::restore(sum, exponent).unwrap(), S::ONE);
    }

    #[test]
    fn compensated_dots_retain_native_corrections_and_range_safe_binary_replay() {
        compensated_native_and_binary_replay::<f32>();
        compensated_native_and_binary_replay::<f64>();
        compensated_native_and_binary_replay::<Complex<f32>>();
        compensated_native_and_binary_replay::<Complex<f64>>();
    }

    fn assert_close_f64(actual: f64, expected: f64, tolerance: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_close_f32(actual: f32, expected: f32, tolerance: f32) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn recovers_cross_scaled_products() {
        let left_f64 = [f64::MAX, f64::MIN_POSITIVE];
        let right_f64 = [f64::MIN_POSITIVE, f64::MAX];
        let left_f32 = [f32::MAX, f32::MIN_POSITIVE];
        let right_f32 = [f32::MIN_POSITIVE, f32::MAX];

        assert_eq!(
            crate::VectorView::from_slice(&left_f64)
                .dotu_scoped(
                    crate::VectorView::from_slice(&right_f64),
                    &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                    crate::LinalgExecutionContext::sequential()
                )
                .unwrap()
                .to_bits(),
            0x401f_ffff_ffff_ffff
        );
        assert_eq!(
            crate::VectorView::from_slice(&left_f32)
                .dotu_scoped(
                    crate::VectorView::from_slice(&right_f32),
                    &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                    crate::LinalgExecutionContext::sequential()
                )
                .unwrap()
                .to_bits(),
            0x40ff_ffff
        );
    }

    fn standard<S: NumericScalar>(
        left: &[S],
        right: &[S],
        conjugate: bool,
    ) -> Result<S, SciOxideError> {
        dot_with_context(
            VectorView::from_slice(left),
            VectorView::from_slice(right),
            conjugate,
            KernelContext::sequential().with_accuracy(Accuracy::Standard),
        )
    }

    #[test]
    fn standard_dots_normalize_only_after_unscaled_overflow() {
        // Maxima near the range limits used to normalize both operands and
        // flush MIN_POSITIVE before multiplying it by MAX.
        let left = [f64::MAX, f64::MIN_POSITIVE];
        let right = [f64::MIN_POSITIVE, f64::MAX];
        assert_eq!(
            standard(&left, &right, false).unwrap().to_bits(),
            0x401f_ffff_ffff_ffff
        );
        let left = [f32::MAX, f32::MIN_POSITIVE];
        let right = [f32::MIN_POSITIVE, f32::MAX];
        assert_eq!(
            standard(&left, &right, false).unwrap().to_bits(),
            0x40ff_ffff
        );
        let left = [
            Complex::new(f64::MAX, 0.0),
            Complex::new(0.0, f64::MIN_POSITIVE),
        ];
        let right = [
            Complex::new(f64::MIN_POSITIVE, 0.0),
            Complex::new(0.0, f64::MAX),
        ];
        let conjugated = standard(&left, &right, true).unwrap();
        assert_eq!(conjugated.re.to_bits(), 0x401f_ffff_ffff_ffff);
        assert_eq!(conjugated.im, 0.0);

        // The binary32 offset dot follows the same rule.
        let offset = VectorView::from_slice(&[f32::MAX, f32::MIN_POSITIVE]);
        let other = VectorView::from_slice(&[f32::MIN_POSITIVE, f32::MAX]);
        let exact = KernelContext::sequential().with_accuracy(Accuracy::Exact);
        assert_eq!(
            offset
                .dot_plus_scoped(
                    other,
                    1.0,
                    &crate::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            offset
                .dot_plus_scoped(
                    other,
                    1.0,
                    &crate::LinalgExecutionOptions::new(),
                    crate::LinalgExecutionContext::unscoped(exact)
                )
                .unwrap()
        );

        // An unscaled overflow still falls back to normalized operands, so a
        // representable cancelling sum survives and a true overflow is reported.
        assert_eq!(
            standard(&[f64::MAX, -f64::MAX], &[2.0, 2.0], false).unwrap(),
            0.0
        );
        assert!(matches!(
            standard(&[f64::MAX], &[2.0_f64], false).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }

    fn reference_dot<S: NumericScalar>(left: &[S], right: &[S]) -> S {
        let mut sum = ProductSum::new();
        left.iter()
            .zip(right)
            .for_each(|(&a, &b)| sum.add_product(a, b).unwrap());
        sum.finish().unwrap()
    }

    fn reference_norm<S: NumericScalar>(values: &[S]) -> S::Real {
        let mut sum = ScaledNorm::new();
        values.iter().for_each(|value| {
            sum.add_magnitude(value.real_component().abs()).unwrap();
            sum.add_magnitude(value.imaginary_component().abs())
                .unwrap();
        });
        sum.finish().unwrap()
    }

    #[test]
    fn all_scalar_families_match_reference_for_offset_tails() {
        let left_f64 = (0..262)
            .map(|index| ((index * 17 + 3) % 41) as f64 / 13.0 - 1.5)
            .collect::<Vec<_>>();
        let right_f64 = (0..262)
            .map(|index| ((index * 11 + 7) % 37) as f64 / 17.0 - 1.0)
            .collect::<Vec<_>>();
        let left_f32 = left_f64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let right_f32 = right_f64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let left_complex_f64 = left_f64
            .iter()
            .zip(right_f64.iter())
            .map(|(&real, &imaginary)| Complex::new(real, imaginary))
            .collect::<Vec<_>>();
        let right_complex_f64 = right_f64
            .iter()
            .zip(left_f64.iter())
            .map(|(&real, &imaginary)| Complex::new(real, imaginary * 0.5))
            .collect::<Vec<_>>();
        let left_complex_f32 = left_complex_f64
            .iter()
            .map(|value| Complex::new(value.re as f32, value.im as f32))
            .collect::<Vec<_>>();
        let right_complex_f32 = right_complex_f64
            .iter()
            .map(|value| Complex::new(value.re as f32, value.im as f32))
            .collect::<Vec<_>>();

        let left_f64 = &left_f64[1..260];
        let right_f64 = &right_f64[2..261];
        let left_f32 = &left_f32[1..260];
        let right_f32 = &right_f32[2..261];
        let left_complex_f64 = &left_complex_f64[1..260];
        let right_complex_f64 = &right_complex_f64[2..261];
        let left_complex_f32 = &left_complex_f32[1..260];
        let right_complex_f32 = &right_complex_f32[2..261];

        let expected_dot_f64 = reference_dot(left_f64, right_f64);
        let expected_norm_f64 = reference_norm(left_f64);
        let expected_dot_f32 = reference_dot(left_f32, right_f32);
        let expected_norm_f32 = reference_norm(left_f32);
        let expected_dot_complex_f64 = reference_dot(left_complex_f64, right_complex_f64);
        let expected_norm_complex_f64 = reference_norm(left_complex_f64);
        let expected_dot_complex_f32 = reference_dot(left_complex_f32, right_complex_f32);
        let expected_norm_complex_f32 = reference_norm(left_complex_f32);

        let dot_f64 = crate::VectorView::from_slice(left_f64)
            .dotu_scoped(
                crate::VectorView::from_slice(right_f64),
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential(),
            )
            .unwrap();
        let norm_f64 = crate::VectorView::from_slice(left_f64)
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_close_f64(dot_f64, expected_dot_f64, 2.0e-14);
        assert_close_f64(norm_f64, expected_norm_f64, 2.0e-14);

        let dot_f32 = crate::VectorView::from_slice(left_f32)
            .dotu_scoped(
                crate::VectorView::from_slice(right_f32),
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential(),
            )
            .unwrap();
        let norm_f32 = crate::VectorView::from_slice(left_f32)
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_close_f32(dot_f32, expected_dot_f32, 3.0e-5);
        assert_close_f32(norm_f32, expected_norm_f32, 3.0e-5);

        let dot_complex_f64 = crate::VectorView::from_slice(left_complex_f64)
            .dotu_scoped(
                crate::VectorView::from_slice(right_complex_f64),
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential(),
            )
            .unwrap();
        let norm_complex_f64 = crate::VectorView::from_slice(left_complex_f64)
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_close_f64(dot_complex_f64.re, expected_dot_complex_f64.re, 3.0e-14);
        assert_close_f64(dot_complex_f64.im, expected_dot_complex_f64.im, 3.0e-14);
        assert_close_f64(norm_complex_f64, expected_norm_complex_f64, 2.0e-14);

        let dot_complex_f32 = crate::VectorView::from_slice(left_complex_f32)
            .dotu_scoped(
                crate::VectorView::from_slice(right_complex_f32),
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential(),
            )
            .unwrap();
        let norm_complex_f32 = crate::VectorView::from_slice(left_complex_f32)
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_close_f32(dot_complex_f32.re, expected_dot_complex_f32.re, 4.0e-5);
        assert_close_f32(dot_complex_f32.im, expected_dot_complex_f32.im, 4.0e-5);
        assert_close_f32(norm_complex_f32, expected_norm_complex_f32, 3.0e-5);
    }

    #[test]
    fn preserves_cancellation_and_scaled_norm_extremes() {
        let cancellation_f64 = [1.0e16_f64, 1.0, -1.0e16];
        let cancellation_f32 = [1.0e8_f32, 1.0, -1.0e8];
        let ones_f64 = [1.0_f64; 3];
        let ones_f32 = [1.0_f32; 3];
        let smallest_f64 = f64::from_bits(1);
        let smallest_f32 = f32::from_bits(1);
        let subnormal_f64 = [smallest_f64; 67];
        let subnormal_f32 = [smallest_f32; 67];
        let near_overflow_f64 = [f64::MAX / 8.0; 17];
        let near_overflow_f32 = [f32::MAX / 8.0; 17];
        let expected_subnormal_f64 = reference_norm(&subnormal_f64);
        let expected_subnormal_f32 = reference_norm(&subnormal_f32);
        let expected_near_overflow_f64 = reference_norm(&near_overflow_f64);
        let expected_near_overflow_f32 = reference_norm(&near_overflow_f32);

        assert_eq!(
            crate::VectorView::from_slice(&cancellation_f64)
                .dotu_scoped(
                    crate::VectorView::from_slice(&ones_f64),
                    &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                    crate::LinalgExecutionContext::sequential()
                )
                .unwrap(),
            1.0
        );
        assert_eq!(
            crate::VectorView::from_slice(&cancellation_f32)
                .dotu_scoped(
                    crate::VectorView::from_slice(&ones_f32),
                    &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                    crate::LinalgExecutionContext::sequential()
                )
                .unwrap(),
            1.0
        );
        assert_eq!(
            crate::VectorView::from_slice(&subnormal_f64)
                .norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    ExecutionContext::sequential()
                )
                .unwrap(),
            expected_subnormal_f64
        );
        assert_eq!(
            crate::VectorView::from_slice(&subnormal_f32)
                .norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    ExecutionContext::sequential()
                )
                .unwrap(),
            expected_subnormal_f32
        );
        assert_close_f64(
            crate::VectorView::from_slice(&near_overflow_f64)
                .norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap(),
            expected_near_overflow_f64,
            3.0e-15,
        );
        assert_close_f32(
            crate::VectorView::from_slice(&near_overflow_f32)
                .norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap(),
            expected_near_overflow_f32,
            3.0e-6,
        );
    }

    #[test]
    fn preserves_errors_empty_results_and_determinism() {
        assert_eq!(
            VectorView::<f64>::from_slice(&[])
                .dotu_scoped(
                    VectorView::from_slice(&[]),
                    &crate::LinalgExecutionOptions::new(),
                    ExecutionContext::sequential()
                )
                .unwrap(),
            0.0
        );
        assert_eq!(
            VectorView::<Complex<f32>>::from_slice(&[])
                .norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    ExecutionContext::sequential()
                )
                .unwrap(),
            0.0
        );

        let non_finite_dot = crate::VectorView::from_slice(&[1.0_f64, f64::INFINITY]).dotu_scoped(
            crate::VectorView::from_slice(&[1.0, 1.0]),
            &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
            crate::LinalgExecutionContext::sequential(),
        );
        assert!(matches!(
            (non_finite_dot).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        let non_finite_norm = crate::VectorView::from_slice(&[Complex::new(1.0_f32, f32::NAN)])
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                ExecutionContext::sequential(),
            );
        assert!(matches!(
            (non_finite_norm).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        let term_overflow = crate::VectorView::from_slice(&[f64::MAX]).dotu_scoped(
            crate::VectorView::from_slice(&[2.0_f64]),
            &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
            crate::LinalgExecutionContext::sequential(),
        );
        assert!(matches!(
            (term_overflow).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        let norm_overflow = crate::VectorView::from_slice(&[f64::MAX, f64::MAX]).norm2_scoped(
            &crate::LinalgExecutionOptions::new(),
            ExecutionContext::sequential(),
        );
        assert!(matches!(
            (norm_overflow).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));

        let values = (0..521)
            .map(|index| ((index * 13 + 5) % 31) as f64 - 15.0)
            .collect::<Vec<_>>();
        let first_dot = crate::VectorView::from_slice(&values)
            .dotu_scoped(
                crate::VectorView::from_slice(&values),
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential(),
            )
            .unwrap();
        let first_norm = crate::VectorView::from_slice(&values)
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        (0..8).for_each(|_| {
            assert_eq!(
                crate::VectorView::from_slice(&values)
                    .dotu_scoped(
                        crate::VectorView::from_slice(&values),
                        &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                        crate::LinalgExecutionContext::sequential()
                    )
                    .unwrap(),
                first_dot
            );
            assert_eq!(
                crate::VectorView::from_slice(&values)
                    .norm2_scoped(
                        &crate::LinalgExecutionOptions::new(),
                        ExecutionContext::sequential()
                    )
                    .unwrap(),
                first_norm
            );
        });
    }
}
