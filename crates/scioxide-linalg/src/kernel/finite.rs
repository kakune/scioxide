//! Finite private vector primitives shared by actions, Krylov solvers and
//! decompositions.
//!
//! Every operand is private storage whose producer (staging, a checked kernel
//! or a validated provider) already established finiteness, so nothing here
//! repeats public input validation. Range proofs, dispatched leaves and fused
//! moments come from [`super::recurrence`] and [`super::standard`]; callers
//! resolve the [`KernelDispatch`] once and own publication.

use scioxide_core::backend::KernelDispatch;
use scioxide_core::{
    AlgebraicScalar, Dimensions, ExecutionContext, NumericScalar, RealScalar, ScaledValue,
    SciOxideError,
};

use super::recurrence::Moments;
use super::standard;

/// Largest component magnitude `max(|re|, |im|)`, zero for an empty slice.
///
/// Dispatched leaves and the scalar fallback both fuse the scan with a
/// finiteness check, so a non-finite component reports an error rather than a
/// meaningless bound on every backend.
pub(crate) fn maximum<S: NumericScalar>(
    values: &[S],
    dispatch: KernelDispatch,
) -> Result<S::Real, SciOxideError> {
    match super::arch::maximum(values, dispatch)? {
        Some(maximum) => Ok(maximum),
        None => standard::finite_slice_maximum(values),
    }
}

/// Blue's three-range Euclidean norm, independent of dispatch.
///
/// # Errors
/// Returns `NumericalInstability(Overflow)` when the norm is not representable.
pub(crate) fn norm<S: NumericScalar>(values: &[S]) -> Result<S::Real, SciOxideError> {
    standard::norm_workspace(values.iter().copied())
}

/// [`norm`] at a cooperative stop point of a long recurrence.
///
/// # Errors
/// Returns the execution stop before measuring, or the [`norm`] error.
pub(crate) fn checked_norm<S: NumericScalar>(
    values: &[S],
    execution: ExecutionContext<'_>,
) -> Result<S::Real, SciOxideError> {
    execution.check()?;
    norm(values)
}

/// Euclidean norm from one dispatched pass of squared moments. Ordinary ranges
/// keep the fused sum of squares; others fall back to [`norm`].
pub(crate) fn fused_norm<S: NumericScalar>(
    values: &[S],
    dispatch: KernelDispatch,
) -> Result<S::Real, SciOxideError> {
    super::recurrence::products(values, values, dispatch)?
        .left
        .norm(values)
}

/// Extended-range inner product `left * conj(right)`. Ordinary products keep
/// the dispatched candidate; others accumulate power-of-two scaled components.
pub(crate) fn scaled_dot<S: NumericScalar>(
    left: &[S],
    right: &[S],
    dispatch: KernelDispatch,
) -> Result<ScaledValue<S>, SciOxideError> {
    super::recurrence::products(left, right, dispatch)?.scaled(left, right)
}

/// [`scaled_dot`] rounded to the working scalar.
///
/// # Errors
/// Returns `NumericalInstability` when the rounded product is not representable.
pub(crate) fn dot<S: NumericScalar>(
    left: &[S],
    right: &[S],
    dispatch: KernelDispatch,
) -> Result<S, SciOxideError> {
    scaled_dot(left, right, dispatch)?.round()
}

/// Diagnostics measured while an [`update`] writes its output.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Updated<R: RealScalar> {
    /// Whether any output element changed; recurrences use it to detect stagnation.
    pub(crate) changed: bool,
    /// Bound and squared-norm candidate of the new output.
    pub(crate) moments: Moments<R>,
}

/// `output = alpha * input + beta * output` with one range proof for every
/// element.
///
/// # Errors
/// Returns `DimensionMismatch` for unequal lengths and `NumericalInstability`
/// for a non-finite coefficient or an unrepresentable result.
pub(crate) fn update<S: NumericScalar>(
    output: &mut [S],
    alpha: S,
    input: &[S],
    beta: S,
    dispatch: KernelDispatch,
) -> Result<Updated<S::Real>, SciOxideError> {
    let bounds = (maximum(input, dispatch)?, maximum(output, dispatch)?);
    let (changed, moments) =
        super::recurrence::update(output, alpha, input, beta, bounds, dispatch)?;
    Ok(Updated { changed, moments })
}

/// `output += alpha * input`. A zero coefficient leaves the output untouched
/// without measuring it, as restarted couplings are mostly structural zeros.
///
/// # Errors
/// As [`update`].
pub(crate) fn axpy<S: NumericScalar>(
    output: &mut [S],
    alpha: S,
    input: &[S],
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    if alpha == S::ZERO {
        return Ok(());
    }
    update(output, alpha, input, S::ONE, dispatch).map(drop)
}

/// Divide by a positive norm, falling back to checked division for tiny norms.
///
/// # Errors
/// Returns `NumericalInstability(Breakdown)` for a zero norm or an
/// unrepresentable quotient.
pub(crate) fn normalize<S: NumericScalar>(
    values: &mut [S],
    norm: S::Real,
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    let maximum = maximum(values, dispatch)?;
    super::recurrence::normalize(values, norm, maximum, dispatch)
}

/// Multiply by `factor`, rescaling binary frames whose products leave range.
///
/// # Errors
/// Returns `NumericalInstability` for a non-finite factor or result.
pub(crate) fn scale<S: NumericScalar>(values: &mut [S], factor: S) -> Result<(), SciOxideError> {
    crate::checked::finite_candidate(factor)?;
    let affine = standard::Affine::new(
        factor,
        standard::slice_maximum(values),
        S::ZERO,
        S::Real::ZERO,
    )?;
    values.iter_mut().try_for_each(|value| {
        let candidate = affine.apply(*value, S::ZERO);
        *value = match affine.publication {
            standard::Publication::Direct => candidate,
            standard::Publication::Checked => crate::checked::finite_candidate(candidate)?,
        };
        Ok(())
    })
}

/// Bit-preserving copy between equal-length private vectors.
///
/// # Errors
/// Returns `DimensionMismatch` (expected `input.len()`) for unequal lengths.
pub(crate) fn copy<S: NumericScalar>(output: &mut [S], input: &[S]) -> Result<(), SciOxideError> {
    if output.len() != input.len() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(input.len()),
            Dimensions::one(output.len()),
        ));
    }
    output.copy_from_slice(input);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use scioxide_core::backend::KernelBackend;
    use scioxide_core::{CancellationToken, Complex, SciOxideErrorView};

    fn backends() -> impl Iterator<Item = KernelDispatch> {
        let processor = crate::kernel::tests::test_processor();
        KernelBackend::available(processor)
            .map(move |backend| KernelDispatch::for_backend(backend, processor).unwrap())
    }

    #[test]
    fn copy_preserves_bits_and_rejects_unequal_lengths() {
        let input = [
            f64::from_bits(0x7ff8_0000_dead_beef),
            -0.0,
            f64::MIN_POSITIVE,
        ];
        let mut output = [1.0; 3];
        copy(&mut output, &input).unwrap();
        assert_eq!(output.map(f64::to_bits), input.map(f64::to_bits));
        let error = copy(&mut [0.0; 2], &input).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(output.map(f64::to_bits), input.map(f64::to_bits));
    }

    /// The dispatched update must equal the portable range-proved loop bit
    /// for bit, so replacing scalar action updates changes no recurrence.
    fn axpy_matches_portable<S: NumericScalar>(scalar: impl Fn(usize) -> S) {
        let alpha = scalar(3);
        for dispatch in backends() {
            for n in [0, 1, 3, 4, 7, 8, 15, 16, 17, 33, 65] {
                let x: Vec<_> = (0..n).map(&scalar).collect();
                let y: Vec<_> = (5..n + 5).map(&scalar).collect();
                let affine = standard::Affine::new(
                    alpha,
                    standard::slice_maximum(&x),
                    S::ONE,
                    standard::slice_maximum(&y),
                )
                .unwrap();
                let expected: Vec<_> = x
                    .iter()
                    .zip(&y)
                    .map(|(&x, &y)| affine.apply(x, y))
                    .collect();
                let mut actual = y.clone();
                axpy(&mut actual, alpha, &x, dispatch).unwrap();
                assert!(actual.iter().zip(&expected).all(|(a, e)| a == e), "n={n}");
                let mut unchanged = y.clone();
                axpy(&mut unchanged, S::ZERO, &x, dispatch).unwrap();
                assert!(unchanged.iter().zip(&y).all(|(a, e)| a == e));
            }
        }
    }

    #[test]
    fn axpy_is_bit_identical_to_the_portable_affine_loop() {
        let real = |i: usize| (i % 17) as f64 / 8. - 1.;
        axpy_matches_portable::<f64>(real);
        axpy_matches_portable::<f32>(|i| real(i) as f32);
        axpy_matches_portable::<Complex<f64>>(|i| Complex::new(real(i), real(i + 7) / 3.));
    }

    #[test]
    fn extreme_updates_rescale_or_report_overflow() {
        for dispatch in backends() {
            let mut out = [-f64::MAX];
            axpy(&mut out, 2., &[f64::MAX], dispatch).unwrap();
            assert_eq!(out, [f64::MAX]);
            assert!(matches!(
                axpy(&mut out, 2., &[f64::MAX], dispatch)
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::NumericalInstability { .. }
            ));
            assert!(axpy(&mut out, 1., &[1., 2.], dispatch).is_err());
            let mut values = [f64::MAX, -f64::MAX / 2.];
            scale(&mut values, 0.5).unwrap();
            assert_eq!(values, [f64::MAX / 2., -f64::MAX / 4.]);
            assert!(scale(&mut values, 4.).is_err());
            assert!(scale(&mut values, f64::NAN).is_err());
        }
    }

    #[test]
    fn norms_agree_and_checked_norm_stops_first() {
        let values = [3e200_f64, -4e200, 0.];
        let close = |length: f64| (length / 5e200 - 1.).abs() <= 4. * f64::EPSILON;
        for dispatch in backends() {
            assert!(close(norm(&values).unwrap()));
            assert!(close(fused_norm(&values, dispatch).unwrap()));
            assert_eq!(maximum(&values, dispatch).unwrap(), 4e200);
        }
        assert!(close(
            checked_norm(&values, ExecutionContext::sequential()).unwrap()
        ));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(matches!(
            checked_norm(
                &values,
                ExecutionContext::sequential().with_cancellation(&cancellation)
            )
            .unwrap_err()
            .view(),
            SciOxideErrorView::ExecutionStopped { .. }
        ));
    }

    #[test]
    fn dots_round_extended_products() {
        for dispatch in backends() {
            let z = |re, im| Complex::new(re, im);
            assert_eq!(
                dot(&[z(1., 2.)], &[z(3., -1.)], dispatch).unwrap(),
                z(1., 7.)
            );
            let huge = [f64::MAX, f64::MAX];
            assert_eq!(dot(&huge, &[2., -2.], dispatch).unwrap(), 0.);
            assert!(dot(&huge, &[2., 2.], dispatch).is_err());
            let mut values = [3e-310_f64, 4e-310];
            normalize(&mut values, 5e-310, dispatch).unwrap();
            assert!((values[0] - 0.6).abs() < 1e-12 && (values[1] - 0.8).abs() < 1e-12);
            assert!(normalize(&mut values, 0., dispatch).is_err());
        }
    }
}
