//! Horner's rule for monomial coefficients: native arithmetic after a range
//! check, restarted in extended range when the range is lost.

use crate::admission::{
    Arithmetic, ArithmeticError, ArithmeticResult, CheckpointTier, CoefficientView, Extended,
    Lanes, Native, WideHorner, candidate, extended, native_coefficients, paired_point, require,
};
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, Scaled, ScaledValue, SciOxideError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct HornerState<V, B> {
    pub(crate) value: V,
    pub(crate) derivative: V,
    pub(crate) second_derivative: V,
    pub(crate) bound: B,
}

pub(crate) type ScaledHorner<R> = HornerState<ScaledValue<Complex<R>>, Scaled<R>>;

#[derive(Clone, Copy, Debug)]
pub(crate) enum HornerEvaluation<C: NumericScalar> {
    Native(HornerState<C, C::Real>),
    Extended(HornerState<ScaledValue<C>, Scaled<C::Real>>),
}

impl<C: NumericScalar> HornerEvaluation<C> {
    #[inline]
    pub(crate) fn value(self) -> Result<C, SciOxideError> {
        match self {
            Self::Native(state) => Ok(state.value),
            Self::Extended(state) => state.value.round(),
        }
    }

    // Outlining this projection can copy the large extended variant even on
    // the native path; keep the variant selection in the caller.
    #[inline(always)]
    pub(crate) fn rounded(self) -> Result<(C, C, C::Real), SciOxideError> {
        match self {
            Self::Native(state) => Ok((state.value, state.derivative, state.bound)),
            Self::Extended(state) => Ok((
                state.value.round()?,
                state.derivative.round()?,
                state.bound.round()?,
            )),
        }
    }

    pub(crate) fn into_complex(self) -> HornerEvaluation<Complex<C::Real>> {
        let complex = |v: C| Complex::new(v.real_component(), v.imaginary_component());
        let scaled = |v: ScaledValue<C>| ScaledValue::from_components(v.real(), v.imaginary());
        match self {
            Self::Native(state) => HornerEvaluation::Native(HornerState {
                value: complex(state.value),
                derivative: complex(state.derivative),
                second_derivative: complex(state.second_derivative),
                bound: state.bound,
            }),
            Self::Extended(state) => HornerEvaluation::Extended(HornerState {
                value: scaled(state.value),
                derivative: scaled(state.derivative),
                second_derivative: scaled(state.second_derivative),
                bound: state.bound,
            }),
        }
    }

    pub(crate) fn into_scaled(
        self,
    ) -> Result<HornerState<ScaledValue<C>, Scaled<C::Real>>, SciOxideError> {
        match self {
            Self::Native(state) => Ok(HornerState {
                value: ScaledValue::from_value(state.value)?,
                derivative: ScaledValue::from_value(state.derivative)?,
                second_derivative: ScaledValue::from_value(state.second_derivative)?,
                bound: Scaled::from_value(state.bound)?,
            }),
            Self::Extended(state) => Ok(state),
        }
    }
}

/// Callers validate raw input once; no coefficient collection or allocation.
#[inline]
pub(crate) fn monomial_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: impl DoubleEndedIterator<Item = C> + Clone,
    point: C,
    mut checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    let admitted = native_coefficients(coefficients.clone());
    evaluate_horner::<C, ORDER, BOUND>(
        coefficients,
        point,
        admitted,
        CheckpointTier::CHECKED,
        &mut checkpoint,
    )
}

#[inline(always)]
pub(crate) fn admitted_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: CoefficientView<'_, C>,
    point: C,
    mut checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    // Even/odd decomposition evaluates P(x)=E(x^2)+x O(x^2) using two
    // independent lanes of this same Horner recurrence, including both native
    // coefficient envelopes. It halves the derivative dependency chain.
    if ORDER == 1 && BOUND && coefficients.horner_parity(point) {
        let x = point.real_component();
        let square = x * x;
        let pairs = coefficients
            .values()
            .chunks_exact(2)
            .map(|pair| Ok([pair[0].real_component(), pair[1].real_component()]));
        if let Some(state) = candidate(
            horner_arithmetic::<C::Real, Lanes<2>, 1, true, true, false>(
                pairs,
                square,
                &mut checkpoint,
            ),
        )? {
            let value = scioxide_core::fast_mul_add(x, state.value[1], state.value[0]);
            let derivative = scioxide_core::fast_mul_add(
                x + x,
                state.derivative[0],
                scioxide_core::fast_mul_add(square + square, state.derivative[1], state.value[1]),
            );
            let bound = scioxide_core::fast_mul_add(x.abs(), state.bound[1], state.bound[0]);
            return Ok(HornerEvaluation::Native(HornerState {
                value: C::from_real(value),
                derivative: C::from_real(derivative),
                second_derivative: C::ZERO,
                bound,
            }));
        }
    }
    evaluate_horner::<C, ORDER, BOUND>(
        coefficients.values().iter().copied(),
        point,
        coefficients.native(),
        coefficients.horner_tier(point),
        checkpoint,
    )
}

#[inline(always)]
fn evaluate_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: impl DoubleEndedIterator<Item = C> + Clone,
    point: C,
    admitted: bool,
    tier: CheckpointTier,
    mut checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    if tier.steps() != 0
        && let Some(state) = candidate(horner_kernel::<C, Native, ORDER, BOUND, true, false>(
            coefficients.clone(),
            point,
            &mut checkpoint,
        ))?
    {
        return Ok(HornerEvaluation::Native(state));
    }
    if tier.steps() != 0 {
        return framed_horner::<C, ORDER, BOUND>(coefficients, point, checkpoint);
    }
    checked_horner::<C, ORDER, BOUND>(coefficients, point, admitted, checkpoint)
}

#[cold]
#[inline(never)]
fn framed_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: impl DoubleEndedIterator<Item = C> + Clone,
    point: C,
    mut checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    if let Some(state) = candidate(horner_kernel::<C, Native, ORDER, BOUND, true, true>(
        coefficients.clone(),
        point,
        &mut checkpoint,
    ))? {
        return Ok(HornerEvaluation::Native(state));
    }
    if let Some(state) = candidate(horner_kernel::<
        C,
        Native<WideHorner>,
        ORDER,
        BOUND,
        true,
        true,
    >(coefficients.clone(), point, &mut checkpoint))?
    {
        return Ok(HornerEvaluation::Native(state));
    }
    extended_horner::<C, ORDER, BOUND>(coefficients, point, checkpoint)
}

// Keep the conservative candidate out of the prepared hot path. Both use the
// same specialized recurrence, without copying its large error/extended arms
// into every complete-report call.
#[inline(never)]
fn checked_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: impl DoubleEndedIterator<Item = C> + Clone,
    point: C,
    admitted: bool,
    mut checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    if admitted
        && let Some(state) = candidate(horner_kernel::<C, Native, ORDER, BOUND, false, false>(
            coefficients.clone(),
            point,
            &mut checkpoint,
        ))?
    {
        return Ok(HornerEvaluation::Native(state));
    }
    extended_horner::<C, ORDER, BOUND>(coefficients, point, checkpoint)
}

#[cold]
#[inline(never)]
fn extended_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: impl DoubleEndedIterator<Item = C>,
    point: C,
    checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    let state = extended(horner_kernel::<C, Extended, ORDER, BOUND, false, false>(
        coefficients,
        point,
        checkpoint,
    ))?;
    Ok(HornerEvaluation::Extended(state))
}

#[inline]
pub(crate) fn complex_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: &[C],
    point: Complex<C::Real>,
) -> Result<HornerEvaluation<Complex<C::Real>>, SciOxideError> {
    admitted_complex_horner::<C, ORDER, BOUND>(
        coefficients,
        point,
        native_coefficients(coefficients.iter().copied()),
    )
}

/// [`complex_horner`] with the caller's `native_coefficients` classification
/// of the same coefficients, so one scan serves validation and the range check.
#[inline]
pub(crate) fn admitted_complex_horner<C: NumericScalar, const ORDER: usize, const BOUND: bool>(
    coefficients: &[C],
    point: Complex<C::Real>,
    admitted: bool,
) -> Result<HornerEvaluation<Complex<C::Real>>, SciOxideError> {
    // Real components carry the same admission: imaginary parts are zero.
    if C::IS_REAL && point.im == C::Real::ZERO {
        return Ok(evaluate_horner::<C::Real, ORDER, BOUND>(
            coefficients.iter().map(|c| c.real_component()),
            point.re,
            admitted,
            CheckpointTier::CHECKED,
            || Ok(()),
        )?
        .into_complex());
    }
    evaluate_horner::<_, ORDER, BOUND>(
        coefficients
            .iter()
            .map(|c| Complex::new(c.real_component(), c.imaginary_component())),
        point,
        admitted,
        CheckpointTier::CHECKED,
        || Ok(()),
    )
}

type ArithmeticState<C, A> = HornerState<<A as Arithmetic<C>>::Value, <A as Arithmetic<C>>::Bound>;

#[inline(always)]
fn horner_kernel<
    C: NumericScalar,
    A: Arithmetic<C>,
    const ORDER: usize,
    const BOUND: bool,
    const BLOCKED: bool,
    const FRAMED: bool,
>(
    coefficients: impl DoubleEndedIterator<Item = C>,
    point: C,
    checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> ArithmeticResult<ArithmeticState<C, A>> {
    horner_arithmetic::<C, A, ORDER, BOUND, BLOCKED, FRAMED>(
        coefficients.map(A::scalar),
        point,
        checkpoint,
    )
}

#[inline(always)]
fn horner_arithmetic<
    C: NumericScalar,
    A: Arithmetic<C>,
    const ORDER: usize,
    const BOUND: bool,
    const BLOCKED: bool,
    const FRAMED: bool,
>(
    coefficients: impl DoubleEndedIterator<Item = ArithmeticResult<A::Value>>,
    point: C,
    mut checkpoint: impl FnMut() -> Result<(), SciOxideError>,
) -> ArithmeticResult<ArithmeticState<C, A>> {
    let x = A::scalar(point)?;
    if !BLOCKED && !A::safe(x) {
        return Err(ArithmeticError::Retry);
    }
    let radius = if BOUND {
        A::magnitude(x)?
    } else {
        A::bound_scalar(C::Real::ZERO)?
    };
    if BOUND && !BLOCKED && !A::safe_bound(radius) {
        return Err(ArithmeticError::Retry);
    }
    let zero = A::scalar(C::ZERO)?;
    let zero_bound = A::magnitude(zero)?;
    let empty = HornerState {
        value: zero,
        derivative: zero,
        second_derivative: zero,
        bound: zero_bound,
    };
    let paired = paired_point(point);
    let mut coefficients = coefficients.rev();
    let Some(first) = coefficients.next() else {
        return Ok(empty);
    };
    checkpoint()?;
    let first = first?;
    let low = if paired { coefficients.next() } else { None };
    // Peel the leading block: its derivatives are known without multiplying
    // zero states. Both arithmetic representations use the same initialization.
    let mut state = if let Some(low) = low {
        checkpoint()?;
        let low = low?;
        HornerState {
            value: A::mul_add(first, x, low)?,
            derivative: if ORDER >= 1 { first } else { zero },
            second_derivative: zero,
            bound: if BOUND {
                A::bound_mul_add(A::magnitude(first)?, radius, A::magnitude(low)?)?
            } else {
                zero_bound
            },
        }
    } else {
        HornerState {
            value: first,
            bound: if BOUND {
                A::magnitude(first)?
            } else {
                zero_bound
            },
            ..empty
        }
    };
    if !BLOCKED && !A::safe(state.value) {
        return Err(ArithmeticError::Retry);
    }
    let mut safe = BLOCKED || !BOUND || A::safe_bound(state.bound);
    let square = if paired { A::mul_add(x, x, zero)? } else { x };
    let twice_x = if paired && ORDER >= 1 {
        A::double(x)?
    } else {
        x
    };
    let four_x = if paired && ORDER >= 2 {
        A::double(twice_x)?
    } else {
        x
    };
    let radius_squared = if paired && BOUND {
        A::bound_mul_add(radius, radius, zero_bound)?
    } else {
        radius
    };
    let mut frame = A::frame();
    let mut remaining = 2 * A::HORNER_TIER.steps();
    while let Some(coefficient) = coefficients.next() {
        checkpoint()?;
        let coefficient = if FRAMED {
            A::frame_value(coefficient?, frame)?
        } else {
            coefficient?
        };
        let low = if paired { coefficients.next() } else { None };
        let paired_step = low.is_some();
        if let Some(low) = low {
            checkpoint()?;
            let low = if FRAMED {
                A::frame_value(low?, frame)?
            } else {
                low?
            };
            // Second-order Horner: p <- p*x² + (high*x + low).
            // Differentiating this same update gives d*x² + 2*v*x + high
            // and s*x² + 4*d*x + 2*v. Independent coefficient pairs shorten
            // the value's dependency chain without introducing another kernel.
            let value = A::mul_add(state.value, square, A::mul_add(coefficient, x, low)?)?;
            // Separate admission of the value and derivative keeps their
            // dependency chains independent, including after vectorization.
            if !BLOCKED && !A::safe(value) {
                return Err(ArithmeticError::Retry);
            }
            let derivative = if ORDER >= 1 {
                A::mul_add(
                    state.derivative,
                    square,
                    A::mul_add(state.value, twice_x, coefficient)?,
                )?
            } else {
                zero
            };
            if !BLOCKED && ORDER >= 1 && !A::safe(derivative) {
                return Err(ArithmeticError::Retry);
            }
            state = HornerState {
                second_derivative: if ORDER >= 2 {
                    A::mul_add(
                        state.second_derivative,
                        square,
                        A::mul_add(state.derivative, four_x, A::double(state.value)?)?,
                    )?
                } else {
                    zero
                },
                derivative,
                value,
                bound: if BOUND {
                    A::bound_mul_add(
                        state.bound,
                        radius_squared,
                        A::bound_mul_add(A::magnitude(coefficient)?, radius, A::magnitude(low)?)?,
                    )?
                } else {
                    zero_bound
                },
            };
            // Range loss is sticky even when subsequent coefficients shrink
            // these fields back into the admission interval.
            if !BLOCKED {
                safe &= A::safe(state.second_derivative) & A::safe_bound(state.bound);
            }
        } else {
            // The final unpaired coefficient, or the per-step range path.
            state = HornerState {
                second_derivative: if ORDER >= 2 {
                    A::mul_add(state.second_derivative, x, A::double(state.derivative)?)?
                } else {
                    zero
                },
                derivative: if ORDER >= 1 {
                    A::mul_add(state.derivative, x, state.value)?
                } else {
                    zero
                },
                value: A::mul_add(state.value, x, coefficient)?,
                bound: if BOUND {
                    A::bound_mul_add(state.bound, radius, A::magnitude(coefficient)?)?
                } else {
                    zero_bound
                },
            };
            if !BLOCKED {
                safe &= A::safe(state.value)
                    & A::safe(state.derivative)
                    & A::safe(state.second_derivative)
                    & A::safe_bound(state.bound);
            }
        };
        if BLOCKED {
            remaining -= if paired_step { 2 } else { 1 };
            if remaining == 0 {
                horner_checkpoint::<C, A, ORDER, FRAMED>(&mut state, &mut frame)?;
                remaining = 2 * A::HORNER_TIER.steps();
            }
        }
    }
    require(safe)?;
    if BLOCKED {
        horner_checkpoint::<C, A, ORDER, FRAMED>(&mut state, &mut frame)?;
    }
    if FRAMED {
        state.value = A::restore(state.value, frame)?;
        if ORDER >= 1 {
            state.derivative = A::restore(state.derivative, frame)?;
        }
        if ORDER >= 2 {
            state.second_derivative = A::restore(state.second_derivative, frame)?;
        }
        if BOUND {
            state.bound = A::restore_bound(state.bound, frame)?;
        }
    }
    Ok(state)
}

/// One checkpoint operation for native Horner states and envelopes.
#[inline(always)]
fn horner_checkpoint<C: NumericScalar, A: Arithmetic<C>, const ORDER: usize, const FRAMED: bool>(
    state: &mut ArithmeticState<C, A>,
    frame: &mut A::Frame,
) -> ArithmeticResult<()> {
    if FRAMED {
        let mut live = [state.value, state.derivative, state.second_derivative];
        A::reframe::<true>(
            &mut live[..=ORDER],
            &mut state.bound,
            frame,
            A::HORNER_TIER.narrow(),
        )?;
        state.value = live[0];
        state.derivative = live[1];
        state.second_derivative = live[2];
        Ok(())
    } else {
        require(
            A::checkpoint(state.value, true)
                & A::checkpoint(state.derivative, true)
                & A::checkpoint(state.second_derivative, true)
                & A::bound_checkpoint(state.bound, true),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::native_range;
    use scioxide_core::RealScalar;

    #[test]
    fn unused_second_derivative_does_not_restart_first_order_horner() {
        let coefficients = [1.0, 1.0, 2.0_f64.powi(509)];
        let first =
            monomial_horner::<_, 1, true>(coefficients.into_iter(), 0.0, || Ok(())).unwrap();
        assert!(matches!(first, HornerEvaluation::Native(_)));
        assert_eq!(first.rounded().unwrap(), (1.0, 1.0, 1.0));
        let second =
            monomial_horner::<_, 2, true>(coefficients.into_iter(), 0.0, || Ok(())).unwrap();
        assert!(matches!(second, HornerEvaluation::Extended(_)));
        assert_eq!(
            second
                .into_scaled()
                .unwrap()
                .second_derivative
                .round()
                .unwrap(),
            2.0_f64.powi(510)
        );
    }

    #[test]
    fn prepared_horner_rescales_large_representable_reports() {
        for (length, point) in [(63, 65536.0_f64), (72, 16384.0)] {
            let coefficients = alloc::vec![1.0_f64; length];
            let actual = admitted_horner::<_, 1, true>(
                CoefficientView::new(&coefficients),
                point,
                || Ok(()),
            )
            .unwrap();
            assert!(matches!(actual, HornerEvaluation::Native(_)));
            let expected = horner_kernel::<_, Extended, 1, true, false, false>(
                coefficients.iter().copied(),
                point,
                || Ok(()),
            )
            .unwrap();
            let (value, derivative, bound) = actual.rounded().unwrap();
            for (actual, expected) in [
                (value, expected.value.round().unwrap()),
                (derivative, expected.derivative.round().unwrap()),
                (bound, expected.bound.round().unwrap()),
            ] {
                assert!((actual / expected - 1.0).abs() <= 512.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn admitted_first_order_horner_matches_scaled_at_block_boundaries() {
        for length in [1, 2, 5, 6, 7, 8, 13, 16, 64, 257] {
            let coefficients: alloc::vec::Vec<f64> = (0..length)
                .map(|i| if i % 3 == 0 { -0.25 } else { 0.5 })
                .collect();
            for point in [-1.0, 0.0, 0.5, 1.0, 1.125] {
                let native = admitted_horner::<_, 1, true>(
                    CoefficientView::new(&coefficients),
                    point,
                    || Ok(()),
                )
                .unwrap();
                let expected = horner_kernel::<_, Extended, 1, true, false, false>(
                    coefficients.iter().copied(),
                    point,
                    || Ok(()),
                )
                .unwrap();
                let actual = native.into_scaled().unwrap();
                for (actual, expected) in [
                    (actual.value, expected.value),
                    (actual.derivative, expected.derivative),
                ] {
                    let (actual, expected) = (actual.round().unwrap(), expected.round().unwrap());
                    assert!(
                        (actual - expected).abs() <= 256.0 * f64::EPSILON * expected.abs().max(1.0)
                    );
                }
                let (actual, expected) = (
                    actual.bound.round().unwrap(),
                    expected.bound.round().unwrap(),
                );
                assert!((actual - expected).abs() <= 256.0 * f64::EPSILON * expected.max(1.0));
            }
        }
    }

    fn parity<R: RealScalar>() {
        let c = [R::ONE, -R::TWO, R::ONE, R::TWO];
        let x = R::HALF;
        let native = monomial_horner::<_, 2, true>(c.into_iter(), x, || Ok(())).unwrap();
        assert!(matches!(native, HornerEvaluation::Native(_)));
        let extended =
            horner_kernel::<R, Extended, 2, true, false, false>(c.into_iter(), x, || Ok(()))
                .unwrap();
        let native = native.into_scaled().unwrap();
        for (actual, expected) in [
            (native.value, extended.value),
            (native.derivative, extended.derivative),
            (native.second_derivative, extended.second_derivative),
        ] {
            assert!(
                (actual.round().unwrap() - expected.round().unwrap()).abs() <= R::EPSILON * R::TWO
            );
        }
        assert_eq!(native.value.round().unwrap(), R::HALF);
        assert_eq!(native.derivative.round().unwrap(), R::HALF);
        assert_eq!(
            native.second_derivative.round().unwrap(),
            R::TWO + R::TWO + R::TWO + R::TWO
        );
        assert_eq!(
            native.bound.round().unwrap(),
            extended.bound.round().unwrap()
        );
    }

    #[test]
    fn real_native_and_extended_match_independent_cubic() {
        parity::<f32>();
        parity::<f64>();
    }

    #[test]
    fn extreme_components_restart_and_keep_unrequested_fields_scaled() {
        for scale in [2.0_f64.powi(-1000), 2.0_f64.powi(1000)] {
            let c = [scale, -scale, scale];
            let evaluation = monomial_horner::<_, 2, true>(c.into_iter(), 1.0, || Ok(())).unwrap();
            assert!(matches!(evaluation, HornerEvaluation::Extended(_)));
            let (value, derivative, bound) = evaluation.rounded().unwrap();
            assert_eq!(value, scale);
            assert_eq!(derivative, scale);
            assert_eq!(bound, 3.0 * scale);
        }
        let large = f64::MAX * 0.75;
        let evaluation =
            monomial_horner::<_, 2, true>([-large, large].into_iter(), 1.5, || Ok(())).unwrap();
        assert!(matches!(evaluation, HornerEvaluation::Extended(_)));
        let value = evaluation.value().unwrap();
        assert!((value / large - 0.5).abs() <= f64::EPSILON);
        let evaluation =
            monomial_horner::<_, 2, true>([0.0, 0.0, f64::MAX].into_iter(), 0.0, || Ok(()))
                .unwrap();
        assert_eq!(evaluation.value().unwrap(), 0.0);
    }

    #[test]
    fn signed_zeros_match_extended_arithmetic() {
        for z in [0.0_f64, -0.0_f64] {
            let c = [z, -z, z];
            let native = monomial_horner::<_, 2, true>(c.into_iter(), z, || Ok(()))
                .unwrap()
                .into_scaled()
                .unwrap();
            let extended =
                horner_kernel::<f64, Extended, 2, true, false, false>(c.into_iter(), z, || Ok(()))
                    .unwrap();
            for (a, b) in [
                (native.value, extended.value),
                (native.derivative, extended.derivative),
                (native.second_derivative, extended.second_derivative),
            ] {
                assert_eq!(a.round().unwrap().to_bits(), b.round().unwrap().to_bits());
            }
            let c = c.map(|r| Complex::new(r, -r));
            let point = Complex::new(z, -z);
            let native = monomial_horner::<_, 2, true>(c.into_iter(), point, || Ok(()))
                .unwrap()
                .into_scaled()
                .unwrap();
            let extended = horner_kernel::<Complex<f64>, Extended, 2, true, false, false>(
                c.into_iter(),
                point,
                || Ok(()),
            )
            .unwrap();
            for (a, b) in [
                (native.value, extended.value),
                (native.derivative, extended.derivative),
                (native.second_derivative, extended.second_derivative),
            ] {
                let a = a.round().unwrap();
                let b = b.round().unwrap();
                assert_eq!(a.re.to_bits(), b.re.to_bits());
                assert_eq!(a.im.to_bits(), b.im.to_bits());
            }
        }
    }

    fn range_boundaries<R: RealScalar>() {
        for edge in [R::MIN_POSITIVE.sqrt(), R::MAX.sqrt() / (R::TWO + R::TWO)] {
            let evaluation =
                monomial_horner::<_, 2, true>([edge].into_iter(), R::ZERO, || Ok(())).unwrap();
            assert!(matches!(evaluation, HornerEvaluation::Native(_)));
        }
        for outside in [
            R::MIN_POSITIVE.sqrt().next_after(R::ZERO),
            (R::MAX.sqrt() / (R::TWO + R::TWO)).next_after(R::MAX),
        ] {
            let evaluation =
                monomial_horner::<_, 2, true>([outside].into_iter(), R::ZERO, || Ok(())).unwrap();
            assert!(matches!(evaluation, HornerEvaluation::Extended(_)));
            assert_eq!(evaluation.value().unwrap(), outside);
        }
    }

    #[test]
    fn admission_boundaries_and_block_point_limits() {
        range_boundaries::<f32>();
        range_boundaries::<f64>();
        for edge in [1.0_f64 / 65536.0, 65536.0] {
            assert!(paired_point::<f64>(edge));
        }
        for outside in [(1.0_f64 / 65536.0).next_down(), 65536.0_f64.next_up()] {
            assert!(!paired_point::<f64>(outside));
        }
        assert!(!paired_point::<f32>(1.0));
    }

    fn complex_parity<R: RealScalar>() {
        for length in [1, 2, 17, 65] {
            let coefficients: alloc::vec::Vec<_> = (0..length)
                .map(|i| {
                    let k = R::try_from_usize_exact(i + 1).unwrap();
                    Complex::new(
                        R::ONE / k,
                        if i % 2 == 0 {
                            R::HALF / k
                        } else {
                            -R::HALF / k
                        },
                    )
                })
                .collect();
            for point in [
                Complex::new(R::HALF, R::HALF),
                Complex::new(-R::HALF, R::ZERO),
            ] {
                let native =
                    monomial_horner::<_, 2, true>(coefficients.iter().copied(), point, || Ok(()))
                        .unwrap();
                assert!(matches!(native, HornerEvaluation::Native(_)));
                let native = native.into_scaled().unwrap();
                let extended = horner_kernel::<Complex<R>, Extended, 2, true, false, false>(
                    coefficients.iter().copied(),
                    point,
                    || Ok(()),
                )
                .unwrap();
                for (a, b) in [
                    (native.value, extended.value),
                    (native.derivative, extended.derivative),
                    (native.second_derivative, extended.second_derivative),
                ] {
                    let a = a.round().unwrap();
                    let b = b.round().unwrap();
                    assert!(
                        (a - b).abs()
                            <= R::EPSILON
                                * R::try_from_usize_exact(64).unwrap()
                                * b.abs().max(R::ONE)
                    );
                }
                let a = native.bound.round().unwrap();
                let b = extended.bound.round().unwrap();
                assert!((a - b).abs() <= R::EPSILON * R::try_from_usize_exact(64).unwrap() * b);
            }
        }
    }

    #[test]
    fn complex_native_and_extended_match_at_several_degrees() {
        complex_parity::<f32>();
        complex_parity::<f64>();
    }

    #[test]
    fn range_loss_is_not_cleared_when_later_terms_shrink_the_state() {
        let a = 2.0_f64.powi(509);
        let mut coefficients = [0.0; 132];
        for (i, c) in coefficients[64..].iter_mut().enumerate() {
            *c = if i % 2 == 0 { a } else { -a };
        }
        assert!(native_coefficients(coefficients.into_iter()));
        let mut polls = 0;
        let native = horner_kernel::<f64, Native, 2, true, false, false>(
            coefficients.into_iter(),
            0.5,
            || {
                polls += 1;
                Ok(())
            },
        );
        assert!(matches!(native, Err(ArithmeticError::Retry)));
        assert_eq!(polls, coefficients.len());
        let evaluation =
            monomial_horner::<_, 2, true>(coefficients.into_iter(), 0.5, || Ok(())).unwrap();
        assert!(matches!(evaluation, HornerEvaluation::Extended(_)));
        let value = evaluation.value().unwrap();
        assert!(native_range(value));
    }
}
