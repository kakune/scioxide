//! Shared scaled complex report construction for composed special functions.
//!
//! A kernel estimate `y·exp(scale)` is evaluated eagerly to its ordinary
//! value; its scaled, logarithmic and derivative forms are formed only when
//! the detailed result is returned.
use crate::ValueWithError;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::is_real;
pub(crate) use crate::support::narrow;
use crate::support::{C, ZERO};

use crate::result::{Evaluation, Forms, Publish};
use crate::{ScalarRepresentation, SpecialFunctionTermination};
use scioxide_core::AlgebraicScalar;
use scioxide_core::Scaled;
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, EstimateReliability, EvaluationCounts, LogValue,
    NumericScalar, RealScalar, ScaledValue, SciOxideError,
};

/// Evaluation of a scaled estimate.
pub(crate) type ScaledEvaluation<T> = Evaluation<T, ScaledForm>;

/// Report a scaled estimate charged to `work`, under the stable algorithm `family`.
pub(crate) fn report<T: NumericScalar>(
    estimate: Result<Estimate, SciOxideError>,
    derivative: bool,
    work: &Work<'_, '_, '_>,
    family: AlgorithmComponent,
) -> Result<ScaledEvaluation<T>, SciOxideError> {
    report_with_status(estimate, derivative, family, work.used, work.exhausted())
}

/// Report an estimate whose forms are never returned. A finite unscaled
/// estimate is its own ordinary value, so its evaluation carries no scaled
/// form. A real destination with a nonzero imaginary component still takes
/// [`report`]'s error-budget check.
pub(crate) fn report_plain<T: NumericScalar>(
    estimate: Result<Estimate, SciOxideError>,
    work: &Work<'_, '_, '_>,
    family: AlgorithmComponent,
) -> Result<ScaledEvaluation<T>, SciOxideError> {
    match estimate {
        Ok(e)
            if !work.exhausted()
                && e.scale.v == ZERO
                && (!is_real::<T>() || e.y.v.im == 0.0)
                && e.y.v.is_finite()
                && e.error.is_finite() =>
        {
            let (value, representation) =
                if T::Real::try_from_f64_lossy(e.y.v.re.abs().max(e.y.v.im.abs())).is_ok() {
                    (Some(narrow::<T>(e.y.v)?), ScalarRepresentation::Finite)
                } else {
                    (None, ScalarRepresentation::Overflow)
                };
            Ok(Evaluation::from_parts(
                value,
                representation,
                SpecialFunctionTermination::Converged,
                EvaluationCounts::from_iterations(work.used),
                AlgorithmId::from_components(family, e.method, 1),
                None,
            ))
        }
        estimate => report(estimate, false, work, family),
    }
}

/// The scaled estimate and the intermediate values its forms reuse. Error
/// and logarithmic quantities are derived only by the forms needing them.
#[derive(Clone, Copy)]
pub(crate) struct ScaledForm {
    e: Estimate,
    derivative: bool,
    zero: bool,
    base: ScaledValue<C>,
    encoded: Option<ScaledValue<C>>,
    ordinary: Option<C>,
}

/// Logarithm and error budget of a scaled estimate in destination `T`.
struct Budget {
    logarithm: C,
    absolute_log: f64,
    mantissa_relative: f64,
    scale_error: f64,
    relative: f64,
}

impl ScaledForm {
    /// Natural logarithm of the estimate; zero for an exact zero.
    fn logarithm(&self) -> C {
        if self.zero {
            ZERO
        } else {
            self.e.y.v.ln() + self.e.scale.v
        }
    }

    /// Error budget.
    fn budget<T: NumericScalar>(&self) -> Result<Budget, SciOxideError> {
        let (e, zero) = (self.e, self.zero);
        let logarithm = self.logarithm();
        let absolute_log = if e.error == 0.0 {
            f64::NEG_INFINITY
        } else {
            e.error.ln() + e.scale.v.re
        };
        // The mantissa error is relative; the rounded scale is an absolute
        // log-unit error. Their sum is the relative error of the scalar value.
        let eps = T::Real::EPSILON.try_to_f64_exact()?;
        let (mantissa_relative, scale_error) = if zero {
            (0.0, 0.0)
        } else {
            (
                Scaled::from_value(e.error)?
                    .div(self.base.magnitude()?)?
                    .round()?
                    + 8.0 * f64::EPSILON
                    + if eps > f64::EPSILON { 2.0 * eps } else { 0.0 },
                (e.scale.v.re * (8.0 * f64::EPSILON)).hypot(e.scale.v.im * (8.0 * f64::EPSILON)),
            )
        };
        let relative = mantissa_relative + scale_error;
        let absolute_log = if zero {
            absolute_log
        } else {
            absolute_log.max(logarithm.re + relative.ln())
        };
        Ok(Budget {
            logarithm,
            absolute_log,
            mantissa_relative,
            scale_error,
            relative,
        })
    }
}

fn exponent_out_of_range(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::ExponentOutOfRange,
            ..
        }
    )
}

/// Apply the rotation `exp(i·scale.im)`.
fn rotate(value: ScaledValue<C>, scale: C) -> Result<ScaledValue<C>, SciOxideError> {
    if scale.im == 0.0 {
        Ok(value)
    } else {
        value.mul(ScaledValue::from_value(C::new(
            scale.im.cos(),
            scale.im.sin(),
        ))?)
    }
}

/// Apply `exp(scale)`; `None` once the exponent leaves the scaled range.
fn expand(value: ScaledValue<C>, scale: C) -> Result<Option<ScaledValue<C>>, SciOxideError> {
    if value.is_zero() || scale == ZERO {
        return Ok(Some(value));
    }
    let factor = match Scaled::from_log(1.0, scale.re) {
        Ok(value) => value,
        Err(error) if exponent_out_of_range(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    match rotate(value, scale)?.mul_real(factor) {
        Ok(value) => Ok(Some(value)),
        Err(error) if exponent_out_of_range(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Round a scaled value to an ordinary one; `None` on overflow.
fn project(value: Option<ScaledValue<C>>) -> Result<Option<C>, SciOxideError> {
    match value.map(|value| value.round()).transpose() {
        Ok(value) => Ok(value),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Evaluate a scaled estimate.
///
/// `family` is the stable algorithm family; it is independent.
pub(crate) fn report_with_status<T: NumericScalar>(
    estimate: Result<Estimate, SciOxideError>,
    derivative: bool,
    family: AlgorithmComponent,
    used: usize,
    exhausted: bool,
) -> Result<ScaledEvaluation<T>, SciOxideError> {
    if exhausted {
        return Err(crate::result::exhausted_work(used));
    }
    let e = estimate?;
    if !e.y.v.is_finite() || !e.error.is_finite() || !e.scale.v.is_finite() {
        return Err(crate::support::instability(
            "scaled special-function representation lost finite accuracy",
        ));
    }
    let zero = e.y.v == ZERO;
    let base = ScaledValue::from_value(e.y.v)?;
    let encoded = expand(base, e.scale.v)?;
    // An unscaled finite estimate projects back to itself exactly. A zero
    // significand, which the scaled encoding stores unsigned, and a scale
    // below the extended exponent range both give a zero signed like the
    // rotated estimate. Without a rotation the estimate keeps its own zero
    // signs, which a complex product by one would cancel.
    let ordinary = if e.scale.v == ZERO {
        Some(e.y.v)
    } else if zero || encoded.is_none() && e.scale.v.re < 0.0 {
        let direction = if e.scale.v.im == 0.0 {
            e.y.v
        } else {
            e.y.v * C::new(e.scale.v.im.cos(), e.scale.v.im.sin())
        };
        Some(C::new(
            0.0_f64.copysign(direction.re),
            0.0_f64.copysign(direction.im),
        ))
    } else {
        project(encoded)?
    };
    let form = ScaledForm {
        e,
        derivative,
        zero,
        base,
        encoded,
        ordinary,
    };
    // A real destination rejects an imaginary part beyond the error budget;
    // an exactly real value needs no budget.
    let imaginary = ordinary.filter(|v| v.im != 0.0);
    if is_real::<T>()
        && let Some(v) = imaginary
        && v.im.abs() > form.budget::<T>()?.absolute_log.exp() + 1e-12 * v.re.abs()
    {
        return Err(crate::support::instability(
            "computed real projection has a significant imaginary component",
        ));
    }
    let (value, representation) = match ordinary
        .filter(|v| T::Real::try_from_f64_lossy(v.re.abs().max(v.im.abs())).is_ok())
    {
        Some(ordinary) => (Some(narrow::<T>(ordinary)?), ScalarRepresentation::Finite),
        None => (None, ScalarRepresentation::Overflow),
    };
    Ok(Evaluation::from_parts(
        value,
        representation,
        SpecialFunctionTermination::Converged,
        EvaluationCounts::from_iterations(used),
        AlgorithmId::from_components(family, e.method, 1),
        Some(form),
    ))
}

impl ScaledForm {
    fn projected_derivative<T: NumericScalar>(&self) -> Result<Option<(T, C)>, SciOxideError> {
        let e = self.e;
        if !self.derivative
            || !e.y.d.is_finite()
            || !e.scale.d.is_finite()
            || e.derivative_error.is_some_and(|error| !error.is_finite())
        {
            return Ok(None);
        }
        let numerator = ScaledValue::from_value(e.y.d)?
            .add(ScaledValue::from_value(e.y.v)?.mul(ScaledValue::from_value(e.scale.d)?)?)?;
        let numerator = ScaledValue::from_components(numerator.real(), numerator.imaginary());
        let Some(v) = project(expand(numerator, e.scale.v)?)? else {
            return Ok(None);
        };
        if e.derivative_error.is_some_and(|error| error < 0.0) {
            return Err(crate::support::instability(
                "scaled special-function representation lost finite accuracy",
            ));
        }
        if T::Real::try_from_f64_lossy(v.re.abs().max(v.im.abs())).is_err()
            || (v == ZERO && !numerator.is_zero())
        {
            return Ok(None);
        }
        let value = narrow::<T>(v)?;
        Ok((value.abs() != T::Real::ZERO || v == ZERO).then_some((value, v)))
    }
}

impl<T: NumericScalar> Publish<T> for ScaledForm {
    fn publish(self, value: Option<T>) -> Result<Forms<T>, SciOxideError> {
        let Budget {
            logarithm,
            absolute_log,
            mantissa_relative,
            scale_error,
            relative,
        } = self.budget::<T>()?;
        let Self {
            e,
            zero,
            encoded,
            ordinary,
            ..
        } = self;
        let eps = T::Real::EPSILON.try_to_f64_exact()?;
        let max = T::Real::MAX.try_to_f64_exact()?;
        let reliability = e.reliability();
        let bounded = crate::result::outward::<T::Real>;
        let value = match (value, ordinary) {
            (Some(value), Some(ordinary)) => {
                let quantum = T::Real::ZERO.next_after(T::Real::ONE).into_f64();
                let rounding = if zero && e.error == 0.0 {
                    0.0
                } else {
                    eps * ordinary.re.abs().max(ordinary.im.abs()) * core::f64::consts::SQRT_2
                        + quantum
                };
                Some(ValueWithError::new(
                    value,
                    bounded(absolute_log.exp() + rounding)?,
                    None,
                    reliability,
                )?)
            }
            _ => None,
        };
        // Narrow normalized components independently; narrowing an ordinary phase
        // first would erase directions that become representable after scaling.
        let scaled = encoded
            .map(|value| {
                let scaled = ScaledValue::from_component_parts(
                    narrow::<T>(value.component_significands())?,
                    value.component_exponents(),
                )?;
                let error = if zero {
                    bounded(absolute_log.exp())?
                } else {
                    bounded(relative * scaled.scaled_magnitude().try_to_f64_exact()?)?
                };
                ValueWithError::new(scaled, error, None, reliability)
            })
            .transpose()?;
        // A log magnitude beyond the destination range has no logarithmic form.
        let log_value = if zero {
            Some(LogValue::from_value(narrow::<T>(e.y.v)?)?)
        } else if logarithm.re.abs() > max {
            None
        } else {
            let phase = rotate(LogValue::from_value(e.y.v)?.scaled_phase(), e.scale.v)?;
            if is_real::<T>() {
                Some(LogValue::from_log(
                    T::from_real(T::Real::try_from_f64_lossy(
                        phase.real().significand().signum(),
                    )?),
                    T::Real::try_from_f64_lossy(logarithm.re)?,
                )?)
            } else {
                Some(LogValue::from_scaled_log(
                    ScaledValue::from_component_parts(
                        narrow::<T>(phase.component_significands())?,
                        phase.component_exponents(),
                    )?,
                    T::Real::try_from_f64_lossy(logarithm.re)?,
                )?)
            }
        };
        // Derivative availability is independent of the ordinary value.
        let d = self
            .projected_derivative()?
            .map(|(projected, v)| {
                let derivative_error = if let Some(error) = e.derivative_error {
                    let absolute = if error == 0.0 {
                        0.0
                    } else {
                        (error.ln() + e.scale.v.re).exp()
                    };
                    absolute
                        + v.norm() * (16.0 * eps + 8.0 * f64::EPSILON * (1.0 + e.scale.v.norm()))
                } else {
                    absolute_log.exp() + v.norm() * (relative + 16.0 * eps)
                };
                ValueWithError::new(projected, bounded(derivative_error)?, None, reliability)
            })
            .transpose()?;
        Ok(Forms {
            value,
            scaled,
            // Natural-log units; no log estimate exists once the interval may reach zero.
            logarithmic: log_value
                .filter(|_| !(zero && e.error > 0.0) && mantissa_relative < 0.5)
                .map(|log_value| {
                    let error = crate::result::log_error(mantissa_relative, logarithm.re)
                        .unwrap_or(f64::MAX)
                        + scale_error;
                    let error = if T::Real::MANTISSA_DIGITS < f64::MANTISSA_DIGITS {
                        let rounding = log_value
                            .log_magnitude()
                            .map_or(0.0, |magnitude| (magnitude.into_f64() - logarithm.re).abs());
                        error + rounding
                    } else {
                        error
                    };
                    ValueWithError::new(log_value, bounded(error)?, None, reliability)
                })
                .transpose()?,
            derivative: d,
        })
    }
}

/// A value below every representation, returned as its rounded zero with a
/// one-quantum bound and no scaled, logarithmic or derivative form.
#[derive(Clone, Copy)]
pub(crate) struct RoundedZero;

/// A scaled estimate, or a value below every representation.
pub(crate) type ZeroOrScaled = crate::result::Either<ScaledForm, RoundedZero>;

impl<T: NumericScalar> Publish<T> for RoundedZero {
    fn publish(self, value: Option<T>) -> Result<Forms<T>, SciOxideError> {
        Ok(Forms {
            value: value
                .map(|value| {
                    ValueWithError::new(
                        value,
                        T::Real::MIN_SUBNORMAL,
                        None,
                        EstimateReliability::ModelBased,
                    )
                })
                .transpose()?,
            ..Forms::NONE
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::{Jet, real};
    use scioxide_core::algorithm_component;

    #[test]
    fn rounded_component_limits_agree_in_plain_and_detailed_forms() {
        let maximum = f64::from(f32::MAX);
        let spacing = maximum - f64::from(f32::from_bits(f32::MAX.to_bits() - 1));
        let work = Work::new(12);
        for sign in [-1.0, 1.0] {
            for imaginary in [false, true] {
                let component = sign * (maximum + spacing * 0.25);
                let value = if imaginary {
                    C::new(0.0, component)
                } else {
                    C::new(component, 0.0)
                };
                let estimate = Estimate::new(
                    Jet { v: value, d: value },
                    0.0,
                    algorithm_component!("test"),
                );
                let plain = report_plain::<scioxide_core::Complex32>(
                    Ok(estimate),
                    &work,
                    algorithm_component!("test"),
                )
                .unwrap()
                .value()
                .unwrap();
                let detailed = report::<scioxide_core::Complex32>(
                    Ok(estimate),
                    true,
                    &work,
                    algorithm_component!("test"),
                )
                .unwrap()
                .publish()
                .unwrap();
                let expected = if imaginary {
                    scioxide_core::Complex32::new(0.0, sign as f32 * f32::MAX)
                } else {
                    scioxide_core::Complex32::new(sign as f32 * f32::MAX, 0.0)
                };
                assert_eq!(plain, expected);
                assert_eq!(detailed.try_value().unwrap(), expected);
                assert_eq!(detailed.try_derivative().unwrap(), expected);
                assert!(detailed.value().unwrap().absolute_error() > 0.0);
            }
        }
    }

    #[test]
    #[allow(clippy::panic)]
    fn plain_unscaled_complex_projection_preserves_component_boundaries() {
        fn check<S: RealScalar>()
        where
            scioxide_core::Complex<S>: NumericScalar<Real = S>,
        {
            let work = Work::new(12);
            for value in [
                C::new(-0.0, 0.0),
                C::new(0.0, -0.0),
                C::new(f64::MAX, f64::from_bits(1)),
                C::new(f64::from_bits(1), f64::MAX),
                C::new(0.0, 1e-300),
                C::new(1e-300, -1e-300),
                C::new(f64::from(f32::MAX), 1e-40),
                C::new(1.0, 2.0 * f64::from(f32::MAX)),
                C::new(f64::from(f32::from_bits(1)), -0.0),
            ] {
                let e = Estimate::new(Jet::constant(value), 0.0, algorithm_component!("test"));
                let plain = report_plain::<scioxide_core::Complex<S>>(
                    Ok(e),
                    &work,
                    algorithm_component!("test"),
                )
                .unwrap();
                let detailed = report::<scioxide_core::Complex<S>>(
                    Ok(e),
                    false,
                    &work,
                    algorithm_component!("test"),
                )
                .unwrap();
                assert_eq!(
                    plain.representation(),
                    detailed.representation(),
                    "{value:?}"
                );
                match (plain.value(), detailed.value()) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a.re.into_f64().to_bits(), b.re.into_f64().to_bits());
                        assert_eq!(a.im.into_f64().to_bits(), b.im.into_f64().to_bits());
                    }
                    (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind()),
                    results => panic!("{value:?}: {results:?}"),
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn report_conversion_retains_mixed_components_and_extended_phases() {
        let source = C::new(f64::MAX, f64::from_bits(1));
        let estimate = |value, scale| {
            Estimate::from(crate::support::JetData {
                y: Jet::constant(value),
                scale: Jet::constant(scale),
                error: 0.0,
                derivative_error: None,
                method: algorithm_component!("component_conversion"),
            })
        };
        let result = report_with_status::<C>(
            Ok(estimate(source, ZERO)),
            false,
            algorithm_component!("test"),
            0,
            false,
        )
        .unwrap()
        .publish()
        .unwrap();
        let projected = result.scaled().unwrap().value().try_value().unwrap();
        assert_eq!(projected.re.to_bits(), source.re.to_bits());
        assert_eq!(projected.im.to_bits(), source.im.to_bits());
        assert!(
            !result
                .logarithmic()
                .unwrap()
                .value()
                .scaled_phase()
                .imaginary()
                .is_zero()
        );

        let result = report_with_status::<scioxide_core::Complex32>(
            Ok(estimate(C::new(f64::from_bits(1), 1.0), real(700.0))),
            false,
            algorithm_component!("test"),
            0,
            false,
        )
        .unwrap()
        .publish()
        .unwrap();
        let real = result.scaled().unwrap().value().real().round().unwrap();
        // 1009*ln(2) < 700 < 1010*ln(2), so exp(700)*2^-1074
        // lies strictly between 2^-65 and 2^-64 and is a normal binary32 value.
        assert!(real > 2.0_f32.powi(-65) && real < 2.0_f32.powi(-64));
        assert!(
            !result
                .logarithmic()
                .unwrap()
                .value()
                .scaled_phase()
                .real()
                .is_zero()
        );
    }

    #[test]
    fn an_uncertain_zero_keeps_its_absolute_estimate() {
        let e = Estimate::from(crate::support::JetData {
            y: Jet::constant(ZERO),
            scale: Jet::constant(ZERO),
            error: 1e-6,
            derivative_error: None,
            method: algorithm_component!("test"),
        });
        let r = report_with_status::<C>(Ok(e), false, algorithm_component!("test"), 1, false)
            .unwrap()
            .publish()
            .unwrap();
        assert_eq!(r.termination(), SpecialFunctionTermination::Converged);
        assert!(r.value().unwrap().absolute_error() >= 1e-6);
        assert!(r.logarithmic().is_none());
    }

    #[test]
    fn scaled_negative_axis_preserves_a_tiny_imaginary_component() {
        for imaginary in [0.0_f64, 1e-100, -1e-100] {
            let e = Estimate::from(crate::support::JetData {
                y: Jet::constant(C::new(-1.0, imaginary)),
                scale: Jet::constant(real(4.0_f64.ln())),
                error: f64::EPSILON,
                derivative_error: None,
                method: algorithm_component!("test"),
            });
            let r = report_with_status::<C>(Ok(e), false, algorithm_component!("test"), 1, false)
                .unwrap()
                .publish()
                .unwrap();
            let value = r.try_value().unwrap();
            assert!((value.re + 4.0).abs() < 2e-15);
            assert!((value.im - 4.0 * imaginary).abs() <= 4.0 * f64::EPSILON * imaginary.abs());
            let significand = r
                .scaled()
                .unwrap()
                .value()
                .try_projective()
                .unwrap()
                .significand();
            assert!(
                (significand.im / significand.re + imaginary).abs()
                    <= f64::EPSILON * imaginary.abs()
            );
        }
    }
}
