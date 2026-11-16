//! Reporting and work accounting for Lambert/polylog kernels.
use crate::ValueWithError;
use crate::result::{Evaluation, Forms, Publish};
use crate::support::TRANSCENDENTAL_ERRORS as ERRORS;
use crate::support::TranscendentalEstimate as Estimate;
use crate::support::Work;
use crate::{ScalarRepresentation, SpecialFunctionResult, SpecialFunctionTermination};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, Complex, EstimateReliability, EvaluationCounts, LogValue,
    NumericScalar, RealScalar, ScaledValue, SciOxideError,
};

pub(crate) type Detailed<S> = SpecialFunctionResult<Complex<S>>;
/// Real kernels return through the shared scalar forms, without complex
/// projection or derivative/error construction on the plain path.
pub(crate) type RealKernelEvaluation<S> = Evaluation<S, RealEstimateForm>;

#[derive(Clone, Copy)]
pub(crate) struct RealEstimateForm(Estimate<f64>);

/// Projection is representation-specific; validation, completion, counts and
/// algorithm assembly are shared by scalar and complex estimates.
struct Projection<V, F> {
    value: Option<V>,
    representation: ScalarRepresentation,
    form: F,
    exact_rounding: bool,
}

#[inline(always)]
fn assemble<V: NumericScalar<Real = f64>, T: NumericScalar, F: Publish<T>>(
    estimate: Result<Estimate<V>, SciOxideError>,
    work: &Work<'_, '_, '_>,
    project: impl FnOnce(Estimate<V>) -> Result<Projection<T, F>, SciOxideError>,
) -> Result<Evaluation<T, F>, SciOxideError> {
    const FAMILY: AlgorithmComponent = scioxide_core::algorithm_component!("lambert_polylog");
    if !work.complete && estimate.is_err() {
        return Err(crate::result::exhausted_work(work.used));
    }
    let e = estimate?;
    if !e.value.has_finite_components() || !e.error.is_finite() {
        return Err(ERRORS.unstable());
    }
    let p = project(e)?;
    let termination = if !work.complete {
        SpecialFunctionTermination::IterationLimit
    } else if e.error == 0.0 && p.exact_rounding {
        SpecialFunctionTermination::Exact
    } else {
        SpecialFunctionTermination::Converged
    };
    Ok(Evaluation::from_parts(
        p.value,
        p.representation,
        termination,
        EvaluationCounts::from_iterations(work.used),
        AlgorithmId::from_components(FAMILY, e.method, 1),
        Some(p.form),
    ))
}

#[inline(always)]
pub(crate) fn real_report<S: RealScalar>(
    estimate: Result<Estimate<f64>, SciOxideError>,
    work: &Work<'_, '_, '_>,
) -> Result<RealKernelEvaluation<S>, SciOxideError> {
    assemble(estimate, work, |e| {
        // The shared report already checked binary64 finiteness. Only a
        // narrower destination needs a second range decision.
        let value = S::try_from_f64_lossy(e.value).ok();
        Ok(Projection {
            value,
            representation: if value.is_some() {
                ScalarRepresentation::Finite
            } else {
                ScalarRepresentation::Overflow
            },
            exact_rounding: value.is_some_and(|v| v.into_f64() == e.value),
            form: RealEstimateForm(e),
        })
    })
}

impl<S: RealScalar> Publish<S> for RealEstimateForm {
    fn publish(self, value: Option<S>) -> Result<Forms<S>, SciOxideError> {
        use crate::result::{Bound, Standard};
        let e = self.0;
        if value.is_none() {
            let wide = Evaluation::finite(
                e.value,
                Bound::derived(e.error),
                e.derivative
                    .map(|d| (d, Bound::derived(e.derivative_error))),
                scioxide_core::algorithm_id!("lambert_polylog", "wide_projection", 1),
                0,
                SpecialFunctionTermination::Converged,
            )
            .narrow::<S>()?;
            return wide
                .form()
                .copied()
                .ok_or_else(|| ERRORS.unstable())?
                .publish(None);
        }
        let maximum = S::MAX.try_to_f64_exact()?;
        let error = |error: f64| {
            let narrowed = S::try_from_f64_lossy(error.min(maximum))?;
            // A positive binary64 bound may round away in a subnormal f32
            // result. Keep one destination quantum instead of claiming zero.
            Ok::<S, SciOxideError>(if error > 0.0 {
                narrowed.max(S::MIN_SUBNORMAL)
            } else {
                narrowed
            })
        };
        let rounding = value.map_or(Ok(0.0), |v| {
            v.try_to_f64_exact().map(|v| (v - e.value).abs())
        })?;
        let derivative = e
            .derivative
            .filter(|d| d.is_finite() && S::try_from_f64_lossy(*d).is_ok())
            .map(|d| {
                Ok::<_, SciOxideError>((
                    S::try_from_f64_lossy(d)?,
                    Bound::derived(error(
                        e.derivative_error + d.abs() * S::EPSILON.try_to_f64_exact()? * 64.0,
                    )?),
                ))
            })
            .transpose()?;
        let mut forms = Standard::Finite {
            bound: Bound::derived(error(e.error + rounding)?),
            derivative,
        }
        .publish(value)?;
        if let Some(logarithm) = e.logarithm {
            // Omega's logarithm remains available when its value rounds to zero.
            forms.logarithmic = Some(ValueWithError::new(
                LogValue::from_log(S::ONE, S::try_from_f64_lossy(logarithm)?)?,
                error(16.0 * f64::EPSILON * logarithm.abs())?,
                None,
                EstimateReliability::ModelBased,
            )?);
        }
        Ok(forms)
    }
}

/// Lambert/polylog forms: the `f64` estimate, narrowed when the result is formed.
#[derive(Clone, Copy)]
pub(crate) struct EstimateForm {
    e: Estimate,
    scaled: Option<crate::complex_support::ScaledForm>,
}

/// Evaluation of a complex Lambert/polylog function.
pub(crate) type TranscendentalEvaluation<S> = Evaluation<Complex<S>, EstimateForm>;

impl EstimateForm {
    /// The narrowed value, its rounding and its absolute error.
    fn narrowed<S: RealScalar>(&self) -> Result<(Complex<S>, f64, S), SciOxideError>
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let project = |v: f64| {
            if self.e.scale == 0.0 || v == 0.0 {
                v
            } else {
                (v.abs().ln() + self.e.scale).exp().copysign(v)
            }
        };
        let source = Complex::new(project(self.e.value.re), project(self.e.value.im));
        let value = Complex::new(
            S::try_from_f64_lossy(source.re)?,
            S::try_from_f64_lossy(source.im)?,
        );
        let rounding = if value.is_finite() {
            (value.re.try_to_f64_exact()? - source.re).abs()
                + (value.im.try_to_f64_exact()? - source.im).abs()
        } else {
            0.0
        };
        let error = S::try_from_f64_lossy(
            (project(self.e.error) + rounding).min(S::MAX.try_to_f64_exact()?),
        )?;
        Ok((value, rounding, error))
    }

    /// The derivative estimate, when finite and representable. A finite
    /// function value can have an unrepresentable derivative (notably W_-1 at
    /// tiny negative f32 inputs). Derivatives are optional.
    fn derivative<S: RealScalar>(
        &self,
    ) -> Result<Option<ValueWithError<Complex<S>, S>>, SciOxideError>
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let e = &self.e;
        e.derivative
            .filter(|d| d.is_finite())
            .map(|d| {
                let maximum = S::MAX.try_to_f64_exact()?;
                if S::try_from_f64_lossy(d.re).is_err() || S::try_from_f64_lossy(d.im).is_err() {
                    return Ok::<_, SciOxideError>(None);
                }
                let v = Complex::new(S::try_from_f64_lossy(d.re)?, S::try_from_f64_lossy(d.im)?);
                if !v.is_finite() {
                    return Ok::<_, SciOxideError>(None);
                }
                Ok(Some(ValueWithError::new(
                    v,
                    (S::try_from_f64_lossy(e.derivative_error.min(maximum))?
                        + v.norm() * S::EPSILON * S::try_from_f64_lossy(64.0)?)
                    .min(S::MAX),
                    None,
                    EstimateReliability::ModelBased,
                )?))
            })
            .transpose()
            .map(Option::flatten)
    }
}

#[inline(always)]
pub(crate) fn report<S: RealScalar>(
    estimate: Result<Estimate, SciOxideError>,
    work: &Work<'_, '_, '_>,
) -> Result<TranscendentalEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    assemble(estimate, work, |e| {
        let mut form = EstimateForm { e, scaled: None };
        if e.scale != 0.0 || e.value.re.abs().max(e.value.im.abs()) > S::MAX.into_f64() {
            let estimate = crate::support::JetEstimate::from(crate::support::JetData {
                y: crate::support::Jet::constant(e.value),
                scale: crate::support::Jet::constant(crate::support::real(e.scale)),
                error: e.error,
                derivative_error: None,
                method: e.method,
            });
            let scaled = crate::complex_support::report_with_status(
                Ok(estimate),
                false,
                scioxide_core::algorithm_component!("lambert_polylog"),
                work.used,
                false,
            )?;
            form.scaled = scaled.form().copied();
            return Ok(Projection {
                value: scaled.ordinary(),
                representation: scaled.representation(),
                form,
                exact_rounding: false,
            });
        }
        let (value, rounding, _) = form.narrowed::<S>()?;
        Ok(Projection {
            value: Some(value),
            representation: ScalarRepresentation::Finite,
            form,
            exact_rounding: rounding == 0.0,
        })
    })
}

impl<S: RealScalar> Publish<Complex<S>> for EstimateForm
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn publish(self, value: Option<Complex<S>>) -> Result<Forms<Complex<S>>, SciOxideError> {
        if let Some(scaled) = self.scaled {
            return scaled.publish(value);
        }
        let Some(value) = value else {
            return Ok(Forms {
                value: None,
                scaled: None,
                logarithmic: None,
                derivative: None,
            });
        };
        let (_, _, error) = self.narrowed::<S>()?;
        let relative = if value.norm() == S::ZERO {
            None
        } else {
            Some((error / value.norm()).min(S::MAX))
        };
        let reliability = EstimateReliability::ModelBased;
        let scaled = ScaledValue::from_value(value)?;
        let logarithmic = if let Some(logarithm) = self.e.logarithm {
            LogValue::from_log(
                Complex::new(
                    S::try_from_f64_lossy(logarithm.im.cos())?,
                    S::try_from_f64_lossy(logarithm.im.sin())?,
                ),
                S::try_from_f64_lossy(logarithm.re)?,
            )?
        } else {
            LogValue::from_value(value)?
        };
        // A kernel logarithm carries its own rounding; otherwise the log error
        // follows the value error and is omitted once that interval reaches zero.
        let log_error = match (self.e.logarithm, relative) {
            (Some(logarithm), _) => Some(S::try_from_f64_lossy(
                16.0 * f64::EPSILON * logarithm.norm(),
            )?),
            (None, Some(relative)) => crate::result::log_error(relative, value.norm().ln()),
            (None, None) => (error == S::ZERO).then_some(S::ZERO),
        };
        Ok(Forms {
            value: Some(ValueWithError::new(value, error, relative, reliability)?),
            scaled: scaled
                .scale_absolute_error(error)
                .ok()
                .map(|scaled_error| {
                    ValueWithError::new(scaled, scaled_error, relative, reliability)
                })
                .transpose()?,
            logarithmic: log_error
                .map(|log_error| ValueWithError::new(logarithmic, log_error, None, reliability))
                .transpose()?,
            derivative: self.derivative()?,
        })
    }
}

/// Evaluation of the real part of a Lambert/polylog function.
pub(crate) type RealEvaluation<S> = Evaluation<S, RealPart>;

/// The real part of a complex Lambert/polylog estimate.
#[derive(Clone, Copy)]
pub(crate) struct RealPart {
    inner: EstimateForm,
    take_real_part: bool,
}

/// Projects a complex evaluation to real coordinates. Unless `take_real_part`, an imaginary part beyond
/// the value's or derivative's error is a loss of significance.
pub(crate) fn project_real<S: RealScalar>(
    r: TranscendentalEvaluation<S>,
    take_real_part: bool,
) -> Result<Evaluation<S, RealPart>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    if let Some(form) = r.form().copied()
        && !take_real_part
    {
        let nonreal = if form.scaled.is_some() {
            // The positive real scale cancels in this comparison. Reconstructing
            // the ordinary value here would discard valid overflow results.
            form.e.value.im.abs() > form.e.error
        } else {
            let (_, _, error) = form.narrowed::<S>()?;
            r.ordinary().is_some_and(|v| v.im.abs() > error)
        };
        if nonreal
            || form
                .derivative::<S>()?
                .is_some_and(|d| d.value().im.abs() > d.absolute_error())
        {
            return Err(ERRORS.unstable());
        }
    }
    Ok(Evaluation::from_parts(
        r.ordinary().map(|v| v.re),
        r.representation(),
        r.termination(),
        r.counts(),
        r.algorithm(),
        r.form().copied().map(|inner| RealPart {
            inner,
            take_real_part,
        }),
    ))
}

impl<S: RealScalar> Publish<S> for RealPart
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn publish(self, value: Option<S>) -> Result<Forms<S>, SciOxideError> {
        let take_real_part = self.take_real_part;
        let complex = value
            .map(|real| {
                self.inner
                    .narrowed::<S>()
                    .map(|(value, _, _)| Complex::new(real, value.im))
            })
            .transpose()?;
        let r = self.inner.publish(complex)?;
        let convert = |v: &ValueWithError<Complex<S>, S>| {
            ValueWithError::new(v.value().re, v.absolute_error(), None, v.reliability())
        };
        Ok(Forms {
            value: r.value.as_ref().map(convert).transpose()?,
            scaled: r
                .scaled
                .and_then(|v| {
                    let scaled =
                        ScaledValue::from_components(v.value().real(), scioxide_core::Scaled::ZERO);
                    let shift = scaled
                        .scale_exponent()
                        .checked_sub(v.value().scale_exponent())?;
                    let error = crate::result::scaled_error(v.absolute_error(), shift)?;
                    Some(ValueWithError::new(scaled, error, None, v.reliability()))
                })
                .transpose()?,
            logarithmic: if take_real_part {
                // The real part has its own relative interval; no log estimate
                // exists once that interval reaches zero.
                r.value
                    .as_ref()
                    .and_then(|v| {
                        let real = v.value().re;
                        (real != S::ZERO)
                            .then(|| {
                                crate::result::log_error(
                                    (v.absolute_error() / real.abs()).min(S::MAX),
                                    real.abs().ln(),
                                )
                            })
                            .flatten()
                            .map(|error| {
                                ValueWithError::new(
                                    LogValue::from_value(real)?,
                                    error,
                                    None,
                                    v.reliability(),
                                )
                            })
                    })
                    .transpose()?
            } else {
                r.logarithmic
                    .map(|v| {
                        let magnitude = v.value().log_magnitude();
                        if magnitude.is_some() && v.value().scaled_phase().round()?.re == S::ZERO {
                            return Ok::<_, SciOxideError>(None);
                        }
                        let phase = if magnitude.is_none() {
                            S::ZERO
                        } else {
                            v.value().scaled_phase().round()?.re.signum()
                        };
                        Ok(Some(ValueWithError::new(
                            {
                                match magnitude {
                                    Some(magnitude) => LogValue::from_log(phase, magnitude),
                                    None => LogValue::from_value(phase),
                                }
                            }?,
                            v.absolute_error(),
                            None,
                            v.reliability(),
                        )?))
                    })
                    .transpose()?
                    .flatten()
            },
            derivative: r.derivative.as_ref().map(convert).transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_projection_checks_scaled_imaginary_parts_and_derivatives() -> Result<(), SciOxideError>
    {
        for scale in [0.0, 1000.0] {
            for (imaginary, derivative_imaginary, accepted) in [
                (0.0, 0.0, true),
                (1e-16, 0.0, true),
                (1.0, 0.0, false),
                (0.0, 1.0, false),
            ] {
                let mut estimate = Estimate::new(
                    Complex::new(1.0, imaginary),
                    1e-15,
                    scioxide_core::algorithm_component!("projection_test"),
                );
                estimate.scale = scale;
                estimate.derivative = Some(Complex::new(1.0, derivative_imaginary));
                estimate.derivative_error = 1e-15;
                let evaluation = report::<f64>(Ok(estimate), &Work::new(1))?;
                assert_eq!(project_real(evaluation, false).is_ok(), accepted);
            }
        }
        Ok(())
    }
}
