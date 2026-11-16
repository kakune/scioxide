//! Complex-argument Bessel/Hankel functions, DLMF 10.2, 10.6, 10.11, 10.17,
//! 10.25, 10.27 and 10.40. Off-axis arguments share adjacent modified Bessel
//! solutions with complex orders; axis kernels keep exact phases and limits.
pub(crate) use super::complex_bessel_common::Kind;
use super::complex_bessel_common::{derivative_terms, hankel_argument};
use crate::ValueWithError;
use crate::complex_support::ScaledForm;
use crate::result::{Either, Estimated, Evaluation, Forms, Publish};
use crate::support::Work;
use crate::support::{C, real};
use crate::{
    BesselOptions, ScalarRepresentation, SpecialFunctionResult, SpecialFunctionTermination,
};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, Complex, EstimateReliability, EvaluationCounts, LogValue,
    NumericScalar, RealScalar, ScaledValue, SciOxideError, algorithm_component,
};
/// Family estimate `value * exp(scale)`, with error in the same exponential coordinates. This is a
/// logarithmic recurrence state, not a binary component representation; final binary conversion is
/// owned by core's ScaledValue.
#[derive(Clone, Copy)]
pub(crate) struct ExponentialEstimate {
    pub(crate) value: C,
    pub(crate) scale: f64,
    pub(crate) error: f64,
}

fn phase(x: f64) -> C {
    let (s, c) = RealScalar::sin_cos(x);
    C::new(c, s)
}
fn pi_phase(x: f64) -> Result<C, SciOxideError> {
    let (s, c) = scioxide_core::sin_cos_pi(x)?;
    Ok(C::new(c, s))
}
pub(crate) fn multiply(mut a: ExponentialEstimate, factor: C) -> ExponentialEstimate {
    // A real factor scales each component, keeping the signs of a value that
    // rounded to zero; a complex product with a zero imaginary part can cancel them.
    a.value = if factor.im == 0.0 {
        a.value.scale(factor.re)
    } else {
        a.value * factor
    };
    a.error *= factor.norm();
    a
}
pub(crate) fn add(a: ExponentialEstimate, b: ExponentialEstimate) -> ExponentialEstimate {
    if a.value == real(0.0) && a.error == 0.0 {
        return b;
    }
    if b.value == real(0.0) && b.error == 0.0 {
        return a;
    }
    let scale = a.scale.max(b.scale);
    let x = RealScalar::exp(a.scale - scale);
    let y = RealScalar::exp(b.scale - scale);
    ExponentialEstimate {
        value: a.value.scale(x) + b.value.scale(y),
        scale,
        error: a.error * x
            + b.error * y
            + 8.0 * f64::EPSILON * (a.value.norm() * x + b.value.norm() * y),
    }
}

pub(crate) fn kernel(
    kind: Kind,
    nu: f64,
    z: C,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    if z.re != 0.0 && z.im != 0.0 {
        return adjacent_pair(kind, nu, z, w).map(|(value, _)| value);
    }
    w.try_step()?;
    if matches!(kind, Kind::H1 | Kind::H2) && z.im == 0.0 && z.re != 0.0 {
        let j = crate::bessel_airy::bessel::bessel_j(nu, z.re.abs(), w)?;
        let y = crate::bessel_airy::bessel::bessel_y(nu, z.re.abs(), w)?;
        let value = if z.re > 0.0 {
            C::new(j, if matches!(kind, Kind::H1) { y } else { -y })
        } else {
            let c = scioxide_core::cos_pi(nu)?;
            let s = scioxide_core::sin_pi(nu)?;
            let first = matches!(kind, Kind::H1) ^ z.im.is_sign_negative();
            let mut value = if first {
                C::new(-c * j + s * y, s * j + c * y)
            } else {
                C::new(3.0 * c * j - s * y, s * j - c * y)
            };
            if z.im.is_sign_negative() {
                value = value.conj();
            }
            value
        };
        return Ok(ExponentialEstimate {
            value,
            scale: 0.0,
            error: 2.0 * REAL_BESSEL_RELATIVE * (j.abs() + y.abs()),
        });
    }
    if matches!(kind, Kind::H2) {
        let mut value = kernel(Kind::H1, nu, z.conj(), w)?;
        value.value = value.value.conj();
        return Ok(value);
    }
    if matches!(kind, Kind::H1) {
        return Ok(multiply(
            kernel(Kind::K, nu, hankel_argument(z, true), w)?,
            pi_phase(-nu * 0.5)? * C::new(0.0, -2.0 / core::f64::consts::PI),
        ));
    }
    if z.im.is_sign_negative() {
        let mut value = kernel(kind, nu, z.conj(), w)?;
        value.value = value.value.conj();
        return Ok(value);
    }
    if z == real(0.0) {
        if matches!(kind, Kind::J | Kind::I) && (nu >= 0.0 || nu == RealScalar::round(nu)) {
            return Ok(ExponentialEstimate {
                value: real(if nu == 0.0 { 1.0 } else { 0.0 }),
                scale: 0.0,
                error: 0.0,
            });
        }
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("z"),
            scioxide_core::DomainReason::Pole,
            "the origin is singular for this order and family",
        ));
    }
    let nu = if matches!(kind, Kind::K) {
        nu.abs()
    } else {
        nu
    };
    if z.re < 0.0 {
        return match kind {
            Kind::J | Kind::I => Ok(multiply(kernel(kind, nu, -z, w)?, pi_phase(nu)?)),
            Kind::Y => Ok(add(
                multiply(kernel(Kind::Y, nu, -z, w)?, pi_phase(-nu)?),
                multiply(
                    kernel(Kind::J, nu, -z, w)?,
                    C::new(0.0, 2.0 * scioxide_core::cos_pi(nu)?),
                ),
            )),
            Kind::K => Ok(add(
                multiply(kernel(Kind::K, nu, -z, w)?, pi_phase(-nu)?),
                multiply(
                    kernel(Kind::I, nu, -z, w)?,
                    C::new(0.0, -core::f64::consts::PI),
                ),
            )),
            _ => unreachable_kind(),
        };
    }
    if matches!(kind, Kind::J) && nu >= 0.0 && z.norm() < 1e-8 {
        w.try_step()?;
        let (gamma, sign) = crate::gamma_beta::gamma::gamma_log_parts(nu + 1.0)?;
        return power_scaled(nu, z, -gamma, sign);
    }
    if matches!(kind, Kind::J) && z.re == 0.0 {
        let value = if nu.abs() >= 32.0 {
            let (logarithm, sign) =
                crate::bessel_airy::bessel::modified_uniform_log(nu, z.im, false, false, w)?;
            ExponentialEstimate {
                value: real(sign),
                scale: logarithm,
                error: REAL_BESSEL_RELATIVE,
            }
        } else {
            ExponentialEstimate {
                value: real(crate::bessel_airy::bessel::bessel_i_scaled(nu, z.im, w)?),
                scale: z.im,
                error: 0.0,
            }
        };
        let mut value = multiply(value, pi_phase(nu * 0.5)?);
        value.error += REAL_BESSEL_RELATIVE * value.value.norm();
        return Ok(value);
    }
    if z.im == 0.0 && z.re >= 1e-8 {
        return real_base(kind, nu, z, w);
    }
    if matches!(kind, Kind::I) {
        return Ok(multiply(
            kernel(Kind::J, nu, C::new(z.im, -z.re), w)?,
            pi_phase(nu * 0.5)?,
        ));
    }
    if z.im != 0.0 {
        return adjacent_pair(kind, nu, z, w).map(|(value, _)| value);
    }
    if nu < 0.0 {
        let order = -nu;
        let cosine = scioxide_core::cos_pi(order)?;
        let sine = scioxide_core::sin_pi(order)?;
        let main = multiply(kernel(kind, order, z, w)?, real(cosine));
        if sine == 0.0 {
            return Ok(main);
        }
        return match kind {
            Kind::J => Ok(add(
                main,
                multiply(kernel(Kind::Y, order, z, w)?, real(-sine)),
            )),
            Kind::Y => Ok(add(
                main,
                multiply(kernel(Kind::J, order, z, w)?, real(sine)),
            )),
            _ => unreachable_kind(),
        };
    }
    if nu < 1.5 {
        return real_base(kind, nu, z, w);
    }
    let index = RealScalar::round(nu);
    if index > w.remaining as f64 {
        w.complete = false;
        return Err(crate::result::exhausted(
            scioxide_core::ConvergenceDiagnostics::new(),
        ));
    }
    let n = index.try_to_usize_exact()?;
    let fraction = nu - index;
    let a = real_base(kind, fraction, z, w)?;
    let b = real_base(kind, fraction + 1.0, z, w)?;
    let common = a.scale.max(b.scale);
    let zero = a.value.scale(RealScalar::exp(a.scale - common));
    let one = b.value.scale(RealScalar::exp(b.scale - common));
    let norm = z.norm();
    let miller = matches!(kind, Kind::J) && nu > norm;
    if !miller {
        let mut previous = zero;
        let mut current = one;
        let mut scale = common;
        let threshold = RealScalar::sqrt(f64::MAX);
        for k in 1..n {
            w.try_step()?;
            let coefficient = real(2.0 * (fraction + k as f64)) / z;
            if current.norm().max(previous.norm()) > threshold / coefficient.norm().max(1.0) {
                previous = previous.unscale(threshold);
                current = current.unscale(threshold);
                scale += RealScalar::ln(threshold);
            }
            let next = coefficient * current
                + if matches!(kind, Kind::K) {
                    previous
                } else {
                    -previous
                };
            previous = current;
            current = next;
        }
        return Ok(ExponentialEstimate {
            value: current,
            scale,
            error: 128.0 * f64::EPSILON * (w.used + 1) as f64 * (current.norm() + previous.norm())
                + (a.error + b.error) * current.norm() / zero.norm().max(one.norm()),
        });
    }
    let start = nu.max(norm) + 32.0 + RealScalar::sqrt(40.0 * (nu + norm));
    if start > w.remaining as f64 {
        w.complete = false;
        return Err(crate::result::exhausted(
            scioxide_core::ConvergenceDiagnostics::new(),
        ));
    }
    let mut top = RealScalar::ceil(start).try_to_usize_exact()?;
    let mut previous: Option<ExponentialEstimate> = None;
    let threshold = RealScalar::sqrt(f64::MAX);
    loop {
        if top > w.remaining {
            w.complete = false;
            return Err(crate::result::exhausted(
                scioxide_core::ConvergenceDiagnostics::new(),
            ));
        }
        let mut current = real(1.0);
        let mut upper = real(0.0);
        let mut target = real(0.0);
        let mut scale = common;
        for k in (1..=top).rev() {
            w.try_step()?;
            let coefficient = real(2.0 * (fraction + k as f64)) / z;
            if current.norm() > threshold / coefficient.norm().max(1.0) {
                current = current.unscale(threshold);
                upper = upper.unscale(threshold);
                if k < n {
                    scale -= RealScalar::ln(threshold);
                }
            }
            if k == n {
                target = current;
            }
            let lower = coefficient * current - upper;
            upper = current;
            current = lower;
        }
        let (seed, denominator) = if zero.norm() >= one.norm() {
            (zero, current)
        } else {
            (one, upper)
        };
        let target_norm = target.norm();
        let mut value = ExponentialEstimate {
            value: target.unscale(target_norm) * (seed / denominator),
            scale: scale + RealScalar::ln(target_norm),
            error: 0.0,
        };
        if let Some(old) = previous {
            let delta = add(value, multiply(old, real(-1.0)));
            let relative = delta.value.norm() * RealScalar::exp(delta.scale - value.scale)
                / value.value.norm();
            if relative < 1e-12 {
                value.error = value.value.norm()
                    * (relative
                        + 128.0 * f64::EPSILON * (w.used + 1) as f64
                        + (a.error + b.error) / zero.norm().max(one.norm()));
                return Ok(value);
            }
        }
        previous = Some(value);
        top = top.saturating_add(24);
    }
}
fn unreachable_kind<T>() -> Result<T, SciOxideError> {
    Err(SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "invalid internal function dispatch",
    ))
}
// Only positive real arguments reach these reduced-order seeds. Nonreal
// arguments use adjacent_pair, including for internal Airy/spherical consumers.
fn real_base(
    kind: Kind,
    nu: f64,
    z: C,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    w.try_step()?;
    if matches!(kind, Kind::J | Kind::Y) && nu >= 1000.0 && nu > 2.0 * z.re.sqrt() {
        let e = super::bessel_uniform::evaluate(nu, z.re, matches!(kind, Kind::Y), w)?;
        if (!e.value.is_finite() || e.value == 0.0)
            && let Some(log) = e.logarithm
        {
            return Ok(ExponentialEstimate {
                value: real(e.sign),
                scale: log.value(),
                error: e.relative,
            });
        }
        return Ok(ExponentialEstimate {
            value: real(e.value),
            scale: 0.0,
            error: e.error,
        });
    }
    if matches!(kind, Kind::I | Kind::K) && z.re >= 1e-8 {
        let second = matches!(kind, Kind::K);
        let e = super::bessel::modified(nu, z.re, second, true, w)?;
        let scale = if second { -z.re } else { z.re };
        if (!e.value.is_finite() || e.value == 0.0)
            && let Some(log) = e.logarithm
        {
            return Ok(ExponentialEstimate {
                value: real(e.sign),
                scale: log.value() + scale,
                error: e.relative,
            });
        }
        return Ok(ExponentialEstimate {
            value: real(e.value),
            scale,
            error: e.error,
        });
    }
    if (nu.abs() == 0.5 || nu.abs() == 1.5) && matches!(kind, Kind::J | Kind::Y) {
        let value = if matches!(kind, Kind::J) {
            crate::bessel_airy::bessel::bessel_j(nu, z.re, w)?
        } else {
            crate::bessel_airy::bessel::bessel_y(nu, z.re, w)?
        };
        return Ok(ExponentialEstimate {
            value: real(value),
            scale: 0.0,
            error: 16.0 * f64::EPSILON * (1.0 / z.re).sqrt(),
        });
    }
    if z.re < 1e-8 {
        return small_base(kind, nu, z, w);
    }
    let (value, scale) = match kind {
        Kind::J => (crate::bessel_airy::bessel::bessel_j(nu, z.re, w)?, 0.0),
        Kind::Y => (crate::bessel_airy::bessel::bessel_y(nu, z.re, w)?, 0.0),
        _ => return unreachable_kind(),
    };
    Ok(ExponentialEstimate {
        value: real(value),
        scale,
        error: REAL_BESSEL_RELATIVE * value.abs(),
    })
}

/// Keeps the adjacent kernel's value and derivative in real exponential coordinates.
/// The complex phase is projected once, with its rounding propagated to each estimate.
fn adjacent_pair(
    kind: Kind,
    nu: f64,
    z: C,
    work: &mut Work,
) -> Result<(ExponentialEstimate, Option<ExponentialEstimate>), SciOxideError> {
    let e = super::complex_order_bessel::kernel_value(
        kind,
        real(nu),
        crate::support::Jet::variable(z),
        real(0.0),
        work,
    )?;
    if !e.y.v.is_finite() || !e.error.is_finite() || !e.scale.v.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::IllConditioned,
            "complex Bessel estimate exceeds its working representation",
        ));
    }
    let unit = phase(e.scale.v.im);
    let project = |value: C, error: f64| ExponentialEstimate {
        value: value * unit,
        scale: e.scale.v.re,
        error: error + 32.0 * f64::EPSILON * (1.0 + e.scale.v.im.abs()) * value.norm(),
    };
    let derivative = e.y.d + e.scale.d * e.y.v;
    let derivative_error = e.derivative_error.unwrap_or(e.error) + e.scale.d.norm() * e.error;
    Ok((
        project(e.y.v, e.error),
        (derivative.is_finite() && derivative_error.is_finite())
            .then(|| project(derivative, derivative_error)),
    ))
}

fn scale_function(kind: Kind, z: C, mut value: ExponentialEstimate) -> ExponentialEstimate {
    match kind {
        Kind::J | Kind::Y => value.scale -= z.im.abs(),
        Kind::I => value.scale -= z.re.abs(),
        Kind::K => {
            value.scale += z.re;
            value.value *= phase(z.im);
        }
        Kind::H1 => {
            value.scale += z.im;
            value.value *= phase(-z.re);
        }
        Kind::H2 => {
            value.scale -= z.im;
            value.value *= phase(z.re);
        }
    }
    value
}

/// Relative envelope of the real Bessel kernels, measured against reference values (64 `f64` ulps).
const REAL_BESSEL_RELATIVE: f64 = 64.0 * f64::EPSILON;

/// A recurrence estimate `value·exp(scale)` with its derivative.
#[derive(Clone, Copy)]
pub(crate) struct ExponentialForm {
    value: ExponentialEstimate,
    derivative: Option<ExponentialEstimate>,
}

/// Evaluation of a complex Bessel-family function: a recurrence estimate, or the complex-order
/// kernel's scaled estimate.
pub(crate) type BesselEvaluation<S> = Evaluation<Complex<S>, Either<ExponentialForm, ScaledForm>>;

/// The ordinary projection of a recurrence estimate, `None` beyond range.
fn project_component<S: RealScalar>(
    p: ExponentialEstimate,
) -> Result<Option<Complex<S>>, SciOxideError> {
    let component = |v: f64| {
        if v == 0.0 || p.scale == 0.0 {
            v
        } else {
            crate::support::wide::project(
                crate::support::wide::Wide::from(v),
                crate::support::wide::Wide::from(p.scale),
            )
        }
    };
    let re = component(p.value.re);
    let im = component(p.value.im);
    if S::try_from_f64_lossy(re).is_err() || S::try_from_f64_lossy(im).is_err() {
        return Ok(None);
    }
    Ok(Some(Complex::new(
        S::try_from_f64_lossy(re)?,
        S::try_from_f64_lossy(im)?,
    )))
}

/// The heuristic bound of a projected recurrence estimate.
fn projected_estimate<S: RealScalar>(
    p: ExponentialEstimate,
    v: Complex<S>,
    max: f64,
) -> Result<ValueWithError<Complex<S>, S>, SciOxideError> {
    let error = if p.error == 0.0 {
        0.0
    } else {
        crate::support::wide::project(
            crate::support::wide::Wide::from(p.error),
            crate::support::wide::Wide::from(p.scale),
        )
    };
    let e = S::try_from_f64_lossy(error.min(max))?
        + (S::EPSILON * v.re.abs().max(v.im.abs()))
            * S::try_from_f64_lossy(core::f64::consts::SQRT_2)?
        + S::ZERO.next_after(S::ONE);
    ValueWithError::new(v, e.min(S::MAX), None, EstimateReliability::Heuristic)
}

pub(crate) fn report<S: RealScalar>(
    value: ExponentialEstimate,
    derivative: Option<ExponentialEstimate>,
    used: usize,
    complete: bool,
    family: AlgorithmComponent,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let algorithm =
        AlgorithmId::from_components(family, algorithm_component!("scaled_recurrence"), 1);
    if !complete {
        return Err(crate::result::exhausted_work(used));
    }
    if !value.value.is_finite() || !value.scale.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "complex recurrence exceeds its working representation",
        ));
    }
    let ordinary = project_component::<S>(value)?;
    Ok(Evaluation::from_parts(
        ordinary,
        if ordinary.is_some() {
            ScalarRepresentation::Finite
        } else {
            ScalarRepresentation::Overflow
        },
        SpecialFunctionTermination::Converged,
        EvaluationCounts::from_iterations(used),
        algorithm,
        Some(Either::Left(ExponentialForm { value, derivative })),
    ))
}

/// The ordinary and derivative estimates of an exponential form.
type Estimates<S> = (
    Estimated<Complex<S>, Complex<S>>,
    Estimated<Complex<S>, Complex<S>>,
);

impl ExponentialForm {
    /// The ordinary and derivative estimates of `value`, without the scaled and logarithmic forms.
    fn estimates<S: RealScalar>(
        &self,
        value: Option<Complex<S>>,
    ) -> Result<Estimates<S>, SciOxideError>
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let max = S::MAX.try_to_f64_exact()?;
        let ordinary = value
            .map(|v| projected_estimate(self.value, v, max))
            .transpose()?;
        let derivative = self
            .derivative
            .map(|d| {
                project_component::<S>(d)?
                    .map(|v| projected_estimate(d, v, max))
                    .transpose()
            })
            .transpose()?
            .flatten();
        Ok((ordinary, derivative))
    }
}

impl<S: RealScalar> Publish<Complex<S>> for ExponentialForm
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn publish(self, ordinary: Option<Complex<S>>) -> Result<Forms<Complex<S>>, SciOxideError> {
        let value = self.value;
        let magnitude = value.value.re.abs().max(value.value.im.abs());
        let unit = if magnitude == 0.0 {
            real(0.0)
        } else {
            value.value.unscale(magnitude)
        };
        let unit_norm = unit.norm();
        let logarithm = if magnitude == 0.0 {
            None
        } else {
            Some(RealScalar::ln(magnitude) + RealScalar::ln(unit_norm) + value.scale)
        };
        let phase = LogValue::from_value(value.value)?.scaled_phase();
        let phase = scioxide_core::ScaledValue::from_component_parts(
            Complex::new(
                S::try_from_f64_lossy(phase.component_significands().re)?,
                S::try_from_f64_lossy(phase.component_significands().im)?,
            ),
            phase.component_exponents(),
        )?;
        // The mantissa error is relative; the rounded scale is an absolute
        // log-unit error. Their sum is the relative error of the scalar value.
        let (mantissa_relative, scale_error) = if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            (
                value.error / magnitude / unit_norm + 32.0 * f64::EPSILON,
                32.0 * f64::EPSILON * value.scale.abs(),
            )
        };
        let max = S::MAX.try_to_f64_exact()?;
        let mantissa_relative = S::try_from_f64_lossy(mantissa_relative.min(max))? + S::EPSILON;
        let scale_error = S::try_from_f64_lossy(scale_error.min(max))?;
        let relative = mantissa_relative + scale_error;
        let (ordinary, derivative) = self.estimates(ordinary)?;
        let logarithmic = {
            match logarithm.map(S::try_from_f64_lossy).transpose()? {
                Some(magnitude) => LogValue::from_scaled_log(phase, magnitude),
                None => LogValue::from_value(phase.round()?),
            }
        }?;
        let scaled = if let Some(log) = logarithm {
            match scioxide_core::Scaled::from_log(1.0_f64, log) {
                Ok(magnitude) => {
                    let factor = scioxide_core::Scaled::from_parts(
                        S::try_from_f64_lossy(magnitude.significand())?,
                        magnitude.exponent(),
                    )?;
                    Some(phase.mul_real(factor)?)
                }
                Err(error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability {
                            issue: scioxide_core::NumericalIssue::ExponentOutOfRange,
                            ..
                        }
                    ) =>
                {
                    None
                }
                Err(error) => return Err(error),
            }
        } else {
            Some(ScaledValue::from_value(Complex::new(S::ZERO, S::ZERO))?)
        };
        // Natural-log units; none once the mantissa interval may reach zero.
        let log_rounding = S::EPSILON
            * logarithm
                .map(S::try_from_f64_lossy)
                .transpose()?
                .unwrap_or(S::ZERO)
                .abs();
        let log_error = crate::result::log_error(mantissa_relative, S::ZERO)
            .map(|error| error + scale_error + log_rounding);
        Ok(Forms {
            value: ordinary,
            scaled: scaled
                .map(|s| {
                    crate::result::scaled_estimate(s, relative, EstimateReliability::Heuristic)
                })
                .transpose()?,
            logarithmic: log_error
                .map(|error| {
                    ValueWithError::new(logarithmic, error, None, EstimateReliability::Heuristic)
                })
                .transpose()?,
            derivative,
        })
    }
}
fn evaluate<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    options: BesselOptions,
    kind: Kind,
    scaled: bool,
    family: AlgorithmComponent,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate_controlled(order, z, options, kind, scaled, family, None)
}
pub(crate) fn cylindrical_pair(
    kind: Kind,
    order: f64,
    z: C,
    work: &mut Work,
) -> Result<(ExponentialEstimate, Option<ExponentialEstimate>), SciOxideError> {
    if z.im != 0.0 && (z.re != 0.0 || matches!(kind, Kind::Y | Kind::K)) {
        return adjacent_pair(kind, order, z, work);
    }
    let value = kernel(kind, order, z, work)?;
    fn derivative(
        kind: Kind,
        order: f64,
        z: C,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<ExponentialEstimate, SciOxideError> {
        let lower = kernel(kind, order - 1.0, z, work)?;
        let upper = kernel(kind, order + 1.0, z, work)?;
        let derivative = if matches!(kind, Kind::I | Kind::K) {
            add(lower, upper)
        } else {
            add(lower, multiply(upper, real(-1.0)))
        };
        Ok::<_, SciOxideError>(multiply(
            derivative,
            real(if matches!(kind, Kind::K) { -0.5 } else { 0.5 }),
        ))
    }
    let derivative = derivative(kind, order, z, work);
    let derivative = match derivative {
        Ok(d) => Some(d),
        Err(ref error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
                    | scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            None
        }
        Err(e) => return Err(e),
    };
    Ok((value, derivative))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_controlled<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    options: BesselOptions,
    kind: Kind,
    scaled: bool,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    if !order.has_finite_components() || !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and argument must be finite",
        ));
    }
    let imaginary = order.imaginary_component().try_to_f64_exact()?;
    // All real orders off the axes share the adjacent complex kernels. Keep
    // the established low-order imaginary-axis route; the other axis cases
    // retain exact phases, origin limits and large-order scaling.
    if imaginary != 0.0
        || (z.im != S::ZERO && (z.re != S::ZERO || order.real_component().abs() <= S::ONE))
    {
        return Ok(crate::bessel_airy::complex_order_bessel::evaluate(
            C::new(order.real_component().try_to_f64_exact()?, imaginary),
            z,
            options,
            (kind, scaled),
            family,
            context,
            None,
        )?
        .map_form(Either::Right));
    }
    let nu = order.real_component().try_to_f64_exact()?;
    let argument = C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    crate::support::require_finite_param(argument, "Bessel parameter argument must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    fn scaled_cylindrical_pair(
        kind: Kind,
        nu: f64,
        argument: C,
        scaled: bool,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(ExponentialEstimate, Option<ExponentialEstimate>), SciOxideError> {
        let (mut value, mut derivative) = cylindrical_pair(kind, nu, argument, work)?;
        if scaled {
            value = scale_function(kind, argument, value);
            derivative = derivative.map(|d| scale_function(kind, argument, d));
        }
        Ok((value, derivative))
    }
    let result = scaled_cylindrical_pair(kind, nu, argument, scaled, &mut work);
    work.check()?;
    match result {
        Ok((v, d)) => report(v, d, work.used, work.complete, family),
        Err(_) if !work.complete => report(
            ExponentialEstimate {
                value: real(0.0),
                scale: 0.0,
                error: f64::MAX,
            },
            None,
            work.used,
            false,
            family,
        ),
        Err(e) => Err(e),
    }
}
pub(crate) fn bessel_j_complex<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::J,
        false,
        algorithm_component!(stringify!(bessel_j_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn bessel_j_complex_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::J,
        false,
        algorithm_component!(stringify!(bessel_j_complex)),
        context,
    )
}

pub(crate) fn bessel_y_complex<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::Y,
        false,
        algorithm_component!(stringify!(bessel_y_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn bessel_y_complex_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::Y,
        false,
        algorithm_component!(stringify!(bessel_y_complex)),
        context,
    )
}

pub(crate) fn bessel_i_complex<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::I,
        false,
        algorithm_component!(stringify!(bessel_i_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn bessel_i_complex_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::I,
        false,
        algorithm_component!(stringify!(bessel_i_complex)),
        context,
    )
}

pub(crate) fn bessel_k_complex<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::K,
        false,
        algorithm_component!(stringify!(bessel_k_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn bessel_k_complex_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::K,
        false,
        algorithm_component!(stringify!(bessel_k_complex)),
        context,
    )
}

pub(crate) fn hankel_1_complex<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::H1,
        false,
        algorithm_component!(stringify!(hankel_1_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn hankel_1_complex_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::H1,
        false,
        algorithm_component!(stringify!(hankel_1_complex)),
        context,
    )
}

pub(crate) fn hankel_2_complex<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::H2,
        false,
        algorithm_component!(stringify!(hankel_2_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn hankel_2_complex_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::H2,
        false,
        algorithm_component!(stringify!(hankel_2_complex)),
        context,
    )
}

pub(crate) fn bessel_i_complex_scaled<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::I,
        true,
        algorithm_component!(stringify!(bessel_i_complex_scaled)),
    )?
    .value()
}

#[inline]
pub(crate) fn bessel_i_complex_scaled_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::I,
        true,
        algorithm_component!(stringify!(bessel_i_complex_scaled)),
        context,
    )
}

pub(crate) fn bessel_k_complex_scaled<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        order,
        z,
        options,
        Kind::K,
        true,
        algorithm_component!(stringify!(bessel_k_complex_scaled)),
    )?
    .value()
}

#[inline]
pub(crate) fn bessel_k_complex_scaled_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S, O>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate_controlled(
        order,
        z,
        options,
        Kind::K,
        true,
        algorithm_component!(stringify!(bessel_k_complex_scaled)),
        context,
    )
}

fn power_scaled(
    power: f64,
    z: C,
    log_factor: f64,
    sign: f64,
) -> Result<ExponentialEstimate, SciOxideError> {
    let scale = power * (RealScalar::ln(z.norm()) - core::f64::consts::LN_2) + log_factor;
    let angular = if z.re == 0.0 {
        pi_phase(power * 0.5)?
    } else {
        phase(power * z.arg())
    };
    Ok(ExponentialEstimate {
        value: angular.scale(sign),
        scale,
        error: 32.0 * f64::EPSILON * (1.0 + scale.abs()),
    })
}
fn small_base(
    kind: Kind,
    nu: f64,
    z: C,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    w.try_step()?;
    let logarithm = z.ln() + real(0.577_215_664_901_532_9 - core::f64::consts::LN_2);
    if matches!(kind, Kind::J) {
        let (g, s) = crate::gamma_beta::gamma::gamma_log_parts(nu + 1.0)?;
        return power_scaled(nu, z, -g, s);
    }
    if nu.abs() < 1e-4 {
        // Temme's near-zero-order cancellation is removed before evaluation.
        let square = nu * nu;
        let average = -core::f64::consts::PI * core::f64::consts::PI / 12.0 * square
            - 0.270_580_808_427_784_5 * square * square;
        let h = -logarithm
            - real(
                1.202_056_903_159_594_3 / 3.0 * square
                    + 1.036_927_755_143_37 / 5.0 * square * square,
            );
        let q = h.scale(nu);
        let q2 = q * q;
        let sinhc = real(1.0)
            + q2 * (real(1.0 / 6.0)
                + q2 * (real(1.0 / 120.0) + q2 * (real(1.0 / 5040.0) + q2.scale(1.0 / 362880.0))));
        let value = if matches!(kind, Kind::K) {
            let sinc = if nu == 0.0 {
                1.0
            } else {
                scioxide_core::sin_pi(nu)? / (core::f64::consts::PI * nu)
            };
            h * sinhc.scale(RealScalar::exp(average) / sinc)
        } else {
            let angle = core::f64::consts::PI * nu * 0.5;
            let cotc = if angle == 0.0 {
                1.0
            } else {
                angle * RealScalar::cos(angle) / RealScalar::sin(angle)
            };
            -(q.cosh().scale(RealScalar::tan(angle))
                + h * sinhc.scale(2.0 / core::f64::consts::PI * cotc))
            .scale(RealScalar::exp(average))
        };
        return Ok(ExponentialEstimate {
            value,
            scale: 0.0,
            error: 128.0 * f64::EPSILON * value.norm() + z.norm() * z.norm() * logarithm.norm(),
        });
    }
    if (nu - 1.0).abs() < 1e-8 {
        let (g, s) = crate::gamma_beta::gamma::gamma_log_parts(nu)?;
        let leading = power_scaled(
            -nu,
            z,
            g - if matches!(kind, Kind::K) {
                core::f64::consts::LN_2
            } else {
                RealScalar::ln(core::f64::consts::PI)
            },
            if matches!(kind, Kind::K) { s } else { -s },
        )?;
        let factor = if matches!(kind, Kind::K) {
            0.5
        } else {
            1.0 / core::f64::consts::PI
        };
        let correction = (logarithm - real(0.5)) * z.scale(factor);
        // Rounding of the logarithmic term, the omitted O(z³ log z) terms and
        // the order offset from one.
        let log_scale = logarithm.norm() + 1.0;
        let correction_error = factor
            * z.norm()
            * (8.0 * f64::EPSILON * log_scale
                + z.norm() * z.norm() * log_scale
                + (nu - 1.0).abs() * log_scale * log_scale);
        return Ok(add(
            leading,
            ExponentialEstimate {
                value: correction,
                scale: 0.0,
                error: correction_error,
            },
        ));
    }
    let q = z * z * real(0.25);
    if matches!(kind, Kind::K) {
        let (ga, sa) = crate::gamma_beta::gamma::gamma_log_parts(nu)?;
        let (gb, sb) = crate::gamma_beta::gamma::gamma_log_parts(-nu)?;
        Ok(add(
            multiply(
                power_scaled(-nu, z, ga - core::f64::consts::LN_2, sa)?,
                real(1.0) + q.unscale(1.0 - nu),
            ),
            multiply(
                power_scaled(nu, z, gb - core::f64::consts::LN_2, sb)?,
                real(1.0) + q.unscale(1.0 + nu),
            ),
        ))
    } else {
        let sine = scioxide_core::sin_pi(nu)?;
        let (ga, sa) = crate::gamma_beta::gamma::gamma_log_parts(nu + 1.0)?;
        let (gb, sb) = crate::gamma_beta::gamma::gamma_log_parts(1.0 - nu)?;
        Ok(add(
            multiply(
                power_scaled(nu, z, -ga, sa)?,
                (real(1.0) - q.unscale(nu + 1.0)).scale(scioxide_core::cos_pi(nu)? / sine),
            ),
            multiply(
                power_scaled(-nu, z, -gb, sb)?,
                (real(1.0) - q.unscale(1.0 - nu)).scale(-1.0 / sine),
            ),
        ))
    }
}

mod extended;
mod spherical;
pub use extended::*;
pub(crate) use spherical::*;
