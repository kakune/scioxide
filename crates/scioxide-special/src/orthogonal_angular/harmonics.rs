//! Normalized spherical harmonics and scale-safe associated-Legendre recurrence.
use crate::complex_support::{ScaledEvaluation, ScaledForm};
use crate::orthogonal_angular::legendre::AssociatedValue;
use crate::result::{Either, Evaluation, Form};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::real;
use crate::{LegendreOptions, SpecialFunctionResult};
use scioxide_core::Scaled;
use scioxide_core::backend::scaled::factor_common_scale;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_core::{algorithm_component, algorithm_id};
pub(crate) fn normalized_step(degree: usize, order: usize) -> Result<(f64, f64), SciOxideError> {
    let k = f64::try_from_usize_exact(degree)?;
    let m = f64::try_from_usize_exact(order)?;
    let denominator = (k - m) * (k + m);
    let a = ((2.0 * k - 1.0) * (2.0 * k + 1.0) / denominator).sqrt();
    let b = if degree == order + 1 {
        0.0
    } else {
        ((2.0 * k + 1.0) * (k - 1.0 - m) * (k - 1.0 + m) / ((2.0 * k - 3.0) * denominator)).sqrt()
    };
    Ok((a, b))
}
/// Unit-sphere normalized associated Legendre factor, including Condon–Shortley. The caller
/// supplies sin(theta) separately to keep accuracy near a pole.
pub(crate) fn normalized_associated(
    degree: usize,
    order: usize,
    x: f64,
    sine: f64,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<AssociatedValue, SciOxideError> {
    let mut diagonal = Scaled::from_value(1.0 / (4.0 * core::f64::consts::PI).sqrt())?;
    for k in 1..=order {
        step()?;
        let k = f64::try_from_usize_exact(k)?;
        diagonal = diagonal.mul(Scaled::from_value(
            -sine * ((2.0 * k + 1.0) / (2.0 * k)).sqrt(),
        )?)?;
    }
    if degree == order {
        return Ok(AssociatedValue {
            value: diagonal,
            envelope: diagonal.abs(),
        });
    }
    step()?;
    let m = f64::try_from_usize_exact(order)?;
    let mut previous = diagonal;
    let mut current = diagonal.mul(Scaled::from_value(x * (2.0 * m + 3.0).sqrt())?)?;
    let mut envelope = diagonal.abs().add(current.abs())?;
    for k in (order + 2)..=degree {
        step()?;
        let (a, b) = normalized_step(k, order)?;
        let first = current.mul(Scaled::from_value(a * x)?)?;
        let second = previous.mul(Scaled::from_value(b)?)?;
        envelope = first.abs().add(second.abs())?;
        let next = first.sub(second)?;
        previous = current;
        current = next;
    }
    Ok(AssociatedValue {
        value: current,
        envelope,
    })
}
fn evaluate<S: RealScalar>(
    degree: usize,
    order: i32,
    theta: S,
    phi: S,
    real_basis: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    if !theta.is_finite() || !phi.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "angles must be finite",
        ));
    }
    let m = usize::try_from(order.unsigned_abs())
        .map_err(|_| SciOxideError::size_overflow(scioxide_core::quantity!("order")))?;
    if m > degree {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("order"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "require |m|<=degree and polar angle in [0,pi]; angles use radians",
        ));
    }
    if theta < S::ZERO || theta > S::PI {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("theta"),
            scioxide_core::DomainReason::OutsideDomain,
            "require |m|<=degree and polar angle in [0,pi]; angles use radians",
        ));
    }
    work.check()?;
    let theta = theta.try_to_f64_exact()?;
    let phi = phi.try_to_f64_exact()?;
    let (sin, cos) = if theta == 0.0 {
        (0.0, 1.0)
    } else if theta == S::PI.try_to_f64_exact()? {
        (0.0, -1.0)
    } else {
        RealScalar::sin_cos(theta)
    };
    let value = normalized_associated(degree, m, cos, sin, &mut || work.try_step())?;
    // Integer powers of exp(i*phi) avoid rounding or overflowing m*phi.
    let (sin_phi, cos_phi) = RealScalar::sin_cos(phi);
    let mut base = Complex::new(cos_phi, sin_phi);
    let mut phase = real(1.0);
    let mut power = order.unsigned_abs();
    while power > 0 {
        work.try_step()?;
        if power & 1 != 0 {
            phase *= base;
        }
        power >>= 1;
        if power > 0 {
            base *= base;
            base = base / base.norm();
        }
    }
    let (s, c) = (phase.im, phase.re);
    let factor = if real_basis {
        if order == 0 {
            real(1.0)
        } else {
            let sign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
            real(core::f64::consts::SQRT_2 * sign * if order < 0 { s } else { c })
        }
    } else if order < 0 {
        let sign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
        Complex::new(sign * c, -sign * s)
    } else {
        Complex::new(c, s)
    };
    let (parts, exponent) = factor_common_scale([value.value, value.envelope])?;
    let mantissa = parts[0].round()?;
    // Normalized forward recurrence error model, not an enclosure certificate.
    let relative = 32.0
        * f64::EPSILON
        * (1.0 + f64::try_from_usize_exact(degree)? + f64::from(order.unsigned_abs())).powi(2);
    Ok(Estimate::from(crate::support::JetData {
        derivative_error: None,
        y: Jet::constant(factor * mantissa),
        scale: Jet::constant(real(exponent as f64 * core::f64::consts::LN_2)),
        error: relative
            * parts[1].round()?
            * if real_basis && order != 0 {
                core::f64::consts::SQRT_2
            } else {
                1.0
            },
        method: algorithm_component!("normalized_legendre_recurrence"),
    }))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the spherical harmonic `Y_l^m(θ,φ)`, normalized on the unit sphere.

```text
Y_l^m(θ,φ)  = √((2l+1)/(4π) · (l−m)!/(l+m)!) P_l^m(cos θ) e^(imφ),   0 ≤ m ≤ l,
Y_l^(−m)(θ,φ) = (−1)^m conj(Y_l^m(θ,φ)),
```

where `P_l^m` is [`crate::legendre_p_assoc`], including the Condon–Shortley phase
(DLMF 14.30.1). `θ ∈ [0, π]` is the polar angle and `φ` the azimuth, both in radians. The
integral of `|Y_l^m|²` over the unit sphere is one.

The normalized associated Legendre factor is computed by a recurrence on the normalized
functions with binary rescaling, so large degrees neither overflow nor underflow. `e^(imφ)` is
formed by repeated squaring of `e^(iφ)`, which avoids rounding the product `mφ`. The cost is
about `l + log₂|m|` work units.

# Examples

```
// Y₀⁰ = 1/√(4π) and Y₁⁰(θ, φ) = √(3/(4π)) cos θ
let y00 = scioxide_special::spherical_harmonic(0, 0, 0.5_f64, 0.25)?;
let y10 = scioxide_special::spherical_harmonic(1, 0, 0.5_f64, 0.25)?;
let c = 1.0 / (4.0 * core::f64::consts::PI).sqrt();
assert!((y00.re - c).abs() < 1e-14);
assert!((y10.re - 3.0_f64.sqrt() * c * 0.5_f64.cos()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. The error estimate of the detailed result models the
rounding error of the normalized recurrence and grows like `32ε(1 + l + |m|)²` relative to the
magnitude of the terms. See the [crate accuracy table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `|m| > l`.
- `DomainError` (`OutsideDomain`) if `θ < 0` or `θ > π`.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §14.30. <https://dlmf.nist.gov/14.30>"###)]

    pub spherical_harmonic [S: RealScalar] (degree: usize,
        order: i32,
        theta: S,
        phi: S) -> Complex<S>
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: LegendreOptions;
    detailed spherical_harmonic_with_options -> SpecialFunctionResult<Complex<S>>;
    value {
        let options: LegendreOptions = LegendreOptions::default();

        harmonic_run(degree, order, theta, phi, options, None)?.value()
    }
    report {
        harmonic_run(degree, order, theta, phi, options, None)?.publish()
    }
    scoped pub spherical_harmonic_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            harmonic_run(degree, order, theta, phi, options, Some(context))?.publish()
        })
    }
}

fn harmonic_run<S: RealScalar>(
    degree: usize,
    order: i32,
    theta: S,
    phi: S,
    options: LegendreOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = evaluate(degree, order, theta, phi, false, &mut work);
    crate::complex_support::report(e, false, &work, algorithm_component!("spherical-harmonics"))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the real (tesseral) spherical harmonic `Y_lm(θ,φ)`.

```text
Y_lm = √2 (−1)^m Re Y_l^m,       m > 0,
Y_l0 = Y_l^0,
Y_lm = √2 (−1)^|m| Im Y_l^|m|,   m < 0,
```

where `Y_l^m` is [`crate::spherical_harmonic`]. The factor `(−1)^m` cancels the
Condon–Shortley phase. The real harmonics are orthonormal on the unit sphere. Angles, domain
and method are those of [`crate::spherical_harmonic`].

# Examples

```
// Y₀₀ = 1/√(4π) and Y₁₁(θ, φ) = √(3/(4π)) sin θ cos φ
let (theta, phi) = (0.5_f64, 0.25_f64);
let y00 = scioxide_special::spherical_harmonic_real(0, 0, theta, phi)?;
let y11 = scioxide_special::spherical_harmonic_real(1, 1, theta, phi)?;
let c = (3.0 / (4.0 * core::f64::consts::PI)).sqrt();
assert!((y00 - 1.0 / (4.0 * core::f64::consts::PI).sqrt()).abs() < 1e-13);
assert!((y11 - c * theta.sin() * phi.cos()).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the recurrence error estimate is that of
[`crate::spherical_harmonic`], multiplied by `√2` for `m ≠ 0`, with additional rounding
from publication in the requested scalar type. See the
[crate accuracy table](crate#accuracy)."###, r###"- `InvalidInput` (`OutOfRange`) if `|m| > l`.
- `DomainError` (`OutsideDomain`) if `θ < 0` or `θ > π`.
- `ConvergenceFailure` (`IterationLimit`) if the evaluation needs more work than the budget.

# References

NIST DLMF, §14.30. <https://dlmf.nist.gov/14.30>"###)]

    pub spherical_harmonic_real [S: RealScalar] (degree: usize,
        order: i32,
        theta: S,
        phi: S) -> S
    where [];
    options options: LegendreOptions;
    detailed spherical_harmonic_real_with_options -> SpecialFunctionResult<S>;
    value {
        spherical_harmonic_real_evaluate(degree, order, theta, phi, LegendreOptions::default(), None)?
            .value()
    }
    report {
        spherical_harmonic_real_evaluate(degree, order, theta, phi, options, None)?.publish()
    }
    scoped pub spherical_harmonic_real_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            spherical_harmonic_real_evaluate(degree, order, theta, phi, options, Some(context))?
                .publish()
        })
    }
}

/// Evaluates before projecting the result.
fn spherical_harmonic_real_evaluate<S: RealScalar>(
    degree: usize,
    order: i32,
    theta: S,
    phi: S,
    options: LegendreOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<ScaledEvaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = evaluate(degree, order, theta, phi, true, &mut work);
    crate::complex_support::report(e, false, &work, algorithm_component!("spherical-harmonics"))
}

/// Associated Legendre report.
pub(crate) fn associated_report<S: RealScalar>(
    degree: usize,
    order: i32,
    x: S,
    options: crate::LegendreOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<Evaluation<S, Either<Form<S>, ScaledForm>>, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "the associated Legendre argument must be finite",
        ));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    fn associated_estimate<S: RealScalar>(
        degree: usize,
        order: i32,
        x: S,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        let value = crate::orthogonal_angular::legendre::associated_scaled(
            degree,
            order,
            x.try_to_f64_exact()?,
            work,
        )?;
        let (parts, exponent) = factor_common_scale([value.value, value.envelope])?;
        // Materialize by binary scaling, avoiding log/exp round trips for raw
        // associated values. Out-of-range outputs retain scaled/log reports.
        let ordinary = value
            .value
            .round()
            .ok()
            .filter(|v| *v != 0.0 || value.value.is_zero());
        let (mantissa, envelope, exponent) = if let Some(ordinary) = ordinary {
            (ordinary, value.envelope.round()?, 0)
        } else {
            (parts[0].round()?, parts[1].round()?, exponent)
        };
        Ok(Estimate::from(crate::support::JetData {
            derivative_error: None,
            y: Jet::constant(real(mantissa)),
            scale: Jet::constant(real(exponent as f64 * core::f64::consts::LN_2)),
            error: 64.0
                * f64::EPSILON
                * (1.0 + degree as f64 + 2.0 * f64::from(order.unsigned_abs())).powi(2)
                * envelope,
            method: algorithm_component!("scaled_compensated_associated"),
        }))
    }
    let evaluation = associated_estimate(degree, order, x, &mut work);
    if evaluation.is_ok() && (degree == 0 || x.abs() == S::ONE) {
        let value = if degree == 0 {
            S::ONE
        } else if order != 0 {
            if order >= 0 && !order.unsigned_abs().is_multiple_of(2) {
                -S::ZERO
            } else {
                S::ZERO
            }
        } else if x < S::ZERO && !degree.is_multiple_of(2) {
            -S::ONE
        } else {
            S::ONE
        };
        return Ok(Evaluation::finite(
            value,
            crate::result::Bound::exact(),
            None,
            algorithm_id!("legendre_p_assoc", "exact_endpoint", 1),
            work.used,
            crate::SpecialFunctionTermination::Exact,
        )
        .map_form(Either::Left));
    }
    Ok(crate::complex_support::report(
        evaluation,
        false,
        &work,
        algorithm_component!("legendre_p_assoc"),
    )?
    .map_form(Either::Right))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_basis_scales_recurrence_error_only_for_nonzero_orders() -> Result<(), SciOxideError> {
        for order in [-1, 0, 1] {
            let complex = evaluate(2, order, 0.5_f64, 0.25, false, &mut Work::new(100))?;
            let real = evaluate(2, order, 0.5_f64, 0.25, true, &mut Work::new(100))?;
            let factor = if order == 0 {
                1.0
            } else {
                core::f64::consts::SQRT_2
            };
            assert!((real.error - factor * complex.error).abs() <= f64::EPSILON * real.error);
        }
        Ok(())
    }
}
