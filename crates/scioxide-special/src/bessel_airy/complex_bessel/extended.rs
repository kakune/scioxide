//! Real projections, weighted integrals, and derivative entry points.

use super::spherical::{nth_evaluate, spherical_evaluate, spherical_evaluate_controlled};
use super::*;

#[allow(clippy::too_many_arguments)]
fn real_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    kind: Kind,
    scaled: bool,
    degree: u32,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !order.is_finite() || !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and argument must be finite",
        ));
    }
    if matches!(kind, Kind::Y | Kind::K) && x <= S::ZERO || x < S::ZERO && order != order.round() {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "argument is outside the real derivative domain",
        ));
    }
    let o = options;
    let z = real(x.try_to_f64_exact()?);
    let nu = order.try_to_f64_exact()?;
    let r = nth_evaluate(nu, z, degree, o, kind, scaled, family, context)?;
    real_projection(r, family)
}
/// The real part of an `f64` complex evaluation, converted to `S` with its error estimate.
fn real_projection<S: RealScalar>(
    r: BesselEvaluation<f64>,
    family: AlgorithmComponent,
) -> Result<Evaluation<S>, SciOxideError> {
    let algorithm = AlgorithmId::from_components(
        family,
        algorithm_component!("neighbor_order_combination"),
        1,
    );
    if r.termination() == SpecialFunctionTermination::IterationLimit && r.ordinary().is_none() {
        return Err(crate::result::exhausted_work(r.counts().iterations()));
    }
    let (estimate, derivative) = match r.form().copied() {
        Some(Either::Left(form)) => form.estimates::<f64>(r.ordinary())?,
        Some(Either::Right(form)) => {
            let forms = Publish::<Complex<f64>>::publish(form, r.ordinary())?;
            (forms.value, forms.derivative)
        }
        None => (None, None),
    };
    let Some(estimate) = estimate else {
        r.value()?;
        return unreachable_kind();
    };
    let max = S::MAX.try_to_f64_exact()?;
    let value = crate::support::project_real::<S>(estimate.value().re)?;
    let error =
        S::try_from_f64_lossy(estimate.absolute_error().min(max))? + S::EPSILON * value.abs();
    let derivative = match derivative {
        Some(d) if d.value().re.is_finite() && S::try_from_f64_lossy(d.value().re).is_ok() => {
            let v = S::try_from_f64_lossy(d.value().re)?;
            Some((
                v,
                crate::result::Bound::heuristic(
                    (S::try_from_f64_lossy(d.absolute_error().min(max))? + S::EPSILON * v.abs())
                        .min(S::MAX),
                ),
            ))
        }
        _ => None,
    };
    Ok(crate::expansions::bounded(
        value,
        crate::result::Bound::heuristic(error),
        algorithm,
        r.counts().iterations(),
        r.termination() != SpecialFunctionTermination::IterationLimit,
    )?
    .with_derivative(derivative))
}
pub(crate) fn bessel_j_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_j_derivative_evaluate(order, x, degree, options, None)?.value()
}

#[doc = "The shared evaluation, charged to an optional solve context."]
pub(crate) fn bessel_j_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    real_evaluate(
        order,
        x,
        options,
        Kind::J,
        false,
        degree,
        algorithm_component!(stringify!(bessel_j_derivative)),
        context,
    )
}
pub(crate) fn bessel_y_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_y_derivative_evaluate(order, x, degree, options, None)?.value()
}

#[doc = "The shared evaluation, charged to an optional solve context."]
pub(crate) fn bessel_y_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    real_evaluate(
        order,
        x,
        options,
        Kind::Y,
        false,
        degree,
        algorithm_component!(stringify!(bessel_y_derivative)),
        context,
    )
}
pub(crate) fn bessel_i_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_i_derivative_evaluate(order, x, degree, options, None)?.value()
}

#[doc = "The shared evaluation, charged to an optional solve context."]
pub(crate) fn bessel_i_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    real_evaluate(
        order,
        x,
        options,
        Kind::I,
        false,
        degree,
        algorithm_component!(stringify!(bessel_i_derivative)),
        context,
    )
}
pub(crate) fn bessel_k_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_k_derivative_evaluate(order, x, degree, options, None)?.value()
}

#[doc = "The shared evaluation, charged to an optional solve context."]
pub(crate) fn bessel_k_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    real_evaluate(
        order,
        x,
        options,
        Kind::K,
        false,
        degree,
        algorithm_component!(stringify!(bessel_k_derivative)),
        context,
    )
}
pub(crate) fn bessel_i_scaled_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_i_scaled_derivative_evaluate(order, x, degree, options, None)?.value()
}

#[doc = "The shared evaluation, charged to an optional solve context."]
pub(crate) fn bessel_i_scaled_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    real_evaluate(
        order,
        x,
        options,
        Kind::I,
        true,
        degree,
        algorithm_component!(stringify!(bessel_i_scaled_derivative)),
        context,
    )
}
pub(crate) fn bessel_k_scaled_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_k_scaled_derivative_evaluate(order, x, degree, options, None)?.value()
}

#[doc = "The shared evaluation, charged to an optional solve context."]
pub(crate) fn bessel_k_scaled_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    real_evaluate(
        order,
        x,
        options,
        Kind::K,
        true,
        degree,
        algorithm_component!(stringify!(bessel_k_scaled_derivative)),
        context,
    )
}
fn lambda_kernel(nu: f64, x: f64, w: &mut Work) -> Result<ExponentialEstimate, SciOxideError> {
    w.try_step()?;
    if x == 0.0 {
        return Ok(ExponentialEstimate {
            value: real(1.0),
            scale: 0.0,
            error: 0.0,
        });
    }
    if nu == -0.5 {
        return Ok(ExponentialEstimate {
            value: real(RealScalar::cos(x)),
            scale: 0.0,
            error: 2.0 * f64::EPSILON * RealScalar::cos(x).abs(),
        });
    }
    if nu == 0.5 {
        return Ok(ExponentialEstimate {
            value: real(RealScalar::sin(x) / x),
            scale: 0.0,
            error: 2.0 * f64::EPSILON * (RealScalar::sin(x) / x).abs(),
        });
    }
    if (x / (nu + 1.0).abs()) * x <= 16.0 {
        let mut term = 1.0;
        let mut sum = term;
        let mut absolute = term;
        let mut k = 0usize;
        loop {
            w.try_step()?;
            k += 1;
            term *= -(x * 0.5 / (k as f64)) * (x * 0.5 / (nu + k as f64));
            sum += term;
            absolute += term.abs();
            if term.abs() <= f64::EPSILON * sum.abs() {
                break;
            }
        }
        return Ok(ExponentialEstimate {
            value: real(sum),
            scale: 0.0,
            error: term.abs() + 16.0 * f64::EPSILON * absolute,
        });
    }
    if (0.0..=32.0).contains(&nu) {
        let j = match crate::bessel_airy::bessel::bessel_j(nu, x, w) {
            Ok(value) => Some(value),
            Err(ref error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Cancellation
                            | scioxide_core::NumericalIssue::IllConditioned,
                        ..
                    }
                ) =>
            {
                None
            }
            Err(error) => return Err(error),
        };
        if let Some(j) = j {
            w.try_step()?;
            let (gamma, _) = crate::gamma_beta::gamma::gamma_log_parts(nu + 1.0)?;
            return Ok(ExponentialEstimate {
                value: real(j),
                scale: gamma - nu * (RealScalar::ln(x) - core::f64::consts::LN_2),
                error: REAL_BESSEL_RELATIVE * j.abs(),
            });
        }
    }
    let mut value = kernel(Kind::J, nu, real(x), w)?;
    w.try_step()?;
    let (gamma, sign) = crate::gamma_beta::gamma::gamma_log_parts(nu + 1.0)?;
    value.value = value.value.scale(sign);
    value.scale += gamma - nu * (RealScalar::ln(x) - core::f64::consts::LN_2);
    value.error += value.value.norm() * 64.0 * f64::EPSILON * (1.0 + gamma.abs());
    Ok(value)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Jahnke–Emden lambda function `Λ_ν(x) = Γ(ν + 1) (x/2)^(−ν) J_ν(x)`.

```text
Λ_ν(x) = Σₖ₌₀^∞ (−x²/4)^k / (k! (ν + 1)ₖ)
```

where `(ν + 1)ₖ` is the rising factorial. `Λ_ν` is an even entire function of `x` with
`Λ_ν(0) = 1`, defined here for real `ν ≥ 0` and every finite real `x`. The result never forms
`Γ(ν + 1)` or `(x/2)^ν` separately, so it stays finite where either factor would overflow or
underflow.

The series above is summed when `x² ≤ 16 (ν + 1)`. `Λ_(1/2)(x) = sin(x)/x` is evaluated
directly. Larger arguments use [`bessel_j`](crate::bessel_j) with a logarithmic gamma factor.
The work is bounded by the iteration budget of [`BesselOptions`](crate::BesselOptions).

# Examples

```
// Λ_(1/2)(x) = sin(x)/x
let lambda = scioxide_special::bessel_lambda(0.5_f64, 1.0)?;
assert!((lambda - 1.0_f64.sin()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; the detailed form reports an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `ν < 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- E. Jahnke and F. Emde, *Tables of Functions with Formulae and Curves*, Dover, 1945."###)]

    pub bessel_lambda [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: BesselOptions;
    detailed bessel_lambda_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        bessel_lambda_evaluate(order, x, options, context)
    }
    scoped pub bessel_lambda_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluates the detailed result before scalar projection.
fn bessel_lambda_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !order.is_finite() || !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and argument must be finite",
        ));
    }
    if order < S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("order"),
            scioxide_core::DomainReason::OutsideDomain,
            "order must be nonnegative",
        ));
    }
    let nu = order.try_to_f64_exact()?;
    let argument = x.try_to_f64_exact()?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    fn lambda_pair(
        nu: f64,
        argument: f64,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(ExponentialEstimate, ExponentialEstimate), SciOxideError> {
        let value = lambda_kernel(nu, argument.abs(), work)?;
        let derivative = if argument == 0.0 {
            ExponentialEstimate {
                value: real(0.0),
                scale: 0.0,
                error: 0.0,
            }
        } else {
            multiply(
                lambda_kernel(nu + 1.0, argument.abs(), work)?,
                real(-(argument / (nu + 1.0)) * 0.5),
            )
        };
        Ok((value, derivative))
    }
    let computation = lambda_pair(nu, argument, &mut work);
    work.check()?;
    let r = match computation {
        Ok((v, d)) => report(
            v,
            Some(d),
            work.used,
            work.complete,
            algorithm_component!("bessel_lambda"),
        )?,
        Err(_) if !work.complete => report(
            ExponentialEstimate {
                value: real(0.0),
                scale: 0.0,
                error: f64::MAX,
            },
            None,
            work.used,
            false,
            algorithm_component!("bessel_lambda"),
        )?,
        Err(error) => return Err(error),
    };
    real_projection(r, algorithm_component!("bessel_lambda"))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the natural logarithm `ln K_ν(x)` of the modified Bessel function of the second kind.

`K_ν(x)` is positive for real `ν` and `x > 0`, so the logarithm is real. It stays representable
where `K_ν(x)` itself overflows (small `x`, large `|ν|`) or underflows (large `x`). The order
may be any finite real number, since `K_(−ν) = K_ν`. Unlike [`bessel_k`](crate::bessel_k), the
function has no limit value at `x = 0`.

Orders with `|ν| ≥ 32` use Debye's uniform expansion (DLMF §10.41) in logarithmic form. For
`x < 10⁻⁸` and `|ν| ≥ 3/2` it uses the leading term `ln(Γ(|ν|)/2) − |ν| ln(x/2)` of DLMF
10.30.2, whose relative error is below `x²/(4(|ν| − 1))`. Other arguments take the logarithm of
the [`bessel_k`](crate::bessel_k) kernel value, with its exponent kept separately.

# Examples

```
use scioxide_special::{bessel_k, bessel_k_log};

// ln K₀(1)
let log_k = bessel_k_log(0.0_f64, 1.0)?;
assert!((log_k - bessel_k(0.0_f64, 1.0)?.ln()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound. The detailed form reports an absolute error estimate
in natural-log units. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`.
- `DomainError` (`OutsideDomain`) for `x < 0`.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>
- N. M. Temme, "On the numerical evaluation of the modified Bessel function of the third kind",
  *J. Comput. Phys.* 19 (1975), 324–337."###)]

    pub bessel_k_log [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: BesselOptions;
    detailed bessel_k_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        bessel_k_log_evaluate(order, x, options, context)
    }
    scoped pub bessel_k_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluates the detailed result before scalar projection.
fn bessel_k_log_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !order.is_finite() || !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and argument must be finite",
        ));
    }
    if x <= S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "x must be positive",
        ));
    }
    let nu = order.abs().try_to_f64_exact()?;
    let a = x.try_to_f64_exact()?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    // Each branch returns ln K and its error in natural-log units.
    fn log_k(nu: f64, a: f64, w: &mut Work<'_, '_, '_>) -> Result<(f64, f64), SciOxideError> {
        if nu >= 32.0 {
            let logarithm =
                crate::bessel_airy::bessel::modified_uniform_log(nu, a, true, false, w)?.0;
            return Ok((
                logarithm,
                REAL_BESSEL_RELATIVE + f64::EPSILON * logarithm.abs(),
            ));
        }
        if a < 1e-8 && nu >= 1.5 {
            w.try_step()?;
            let (g, _) = crate::gamma_beta::gamma::gamma_log_parts(nu)?;
            let power = nu * (RealScalar::ln(a) - core::f64::consts::LN_2);
            // The omitted small-argument correction is below a²/(4(ν−1)).
            return Ok((
                g - core::f64::consts::LN_2 - power,
                f64::EPSILON * (4.0 * g.abs() + 2.0 * power.abs() + 2.0)
                    + a * a / (4.0 * (nu - 1.0)),
            ));
        }
        let k = kernel(Kind::K, nu, real(a), w)?;
        let logarithm = RealScalar::ln(k.value.norm()) + k.scale;
        Ok((
            logarithm,
            k.error / k.value.norm()
                + 32.0 * f64::EPSILON * (1.0 + k.scale.abs())
                + f64::EPSILON * logarithm.abs(),
        ))
    }
    let calculation = log_k(nu, a, &mut w);
    w.check()?;
    let (value, error) = match calculation {
        Ok(v) => v,
        Err(_) if !w.complete => {
            return crate::expansions::bounded(
                S::ZERO,
                crate::result::Bound::derived(S::MAX),
                scioxide_core::algorithm_id!("bessel_k_log", "logarithmic_k", 1),
                w.used,
                false,
            );
        }
        Err(e) => return Err(e),
    };
    let value = S::try_from_f64_lossy(value)?;
    let error = S::try_from_f64_lossy(error)? + S::EPSILON * value.abs();
    crate::expansions::bounded(
        value,
        crate::result::Bound::heuristic(error),
        scioxide_core::algorithm_id!("bessel_k_log", "logarithmic_k", 1),
        w.used,
        true,
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the weighted Bessel integral `∫₀¹ t^μ J_ν(2at) dt`, where `μ` is `exponent` and `ν` is `order`.

The integral converges at `t = 0` when `μ + ν > −1`. Negative integer orders use
`J_(−n) = (−1)^n J_n` and need `μ + |ν| > −1`. Negative `a` requires an integer order and uses
`J_n(−x) = (−1)^n J_n(x)`.

For `a² ≤ 4 (ν + 1)` the integrated power series is summed. Larger `a` uses composite 32-point
Gauss–Legendre panels after the substitution `t = u^q`, which smooths the endpoint at `t = 0`.
Each panel is bisected until two levels agree to `10⁻¹²` times the panel length plus `10⁻¹⁰`
relative. For `μ = ν = 0` the result is `∫₀^(2a) J₀(s) ds / (2a)`. The work is bounded by the
iteration budget of [`BesselIntegralOptions`](crate::BesselIntegralOptions).

# Examples

```
use scioxide_special::{bessel_j, bessel_weighted_integral};

// ∫₀¹ t J₀(t) dt = J₁(1), since (x J₁(x))′ = x J₀(x)
let integral = bessel_weighted_integral(0.5_f64, 1.0, 0.0)?;
assert!((integral - bessel_j(1.0_f64, 1.0)?).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; the detailed form reports the series truncation or
quadrature error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a < 0` and `ν` is not an integer.
- `DomainError` (`Pole`) if the integral diverges at `t = 0` (`μ + |ν| ≤ −1` for integer `ν`,
  `μ + ν ≤ −1` otherwise), or if `a = 0` and `ν` is a negative non-integer.
- `NumericalInstability` (`Breakdown`) if a panel cannot be bisected further before its two
  levels agree.

# References

- NIST DLMF, Chapter 10. <https://dlmf.nist.gov/10>"###)]

    pub bessel_weighted_integral [S: RealScalar] (a: S,
        exponent: S,
        order: S) -> S
    where [];
    options options: crate::BesselIntegralOptions;
    detailed bessel_weighted_integral_with_options -> SpecialFunctionResult<S>;
    value {
        let options = crate::BesselIntegralOptions::default();
        bessel_weighted_integral_evaluate(a, exponent, order, options, None)?.value()
    }
    report {
        bessel_weighted_integral_evaluate(a, exponent, order, options, None)?.publish()
    }
    scoped pub bessel_weighted_integral_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            bessel_weighted_integral_evaluate(a, exponent, order, options, Some(context))
        })?
        .publish()
    }
}

/// Evaluates the detailed result before scalar projection.
fn bessel_weighted_integral_evaluate<S: RealScalar>(
    a: S,
    exponent: S,
    order: S,
    options: crate::BesselIntegralOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !a.is_finite() || !exponent.is_finite() || !order.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "parameters must be finite",
        ));
    }
    let mut nu = order.try_to_f64_exact()?;
    let exponent = exponent.try_to_f64_exact()?;
    let argument = a.try_to_f64_exact()?;
    let integer = nu == RealScalar::round(nu);
    let mut sign = 1.0;
    if argument < 0.0 && !integer {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("a"),
            scioxide_core::DomainReason::OutsideDomain,
            "negative arguments require integer order",
        ));
    }
    if nu < 0.0 && integer {
        sign *= scioxide_core::cos_pi(nu)?;
        nu = -nu;
    }
    if argument.is_sign_negative() {
        sign *= scioxide_core::cos_pi(nu)?;
    }
    let p = scioxide_core::checked_sum(&[exponent, nu, 1.0])?;
    if p <= 0.0 || argument == 0.0 && nu < 0.0 {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("exponent"),
            scioxide_core::DomainReason::Pole,
            "weighted endpoint is not integrable",
        ));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let result = weighted_kernel(nu, argument.abs(), p, &mut work);
    work.check()?;
    let r = match result {
        Ok(v) => report(
            multiply(v, real(sign)),
            None,
            work.used,
            work.complete,
            algorithm_component!("bessel_weighted_integral"),
        )?,
        Err(_) if !work.complete => report(
            ExponentialEstimate {
                value: real(0.0),
                scale: 0.0,
                error: f64::MAX,
            },
            None,
            work.used,
            false,
            algorithm_component!("bessel_weighted_integral"),
        )?,
        Err(e) => return Err(e),
    };
    real_projection(r, algorithm_component!("bessel_weighted_integral"))
}

fn weighted_kernel(
    nu: f64,
    a: f64,
    p: f64,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    w.try_step()?;
    if a == 0.0 {
        return Ok(ExponentialEstimate {
            value: real(if nu == 0.0 { 1.0 / p } else { 0.0 }),
            scale: 0.0,
            error: if nu == 0.0 { f64::EPSILON / p } else { 0.0 },
        });
    }
    if nu == 0.0 && p == 1.0 {
        if a > f64::MAX * 0.5 {
            return Ok(ExponentialEstimate {
                value: real(0.5 / a),
                scale: 0.0,
                error: f64::EPSILON * (0.5 / a),
            });
        }
        if w.remaining == 0 {
            w.try_step()?;
        }
        // The nested integral shares this allowance, so each of its steps is admitted live.
        let r = crate::bessel_airy::bessel_integrals::evaluate_charged(
            2.0 * a,
            crate::bessel_airy::bessel_integrals::Kind::JInt,
            algorithm_component!("bessel_j0_integral"),
            w,
        )?;
        // The integral bound is part of this estimate's error.
        let published = r.publish()?;
        let value = published.value().ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "integral estimate unavailable",
            )
        })?;
        return Ok(ExponentialEstimate {
            value: real(*value.value() / (2.0 * a)),
            scale: 0.0,
            error: value.absolute_error() / (2.0 * a),
        });
    }
    let (gamma, sign) = crate::gamma_beta::gamma::gamma_log_parts(nu + 1.0)?;
    let logarithm = nu * RealScalar::ln(a) - gamma;
    if (a / (nu + 1.0).abs()) * a <= 4.0 {
        let mut term = 1.0;
        let mut sum = term;
        let mut absolute = term;
        let mut k = 0usize;
        loop {
            w.try_step()?;
            k += 1;
            let ks = k as f64;
            term *= -(a / ks) * (a / (nu + ks)) * ((p + 2.0 * ks - 2.0) / (p + 2.0 * ks));
            sum += term;
            absolute += term.abs();
            if term.abs() <= f64::EPSILON * sum.abs() {
                break;
            }
        }
        return Ok(ExponentialEstimate {
            value: real(sign * sum),
            scale: logarithm - RealScalar::ln(p),
            error: term.abs() + 32.0 * f64::EPSILON * absolute,
        });
    }
    let q = if p >= 1.0 && p == RealScalar::floor(p) {
        1.0
    } else {
        RealScalar::ceil(8.0 / p).max(1.0)
    };
    // The endpoint substitution makes the integrand at least C^7. GL32
    // resolves it with far fewer nested Bessel evaluations than composite
    // Simpson refinement, whose repeated coarse grids exhausted default work.
    let count = RealScalar::ceil(a.max(p) * q / 8.0).max(1.0);
    if count > (w.remaining / 96) as f64 {
        w.complete = false;
        return Err(crate::result::exhausted(
            scioxide_core::ConvergenceDiagnostics::new(),
        ));
    }
    let intervals = count.try_to_usize_exact()?;
    let panel = |left: f64, right: f64, w: &mut Work| -> Result<(f64, f64), SciOxideError> {
        let half = 0.5 * (right - left);
        let mid = left + half;
        let mut sum = 0.0;
        let mut absolute = 0.0;
        for &(node, weight) in &crate::support::GL32 {
            w.try_step()?;
            let u = mid + half * node;
            let lu = RealScalar::ln(u);
            let t = RealScalar::exp(q * lu);
            let x = a * (2.0 * t);
            if !x.is_finite() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "internal Bessel argument exceeds the working range",
                ));
            }
            let lambda = lambda_kernel(nu, x, w)?;
            let log_weight = logarithm + RealScalar::ln(q) + (q * p - 1.0) * lu + lambda.scale;
            let value = if lambda.value.re == 0.0 {
                0.0
            } else {
                RealScalar::exp(RealScalar::ln(lambda.value.re.abs()) + log_weight)
                    .copysign(lambda.value.re * sign)
            } * (weight * half);
            sum += value;
            absolute += value.abs();
        }
        Ok((sum, absolute))
    };
    let mut total = 0.0;
    let mut error = 0.0;
    let mut stack = [(0.0, 0.0, 0.0); 64];
    for interval in 0..intervals {
        let left = interval as f64 / intervals as f64;
        let right = (interval + 1) as f64 / intervals as f64;
        stack[0] = (left, right, panel(left, right, w)?.0);
        let mut count = 1;
        while count > 0 {
            count -= 1;
            let (left, right, coarse) = stack[count];
            let mid = left + 0.5 * (right - left);
            let (l, lm) = panel(left, mid, w)?;
            let (r, rm) = panel(mid, right, w)?;
            let fine = l + r;
            let delta = (fine - coarse).abs();
            let bound = delta + 64.0 * f64::EPSILON * (lm + rm);
            if delta <= 1e-12 * (right - left) + 1e-10 * fine.abs() {
                total += fine;
                error += bound;
            } else {
                if count + 2 > stack.len() || mid == left || mid == right {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "weighted Bessel quadrature cannot subdivide further",
                    ));
                }
                stack[count] = (mid, right, r);
                stack[count + 1] = (left, mid, l);
                count += 2;
            }
        }
    }
    Ok(ExponentialEstimate {
        value: real(total),
        scale: 0.0,
        error,
    })
}
pub(crate) fn spherical_hankel_1_complex<S: RealScalar>(
    n: u32,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <crate::SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    spherical_evaluate(
        n,
        z,
        options,
        Kind::H1,
        false,
        algorithm_component!(stringify!(spherical_hankel_1_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_hankel_1_complex_evaluation<S: RealScalar>(
    n: u32,
    z: Complex<S>,
    options: crate::SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    spherical_evaluate_controlled(
        n,
        z,
        options,
        Kind::H1,
        false,
        algorithm_component!(stringify!(spherical_hankel_1_complex)),
        context,
    )
}

pub(crate) fn spherical_hankel_2_complex<S: RealScalar>(
    n: u32,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <crate::SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    spherical_evaluate(
        n,
        z,
        options,
        Kind::H2,
        false,
        algorithm_component!(stringify!(spherical_hankel_2_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_hankel_2_complex_evaluation<S: RealScalar>(
    n: u32,
    z: Complex<S>,
    options: crate::SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    spherical_evaluate_controlled(
        n,
        z,
        options,
        Kind::H2,
        false,
        algorithm_component!(stringify!(spherical_hankel_2_complex)),
        context,
    )
}

pub(crate) fn spherical_hankel_1<S: RealScalar>(
    n: u32,
    x: S,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <crate::SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    spherical_evaluate(
        n,
        Complex::new(x, S::ZERO),
        options,
        Kind::H1,
        false,
        algorithm_component!("spherical_hankel_1"),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_hankel_1_evaluation<S: RealScalar>(
    n: u32,
    x: S,
    options: crate::SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    spherical_evaluate_controlled(
        n,
        Complex::new(x, S::ZERO),
        options,
        Kind::H1,
        false,
        algorithm_component!("spherical_hankel_1"),
        context,
    )
}

pub(crate) fn spherical_hankel_2<S: RealScalar>(
    n: u32,
    x: S,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <crate::SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    spherical_evaluate(
        n,
        Complex::new(x, S::ZERO),
        options,
        Kind::H2,
        false,
        algorithm_component!("spherical_hankel_2"),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_hankel_2_evaluation<S: RealScalar>(
    n: u32,
    x: S,
    options: crate::SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    spherical_evaluate_controlled(
        n,
        Complex::new(x, S::ZERO),
        options,
        Kind::H2,
        false,
        algorithm_component!("spherical_hankel_2"),
        context,
    )
}

pub(crate) fn hankel_1_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    hankel_1_derivative_evaluate(order, x, degree, options, None)?.value()
}

pub(crate) fn hankel_2_derivative<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    hankel_2_derivative_evaluate(order, x, degree, options, None)?.value()
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn hankel_1_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    crate::support::require_finite_param(x, "Bessel parameter x must be finite")?;
    if x <= S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "real argument must be positive",
        ));
    }
    nth_evaluate(
        order,
        Complex::new(x, S::ZERO),
        degree,
        options,
        Kind::H1,
        false,
        algorithm_component!("hankel_1_derivative"),
        context,
    )
}
/// Evaluates the detailed result before scalar projection.
pub(crate) fn hankel_2_derivative_evaluate<S: RealScalar>(
    order: S,
    x: S,
    degree: u32,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    crate::support::require_finite_param(x, "Bessel parameter x must be finite")?;
    if x <= S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "real argument must be positive",
        ));
    }
    nth_evaluate(
        order,
        Complex::new(x, S::ZERO),
        degree,
        options,
        Kind::H2,
        false,
        algorithm_component!("hankel_2_derivative"),
        context,
    )
}
