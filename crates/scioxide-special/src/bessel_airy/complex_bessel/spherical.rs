//! Spherical complex Bessel and Hankel kernels.

use super::*;

fn spherical_small(
    kind: Kind,
    n: u32,
    z: C,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    w.try_step()?;
    let maximum = z.re.abs().max(z.im.abs());
    let u = z.unscale(maximum);
    let norm = u.norm();
    let unit = u.unscale(norm);
    let logarithm = RealScalar::ln(maximum) + RealScalar::ln(norm);
    let ns = f64::from(n);
    let (g1, _) =
        crate::gamma_beta::gamma::gamma_log_parts(if matches!(kind, Kind::J | Kind::I) {
            2.0 * ns + 2.0
        } else {
            2.0 * ns + 1.0
        })?;
    w.try_step()?;
    let (g2, _) = crate::gamma_beta::gamma::gamma_log_parts(ns + 1.0)?;
    let coefficient = g1 - g2 - ns * core::f64::consts::LN_2;
    let (power, scale) = if matches!(kind, Kind::J | Kind::I) {
        (unit.powu(n), ns * logarithm - coefficient)
    } else {
        (
            unit.inv().powu(n) * unit.inv(),
            -(ns + 1.0) * logarithm + coefficient,
        )
    };
    let mut term = real(1.0);
    let mut sum = term;
    let square = z * z;
    let mut k = 0usize;
    let mut omitted = 1.0;
    while w.step() {
        k += 1;
        let ks = k as f64;
        let denominator = 2.0
            * ks
            * if matches!(kind, Kind::J | Kind::I) {
                2.0 * ns + 2.0 * ks + 1.0
            } else {
                2.0 * ks - 2.0 * ns - 1.0
            };
        term = (term * square)
            .scale(if matches!(kind, Kind::J | Kind::Y) {
                -1.0
            } else {
                1.0
            })
            .unscale(denominator);
        sum += term;
        omitted = term.norm();
        if k >= 3 && omitted <= f64::EPSILON * sum.norm() {
            break;
        }
    }
    let result = ExponentialEstimate {
        value: power * sum.scale(if matches!(kind, Kind::Y) { -1.0 } else { 1.0 }),
        scale,
        error: power.norm() * (omitted + 64.0 * f64::EPSILON * (ns + 1.0) * sum.norm()),
    };
    if matches!(kind, Kind::K) {
        let regular = multiply(
            spherical_small(Kind::I, n, z, w)?,
            real(if n.is_multiple_of(2) { -1.0 } else { 1.0 }),
        );
        Ok(multiply(
            add(result, regular),
            real(core::f64::consts::PI / 2.0),
        ))
    } else {
        Ok(result)
    }
}
fn spherical_kernel(
    kind: Kind,
    n: u32,
    z: C,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    if matches!(kind, Kind::H2) {
        let mut value = spherical_kernel(Kind::H1, n, z.conj(), w)?;
        value.value = value.value.conj();
        return Ok(value);
    }
    if matches!(kind, Kind::H1) {
        // On the real axis J and iY occupy orthogonal components. Their real
        // recurrences retain large-order scales without the pessimistic error
        // growth of a K recurrence along the imaginary axis.
        if z.im == 0.0 && z.re != 0.0 {
            return Ok(add(
                spherical_kernel(Kind::J, n, z, w)?,
                multiply(spherical_kernel(Kind::Y, n, z, w)?, C::i()),
            ));
        }
        return Ok(multiply(
            spherical_kernel(Kind::K, n, hankel_argument(z, true), w)?,
            pi_phase(-f64::from(n) * 0.5)?.scale(-2.0 / core::f64::consts::PI),
        ));
    }
    if z == real(0.0) {
        w.try_step()?;
        return if matches!(kind, Kind::J | Kind::I) {
            Ok(ExponentialEstimate {
                value: real(if n == 0 { 1.0 } else { 0.0 }),
                scale: 0.0,
                error: 0.0,
            })
        } else {
            Err(SciOxideError::domain_error(
                scioxide_core::argument!("z"),
                scioxide_core::DomainReason::Pole,
                "the origin is singular",
            ))
        };
    }
    if z.norm() < 0.25 {
        return spherical_small(kind, n, z, w);
    }
    let prefactor = z
        .sqrt()
        .inv()
        .scale(RealScalar::sqrt(core::f64::consts::PI / 2.0));
    Ok(multiply(kernel(kind, f64::from(n) + 0.5, z, w)?, prefactor))
}
pub(super) fn spherical_evaluate<S: RealScalar>(
    n: u32,
    z: Complex<S>,
    options: crate::SphericalBesselOptions,
    kind: Kind,
    scaled: bool,
    family: AlgorithmComponent,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    spherical_evaluate_controlled(n, z, options, kind, scaled, family, None)
}
pub(crate) fn spherical_pair(
    kind: Kind,
    n: u32,
    z: C,
    work: &mut Work,
) -> Result<(ExponentialEstimate, ExponentialEstimate), SciOxideError> {
    let value = spherical_kernel(kind, n, z, work)?;
    let derivative = if z == real(0.0) {
        ExponentialEstimate {
            value: real(if n == 1 { 1.0 / 3.0 } else { 0.0 }),
            scale: 0.0,
            error: if n == 1 { f64::EPSILON / 3.0 } else { 0.0 },
        }
    } else {
        let next = n.checked_add(1).ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("order"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "neighboring order overflow",
            )
        })?;
        let upper = spherical_kernel(kind, next, z, work)?;
        let ns = f64::from(n);
        let mut d = multiply(
            upper,
            real(if matches!(kind, Kind::I) {
                (ns + 1.0) / (2.0 * ns + 1.0)
            } else {
                -(ns + 1.0) / (2.0 * ns + 1.0)
            }),
        );
        if n > 0 {
            d = add(
                d,
                multiply(
                    spherical_kernel(kind, n - 1, z, work)?,
                    real(if matches!(kind, Kind::K) {
                        -ns / (2.0 * ns + 1.0)
                    } else {
                        ns / (2.0 * ns + 1.0)
                    }),
                ),
            );
        }
        d
    };
    Ok((value, derivative))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn spherical_evaluate_controlled<S: RealScalar>(
    n: u32,
    z: Complex<S>,
    options: crate::SphericalBesselOptions,
    kind: Kind,
    scaled: bool,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<BesselEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    if !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "argument must be finite",
        ));
    }
    let argument = C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    fn scaled_spherical_pair(
        kind: Kind,
        n: u32,
        argument: C,
        scaled: bool,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(ExponentialEstimate, ExponentialEstimate), SciOxideError> {
        let (mut value, mut derivative) = spherical_pair(kind, n, argument, work)?;
        if scaled {
            value = scale_function(kind, argument, value);
            derivative = scale_function(kind, argument, derivative);
        }
        Ok((value, derivative))
    }
    let computation = scaled_spherical_pair(kind, n, argument, scaled, &mut work);
    work.check()?;
    match computation {
        Ok((v, d)) => report(v, Some(d), work.used, work.complete, family),
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
pub(crate) fn spherical_jn_complex<S: RealScalar>(
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
        Kind::J,
        false,
        algorithm_component!(stringify!(spherical_jn_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_jn_complex_evaluation<S: RealScalar>(
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
        Kind::J,
        false,
        algorithm_component!(stringify!(spherical_jn_complex)),
        context,
    )
}

pub(crate) fn spherical_yn_complex<S: RealScalar>(
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
        Kind::Y,
        false,
        algorithm_component!(stringify!(spherical_yn_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_yn_complex_evaluation<S: RealScalar>(
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
        Kind::Y,
        false,
        algorithm_component!(stringify!(spherical_yn_complex)),
        context,
    )
}

pub(crate) fn spherical_in_complex<S: RealScalar>(
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
        Kind::I,
        false,
        algorithm_component!(stringify!(spherical_in_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_in_complex_evaluation<S: RealScalar>(
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
        Kind::I,
        false,
        algorithm_component!(stringify!(spherical_in_complex)),
        context,
    )
}

pub(crate) fn spherical_kn_complex<S: RealScalar>(
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
        Kind::K,
        false,
        algorithm_component!(stringify!(spherical_kn_complex)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_kn_complex_evaluation<S: RealScalar>(
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
        Kind::K,
        false,
        algorithm_component!(stringify!(spherical_kn_complex)),
        context,
    )
}

pub(crate) fn spherical_in_complex_scaled<S: RealScalar>(
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
        Kind::I,
        true,
        algorithm_component!(stringify!(spherical_in_complex_scaled)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_in_complex_scaled_evaluation<S: RealScalar>(
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
        Kind::I,
        true,
        algorithm_component!(stringify!(spherical_in_complex_scaled)),
        context,
    )
}

pub(crate) fn spherical_kn_complex_scaled<S: RealScalar>(
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
        Kind::K,
        true,
        algorithm_component!(stringify!(spherical_kn_complex_scaled)),
    )?
    .value()
}

#[inline]
pub(crate) fn spherical_kn_complex_scaled_evaluation<S: RealScalar>(
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
        Kind::K,
        true,
        algorithm_component!(stringify!(spherical_kn_complex_scaled)),
        context,
    )
}

fn nth_kernel(
    kind: Kind,
    nu: f64,
    z: C,
    degree: u32,
    w: &mut Work,
) -> Result<ExponentialEstimate, SciOxideError> {
    if degree == 0 {
        return kernel(kind, nu, z, w);
    }
    if u64::from(degree) > w.remaining as u64 {
        w.complete = false;
        return Err(crate::result::exhausted(
            scioxide_core::ConvergenceDiagnostics::new(),
        ));
    }
    for _ in 0..3 {
        w.try_step()?;
    }
    let mut result = ExponentialEstimate {
        value: real(0.0),
        scale: 0.0,
        error: 0.0,
    };
    for (offset, weight) in derivative_terms(kind, degree)? {
        w.try_step()?;
        result = add(
            result,
            multiply(kernel(kind, nu + offset, z, w)?, real(weight)),
        );
    }
    result.error += result.value.norm()
        * 64.0
        * f64::EPSILON
        * (f64::from(degree) + 1.0)
        * RealScalar::ln(f64::from(degree) + 2.0);
    Ok(result)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn nth_evaluate<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
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
    if imaginary != 0.0 || (z.re != S::ZERO && z.im != S::ZERO) {
        return Ok(crate::bessel_airy::complex_order_bessel::evaluate(
            C::new(order.real_component().try_to_f64_exact()?, imaginary),
            z,
            options,
            (kind, scaled),
            family,
            context,
            Some(degree),
        )?
        .map_form(Either::Right));
    }
    let nu = order.real_component().try_to_f64_exact()?;
    let z = C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let value = nth_kernel(kind, nu, z, degree, &mut work);
    work.check()?;
    match value {
        Ok(value) => report(
            if scaled {
                scale_function(kind, z, value)
            } else {
                value
            },
            None,
            work.used,
            work.complete,
            family,
        ),
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
        Err(error) => Err(error),
    }
}
pub(crate) fn bessel_j_complex_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(bessel_j_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::J, false, family, None)?.value()
}

#[inline]
pub(crate) fn bessel_j_complex_derivative_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(bessel_j_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::J, false, family, context)
}

pub(crate) fn bessel_y_complex_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(bessel_y_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::Y, false, family, None)?.value()
}

#[inline]
pub(crate) fn bessel_y_complex_derivative_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(bessel_y_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::Y, false, family, context)
}

pub(crate) fn bessel_i_complex_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(bessel_i_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::I, false, family, None)?.value()
}

#[inline]
pub(crate) fn bessel_i_complex_derivative_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(bessel_i_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::I, false, family, context)
}

pub(crate) fn bessel_k_complex_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(bessel_k_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::K, false, family, None)?.value()
}

#[inline]
pub(crate) fn bessel_k_complex_derivative_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(bessel_k_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::K, false, family, context)
}

pub(crate) fn hankel_1_complex_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(hankel_1_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::H1, false, family, None)?.value()
}

#[inline]
pub(crate) fn hankel_1_complex_derivative_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(hankel_1_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::H1, false, family, context)
}

pub(crate) fn hankel_2_complex_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(hankel_2_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::H2, false, family, None)?.value()
}

#[inline]
pub(crate) fn hankel_2_complex_derivative_evaluation<S: RealScalar, O: NumericScalar<Real = S>>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(hankel_2_complex_derivative));
    nth_evaluate(order, z, degree, options, Kind::H2, false, family, context)
}

pub(crate) fn bessel_i_complex_scaled_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(bessel_i_complex_scaled_derivative));
    nth_evaluate(order, z, degree, options, Kind::I, true, family, None)?.value()
}

#[inline]
pub(crate) fn bessel_i_complex_scaled_derivative_evaluation<
    S: RealScalar,
    O: NumericScalar<Real = S>,
>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(bessel_i_complex_scaled_derivative));
    nth_evaluate(order, z, degree, options, Kind::I, true, family, context)
}

pub(crate) fn bessel_k_complex_scaled_derivative<S: RealScalar>(
    order: impl NumericScalar<Real = S>,
    z: Complex<S>,
    degree: u32,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: BesselOptions = BesselOptions::default();
    let family = algorithm_component!(stringify!(bessel_k_complex_scaled_derivative));
    nth_evaluate(order, z, degree, options, Kind::K, true, family, None)?.value()
}

#[inline]
pub(crate) fn bessel_k_complex_scaled_derivative_evaluation<
    S: RealScalar,
    O: NumericScalar<Real = S>,
>(
    order: O,
    z: Complex<S>,
    degree: u32,
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
    let family = algorithm_component!(stringify!(bessel_k_complex_scaled_derivative));
    nth_evaluate(order, z, degree, options, Kind::K, true, family, context)
}
