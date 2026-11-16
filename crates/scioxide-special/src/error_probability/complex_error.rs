//! Entire error-function continuations sharing the Faddeeva and real erfcx kernels.
//! DLMF 7.2.1–3; analytic coordinate derivatives and explicit logarithmic scale.
use crate::complex_support::{RoundedZero, ZeroOrScaled};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, real};

use crate::ErrorFunctionOptions;
use crate::result::{Bound, Either, Evaluation, Ulps};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, Budget, Complex, NumericScalar, RealScalar, SciOxideError,
    SolveContext, algorithm_component,
};
const FACTOR: f64 = core::f64::consts::FRAC_2_SQRT_PI;
/// Output envelope of the direct real erf and erfc kernels, checked against reference values.
const REAL_ULPS: Ulps = Ulps::new(4.0, 1.0);
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Erf,
    Erfc,
    Erfcx,
    Erfi,
    Dawson,
    LogErfc,
}

fn scale(z: C, f: f64) -> C {
    C::new(z.re * f, z.im * f)
}
fn times_i(z: C) -> C {
    C::new(-z.im, z.re)
}
fn minus_i(z: C) -> C {
    C::new(z.im, -z.re)
}
/// exp(-z²): keep product residuals in the exponent and phase.
#[inline]
pub(super) fn gaussian(z: C) -> Result<(crate::support::wide::Wide, C), SciOxideError> {
    // Preserve exponent rounding as well as the exact product phase. The
    // difference-of-products check also handles cancellation at extreme inputs.
    let exponent = if (z.re * z.re).is_finite() && (z.im * z.im).is_finite() {
        crate::support::wide::two_product(z.im, z.im)
            - crate::support::wide::two_product(z.re, z.re)
    } else {
        crate::support::wide::Wide::from(scioxide_core::checked_difference_of_products(
            z.im, z.im, z.re, z.re,
        )?)
    };
    crate::support::check_finite(exponent.hi)?;
    let product = z.re * z.im;
    if !product.is_finite() {
        return Err(crate::support::instability(
            "Gaussian phase exceeds the working scalar range",
        ));
    }
    if product == 0.0 {
        return Ok((exponent, C::new(1.0, -product)));
    }
    let residual = RealScalar::mul_add(z.re, z.im, -product);
    let (s, c) = RealScalar::sin_cos(product);
    let (se, ce) = if residual == 0.0 {
        (0.0, 1.0)
    } else {
        RealScalar::sin_cos(residual)
    };
    let half = C::new(c * ce - s * se, -(s * ce + c * se));
    Ok((exponent, half * half))
}
fn small_erf(z: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let square = z * z;
    let mut term = z;
    let mut value = z;
    let mut magnitude = z.norm();
    let mut tail = 0.0;
    for k in 1_usize.. {
        if !work.poll_step()? {
            return Ok(crate::support::interrupted());
        }
        let n = k as f64;
        term = scale(term * (-square), (2.0 * n - 1.0) / (n * (2.0 * n + 1.0)));
        value += term;
        magnitude += term.norm();
        let ratio = square.norm() / (n + 1.0);
        tail = term.norm() * ratio / (1.0 - ratio);
        if k >= 2 && tail <= f64::EPSILON * value.norm() {
            break;
        }
    }
    let derivative = scale((-square).exp(), FACTOR);
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: scale(value, FACTOR),
            d: derivative,
        },
        scale: Jet::constant(real(0.0)),
        error: FACTOR * (tail + 8.0 * f64::EPSILON * (work.used as f64 + 1.0) * magnitude),
        derivative_error: Some(8.0 * f64::EPSILON * derivative.norm()),
        method: algorithm_component!("erf_power_series"),
    }))
}
/// Right-sector erfcx / real-sector scaled-erfi Laurent expansion. The derivative is summed
/// directly, avoiding 2*z*value minus its constant limit.
fn laurent(z: C, alternating: bool, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let inverse = real(1.0) / z;
    let square = inverse * inverse;
    let mut term = real(1.0);
    let mut value = term;
    let mut derivative = term;
    let mut magnitude = 1.0;
    let mut derivative_magnitude = 1.0;
    let mut prior = 1.0;
    let mut k = 0_usize;
    let (tail, derivative_tail) = loop {
        if !work.poll_step()? {
            return Ok(crate::support::interrupted());
        }
        k += 1;
        let factor = (2.0 * k as f64 - 1.0) * if alternating { -0.5 } else { 0.5 };
        term = scale(term * square, factor);
        let size = term.norm();
        if size > prior {
            return Err(crate::support::instability(
                "Laurent derivative cannot resolve its asymptotic tail",
            ));
        }
        value += term;
        derivative += scale(term, 2.0 * k as f64 + 1.0);
        magnitude += size;
        derivative_magnitude += (2.0 * k as f64 + 1.0) * size;
        let ratio = square.norm() * (k as f64 + 0.5);
        let tail = size * ratio / (1.0 - ratio);
        let dtail = tail * (2.0 * k as f64 + 3.0) / (1.0 - ratio);
        if dtail <= f64::EPSILON * derivative.norm() {
            break (tail, dtail);
        }
        prior = size;
    };
    let prefactor = scale(inverse, 0.5 * FACTOR);
    let dprefactor = -prefactor * inverse;
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: prefactor * value,
            d: dprefactor * derivative,
        },
        scale: Jet::constant(real(0.0)),
        error: prefactor.norm() * (tail + 16.0 * f64::EPSILON * (k + 1) as f64 * magnitude),
        derivative_error: Some(
            dprefactor.norm()
                * (derivative_tail + 16.0 * f64::EPSILON * (k + 1) as f64 * derivative_magnitude),
        ),
        method: algorithm_component!("scaled_erfi_laurent"),
    }))
}
pub(crate) fn base(z: C, kind: Kind, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if !work.poll_step()? {
        return Ok(crate::support::interrupted());
    }
    if z.im == 0.0 && matches!(kind, Kind::Erf | Kind::Erfc) {
        let v = if matches!(kind, Kind::Erf) {
            crate::error_probability::error_fn::erf(z.re)?
        } else {
            crate::error_probability::error_fn::erfc(z.re)?
        };
        let d = FACTOR * (-z.re * z.re).exp() * if matches!(kind, Kind::Erf) { 1.0 } else { -1.0 };
        if v != 0.0 || matches!(kind, Kind::Erf) {
            return Ok(Estimate::from(crate::support::JetData {
                y: Jet {
                    v: C::new(
                        v,
                        if matches!(kind, Kind::Erf) {
                            z.im
                        } else {
                            -z.im
                        },
                    ),
                    d: real(d),
                },
                scale: Jet::constant(real(0.0)),
                error: 64.0 * f64::EPSILON * v.abs(),
                derivative_error: Some(16.0 * f64::EPSILON * d.abs()),
                method: algorithm_component!("real_axis_cephes"),
            }));
        }
    }
    if z.norm() <= 1.0 {
        let mut e = small_erf(z, work)?;
        if !work.complete {
            return Ok(crate::support::interrupted());
        }
        if matches!(kind, Kind::Erfc | Kind::Erfcx) {
            e.y.v = C::new(1.0 - e.y.v.re, -e.y.v.im);
            e.y.d = -e.y.d;
            e.error += 2.0 * f64::EPSILON * e.y.v.norm();
        }
        if matches!(kind, Kind::Erfcx) {
            let factor = (z * z).exp();
            let before = e.y.v;
            e.y.v *= factor;
            e.y.d = (e.y.d + scale(z * before, 2.0)) * factor;
            e.error = (e.error + 8.0 * f64::EPSILON * before.norm()) * factor.norm();
            e.derivative_error = Some(
                e.derivative_error.unwrap_or(0.0) * factor.norm()
                    + 2.0 * z.norm() * e.error
                    + 16.0 * f64::EPSILON * e.y.d.norm(),
            );
        }
        return Ok(e);
    }
    let reflected = z.re < 0.0;
    let positive = if reflected { -z } else { z };
    let w = crate::error_probability::faddeeva::upper(times_i(positive), work)?;
    if !work.complete {
        return Ok(crate::support::interrupted());
    }
    let we = 64.0 * f64::EPSILON * w.norm();
    if matches!(kind, Kind::Erfcx) && !reflected {
        if z.norm() > 16.0 {
            let mut e = laurent(z, true, work)?;
            e.y.v = w;
            e.error = we;
            e.method = algorithm_component!("erfcx_faddeeva_laurent_derivative");
            return Ok(e);
        }
        let product = z * w;
        let derivative = scale(product, 2.0) - real(FACTOR);
        return Ok(Estimate::from(crate::support::JetData {
            y: Jet {
                v: w,
                d: derivative,
            },
            scale: Jet::constant(real(0.0)),
            error: we,
            derivative_error: Some(128.0 * f64::EPSILON * (2.0 * product.norm() + FACTOR)),
            method: algorithm_component!("erfcx_faddeeva"),
        }));
    }
    let (g, phase) = gaussian(z)?;
    let g = g.value();
    let exponent = if matches!(kind, Kind::Erfcx) { -g } else { g };
    let phase = if matches!(kind, Kind::Erfcx) {
        phase.conj()
    } else {
        phase
    };
    let carrier = if matches!(kind, Kind::Erfc) && !reflected {
        exponent
    } else {
        exponent.max(0.0)
    };
    let exponential = scale(phase, (exponent - carrier).exp());
    // The carried exponent is published as the scale, whose rounding the report
    // adds in log units; the mantissa only sees the uncarried exponent. The
    // Gaussian phase comes from an exact compensated product.
    let amplification = 1.0 + (exponent - carrier).abs();
    let one = if !matches!(kind, Kind::Erfc) || reflected || z.re == 0.0 {
        (-carrier).exp()
    } else {
        0.0
    };
    let (value, derivative, error, derivative_error) = if matches!(kind, Kind::Erfcx) {
        let value = scale(exponential, 2.0) - scale(w, one);
        let product = z * value;
        let derivative = scale(product, 2.0) - real(FACTOR * one);
        let error = we * one
            + 64.0 * f64::EPSILON * amplification * (2.0 * exponential.norm() + w.norm() * one);
        (
            value,
            derivative,
            error,
            128.0 * f64::EPSILON * amplification * (2.0 * product.norm() + FACTOR * one),
        )
    } else {
        let tail = w * exponential;
        let mut value = match kind {
            Kind::Erf if reflected => tail - real(one),
            Kind::Erf => real(one) - tail,
            _ if reflected => real(2.0 * one) - tail,
            _ => tail,
        };
        // Entire-function axis identities preserve small orthogonal components.
        if z.re == 0.0 {
            value.re = if matches!(kind, Kind::Erf) { z.re } else { one };
        }
        let derivative = scale(
            exponential,
            if matches!(kind, Kind::Erf) {
                FACTOR
            } else {
                -FACTOR
            },
        );
        let constant = if matches!(kind, Kind::Erf) {
            one
        } else if reflected {
            2.0 * one
        } else {
            0.0
        };
        let error = we * exponential.norm()
            + 64.0 * f64::EPSILON * amplification * tail.norm()
            + 8.0 * f64::EPSILON * constant;
        (
            value,
            derivative,
            error,
            64.0 * f64::EPSILON * amplification * derivative.norm(),
        )
    };
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: value,
            d: derivative,
        },
        scale: Jet::constant(real(carrier)),
        error,
        derivative_error: Some(derivative_error),
        method: algorithm_component!("faddeeva_entire_continuation"),
    }))
}
fn logarithmic_erfc(z: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if z.norm() <= 0.5 {
        let e = base(z, Kind::Erf, work)?;
        if !work.complete {
            return Ok(crate::support::interrupted());
        }
        let denominator = real(1.0) - e.y.v;
        let value = scioxide_core::xlog1py(real(1.0), -e.y.v)?;
        let derivative = -e.y.d / denominator;
        let relative = e.error / denominator.norm();
        return Ok(Estimate::from(crate::support::JetData {
            y: Jet {
                v: value,
                d: derivative,
            },
            scale: Jet::constant(real(0.0)),
            error: relative + 16.0 * f64::EPSILON * value.norm(),
            derivative_error: Some(
                e.derivative_error.unwrap_or(0.0) / denominator.norm()
                    + derivative.norm() * (relative + 16.0 * f64::EPSILON),
            ),
            method: algorithm_component!("log_erfc_log1p"),
        }));
    }
    let e = base(z, if z.re >= 0.0 { Kind::Erfcx } else { Kind::Erfc }, work)?;
    if !work.complete {
        return Ok(crate::support::interrupted());
    }
    let norm = e.y.v.norm();
    if norm == 0.0 {
        return Err(crate::support::instability(
            "erfc logarithm cannot resolve its value near a zero",
        ));
    }
    let relative = e.error / norm;
    let (value, derivative, derivative_error) = if z.re >= 0.0 {
        let (g, phase) = gaussian(z)?;
        let g = g.value();
        let unit = e.y.v.unscale(norm) * phase;
        if unit.re < 0.0 && unit.im.abs() <= 128.0 * f64::EPSILON {
            return Err(crate::support::instability(
                "principal erfc logarithm cut side is unresolved",
            ));
        }
        let derivative = -real(FACTOR) / e.y.v;
        (
            C::new(norm.ln() + g, unit.arg()),
            derivative,
            derivative.norm() * (relative + 16.0 * f64::EPSILON),
        )
    } else {
        let unit = e.y.v.unscale(norm);
        let phase_error = 128.0 * f64::EPSILON * (1.0 + 2.0 * (-e.scale.v.re).exp() / norm);
        if unit.re < 0.0 && unit.im.abs() <= phase_error {
            return Err(crate::support::instability(
                "principal erfc logarithm cut side is unresolved",
            ));
        }
        let derivative = e.y.d / e.y.v;
        (
            e.y.v.ln() + e.scale.v,
            derivative,
            e.derivative_error.unwrap_or(0.0) / norm
                + derivative.norm() * (relative + 16.0 * f64::EPSILON),
        )
    };
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: value,
            d: derivative,
        },
        scale: Jet::constant(real(0.0)),
        error: relative + 16.0 * f64::EPSILON * value.norm(),
        derivative_error: Some(derivative_error),
        method: algorithm_component!("principal_log_erfc"),
    }))
}
pub(crate) fn evaluate<T: NumericScalar>(
    z: T,
    kind: Kind,
    budget: Budget,
    family: AlgorithmComponent,
    parent: Option<&mut SolveContext<'_, '_>>,
) -> Result<Evaluation<T, ZeroOrScaled>, SciOxideError> {
    if !z.has_finite_components() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "error-function input must be finite",
        ));
    }
    let z = C::new(
        z.real_component().try_to_f64_exact()?,
        z.imaginary_component().try_to_f64_exact()?,
    );
    // Real erfc beyond x² overflow lies below every scalar, scaled and
    // logarithmic representation: its rounded zero, with the conjugate-
    // symmetric imaginary sign.
    if matches!(kind, Kind::Erfc) && z.im == 0.0 && z.re > 0.0 && !(z.re * z.re).is_finite() {
        return Ok(Evaluation::from_parts(
            Some(T::from_components(
                <T::Real as scioxide_core::AlgebraicScalar>::ZERO,
                <T::Real as scioxide_core::AlgebraicScalar>::ZERO
                    .copysign(T::Real::try_from_f64_lossy(-z.im)?),
            )),
            crate::ScalarRepresentation::Finite,
            crate::SpecialFunctionTermination::Converged,
            scioxide_core::EvaluationCounts::new(),
            AlgorithmId::from_components(family, algorithm_component!("real_axis_underflow"), 1),
            Some(Either::Right(RoundedZero)),
        ));
    }
    let mut work = Work::with_context(budget.get(), parent);
    fn error_estimate(
        kind: Kind,
        z: C,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        if z == real(0.0) {
            if !work.poll_step()? {
                return Ok(crate::support::interrupted());
            }
            let complement = matches!(kind, Kind::Erfc | Kind::Erfcx);
            return Ok(Estimate::from(crate::support::JetData {
                y: Jet {
                    v: if matches!(kind, Kind::LogErfc) {
                        -z
                    } else if complement {
                        C::new(1.0, -z.im)
                    } else {
                        z
                    },
                    d: real(if complement || matches!(kind, Kind::LogErfc) {
                        -FACTOR
                    } else {
                        FACTOR
                    }),
                },
                scale: Jet::constant(real(0.0)),
                error: 0.0,
                derivative_error: Some(8.0 * f64::EPSILON * FACTOR),
                method: algorithm_component!("origin_value"),
            }));
        }
        if matches!(kind, Kind::LogErfc) {
            return logarithmic_erfc(z, work);
        }
        if matches!(kind, Kind::Dawson) && z.re.abs() > 16.0 && z.im.abs() <= 0.5 * z.re.abs() {
            return laurent(z, false, work);
        }
        if matches!(kind, Kind::Erfi | Kind::Dawson) {
            let mut e = base(times_i(z), Kind::Erf, work)?;
            if !work.complete {
                return Ok(crate::support::interrupted());
            }
            e.y.v = minus_i(e.y.v);
            // (-i)*i=1 for the coordinate derivative.
            if matches!(kind, Kind::Dawson) {
                let (g, phase) = gaussian(z)?;
                let g = g.value();
                e.scale.v.re += g;
                e.y.d = (e.y.d - scale(z * e.y.v, 2.0)) * phase;
                e.y.v *= phase;
                e.derivative_error = Some(
                    e.derivative_error.unwrap_or(0.0)
                        + 2.0 * z.norm() * e.error
                        + 64.0 * f64::EPSILON * e.y.d.norm(),
                );
            }
            Ok(e)
        } else {
            base(z, kind, work)
        }
    }
    let result = error_estimate(kind, z, &mut work).map(|mut e| {
        if matches!(kind, Kind::Dawson) {
            let factor = core::f64::consts::PI.sqrt() * 0.5;
            e.y.v = scale(e.y.v, factor);
            e.y.d = if z == real(0.0) {
                real(1.0)
            } else {
                scale(e.y.d, factor)
            };
            e.error = factor * e.error + 2.0 * f64::EPSILON * e.y.v.norm();
            e.derivative_error = e
                .derivative_error
                .map(|error| factor * error + 2.0 * f64::EPSILON * e.y.d.norm());
        }
        e
    });
    work.check()?;
    let derivative = result.as_ref().is_ok_and(|e| e.y.d != real(0.0));
    Ok(crate::complex_support::report_with_status(
        result,
        derivative,
        family,
        work.used,
        !work.complete,
    )?
    .map_form(Either::Left))
}
pub(crate) fn erf_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <ErrorFunctionOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        z,
        Kind::Erf,
        options.iteration_budget(),
        algorithm_component!(stringify!(erf_complex)),
        context,
    )?
    .value()
}

#[inline]
pub(crate) fn erf_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Erf,
        options.iteration_budget(),
        algorithm_component!(stringify!(erf_complex)),
        context,
    )
}

pub(crate) fn erfc_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <ErrorFunctionOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        z,
        Kind::Erfc,
        options.iteration_budget(),
        algorithm_component!(stringify!(erfc_complex)),
        context,
    )?
    .value()
}

#[inline]
pub(crate) fn erfc_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Erfc,
        options.iteration_budget(),
        algorithm_component!(stringify!(erfc_complex)),
        context,
    )
}

pub(crate) fn erfcx_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <ErrorFunctionOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        z,
        Kind::Erfcx,
        options.iteration_budget(),
        algorithm_component!(stringify!(erfcx_complex)),
        context,
    )?
    .value()
}

#[inline]
pub(crate) fn erfcx_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Erfcx,
        options.iteration_budget(),
        algorithm_component!(stringify!(erfcx_complex)),
        context,
    )
}

pub(crate) fn erfi_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <ErrorFunctionOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        z,
        Kind::Erfi,
        options.iteration_budget(),
        algorithm_component!(stringify!(erfi_complex)),
        context,
    )?
    .value()
}

#[inline]
pub(crate) fn erfi_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Erfi,
        options.iteration_budget(),
        algorithm_component!(stringify!(erfi_complex)),
        context,
    )
}

#[inline]
pub(crate) fn dawson_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <ErrorFunctionOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        z,
        Kind::Dawson,
        options.iteration_budget(),
        algorithm_component!(stringify!(dawson_complex)),
        context,
    )?
    .value()
}

#[inline]
pub(crate) fn dawson_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Dawson,
        options.iteration_budget(),
        algorithm_component!(stringify!(dawson_complex)),
        context,
    )
}

pub(crate) fn log_erfc_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <ErrorFunctionOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(
        z,
        Kind::LogErfc,
        options.iteration_budget(),
        algorithm_component!(stringify!(log_erfc_complex)),
        context,
    )?
    .value()
}

#[inline]
pub(crate) fn log_erfc_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::LogErfc,
        options.iteration_budget(),
        algorithm_component!(stringify!(log_erfc_complex)),
        context,
    )
}

/// Real erf or erfc from the direct fixed-cost kernel the plain functions use.
///
/// The value is the plain value bit for bit, with the checked output envelope; the analytic
/// derivative carries its exponent-rounding bound. A far erfc tail that rounds to zero
/// remains a value.
pub(crate) fn real_error_function<S: RealScalar>(
    x: S,
    complement: bool,
    budget: Budget,
) -> Result<Evaluation<S>, SciOxideError> {
    let (value, algorithm) = if complement {
        (
            crate::error_probability::error_fn::erfc(x)?,
            scioxide_core::algorithm_id!("erfc", "fixed_rational", 1),
        )
    } else {
        (
            crate::error_probability::error_fn::erf(x)?,
            scioxide_core::algorithm_id!("erf", "fixed_rational", 1),
        )
    };
    let mut work = Work::new(budget.get());
    if !work.poll_step()? {
        return Err(crate::result::exhausted_work(work.used));
    }
    let x = x.try_to_f64_exact()?;
    let slope = FACTOR * (-x * x).exp();
    let derivative = S::try_from_f64_lossy(if complement { -slope } else { slope })?;
    // Rounding x² perturbs the exponent by up to |x|² ulps.
    let exponent_rounding = if slope == 0.0 {
        0.0
    } else {
        slope * (x * x + 2.0) * f64::EPSILON
    };
    let bound = Bound::derived(S::try_from_f64_lossy(exponent_rounding)?)
        .plus(Bound::rounded(derivative))
        .with_subnormal_rounding(derivative.abs(), S::ONE);
    Ok(Evaluation::finite(
        value,
        Bound::envelope(value, REAL_ULPS)?,
        Some((derivative, bound)),
        algorithm,
        work.used,
        crate::SpecialFunctionTermination::Converged,
    ))
}
