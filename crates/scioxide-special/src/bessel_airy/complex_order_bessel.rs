//! Complex-order cylindrical functions from adjacent modified Bessel solutions.
//! Small arguments use defining/Temme series, moderate arguments use Steed and
//! Wronskian-normalized Miller recurrences, and large arguments use Hankel
//! expansions. Large orders outside the Hankel band use Debye's K expansion,
//! whose cost depends on neither the order nor the argument; beyond the
//! turning circle I combines the two Debye K rays instead of a Miller
//! recurrence of depth about `|z|`. DLMF 10.27/10.34 connections preserve
//! principal signed-zero lips.
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, ZERO, real};
mod seeds;
use crate::BesselOptions;
use crate::bessel_airy::complex_bessel_common::{Kind, hankel_argument};
use scioxide_core::{AlgorithmComponent, algorithm_component};
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError, SolveContext};
const PI: f64 = core::f64::consts::PI;
fn breakdown() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::IllConditioned,
        "complex Bessel recurrence encountered a non-finite value or singular normalization",
    )
}
fn weighted(mut e: Estimate, f: C) -> Estimate {
    e.y = e.y * Jet::constant(f);
    e.error *= f.norm();
    e.derivative_error = e.derivative_error.map(|x| x * f.norm());
    e
}
fn combine(a: Estimate, b: Estimate) -> Estimate {
    let common = a.scale.v.re.max(b.scale.v.re);
    let project = |e: Estimate| {
        let factor = C::new((e.scale.v.re - common).exp(), 0.0)
            * C::new(e.scale.v.im.cos(), e.scale.v.im.sin());
        let y = Jet {
            v: e.y.v * factor,
            d: (e.y.d + e.scale.d * e.y.v) * factor,
        };
        let error =
            e.error * factor.norm() + 32.0 * f64::EPSILON * (1.0 + e.scale.v.im.abs()) * y.v.norm();
        let derror = (e.derivative_error.unwrap_or(e.error) + e.error * e.scale.d.norm())
            * factor.norm()
            + 32.0 * f64::EPSILON * (1.0 + e.scale.v.im.abs()) * y.d.norm();
        (y, error, derror)
    };
    let (a, ae, ad) = project(a);
    let (b, be, bd) = project(b);
    Estimate::from(crate::support::JetData {
        y: a + b,
        scale: Jet::constant(real(common)),
        error: ae + be + 8.0 * f64::EPSILON * (a.v.norm() + b.v.norm()),
        derivative_error: Some(ad + bd + 8.0 * f64::EPSILON * (a.d.norm() + b.d.norm())),
        method: algorithm_component!("complex_order_bessel_connection"),
    })
}
/// Multiply by `exp(i pi turns nu)`; the real growth enters the scale. Beside
/// a small scale phase the order phase is added to it, costing the `O(eps)`
/// that [`combine`] charges per unit of phase. Beside a large argument phase
/// that sum would cost `O(eps |z|)`, so [`rotate`] applies the unit factor to
/// the value instead.
#[inline]
fn turn(e: &mut Estimate, nu: C, turns: f64) {
    e.scale.v.re -= PI * turns * nu.im;
    let half_turns = turns * nu.re;
    if e.scale.v.im.abs() <= 64.0 {
        e.scale.v.im += PI * half_turns;
    } else {
        rotate(e, half_turns);
    }
}
/// Multiply by `exp(i pi t)`, reducing `t` exactly to `[-1, 1]` first.
#[cold]
#[inline(never)]
fn rotate(e: &mut Estimate, t: f64) {
    let reduced = t - 2.0 * (t * 0.5).round();
    let (sine, cosine) = (PI * reduced).sin_cos();
    e.y = e.y * Jet::constant(C::new(cosine, sine));
    e.error += 8.0 * f64::EPSILON * e.y.v.norm();
    e.derivative_error = e
        .derivative_error
        .map(|error| error + 8.0 * f64::EPSILON * e.y.d.norm());
}
/// `1` on the upper principal lip and half-plane, `-1` on the lower ones.
fn half(z: C) -> f64 {
    if z.im.is_sign_negative() { -1.0 } else { 1.0 }
}
fn first(nu: C, z: Jet, shift: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if z.v.re < 0.0 {
        let mut e = first(nu, -z, shift, work)?;
        turn(&mut e, nu, half(z.v));
        return Ok(e);
    }
    let rotation = C::new(0.0, -half(z.v));
    let mut e = modified_i(nu, z * Jet::constant(rotation), shift, work)?;
    turn(&mut e, nu, half(z.v) / 2.0);
    Ok(e)
}

/// DLMF 10.34.2, using independent adjacent K solutions on both rays.
fn i_connection(nu: C, z: Jet, shift: C, a: seeds::Pair, b: seeds::Pair) -> Estimate {
    let side = -half(z.v);
    let mut a = a.estimate(nu, z, shift, false);
    turn(&mut a, nu, -side);
    let b = b.estimate(nu, -z, shift, false);
    weighted(
        combine(a, weighted(b, real(-1.0))),
        C::new(0.0, -1.0 / (side * PI)),
    )
}

fn modified_i(nu: C, z: Jet, shift: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if z.v.re < 0.0 {
        let mut e = modified_i(nu, -z, shift, work)?;
        turn(&mut e, nu, half(z.v));
        return Ok(e);
    }
    let reflected = nu.re < 0.0;
    let nu = if reflected { -nu } else { nu };
    let (first, normalization) = if z.v.norm() <= 2.0 {
        (
            seeds::i_series(nu, z.v, work)?.estimate(nu, z, shift, true),
            None,
        )
    } else if let Some(a) = seeds::asymptotic(nu, z.v, work)?
        && let Some(b) = seeds::asymptotic(nu, -z.v, work)?
    {
        // Both exponentials are needed on the imaginary axis. Pick the
        // half-turn which stays on the principal sheet (DLMF 10.34.2).
        (i_connection(nu, z, shift, a, b), None)
    } else {
        let k = seeds::k_pair(nu, z.v, work)?;
        // Beyond the turning circle near the imaginary axis I oscillates and
        // Miller's depth grows with |z|. Debye's adjacent K values there are
        // cheap and their DLMF 10.34.2 combination does not cancel.
        let oscillatory = z.v.re <= z.v.im.abs() && !seeds::turning(nu, z.v);
        let first = if oscillatory && let Some(b) = seeds::debye(nu, -z.v, work)? {
            i_connection(nu, z, shift, k, b)
        } else {
            match seeds::i_pair(nu, z.v, k, work)? {
                Some(pair) => pair.estimate(nu, z, shift, true),
                None => i_connection(nu, z, shift, k, seeds::k_pair(nu, -z.v, work)?),
            }
        };
        (first, Some(k))
    };
    if !reflected {
        return Ok(first);
    }
    // The same adjacent K pair normalizes Miller and supplies the reflected
    // order's connection. Reusing it charges only the work actually performed.
    let second = match normalization {
        Some(k) => k.estimate(nu, z, shift, false),
        None => modified_k(nu, z, shift, work)?,
    };
    Ok(combine(
        first,
        weighted(second, scioxide_core::sin_pi(nu)? * (2.0 / PI)),
    ))
}

fn modified_k(nu: C, z: Jet, shift: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let nu = if nu.re < 0.0 { -nu } else { nu };
    // Temme's logarithmic seeds hold on the whole principal sheet. CF2 also
    // continues across the imaginary axis; use it directly through |arg z|<=3pi/4
    // to avoid cancelling K and I when the complex order makes K recessive.
    // Large orders inside the turning circle reflect the whole left
    // half-plane: there CF2's forward recurrence loses accuracy and Debye's
    // expansion does not apply, while the reflected K dominates I.
    if z.v.norm() > 2.0 && (z.v.re < -z.v.im.abs() || (z.v.re < 0.0 && seeds::turning(nu, z.v))) {
        let side = half(z.v);
        let mut k = modified_k(nu, -z, shift, work)?;
        turn(&mut k, nu, -side);
        let i = modified_i(nu, -z, shift, work)?;
        return Ok(combine(k, weighted(i, C::new(0.0, -side * PI))));
    }
    Ok(seeds::k_pair(nu, z.v, work)?.estimate(nu, z, shift, false))
}

fn wave(kind: Kind, nu: C, z: Jet, shift: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let is_first = matches!(kind, Kind::H1);
    let sign = if is_first { 1.0 } else { -1.0 };
    // H1_-nu = exp(i*pi*nu) H1_nu, H2_-nu = exp(-i*pi*nu) H2_nu.
    // Reflect before a coordinate connection; otherwise the two terms
    // can cancel by exp(2*pi*|Im nu|) even at small arguments. The phase
    // turns each term before any connection projects it.
    let order = if nu.re < 0.0 { -nu } else { nu };
    let reflect = |e: &mut Estimate| {
        if nu.re < 0.0 {
            turn(e, nu, -sign);
        }
    };
    if z.v.re < 0.0 && z.v.im.is_sign_negative() == is_first {
        let mut j = first(order, z, shift, work)?;
        reflect(&mut j);
        let other = if is_first { Kind::H2 } else { Kind::H1 };
        let mut other = wave(other, order, z, shift, work)?;
        reflect(&mut other);
        return Ok(combine(weighted(j, real(2.0)), weighted(other, real(-1.0))));
    }
    let rotation = C::new(0.0, -sign);
    let argument = Jet {
        v: hankel_argument(z.v, is_first),
        d: z.d * rotation,
    };
    let mut e = modified_k(order, argument, shift, work)?;
    turn(&mut e, order, -sign / 2.0);
    reflect(&mut e);
    Ok(weighted(e, C::new(0.0, -sign * 2.0 / PI)))
}
fn second(nu: C, z: Jet, shift: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if nu.im == 0.0 && nu.re < 0.0 {
        // Reflect real orders before combining waves (DLMF 10.4).
        // At negative half-orders Y is a small J solution; subtracting the
        // two individually large Hankel waves would destroy it near zero.
        let order = -nu;
        let (sine, cosine) = scioxide_core::sin_cos_pi(order.re)?;
        if cosine == 0.0 {
            return Ok(weighted(first(order, z, shift, work)?, real(sine)));
        }
        let y = weighted(second(order, z, shift, work)?, real(cosine));
        return if sine == 0.0 {
            Ok(y)
        } else {
            Ok(combine(
                y,
                weighted(first(order, z, shift, work)?, real(sine)),
            ))
        };
    }
    Ok(weighted(
        combine(
            wave(Kind::H1, nu, z, shift, work)?,
            weighted(wave(Kind::H2, nu, z, shift, work)?, real(-1.0)),
        ),
        C::new(0.0, -0.5),
    ))
}

pub(crate) fn kernel_value(
    kind: Kind,
    nu: C,
    z: Jet,
    shift: C,
    work: &mut Work,
) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    match kind {
        Kind::J => first(nu, z, shift, work),
        Kind::K => modified_k(nu, z, shift, work),
        Kind::I => modified_i(nu, z, shift, work),
        Kind::H1 | Kind::H2 => wave(kind, nu, z, shift, work),
        Kind::Y => second(nu, z, shift, work),
    }
}
fn nth(
    kind: Kind,
    nu: C,
    z: C,
    shift: C,
    degree: u32,
    work: &mut Work,
) -> Result<Estimate, SciOxideError> {
    if degree == 0 {
        return kernel_value(kind, nu, Jet::constant(z), shift, work);
    }
    for _ in 0..3 {
        work.try_step()?;
    }
    let mut sum = None;
    for (offset, weight) in
        crate::bessel_airy::complex_bessel_common::derivative_terms(kind, degree)?
    {
        work.try_step()?;
        let term = weighted(
            kernel_value(kind, nu + real(offset), Jet::constant(z), shift, work)?,
            real(weight),
        );
        sum = Some(if let Some(previous) = sum {
            combine(previous, term)
        } else {
            term
        });
    }
    let mut e = sum.ok_or_else(breakdown)?;
    e.error += e.y.v.norm()
        * 64.0
        * f64::EPSILON
        * (f64::from(degree) + 1.0)
        * (f64::from(degree) + 2.0).ln();
    e.method = algorithm_component!("complex_order_bessel_derivative");
    Ok(e)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate<S: RealScalar>(
    nu: C,
    z: Complex<S>,
    options: BesselOptions,
    selection: (Kind, bool),
    family: AlgorithmComponent,
    context: Option<&mut SolveContext<'_, '_>>,
    degree: Option<u32>,
) -> Result<crate::complex_support::ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let (kind, scaled) = selection;
    let z = C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    if !nu.is_finite() || !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and coordinate must be finite",
        ));
    }
    if z == ZERO && !(matches!(kind, Kind::J | Kind::I) && nu.re > f64::from(degree.unwrap_or(0))) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("z"),
            scioxide_core::DomainReason::Pole,
            "the requested complex-order function has no finite origin limit",
        ));
    }
    let shift = if !scaled {
        ZERO
    } else {
        match kind {
            Kind::J | Kind::Y => real(-z.im.abs()),
            Kind::I => real(-z.re.abs()),
            Kind::K => z,
            Kind::H1 => C::new(z.im, -z.re),
            Kind::H2 => C::new(-z.im, z.re),
        }
    };
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let result = if z == ZERO {
        work.try_step().map(|()| {
            Estimate::from(crate::support::JetData {
                y: Jet::constant(ZERO),
                scale: Jet::constant(ZERO),
                error: 0.0,
                derivative_error: Some(if nu.re > 1.0 { 0.0 } else { f64::INFINITY }),
                method: algorithm_component!("complex_order_bessel_origin_limit"),
            })
        })
    } else if let Some(degree) = degree {
        nth(kind, nu, z, shift, degree, &mut work)
    } else {
        kernel_value(kind, nu, Jet::variable(z), shift, &mut work)
    };
    work.check()?;
    crate::complex_support::report(result, degree.is_none(), &work, family)
}
