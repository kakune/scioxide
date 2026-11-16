//! Real incomplete gamma values and conditioned tails.

use crate::support::Work;
use crate::support::gamma_core::{GAMMA_ULPS, SMALL_FACTORIALS_F64};
use crate::support::gamma_real as real;
use crate::support::wide::Wide;
use crate::support::{check_positive, checked_probability, require_finite_param};
use crate::util::is_exact_zero;
use crate::{
    GammaOptions, SpecialFunctionTermination,
    result::{Bound, Evaluation, Projection, ulp_bound},
};
use scioxide_core::{AlgorithmComponent, RealScalar, SciOxideError, algorithm_component};
pub(crate) fn gamma_p<S: RealScalar>(a: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <GammaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    gamma_p_evaluate(a, x, options, None)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
fn gamma_p_kernel<S: RealScalar>(
    a: S,
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_p_kernel::<f64>(a.into_f64(), x.into_f64(), work),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    validate_regularized_gamma(a, x)?;
    if a == S::ZERO {
        return Ok(Evaluation::finite(
            S::ONE,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("gamma", "zero_shape_limit", 1),
            work.used,
            SpecialFunctionTermination::Limit,
        ));
    }
    if is_exact_zero(x) {
        return Ok(Evaluation::finite(
            x,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("gamma", "lower_zero_endpoint", 1),
            0,
            SpecialFunctionTermination::Exact,
        ));
    }
    let ratios = gamma_ratios_with_work(a, x, work)?;
    Ok(Evaluation::finite(
        ratios.p,
        Bound::derived(ratios.p_error),
        None,
        scioxide_core::AlgorithmId::from_components(
            algorithm_component!("gamma"),
            ratios.branch_id,
            1,
        ),
        ratios.iteration_count,
        SpecialFunctionTermination::Converged,
    ))
}
pub(crate) fn gamma_q<S: RealScalar>(a: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <GammaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    gamma_q_evaluate(a, x, options, None)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
fn gamma_q_kernel<S: RealScalar>(
    a: S,
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_q_kernel::<f64>(a.into_f64(), x.into_f64(), work),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    validate_regularized_gamma(a, x)?;
    if a == S::ZERO {
        return Ok(Evaluation::finite(
            S::ZERO,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("gamma", "zero_shape_limit", 1),
            work.used,
            SpecialFunctionTermination::Limit,
        ));
    }
    if is_exact_zero(x) {
        return Ok(Evaluation::finite(
            S::ONE,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("gamma", "upper_zero_endpoint", 1),
            0,
            SpecialFunctionTermination::Exact,
        ));
    }
    let ratios = gamma_ratios_with_work(a, x, work)?;
    Ok(Evaluation::finite(
        ratios.q,
        Bound::derived(ratios.q_error),
        None,
        scioxide_core::AlgorithmId::from_components(
            algorithm_component!("gamma"),
            ratios.branch_id,
            1,
        ),
        ratios.iteration_count,
        SpecialFunctionTermination::Converged,
    ))
}
pub(crate) fn gamma_lower<S: RealScalar>(a: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <GammaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    gamma_lower_evaluate(a, x, options, None)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
fn gamma_lower_kernel<S: RealScalar>(
    a: S,
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_lower_kernel::<f64>(a.into_f64(), x.into_f64(), work),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    validate_incomplete_gamma(a, x)?;
    if is_exact_zero(x) {
        return Ok(Evaluation::finite(
            x,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("gamma", "lower_zero_endpoint", 1),
            0,
            SpecialFunctionTermination::Exact,
        ));
    }

    let (a, x) = (a.into_f64(), x.into_f64());
    let normalization = Normalization::wide(a);
    let tail = log_tail(a, x, &normalization, TailArithmetic::Mixed, work)?;
    let (log_ratio, error) = tail.oriented(false);
    unregularized(log_ratio + normalization.value, error, &tail)
}
pub(crate) fn gamma_upper<S: RealScalar>(a: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <GammaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    gamma_upper_evaluate(a, x, options, None)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
fn gamma_upper_kernel<S: RealScalar>(
    a: S,
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_upper_kernel::<f64>(a.into_f64(), x.into_f64(), work),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    if a == S::ZERO {
        require_finite_param(x, "gamma coordinate must be finite")?;
        if x < S::ZERO {
            return Err(crate::support::GAMMA_ERRORS.outside(scioxide_core::argument!("x")));
        }
        if x == S::ZERO {
            return Err(crate::support::GAMMA_ERRORS.domain(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::Pole,
            ));
        }
        return crate::integrals::e1_logarithmic(x, work);
    }
    validate_incomplete_gamma(a, x)?;
    // The closed branches are limited by their binary64 inner arithmetic:
    // `inner` is that natural-log error before the final rounding.
    let eps = S::EPSILON;
    if is_exact_zero(x) {
        let value = real::ln_positive_wide(Wide::from(a.into_f64()));
        let log_value = S::try_from_f64_lossy(value.value())?;
        return Ok(logarithmic_result(
            log_value,
            crate::result::positive_projection(value.exp())?,
            eps * (S::ONE + log_value.abs()),
            0,
            algorithm_component!("upper_zero_endpoint"),
        ));
    }
    if a <= S::HALF && x <= S::ONE {
        let (value, iterations) = gamma_upper_small_shape(a.into_f64(), x.into_f64(), work)?;
        let log_value = S::try_from_f64_lossy(value.ln().value())?;
        return Ok(logarithmic_result(
            log_value,
            S::try_from_f64_lossy(value.value())?,
            S::TWO * eps * (S::ONE + log_value.abs()),
            iterations,
            algorithm_component!("upper_small_shape_series"),
        ));
    }
    let (a, x) = (a.into_f64(), x.into_f64());
    let normalization = Normalization::wide(a);
    let tail = log_tail(a, x, &normalization, TailArithmetic::Mixed, work)?;
    let (log_ratio, error) = tail.oriented(true);
    unregularized(log_ratio + normalization.value, error, &tail)
}

/// Returns `exp(log_value)` and its logarithm. `inner` is the natural-log
/// error before the final rounding to the destination.
fn logarithmic_result<S: RealScalar>(
    log_value: S,
    projection: S,
    inner: S,
    iterations: usize,
    method: AlgorithmComponent,
) -> Evaluation<S> {
    Evaluation::logarithmic(
        S::ONE,
        log_value,
        Bound::rounded(log_value).plus(Bound::derived(inner)),
        Projection::Direct(
            projection,
            Bound::rounded(projection).plus(Bound::derived(projection * inner.exp_m1())),
        ),
        scioxide_core::AlgorithmId::from_components(algorithm_component!("gamma"), method, 1),
        iterations,
    )
}

/// An unregularized tail from its two-word logarithm, projected directly
/// rather than through the rounded logarithm. The leading word's `f64`
/// exponential adds one unit to the natural-log error.
fn unregularized<S: RealScalar>(
    value: Wide,
    error: f64,
    tail: &LogTail,
) -> Result<Evaluation<S>, SciOxideError> {
    Ok(logarithmic_result(
        S::try_from_f64_lossy(value.value())?,
        crate::result::positive_projection(value.exp())?,
        S::try_from_f64_lossy(error + f64::EPSILON)?,
        tail.iterations,
        tail.method.component(),
    ))
}

/// Cancel the 1/a pole analytically before evaluating the small-shape tail.
/// Gamma(a,x) = (Gamma(1+a)-x^a)/a - x^a sum_(k>=1) (-x)^k/(k!(a+k)).
fn gamma_upper_small_shape(
    a: f64,
    x: f64,
    work: &mut Work,
) -> Result<(Wide, usize), SciOxideError> {
    let a = Wide::from(a);
    let x = Wide::from(x);
    let log_x = x.ln();
    let log_power = a * log_x;
    let complete = real::ln_positive_wide(Wide::from(1.0) + a);
    let difference = if a.hi * log_x.hi.abs().max(1.0) < 1.0e-18 {
        // The a->0 limit avoids products below the subnormal lattice.
        // On 0<x<=2 the relative first-order correction is < 4e-18
        // under this guard (including the logarithmic x->0 limit).
        let (hi, lo) = crate::support::reciprocal_gamma::RECIPROCAL_GAMMA[1];
        -Wide { hi, lo } - log_x
    } else {
        (-(log_power - complete).exp_m1() / a) * (Wide::from(1.0) + complete.exp_m1())
    };
    let mut term = -x / (a + Wide::from(1.0));
    let mut sum = term;
    let mut iteration = 0usize;
    loop {
        iteration += 1;
        work.try_step()?;
        let k = iteration as f64;
        term = -(term * x).div_f64(k + 1.0) * (a + Wide::from(k)) / (a + Wide::from(k + 1.0));
        sum = sum + term;
        if term.hi.abs() <= 1.0e-28 * sum.hi.abs() {
            return Ok((
                difference - sum * (Wide::from(1.0) + log_power.exp_m1()),
                iteration,
            ));
        }
    }
}

fn validate_regularized_gamma<S: RealScalar>(a: S, x: S) -> Result<(), SciOxideError> {
    require_finite_param(a, "gamma shape must be finite")?;
    require_finite_param(x, "gamma coordinate must be finite")?;
    if a < S::ZERO {
        return Err(crate::support::GAMMA_ERRORS.outside(scioxide_core::argument!("a")));
    }
    if x < S::ZERO {
        return Err(crate::support::GAMMA_ERRORS.outside(scioxide_core::argument!("x")));
    }
    if a == S::ZERO && x == S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::Undefined,
            "regularized gamma has no joint limit at zero shape and coordinate",
        ));
    }
    Ok(())
}

fn validate_incomplete_gamma<S: RealScalar>(a: S, x: S) -> Result<(), SciOxideError> {
    check_positive(
        a,
        scioxide_core::argument!("a"),
        "gamma parameter a must be positive",
    )?;
    require_finite_param(x, "gamma parameter x must be finite")?;
    if x < S::ZERO {
        Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "incomplete-gamma argument must be non-negative",
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers for incomplete gamma
// ---------------------------------------------------------------------------

/// Complementary regularized tails with absolute error bounds.
#[derive(Clone, Copy)]
struct GammaRatios<S> {
    p: S,
    q: S,
    p_error: S,
    q_error: S,
    branch_id: AlgorithmComponent,
    iteration_count: usize,
}

impl<S: RealScalar> GammaRatios<S> {
    /// The pair from a primary tail and its complement `1 − tail`, which adds
    /// one rounding to the propagated error.
    fn from_tail(
        tail: S,
        error: S,
        lower: bool,
        branch_id: AlgorithmComponent,
        iteration_count: usize,
    ) -> Result<Self, SciOxideError> {
        let complement = checked_probability(S::ONE - tail)?;
        let complement_error = error + ulp_bound(complement, S::HALF);
        let ((p, p_error), (q, q_error)) = if lower {
            ((tail, error), (complement, complement_error))
        } else {
            ((complement, complement_error), (tail, error))
        };
        Ok(Self {
            p,
            q,
            p_error,
            q_error,
            branch_id,
            iteration_count,
        })
    }
}

fn gamma_ratios_with_work<S: RealScalar>(
    a: S,
    x: S,
    work: &mut Work,
) -> Result<GammaRatios<S>, SciOxideError> {
    gamma_ratios_core(a, x, true, work)
}

fn gamma_ratios_core<S: RealScalar>(
    a: S,
    x: S,
    specialized: bool,
    work: &mut Work,
) -> Result<GammaRatios<S>, SciOxideError> {
    let one = S::ONE;

    if a == one {
        work.try_step()?;
        let p = -(-x).exp_m1();
        let q = (-x).exp();
        return Ok(GammaRatios {
            p,
            q,
            p_error: ulp_bound(p, S::try_from_f64_lossy(GAMMA_ULPS)?),
            q_error: ulp_bound(q, S::try_from_f64_lossy(GAMMA_ULPS)?),
            branch_id: algorithm_component!("unit_exponential"),
            iteration_count: 1,
        });
    }

    if a == S::HALF {
        work.try_step()?;
        // P(1/2,x)=erf(sqrt(x)): retain subnormal x without a log/exp
        // round trip whose error grows with the magnitude of log(x).
        let root = x.sqrt();
        let (p, q) = (
            crate::error_probability::error_fn::erf(root)?,
            crate::error_probability::error_fn::erfc(root)?,
        );
        // The rounded root perturbs both tails by at most ε(1+x)·erfc(√x).
        let conditioning = S::EPSILON * (one + x) * q;
        return Ok(GammaRatios {
            p,
            q,
            p_error: ulp_bound(p, S::try_from_f64_lossy(GAMMA_ULPS)?) + conditioning,
            q_error: ulp_bound(q, S::try_from_f64_lossy(GAMMA_ULPS)?) + conditioning,
            branch_id: algorithm_component!("half_integer_erf_erfc"),
            iteration_count: 1,
        });
    }

    if a < one && x <= S::TWO && a * x.ln() > -S::LN_2 {
        // Avoid subtracting a lower tail close to one. The power criterion
        // selects a region where the complementary lower result is sizable.
        let (upper, iterations) = gamma_upper_small_shape(a.into_f64(), x.into_f64(), work)?;
        let shifted = real::ln_positive_wide(Wide::from(1.0) + Wide::from(a.into_f64()));
        let q = S::try_from_f64_lossy(
            (upper.mul_f64(a.into_f64()))
                .mul_f64((-shifted).exp())
                .value(),
        )?;
        return GammaRatios::from_tail(
            q,
            ulp_bound(q, S::try_from_f64_lossy(GAMMA_ULPS)?),
            false,
            algorithm_component!("small_shape_upper_series"),
            iterations,
        );
    }

    // The finite lattice sums need a normal exp(-x).
    if specialized && x >= a && x <= -S::MIN_POSITIVE.ln() {
        let twice_a = a + a;
        let lattice_limit = S::try_from_f64_lossy(65.0)?;
        if twice_a == twice_a.floor() && twice_a <= lattice_limit {
            let twice_a = twice_a.try_to_f64_exact()? as usize;
            let required = twice_a.div_ceil(2);
            if required > work.remaining {
                return gamma_ratios_core(a, x, false, work);
            }
            let (q, error, branch_id, iteration_count) = if twice_a.is_multiple_of(2) {
                let terms = twice_a / 2;
                let q = gamma_q_integer(terms, x, work)?;
                // exp(-x) and two roundings per positive term.
                let error = ulp_bound(q, S::try_from_usize_exact(2 * terms + 4)?);
                (
                    q,
                    error,
                    algorithm_component!("integer_upper_finite"),
                    terms,
                )
            } else {
                let steps = twice_a / 2;
                let (q, complementary_error_function) = gamma_q_half_integer(steps, x, work)?;
                let error = ulp_bound(q, S::try_from_usize_exact(2 * steps + 8)?)
                    + S::EPSILON * (one + x) * complementary_error_function;
                (
                    q,
                    error,
                    algorithm_component!("half_integer_upper_finite"),
                    steps,
                )
            };
            return GammaRatios::from_tail(q, error, false, branch_id, iteration_count);
        }
    }

    if use_temme(a.into_f64(), x.into_f64()) {
        let temme = temme(a.into_f64(), x.into_f64(), work)?;
        let tail = checked_probability(S::try_from_f64_lossy(temme.value())?)?;
        // The final rounding and gradual underflow of the scaled product.
        let error = tail * S::try_from_f64_lossy(temme.relative)? + ulp_bound(tail, S::TWO);
        return GammaRatios::from_tail(
            tail,
            error,
            x < a,
            TailMethod::Temme.component(),
            TEMME_TERMS,
        );
    }

    if x < a {
        let (sum, iteration_count) = gamma_lower_series_sum(a.into_f64(), x.into_f64(), work)?;
        let p = checked_probability(S::try_from_f64_lossy(
            real::gamma_prefix(a.into_f64(), x.into_f64()) * sum,
        )?)?;
        let error = ulp_bound(p, S::try_from_usize_exact(16 + 2 * iteration_count)?);
        GammaRatios::from_tail(
            p,
            error,
            true,
            algorithm_component!("lower_compensated_series"),
            iteration_count,
        )
    } else {
        let (fraction, iteration_count) = gamma_upper_fraction(a.into_f64(), x.into_f64(), work)?;
        let q = checked_probability(S::try_from_f64_lossy(
            real::gamma_prefix(a.into_f64(), x.into_f64()) * fraction.value(),
        )?)?;
        let error = ulp_bound(q, S::try_from_usize_exact(16 + 2 * iteration_count)?);
        GammaRatios::from_tail(
            q,
            error,
            false,
            algorithm_component!("upper_scaled_lentz"),
            iteration_count,
        )
    }
}

fn gamma_q_integer<S: RealScalar>(terms: usize, x: S, work: &mut Work) -> Result<S, SciOxideError> {
    work.try_step()?;
    let (sum, _) = (1..terms).try_fold((S::ONE, S::ONE), |(sum, term), index| {
        work.try_step()?;
        #[allow(clippy::cast_precision_loss)]
        let divisor = S::try_from_f64_lossy(index as f64)?;
        let next = term * x / divisor;
        Ok::<_, SciOxideError>((sum + next, next))
    })?;
    checked_probability((-x).exp() * sum)
}

/// The upper tail and its leading `erfc(√x)` term.
fn gamma_q_half_integer<S: RealScalar>(
    steps: usize,
    x: S,
    work: &mut Work,
) -> Result<(S, S), SciOxideError> {
    let sqrt_x = x.sqrt();
    let complementary_error_function = crate::error_probability::error_fn::erfc(sqrt_x)?;
    let sqrt_pi = S::PI.sqrt();
    let initial_term = (S::TWO) * sqrt_x * (-x).exp() / sqrt_pi;
    let (q, _) = (0..steps).try_fold(
        (complementary_error_function, initial_term),
        |(q, term), step| {
            work.try_step()?;
            #[allow(clippy::cast_precision_loss)]
            let denominator = S::try_from_f64_lossy(step as f64 + 1.5)?;
            Ok::<_, SciOxideError>((q + term, term * x / denominator))
        },
    )?;
    Ok((checked_probability(q)?, complementary_error_function))
}

/// Compensated scalar lower series `S = sum_n x^n / (a (a+1) ... (a+n))` and
/// its term count.
fn gamma_lower_series_sum(a: f64, x: f64, work: &mut Work) -> Result<(f64, usize), SciOxideError> {
    let eps = f64::EPSILON;
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut correction = 0.0;

    let mut iteration = 0usize;
    loop {
        iteration += 1;
        work.try_step()?;
        let ap = a + iteration as f64;
        term = term * x / ap;
        let corrected_term = term - correction;
        let updated = sum + corrected_term;
        correction = (updated - sum) - corrected_term;
        sum = updated;
        if term.abs() <= eps * sum.abs() {
            let ratio = x / (ap + 1.0);
            if ratio < 1.0 && term.abs() * ratio <= 0.25 * eps * sum.abs() * (1.0 - ratio) {
                return Ok((sum, iteration));
            }
        }
    }
}

/// Two-word lower series normalized to start at one,
/// `1 + sum_n x^n / ((a+1) ... (a+n))`. Dividing out `1/a` cancels the pole
/// analytically for very small shapes and quantiles.
fn gamma_lower_wide_sum(a: f64, x: f64, work: &mut Work) -> Result<(Wide, usize), SciOxideError> {
    let mut term = Wide::from(1.0);
    let mut sum = term;
    let mut count = 0usize;
    loop {
        work.try_step()?;
        count += 1;
        term = term.mul_f64(x) / (Wide::from(a) + Wide::from(count as f64));
        sum = sum + term;
        if term.hi.abs() <= 32.0 * f64::EPSILON * f64::EPSILON * sum.hi.abs() {
            return Ok((sum, count));
        }
    }
}

/// The upper Legendre fraction `F` of `Γ(a,x) = x^a e^(-x) F` by modified
/// Lentz steps, accumulating its product in two words.
#[inline]
fn gamma_upper_fraction(a: f64, x: f64, work: &mut Work) -> Result<(Wide, usize), SciOxideError> {
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let floor = |value: f64| crate::util::lentz_floor(value, tiny);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / floor(b);
    let mut fraction = Wide::from(d);
    let mut index = 0.0;
    let mut iteration = 0usize;
    loop {
        iteration += 1;
        work.try_step()?;
        index += 1.0;
        let numerator = -index * (index - a);
        b += 2.0;
        d = 1.0 / floor(numerator.mul_add(d, b));
        c = floor(b + numerator / c);
        let delta = d * c;
        fraction = fraction.mul_f64(d).mul_f64(c);
        if !fraction.hi.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "continued-fraction state exceeded the scalar range",
            ));
        }
        if (delta - 1.0).abs() <= 0.5 * f64::EPSILON {
            return Ok((fraction, iteration));
        }
    }
}

const TEMME_TERMS: usize = 10;

/// The transition band `|x − a| < 2a/5` above `a = 20`. Below `a = 100` its
/// lower edge is `x − a > −a/4`, where the truncated Taylor polynomials of the
/// higher `c_k(η)` stay within an ulp (`|η| < 0.28`). Outside the band the
/// lower series needs at most about 170 terms and the fraction about 20 steps
/// for every larger shape, so no evaluation cost grows with `a`.
fn use_temme(a: f64, x: f64) -> bool {
    let lower = if a < 100.0 { 0.25 } else { 0.4 };
    a > 20.0 && x - a > -lower * a && x - a < 0.4 * a
}

/// The smaller regularized tail by Temme's uniform expansion (DLMF 8.12.8):
/// `exp(exponent) · bracket`, P below the transition and Q from it.
#[derive(Clone, Copy)]
struct Temme {
    /// `−aφ` with `φ = σ − ln(1+σ)`, `σ = (x−a)/a`, in two words.
    exponent: Wide,
    /// `erfcx(√(aφ))/2 ∓ C(η)/√(2πa)`, the tail scaled by `exp(aφ)`.
    bracket: f64,
    /// Relative error bound of the tail.
    relative: f64,
    shape: f64,
}

impl Temme {
    fn value(self) -> f64 {
        self.bracket * self.exponent.exp()
    }

    fn log(self) -> Wide {
        self.exponent + Wide::from(self.bracket.ln())
    }

    /// `ln(x · density) = −aφ + ln √(a/2π) − μ(a)`, with Stirling's
    /// correction `μ`; no complete log-gamma value is formed.
    fn power(self) -> Wide {
        let inverse = 1.0 / self.shape;
        self.exponent
            + Wide::from(
                0.5 * (self.shape / core::f64::consts::TAU).ln()
                    - crate::support::gamma_series::stirling_f64(inverse, inverse * inverse),
            )
    }
}

/// Temme's expansion with Taylor polynomials of its coefficients `c_k(η)`,
/// `η = sign(σ)√(2φ)`. `x − a` is exact on the band (Sterbenz), and
/// `y = aφ` keeps two words, so the Gaussian exponent loses no digits.
#[allow(clippy::excessive_precision)]
fn temme(a: f64, x: f64, work: &mut Work) -> Result<Temme, SciOxideError> {
    const C0: &[f64] = &[
        -0.333_333_333_333_333_3,
        0.083_333_333_333_333_33,
        -0.014_814_814_814_814_815,
        0.001_157_407_407_407_407_3,
        0.000_352_733_686_067_019_4,
        -0.000_178_755_144_032_921_81,
        0.391_926_317_852_243_8e-4,
        -0.218_544_851_067_999_22e-5,
        -0.185_406_221_071_516e-5,
        0.829_671_134_095_308_6e-6,
        -0.176_659_527_368_260_8e-6,
        0.670_785_354_340_149_9e-8,
        0.102_618_097_842_403_08e-7,
        -0.438_203_601_845_335_3e-8,
        0.914_769_958_223_679e-9,
    ];
    const C1: &[f64] = &[
        -0.001_851_851_851_851_852,
        -0.003_472_222_222_222_222,
        0.002_645_502_645_502_645_4,
        -0.000_990_226_337_448_559_6,
        0.000_205_761_316_872_427_98,
        -0.401_877_572_016_460_9e-6,
        -0.180_985_503_344_899_78e-4,
        0.764_916_091_608_111e-5,
        -0.161_209_008_945_634_46e-5,
        0.464_712_780_280_743_4e-8,
        0.137_863_344_691_572_1e-6,
        -0.575_254_560_351_770_5e-7,
        0.119_516_285_997_781_47e-7,
    ];
    const C2: &[f64] = &[
        0.004_133_597_883_597_884,
        -0.002_681_327_160_493_827_3,
        0.000_771_604_938_271_604_9,
        0.200_938_786_008_230_45e-5,
        -0.000_107_366_532_263_651_6,
        0.529_234_488_291_201_3e-4,
        -0.127_606_351_886_187_28e-4,
        0.342_357_873_409_613_8e-7,
        0.137_219_573_090_629_33e-5,
        -0.629_899_213_838_005_5e-6,
        0.142_806_142_060_642_42e-6,
    ];
    const C3: &[f64] = &[
        0.000_649_434_156_378_600_8,
        0.000_229_472_093_621_399_18,
        -0.000_469_189_494_395_255_7,
        0.000_267_720_632_062_838_85,
        -0.756_180_167_188_397_6e-4,
        -0.239_650_511_386_729_67e-6,
        0.110_826_541_153_473_02e-4,
        -0.567_495_282_699_159_7e-5,
        0.142_309_007_324_358_85e-5,
    ];
    const C4: &[f64] = &[
        -0.000_861_888_290_916_711_7,
        0.000_784_039_221_720_066_6,
        -0.000_299_072_480_303_190_2,
        -0.146_384_525_788_434_18e-5,
        0.664_149_821_546_512_2e-4,
        -0.396_836_504_717_943_5e-4,
        0.113_757_269_706_784_19e-4,
    ];
    const C5: &[f64] = &[
        -0.000_336_798_553_366_358_15,
        -0.697_281_375_836_585_8e-4,
        0.000_277_275_324_495_939_2,
        -0.000_199_325_705_161_888_47,
        0.679_778_047_793_720_8e-4,
        0.141_906_292_064_396_7e-6,
        -0.135_940_481_897_686_93e-4,
        0.801_847_025_633_420_2e-5,
        -0.229_148_117_650_809_5e-5,
    ];
    const C6: &[f64] = &[
        0.000_531_307_936_463_992_2,
        -0.000_592_166_437_353_693_9,
        0.000_270_878_209_671_804_5,
        0.790_235_323_266_032_8e-6,
        -0.815_396_936_756_196_9e-4,
        0.561_168_275_310_625e-4,
        -0.183_291_165_828_433_77e-4,
    ];
    const C7: &[f64] = &[
        0.000_344_367_606_892_377_67,
        0.517_179_090_826_059_2e-4,
        -0.000_334_931_610_811_422_36,
        0.000_281_269_515_476_323_7,
        -0.000_109_765_822_446_847_31,
    ];
    const C8: &[f64] = &[
        -0.000_652_623_918_595_309_4,
        0.000_839_498_720_672_087_3,
        -0.000_438_297_098_541_721,
    ];
    const C9: &[f64] = &[-0.000_596_761_290_192_746_3];

    let sigma = Wide::from(x - a) / Wide::from(a);
    let phi = -crate::support::wide::log1pmx(sigma);
    let y = phi.mul_f64(a);
    let eta = (2.0 * phi.hi).sqrt().copysign(x - a);
    let inverse_a = 1.0 / a;
    let correction = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9]
        .iter()
        .rev()
        .try_fold(0.0, |value: f64, coefficients| {
            work.try_step()?;
            let polynomial = crate::support::polynomial::polynomial(
                eta,
                coefficients,
                crate::support::polynomial::PolynomialPolicy::Unfused,
            );
            Ok::<_, SciOxideError>(value.mul_add(inverse_a, polynomial))
        })?;
    let gaussian = 0.5 * crate::error_probability::real_faddeeva::erfcx(y.hi.sqrt());
    let scaled = correction / (core::f64::consts::TAU * a).sqrt();
    let bracket = if x < a {
        gaussian - scaled
    } else {
        gaussian + scaled
    };
    if bracket.is_nan() || bracket <= 0.0 {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "Temme expansion of the incomplete-gamma tail is not positive",
        ));
    }
    // erfcx and its rounded argument (5ε) and the expansion with its scaling
    // (8ε) relative to their parts; the bracket, exponential and product
    // roundings. Omitted 1/a and Taylor terms stay below ε/2 on the band.
    let relative = f64::EPSILON * (4.0 + 8.0 * (gaussian + scaled.abs()) / bracket);
    Ok(Temme {
        exponent: -y,
        bracket,
        relative,
        shape: a,
    })
}

/// Arithmetic of a logarithmic tail's power factor and lower series.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailArithmetic {
    /// Binary64 power factor and logarithms, with their roundings bounded:
    /// safeguarded inverse steps away from the output lattice.
    Scalar,
    /// Two-word power factor and logarithms with the compensated scalar
    /// lower series, whose relative error is modelled.
    Mixed,
    /// Two-word power factor, logarithms and lower series.
    Compensated,
}

/// The evaluation behind a logarithmic tail.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailMethod {
    Temme,
    ScalarSeries,
    Series,
    Fraction,
}

impl TailMethod {
    pub(crate) fn component(self) -> AlgorithmComponent {
        match self {
            Self::Temme => algorithm_component!("temme_transition"),
            Self::ScalarSeries => algorithm_component!("lower_compensated_series"),
            Self::Series => algorithm_component!("lower_twofold_series"),
            Self::Fraction => algorithm_component!("upper_twofold_lentz"),
        }
    }
}

/// `ln Γ(a)` and `ln Γ(a+1)`, and the absolute uncertainty of the first.
#[derive(Clone, Copy)]
pub(crate) struct Normalization {
    pub(crate) value: Wide,
    pub(crate) shifted: Wide,
    pub(crate) error: f64,
}

impl Normalization {
    /// Two-word normalization; tabulated factorials are exact `f64` integers.
    pub(crate) fn wide(a: f64) -> Self {
        let factorials = &SMALL_FACTORIALS_F64;
        let value = if a == 1.0 || a == 2.0 {
            Wide::ZERO
        } else if a >= 1.0 && a <= factorials.len() as f64 && a == a.floor() {
            Wide::from(factorials[a as usize - 1]).ln()
        } else {
            real::ln_positive_wide(Wide::from(a))
        };
        let shifted = if a == 2.0 {
            crate::support::constants::LN_2
        } else if a >= 1.0 {
            value + Wide::from(a).ln()
        } else {
            real::ln_positive_wide(Wide::from(a) + Wide::from(1.0))
        };
        Self {
            value,
            shifted,
            error: 0.0,
        }
    }
}

/// The natural logarithm of one directly evaluated regularized tail.
#[derive(Clone, Copy)]
pub(crate) struct LogTail {
    pub(crate) value: Wide,
    /// Whether `value` is `ln Q` rather than `ln P`.
    pub(crate) upper: bool,
    /// `ln(x · density) = a ln x − x − ln Γ(a)`, the inverse derivative factor.
    pub(crate) power: Wide,
    /// Absolute error bound of `value`, including the normalization's.
    pub(crate) error: f64,
    pub(crate) iterations: usize,
    pub(crate) method: TailMethod,
}

impl LogTail {
    /// The logarithm of the requested tail and its absolute error bound. The
    /// complement amplifies the error by `T/(1−T)` for the direct tail `T`.
    pub(crate) fn oriented(self, upper: bool) -> (Wide, f64) {
        if upper == self.upper {
            return (self.value, self.error);
        }
        let opposite = crate::error_probability::probability::wide_log1mexp(self.value);
        (
            opposite,
            (self.error + f64::EPSILON) * (self.value.hi - opposite.hi).exp(),
        )
    }
}

/// The single logarithmic tail assembly shared by the unregularized, log and
/// inverse forms: Temme's expansion on its band, the lower series below
/// `a + 1`, and the upper fraction above it.
pub(crate) fn log_tail(
    a: f64,
    x: f64,
    normalization: &Normalization,
    arithmetic: TailArithmetic,
    work: &mut Work,
) -> Result<LogTail, SciOxideError> {
    if use_temme(a, x) {
        let temme = temme(a, x, work)?;
        return Ok(LogTail {
            value: temme.log(),
            upper: x >= a,
            power: temme.power(),
            error: temme.relative,
            iterations: TEMME_TERMS,
            method: TailMethod::Temme,
        });
    }
    let eps = f64::EPSILON;
    let scalar = arithmetic == TailArithmetic::Scalar;
    // a ln x − x, and the rounding of the product and its logarithm.
    let (exponent, mut error) = if scalar {
        let power = a * x.ln();
        (
            Wide::from(power) - Wide::from(x),
            eps * (power.abs() + normalization.value.hi.abs()),
        )
    } else {
        let power = Wide::from(a) * Wide::from(x).ln();
        (
            power - Wide::from(x),
            4.0 * eps * eps * (power.hi.abs() + x),
        )
    };
    error += normalization.error;
    let logarithm = |value: f64| {
        if scalar {
            (Wide::from(value.ln()), 0.5 * eps * value.ln().abs())
        } else {
            (Wide::from(value).ln(), 0.0)
        }
    };
    let power = exponent - normalization.value;
    let (value, upper, iterations, method) = if x < a + 1.0 {
        if a >= 0.125 && arithmetic != TailArithmetic::Compensated {
            let (sum, count) = gamma_lower_series_sum(a, x, work)?;
            // Sum n t_n / sum t_n = x − a + 1/S. Each term recurrence rounds at
            // most three times (1.5ε per index); compensated summation, the
            // initial division and the bounded remainder cost two.
            let mean = (x - a + 1.0 / sum).max(0.0) + eps * (x + a + 1.0 / sum);
            error += eps * (2.0 + 1.5 * mean + 4.0 * eps * (count as f64).powi(2));
            let (log_sum, rounding) = logarithm(sum);
            error += rounding;
            (power + log_sum, false, count, TailMethod::ScalarSeries)
        } else {
            let (sum, count) = gamma_lower_wide_sum(a, x, work)?;
            error += 32.0 * eps * eps * count as f64;
            (
                exponent - normalization.shifted + sum.ln(),
                false,
                count,
                TailMethod::Series,
            )
        }
    } else {
        let (fraction, count) = gamma_upper_fraction(a, x, work)?;
        // Two roundings per modified-Lentz step and the stopping remainder.
        error += eps * (4.0 + 2.0 * count as f64);
        let log_fraction = if scalar {
            logarithm(fraction.value())
        } else {
            (fraction.ln(), 0.0)
        };
        error += log_fraction.1;
        (power + log_fraction.0, true, count, TailMethod::Fraction)
    };
    Ok(LogTail {
        // A rounded tail near one may not exceed the lattice's one.
        value: if value.hi > 0.0 { Wide::ZERO } else { value },
        upper,
        power,
        error,
        iterations,
        method,
    })
}

/// Binary64 log tail for safeguarded inverse steps away from the output
/// lattice: the requested log tail, its absolute bound and the power factor.
pub(crate) fn probability_log_tail_fast(
    a: f64,
    x: f64,
    upper: bool,
    work: &mut Work,
    normalization: &Normalization,
) -> Result<(f64, f64, f64), SciOxideError> {
    let tail = log_tail(a, x, normalization, TailArithmetic::Scalar, work)?;
    let (direct, power) = (tail.value.value(), tail.power.value());
    if upper == tail.upper {
        return Ok((direct, tail.error, power));
    }
    // A scalar complement keeps these steps free of two-word logarithms.
    let opposite = crate::error_probability::probability::log1mexp(direct)?;
    Ok((
        opposite,
        tail.error * (direct - opposite).exp() + 4.0 * f64::EPSILON * opposite.abs(),
        power,
    ))
}

/// An `f64` incomplete-gamma tail evaluated in linear space.
#[derive(Clone, Copy)]
pub(crate) struct LinearGammaTail {
    /// The requested tail probability.
    pub(crate) value: f64,
    /// Calibrated relative uncertainty of `value`.
    pub(crate) error: f64,
    /// `x^a e^-x / Gamma(a)`, so `power / value` is the log-coordinate derivative.
    pub(crate) power: f64,
    /// The summed opposite tail when `value` is its complement.
    complemented: Option<f64>,
}

impl LinearGammaTail {
    /// Natural logarithm of `value` and its absolute uncertainty; a
    /// complement uses `ln1p` of the summed tail instead of rounding `1 - d`.
    pub(crate) fn log(&self) -> (f64, f64) {
        let log = self
            .complemented
            .map_or_else(|| self.value.ln(), |summed| (-summed).ln_1p());
        (log, self.error + f64::EPSILON * log.abs())
    }

    /// Relative uncertainty of the linear `value`, including the rounded
    /// complement subtraction.
    pub(crate) fn linear_error(&self) -> f64 {
        self.error
            + if self.complemented.is_some() {
                0.5 * f64::EPSILON
            } else {
                0.0
            }
    }
}

/// `Gamma(a)` and its relative uncertainty for `f64` power-factor tails:
/// the factorial table, the exact half-integer lattice times `sqrt(pi)`, or
/// the mixed-precision reciprocal-gamma recurrence below 128.
pub(crate) fn linear_normalization(a: f64) -> Option<(f64, f64)> {
    let twice = a + a;
    let lattice = twice == twice.floor() && (1.0..=65.0).contains(&twice);
    let (value, uncertainty) =
        if lattice && a == a.floor() && a <= SMALL_FACTORIALS_F64.len() as f64 {
            (SMALL_FACTORIALS_F64[a as usize - 1], 0.0)
        } else if lattice && a != a.floor() {
            // Gamma(n + 1/2) = sqrt(pi) (1/2)(3/2)...(n - 1/2): the doubled
            // factors are odd integers, exact while their product is.
            use crate::support::constants::SQRT_PI;
            let steps = (a - 0.5) as usize;
            let odd = (0..steps).try_fold(1.0_f64, |product, step| {
                let next = product * (2 * step + 1) as f64;
                (next < 9_007_199_254_740_992.0).then_some(next)
            })?;
            (
                SQRT_PI
                    .mul_f64(odd)
                    .mul_f64(0.5_f64.powi(steps as i32))
                    .value(),
                0.5 * f64::EPSILON,
            )
        } else if (1.0e-150..128.0).contains(&a) {
            let (value, uncertainty) = real::gamma_positive_fast(Wide::from(a));
            (value.value(), uncertainty + 0.5 * f64::EPSILON)
        } else {
            return None;
        };
    (value.is_normal() && uncertainty.is_finite()).then_some((value, uncertainty))
}

/// The requested incomplete-gamma tail in `f64` with a calibrated relative
/// uncertainty, or `None` where its power factor leaves the normal range or no
/// positive sum converges quickly.
///
/// Integer and half-integer shapes through 32.5 sum their finite upper tails
/// (the latter after `erfc(sqrt x)`). Otherwise, and for small lower tails,
/// the positive series `x^a e^-x / Gamma(a+1) sum x^n / (a+1)_n` is used
/// while `x < a + 1`. The opposite tail is the complement, whose uncertainty
/// includes the cancellation amplification. `pow` and `exp` keep their
/// sub-ulp errors instead of rounding `a ln x - x` in log space.
pub(crate) fn linear_tail(
    a: f64,
    x: f64,
    upper: bool,
    (gamma, gamma_error): (f64, f64),
    work: &mut Work,
) -> Result<Option<LinearGammaTail>, SciOxideError> {
    if !(x > 0.0 && x <= 708.0) {
        return Ok(None);
    }
    let leading = x.powf(a);
    let exponential = (-x).exp();
    if !(leading.is_normal() && exponential.is_normal()) {
        return Ok(None);
    }
    let power = leading * exponential / gamma;
    if !power.is_normal() {
        return Ok(None);
    }
    // Two sub-ULP elementary values, two products and the normalization.
    let power_error = 2.5 * f64::EPSILON + gamma_error;
    let twice = a + a;
    let lattice = twice == twice.floor() && twice <= 65.0;
    // Finite sums serve the upper tail from x >= a; below it the lower series
    // keeps a small lower tail and the logarithm of a near-one upper tail.
    let (upper_sum, summed) = if lattice && x >= a {
        // Finite upper tails: two roundings per positive term recurrence,
        // weighted by the share of later terms, plus summation.
        let (sum, weighted) = if a == a.floor() {
            let terms = a as usize;
            let (sum, _, weighted) = (1..terms).try_fold(
                (1.0_f64, 1.0_f64, 0.0_f64),
                |(sum, term, weighted), index| {
                    work.try_step()?;
                    let next = term * x / index as f64;
                    Ok::<_, SciOxideError>((sum + next, next, weighted + index as f64 * next))
                },
            )?;
            // exp(-x) and the final product add one unit.
            (
                exponential * sum,
                f64::EPSILON * (weighted / sum + 0.5 * terms as f64 + 1.0),
            )
        } else {
            let root = x.sqrt();
            let erfc = crate::error_probability::error_fn::erfc(root)?;
            // erfc's envelope and the rounded root's relative effect, about x.
            let erfc_error = f64::EPSILON * (4.0 + x) * erfc;
            let steps = (a - 0.5) as usize;
            // e^-x x^(1/2) / Gamma(3/2); later terms divide by k + 3/2.
            let initial = root * exponential * core::f64::consts::FRAC_2_SQRT_PI;
            let (series, _, weighted) = (0..steps).try_fold(
                (0.0_f64, initial, 0.0_f64),
                |(series, term, weighted), step| {
                    work.try_step()?;
                    Ok::<_, SciOxideError>((
                        series + term,
                        term * x / (step as f64 + 1.5),
                        weighted + (step as f64 + 3.0) * term,
                    ))
                },
            )?;
            let sum = erfc + series;
            (
                sum,
                (erfc_error + f64::EPSILON * (weighted + 0.5 * steps as f64 * sum)) / sum,
            )
        };
        (true, Some((sum, weighted)))
    } else if x < a + 1.0 {
        // Positive lower series; term ratios x/(a+1+n) decrease, each rounded
        // twice, and a rounding at index k perturbs every later term.
        let start = a + 1.0;
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut compensation = 0.0;
        let mut weighted = 0.0;
        let mut count = 0usize;
        loop {
            work.try_step()?;
            let ratio = x / (start + count as f64);
            count += 1;
            term *= ratio;
            let updated = sum + term;
            compensation += (sum - updated) + term;
            sum = updated;
            weighted += count as f64 * term;
            if ratio < 1.0 && term <= 0.125 * f64::EPSILON * sum * (1.0 - ratio) {
                let error = f64::EPSILON * (1.5 * weighted / sum + 1.5)
                    + term * ratio / ((1.0 - ratio) * sum)
                    + power_error;
                break (false, Some((power / a * (sum + compensation), error)));
            }
            if count > 512 {
                break (false, None);
            }
        }
    } else {
        (false, None)
    };
    let Some((summed, summed_error)) = summed else {
        return Ok(None);
    };
    let summed_error = if upper_sum {
        summed_error + 0.5 * f64::EPSILON
    } else {
        summed_error
    };
    if !(summed.is_finite() && summed > 0.0 && summed <= 1.0) {
        return Ok(None);
    }
    Ok(Some(if upper == upper_sum {
        LinearGammaTail {
            value: summed,
            error: summed_error,
            power,
            complemented: None,
        }
    } else {
        let value = 1.0 - summed;
        if value <= 0.0 {
            return Ok(None);
        }
        LinearGammaTail {
            value,
            error: summed_error * (summed / value),
            power,
            complemented: Some(summed),
        }
    }))
}

/// Four-ulp scalar log-gamma envelope, expressed as an absolute uncertainty.
pub(crate) fn probability_normalization(a: f64) -> (f64, f64) {
    let value = real::ln_positive(a);
    (
        value,
        if a == 1.0 || a == 2.0 {
            0.0
        } else {
            4.0 * (value.abs().next_up() - value.abs())
        },
    )
}

/// Compensated log tail, power factor and absolute output-coordinate
/// uncertainty for inverse refinement. The scalar lower series replaces the
/// two-word recurrence only inside an eight-output-ulp bound; Temme's bounded
/// expansion reports its uncertainty, and two-word recurrences report none.
pub(crate) fn probability_log_tail_wide(
    a: f64,
    x: f64,
    upper: bool,
    work: &mut Work,
    normalization: &Normalization,
) -> Result<(Wide, Wide, f64), SciOxideError> {
    let mut tail = log_tail(a, x, normalization, TailArithmetic::Mixed, work)?;
    // d ln T / d ln x = exp(power − ln T) converts the log error to x.
    let uncertainty = tail.error * x * (tail.value.hi - tail.power.hi).exp();
    let uncertainty = match tail.method {
        TailMethod::Temme => uncertainty,
        TailMethod::ScalarSeries
            if uncertainty.is_finite()
                && uncertainty <= 8.0 * (x - x.next_down()).max(f64::from_bits(1)) =>
        {
            uncertainty
        }
        TailMethod::ScalarSeries => {
            tail = log_tail(a, x, normalization, TailArithmetic::Compensated, work)?;
            0.0
        }
        TailMethod::Series | TailMethod::Fraction => 0.0,
    };
    Ok((tail.oriented(upper).0, tail.power, uncertainty))
}

/// Shared logarithmic ratios, keeping specialized central/transition paths.
pub(crate) fn probability_log_tail<S: RealScalar>(
    a: S,
    x: S,
    upper: bool,
    work: &mut crate::support::Work,
) -> Result<(S, AlgorithmComponent), SciOxideError> {
    let (p, q, branch) = probability_log_pair(a, x, upper, work)?;
    Ok((if upper { q } else { p }, branch))
}

/// Both logarithmic tails; a conditioned `f64` linear tail serves the
/// requested (`upper`) one directly, the other then being its complement.
fn probability_log_pair<S: RealScalar>(
    a: S,
    x: S,
    upper: bool,
    work: &mut crate::support::Work,
) -> Result<(S, S, AlgorithmComponent), SciOxideError> {
    use crate::error_probability::probability::log1mexp;
    validate_regularized_gamma(a, x)?;
    if a == S::ZERO {
        return Ok((
            S::ZERO,
            S::NEG_INFINITY,
            algorithm_component!("zero_shape_limit"),
        ));
    }
    if x == S::ZERO {
        return Ok((-S::INFINITY, S::ZERO, algorithm_component!("lower_zero")));
    }
    if work.remaining == 0 {
        work.complete = false;
        return Err(work.limit_error());
    }
    let (af, xf) = (a.into_f64(), x.into_f64());
    if let Some(normalization) = linear_normalization(af)
        && let Some(tail) = linear_tail(af, xf, upper, normalization, work)?
    {
        let (log, error) = tail.log();
        if error <= crate::gamma_beta::beta::LINEAR_TAIL_ACCEPT * log.abs() {
            let log = S::try_from_f64_lossy(log)?;
            let other = log1mexp(log)?;
            return Ok(if upper {
                (other, log, algorithm_component!("upper_linear_tail"))
            } else {
                (log, other, algorithm_component!("lower_linear_tail"))
            });
        }
    }
    let r = gamma_ratios_with_work(a, x, work)?;
    if r.p >= S::MIN_POSITIVE && r.q >= S::MIN_POSITIVE {
        return if r.p <= r.q {
            let p = r.p.ln();
            Ok((p, log1mexp(p)?, r.branch_id))
        } else {
            let q = r.q.ln();
            Ok((log1mexp(q)?, q, r.branch_id))
        };
    }
    if work.remaining == 0 {
        work.complete = false;
        return Err(work.limit_error());
    }
    // Ratios below the normal range keep their logarithm in two words.
    let (a, x) = (a.into_f64(), x.into_f64());
    let tail = log_tail(a, x, &Normalization::wide(a), TailArithmetic::Mixed, work)?;
    Ok((
        S::try_from_f64_lossy(tail.oriented(false).0.value())?,
        S::try_from_f64_lossy(tail.oriented(true).0.value())?,
        tail.method.component(),
    ))
}

pub(crate) fn gamma_p_evaluate<S: RealScalar>(
    a: S,
    x: S,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = gamma_p_kernel(a, x, &mut work);
    work.check()?;
    result.map(|evaluation| {
        evaluation.with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
    })
}

pub(crate) fn gamma_q_evaluate<S: RealScalar>(
    a: S,
    x: S,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = gamma_q_kernel(a, x, &mut work);
    work.check()?;
    result.map(|evaluation| {
        evaluation.with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
    })
}

pub(crate) fn gamma_lower_evaluate<S: RealScalar>(
    a: S,
    x: S,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = gamma_lower_kernel(a, x, &mut work);
    work.check()?;
    result.map(|evaluation| {
        evaluation.with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
    })
}

pub(crate) fn gamma_upper_evaluate<S: RealScalar>(
    a: S,
    x: S,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = gamma_upper_kernel(a, x, &mut work);
    work.check()?;
    result.map(|evaluation| {
        evaluation.with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
    })
}
