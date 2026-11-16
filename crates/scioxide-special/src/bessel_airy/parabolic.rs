//! Parabolic cylinder D/V/W, DLMF 12.2, 12.4, 12.5, 12.9 and 12.14.
use crate::ValueWithError;
use crate::expansions::real;
use crate::result::{Either, Evaluation, Form, Forms, Publish};
use crate::support::Work;
use crate::support::wide::Wide;
use crate::{
    ParabolicCylinderOptions, ScalarRepresentation, SpecialFunctionResult,
    SpecialFunctionTermination,
};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, EstimateReliability, EvaluationCounts, LogValue, RealScalar,
    ScaledValue, SciOxideError, algorithm_component,
};
type Detailed<S> = SpecialFunctionResult<S>;
#[derive(Clone, Copy)]
struct Pair<S> {
    v: S,
    d: S,
    scale: S,
    error: S,
    d_error: S,
}
fn empty<S: RealScalar>() -> Pair<S> {
    Pair {
        v: S::ZERO,
        d: S::ZERO,
        scale: S::ZERO,
        error: S::MAX,
        d_error: S::MAX,
    }
}
fn normalize<S: RealScalar>(p: &mut Pair<S>) {
    let magnitude = p.v.abs().max(p.d.abs());
    let threshold = S::MAX.sqrt();
    if magnitude > threshold || magnitude > S::ZERO && magnitude < S::ONE / threshold {
        p.v /= magnitude;
        p.d /= magnitude;
        p.error /= magnitude;
        p.d_error /= magnitude;
        p.scale += magnitude.ln();
    }
}
fn reciprocal_gamma<S: RealScalar>(x: S, w: &mut Work) -> Result<(S, S), SciOxideError> {
    if !w.step() {
        return Ok((S::ZERO, S::ZERO));
    }
    if x <= S::ZERO && x == x.round() {
        return Ok((S::ZERO, S::ZERO));
    }
    let (log, sign) = crate::gamma_beta::gamma::gamma_log_parts(x)?;
    Ok((sign, -log))
}
fn initial<S: RealScalar>(nu: S, v: bool, w: &mut Work) -> Result<Pair<S>, SciOxideError> {
    let ln2 = real::<S>(2.0)?.ln();
    let (s0, l0, s1, l1) = if v {
        let (g0, l0) = reciprocal_gamma(S::ONE + nu * S::HALF, w)?;
        let (g1, l1) = reciprocal_gamma((nu + S::ONE) * S::HALF, w)?;
        (
            -scioxide_core::sin_pi(nu * S::HALF)? * g0,
            l0 - nu * S::HALF * ln2,
            scioxide_core::cos_pi(nu * S::HALF)? * g1,
            l1 + (S::ONE - nu) * S::HALF * ln2,
        )
    } else {
        let (g0, l0) = reciprocal_gamma((S::ONE - nu) * S::HALF, w)?;
        let (g1, l1) = reciprocal_gamma(-nu * S::HALF, w)?;
        (
            g0,
            l0 + nu * S::HALF * ln2 + S::HALF * S::PI.ln(),
            -g1,
            l1 + (nu + S::ONE) * S::HALF * ln2 + S::HALF * S::PI.ln(),
        )
    };
    if !w.complete {
        return Ok(empty());
    }
    let scale = if s0 == S::ZERO {
        l1
    } else if s1 == S::ZERO {
        l0
    } else {
        l0.max(l1)
    };
    let value = if s0 == S::ZERO {
        S::ZERO
    } else {
        s0 * (l0 - scale).exp()
    };
    let derivative = if s1 == S::ZERO {
        S::ZERO
    } else {
        s1 * (l1 - scale).exp()
    };
    Ok(Pair {
        v: value,
        d: derivative,
        scale,
        error: real::<S>(32.0)? * S::EPSILON * value.abs(),
        d_error: real::<S>(32.0)? * S::EPSILON * derivative.abs(),
    })
}

/// DLMF 12.4.1/12.4.3/12.4.4 with a = -nu-1/2, differentiated termwise.
/// The rectangle |nu|<=2, |x|<=3 limits cancellation between the origin solutions.
/// All four series use independent seeds, including at subnormal coordinates.
fn origin_d(nu: f64, x: f64, work: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    let initial = initial(nu, false, work)?;
    if !work.complete {
        return Ok(initial);
    }
    let square = x * x;
    // Even/odd g and g', where D = exp(-x²/4) g. The even derivative
    // starts at the derivative of the quadratic term, avoiding division by x.
    let mut terms = [initial.v, x * initial.d, -nu * x * initial.v, initial.d];
    let mut sums = terms.map(Wide::from);
    let mut magnitudes = terms.map(f64::abs);
    for k in 1usize.. {
        if !work.step() {
            return Ok(empty());
        }
        let n = 2.0 * k as f64;
        let factors = [
            square * (n - 2.0 - nu) / (n * (n - 1.0)),
            square * (n - 1.0 - nu) / (n * (n + 1.0)),
            square * (n - nu) / (n * (n + 1.0)),
            square * (n - 1.0 - nu) / (n * (n - 1.0)),
        ];
        for i in 0..4 {
            terms[i] *= factors[i];
            sums[i] = sums[i].add(Wide::from(terms[i]));
            magnitudes[i] += terms[i].abs();
        }
        // For |nu|<=2 every subsequent ratio is bounded by x²/n and this
        // bound decreases. Bound each parity tail before combining solutions.
        let ratio = square / n;
        if ratio >= 1.0 {
            continue;
        }
        let tails = terms.map(|term| term.abs() * ratio / (1.0 - ratio));
        if (0..4).all(|i| tails[i] <= f64::EPSILON * magnitudes[i]) {
            // The origin carries 32 epsilon relative error; the remaining
            // allowance propagates coefficient, recurrence and sum rounding.
            let errors = core::array::from_fn::<_, 4, _>(|i| {
                tails[i]
                    + (32.0 + 8.0 * (n + 1.0)) * f64::EPSILON * magnitudes[i]
                    + 4.0 * f64::from_bits(1)
            });
            let value = sums[0].add(sums[1]);
            let derivative = sums[2].add(sums[3]).add(value.mul_f64(-x * 0.5));
            crate::support::check_finite(value.value())?;
            crate::support::check_finite(derivative.value())?;
            let error = errors[0] + errors[1];
            return Ok(Pair {
                v: value.value(),
                d: derivative.value(),
                scale: initial.scale - (x * 0.5) * (x * 0.5),
                error,
                d_error: errors[2] + errors[3] + x.abs() * 0.5 * error,
            });
        }
    }
    Err(crate::support::instability(
        "parabolic origin series did not converge",
    ))
}

fn combine<S: RealScalar>(left: Pair<S>, right: Pair<S>) -> Pair<S> {
    // A vanished reflection coefficient must not choose the scale of a
    // nonzero, exponentially smaller solution.
    if left.v == S::ZERO && left.d == S::ZERO && left.error == S::ZERO && left.d_error == S::ZERO {
        return right;
    }
    if right.v == S::ZERO
        && right.d == S::ZERO
        && right.error == S::ZERO
        && right.d_error == S::ZERO
    {
        return left;
    }
    let scale = left.scale.max(right.scale);
    let a = (left.scale - scale).exp();
    let b = (right.scale - scale).exp();
    Pair {
        v: a * left.v + b * right.v,
        d: a * left.d + b * right.d,
        scale,
        error: a * left.error
            + b * right.error
            + S::EPSILON * (a * left.v.abs() + b * right.v.abs()),
        d_error: a * left.d_error
            + b * right.d_error
            + S::EPSILON * (a * left.d.abs() + b * right.d.abs()),
    }
}

/// DLMF 12.9.1/12.9.2, with a=-nu-1/2. The conservative hand-off
/// x>=max(32,2(|nu|+1)) leaves the least term far below f64 precision.
/// Differentiate both the inverse-power series and its exponential prefactor.
fn asymptotic(nu: f64, x: f64, v: bool, work: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    let inverse = 1.0 / x;
    let power = if v { -nu - 1.0 } else { nu };
    let slope = (if v { x * 0.5 } else { -x * 0.5 }) + power * inverse;
    let mut term = 1.0;
    let mut sum = Wide::from(1.0);
    let mut derivative = Wide::ZERO;
    let mut absolute = 1.0;
    let mut d_absolute = 0.0;
    for k in 0usize.. {
        if !work.step() {
            return Ok(empty());
        }
        let n = 2.0 * k as f64;
        let coefficient = if v {
            (n + nu + 1.0) * (n + nu + 2.0)
        } else {
            -(n - nu) * (n + 1.0 - nu)
        };
        let next = term * (coefficient * inverse) * inverse / (n + 2.0);
        crate::support::check_finite(next)?;
        let tail = next.abs();
        let d_tail = (n + 2.0) * inverse * tail;
        if tail <= f64::EPSILON * absolute && d_tail <= f64::EPSILON * (1.0 + d_absolute) {
            // In this sector the first omitted terms estimate the remainder;
            // keep a factor four and recurrence rounding as a heuristic.
            let rounding = (32.0 + 8.0 * (k + 1) as f64) * f64::EPSILON;
            let error = 4.0 * tail + rounding * absolute;
            let d_error = 4.0 * d_tail
                + rounding * d_absolute
                + slope.abs() * error
                + 4.0 * f64::EPSILON * (slope.abs() * absolute + d_absolute);
            let value = sum.value();
            let d = derivative.add(sum.mul_f64(slope)).value();
            crate::support::check_finite(d)?;
            let gaussian = (x * 0.5) * (x * 0.5);
            return Ok(Pair {
                v: value,
                d,
                scale: (if v { gaussian } else { -gaussian })
                    + power * x.ln()
                    + if v {
                        0.5 * (2.0 / core::f64::consts::PI).ln()
                    } else {
                        0.0
                    },
                error,
                d_error,
            });
        }
        if k > 0 && next.abs() >= term.abs() {
            return Err(crate::support::instability(
                "parabolic asymptotic terms stopped decreasing",
            ));
        }
        term = next;
        sum = sum.add(Wide::from(term));
        derivative = derivative.add(Wide::from(-(n + 2.0) * inverse * term));
        absolute += term.abs();
        d_absolute += ((n + 2.0) * inverse * term).abs();
    }
    Err(crate::support::instability(
        "parabolic asymptotic series did not converge",
    ))
}

fn scaled_v(nu: f64, x: f64, w: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    if x >= 32.0_f64.max(2.0 * (nu.abs() + 1.0)) {
        return asymptotic(nu, x, true, w);
    }
    let mut state = initial(nu, true, w)?;
    if !w.complete {
        return Ok(state);
    }
    let mut position = 0.0;
    while position < x {
        if w.remaining < 24 {
            w.complete = false;
            return Ok(empty());
        }
        let h = (x - position).min(0.5 / (1.0 + position + nu.abs().sqrt()));
        let mut older = state.v;
        let mut old = h * state.d;
        let mut value = older + old;
        let mut derivative = state.d;
        let mut absolute = value.abs();
        let mut derivative_absolute = derivative.abs();
        let mut tail = 0.0;
        for k in 2..=25 {
            if !w.step() {
                return Ok(empty());
            }
            let ks = f64::try_from_usize_exact(k)?;
            let term =
                -position * h * old / ks - h * h * (ks + nu - 1.0) * older / (ks * (ks - 1.0));
            let d = ks * term / h;
            value += term;
            derivative += d;
            absolute += term.abs();
            derivative_absolute += d.abs();
            tail = term.abs() + d.abs();
            older = old;
            old = term;
        }
        let ratio = value.abs().max(derivative.abs()) / state.v.abs().max(state.d.abs());
        state.v = value;
        state.d = derivative;
        state.error = state.error * ratio + 32.0 * f64::EPSILON * absolute + tail;
        state.d_error = state.d_error * ratio + 32.0 * f64::EPSILON * derivative_absolute + tail;
        normalize(&mut state);
        position += h;
    }
    state.d += x * 0.5 * state.v;
    state.d_error += x * 0.5 * state.error;
    state.scale += (x * 0.5) * (x * 0.5);
    Ok(state)
}

fn negative_d(alpha: f64, x: f64, w: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    // Integral for D_-alpha on the positive real axis. A subtracted Gamma
    // density regularizes the near-zero-order singularity before t=u^4.
    if !w.step() {
        return Ok(empty());
    }
    let (gamma, _) = crate::gamma_beta::gamma::gamma_log_parts(alpha)?;
    if alpha >= 1.0 {
        return saddle_d(alpha, x, gamma, w);
    }
    // Truncate the subtracted Gamma tail below epsilon squared. The original
    // integrand decays still faster, as exp(-t²/2-x*t).
    let tail_log = -2.0 * f64::EPSILON.ln();
    let endpoint = tail_log.sqrt().sqrt();
    let mut intervals = 64usize;
    let mut previous: Option<(f64, f64)> = None;
    let mut result = empty();
    loop {
        if intervals + 1 > w.remaining {
            w.complete = false;
            return Ok(result);
        }
        let h = endpoint / f64::try_from_usize_exact(intervals)?;
        let mut sum = Wide::ZERO;
        let mut moment = Wide::ZERO;
        let mut absolute = 0.0;
        for j in 1..=intervals {
            if !w.step() {
                return Ok(result);
            }
            let u = h * f64::try_from_usize_exact(j)?;
            let t = (u * u) * (u * u);
            let weight = 4.0 * ((4.0 * alpha - 1.0) * u.ln()).exp();
            let exponent = -x * t - 0.5 * t * t;
            let delta = (1.0 - x) * t - 0.5 * t * t;
            let difference = if delta > 0.0 {
                exponent.exp() * (-(-delta).exp_m1())
            } else {
                (-t).exp() * delta.exp_m1()
            };
            let multiplier = if j == intervals {
                1.0
            } else if j.is_multiple_of(2) {
                2.0
            } else {
                4.0
            };
            let term = multiplier * weight * difference;
            sum = sum.add(Wide::from(term));
            absolute += term.abs();
            moment = moment.add(Wide::from(multiplier * weight * t * exponent.exp()));
        }
        let sum = sum.mul_f64(h / 3.0).value();
        let moment = moment.mul_f64(h / 3.0).value();
        let reciprocal = (-gamma).exp();
        if let Some((old, old_m)) = previous {
            // Refine to the compensated rule's attainable arithmetic floor,
            // using the full inter-rule difference in the published estimate.
            let rounding = 64.0 * f64::EPSILON * (1.0 + absolute * h / 3.0 * reciprocal);
            let d_rounding = 64.0 * f64::EPSILON * (moment * reciprocal).abs();
            let tail = 2.0 * (-tail_log).exp() * reciprocal;
            let e = (sum - old).abs() * reciprocal + rounding + tail;
            let de = (moment - old_m).abs() * reciprocal + d_rounding + tail;
            let value = 1.0 + sum * reciprocal;
            result = Pair {
                v: value,
                d: -x * 0.5 * value - moment * reciprocal,
                scale: -(x * 0.5) * (x * 0.5),
                error: e,
                d_error: de + x * 0.5 * e,
            };
            let fifteen = 15.0;
            if (sum - old).abs() / fifteen * reciprocal <= rounding
                && (moment - old_m).abs() / fifteen * reciprocal <= d_rounding.max(rounding)
            {
                return Ok(result);
            }
        }
        previous = Some((sum, moment));
        intervals = intervals.saturating_mul(2);
    }
}
/// exp(x)-1-x without cancellation; at |x|<1/8 the omitted x^11/11!
/// is below two epsilon times the leading x²/2 term.
fn exp_correction(x: f64) -> f64 {
    if x.abs() < 0.125 {
        let tail = (1.0 / 362880.0) + x * (1.0 / 3628800.0);
        let tail = (1.0 / 40320.0) + x * tail;
        let tail = (1.0 / 5040.0) + x * tail;
        let tail = (1.0 / 720.0) + x * tail;
        let tail = (1.0 / 120.0) + x * tail;
        let tail = (1.0 / 24.0) + x * tail;
        let tail = (1.0 / 6.0) + x * tail;
        x * x * (0.5 + x * tail)
    } else {
        x.exp_m1() - x
    }
}
fn saddle_d(alpha: f64, x: f64, gamma: f64, w: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    let root = x.hypot(2.0 * alpha.sqrt());
    let t = 2.0 * (alpha / (root + x));
    let curvature = (alpha + t * t).sqrt();
    let tail_log = -2.0 * f64::EPSILON.ln();
    let left = (2.0 * tail_log)
        .sqrt()
        .max(tail_log * 2.0_f64.sqrt() / alpha.sqrt());
    let right = (2.0 * tail_log).sqrt();
    // The log density and its moment are concave: alpha>=t² for x>=0.
    // Their endpoint tangents therefore bound both discarded integrals.
    let tail_at = |u: f64| {
        let v = u / curvature;
        let exponential = v.exp();
        let em = v.exp_m1();
        let density = (-alpha * exp_correction(v) - 0.5 * (t * em).powi(2)).exp();
        let slope = -em * (alpha + t * t * exponential) / curvature;
        (
            density / slope.abs(),
            density * t * exponential / (slope + 1.0 / curvature).abs(),
        )
    };
    let (left_tail, left_moment_tail) = tail_at(-left);
    let (right_tail, right_moment_tail) = tail_at(right);
    let tail = left_tail + right_tail;
    let moment_tail = left_moment_tail + right_moment_tail;
    let scale =
        alpha * t.ln() - x * t - 0.5 * t * t - gamma - (x * 0.5) * (x * 0.5) - curvature.ln();
    let mut intervals = 64usize;
    let mut previous: Option<(f64, f64)> = None;
    let mut result = empty();
    loop {
        if intervals + 1 > w.remaining {
            w.complete = false;
            return Ok(result);
        }
        let h = (left + right) / f64::try_from_usize_exact(intervals)?;
        let mut sum = Wide::ZERO;
        let mut moment = Wide::ZERO;
        for j in 0..=intervals {
            if !w.step() {
                return Ok(result);
            }
            let u = -left + h * f64::try_from_usize_exact(j)?;
            let v = u / curvature;
            let em = v.exp_m1();
            let exponent = -alpha * exp_correction(v) - 0.5 * (t * em) * (t * em);
            let density = exponent.exp();
            let multiplier = if j == 0 || j == intervals {
                1.0
            } else if j.is_multiple_of(2) {
                2.0
            } else {
                4.0
            };
            sum = sum.add(Wide::from(multiplier * density));
            moment = moment.add(Wide::from(multiplier * density * t * (em + 1.0)));
        }
        let sum = sum.mul_f64(h / 3.0).value();
        let moment = moment.mul_f64(h / 3.0).value();
        if let Some((old, old_m)) = previous {
            // Compensated summation leaves integrand evaluation as the main
            // rounding floor. The full inter-rule difference stays in the bound.
            let rounding = 64.0 * f64::EPSILON;
            let e = (sum - old).abs() + rounding * sum.abs() + tail;
            let de = (moment - old_m).abs() + rounding * moment.abs() + moment_tail;
            result = Pair {
                v: sum,
                d: -x * 0.5 * sum - moment,
                scale,
                error: e,
                d_error: de + x * 0.5 * e,
            };
            let fifteen = 15.0;
            if (sum - old).abs() / fifteen <= rounding * sum.abs()
                && (moment - old_m).abs() / fifteen <= rounding * moment.abs().max(sum.abs())
            {
                return Ok(result);
            }
        }
        previous = Some((sum, moment));
        intervals = intervals.saturating_mul(2);
    }
}
fn positive_d(nu: f64, x: f64, w: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    if x == 0.0 {
        return initial(nu, false, w);
    }
    if x >= 32.0_f64.max(2.0 * (nu.abs() + 1.0)) {
        return asymptotic(nu, x, false, w);
    }
    if nu < 0.0 {
        return negative_d(-nu, x, w);
    }
    let integer = nu == nu.round();
    let count = nu.floor();
    if count > f64::try_from_usize_exact(w.remaining)? {
        w.complete = false;
        return Ok(empty());
    }
    let count = count.try_to_usize_exact()?;
    let (mut previous, mut current, mut scale, start, mut previous_error, mut current_error) =
        if integer {
            (1.0, x, -(x * 0.5) * (x * 0.5), 0.0, 0.0, 0.0)
        } else {
            let base = nu - f64::try_from_usize_exact(count)? - 1.0;
            let p = negative_d(-base, x, w)?;
            if !w.complete {
                return Ok(p);
            }
            (
                p.v,
                x * 0.5 * p.v - p.d,
                p.scale,
                base,
                p.error,
                x * 0.5 * p.error + p.d_error + f64::EPSILON * (x * 0.5 * p.v.abs() + p.d.abs()),
            )
        };
    let steps = count + usize::from(!integer);
    if steps == 0 {
        return Ok(Pair {
            v: 1.0,
            d: -x * 0.5,
            scale,
            error: f64::EPSILON,
            d_error: f64::EPSILON * x.abs(),
        });
    }
    let threshold = f64::MAX.sqrt();
    for k in 1..steps {
        if !w.step() {
            return Ok(empty());
        }
        if current.abs().max(previous.abs()) > threshold / x.abs().max(1.0) {
            current /= threshold;
            previous /= threshold;
            current_error /= threshold;
            previous_error /= threshold;
            scale += threshold.ln();
        }
        let order = start + f64::try_from_usize_exact(k)?;
        let next = x * current - order * previous;
        let next_error = x.abs() * current_error
            + order.abs() * previous_error
            + 4.0 * f64::EPSILON * (x.abs() * current.abs() + order.abs() * previous.abs());
        previous = current;
        current = next;
        previous_error = current_error;
        current_error = next_error;
    }
    let derivative = nu * previous - x * 0.5 * current;
    let d_error = nu.abs() * previous_error
        + x * 0.5 * current_error
        + 4.0 * f64::EPSILON * (nu.abs() * previous.abs() + x * 0.5 * current.abs());
    Ok(Pair {
        v: current,
        d: derivative,
        scale,
        error: current_error,
        d_error,
    })
}
fn d_kernel(nu: f64, x: f64, w: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    let mut positive = positive_d(nu, x.abs(), w)?;
    if x >= 0.0 || !w.complete {
        return Ok(positive);
    }
    let cosine = scioxide_core::cos_pi(nu)?;
    positive.v *= cosine;
    positive.d = -positive.d * cosine;
    positive.error *= cosine.abs();
    positive.d_error *= cosine.abs();
    let (sign, gamma) = reciprocal_gamma(-nu, w)?;
    if sign == 0.0 || !w.complete {
        return Ok(positive);
    }
    let mut v = scaled_v(nu, -x, w)?;
    if !w.complete {
        return Ok(empty());
    }
    v.v *= sign;
    v.d = -v.d * sign;
    v.scale += core::f64::consts::PI.ln() + gamma;
    Ok(combine(positive, v))
}
fn v_kernel(nu: f64, x: f64, w: &mut Work) -> Result<Pair<f64>, SciOxideError> {
    let mut v = scaled_v(nu, x.abs(), w)?;
    if x >= 0.0 || !w.complete {
        return Ok(v);
    }
    // V(a,-x)=sin(pi*a) V(a,x)+cos(pi*a)^2 Gamma(a+1/2)/pi U(a,x).
    // In order notation: -cos(pi*nu) V_nu(x) - sin(pi*nu)/Gamma(nu+1) D_nu(x).
    let cosine = scioxide_core::cos_pi(nu)?;
    v.v *= -cosine;
    v.d *= cosine;
    v.error *= cosine.abs();
    v.d_error *= cosine.abs();
    let sine = scioxide_core::sin_pi(nu)?;
    let (sign, log) = reciprocal_gamma(nu + 1.0, w)?;
    if sine == 0.0 || sign == 0.0 || !w.complete {
        return Ok(v);
    }
    let mut d = positive_d(nu, -x, w)?;
    if !w.complete {
        return Ok(empty());
    }
    d.v *= -sine * sign;
    d.d *= sine * sign;
    d.error *= sine.abs();
    d.d_error *= sine.abs();
    d.scale += log;
    Ok(combine(v, d))
}
fn w_kernel<S: RealScalar>(a: S, x: S, w: &mut Work) -> Result<Pair<S>, SciOxideError> {
    use crate::bessel_airy::parabolic_coefficients::{W_EDGES, W_ZERO};
    let argument = a.abs().try_to_f64_exact()?;
    let segment = W_EDGES
        .windows(2)
        .position(|p| argument <= p[1])
        .unwrap_or(W_ZERO.len() - 1);
    if w.remaining < W_ZERO[segment].len() {
        w.complete = false;
        return Ok(empty());
    }
    let lo = real::<S>(W_EDGES[segment])?;
    let hi = real::<S>(W_EDGES[segment + 1])?;
    let (value, error) = crate::expansions::chebyshev(
        (real::<S>(2.0)? * a.abs() - lo - hi) / (hi - lo),
        &W_ZERO[segment],
        w,
    )?;
    let mut state = Pair {
        v: value,
        d: -S::HALF / value,
        scale: S::ZERO,
        error,
        d_error: S::HALF * error / (value * value) + S::EPSILON / value,
    };
    let mut position = S::ZERO;
    while position.abs() < x.abs() {
        if w.remaining < 48 {
            w.complete = false;
            return Ok(empty());
        }
        let h = (x - position)
            .abs()
            .min(S::HALF / (S::ONE + position.abs() + a.abs().sqrt()))
            .copysign(x);
        let q0 = a - position * position / real(4.0)?;
        let q1 = -position * S::HALF;
        // The step is linear in (y, y'): advance both unit solutions and carry
        // the accumulated error through the transition matrix, which grows
        // with the dominant solution.
        let (Some(first), Some(second)) = (
            taylor_step(q0, q1, h, S::ONE, S::ZERO, w)?,
            taylor_step(q0, q1, h, S::ZERO, S::ONE, w)?,
        ) else {
            return Ok(empty());
        };
        let (v, d) = (state.v, state.d);
        let value = first.value * v + second.value * d;
        let derivative = first.derivative * v + second.derivative * d;
        let rounding = real::<S>(32.0)? * S::EPSILON;
        let error = first.value.abs() * state.error
            + second.value.abs() * state.d_error
            + rounding * (first.absolute * v.abs() + second.absolute * d.abs());
        let d_error = first.derivative.abs() * state.error
            + second.derivative.abs() * state.d_error
            + rounding * (first.d_absolute * v.abs() + second.d_absolute * d.abs());
        state = Pair {
            v: value,
            d: derivative,
            scale: state.scale,
            error,
            d_error,
        };
        position += h;
    }
    Ok(state)
}

/// One step of the Taylor series of y'' = (x²/4 − a) y from (v0, d0).
struct TaylorStep<S> {
    value: S,
    derivative: S,
    absolute: S,
    d_absolute: S,
}

/// Twenty-five terms of the step from `position`, where `q0 = a − x²/4` and `q1 = −x/2`; `None`
/// once the work allowance is exhausted.
fn taylor_step<S: RealScalar>(
    q0: S,
    q1: S,
    h: S,
    v0: S,
    d0: S,
    w: &mut Work,
) -> Result<Option<TaylorStep<S>>, SciOxideError> {
    let mut t0 = S::ZERO;
    let mut t1 = S::ZERO;
    let mut t2 = v0;
    let mut t3 = h * d0;
    let mut value = v0 + t3;
    let mut derivative = d0;
    let mut absolute = v0.abs() + t3.abs();
    let mut d_absolute = derivative.abs();
    for k in 2..=25 {
        if !w.step() {
            return Ok(None);
        }
        let ks = S::try_from_usize_exact(k)?;
        let term = (q0 * h * h * t2 + q1 * h * h * h * t1 - h * h * h * h * t0 / real::<S>(4.0)?)
            / (ks * (ks - S::ONE));
        let d = ks * term / h;
        value += term;
        derivative += d;
        absolute += term.abs();
        d_absolute += d.abs();
        t0 = t1;
        t1 = t2;
        t2 = t3;
        t3 = term;
    }
    Ok(Some(TaylorStep {
        value,
        derivative,
        absolute,
        d_absolute,
    }))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the parabolic cylinder function `D_ν(x)` for real order `ν` and real `x`.

`D_ν(x) = U(−ν − 1/2, x)` is the solution of Weber's equation

```text
y″ + (ν + 1/2 − x²/4) y = 0
```

that decays as `D_ν(x) ~ x^ν exp(−x²/4)` for `x → +∞` (DLMF 12.2, 12.9). It is entire in `ν`
and `x`, so every finite `ν` and `x` is accepted. For a nonnegative integer `n`,
`D_n(x) = exp(−x²/4) Heₙ(x)` with the Hermite polynomial `Heₙ` (DLMF 12.7.2).

For `|ν| ≤ 2` and `|x| ≤ 3` the function sums compensated power series about the origin
(DLMF 12.4). For `|x| ≥ max(32, 2(|ν| + 1))` it sums the asymptotic expansion (DLMF 12.9).
Elsewhere, negative orders use quadrature of the integral representation (DLMF 12.5), and
positive orders recur upward in the order (DLMF 12.8) at a cost of about `ν` steps. Negative `x`
uses the connection formula with `V` (DLMF 12.2). When `|D_ν(x)|` overflows,
[`parabolic_d_with_options`](crate::parabolic_d_with_options) still returns the scaled and
logarithmic forms.

# Examples

```
// D₀(x) = exp(−x²/4)
let d = scioxide_special::parabolic_d(0.0_f64, 1.0)?;
assert!((d - (-0.25_f64).exp()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values for `|ν| ≤ 20` and `|x| ≤ 10⁴` in `f32` and `f64`, including results that overflow into
the scaled and logarithmic forms; near zeros they bound the absolute error.
[`parabolic_d_with_options`](crate::parabolic_d_with_options) returns a heuristic error
estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|D_ν(x)|` exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the work exceeds the budget of
  [`ParabolicCylinderOptions`](crate::ParabolicCylinderOptions) (262144 units by default).
  The upward recurrence takes about `ν` steps, so larger positive orders need
  `|x| ≥ 2(ν + 1)`.

# References

NIST DLMF, Chapter 12. <https://dlmf.nist.gov/12>"###)]

    pub parabolic_d [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: ParabolicCylinderOptions;
    detailed parabolic_d_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(parabolic_d));
        evaluate(order, x, options, 0, family, context)
    }
    scoped pub parabolic_d_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the second parabolic cylinder function `V_ν(x) = V(−ν − 1/2, x)` for real `ν` and `x`.

`V(a, x)` is the solution of Weber's equation `y″ − (x²/4 + a) y = 0` in the normalization of
DLMF 12.2, with order `ν = −a − 1/2` as in [`parabolic_d`](crate::parabolic_d). Its values at
the origin are

```text
V_ν(0)  = −2^(−ν/2) sin(πν/2) / Γ(1 + ν/2),
V_ν′(0) = 2^((1−ν)/2) cos(πν/2) / Γ((ν + 1)/2),
```

where `1/Γ` is zero at the poles of `Γ`. `V_ν(x) ~ √(2/π) x^(−ν−1) exp(x²/4)` for `x → +∞`
(DLMF 12.9). Every finite `ν` and `x` is accepted.

For `|x| ≥ max(32, 2(|ν| + 1))` the function sums the asymptotic expansion (DLMF 12.9).
Otherwise it continues the solution from the origin with Taylor steps of the differential
equation. Negative `x` uses the connection formula with `D_ν` (DLMF 12.2). When `|V_ν(x)|`
overflows, [`parabolic_v_with_options`](crate::parabolic_v_with_options) still returns the
scaled and logarithmic forms.

# Examples

```
// V₀(1) = 0.742538470703127…
let v = scioxide_special::parabolic_v(0.0_f64, 1.0)?;
assert!((v - 0.742_538_470_703_127).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values for `|ν| ≤ 20` and `|x| ≤ 10⁴` in `f32` and `f64`, including results that overflow into
the scaled and logarithmic forms; near zeros they bound the absolute error.
[`parabolic_v_with_options`](crate::parabolic_v_with_options) returns a heuristic error
estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|V_ν(x)|` exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the work exceeds the budget of
  [`ParabolicCylinderOptions`](crate::ParabolicCylinderOptions) (262144 units by default).

# References

NIST DLMF, Chapter 12. <https://dlmf.nist.gov/12>"###)]

    pub parabolic_v [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: ParabolicCylinderOptions;
    detailed parabolic_v_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(parabolic_v));
        evaluate(order, x, options, 1, family, context)
    }
    scoped pub parabolic_v_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the real parabolic cylinder function `W(a, x)` for `|a| ≤ 5` and `|x| ≤ 5`.

The argument `order` is the parameter `a`. `W(a, x)` solves

```text
y″ + (x²/4 − a) y = 0
```

with the standard normalization at the origin (DLMF 12.14)

```text
W(a, 0)  = 2^(−3/4) |Γ(1/4 + ia/2) / Γ(3/4 + ia/2)|^(1/2),
W′(a, 0) = −1 / (2 W(a, 0)).
```

The function evaluates `W(a, 0)` from Chebyshev expansions in `a` and continues the solution to
`x` with Taylor steps of the differential equation.

# Examples

```
// W(0, 0) = 2^(−3/4) √(Γ(1/4)/Γ(3/4)) = 1.0227656721131687…
let w = scioxide_special::parabolic_w(0.0_f64, 0.0)?;
assert!((w - 1.022_765_672_113_168_7).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`. [`parabolic_w_with_options`](crate::parabolic_w_with_options)
returns a heuristic error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `|a| > 5` or `|x| > 5`.

# References

NIST DLMF, Chapter 12. <https://dlmf.nist.gov/12>"###)]

    pub parabolic_w [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: ParabolicCylinderOptions;
    detailed parabolic_w_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(parabolic_w));
        evaluate(order, x, options, 2, family, context)
    }
    scoped pub parabolic_w_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

fn evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: ParabolicCylinderOptions,
    kind: u8,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<ParabolicEvaluation<S>, SciOxideError> {
    if !order.is_finite() || !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and argument must be finite",
        ));
    }
    let nu = order.try_to_f64_exact()?;
    let argument = x.try_to_f64_exact()?;
    if kind == 2 && nu.abs() > 5.0 {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("order"),
            scioxide_core::DomainReason::OutsideDomain,
            "W requires |a| <= 5 and |x| <= 5",
        ));
    }
    if kind == 2 && argument.abs() > 5.0 {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "W requires |a| <= 5 and |x| <= 5",
        ));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let series = kind == 0
        && nu.abs() <= 2.0
        && argument.abs() <= 3.0
        && argument != 0.0
        && !(nu >= 0.0 && nu == nu.round());
    let computed = if series {
        origin_d(nu, argument, &mut work)
            .map(|pair| (pair, algorithm_component!("budgeted_origin_series")))
    } else {
        match kind {
            0 => d_kernel(nu, argument, &mut work),
            1 => v_kernel(nu, argument, &mut work),
            _ => w_kernel(nu, argument, &mut work),
        }
        .map(|pair| {
            let method = if kind != 2 && argument.abs() >= 32.0_f64.max(2.0 * (nu.abs() + 1.0)) {
                algorithm_component!("budgeted_large_argument_asymptotic")
            } else {
                algorithm_component!("budgeted_recurrence_integral")
            };
            (pair, method)
        })
    };
    work.check()?;
    let (pair, method) = computed?;
    report(pair, &work, family, method)
}
/// Evaluation of a real parabolic cylinder function.
type ParabolicEvaluation<S> = Evaluation<S, Either<Form<S>, Recurrence>>;

/// A recurrence pair `v·exp(scale)` with its derivative.
#[derive(Clone, Copy)]
struct Recurrence {
    p: Pair<f64>,
    scalar: f64,
    /// Cached only when ordinary projection already needed the logarithm.
    logarithm: Option<f64>,
}

impl Recurrence {
    /// exp(scale) rounds once and carries the rounding of the scale itself.
    fn scale_rounding(&self) -> f64 {
        32.0 * f64::EPSILON * (1.0 + self.p.scale.abs())
    }

    fn logarithm(&self) -> Option<f64> {
        self.logarithm
            .or_else(|| (self.p.v != 0.0).then(|| RealScalar::ln(self.p.v.abs()) + self.p.scale))
    }
}

fn report<S: RealScalar>(
    p: Pair<f64>,
    work: &Work,
    family: AlgorithmComponent,
    method: AlgorithmComponent,
) -> Result<ParabolicEvaluation<S>, SciOxideError> {
    let algorithm = AlgorithmId::from_components(family, method, 1);
    if !work.complete {
        return Ok(crate::expansions::bounded(
            S::ZERO,
            crate::result::Bound::derived(S::MAX),
            algorithm,
            work.used,
            false,
        )?
        .map_form(Either::Left));
    }
    crate::support::check_finite(p.v)?;
    crate::support::check_finite(p.scale)?;
    let logarithm = (p.scale != 0.0 && p.v != 0.0).then(|| RealScalar::ln(p.v.abs()) + p.scale);
    let scalar = if p.scale == 0.0 {
        p.v
    } else {
        logarithm.map_or(0.0, |l| RealScalar::exp(l).copysign(p.v))
    };
    let recurrence = Recurrence {
        p,
        scalar,
        logarithm,
    };
    let value = S::try_from_f64_lossy(scalar).ok();
    Ok(Evaluation::from_parts(
        value,
        if value.is_some() {
            ScalarRepresentation::Finite
        } else {
            ScalarRepresentation::Overflow
        },
        SpecialFunctionTermination::Converged,
        EvaluationCounts::from_iterations(work.used),
        algorithm,
        Some(Either::Right(recurrence)),
    ))
}

impl<S: RealScalar> Publish<S> for Recurrence {
    fn publish(self, value: Option<S>) -> Result<Forms<S>, SciOxideError> {
        let p = self.p;
        let logarithm = self.logarithm();
        let scale_rounding = self.scale_rounding();
        let relative = if p.v == 0.0 {
            0.0
        } else {
            p.error / p.v.abs() + scale_rounding
        };
        let max = S::MAX.try_to_f64_exact()?;
        let scalar = self.scalar;
        let ordinary = value
            .map(|value| {
                let absolute = if p.error == 0.0 {
                    0.0
                } else {
                    RealScalar::exp(RealScalar::ln(p.error) + p.scale)
                } + scale_rounding * scalar.abs();
                let error = S::try_from_f64_lossy(absolute.min(max))?
                    + S::EPSILON * value.abs()
                    + real::<S>(4.0)? * S::ZERO.next_after(S::ONE);
                ValueWithError::new(value, error, None, EstimateReliability::Heuristic)
            })
            .transpose()?;
        let sign = if p.v < 0.0 {
            -S::ONE
        } else if p.v > 0.0 {
            S::ONE
        } else {
            S::ZERO
        };
        let log = if let Some(l) = logarithm {
            LogValue::from_log(sign, S::try_from_f64_lossy(l)?)?
        } else {
            LogValue::from_value(S::ZERO)?
        };
        let rel = S::try_from_f64_lossy(relative.min(max))? + S::EPSILON;
        let scaled = if let Some(l) = logarithm {
            // Reduce the binary64 logarithm before narrowing the significand;
            // narrowing a large log first loses O(epsilon*|log|) relatively.
            ScaledValue::from_log(if p.v < 0.0 { -1.0 } else { 1.0 }, l)
                .ok()
                .map(|wide| {
                    ScaledValue::from_parts(
                        S::try_from_f64_lossy(wide.component_significands())?,
                        wide.scale_exponent(),
                    )
                })
                .transpose()?
        } else {
            Some(ScaledValue::from_value(S::ZERO)?)
        };
        let derivative_scalar = if p.scale == 0.0 {
            p.d
        } else if p.d == 0.0 {
            0.0
        } else {
            RealScalar::exp(RealScalar::ln(p.d.abs()) + p.scale).copysign(p.d)
        };
        let derivative = if derivative_scalar.is_finite() && derivative_scalar.abs() <= max {
            let value = S::try_from_f64_lossy(derivative_scalar)?;
            let error = if p.d_error == 0.0 {
                0.0
            } else {
                RealScalar::exp(RealScalar::ln(p.d_error) + p.scale)
            } + scale_rounding * derivative_scalar.abs();
            Some(ValueWithError::new(
                value,
                S::try_from_f64_lossy(error.min(max))?
                    + S::EPSILON * value.abs()
                    + real::<S>(4.0)? * S::ZERO.next_after(S::ONE),
                None,
                EstimateReliability::Heuristic,
            )?)
        } else {
            None
        };
        let scaled = scaled
            .map(|value| crate::result::scaled_estimate(value, rel, EstimateReliability::Heuristic))
            .transpose()?;
        let log_bound = crate::result::log_error(rel, log.log_magnitude().unwrap_or(S::ZERO));
        let logarithmic = log_bound
            .map(|error| ValueWithError::new(log, error, None, EstimateReliability::Heuristic))
            .transpose()?;
        Ok(Forms {
            value: ordinary,
            scaled,
            logarithmic,
            derivative,
        })
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the parabolic cylinder function `D_n(z)` for a nonnegative integer order `n` and complex `z`.

```text
D_n(z) = exp(−z²/4) Heₙ(z) = 2^(−n/2) exp(−z²/4) Hₙ(z/√2),
```

where `Heₙ` and `Hₙ` are the Hermite polynomials (DLMF 12.7.2, §18.3). `D_n` is entire in
`z`. For real non-integer orders use [`parabolic_d`](crate::parabolic_d).

The function runs the three-term recurrence `He_{k+1}(z) = z Heₖ(z) − k He_{k−1}(z)` with
rescaling, and multiplies by `exp(−z²/4)` with the phase `−Im(z²)/4` split into high and low
parts. The cost is `n + 1` steps. When `|D_n(z)|` overflows,
[`parabolic_d_complex_with_options`](crate::parabolic_d_complex_with_options) still returns the
scaled form.

# Examples

```
// D₀(0) = 1
let d = scioxide_special::parabolic_d_complex(0, scioxide_core::Complex::new(0.0_f64, 0.0))?;
assert!((d.re - 1.0).abs() < 1e-14 && d.im.abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`LossOfSignificance`) if `z²/4` is not representable in `f64`
  (about `|z| > 10¹⁵⁴`) and the result does not underflow to zero.
- `NumericalInstability` (`Overflow`) if `|D_n(z)|` exceeds the range of the scalar type.
- `ConvergenceFailure` (`IterationLimit`) if the work exceeds the budget of
  [`ParabolicCylinderOptions`](crate::ParabolicCylinderOptions) (262144 units by default).

# References

NIST DLMF, Chapter 12. <https://dlmf.nist.gov/12>"###)]

    pub parabolic_d_complex [S: RealScalar] (order: u32,
        z: scioxide_core::Complex<S>) -> scioxide_core::Complex<S>
    where [scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>, ];
    options options: ParabolicCylinderOptions;
    detailed parabolic_d_complex_with_options -> SpecialFunctionResult<scioxide_core::Complex<S>>;
    evaluate |options, context| {
        parabolic_d_complex_evaluate(order, z, options, context)
    }
    scoped pub parabolic_d_complex_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluate before projecting the value or building the detailed result.
fn parabolic_d_complex_evaluate<S: RealScalar>(
    order: u32,
    z: scioxide_core::Complex<S>,
    options: ParabolicCylinderOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<crate::bessel_airy::complex_bessel::BesselEvaluation<S>, SciOxideError>
where
    scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let computed = complex_integer(order, z, &mut work);
    work.check()?;
    let (value, derivative) = computed?;
    crate::bessel_airy::complex_bessel::report(
        value,
        Some(derivative),
        work.used,
        work.complete,
        algorithm_component!("parabolic_d_complex"),
    )
}

fn complex_integer<S: RealScalar>(
    order: u32,
    z: scioxide_core::Complex<S>,
    w: &mut Work,
) -> Result<
    (
        crate::bessel_airy::complex_bessel::ExponentialEstimate,
        crate::bessel_airy::complex_bessel::ExponentialEstimate,
    ),
    SciOxideError,
> {
    use crate::bessel_airy::complex_bessel::ExponentialEstimate;
    use scioxide_core::Complex;
    if !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "argument must be finite",
        ));
    }
    let z = Complex::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    let empty = ExponentialEstimate {
        value: Complex::new(0.0, 0.0),
        scale: 0.0,
        error: f64::MAX,
    };
    if !w.step() {
        return Ok((empty, empty));
    }
    let mut scale = (z.im - z.re) * (z.im * 0.25 + z.re * 0.25);
    if scale == f64::NEG_INFINITY {
        return Ok((
            ExponentialEstimate {
                error: 0.0,
                ..empty
            },
            ExponentialEstimate {
                error: 0.0,
                ..empty
            },
        ));
    }
    let phase_angle = (-z.re * 0.5) * z.im;
    if !phase_angle.is_finite() || !scale.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::LossOfSignificance,
            "quadratic exponent exceeds the working phase representation",
        ));
    }
    let low = RealScalar::mul_add(-z.re * 0.5, z.im, -phase_angle);
    let (s, c) = RealScalar::sin_cos(phase_angle);
    let mut phase = Complex::new(c, s);
    if low != 0.0 {
        let (s, c) = RealScalar::sin_cos(low);
        phase *= Complex::new(c, s);
    }
    let mut previous = Complex::new(0.0, 0.0);
    let mut current = Complex::new(1.0, 0.0);
    let threshold = RealScalar::sqrt(f64::MAX);
    for n in 0..order {
        if !w.step() {
            return Ok((empty, empty));
        }
        if current.norm().max(previous.norm()) > threshold / z.norm().max(1.0) {
            current = current.unscale(threshold);
            previous = previous.unscale(threshold);
            scale += RealScalar::ln(threshold);
        }
        let next = z * current - previous.scale(f64::from(n));
        previous = current;
        current = next;
    }
    let derivative = previous.scale(f64::from(order)) - z.scale(0.5) * current;
    let error = if z == Complex::new(0.0, 0.0) && !order.is_multiple_of(2) {
        0.0
    } else {
        64.0 * f64::EPSILON * (f64::from(order) + 1.0) * (current.norm() + previous.norm())
    };
    Ok((
        ExponentialEstimate {
            value: phase * current,
            scale,
            error,
        },
        ExponentialEstimate {
            value: phase * derivative,
            scale,
            error: error * (1.0 + f64::from(order) + z.norm()),
        },
    ))
}
