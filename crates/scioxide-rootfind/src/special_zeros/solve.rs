use super::evaluation::{C, Provider, widen};
use super::*;
use crate::context::Session;
use crate::invalid;
use crate::scalar::correction::newton;
use crate::scalar::{BracketState, Sample};
use RootTermination as T;
use SpecialZeroBranch as B;
use SpecialZeroFamily as F;
use scioxide_core::{
    Argument, BudgetKind, EvaluationRole, NumericScalar, ProblemToken, SolveBudgets,
};
const PI: f64 = core::f64::consts::PI;

/// Validate a request; `[family, branch, start, count]` name its fields in errors.
fn validate<S: RealScalar>(
    p: &Request<'_, S>,
    [family, branch, start, count]: [Argument; 4],
) -> Result<(), SciOxideError> {
    p.start
        .checked_add(p.count)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("last_index")))?;
    if p.start.saturating_add(p.count) as u128 > (1_u128 << 52) {
        return Err(invalid(
            count,
            scioxide_core::InvalidInputReason::OutOfRange,
            "index exceeds exact binary64 indexing",
        ));
    }
    let f = widen(p.family)?;
    match f {
        F::BesselJ(n) if !n.is_finite() || n <= -0.5 => {
            return Err(invalid(
                family,
                scioxide_core::InvalidInputReason::OutOfRange,
                "J order must be finite and greater than -1/2",
            ));
        }
        F::BesselY(n) | F::BesselJPrime(n) | F::BesselYPrime(n) if !n.is_finite() || n < 0.0 => {
            return Err(invalid(
                family,
                scioxide_core::InvalidInputReason::OutOfRange,
                "order must be finite and nonnegative",
            ));
        }
        _ => {}
    }
    match p.branch {
        B::Origin => {
            if p.start != 0 || p.count > 1 || origin_multiplicity(f).is_none() {
                return Err(invalid(
                    branch,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "origin requires a supported family, start=0 and count<=1",
                ));
            }
        }
        B::Real => {
            if p.start == 0 {
                return Err(invalid(
                    start,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "nonzero sequences use one-based indices",
                ));
            }
        }
        B::Complex { negate, rotate, .. } => {
            if p.start == 0 {
                return Err(invalid(
                    start,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "nonzero sequences use one-based indices",
                ));
            }
            let supported = match f {
                F::BesselY(n) | F::BesselYPrime(n) => (n == 0.0 || n == 1.0) && !negate && !rotate,
                F::AiryBi | F::AiryBiPrime => !negate && !rotate,
                F::Erf => !rotate,
                F::FresnelC | F::FresnelS | F::FresnelCPrime | F::FresnelSPrime => true,
                _ => false,
            };
            if !supported {
                return Err(invalid(
                    branch,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "unsupported complex zero family or symmetry",
                ));
            }
        }
    }
    Ok(())
}
fn origin_multiplicity(f: F<f64>) -> Option<usize> {
    use KelvinZeroFunction as K;
    match f {
        F::Erf | F::FresnelC => Some(1),
        F::FresnelS => Some(3),
        F::FresnelSPrime => Some(2),
        F::BesselJPrime(0.0) => Some(1),
        F::Kelvin(K::Bei) => Some(2),
        F::Kelvin(K::BerPrime) => Some(3),
        F::Kelvin(K::BeiPrime) => Some(1),
        _ => None,
    }
}
fn no_zeros(f: F<f64>, branch: B) -> bool {
    branch == B::Real && matches!(f, F::Erf | F::FresnelC | F::FresnelS)
}
pub(super) struct Attempt<S: RealScalar> {
    zero: Option<SpecialZero<S>>,
    termination: T,
    candidate: Option<Complex<S>>,
}
fn stopped<S: RealScalar>(termination: T, candidate: Option<Complex<S>>) -> Attempt<S> {
    Attempt {
        zero: None,
        termination,
        candidate,
    }
}
fn stop(error: &SciOxideError) -> Option<T> {
    match error.view() {
        scioxide_core::SciOxideErrorView::ConvergenceFailure { reason, .. } => match reason {
            scioxide_core::ConvergenceReason::IterationLimit => {
                Some(T::WorkLimit(BudgetKind::Iteration))
            }
            scioxide_core::ConvergenceReason::EvaluationLimit => {
                Some(T::WorkLimit(BudgetKind::Evaluation))
            }
            scioxide_core::ConvergenceReason::ResourceLimit(kind) => Some(T::WorkLimit(kind)),
            _ => None,
        },
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: scioxide_core::ExecutionStopReason::Cancelled,
        } => Some(T::Cancelled),
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: scioxide_core::ExecutionStopReason::DeadlineReached,
        } => Some(T::TimeLimit),
        _ => None,
    }
}
/// One counted real function value.
fn sample(provider: Provider, x: f64, c: &mut SolveContext<'_, '_>) -> Result<f64, SciOxideError> {
    Ok(provider.evaluate(C::new(x, 0.0), c)?.value.re)
}
/// Whether the Newton candidate `next` moves at most `2ε max(|x|, 1)` from `x`.
fn converged<V: scioxide_core::NumericScalar<Real = f64>>(x: V, next: V) -> bool {
    (next - x).abs() <= 2.0 * f64::EPSILON * x.abs().max(1.0)
}
/// A located real zero, or the stop that ended its search with the last
/// estimate of this zero, if any.
enum RealRoot {
    Found(Located),
    Stopped {
        termination: T,
        estimate: Option<f64>,
    },
}
impl RealRoot {
    fn complete(x: f64, bracket: Option<Interval<f64>>) -> Self {
        Self::Found(Located {
            z: C::new(x, 0.0),
            bracket,
            evaluation: None,
        })
    }
    fn evaluated(
        x: f64,
        bracket: Interval<f64>,
        evaluation: super::evaluation::Evaluation,
    ) -> Self {
        Self::Found(Located {
            z: C::new(x, 0.0),
            bracket: Some(bracket),
            evaluation: Some((C::new(x, 0.0), evaluation)),
        })
    }
    fn iteration_limit(estimate: f64) -> Self {
        Self::Stopped {
            termination: T::IterationLimit,
            estimate: Some(estimate),
        }
    }
}
/// A located zero before the final check: its coordinate, its sign bracket,
/// if any, and any evaluation at an exact coordinate.
struct Located {
    z: C,
    bracket: Option<Interval<f64>>,
    evaluation: Option<(C, super::evaluation::Evaluation)>,
}
/// Safeguarded Newton refinement of a sign bracket with paired value and
/// derivative evaluations: each step counts one iteration and one `Function`
/// evaluation, and a further proposal unless the Newton correction already
/// meets `2ε max(|x|, 1)`. Proposals outside the bracket bisect it; a step
/// landing on an endpoint completes there; `steps` exhausted is an iteration limit.
/// A `known` evaluation at the starting coordinate is reused, not repeated.
fn refine(
    provider: Provider,
    bracket: Interval<f64>,
    signs: (f64, f64),
    seed: f64,
    mut known: Option<(f64, super::evaluation::Evaluation)>,
    steps: Budget,
    c: &mut SolveContext<'_, '_>,
) -> Result<RealRoot, SciOxideError> {
    if signs.0 == 0.0 {
        return Ok(RealRoot::complete(bracket.lower(), Some(bracket)));
    }
    if signs.1 == 0.0 {
        return Ok(RealRoot::complete(bracket.upper(), Some(bracket)));
    }
    if !crate::initialization::brackets_root(signs.0, signs.1) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "indexed-zero interval does not bracket a sign change",
        ));
    }
    let mut state = BracketState {
        lower: Sample::new(bracket.lower(), signs.0),
        upper: Sample::new(bracket.upper(), signs.1),
    };
    let interval = |state: BracketState<f64>| Interval::new(state.lower.x, state.upper.x);
    let session = Session::new(c);
    let mut x = seed.clamp(bracket.lower(), bracket.upper());
    for _ in 0..steps.get() {
        session.iteration_only()?;
        let e = match known.take() {
            Some((at, e)) if at == x => e,
            _ => session
                .invoke_context(EvaluationRole::Function, |c| provider.at(C::new(x, 0.0), c))?,
        };
        let value = e.value.re;
        let proposal = newton(x, value, e.derivative.re, None).ok().flatten();
        // A checked equation at the exact returned coordinate still has to
        // pass independent verification, including its evaluation uncertainty.
        if value == 0.0 || proposal.is_some_and(|next| converged(x, next)) {
            return Ok(RealRoot::evaluated(x, interval(state)?, e));
        }
        state = state.update(Sample::new(x, value))?;
        session.proposal()?;
        (x, _) = state.safeguard(proposal)?;
        if x == state.lower.x || x == state.upper.x {
            return Ok(RealRoot::complete(x, Some(interval(state)?)));
        }
    }
    Ok(RealRoot::iteration_limit(x))
}

/// Evaluate the two nearby Airy endpoints by its ODE Taylor polynomial. The
/// fourth-derivative remainder and propagated input errors must prove each
/// numerical sign; otherwise the ordinary provider establishes the bracket.
fn airy_local_bracket(
    provider: Provider,
    x: f64,
    e: super::evaluation::Evaluation,
    cell: Interval<f64>,
    c: &mut SolveContext<'_, '_>,
) -> Result<Option<Interval<f64>>, SciOxideError> {
    let v = e.value.re;
    let d = e.derivative.re;
    let correction = (v.abs() + e.error) / d.abs();
    let radius = (8.0 * correction).max(64.0 * f64::EPSILON * x.abs().max(1.0));
    let points = [x - radius, x + radius];
    let h = (points[0] - x).abs().max((points[1] - x).abs());
    let prime = matches!(provider.family, F::AiryAiPrime | F::AiryBiPrime);
    let m = (x.abs() + h + if prime { 1.0 / (x.abs() - h) } else { 0.0 }).max(1.0);
    if !h.is_finite()
        || d.abs() <= 2.0 * e.derivative_error
        || h >= x.abs()
        || m * h > 0.25
        || points[0] <= cell.lower()
        || points[1] >= cell.upper()
    {
        return Ok(None);
    }
    let second = x * v + if prime { d / x } else { 0.0 };
    let third = x * d + if prime { 2.0 * v } else { v };
    // Gronwall bounds |y|+|y'| by twice its initial value on this cell.
    let remainder = h.powi(4)
        * ((v.abs() + d.abs() + e.error + e.derivative_error) * (4.0 + (x.abs() + h).powi(2))
            / 12.0);
    let uncertainty = (e.error + h * e.derivative_error) * (1.0 + m * h).powi(3)
        + remainder
        + 16.0 * f64::EPSILON * (v.abs() + h * d.abs() + h * h * second.abs());
    let mut signs = [0.0; 2];
    let session = Session::new(c);
    for (sign, point) in signs.iter_mut().zip(points) {
        session.admit(EvaluationRole::Function)?;
        session.iteration_only()?;
        let displacement = point - x;
        *sign = v + displacement * (d + displacement * (0.5 * second + displacement * third / 6.0));
    }
    if signs.iter().all(|value| value.abs() > uncertainty)
        && signs[0].is_sign_negative() != signs[1].is_sign_negative()
    {
        Ok(Some(Interval::new(points[0], points[1])?))
    } else {
        Ok(None)
    }
}

fn airy_root(
    provider: Provider,
    index: usize,
    steps: Budget,
    c: &mut SolveContext<'_, '_>,
) -> Result<RealRoot, SciOxideError> {
    let f = provider.family;
    let seed = airy_seed(f, index);
    let lower = (seed + airy_seed(f, index + 1)) * 0.5;
    let upper = if index == 1 {
        0.0
    } else {
        (seed + airy_seed(f, index - 1)) * 0.5
    };
    let cell = Interval::new(lower, upper)?;
    let mut x = seed;
    let session = Session::new(c);
    for used in 0..steps.get() {
        session.iteration_only()?;
        let e =
            session.invoke_context(EvaluationRole::Function, |c| provider.at(C::new(x, 0.0), c))?;
        let next = newton(x, e.value.re, e.derivative.re, None).ok().flatten();
        if next.is_some_and(|next| converged(x, next)) {
            if let Some(bracket) = airy_local_bracket(provider, x, e, cell, c)? {
                return Ok(RealRoot::evaluated(x, bracket, e));
            }
            return airy_fallback(provider, cell, (x, e), steps.get() - used - 1, c);
        }
        session.proposal()?;
        match next.filter(|&next| next > lower && next < upper) {
            Some(next) => x = next,
            None => return airy_fallback(provider, cell, (x, e), steps.get() - used - 1, c),
        }
    }
    Ok(RealRoot::iteration_limit(x))
}
fn airy_seed(f: F<f64>, index: usize) -> f64 {
    super::seeds::airy(f, index)
}
/// Bracketed refinement over the Airy cell from the last Newton point `x`,
/// reusing its evaluation `e`.
fn airy_fallback(
    provider: Provider,
    cell: Interval<f64>,
    (x, e): (f64, super::evaluation::Evaluation),
    remaining: usize,
    c: &mut SolveContext<'_, '_>,
) -> Result<RealRoot, SciOxideError> {
    if remaining == 0 {
        return Ok(RealRoot::iteration_limit(x));
    }
    let signs = (
        sample(provider, cell.lower(), c)?,
        sample(provider, cell.upper(), c)?,
    );
    refine(
        provider,
        cell,
        signs,
        x,
        Some((x, e)),
        Budget::try_from(remaining)?,
        c,
    )
}
fn real_root(
    provider: Provider,
    index: usize,
    steps: Budget,
    c: &mut SolveContext<'_, '_>,
) -> Result<RealRoot, SciOxideError> {
    let f = provider.family;
    if let Some(seed) = super::seeds::bessel(f, index) {
        let spacing = super::seeds::bessel(f, index + 1).map_or(PI, |next| next - seed);
        let lower = match f {
            F::BesselJ(n) | F::BesselY(n) | F::BesselJPrime(n) | F::BesselYPrime(n) => n.max(0.125),
            _ => 0.125,
        };
        let bracket = Interval::new((seed - 0.35 * spacing).max(lower), seed + 0.35 * spacing)?;
        let signs = (
            sample(provider, bracket.lower(), c)?,
            sample(provider, bracket.upper(), c)?,
        );
        if signs.0.is_sign_negative() != signs.1.is_sign_negative() {
            return refine(provider, bracket, signs, seed, None, steps, c);
        }
    }
    if let F::BesselJPrime(n) | F::BesselYPrime(n) = f {
        if matches!(f, F::BesselJPrime(0.0)) {
            return real_root(
                Provider {
                    family: F::BesselJ(1.0),
                    ..provider
                },
                index,
                steps,
                c,
            );
        }
        if matches!(f, F::BesselYPrime(0.0)) {
            return real_root(
                Provider {
                    family: F::BesselY(1.0),
                    ..provider
                },
                index,
                steps,
                c,
            );
        }
        let jf = matches!(f, F::BesselJPrime(_));
        let base = Provider {
            family: if jf { F::BesselJ(n) } else { F::BesselY(n) },
            ..provider
        };
        // A base zero only bounds this derivative zero: a base stop leaves
        // no estimate of it.
        let base_zero = |index, c: &mut SolveContext<'_, '_>| {
            Ok::<_, SciOxideError>(match real_root(base, index, steps, c)? {
                RealRoot::Found(located) => Ok(located.z.re),
                RealRoot::Stopped { termination, .. } => Err(RealRoot::Stopped {
                    termination,
                    estimate: None,
                }),
            })
        };
        let lower = if jf && index == 1 {
            n
        } else {
            match base_zero(if jf { index - 1 } else { index }, c)? {
                Ok(x) => x,
                Err(stopped) => return Ok(stopped),
            }
        };
        let upper = match base_zero(if jf { index } else { index + 1 }, c)? {
            Ok(x) => x,
            Err(stopped) => return Ok(stopped),
        };
        let bracket = Interval::new(lower, upper)?;
        let signs = (sample(provider, lower, c)?, sample(provider, upper, c)?);
        return refine(
            provider,
            bracket,
            signs,
            bracket.midpoint()?,
            None,
            steps,
            c,
        );
    }
    if matches!(f, F::AiryAi | F::AiryAiPrime | F::AiryBi | F::AiryBiPrime) {
        return airy_root(provider, index, steps, c);
    }
    if matches!(f, F::FresnelCPrime | F::FresnelSPrime) {
        return Ok(RealRoot::complete(
            (2.0 * index as f64
                - if matches!(f, F::FresnelCPrime) {
                    1.0
                } else {
                    0.0
                })
            .sqrt(),
            None,
        ));
    }
    // In the separated asymptotic region the phase offset identifies the index.
    // Else enumerate from the origin/turning-point lower bound. The phase step
    // below is smaller than a quarter oscillation (DLMF 10.21, 10.68).
    if let F::Kelvin(k) = f {
        let seed = super::seeds::kelvin(k, index);
        let half = PI * 0.4;
        let interval = Interval::new(seed - half, seed + half)?;
        let a = sample(provider, interval.lower(), c)?;
        let b = sample(provider, interval.upper(), c)?;
        if a.is_sign_negative() != b.is_sign_negative() {
            return refine(provider, interval, (a, b), seed, None, steps, c);
        }
    }
    let mut left = match f {
        F::BesselJ(n) | F::BesselY(n) => n.max(0.125),
        _ => 0.125,
    };
    let mut fl = sample(provider, left, c)?;
    let mut found = 0;
    loop {
        Session::new(c).proposal()?;
        let step = PI * 0.25 / (1.0 + 0.25 / (left * left)).sqrt();
        let right = left + step;
        if !right.is_finite() || right == left {
            return Err(invalid(
                scioxide_core::argument!("family"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "zero bracket cannot advance",
            ));
        }
        let fr = sample(provider, right, c)?;
        if fl.is_sign_negative() != fr.is_sign_negative() || fr == 0.0 {
            found += 1;
            if found == index {
                return refine(
                    provider,
                    Interval::new(left, right)?,
                    (fl, fr),
                    (left + right) * 0.5,
                    None,
                    steps,
                    c,
                );
            }
        }
        left = right;
        fl = fr;
    }
}
struct ComplexRoot {
    z: C,
    termination: T,
    evaluation: Option<super::evaluation::Evaluation>,
}

/// Paired analytic Newton within the indexed start's cell. It stops when the
/// correction reaches coordinate rounding (`2ε max(|z|, 1)`), or at the
/// evaluation-noise floor: a value within its own evaluation error whose
/// correction stops decreasing. The final check includes that uncertainty.
fn complex_root(
    provider: Provider,
    index: usize,
    steps: Budget,
    c: &mut SolveContext<'_, '_>,
    candidate: &mut Option<C>,
) -> Result<ComplexRoot, SciOxideError> {
    let seed = super::seeds::complex(provider.family, index);
    let radius = 0.45 * (super::seeds::complex(provider.family, index + 1) - seed).norm();
    let mut z = seed;
    let mut previous = f64::INFINITY;
    let session = Session::new(c);
    for _ in 0..steps.get() {
        *candidate = Some(z);
        session.iteration_only()?;
        let e = session.invoke_context(EvaluationRole::Function, |c| provider.at(z, c))?;
        if e.derivative == C::new(0.0, 0.0) {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "indexed-zero derivative vanished",
            ));
        }
        let next = newton(z, e.value, e.derivative, None).ok().flatten();
        // One correction modulus serves both tests. At the evaluation-noise
        // floor corrections stop decreasing: accept a value within its own
        // error once they do.
        let correction = next.map_or(f64::INFINITY, |next| NumericScalar::abs(next - z));
        if correction <= 2.0 * f64::EPSILON * NumericScalar::abs(z).max(1.0)
            || (correction >= previous && e.value.norm() <= e.error)
        {
            return Ok(ComplexRoot {
                z,
                termination: T::Complete,
                evaluation: Some(e),
            });
        }
        previous = correction;
        session.proposal()?;
        match next.filter(|next| next.im > 0.0 && (*next - seed).norm() <= radius) {
            Some(next) => z = next,
            None => {
                return Ok(ComplexRoot {
                    z,
                    termination: T::Unverified,
                    evaluation: None,
                });
            }
        }
    }
    *candidate = Some(z);
    Ok(ComplexRoot {
        z,
        termination: T::IterationLimit,
        evaluation: None,
    })
}

fn transform(z: C, branch: B) -> C {
    match branch {
        B::Complex {
            conjugate,
            negate,
            rotate,
        } => {
            let z = if conjugate { z.conj() } else { z };
            let z = if negate { -z } else { z };
            if rotate { C::new(-z.im, z.re) } else { z }
        }
        _ => z,
    }
}
fn narrow<S: RealScalar>(z: C) -> Result<Complex<S>, SciOxideError> {
    Ok(Complex::new(
        S::try_from_f64_lossy(z.re)?,
        S::try_from_f64_lossy(z.im)?,
    ))
}
/// Locate and verify one indexed zero.
///
/// Stops become an unsuccessful attempt with the last estimate of this zero;
/// other errors propagate.
fn one<S: RealScalar>(
    p: &Request<'_, S>,
    offset: usize,
    c: &mut SolveContext<'_, '_>,
) -> Result<Attempt<S>, SciOxideError> {
    let family = widen(p.family)?;
    let provider = Provider {
        family,
        budget: p.options.work_per_zero,
    };
    let index = p.start + offset;
    let minimum = if p.branch == B::Origin || matches!(family, F::FresnelCPrime | F::FresnelSPrime)
    {
        1
    } else {
        2
    };
    if !c.can_consume(BudgetKind::Iteration, minimum)?.is_granted() {
        return Ok(stopped(T::WorkLimit(BudgetKind::Iteration), None));
    }
    let mut candidate = None;
    let result = if p.branch == B::Origin {
        origin(family, index, c)
    } else {
        locate(p, provider, index, c, &mut candidate).and_then(|located| match located {
            Ok(located) => verify(p, provider, index, located, c, &mut candidate),
            Err(attempt) => Ok(attempt),
        })
    };
    match result {
        Err(e) => match stop(&e) {
            Some(t) => Ok(stopped(t, candidate)),
            None => Err(e),
        },
        other => other,
    }
}
/// The zero at the origin, with its known multiplicity and inverse slope.
fn origin<S: RealScalar>(
    family: F<f64>,
    index: usize,
    c: &mut SolveContext<'_, '_>,
) -> Result<Attempt<S>, SciOxideError> {
    Session::new(c).iteration_only()?;
    let zero = S::ZERO;
    Ok(Attempt {
        zero: Some(SpecialZero {
            index,
            root: Complex::new(zero, zero),
            residual: crate::context::residual(zero, S::ONE)?,
            correction: zero,
            estimated_error: zero,
            inverse_slope: match family {
                F::Erf => Some(S::try_from_f64_lossy(PI.sqrt() * 0.5)?),
                F::FresnelC => Some(S::ONE),
                F::BesselJPrime(0.0) | F::Kelvin(KelvinZeroFunction::BeiPrime) => Some(S::TWO),
                _ => None,
            },
            bracket: None,
            multiplicity: origin_multiplicity(family).ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("family"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "invalid origin",
                )
            })?,
            scaled: matches!(family, F::Kelvin(_)),
        }),
        termination: T::Complete,
        candidate: None,
    })
}
/// Locate the indexed zero, or the unsuccessful attempt that ends its search.
///
/// Real zeros use bracketed refinement; complex zeros use paired Newton
/// refinement within the indexed start's cell. `candidate` keeps the last
/// estimate of this zero in the output coordinates.
fn locate<S: RealScalar>(
    p: &Request<'_, S>,
    provider: Provider,
    index: usize,
    c: &mut SolveContext<'_, '_>,
    candidate: &mut Option<Complex<S>>,
) -> Result<Result<Located, Attempt<S>>, SciOxideError> {
    let family = provider.family;
    if p.branch == B::Real {
        return Ok(
            match real_root(provider, index, p.options.refinement_steps, c)? {
                RealRoot::Found(located) => Ok(located),
                RealRoot::Stopped {
                    termination,
                    estimate,
                } => Err(stopped(
                    termination,
                    estimate.map(|x| narrow(C::new(x, 0.0))).transpose()?,
                )),
            },
        );
    }
    if matches!(family, F::FresnelCPrime | F::FresnelSPrime) {
        return Ok(Ok(Located {
            z: super::seeds::complex(family, index),
            bracket: None,
            evaluation: None,
        }));
    }
    let mut current = None;
    let result = complex_root(provider, index, p.options.refinement_steps, c, &mut current);
    *candidate = current
        .map(|z| narrow(transform(z, p.branch)))
        .transpose()?;
    let refined = result?;
    if refined.termination != T::Complete {
        return Ok(Err(stopped(refined.termination, *candidate)));
    }
    Ok(Ok(Located {
        z: refined.z,
        bracket: None,
        evaluation: refined.evaluation.map(|e| (refined.z, e)),
    }))
}
/// Checks a located zero at its returned coordinate.
///
/// The rounded coordinate must stay in the indexed zero's cell, and the
/// original equation at that exact coordinate (reusing an evaluation made
/// there) must bound the coordinate error by the configured tolerance.
fn verify<S: RealScalar>(
    p: &Request<'_, S>,
    provider: Provider,
    index: usize,
    located: Located,
    c: &mut SolveContext<'_, '_>,
    candidate: &mut Option<Complex<S>>,
) -> Result<Attempt<S>, SciOxideError> {
    let family = provider.family;
    let Located {
        z,
        bracket,
        evaluation,
    } = located;
    let root: Complex<S> = narrow(transform(z, p.branch))?;
    *candidate = Some(root);
    let final_z = C::new(root.re.try_to_f64_exact()?, root.im.try_to_f64_exact()?);
    let rounding = (final_z - transform(z, p.branch)).norm();
    let spacing = if p.branch != B::Real || matches!(family, F::FresnelCPrime | F::FresnelSPrime) {
        (super::seeds::complex(family, index + 1) - super::seeds::complex(family, index)).norm()
    } else if matches!(
        family,
        F::AiryAi | F::AiryAiPrime | F::AiryBi | F::AiryBiPrime
    ) {
        (airy_seed(family, index + 1) - airy_seed(family, index)).abs()
    } else {
        PI
    };
    // A small local residual at a rounded coordinate does not establish that
    // it still identifies this indexed root rather than a neighbouring one.
    if spacing <= 0.0 || rounding > 0.45 * spacing {
        return Ok(stopped(T::Unverified, *candidate));
    }
    // Projection or transformation to another coordinate requires a new
    // original-equation check.
    let e = match evaluation {
        Some((at, e)) if final_z == at => e,
        _ => provider.evaluate(final_z, c)?,
    };
    let slope = e.derivative.norm();
    let correction = e.value.norm() / slope;
    let error = (e.value.norm() + e.error) / slope + rounding;
    let threshold = p.options.tolerance.absolute().try_to_f64_exact()?
        + p.options.tolerance.relative().try_to_f64_exact()? * final_z.norm();
    if !error.is_finite() || error > threshold {
        return Ok(stopped(T::Unverified, *candidate));
    }
    let project = S::try_from_f64_lossy;
    let interval = bracket
        .map(|b| {
            let mut lower = S::try_from_f64_lossy(b.lower())?;
            let mut upper = S::try_from_f64_lossy(b.upper())?;
            if lower.try_to_f64_exact()? > b.lower() {
                lower = lower.next_after(S::NEG_INFINITY);
            }
            if upper.try_to_f64_exact()? < b.upper() {
                upper = upper.next_after(S::INFINITY);
            }
            Interval::new(lower, upper)
        })
        .transpose()?;
    Ok(Attempt {
        zero: Some(SpecialZero {
            index,
            root,
            residual: crate::context::residual(project(e.value.norm())?, S::ONE)?,
            correction: project(correction)?,
            estimated_error: project(error)?,
            inverse_slope: if (1.0 / slope).is_finite()
                && 1.0 / slope <= S::MAX.try_to_f64_exact()?
            {
                Some(project(1.0 / slope)?)
            } else {
                None
            },
            bracket: interval,
            multiplicity: 1,
            scaled: matches!(family, F::Kelvin(_)),
        }),
        termination: T::Complete,
        candidate: None,
    })
}
fn identity<S: RealScalar>(p: &Request<'_, S>) -> scioxide_core::SolveIdentity {
    crate::context::identity(
        scioxide_core::algorithm_id!("rootfind", "special-zeros", 1),
        ProblemToken::new(0),
        p.count,
        crate::context::options_token(p),
    )
}
/// The owned form: validates `p` before allocating the output and workspace,
/// then runs it sequentially.
pub(super) fn owned<S: RealScalar>(
    p: &Request<'_, S>,
    arguments: [Argument; 4],
) -> Result<SpecialZeros<S>, SciOxideError> {
    validate(p, arguments)?;
    let mut output =
        scioxide_core::backend::try_filled(p.count, None, scioxide_core::quantity!("output"))?;
    let mut workspace = SpecialZeroWorkspace::workspace(p.count, p.options)?;
    let report = run(
        p,
        arguments,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native().into(),
    )?;
    let mut roots = try_with_capacity(report.completed, scioxide_core::quantity!("roots"))?;
    roots.extend(output.into_iter().flatten());
    Ok(SpecialZeros { roots, report })
}
/// Runs `p` in `scope`, writing zero `start + i` to `output[i]`.
///
/// `p.count` is `output.len()`. The preflight validates `p`, naming
/// `arguments` in its errors, and checks the workspace.
pub(super) fn run<S: RealScalar>(
    p: &Request<'_, S>,
    arguments: [Argument; 4],
    output: &mut [Option<SpecialZero<S>>],
    w: &mut SpecialZeroWorkspace<S>,
    scope: scioxide_core::SolveScope<'_, '_>,
) -> Result<SpecialZeroReport<S>, SciOxideError> {
    crate::attempt::attempt(
        w,
        output,
        scope,
        scioxide_core::AttemptKind::Call,
        |w, output| {
            validate(p, arguments)?;
            let requirements =
                SpecialZeroWorkspace::workspace_requirements(output.len(), p.options)?;
            if w.check(&requirements)? != scioxide_core::Capacity::Sufficient {
                return Err(invalid(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "reserve workspace before solving",
                ));
            }
            Ok(crate::attempt::Entry::call(
                identity(p),
                crate::context::cap(
                    p.options.budgets,
                    BudgetKind::Evaluation,
                    p.options.evaluation_budget,
                ),
            ))
        },
        |w, output, context| run_inner(p, output, w, context),
        |_, _, result| crate::attempt::Conclusion {
            succeeded: result
                .as_ref()
                .is_ok_and(|report| report.termination.is_success()),
            progress: result.as_ref().ok().filter(|r| r.completed > 0).cloned(),
        },
    )
}
fn run_inner<S: RealScalar>(
    p: &Request<'_, S>,
    output: &mut [Option<SpecialZero<S>>],
    w: &mut SpecialZeroWorkspace<S>,
    c: &mut SolveContext<'_, '_>,
) -> Result<SpecialZeroReport<S>, SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::backend::{PartitionSlot, with_partitioned_solves};
    let mut report = SpecialZeroReport {
        algorithm: c.identity().algorithm(),
        termination: T::Complete,
        completed: 0,
        requested: p.count,
        counts: scioxide_core::SolveCounts::new(),
        candidate: None,
    };
    if p.count == 0 {
        return Ok(report);
    }
    if let Some(reason) = c.stop_reason()? {
        report.termination = if reason == scioxide_core::ExecutionStopReason::DeadlineReached {
            T::TimeLimit
        } else {
            T::Cancelled
        };
        return Ok(report);
    }
    if no_zeros(widen(p.family)?, p.branch) {
        report.termination = T::NoZeros;
        return Ok(report);
    }
    for outcome in &mut w.outcomes[..p.count] {
        *outcome = None;
    }
    let id = identity(p);
    with_partitioned_solves(
        c,
        &mut w.children[..p.count],
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, p.options.work_per_zero),
        |execution, children| {
            for (chunk, (children, outcomes)) in children
                .into_chunks(64)?
                .zip(w.outcomes[..p.count].chunks_mut(64))
                .enumerate()
            {
                let plan = execution.plan(
                    p.options.policy,
                    children.len(),
                    scioxide_core::PlanLimits {
                        work_per_item: core::num::NonZeroUsize::MIN,
                        auto_threshold: NonZeroUsize::MIN,
                        worker_limit: NonZeroUsize::new(64).ok_or_else(|| {
                            invalid(
                                scioxide_core::argument!("context"),
                                scioxide_core::InvalidInputReason::OutOfRange,
                                "partition count",
                            )
                        })?,
                    },
                );
                let mut slots: [_; 64] = core::array::from_fn(|_| None);
                for (index, (child, outcome)) in children.into_iter().zip(outcomes).enumerate() {
                    slots[index] = Some((child, outcome));
                }
                let mut remaining = &mut slots[..plan.work_items()];
                let mut partitions = plan.partitions();
                let partition_slots: [_; 64] = core::array::from_fn(|_| {
                    partitions.next().map_or_else(PartitionSlot::empty, |part| {
                        let (entries, tail) =
                            core::mem::take(&mut remaining).split_at_mut(part.range().len());
                        remaining = tail;
                        PartitionSlot::new(entries)
                    })
                });
                execution.execute_partitioned(
                    plan,
                    &partition_slots[..plan.worker_count()],
                    &|partition, entries, scope| {
                        for (index, entry) in partition
                            .range()
                            .zip(entries.iter_mut().filter_map(Option::take))
                        {
                            let (child, outcome) = entry;
                            *outcome = Some(
                                child.run(scope, id, |child| one(p, chunk * 64 + index, child)),
                            );
                        }
                        Ok(())
                    },
                )?;
            }
            Ok(())
        },
    )?;
    // Check every outcome before committing anything, including provider errors after a gap.
    for outcome in &mut w.outcomes[..p.count] {
        if matches!(outcome, Some(Err(_)))
            && let Some(Err(e)) = outcome.take()
        {
            return Err(e);
        }
        if outcome.is_none() {
            return Err(invalid(
                scioxide_core::argument!("context"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "missing zero outcome",
            ));
        }
    }
    let mut previous = None;
    for outcome in &w.outcomes[..p.count] {
        let Some(Ok(a)) = outcome else {
            return Err(invalid(
                scioxide_core::argument!("context"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "missing zero outcome",
            ));
        };
        let Some(zero) = a.zero else {
            report.termination = a.termination;
            report.candidate = a.candidate;
            break;
        };
        let norm = zero.root.re.hypot(zero.root.im);
        if previous.is_some_and(|v| norm <= v) {
            report.termination = T::Unverified;
            report.candidate = Some(zero.root);
            break;
        }
        previous = Some(norm);
        report.completed += 1;
    }
    if report.termination == T::Complete
        && let Some(reason) = c.stop_reason()?
    {
        report.termination = if reason == scioxide_core::ExecutionStopReason::DeadlineReached {
            T::TimeLimit
        } else {
            T::Cancelled
        };
    }
    for (out, outcome) in output.iter_mut().zip(&w.outcomes).take(report.completed) {
        if let Some(Ok(a)) = outcome {
            *out = a.zero;
        }
    }
    report.counts = c.counts();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{ConvergenceDiagnostics, ConvergenceReason, ExecutionStopReason};

    #[test]
    fn stop_classification_preserves_unrelated_failures() {
        for (reason, expected) in [
            (
                ConvergenceReason::IterationLimit,
                Some(T::WorkLimit(BudgetKind::Iteration)),
            ),
            (
                ConvergenceReason::EvaluationLimit,
                Some(T::WorkLimit(scioxide_core::BudgetKind::Evaluation)),
            ),
            (ConvergenceReason::Stagnation, None),
            (ConvergenceReason::NestedSolver, None),
        ] {
            let error = SciOxideError::convergence_failure(
                reason,
                ConvergenceDiagnostics::new(),
                "test failure",
            );
            assert_eq!(stop(&error), expected);
        }
        for (reason, expected) in [
            (ExecutionStopReason::Cancelled, T::Cancelled),
            (ExecutionStopReason::DeadlineReached, T::TimeLimit),
        ] {
            assert_eq!(
                stop(&SciOxideError::execution_stopped(reason)),
                Some(expected)
            );
        }
        assert_eq!(
            stop(&invalid(
                scioxide_core::argument!("start"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "test failure"
            )),
            None
        );
    }
}
