//! Brent, Bus–Dekker, Chandrupatla and Alefeld–Potra–Shi bracketing methods.
use super::{BracketRule, BracketState, Sample};
use crate::ScalarRootOptions;
use scioxide_core::{RealScalar, SciOxideError};

/// Finds a zero of `f` in `[a, b]` by Brent's method.
///
/// Each iteration takes one step of inverse quadratic interpolation through
/// the last three points, or of the secant method when only two are distinct,
/// and evaluates `f` once. The step is accepted only if it is less than half
/// the step before last and lies inside the bracket; otherwise the iteration
/// bisects. Steps are at least `½ ε |x|`, so the iteration keeps moving near
/// convergence. Convergence to a simple zero is superlinear, and the bisection
/// fallback guarantees convergence for every continuous `f`. It is a good
/// default when a bracket is known.
///
/// The solve starts from an [`Interval`](scioxide_core::Interval) `[a, b]` whose
/// endpoint values have opposite signs or include a zero, keeps a bracket
/// around the zero, and stops as described in [`ScalarRootOptions`](crate::ScalarRootOptions).
/// The result is the bracket endpoint with the smaller `|f|`, and
/// [`RootResult::bracket`](crate::RootResult::bracket) returns the final bracket.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign.
/// - The errors listed on [`ScalarRootSolver::solve`](crate::ScalarRootSolver::solve).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_rootfind::{Brent, ScalarFn, ScalarRootOptions, ScalarRootSolver};
///
/// // cos x = x has the unique solution x ≈ 0.7390851332151607 (Dottie number).
/// let f = ScalarFn::new(|x: f64| Ok(x.cos() - x));
/// let result = Brent.solve(&f, Interval::new(0.0, 1.0)?, &ScalarRootOptions::default())?;
/// assert!(result.is_success());
/// assert!((result.point() - 0.739_085_133_215_160_7).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. P. Brent, *Algorithms for Minimization without Derivatives*,
/// Prentice-Hall, 1973, chapter 4.
#[derive(Debug, Clone, Copy, Default)]
pub struct Brent;
/// Finds a zero of `f` in `[a, b]` by Brent's method with hyperbolic
/// interpolation (Bus and Dekker's Algorithm M).
///
/// The iteration is that of [`Brent`], except that the three-point step uses
/// hyperbolic (rational) interpolation instead of inverse quadratic
/// interpolation.
///
/// The solve starts from an [`Interval`](scioxide_core::Interval) `[a, b]` whose
/// endpoint values have opposite signs or include a zero, keeps a bracket
/// around the zero, and stops as described in [`ScalarRootOptions`](crate::ScalarRootOptions).
/// The result is the bracket endpoint with the smaller `|f|`, and
/// [`RootResult::bracket`](crate::RootResult::bracket) returns the final bracket.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign.
/// - The errors listed on [`ScalarRootSolver::solve`](crate::ScalarRootSolver::solve).
///
/// # References
///
/// J. C. P. Bus and T. J. Dekker, "Two efficient algorithms with guaranteed
/// convergence for finding a zero of a function", *ACM Trans. Math. Softw.*
/// 1(4), 330–345 (1975).
#[derive(Debug, Clone, Copy, Default)]
pub struct Brenth;
/// Finds a zero of `f` in `[a, b]` by Chandrupatla's method.
///
/// Each iteration takes an inverse quadratic interpolation step through the
/// two bracket endpoints and the previous point when the values satisfy
/// Chandrupatla's test
///
/// ```text
/// 1 − √(1 − ξ) < Φ < √ξ,
/// ```
///
/// which indicates that the interpolant is monotone on the bracket. Here `ξ`
/// and `Φ` are the ratios of the coordinate and value differences of the three
/// points. Otherwise the iteration bisects. One evaluation per iteration.
///
/// The solve starts from an [`Interval`](scioxide_core::Interval) `[a, b]` whose
/// endpoint values have opposite signs or include a zero, keeps a bracket
/// around the zero, and stops as described in [`ScalarRootOptions`](crate::ScalarRootOptions).
/// The result is the bracket endpoint with the smaller `|f|`, and
/// [`RootResult::bracket`](crate::RootResult::bracket) returns the final bracket.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign.
/// - The errors listed on [`ScalarRootSolver::solve`](crate::ScalarRootSolver::solve).
///
/// # References
///
/// T. R. Chandrupatla, "A new hybrid quadratic/bisection algorithm for finding
/// the zero of a nonlinear function without using derivatives", *Adv. Eng.
/// Softw.* 28(3), 145–149 (1997).
#[derive(Debug, Clone, Copy, Default)]
pub struct Chandrupatla;
/// Finds a zero of `f` in `[a, b]` by Algorithm 748 of Alefeld, Potra and Shi.
///
/// The first iteration takes a secant step. Each later iteration evaluates `f`
/// up to three times: an inverse cubic interpolation step through four points
/// (falling back to a quadratic Newton step or a secant step), a doubled
/// secant step from the endpoint with the smaller `|f|`, and a bisection if the
/// iteration has not halved the bracket. This is the variant with `k = 1` in
/// the notation of the paper. Like [`Brent`], it suits a continuous `f` with a
/// known bracket.
///
/// The solve starts from an [`Interval`](scioxide_core::Interval) `[a, b]` whose
/// endpoint values have opposite signs or include a zero, keeps a bracket
/// around the zero, and stops as described in [`ScalarRootOptions`](crate::ScalarRootOptions).
/// The result is the bracket endpoint with the smaller `|f|`, and
/// [`RootResult::bracket`](crate::RootResult::bracket) returns the final bracket.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign.
/// - The errors listed on [`ScalarRootSolver::solve`](crate::ScalarRootSolver::solve).
///
/// # References
///
/// G. E. Alefeld, F. A. Potra and Y. Shi, "Algorithm 748: Enclosing zeros of
/// continuous functions", *ACM Trans. Math. Softw.* 21(3), 327–344 (1995).
#[derive(Debug, Clone, Copy, Default)]
pub struct Toms748;

/// Rule of [`Brent`], started from the evaluated initial bracket.
#[inline(always)]
pub(super) fn brent<S: RealScalar>(bracket: BracketState<S>) -> impl BracketRule<S> {
    BrentRule::<_, false>(BrentState::new(bracket))
}

bracket_solver!(Brent, brent);
bracket_solver!(Brenth, |b| BrentRule::<_, true>(BrentState::new(b)));
bracket_solver!(Chandrupatla, |_| Interpolation::<_, false>::new());
bracket_solver!(Toms748, |_| Interpolation::<_, true>::new());

/// Brent's methods need only their signed-step recurrence, not the discarded
/// endpoints that Algorithm 748 and Chandrupatla's method use.
struct BrentRule<S: RealScalar, const RATIONAL: bool>(BrentState<S>);

impl<S: RealScalar, const RATIONAL: bool> BracketRule<S> for BrentRule<S, RATIONAL> {
    #[inline(always)]
    fn propose(
        &mut self,
        b: BracketState<S>,
        _: BracketState<S>,
        _: usize,
        step: usize,
        _: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError> {
        if step == 0 {
            self.0.propose(b, RATIONAL).map(Some)
        } else {
            Ok(None)
        }
    }

    fn observe(
        &mut self,
        _: BracketState<S>,
        _: BracketState<S>,
        sample: Sample<S>,
    ) -> Result<(), SciOxideError> {
        self.0.current = sample;
        Ok(())
    }
}

/// Interpolation history of one solve: the two most recently discarded
/// endpoints and the latest evaluation. The method is a const parameter, so
/// the loop has no run-time dispatch.
struct Interpolation<S: RealScalar, const TOMS: bool> {
    discarded: [Option<Sample<S>>; 2],
    latest: Option<Sample<S>>,
}

impl<S: RealScalar, const TOMS: bool> Interpolation<S, TOMS> {
    fn new() -> Self {
        Self {
            discarded: [None; 2],
            latest: None,
        }
    }
}

impl<S: RealScalar, const TOMS: bool> BracketRule<S> for Interpolation<S, TOMS> {
    fn propose(
        &mut self,
        b: BracketState<S>,
        start: BracketState<S>,
        cycle: usize,
        step: usize,
        options: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError> {
        // TOMS 748 takes an interpolation, a doubled secant and, unless the
        // bracket has already halved, a bisection in every cycle after the first.
        let phases = if TOMS && cycle > 0 { 3 } else { 1 };
        if step >= phases || (step == 2 && b.half_width_contracted(start)?) {
            return Ok(None);
        }
        let proposed = match (TOMS, step) {
            (true, 2) => None,
            (true, 1) => doubled_secant(b),
            (true, _) => toms_candidate(b, self.discarded),
            (false, _) => chandrupatla_candidate(b, self.latest, self.discarded[0]),
        };
        let x = match (TOMS, cycle) {
            (false, _) => push_inside(b, proposed, options, S::HALF)?,
            (true, 0) => {
                let x = push_inside(b, proposed, options, S::TWO)?;
                if proposed == Some(x) {
                    x
                } else {
                    b.midpoint()?
                }
            }
            (true, _) => push_inside(b, proposed, options, S::TWO)?,
        };
        Ok(Some(x))
    }

    fn observe(
        &mut self,
        old: BracketState<S>,
        new: BracketState<S>,
        sample: Sample<S>,
    ) -> Result<(), SciOxideError> {
        let discarded = if new.lower.x == old.lower.x {
            old.upper
        } else {
            old.lower
        };
        self.discarded = [Some(discarded), self.discarded[0]];
        self.latest = Some(sample);
        Ok(())
    }
}

/// A failed interpolation is a rejected proposal, not a solver failure.
fn finite<S: RealScalar>(x: S) -> Option<S> {
    x.is_finite().then_some(x)
}

// Normalize both axes before Lagrange interpolation. Singular differences,
// non-finite intermediates and proposals outside the bracket are rejected.
fn interpolate<S: RealScalar>(points: &[Sample<S>]) -> Option<S> {
    if let [a, b] = points {
        return if a.value.abs() < b.value.abs() {
            super::correction::secant(*b, *a)
        } else {
            super::correction::secant(*a, *b)
        }
        .ok()
        .flatten();
    }
    let f_scale = points
        .iter()
        .fold(S::ZERO, |scale, p| scale.max(p.value.abs()));
    let x_scale = points.iter().fold(S::ZERO, |scale, p| scale.max(p.x.abs()));
    if f_scale == S::ZERO || x_scale == S::ZERO {
        return None;
    }
    let mut sum = S::ZERO;
    for (i, p) in points.iter().enumerate() {
        let mut weight = S::ONE;
        let f = p.value / f_scale;
        for (j, q) in points.iter().enumerate() {
            if i != j {
                let g = q.value / f_scale;
                let denominator = f - g;
                if denominator == S::ZERO {
                    return None;
                }
                weight *= -g / denominator;
            }
        }
        sum += weight * (p.x / x_scale);
    }
    finite(sum * x_scale)
}

/// Bus–Dekker hyperbolic correction in normalized function units.
fn hyperbolic_step<S: RealScalar>(u: Sample<S>, v: Sample<S>, old: Sample<S>) -> Option<S> {
    let scale = u.value.abs().max(v.value.abs()).max(old.value.abs());
    let (f, g, h) = (u.value / scale, v.value / scale, old.value / scale);
    let a = (h - f) / (old.x - u.x);
    let c = (g - f) / (v.x - u.x);
    let denominator = g * a - h * c;
    if denominator == S::ZERO || !denominator.is_finite() || !a.is_finite() || !c.is_finite() {
        return None;
    }
    finite(-f * ((g - h) / denominator))
}

/// Algorithm 748 doubled secant, with geometric contraction for extreme imbalance.
fn doubled_secant<S: RealScalar>(b: BracketState<S>) -> Option<S> {
    let u = b.best();
    let other = if u.x == b.lower.x { b.upper } else { b.lower };
    let scale = u.value.abs().max(other.value.abs());
    let fu = u.value / scale;
    let denominator = b.upper.value / scale - b.lower.value / scale;
    let half_width = b.midpoint().ok()? - b.lower.x;
    let step = -fu / denominator * half_width * (S::TWO + S::TWO);
    if !step.is_finite() || step.abs() > half_width {
        return None;
    }
    let x = finite(u.x + step)?;
    if x == u.x && fu.abs() < other.value.abs() / scale / S::TWO.powi(50) {
        let direction = if u.x == b.lower.x { S::ONE } else { -S::ONE };
        return finite(u.x + direction * (half_width / (S::TWO + S::TWO).powi(2)));
    }
    Some(x)
}

/// Algorithm 748 inverse cubic, quadratic Newton, then secant proposal.
fn toms_candidate<S: RealScalar>(b: BracketState<S>, old: [Option<Sample<S>>; 2]) -> Option<S> {
    if let [Some(d), Some(e)] = old {
        let points = [b.lower, b.upper, d, e];
        let scale = super::residual_scale(&points.map(|p| p.value));
        let separated = points.iter().enumerate().all(|(i, p)| {
            points[i + 1..].iter().all(|q| {
                (p.value / scale - q.value / scale).abs()
                    > (S::TWO + S::TWO).powi(2) * S::TWO * S::EPSILON
            })
        });
        if separated
            && let Some(x) = interpolate(&points)
            && b.contains_strict(x)
        {
            return Some(x);
        }
    }
    if let Some(d) = old[0]
        && let Some(x) = newton_polynomial(b, d)
        && b.contains_strict(x)
    {
        return Some(x);
    }
    interpolate(&[b.lower, b.upper])
}

/// Algorithm 748 quadratic Newton proposal in normalized coordinates and values.
fn newton_polynomial<S: RealScalar>(b: BracketState<S>, d: Sample<S>) -> Option<S> {
    let x_scale = b.lower.x.abs().max(b.upper.x.abs()).max(d.x.abs());
    let f_scale = b
        .lower
        .value
        .abs()
        .max(b.upper.value.abs())
        .max(d.value.abs());
    let (a, z, y) = (b.lower.x / x_scale, b.upper.x / x_scale, d.x / x_scale);
    let (fa, fz, fy) = (
        b.lower.value / f_scale,
        b.upper.value / f_scale,
        d.value / f_scale,
    );
    let first = (fz - fa) / (z - a);
    let next = (fy - fz) / (y - z);
    let second = (next - first) / (y - a);
    if !first.is_finite() || !second.is_finite() {
        return None;
    }
    let mut x = if second.is_sign_negative() == fa.is_sign_negative() {
        a
    } else {
        z
    };
    for _ in 0..2 {
        let value = fa + (x - a) * (first + second * (x - z));
        let derivative = first + second * ((x - a) + (x - z));
        let next = x - value / derivative;
        if !b.contains_strict(next * x_scale) {
            break;
        }
        x = next;
    }
    finite(x * x_scale)
}

/// Chandrupatla's acceptance test and inverse quadratic proposal.
fn chandrupatla_candidate<S: RealScalar>(
    b: BracketState<S>,
    latest: Option<Sample<S>>,
    old: Option<Sample<S>>,
) -> Option<S> {
    let (p, r) = (latest?, old?);
    let q = if p.x == b.lower.x { b.upper } else { b.lower };
    let x_scale = p.x.abs().max(q.x.abs()).max(r.x.abs());
    let f_scale = p.value.abs().max(q.value.abs()).max(r.value.abs());
    let xi = (p.x / x_scale - q.x / x_scale) / (r.x / x_scale - q.x / x_scale);
    let phi = (p.value / f_scale - q.value / f_scale) / (r.value / f_scale - q.value / f_scale);
    if xi > S::ZERO && xi < S::ONE && phi > S::ONE - (S::ONE - xi).sqrt() && phi < xi.sqrt() {
        interpolate(&[p, q, r])
    } else {
        None
    }
}

/// Moves a proposal at least a tolerance-sized margin inside the bracket, and
/// bisects when the margins overlap.
fn push_inside<S: RealScalar>(
    b: BracketState<S>,
    proposed: Option<S>,
    options: &ScalarRootOptions<S>,
    factor: S,
) -> Result<S, SciOxideError> {
    let tolerance = options.step_tolerance.tolerance();
    let margin = (tolerance.absolute()
        + tolerance.relative() * b.lower.x.abs().max(b.upper.x.abs()))
        * factor;
    let a = b.lower.x + margin;
    let z = b.upper.x - margin;
    if !margin.is_finite() || a >= z {
        return b.midpoint();
    }
    let x = proposed.map_or_else(|| b.midpoint(), Ok)?.max(a).min(z);
    if b.contains_strict(x) {
        Ok(x)
    } else {
        b.midpoint()
    }
}

/// Brent's signed step history; unrepresentable interpolation is bisected.
struct BrentState<S: RealScalar> {
    previous: Sample<S>,
    current: Sample<S>,
    opposite: Sample<S>,
    previous_step: S,
    last_step: S,
}
impl<S: RealScalar> BrentState<S> {
    fn new(b: BracketState<S>) -> Self {
        let step = (b.upper.x - b.lower.x).min(S::MAX);
        Self {
            previous: b.lower,
            current: b.upper,
            opposite: b.lower,
            previous_step: step,
            last_step: step,
        }
    }
    #[inline(always)]
    fn propose(&mut self, b: BracketState<S>, rational: bool) -> Result<S, SciOxideError> {
        if scioxide_core::backend::opposite_signs(self.previous.value, self.current.value) {
            self.opposite = self.previous;
            let difference = self.current.x - self.previous.x;
            self.last_step = if difference.is_finite() {
                difference
            } else {
                S::MAX
            };
            self.previous_step = self.last_step;
        }
        if self.opposite.value.abs() < self.current.value.abs() {
            let old = self.current;
            self.previous = old;
            self.current = self.opposite;
            self.opposite = old;
        }
        // The rounding guard is distinct from the requested bracket width.
        // A caller tolerance must not replace an accurate, tiny interpolant by
        // a much larger displacement while the sign bracket is still wide.
        let delta = (S::HALF * S::EPSILON) * self.current.x.abs();
        let difference = self.opposite.x - self.current.x;
        let bisect = if difference.is_finite() {
            difference * S::HALF
        } else {
            self.opposite.x * S::HALF - self.current.x * S::HALF
        };
        let interpolated_step = if self.previous_step.abs() > delta
            && self.current.value.abs() < self.previous.value.abs()
        {
            if self.previous.x == self.opposite.x {
                // The current endpoint has the smaller residual and the
                // endpoints have opposite signs: this ratio cannot overflow.
                let ratio = self.current.value / self.opposite.value;
                if !difference.is_finite() {
                    None
                } else {
                    finite(-(difference * ratio) / (S::ONE - ratio))
                }
            } else if rational {
                hyperbolic_step(self.current, self.opposite, self.previous)
            } else {
                // Brent's ratio form avoids the repeated divisions and large
                // cancelling terms of general Lagrange interpolation.
                let s = self.current.value / self.previous.value;
                let q = self.previous.value / self.opposite.value;
                let r = self.current.value / self.opposite.value;
                let p = s
                    * ((S::TWO * bisect) * q * (q - r)
                        - (self.current.x - self.previous.x) * (r - S::ONE));
                let denominator = (q - S::ONE) * (r - S::ONE) * (s - S::ONE);
                if denominator == S::ZERO || !denominator.is_finite() {
                    None
                } else {
                    finite(-p / denominator)
                }
            }
        } else {
            None
        };
        let bound = (S::TWO + S::ONE) * bisect.abs() - delta;
        if let Some(step) = interpolated_step
            && step.abs() < self.previous_step.abs() * S::HALF
            && step.abs() < bound * S::HALF
        {
            self.previous_step = self.last_step;
            self.last_step = step;
        } else {
            self.previous_step = bisect;
            self.last_step = bisect;
        }
        let step = if self.last_step.abs() <= delta {
            delta.copysign(bisect)
        } else {
            self.last_step
        };
        // Retain the signed correction in the recurrence and round the
        // coordinate once, after interpolation and the minimum-step safeguard.
        let mut x = self.current.x + step;
        if x == self.current.x {
            x = self.current.x.next_after(self.opposite.x);
        }
        if !b.contains_strict(x) {
            x = b.midpoint()?;
        }
        self.previous = self.current;
        Ok(x)
    }
}
