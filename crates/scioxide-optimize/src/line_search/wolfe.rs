//! Safeguarded bracket-and-zoom line search for the strong Wolfe conditions.

use alloc::borrow::ToOwned;
#[cfg(test)]
use alloc::vec::Vec;

use crate::DerivativeWorkspace;
use scioxide_core::checked_midpoint;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;

use crate::traits::{LineSearch, ObjectiveFunction};
use crate::utils;
use crate::{LineSearchFailure, LineSearchResult};

#[derive(Clone, Copy)]
struct LinePoint<S: RealScalar> {
    alpha: S,
    value: S,
    derivative: S,
}

// Power-of-two normalization commutes with ordinary rounding while every
// intermediate remains normal. Match the checked interpolation's operation
// order and retry whenever native arithmetic loses range.
#[inline]
fn ordinary_interpolation<S: RealScalar>(
    low: LinePoint<S>,
    high: LinePoint<S>,
    lower: S,
    upper: S,
) -> Option<Option<S>> {
    let normal =
        |value: S| value.is_finite() && (value == S::ZERO || value.abs() >= S::MIN_POSITIVE);
    let delta = high.alpha - low.alpha;
    let slope = low.derivative * delta;
    let change = high.value - low.value;
    let curvature = change - slope;
    if !normal(delta)
        || !normal(change)
        || !normal(curvature)
        || scioxide_core::backend::standard::product_needs_retry(low.derivative, delta, slope)
    {
        return None;
    }
    if curvature == S::ZERO {
        return Some(None);
    }
    let denominator = S::TWO * curvature;
    let numerator = slope * delta;
    if scioxide_core::backend::standard::product_needs_retry(S::TWO, curvature, denominator)
        || scioxide_core::backend::standard::product_needs_retry(slope, delta, numerator)
    {
        return None;
    }
    let step = numerator / denominator;
    let candidate = low.alpha - step;
    if !normal(step) || !normal(candidate) || (numerator != S::ZERO && step == S::ZERO) {
        return None;
    }
    Some((candidate > lower && candidate < upper).then_some(candidate))
}

struct EvaluatedLinePoint<S: RealScalar> {
    point: LinePoint<S>,
}

/// Far end of a zoom bracket: an evaluated trial, or the nearest rejected step.
#[derive(Clone, Copy)]
enum Upper<S: RealScalar> {
    Point(LinePoint<S>),
    Rejected(S),
}
impl<S: RealScalar> Upper<S> {
    const fn alpha(self) -> S {
        match self {
            Self::Point(point) => point.alpha,
            Self::Rejected(alpha) => alpha,
        }
    }
}

struct SearchProgress<'a, S: RealScalar> {
    scope: crate::LineSearchScope<'a>,
    initial: scioxide_core::EvaluationCounts,
    best_trial: Option<(S, S)>,
    rejected: usize,
}
impl<'a, S: RealScalar> SearchProgress<'a, S> {
    fn new(scope: crate::LineSearchScope<'a>) -> Result<Self, SciOxideError> {
        Ok(Self {
            scope,
            initial: scope.counts()?,
            best_trial: None,
            rejected: 0,
        })
    }
    fn record_rejected(&mut self, evaluation: &EvaluatedLinePoint<S>) -> Result<(), SciOxideError> {
        if self
            .best_trial
            .is_none_or(|(_, best)| evaluation.point.value < best)
        {
            self.best_trial = Some((evaluation.point.alpha, evaluation.point.value));
        }
        Ok(())
    }
    fn accepted(
        &self,
        evaluation: &EvaluatedLinePoint<S>,
    ) -> Result<LineSearchResult<S>, SciOxideError> {
        let (fun, grad, rejected) = self.scope.statistics_since(self.initial)?;
        Ok(LineSearchResult::accepted(
            evaluation.point.alpha,
            evaluation.point.value,
            true,
            Some(evaluation.point.derivative),
            fun,
            grad,
        )
        .with_rejected_trials(rejected))
    }
    fn unsuccessful(
        &self,
        reason: LineSearchFailure,
    ) -> Result<LineSearchResult<S>, SciOxideError> {
        let (fun, grad, rejected) = self.scope.statistics_since(self.initial)?;
        Ok(
            LineSearchResult::unsuccessful(reason, self.best_trial, fun, grad)
                .with_rejected_trials(rejected),
        )
    }
}

#[derive(Clone, Copy)]
struct WolfeConditions<S: RealScalar> {
    origin: LinePoint<S>,
    c1: S,
    c2: S,
}

struct SearchContext<'a, S, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    objective: &'a F,
    initial: &'a [S],
    direction: &'a [S],
    conditions: WolfeConditions<S>,
    search: &'a StrongWolfeLineSearch<S>,
    derivative_strategy: DerivativeStrategy,
    scope: crate::LineSearchScope<'a>,
}

impl<S: RealScalar + NumericScalar<Real = S>> WolfeConditions<S> {
    fn armijo_satisfied(&self, point: LinePoint<S>) -> Result<bool, SciOxideError> {
        crate::utils::armijo_satisfied(
            point.value,
            self.origin.value,
            self.c1,
            point.alpha,
            self.origin.derivative,
            S::try_from_usize_exact(8)?,
        )
    }

    fn curvature_satisfied(&self, point: LinePoint<S>) -> Result<bool, SciOxideError> {
        let derivative = NumericScalar::abs(point.derivative);
        let initial = NumericScalar::abs(self.origin.derivative);
        let bound = self.c2 * initial;
        let scale = derivative.max(initial);
        if !scioxide_core::backend::standard::product_needs_retry(self.c2, initial, bound)
            && (derivative - bound).abs()
                > S::try_from_usize_exact(16)? * <S as RealScalar>::EPSILON * scale
        {
            return Ok(derivative <= bound);
        }
        use scioxide_core::Scaled;
        let derivative = Scaled::from_value(point.derivative)?.abs();
        let initial = Scaled::from_value(self.origin.derivative)?.abs();
        let bound = initial.mul(Scaled::from_value(self.c2)?)?;
        let scale = if derivative.abs_le(initial) {
            initial
        } else {
            derivative
        };
        let slack = scale.mul(Scaled::from_value(
            S::try_from_usize_exact(8)? * <S as RealScalar>::EPSILON,
        )?)?;
        Ok(derivative.abs_le(bound.add(slack)?))
    }

    fn satisfied(&self, point: LinePoint<S>) -> Result<bool, SciOxideError> {
        Ok(self.armijo_satisfied(point)? && self.curvature_satisfied(point)?)
    }
}

/// Finds a step length that satisfies the strong Wolfe conditions.
///
/// For a current point `x` and descent direction `d`, the search looks for a
/// step `α` with
///
/// ```text
/// f(x + α d) ≤ f(x) + c₁ α ∇f(x)ᵀd          (sufficient decrease)
/// |∇f(x + α d)ᵀd| ≤ c₂ |∇f(x)ᵀd|            (curvature)
/// ```
///
/// where `0 < c₁ < c₂ < 1`. Both comparisons allow a rounding slack of 8 `ε`
/// relative to the magnitudes compared, where `ε` is the machine epsilon.
///
/// The method brackets an acceptable step and then shrinks the bracket
/// ("zoom") with safeguarded quadratic interpolation, falling back to the
/// midpoint. A trial point outside the domain of the objective, or with a
/// non-finite value, closes the bracket: later trials bisect between the last
/// acceptable step and the rejected one. Each finite trial costs one objective and one
/// gradient evaluation.
///
/// The default uses initial step 1, minimum step `ε` times the smallest
/// positive normal number, `c₁ = 10⁻⁴`, `c₂ = 0.9` and at most 40 trials.
/// [`Self::for_nonlinear_cg`] uses `c₂ = 0.1`.
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §3.5 (Algorithms 3.5 and 3.6).
#[derive(Debug, Clone)]
pub struct StrongWolfeLineSearch<S: RealScalar> {
    alpha_init: S,
    alpha_min: S,
    step_tolerance: Option<scioxide_core::StepTolerance<S>>,
    c1: S,
    c2: S,
    max_iterations: scioxide_core::Budget,
}

impl<S: RealScalar> Default for StrongWolfeLineSearch<S> {
    fn default() -> Self {
        let one = S::ONE;
        let two = S::TWO;
        let five = two + two + one;
        let ten = five + five;
        Self {
            alpha_init: one,
            alpha_min: S::MIN_POSITIVE * <S as RealScalar>::EPSILON,
            step_tolerance: None,
            c1: one / (ten * ten * ten * ten),
            c2: (ten - one) / ten,
            max_iterations: scioxide_core::Budget::ONE.saturating_add(39),
        }
    }
}

impl<S: RealScalar> StrongWolfeLineSearch<S> {
    /// Returns the default search with `c₂ = 0.1`, the stricter curvature
    /// condition used by nonlinear conjugate gradient.
    pub fn for_nonlinear_cg() -> Self {
        let default = Self::default();
        Self {
            c2: S::ONE - default.c2,
            ..default
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> StrongWolfeLineSearch<S> {
    /// Creates a strong Wolfe search with initial step `alpha_init`, minimum
    /// step `alpha_min`, parameters `c1` and `c2`, and at most
    /// `max_iterations` trials.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `InvalidInput` unless `0 < alpha_min ≤ alpha_init` and
    ///   `0 < c1 < c2 < 1`.
    pub fn new(
        alpha_init: S,
        alpha_min: S,
        c1: S,
        c2: S,
        max_iterations: scioxide_core::Budget,
    ) -> Result<Self, SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        for (name, value) in [
            ("alpha_init", alpha_init),
            ("alpha_min", alpha_min),
            ("c1", c1),
            ("c2", c2),
        ] {
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    alloc::format!("{name} must be finite"),
                ));
            }
        }
        if alpha_init <= zero || alpha_min <= zero || alpha_min > alpha_init {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("alpha_min"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "line search requires 0 < alpha_min <= alpha_init",
            ));
        }
        if c1 <= zero || c1 >= c2 || c2 >= one {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("c2"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "strong Wolfe parameters must satisfy 0 < c1 < c2 < 1",
            ));
        }
        Ok(Self {
            alpha_init,
            alpha_min,
            step_tolerance: None,
            c1,
            c2,
            max_iterations,
        })
    }

    /// Sets a tolerance on the bracket width.
    ///
    /// The search stops unsuccessfully when the bracket is narrower than
    /// `tolerance`, with the larger bracket endpoint as the relative scale.
    /// The absolute part is in units of the step length `α`.
    pub fn with_step_tolerance(mut self, tolerance: scioxide_core::StepTolerance<S>) -> Self {
        self.step_tolerance = Some(tolerance);
        self
    }
    /// Returns the bracket-width tolerance, if set.
    pub const fn step_tolerance(&self) -> Option<scioxide_core::StepTolerance<S>> {
        self.step_tolerance
    }

    /// Returns the initial step.
    #[must_use]
    pub const fn initial_step(&self) -> S {
        self.alpha_init
    }

    /// Returns the minimum step.
    #[must_use]
    pub const fn minimum_step(&self) -> S {
        self.alpha_min
    }

    /// Returns the sufficient-decrease parameter `c₁`.
    #[must_use]
    pub const fn sufficient_decrease(&self) -> S {
        self.c1
    }

    /// Returns the curvature parameter `c₂`.
    #[must_use]
    pub const fn curvature(&self) -> S {
        self.c2
    }

    /// Returns the maximum number of trials.
    pub const fn max_iterations(&self) -> scioxide_core::Budget {
        self.max_iterations
    }

    fn roundoff_tolerance(values: &[S]) -> S {
        let zero: S = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one: S = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let four = two + two;
        let eight = four + four;

        let scale = values.iter().fold(zero, |current, &value| {
            current.max(NumericScalar::abs(value))
        });
        eight * <S as RealScalar>::EPSILON * scale
    }

    fn validate_inputs<F: ObjectiveFunction<S> + ?Sized>(
        f: &F,
        x: &[S],
        direction: &[S],
        grad_dot_dir: S,
    ) -> Result<(), SciOxideError> {
        let zero: S = <S as scioxide_core::AlgebraicScalar>::ZERO;
        if x.len() != f.dim() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(f.dim()),
                scioxide_core::Dimensions::one(x.len()),
            ));
        }
        if x.len() != direction.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(x.len()),
                scioxide_core::Dimensions::one(direction.len()),
            ));
        }

        utils::validate_finite_origin(x, scioxide_core::NonFiniteOrigin::Input)?;
        utils::validate_finite_origin(direction, scioxide_core::NonFiniteOrigin::Input)?;

        if !grad_dot_dir.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "non-finite value encountered",
            ));
        }
        if grad_dot_dir >= zero {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("grad_dot_dir"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "search direction is not a descent direction (grad_dot_dir must be negative)"
                    .to_owned(),
            ));
        }

        Ok(())
    }

    /// Evaluate the objective and directional derivative at a trial step.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_trial<F: ObjectiveFunction<S> + ?Sized>(
        f: &F,
        x: &[S],
        direction: &[S],
        alpha: S,
        x_trial: &mut [S],
        gradient: &mut [S],
        derivative_strategy: DerivativeStrategy,
        finite_difference: &mut DerivativeWorkspace<S>,
        scope: crate::LineSearchScope<'_>,
    ) -> Result<Option<EvaluatedLinePoint<S>>, SciOxideError> {
        let zero: S = <S as scioxide_core::AlgebraicScalar>::ZERO;
        if !alpha.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Intermediate,
                "trial step is non-finite",
            ));
        }
        if alpha <= zero {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "strong Wolfe trial step is not positive",
            ));
        }

        scope.propose()?;
        // An unrepresentable point, a point outside the domain and a non-finite
        // value all reject the trial; the search then shortens the step.
        if !crate::utils::fill_trial(x_trial, x, alpha, direction)? || !f.is_in_domain(x_trial)? {
            return Ok(None);
        }
        let value = f.eval(x_trial)?;
        if !value.is_finite() {
            return Ok(None);
        }

        utils::get_gradient_into(
            f,
            x_trial,
            value,
            derivative_strategy,
            gradient,
            finite_difference,
            scope.execution()?,
        )?;

        let derivative = utils::search_dot(gradient, direction)?;
        if !derivative.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Intermediate,
                "non-finite value encountered",
            ));
        }

        Ok(Some(EvaluatedLinePoint {
            point: LinePoint {
                alpha,
                value,
                derivative,
            },
        }))
    }

    /// Select a strictly interior, safeguarded point in the zoom bracket.
    fn safeguarded_trial(
        low: LinePoint<S>,
        high: LinePoint<S>,
        alpha_min: S,
    ) -> Result<Result<S, LineSearchFailure>, SciOxideError> {
        let one: S = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let four = two + two;
        let ten = four + four + two;

        let lower = low.alpha.min(high.alpha);
        let upper = low.alpha.max(high.alpha);
        let width = upper - lower;
        if !width.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "strong Wolfe bracket width became non-finite",
            ));
        }
        let interval_floor = Self::roundoff_tolerance(&[lower, upper]);
        if width <= interval_floor {
            return Ok(Err(LineSearchFailure::Stagnation));
        }

        let midpoint = lower + S::HALF * width;
        if !midpoint.is_finite() || midpoint <= lower || midpoint >= upper {
            return Ok(Err(LineSearchFailure::Stagnation));
        }

        let interpolated = if let Some(candidate) = ordinary_interpolation(low, high, lower, upper)
        {
            candidate
        } else {
            use scioxide_core::Scaled;
            let delta = Scaled::from_value(high.alpha)?.sub(Scaled::from_value(low.alpha)?)?;
            let slope_delta = Scaled::from_value(low.derivative)?.mul(delta)?;
            let curvature = Scaled::from_value(high.value)?
                .sub(Scaled::from_value(low.value)?)?
                .sub(slope_delta)?;
            let denominator = curvature.mul(Scaled::from_value(two)?)?;
            if denominator.is_zero() {
                None
            } else {
                let candidate = Scaled::from_value(low.alpha)?
                    .sub(slope_delta.mul(delta)?.div(denominator)?)?;
                // A candidate outside the bracket is irrelevant even if not representable.
                if candidate.cmp_value(Scaled::from_value(lower)?) == core::cmp::Ordering::Greater
                    && candidate.cmp_value(Scaled::from_value(upper)?) == core::cmp::Ordering::Less
                {
                    Some(candidate.round()?)
                } else {
                    None
                }
            }
        };

        let guard = width / ten;
        let guarded_lower = lower + guard;
        let guarded_upper = upper - guard;
        let trial = match interpolated {
            Some(candidate)
                if candidate > guarded_lower
                    && candidate < guarded_upper
                    && candidate >= alpha_min =>
            {
                candidate
            }
            _ => midpoint,
        };

        if trial < alpha_min {
            return Ok(Err(LineSearchFailure::StepSizeLimit));
        }
        if trial <= lower || trial >= upper {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "strong Wolfe safeguarded trial escaped its bracket",
            ));
        }

        Ok(Ok(trial))
    }

    #[allow(clippy::too_many_arguments)]
    fn zoom<F: ObjectiveFunction<S> + ?Sized>(
        context: &SearchContext<'_, S, F>,
        mut low: LinePoint<S>,
        mut upper: Upper<S>,
        x_trial: &mut [S],
        gradient: &mut [S],
        finite_difference: &mut DerivativeWorkspace<S>,
        progress: &mut SearchProgress<'_, S>,
    ) -> Result<LineSearchResult<S>, SciOxideError> {
        for _ in 0..context.search.max_iterations.get() {
            // Expansion and zoom share one proposal budget. Every earlier
            // proposal was rejected before scope reaches this loop.
            if progress.rejected >= context.search.max_iterations.get() {
                Self::restore_best_trial(context, x_trial, progress)?;
                return progress.unsuccessful(LineSearchFailure::IterationLimit);
            }
            if let Some(tolerance) = context.search.step_tolerance {
                let far = upper.alpha();
                if tolerance
                    .is_satisfied(NumericScalar::abs(far - low.alpha), far.max(low.alpha))?
                {
                    Self::restore_best_trial(context, x_trial, progress)?;
                    return progress.unsuccessful(LineSearchFailure::StepSizeLimit);
                }
            }
            let selected = match upper {
                Upper::Rejected(boundary) => {
                    let trial = checked_midpoint(low.alpha, boundary)?;
                    if trial == low.alpha || trial == boundary {
                        Err(LineSearchFailure::Stagnation)
                    } else if trial < context.search.alpha_min {
                        Err(LineSearchFailure::StepSizeLimit)
                    } else {
                        Ok(trial)
                    }
                }
                Upper::Point(high) => Self::safeguarded_trial(low, high, context.search.alpha_min)?,
            };
            let alpha = match selected {
                Ok(alpha) => alpha,
                Err(reason) => {
                    Self::restore_best_trial(context, x_trial, progress)?;
                    return progress.unsuccessful(reason);
                }
            };
            let Some(evaluation) = Self::evaluate_trial(
                context.objective,
                context.initial,
                context.direction,
                alpha,
                x_trial,
                gradient,
                context.derivative_strategy,
                finite_difference,
                context.scope,
            )?
            else {
                context.scope.reject()?;
                progress.rejected += 1;
                upper = Upper::Rejected(alpha);
                continue;
            };

            let trial = evaluation.point;

            if !context.conditions.armijo_satisfied(trial)? || trial.value >= low.value {
                upper = Upper::Point(trial);
                context.scope.reject()?;
                progress.rejected += 1;
                progress.record_rejected(&evaluation)?;
                continue;
            }
            if context.conditions.satisfied(trial)? {
                return progress.accepted(&evaluation);
            }

            let derivative_points_toward_upper = if upper.alpha() > low.alpha {
                trial.derivative >= <S as scioxide_core::AlgebraicScalar>::ZERO
            } else {
                trial.derivative <= <S as scioxide_core::AlgebraicScalar>::ZERO
            };
            // The bracket keeps a rejected upper end only while the
            // minimizer may still lie between the trial and that end.
            if derivative_points_toward_upper {
                upper = Upper::Point(low);
            }
            low = trial;
            context.scope.reject()?;
            progress.rejected += 1;
            progress.record_rejected(&evaluation)?;
        }

        Self::restore_best_trial(context, x_trial, progress)?;
        progress.unsuccessful(LineSearchFailure::IterationLimit)
    }

    fn restore_best_trial<F: ObjectiveFunction<S> + ?Sized>(
        context: &SearchContext<'_, S, F>,
        trial_point: &mut [S],
        progress: &SearchProgress<'_, S>,
    ) -> Result<(), SciOxideError> {
        if let Some((step, _)) = progress.best_trial {
            trial_point
                .iter_mut()
                .zip(context.initial.iter().zip(context.direction))
                .try_for_each(|(candidate, (&current, &component))| {
                    *candidate = crate::utils::trial_component(current, step, component)?;
                    Ok::<(), SciOxideError>(())
                })?;
        }
        Ok(())
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> LineSearch<S> for StrongWolfeLineSearch<S> {
    fn search_into<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x: &[S],
        initial_value: S,
        direction: &[S],
        grad_dot_dir: S,
        derivative_strategy: DerivativeStrategy,
        trial_point: &mut [S],
        trial_gradient: &mut [S],
        candidate_point: &mut [S],
        candidate_gradient: &mut [S],
        finite_difference: &mut DerivativeWorkspace<S>,
        scope: crate::LineSearchScope<'_>,
        initial_step: Option<S>,
    ) -> Result<LineSearchResult<S>, SciOxideError> {
        if let Some(execution) = scope.standalone() {
            return crate::line_search::standalone(self, x.len(), execution, |session| {
                self.search_into(
                    &session.observe(f) as &dyn ObjectiveFunction<S>,
                    x,
                    initial_value,
                    direction,
                    grad_dot_dir,
                    derivative_strategy,
                    trial_point,
                    trial_gradient,
                    candidate_point,
                    candidate_gradient,
                    finite_difference,
                    crate::LineSearchScope::observed(session),
                    initial_step,
                )
            });
        }

        let zero: S = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let two =
            <S as scioxide_core::AlgebraicScalar>::ONE + <S as scioxide_core::AlgebraicScalar>::ONE;
        let publish = |result: LineSearchResult<S>,
                       trial_point: &mut [S],
                       trial_gradient: &mut [S],
                       candidate_point: &[S],
                       candidate_gradient: &[S]| {
            if result.has_trial() {
                trial_point.copy_from_slice(candidate_point);
                if result.gradient_available() {
                    trial_gradient.copy_from_slice(candidate_gradient);
                }
            }
            result
        };
        Self::validate_inputs(f, x, direction, grad_dot_dir)?;
        if trial_point.len() != x.len()
            || trial_gradient.len() != x.len()
            || candidate_point.len() != x.len()
            || candidate_gradient.len() != x.len()
            || finite_difference.input_dimension() != x.len()
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(x.len()),
                scioxide_core::Dimensions::one(trial_point.len()),
            ));
        }

        if !initial_value.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "f_x must be finite",
            ));
        }

        let origin = LinePoint {
            alpha: zero,
            value: initial_value,
            derivative: grad_dot_dir,
        };
        let conditions = WolfeConditions {
            origin,
            c1: self.c1,
            c2: self.c2,
        };
        let context = SearchContext {
            objective: f,
            initial: x,
            direction,
            conditions,
            search: self,
            derivative_strategy,
            scope,
        };
        let mut previous = origin;
        let mut alpha =
            crate::line_search::initial_step(initial_step, self.alpha_init, self.alpha_min)?;
        let mut progress = SearchProgress::new(scope)?;

        for iteration in 0..self.max_iterations.get() {
            let Some(evaluation) = Self::evaluate_trial(
                f,
                x,
                direction,
                alpha,
                candidate_point,
                candidate_gradient,
                derivative_strategy,
                finite_difference,
                scope,
            )?
            else {
                // The rejected step bounds the search: bisect toward the last
                // acceptable trial (the origin at first), never expanding past it.
                context.scope.reject()?;
                progress.rejected += 1;
                let result = Self::zoom(
                    &context,
                    previous,
                    Upper::Rejected(alpha),
                    candidate_point,
                    candidate_gradient,
                    finite_difference,
                    &mut progress,
                )?;
                return Ok(publish(
                    result,
                    trial_point,
                    trial_gradient,
                    candidate_point,
                    candidate_gradient,
                ));
            };

            let trial = evaluation.point;

            if !conditions.armijo_satisfied(trial)?
                || (iteration > 0 && trial.value >= previous.value)
            {
                context.scope.reject()?;
                progress.rejected += 1;
                progress.record_rejected(&evaluation)?;
                let result = Self::zoom(
                    &context,
                    previous,
                    Upper::Point(trial),
                    candidate_point,
                    candidate_gradient,
                    finite_difference,
                    &mut progress,
                )?;
                return Ok(publish(
                    result,
                    trial_point,
                    trial_gradient,
                    candidate_point,
                    candidate_gradient,
                ));
            }
            if conditions.satisfied(trial)? {
                let result = progress.accepted(&evaluation)?;
                trial_point.copy_from_slice(candidate_point);
                trial_gradient.copy_from_slice(candidate_gradient);
                return Ok(result);
            }
            if trial.derivative >= zero {
                context.scope.reject()?;
                progress.rejected += 1;
                progress.record_rejected(&evaluation)?;
                let result = Self::zoom(
                    &context,
                    trial,
                    Upper::Point(previous),
                    candidate_point,
                    candidate_gradient,
                    finite_difference,
                    &mut progress,
                )?;
                return Ok(publish(
                    result,
                    trial_point,
                    trial_gradient,
                    candidate_point,
                    candidate_gradient,
                ));
            }

            previous = trial;
            context.scope.reject()?;
            progress.rejected += 1;
            progress.record_rejected(&evaluation)?;
            if iteration + 1 < self.max_iterations.get() {
                let expanded = alpha * two;
                if !expanded.is_finite() || expanded <= alpha {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "strong Wolfe step expansion became non-finite",
                    ));
                }
                alpha = expanded;
            }
        }

        if let Some((step, _)) = progress.best_trial {
            candidate_point
                .iter_mut()
                .zip(x.iter().zip(direction))
                .try_for_each(|(candidate, (&current, &component))| {
                    *candidate = crate::utils::trial_component(current, step, component)?;
                    Ok::<(), SciOxideError>(())
                })?;
        }
        let result = progress.unsuccessful(LineSearchFailure::IterationLimit)?;
        if result.has_trial() {
            trial_point.copy_from_slice(candidate_point);
        }
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use crate::LineSearchTermination;
    use crate::line_search::testing::Standalone;
    use core::cell::Cell;

    struct Quadratic;

    impl ObjectiveFunction<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(true)
        }
    }

    struct QuadraticF32;

    impl ObjectiveFunction<f32> for QuadraticF32 {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f32]) -> Result<f32, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }

        fn gradient_into(&self, x: &[f32], output: &mut [f32]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(true)
        }
    }

    struct Quartic;

    impl ObjectiveFunction<f64> for Quartic {
        fn dim(&self) -> usize {
            1
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0].powi(4))
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = 4.0 * x[0].powi(3);
            Ok(true)
        }
    }

    struct DescendingLinear;

    impl ObjectiveFunction<f64> for DescendingLinear {
        fn dim(&self) -> usize {
            1
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(-x[0])
        }

        fn gradient_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = -1.0;
            Ok(true)
        }
    }

    struct PositiveDomainQuadratic {
        domain_checks: Cell<usize>,
        evaluations: Cell<usize>,
    }

    impl ObjectiveFunction<f64> for PositiveDomainQuadratic {
        fn dim(&self) -> usize {
            1
        }

        fn is_in_domain(&self, x: &[f64]) -> Result<bool, SciOxideError> {
            self.domain_checks.set(self.domain_checks.get() + 1);
            Ok(x[0] > 0.0)
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.evaluations.set(self.evaluations.get() + 1);
            if x[0] <= 0.0 {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("x"),
                    scioxide_core::DomainReason::OutsideDomain,
                    "x must be positive",
                ));
            }
            Ok((x[0] - 0.25).powi(2))
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = 2.0 * (x[0] - 0.25);
            Ok(true)
        }
    }

    #[test]
    fn invalid_domain_trials_are_rejected_before_evaluation_and_recover() {
        let objective = PositiveDomainQuadratic {
            domain_checks: Cell::new(0),
            evaluations: Cell::new(0),
        };
        let result = StrongWolfeLineSearch::default()
            .search(&objective, &[1.0], 0.5625, &[-1.5], -2.25)
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.step(), Some(0.5));
        assert_eq!(result.value(), Some(0.0));
        assert_eq!(result.rejected_trials(), 1);
        assert_eq!(result.fun_evals(), 1);
        assert_eq!(objective.domain_checks.get(), 2);
        assert_eq!(objective.evaluations.get(), 1);
    }

    fn test_roundoff_tolerance(values: &[f64]) -> f64 {
        8.0 * f64::EPSILON
            * values
                .iter()
                .fold(0.0_f64, |scale, value| scale.max(value.abs()))
    }

    fn assert_f64_strong_wolfe(
        objective: &dyn ObjectiveFunction<f64>,
        x: &[f64],
        direction: &[f64],
        initial_derivative: f64,
        alpha: f64,
        c1: f64,
        c2: f64,
    ) {
        let initial_value = objective.eval(x).unwrap();
        let trial: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(&initial, &component)| initial + alpha * component)
            .collect();
        let trial_value = objective.eval(&trial).unwrap();
        let trial_gradient = objective.gradient(&trial).unwrap().unwrap();
        let trial_derivative = utils::dot(&trial_gradient, direction).unwrap();
        let decrease = c1 * alpha * initial_derivative;
        let armijo_bound = initial_value + decrease;
        let armijo_slack =
            test_roundoff_tolerance(&[trial_value, initial_value, decrease, armijo_bound]);
        let curvature_bound = c2 * initial_derivative.abs();
        let curvature_slack =
            test_roundoff_tolerance(&[trial_derivative, initial_derivative, curvature_bound]);

        assert!(trial_value <= armijo_bound + armijo_slack);
        assert!(trial_derivative.abs() <= curvature_bound + curvature_slack);
    }

    #[test]
    fn successful_f64_step_satisfies_both_wolfe_conditions() {
        let line_search = StrongWolfeLineSearch::<f64>::default();
        let x = [2.0, 2.0];
        let gradient = [4.0, 4.0];
        let direction = [-4.0, -4.0];
        let initial_derivative = utils::dot(&gradient, &direction).unwrap();

        let result = line_search
            .search(
                &Quadratic,
                &x,
                Quadratic.eval(&x).unwrap(),
                &direction,
                initial_derivative,
            )
            .unwrap();
        let alpha = result.step().unwrap();

        assert_eq!(result.fun_evals(), 2);
        assert_eq!(result.grad_evals(), 2);
        assert_eq!(result.value(), Some(0.0));
        assert_eq!(result.gradient(), Some(&[0.0, 0.0][..]));
        assert_eq!(result.directional_derivative(), Some(0.0));

        assert_f64_strong_wolfe(
            &Quadratic,
            &x,
            &direction,
            initial_derivative,
            alpha,
            1e-4,
            0.9,
        );
    }

    #[test]
    fn successful_f32_step_satisfies_both_wolfe_conditions() {
        let line_search = StrongWolfeLineSearch::<f32>::default();
        let x = [2.0_f32, 2.0];
        let gradient = [4.0_f32, 4.0];
        let direction = [-4.0_f32, -4.0];
        let initial_derivative = utils::dot(&gradient, &direction).unwrap();

        let result = line_search
            .search(
                &QuadraticF32,
                &x,
                QuadraticF32.eval(&x).unwrap(),
                &direction,
                initial_derivative,
            )
            .unwrap();
        let alpha = result.step().unwrap();
        let trial = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]];
        let trial_value = QuadraticF32.eval(&trial).unwrap();
        let trial_gradient = QuadraticF32.gradient(&trial).unwrap().unwrap();
        let trial_derivative = utils::dot(&trial_gradient, &direction).unwrap();
        let initial_value = QuadraticF32.eval(&x).unwrap();
        assert!(
            trial_value <= initial_value + 1e-4 * alpha * initial_derivative + 8.0 * f32::EPSILON
        );
        assert!(trial_derivative.abs() <= 0.9 * initial_derivative.abs() + 8.0 * f32::EPSILON);
    }

    #[test]
    fn finite_difference_trials_report_all_function_evaluations() {
        struct FiniteDifferenceQuadratic;

        impl ObjectiveFunction<f64> for FiniteDifferenceQuadratic {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0])
            }
        }

        let result = StrongWolfeLineSearch::<f64>::default()
            .search(&FiniteDifferenceQuadratic, &[1.0], 1.0, &[-2.0], -4.0)
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.fun_evals(), 4);
        assert_eq!(result.grad_evals(), 2);
        assert!(result.gradient().is_some());
        assert!(result.directional_derivative().is_some());
    }

    #[test]
    fn explicit_automatic_strategy_is_used_for_trial_gradients() {
        struct AutomaticQuadratic;

        impl ObjectiveFunction<f64> for AutomaticQuadratic {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0])
            }

            fn automatic_gradient_into(
                &self,
                x: &[f64],
                output: &mut [f64],
            ) -> Result<bool, SciOxideError> {
                output[0] = 2.0 * x[0];
                Ok(true)
            }
        }

        let result = StrongWolfeLineSearch::<f64>::default()
            .search_with_strategy(
                &AutomaticQuadratic,
                &[1.0],
                1.0,
                &[-1.0],
                -2.0,
                DerivativeStrategy::Automatic,
            )
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.fun_evals(), 1);
        assert_eq!(result.grad_evals(), 1);
    }

    #[test]
    fn outer_search_exhaustion_is_failure() {
        let line_search = StrongWolfeLineSearch::new(
            1.0,
            f64::EPSILON,
            1e-4,
            0.9,
            scioxide_core::Budget::try_from(2).unwrap(),
        )
        .unwrap();
        let result = line_search
            .search(&DescendingLinear, &[0.0], 0.0, &[1.0], -1.0)
            .unwrap();

        assert!(matches!(
            result.termination(),
            LineSearchTermination::Unsuccessful(LineSearchFailure::IterationLimit)
        ));
        assert!(result.has_trial());
        assert_eq!(result.fun_evals(), 2);
        assert_eq!(result.grad_evals(), 2);
        assert_eq!(result.step(), Some(2.0));
        assert_eq!(result.value(), Some(-2.0));
    }

    #[test]
    fn zoom_exhaustion_is_failure() {
        let line_search =
            StrongWolfeLineSearch::new(1.0, f64::EPSILON, 1e-4, 0.1, scioxide_core::Budget::ONE)
                .unwrap();
        let mut trial_point = [f64::NAN];
        let mut trial_gradient = [f64::NAN];
        let mut candidate_point = [0.0];
        let mut candidate_gradient = [0.0];
        let mut finite_difference = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Gradient,
            1,
            1,
        )
        .unwrap();
        let result = line_search
            .search_into(
                &Quartic,
                &[1.0],
                1.0,
                &[-4.0],
                -16.0,
                DerivativeStrategy::Analytic,
                &mut trial_point,
                &mut trial_gradient,
                &mut candidate_point,
                &mut candidate_gradient,
                &mut finite_difference,
                scioxide_core::ExecutionContext::sequential().into(),
                None,
            )
            .unwrap();

        assert!(matches!(
            result.failure(),
            Some(LineSearchFailure::IterationLimit)
        ));
        assert!(result.has_trial());
        assert_eq!(result.fun_evals(), 1);
        assert_eq!(result.grad_evals(), 1);
        assert_eq!(trial_point[0], 1.0 - 4.0 * result.step());
    }

    #[test]
    fn bracket_collapse_below_minimum_step_is_failure() {
        struct ShiftedQuadratic;

        impl ObjectiveFunction<f64> for ShiftedQuadratic {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok((x[0] - 0.25).powi(2))
            }

            fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
                output[0] = 2.0 * (x[0] - 0.25);
                Ok(true)
            }
        }

        let line_search = StrongWolfeLineSearch::new(
            1.0,
            0.9,
            1e-4,
            0.9,
            scioxide_core::Budget::try_from(40).unwrap(),
        )
        .unwrap();
        let result = line_search
            .search(&ShiftedQuadratic, &[0.0], 0.0625, &[1.0], -0.5)
            .unwrap();

        assert!(matches!(
            result.termination(),
            LineSearchTermination::Unsuccessful(LineSearchFailure::StepSizeLimit)
        ));
        assert!(result.has_trial());
    }

    #[test]
    fn extreme_finite_values_do_not_overflow_interpolation() {
        struct ExplosiveObjective;

        impl ObjectiveFunction<f64> for ExplosiveObjective {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(if x[0] == 0.0 { 0.0 } else { f64::MAX })
            }

            fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
                output[0] = if x[0] == 0.0 { -1.0 } else { 1.0 };
                Ok(true)
            }
        }

        let result = StrongWolfeLineSearch::<f64>::default().search(
            &ExplosiveObjective,
            &[0.0],
            0.0,
            &[1.0],
            -1.0,
        );

        assert!(!result.unwrap().termination().is_success());
    }

    #[test]
    fn safeguarded_trial_stays_inside_bracket() {
        let low = LinePoint {
            alpha: 0.0,
            value: 0.0,
            derivative: -1.0,
        };
        let high = LinePoint {
            alpha: 1.0,
            value: -2.0,
            derivative: -1.0,
        };

        let alpha = StrongWolfeLineSearch::<f64>::safeguarded_trial(low, high, f64::EPSILON)
            .unwrap()
            .unwrap();

        assert!(alpha > 0.0);
        assert!(alpha < 1.0);
    }

    #[test]
    fn non_descent_direction_is_invalid_input() {
        let result = StrongWolfeLineSearch::<f64>::default().search(
            &Quadratic,
            &[1.0, 1.0],
            2.0,
            &[1.0, 1.0],
            4.0,
        );

        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }

    #[test]
    fn non_finite_initial_directional_derivative_is_non_finite_error() {
        for derivative in [f64::NAN, f64::NEG_INFINITY] {
            let result = StrongWolfeLineSearch::<f64>::default().search(
                &Quadratic,
                &[1.0, 1.0],
                2.0,
                &[-1.0, -1.0],
                derivative,
            );
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                })
            ));
        }
    }

    #[test]
    fn non_finite_wolfe_parameters_are_input_errors() {
        for result in [
            StrongWolfeLineSearch::new(
                1.0,
                f64::EPSILON,
                f64::NAN,
                0.9,
                scioxide_core::Budget::try_from(40).unwrap(),
            ),
            StrongWolfeLineSearch::new(
                1.0,
                f64::EPSILON,
                1e-4,
                f64::INFINITY,
                scioxide_core::Budget::try_from(40).unwrap(),
            ),
        ] {
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                })
            ));
        }
    }

    #[test]
    fn non_finite_trial_values_are_rejected_and_the_origin_stays_strict() {
        struct NonFiniteObjective;

        impl ObjectiveFunction<f64> for NonFiniteObjective {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(if x[0] == 0.0 { 0.0 } else { f64::INFINITY })
            }

            fn gradient_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
                output[0] = -1.0;
                Ok(true)
            }
        }

        // Every trial is rejected: the search bisects toward the origin and
        // stops unsuccessfully without publishing a trial.
        let result = StrongWolfeLineSearch::<f64>::default()
            .search(&NonFiniteObjective, &[0.0], 0.0, &[1.0], -1.0)
            .unwrap();
        assert_eq!(
            result.termination(),
            LineSearchTermination::Unsuccessful(LineSearchFailure::IterationLimit)
        );
        assert!(!result.has_trial());
        assert_eq!(result.rejected_trials(), 40);
        assert_eq!(result.fun_evals(), 40);
        assert_eq!(result.grad_evals(), 0);

        struct NonFiniteAtOrigin;

        impl ObjectiveFunction<f64> for NonFiniteAtOrigin {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(f64::NAN)
            }
        }

        let result = StrongWolfeLineSearch::<f64>::default().search(
            &NonFiniteAtOrigin,
            &[0.0],
            f64::NAN,
            &[1.0],
            -1.0,
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn a_rejected_expansion_bisects_toward_the_last_acceptable_step() {
        // f(t) = (t - 1.5)^2 below a non-finite wall at t >= 1.8. With c2 = 0.1
        // the step t = 1 must expand; t = 2 is rejected, and the search bisects
        // to t = 1.5 instead of repeating t = 1 or expanding past the wall.
        let wall = crate::ObjectiveFn::new(1, |x: &[f64]| {
            Ok(if x[0] < 1.8 {
                (x[0] - 1.5).powi(2)
            } else {
                f64::NAN
            })
        })
        .with_gradient(|x: &[f64], output| {
            output.copy_from_slice(&[2.0 * (x[0] - 1.5)]);
            Ok(())
        });
        let result = StrongWolfeLineSearch::<f64>::for_nonlinear_cg()
            .search(&wall, &[0.0], 2.25, &[1.0], -3.0)
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert_eq!(result.step(), Some(1.5));
        assert_eq!(result.fun_evals(), 3);
        assert_eq!(result.rejected_trials(), 2);
    }

    #[test]
    fn non_finite_gradient_is_rejected() {
        struct NonFiniteGradient;

        impl ObjectiveFunction<f64> for NonFiniteGradient {
            fn dim(&self) -> usize {
                1
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0])
            }

            fn gradient_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
                output[0] = f64::NAN;
                Ok(true)
            }
        }

        let result = StrongWolfeLineSearch::<f64>::default().search(
            &NonFiniteGradient,
            &[1.0],
            1.0,
            &[-1.0],
            -2.0,
        );

        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn incomplete_gradient_is_non_finite() {
        let malformed_gradient = crate::ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]))
            .with_gradient(|_: &[f64], _output| Ok(()));

        let result = StrongWolfeLineSearch::<f64>::default().search(
            &malformed_gradient,
            &[1.0],
            1.0,
            &[-1.0],
            -2.0,
        );

        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn input_dimension_mismatches_are_structural_errors() {
        let line_search = StrongWolfeLineSearch::<f64>::default();

        let wrong_objective_dimension = line_search.search(&Quadratic, &[1.0], 1.0, &[-1.0], -2.0);
        assert!(matches!(
            (wrong_objective_dimension)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));

        let wrong_direction_dimension =
            line_search.search(&Quadratic, &[1.0, 1.0], 2.0, &[-1.0], -2.0);
        assert!(matches!(
            (wrong_direction_dimension)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_options_are_rejected() {
        for result in [
            StrongWolfeLineSearch::new(
                0.0,
                f64::EPSILON,
                1e-4,
                0.9,
                scioxide_core::Budget::try_from(40).unwrap(),
            ),
            StrongWolfeLineSearch::new(
                1.0,
                2.0,
                1e-4,
                0.9,
                scioxide_core::Budget::try_from(40).unwrap(),
            ),
            StrongWolfeLineSearch::new(
                1.0,
                f64::EPSILON,
                0.4,
                0.3,
                scioxide_core::Budget::try_from(40).unwrap(),
            ),
        ] {
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
            ));
        }
    }
}
